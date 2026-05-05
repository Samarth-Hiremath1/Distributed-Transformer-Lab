import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 50257 # GPT-2 vocab_size
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.n_head = config.n_head
        self.d_model = config.d_model
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_idx=None, kv_cache=None):
        # B = batch size, T_q = sequence length of queries, C = embedding dimensionality (d_model)
        B, T_q, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # hs (head size) = C // n_head
        # Transform to shape (B, n_head, T, hs)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2)

        # If cache provided: append new K/V, compute attention over full cache
        if kv_cache is not None and layer_idx is not None:
            kv_cache.append(layer_idx, k, v)
            k, v = kv_cache.get(layer_idx)

        T_k = k.size(2)

        # Causal scaled dot-product attention
        # We multiply Q (B, n_head, T_q, hs) by K^T (B, n_head, hs, T_k) to get attention scores (B, n_head, T_q, T_k)
        att = (q @ k.transpose(-2, -1))
        
        # Why divide by sqrt(d_k)?
        # If components of q and k have unit variance, their dot product has variance equal to d_k.
        # Dividing by sqrt(d_k) scales the variance back to 1, preventing the softmax from saturating 
        # (where gradients become vanishingly small).
        att = att * (1.0 / math.sqrt(k.size(-1)))
        
        # What is the mask doing?
        # The mask forces attention weights to be -infinity for all future tokens (upper triangular part).
        # We only apply the mask if we are passing multiple queries (e.g. prefill step). 
        # For single token generation (T_q == 1) using KV cache, we attend to all past tokens without masking.
        if T_q > 1:
            att = att.masked_fill(self.bias[:, :, T_k-T_q:T_k, :T_k] == 0, float('-inf'))
        
        # Apply softmax to get probability distribution over past tokens
        att = F.softmax(att, dim=-1)
        
        # Multiply attention weights by values
        # att (B, n_head, T_q, T_k) @ v (B, n_head, T_k, hs) -> y (B, n_head, T_q, hs)
        y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        
        # Output projection back to d_model
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pre-norm vs Post-norm:
        # In Pre-norm (used here and in GPT-2/3), LayerNorm is applied *before* the attention and MLP layers.
        # This makes the gradients well-behaved and allows training very deep networks without learning rate warmup tricks,
        # because the residual stream is kept "clean" and gradients can flow straight through.
        # Original Transformer used Post-norm, which is harder to train.
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, layer_idx=None, kv_cache=None):
        # Residual connections around both sub-layers
        x = x + self.attn(self.ln_1(x), layer_idx=layer_idx, kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, kv_cache=None):
        device = idx.device
        b, t = idx.size()
        
        # If using cache, we only forward the new tokens. The position should reflect 
        # the absolute position in the sequence, which includes the length of the cache.
        past_length = 0 if (kv_cache is None or kv_cache.k_cache[0] is None) else kv_cache.k_cache[0].size(2)
        
        assert past_length + t <= self.config.block_size, f"Cannot forward sequence of length {past_length + t}, block size is only {self.config.block_size}"
        
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        
        # Token embedding + learned positional embedding
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        # N TransformerBlocks
        for i, block in enumerate(self.transformer.h):
            x = block(x, layer_idx=i, kv_cache=kv_cache)
            
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            
        return logits, loss

    @classmethod
    def load_pretrained_gpt2(cls):
        """
        Loads pre-trained GPT-2 small weights from HuggingFace.
        Uses a config switcher to instantiate the full model architecture before weight transfer.
        """
        import transformers
        print("Loading weights from pretrained gpt2...")
        
        # Switch to full GPT-2 config for weight loading
        config = GPTConfig(
            n_layer=12,
            n_head=12,
            d_model=768,
            d_ff=3072,
            block_size=1024,
            vocab_size=50257,
            bias=True
        )
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask

        # init a huggingface/transformers model
        model_hf = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        
        # OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear.
        # This means that we have to transpose these weights when we import them.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-verification', action='store_true', help="Skip weight verification")
    args = parser.parse_args()

    if not args.skip_verification:
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT.load_pretrained_gpt2()
            model.eval()
            
            prompts = ["Hello, my name is", "The meaning of life is", "Once upon a time"]
            print("\nVerifying outputs...")
            for prompt in prompts:
                input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
                
                # Check HF model output
                from transformers import GPT2LMHeadModel
                hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
                hf_model.eval()
                with torch.no_grad():
                    hf_logits = hf_model(input_ids).logits
                    my_logits, _ = model(input_ids)
                    
                    diff = (hf_logits - my_logits).abs().max().item()
                    print(f"Prompt: '{prompt}' | Max logit diff: {diff:.6f}")
                    assert diff < 1e-4, "Outputs do not match!"
            print("Verification successful!")
        except ImportError:
            print("Please install transformers: pip install transformers")
    else:
        print("Skipping weight verification.")
