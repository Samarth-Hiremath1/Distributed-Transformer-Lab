import time
import torch
import torch.nn.functional as F

def sample_greedy(logits):
    """Greedy decoding: argmax at each step."""
    return torch.argmax(logits, dim=-1, keepdim=True)

def sample_top_k(logits, k=50):
    """Top-k sampling: keep top-k logits, zero rest, sample."""
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def sample_nucleus(logits, p=0.9):
    """Nucleus sampling: keep minimum tokens covering p probability mass, sample."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(model, idx, max_new_tokens, strategy="greedy", **kwargs):
    """Generates new tokens using the specified strategy."""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] # Pluck last step
        
        if strategy == "greedy":
            idx_next = sample_greedy(logits)
        elif strategy == "top_k":
            k = kwargs.get("k", 50)
            idx_next = sample_top_k(logits, k=k)
        elif strategy == "nucleus":
            p = kwargs.get("p", 0.9)
            idx_next = sample_nucleus(logits, p=p)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def compute_output_perplexity(model, tokens):
    """Computes perplexity of the generated text under the model."""
    if tokens.size(1) <= 1:
        return float('inf')
        
    # We only care about the model's loss on the generated tokens, 
    # but the simplest way is to run the model on the sequence
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    
    # Chunk if it exceeds block_size
    losses = []
    for i in range(0, x.size(1), model.config.block_size):
        x_chunk = x[:, i:i+model.config.block_size]
        y_chunk = y[:, i:i+model.config.block_size]
        if x_chunk.size(1) == 0:
            break
        _, loss = model(x_chunk, targets=y_chunk)
        losses.append(loss.item())
        
    if not losses:
        return float('inf')
        
    mean_loss = sum(losses) / len(losses)
    return torch.exp(torch.tensor(mean_loss)).item()

def compute_consistency(runs_outputs):
    """Computes average pairwise token overlap across runs."""
    n = len(runs_outputs)
    if n <= 1:
        return 100.0
        
    total_overlap = 0
    pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            seq1 = runs_outputs[i]
            seq2 = runs_outputs[j]
            min_len = min(len(seq1), len(seq2))
            if min_len == 0:
                continue
            
            # Position-wise exact match overlap
            matches = sum(1 for k in range(min_len) if seq1[k] == seq2[k])
            overlap_pct = (matches / min_len) * 100
            total_overlap += overlap_pct
            pairs += 1
            
    if pairs == 0:
        return 100.0
        
    return total_overlap / pairs

def evaluate_strategy(model, tokenizer, strategy, prompts, max_tokens=20, runs=5, **kwargs):
    """Evaluates a decoding strategy on latency, perplexity, and consistency."""
    all_latencies = []
    all_perplexities = []
    all_consistencies = []
    
    for prompt_text in prompts:
        prompt_idx = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0)
        
        prompt_runs = []
        prompt_latencies = []
        
        # Set seed for reproducibility of the *strategy* across different models,
        # but let the stochastic methods sample differently across the 5 runs
        torch.manual_seed(42)
        
        for _ in range(runs):
            t0 = time.perf_counter()
            out_idx = generate(model, prompt_idx, max_tokens, strategy=strategy, **kwargs)
            t1 = time.perf_counter()
            
            prompt_latencies.append((t1 - t0) * 1000)
            
            # Perplexity of the full output (prompt + generated)
            ppl = compute_output_perplexity(model, out_idx)
            all_perplexities.append(ppl)
            
            # Just the generated tokens for consistency
            generated_only = out_idx[0, prompt_idx.size(1):].tolist()
            prompt_runs.append(generated_only)
            
        all_latencies.extend(prompt_latencies)
        
        # Consistency across runs for this prompt
        consistency = compute_consistency(prompt_runs)
        all_consistencies.append(consistency)
        
    mean_latency = sum(all_latencies) / len(all_latencies)
    mean_perplexity = sum(all_perplexities) / len(all_perplexities)
    mean_consistency = sum(all_consistencies) / len(all_consistencies)
    
    return mean_latency, mean_perplexity, mean_consistency

if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from model_pytorch import GPT, GPTConfig
    
    print("Testing decoding implementations...")
    config = GPTConfig(
        n_layer=4, n_head=4, d_model=128, d_ff=512, block_size=64, vocab_size=50257
    )
    model = GPT(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompts = ["Hello, world", "The meaning of life is", "Once upon a time"]
    
    for strategy in ["greedy", "top_k", "nucleus"]:
        lat, ppl, cons = evaluate_strategy(model, tokenizer, strategy, prompts, runs=2)
        print(f"{strategy}: latency={lat:.2f}ms, ppl={ppl:.2f}, consistency={cons:.2f}%")
