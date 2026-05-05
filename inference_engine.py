import torch
from torch.nn import functional as F
from model_pytorch import GPT, GPTConfig
from kv_cache import KVCache

class InferenceEngine:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def generate_no_cache(self, idx, max_new_tokens):
        """
        Baseline autoregressive generation without KV caching.
        Recomputes attention for all tokens at every step. O(N^2) complexity.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] # Pluck last step
            probs = F.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_with_cache(self, idx, max_new_tokens):
        """
        Cache-enabled generation.
        Only forwards the new token at each step. O(N) complexity per token.
        """
        cache = KVCache(self.model.config.n_layer)
        
        # 1. Prefill phase: process the entire prompt to populate the cache
        idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
        logits, _ = self.model(idx_cond, kv_cache=cache)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 2. Generation phase: only pass the most recently generated token
        for _ in range(max_new_tokens - 1):
            if idx.size(1) >= self.model.config.block_size:
                # Basic implementation: stop generation if exceeding block size
                break
                
            logits, _ = self.model(idx_next, kv_cache=cache)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
