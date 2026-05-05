import time
import json
import torch
import matplotlib.pyplot as plt
from model_pytorch import GPT, GPTConfig
from inference_engine import InferenceEngine

# -----------------------------------------------------------------------------
# KV Cache Math Explanation:
#
# Why is the speedup quadratic?
# In standard autoregressive generation (no cache), at step N we must pass all 
# N tokens through the transformer. The self-attention mechanism computes a 
# dot product between every query and every key, which requires O(N^2) operations.
# Thus, generating T tokens from a prompt of length P takes:
# O((P+1)^2 + (P+2)^2 + ... + (P+T)^2) operations.
#
# With a KV Cache, we store the computed Keys and Values for all past tokens.
# At step N, we ONLY pass the single new token (query) through the network.
# Attention becomes a dot product between 1 query and N keys, which is O(N).
# Thus, generating T tokens from a prompt of length P takes:
# O((P+1) + (P+2) + ... + (P+T)) operations.
#
# As sequence length (P or T) grows, the O(N^2) no-cache approach becomes 
# catastrophically slow, while the cache approach scales linearly per step,
# leading to a quadratic relative speedup factor.
# -----------------------------------------------------------------------------

def benchmark():
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        d_model=128,
        d_ff=512,
        block_size=512, # need large block size to fit prompt + generated tokens
        vocab_size=50257
    )
    
    model = GPT(config)
    engine = InferenceEngine(model)
    
    seq_lengths = [32, 64, 128, 256]
    max_new_tokens = 20
    trials = 5
    
    results = {}
    
    print("\n" + "="*70)
    print("KV CACHE BENCHMARK (MacBook CPU)")
    print(f"Generating {max_new_tokens} new tokens for various prompt lengths")
    print("="*70)
    print(f"{'Seq Len':<10} | {'No Cache (ms)':<15} | {'Cache (ms)':<15} | {'Speedup Factor':<15}")
    print("-" * 65)
    
    for seq_len in seq_lengths:
        torch.manual_seed(42)
        prompt = torch.randint(0, config.vocab_size, (1, seq_len))
        
        # 1. Verification: Ensure identical outputs
        torch.manual_seed(1337)
        out_no_cache = engine.generate_no_cache(prompt, max_new_tokens=max_new_tokens)
        
        torch.manual_seed(1337)
        out_cache = engine.generate_with_cache(prompt, max_new_tokens=max_new_tokens)
        
        assert torch.equal(out_no_cache, out_cache), f"Mismatch for seq_len {seq_len}!"
        
        # 2. Benchmark No Cache
        no_cache_times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = engine.generate_no_cache(prompt, max_new_tokens=max_new_tokens)
            t1 = time.perf_counter()
            no_cache_times.append((t1 - t0) * 1000)
        no_cache_ms = sum(no_cache_times) / trials
        
        # 3. Benchmark With Cache
        cache_times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = engine.generate_with_cache(prompt, max_new_tokens=max_new_tokens)
            t1 = time.perf_counter()
            cache_times.append((t1 - t0) * 1000)
        cache_ms = sum(cache_times) / trials
        
        speedup = no_cache_ms / cache_ms
        
        print(f"{seq_len:<10} | {no_cache_ms:<15.2f} | {cache_ms:<15.2f} | {speedup:<15.2f}x")
        
        results[str(seq_len)] = {
            "no_cache_ms": no_cache_ms,
            "cache_ms": cache_ms,
            "speedup_factor": speedup
        }
        
    # Plotting
    plt.figure(figsize=(10, 6))
    no_cache_vals = [results[str(s)]["no_cache_ms"] for s in seq_lengths]
    cache_vals = [results[str(s)]["cache_ms"] for s in seq_lengths]
    
    plt.plot(seq_lengths, no_cache_vals, marker='o', label='No Cache (Baseline)', linewidth=2)
    plt.plot(seq_lengths, cache_vals, marker='s', label='KV Cache', linewidth=2)
    
    plt.title(f'KV Cache Latency Speedup (Generating {max_new_tokens} tokens)')
    plt.xlabel('Prompt Sequence Length')
    plt.ylabel('Generation Latency (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cache_speedup_plot.png', dpi=300)
    
    with open("cache_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaved cache_speedup_plot.png and cache_benchmark_results.json")

if __name__ == "__main__":
    benchmark()
