import time
import json
import tracemalloc
import torch
import numpy as np
from model_pytorch import GPT, GPTConfig

def measure_inference(model, seq_lengths=[32, 64, 128, 256]):
    model.eval()
    results = {}
    
    # Warmup
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 32))
    with torch.no_grad():
        for _ in range(3):
            model(dummy_input)

    print(f"{'Seq Len':<10} | {'Latency (ms)':<15} | {'Std Dev (ms)':<15} | {'Tokens/sec':<15} | {'Memory (MB)':<15}")
    print("-" * 75)

    for seq_len in seq_lengths:
        latencies = []
        
        # We need a tensor of size (1, seq_len)
        x = torch.randint(0, model.config.vocab_size, (1, seq_len))
        
        # Start memory tracking
        tracemalloc.start()
        
        with torch.no_grad():
            for _ in range(10):
                start_time = time.perf_counter()
                model(x)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000) # to ms
                
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / (1024 * 1024)
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Tokens per second: seq_len / (latency_in_seconds)
        # Wait, if it's a forward pass, it processes `seq_len` tokens at once.
        tokens_per_sec = seq_len / (mean_latency / 1000.0)
        
        print(f"{seq_len:<10} | {mean_latency:<15.2f} | {std_latency:<15.2f} | {tokens_per_sec:<15.2f} | {memory_mb:<15.2f}")
        
        results[str(seq_len)] = {
            "latency_ms": mean_latency,
            "std_ms": std_latency,
            "tokens_per_sec": tokens_per_sec,
            "memory_mb": memory_mb
        }
        
    return results

if __name__ == "__main__":
    # Use Tiny config for benchmarking
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        d_model=128,
        d_ff=512,
        block_size=256, # accommodate up to 256 for the benchmark
        vocab_size=50257
    )
    model = GPT(config)
    
    print("Benchmarking PyTorch Eager (Forward Pass)")
    results = measure_inference(model)
    
    with open("benchmark_pytorch.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to benchmark_pytorch.json")
