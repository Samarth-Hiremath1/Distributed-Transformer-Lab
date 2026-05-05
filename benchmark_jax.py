import time
import json
import tracemalloc
import jax
import jax.numpy as jnp
import numpy as np
from model_jax import GPT, GPTConfig

def benchmark_jax():
    print("Initializing JAX/Flax model...")
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        d_model=128,
        d_ff=512,
        block_size=256, # accommodate up to 256 for the benchmark
        vocab_size=50257
    )
    model = GPT(config)
    
    # Initialize explicit PRNG key
    rng = jax.random.PRNGKey(0)
    
    # Initialize model parameters with a dummy input
    dummy_input = jnp.zeros((1, 32), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables['params']

    @jax.jit
    def jit_forward(p, idx):
        return model.apply({'params': p}, idx)

    seq_lengths = [32, 64, 128, 256]
    results = {}

    print(f"{'Seq Len':<10} | {'First Call (ms)':<17} | {'Subsequent (ms)':<17} | {'Tokens/sec':<15} | {'Memory (MB)':<15}")
    print("-" * 85)

    for seq_len in seq_lengths:
        # Create static shaped input for this specific sequence length
        x = jnp.zeros((1, seq_len), dtype=jnp.int32)
        
        # Start memory tracking
        tracemalloc.start()
        
        # 1. FIRST CALL (Tracing + XLA Compilation)
        # JAX will trace the python code and compile it for the specific input shape (1, seq_len).
        # This is very slow.
        start_time = time.perf_counter()
        _ = jit_forward(params, x)
        # We must block_until_ready() because JAX executes asynchronously
        _ = jax.block_until_ready(_)
        end_time = time.perf_counter()
        first_call_ms = (end_time - start_time) * 1000
        
        # 2. SUBSEQUENT CALLS (Pure compiled execution)
        # Now the graph is compiled. Python code is bypassed. It runs directly on CPU.
        latencies = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = jit_forward(params, x)
            _ = jax.block_until_ready(_)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
        subsequent_call_ms = np.mean(latencies)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / (1024 * 1024)
        
        tokens_per_sec = seq_len / (subsequent_call_ms / 1000.0)
        
        print(f"{seq_len:<10} | {first_call_ms:<17.2f} | {subsequent_call_ms:<17.2f} | {tokens_per_sec:<15.2f} | {memory_mb:<15.2f}")
        
        results[str(seq_len)] = {
            "first_call_ms": first_call_ms,
            "subsequent_call_ms": subsequent_call_ms,
            "tokens_per_sec": tokens_per_sec,
            "memory_mb": memory_mb
        }
        
    with open("benchmark_jax.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to benchmark_jax.json")

if __name__ == "__main__":
    benchmark_jax()
