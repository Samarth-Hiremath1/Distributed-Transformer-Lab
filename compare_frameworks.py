import json
import matplotlib.pyplot as plt

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    try:
        pt_data = load_json("benchmark_pytorch.json")
        jax_data = load_json("benchmark_jax.json")
    except FileNotFoundError:
        print("Error: Run benchmark_pytorch.py and benchmark_jax.py first.")
        return

    seq_lengths = sorted([int(k) for k in pt_data.keys()])
    
    print("\n" + "="*80)
    print("FRAMEWORK COMPARISON (CPU Execution)")
    print("="*80)
    
    for seq in seq_lengths:
        seq_str = str(seq)
        print(f"\nSequence Length: {seq}")
        print(f"{'Framework':<20} | {'Latency (ms)':<15} | {'Memory (MB)':<15} | {'Tokens/sec':<15}")
        print("-" * 72)
        
        # PyTorch
        pt = pt_data[seq_str]
        print(f"{'PyTorch eager':<20} | {pt['latency_ms']:<15.2f} | {pt['memory_mb']:<15.2f} | {pt['tokens_per_sec']:<15.2f}")
        
        # JAX First Run (Tracing + Compiling)
        jx = jax_data[seq_str]
        # JAX doesn't track tokens/sec for first run meaningfully because it's mostly compile time
        first_run_tps = seq / (jx['first_call_ms'] / 1000)
        print(f"{'JAX (first run)':<20} | {jx['first_call_ms']:<15.2f} | {jx['memory_mb']:<15.2f} | {first_run_tps:<15.2f}")
        
        # JAX Compiled
        print(f"{'JAX + jit':<20} | {jx['subsequent_call_ms']:<15.2f} | {jx['memory_mb']:<15.2f} | {jx['tokens_per_sec']:<15.2f}")
        
    print("\n" + "="*80)
    print("ANALYSIS: JAX XLA vs PyTorch on CPU")
    print("="*80)
    print("Notice that while JAX + jit is generally faster than PyTorch eager execution,")
    print("the advantage on a MacBook CPU is relatively modest compared to what you would")
    print("see on an NVIDIA GPU. ")
    print("")
    print("Why? XLA's biggest performance wins come from 'kernel fusion' — compiling")
    print("multiple operations (like MatMul -> Add -> GELU) into a single GPU kernel.")
    print("This drastically reduces GPU memory bandwidth bottlenecks (HBM reads/writes).")
    print("On a CPU, memory hierarchies (L1/L2/L3 cache) and the lack of massive parallel")
    print("streaming multiprocessors mean that fusing ops provides a smaller relative")
    print("speedup. The massive 'first run' latency clearly demonstrates the cost of XLA")
    print("tracing and compilation.")
    print("="*80 + "\n")

    # Generate Plot
    pt_latencies = [pt_data[str(s)]["latency_ms"] for s in seq_lengths]
    jax_latencies = [jax_data[str(s)]["subsequent_call_ms"] for s in seq_lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, pt_latencies, marker='o', label='PyTorch Eager', linewidth=2)
    plt.plot(seq_lengths, jax_latencies, marker='s', label='JAX + jit (Compiled)', linewidth=2)
    
    plt.title('Inference Latency vs Sequence Length (MacBook CPU)')
    plt.xlabel('Sequence Length (tokens)')
    plt.ylabel('Latency (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300)
    print("Saved comparison_plot.png")

if __name__ == "__main__":
    main()
