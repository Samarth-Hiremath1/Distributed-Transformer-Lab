import os
import json

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def main():
    # Load all results
    pytorch_res = load_json("benchmark_pytorch.json")
    cache_res = load_json("cache_benchmark_results.json")
    quant_res = load_json("quantize_results.json")
    load_res = load_json("load_test_results.json")
    
    # Extract baseline stats
    baseline_latency = pytorch_res.get("128", {}).get("latency_ms", "N/A")
    baseline_throughput = pytorch_res.get("128", {}).get("tokens_per_sec", "N/A")
    baseline_memory = pytorch_res.get("128", {}).get("memory_mb", "N/A")
    
    # Extract cache stats
    cache_latency = cache_res.get("128", {}).get("cache_ms", "N/A")
    
    # Extract quant stats
    int8_latency = quant_res.get("INT8", {}).get("latency_ms", "N/A")
    int8_memory = quant_res.get("INT8", {}).get("memory_mb", "N/A")
    int8_ppl = quant_res.get("INT8", {}).get("perplexity", "N/A")
    fp32_ppl = quant_res.get("FP32", {}).get("perplexity", "N/A")
    
    # Extract distributed stats
    dist_throughput = "N/A"
    if load_res:
        # Find 3 workers scenario
        for row in load_res:
            if row.get("workers") == 3:
                # Convert rps to approximate tokens/sec (avg 18 tokens per request roughly)
                dist_throughput = row.get("rps", 0) * 18
                break

    print("\n" + "="*85)
    print("DISTRIBUTED TRANSFORMER LAB - MASTER RESULTS")
    print("="*85)
    print(f"{'config':<30} | {'p50_ms':<10} | {'throughput (tok/s)':<20} | {'memory_mb':<10} | {'perplexity':<10}")
    print("-" * 85)
    
    def fmt(val, dec=2):
        return f"{val:.{dec}f}" if isinstance(val, (int, float)) else str(val)
        
    print(f"{'Baseline (FP32, no cache)':<30} | {fmt(baseline_latency):<10} | {fmt(baseline_throughput):<20} | {fmt(baseline_memory):<10} | {fmt(fp32_ppl):<10}")
    print(f"{'+ KV Cache':<30} | {fmt(cache_latency):<10} | {'N/A':<20} | {'N/A':<10} | {'N/A':<10}")
    print(f"{'+ INT8 Quantization':<30} | {fmt(int8_latency):<10} | {'N/A':<20} | {fmt(int8_memory):<10} | {fmt(int8_ppl):<10}")
    print(f"{'+ Distributed (3 workers)':<30} | {'N/A':<10} | {fmt(dist_throughput):<20} | {'N/A':<10} | {'N/A':<10}")
    print("="*85)

if __name__ == "__main__":
    main()
