import os
import time
import json
import urllib.request
import tracemalloc
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from model_pytorch import GPT, GPTConfig

def get_test_data(config):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # 90% train, 10% test
    n = int(0.9 * len(data))
    test_data = data[n:]
    return test_data

@torch.no_grad()
def calculate_perplexity(model, test_data, block_size):
    model.eval()
    losses = []
    
    # Calculate loss over chunks
    for i in range(0, len(test_data) - block_size, block_size):
        x = test_data[i:i+block_size].unsqueeze(0)
        y = test_data[i+1:i+block_size+1].unsqueeze(0)
        
        _, loss = model(x, targets=y)
        losses.append(loss.item())
        
        # Don't evaluate the whole test set to save time on CPU, just 100 batches
        if len(losses) >= 100:
            break
            
    mean_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(mean_loss)).item()
    return perplexity

@torch.no_grad()
def measure_latency_and_memory(model, seq_len=128, trials=10):
    model.eval()
    x = torch.randint(0, 50257, (1, seq_len))
    
    # Warmup
    for _ in range(3):
        model(x)
        
    tracemalloc.start()
    
    start = time.perf_counter()
    for _ in range(trials):
        model(x)
    end = time.perf_counter()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency_ms = ((end - start) / trials) * 1000
    memory_mb = peak / (1024 * 1024)
    
    return latency_ms, memory_mb

def get_model_size_mb(model, filename="temp_model.pt"):
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)
    return size_mb

def main():
    print("Initializing FP32 Model...")
    
    # Set quantization engine for Mac (required for Apple Silicon / Intel CPUs)
    torch.backends.quantized.engine = 'qnnpack'
    
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        d_model=128,
        d_ff=512,
        block_size=512,
        vocab_size=50257
    )
    model_fp32 = GPT(config)
    model_fp32.eval()
    
    print("Loading test data for perplexity...")
    test_data = get_test_data(config)
    
    print("Evaluating FP32 Model...")
    size_fp32 = get_model_size_mb(model_fp32)
    latency_fp32, mem_fp32 = measure_latency_and_memory(model_fp32, seq_len=128)
    ppl_fp32 = calculate_perplexity(model_fp32, test_data, config.block_size)
    
    print("Quantizing to INT8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    
    print("Evaluating INT8 Model...")
    size_int8 = get_model_size_mb(model_int8)
    latency_int8, mem_int8 = measure_latency_and_memory(model_int8, seq_len=128)
    ppl_int8 = calculate_perplexity(model_int8, test_data, config.block_size)
    
    print("\n--- Quantization Results ---")
    print(f"{'config':<10} | {'size_mb':<10} | {'latency_ms':<12} | {'memory_mb':<10} | {'perplexity':<10}")
    print(f"{'FP32':<10} | {size_fp32:<10.2f} | {latency_fp32:<12.2f} | {mem_fp32:<10.2f} | {ppl_fp32:<10.2f}")
    print(f"{'INT8':<10} | {size_int8:<10.2f} | {latency_int8:<12.2f} | {mem_int8:<10.2f} | {ppl_int8:<10.2f}")
    
    results = {
        "FP32": {
            "size_mb": size_fp32,
            "latency_ms": latency_fp32,
            "memory_mb": mem_fp32,
            "perplexity": ppl_fp32
        },
        "INT8": {
            "size_mb": size_int8,
            "latency_ms": latency_int8,
            "memory_mb": mem_int8,
            "perplexity": ppl_int8
        }
    }
    
    with open("quantize_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
