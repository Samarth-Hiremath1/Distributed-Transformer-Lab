import time
import json
import urllib.request
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model_pytorch import GPT, GPTConfig

def get_data(seq_len, batch_size):
    # Load TinyShakespeare via single urllib call, into memory
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    
    # HuggingFace allowed for tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Tokenize the entire text (might take a few seconds)
    print("Tokenizing TinyShakespeare...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Dataset has {len(data)} tokens.")
    
    # Simple batch generator for training
    def get_batch():
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        return x, y
        
    return get_batch

def main():
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        d_model=128,
        d_ff=512,
        block_size=64,
        vocab_size=50257
    )
    
    batch_size = 4
    n_steps = 100
    
    get_batch = get_data(config.block_size, batch_size)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    
    total_tokens = 0
    start_time = time.perf_counter()
    step_times = []
    
    print("\nStarting Training (Single Process)")
    print("-" * 50)
    for step in range(n_steps):
        t0 = time.perf_counter()
        
        x, y = get_batch()
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        t1 = time.perf_counter()
        
        dt = t1 - t0
        step_times.append(dt)
        tokens_processed = batch_size * config.block_size
        total_tokens += tokens_processed
        
        if step % 10 == 0 or step == n_steps - 1:
            tokens_per_sec = tokens_processed / dt
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Throughput: {tokens_per_sec:.2f} tok/s")
            
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_tokens_per_sec = total_tokens / total_time
    
    print("-" * 50)
    print(f"Training Complete.")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average Throughput: {avg_tokens_per_sec:.2f} tok/s")
    print(f"Final Loss: {loss.item():.4f}")
    
    results = {
        "final_loss": loss.item(),
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "total_time_s": total_time
    }
    with open("train_single_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
