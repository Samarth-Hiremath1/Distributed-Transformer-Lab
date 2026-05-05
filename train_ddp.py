import os
import time
import urllib.request
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Tokenizer
from model_pytorch import GPT, GPTConfig

# -----------------------------------------------------------------------------
# DDP on CPU Explanation:
#
# 1. What AllReduce does during backward pass:
# When loss.backward() is called, each process computes gradients based on its
# local batch of data. DDP hooks into the backward pass and automatically
# synchronizes these gradients across all processes using the AllReduce algorithm.
# The gradients are averaged, ensuring every model replica has the exact same
# updated weights after optimizer.step().
#
# 2. Why gradient synchronization is the bottleneck on CPU:
# The Gloo backend uses shared memory or standard networking (TCP/IP) to pass
# gradient tensors between processes. For a tiny model, the actual matrix 
# multiplication (compute) takes very little time. However, moving even small
# tensors between isolated processes via shared memory has a massive fixed 
# overhead. Thus, communication dominates compute, drastically lowering 
# scaling efficiency.
#
# 3. What NCCL does differently on GPU:
# The NCCL backend (used for GPUs) is topology-aware and leverages NVLink or
# PCIe peering. It achieves hundreds of GB/s bandwidth with microsecond 
# latencies, allowing AllReduce to be almost entirely hidden behind the 
# compute of subsequent layers during the backward pass.
#
# 4. Why scaling efficiency degrades with more processes on CPU:
# Efficiency = (Throughput_N / (N * Throughput_1)).
# Because the fixed synchronization cost is paid per step, adding more 
# processes increases the total communication volume and coordination wait
# times. On a MacBook CPU with Gloo, efficiency drops sharply to 50-70% at 
# 4 processes, whereas a multi-GPU NCCL setup often maintains >90% efficiency.
# -----------------------------------------------------------------------------

def setup():
    # Initialize the process group with 'gloo' for CPU training
    dist.init_process_group(backend="gloo")
    
def cleanup():
    dist.destroy_process_group()

def get_data(seq_len, batch_size, rank, world_size):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Simple data sharding: each rank gets a specific chunk of the dataset
    chunk_size = len(data) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    rank_data = data[start_idx:end_idx]
    
    def get_batch():
        ix = torch.randint(len(rank_data) - seq_len, (batch_size,))
        x = torch.stack([rank_data[i:i+seq_len] for i in ix])
        y = torch.stack([rank_data[i+1:i+seq_len+1] for i in ix])
        return x, y
        
    return get_batch

def main():
    setup()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        d_model=128,
        d_ff=512,
        block_size=64,
        vocab_size=50257
    )
    
    batch_size = 4
    n_steps = 50 # We use 50 steps for the scaling analysis to save time
    
    get_batch = get_data(config.block_size, batch_size, rank, world_size)
    
    model = GPT(config)
    # Wrap model with DDP
    # No device_ids since we are on CPU
    model = DDP(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    
    start_time = time.perf_counter()
    total_tokens = 0
    
    for step in range(n_steps):
        t0 = time.perf_counter()
        
        x, y = get_batch()
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward() # DDP synchronizes gradients here
        optimizer.step()
        
        t1 = time.perf_counter()
        
        dt = t1 - t0
        tokens_processed = batch_size * config.block_size
        total_tokens += tokens_processed
        
        if step % 10 == 0 or step == n_steps - 1:
            tokens_per_sec = tokens_processed / dt
            print(f"[Rank {rank}] Step {step:3d} | Loss: {loss.item():.4f} | Local Throughput: {tokens_per_sec:.2f} tok/s")
            
    # Compute and aggregate throughput
    end_time = time.perf_counter()
    total_time = end_time - start_time
    local_tps = total_tokens / total_time
    
    # We want to sum the throughputs across all ranks to get the global throughput
    tps_tensor = torch.tensor([local_tps], dtype=torch.float32)
    dist.all_reduce(tps_tensor, op=dist.ReduceOp.SUM)
    global_tps = tps_tensor.item()
    
    if rank == 0:
        print("-" * 50)
        print(f"DDP Training Complete on {world_size} processes.")
        print(f"Global Aggregated Throughput: {global_tps:.2f} tok/s")
        # Save output to be parsed by scaling_analysis.py
        print(f"__SCALING_RESULT__:{world_size}:{global_tps:.2f}")

    cleanup()

if __name__ == "__main__":
    main()
