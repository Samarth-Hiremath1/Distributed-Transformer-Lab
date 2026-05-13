# Building a Distributed, Optimized GPT-2 on a CPU

## What I Built and Why
In the modern ML infrastructure space, it's easy to assume you need a massive GPU cluster to learn systems engineering. I wanted to prove that wrong. I built a complete production-grade ML inference and training system for a GPT-2 scale model entirely on a MacBook CPU. My goal was to demonstrate core systems optimization skills—distributed training, JAX XLA compilation, KV caching, INT8 quantization, and distributed inference serving with dynamic batching. By forcing myself to run this on a CPU, I had to deeply understand the bottlenecks that are often masked by raw GPU compute power. This project is a comprehensive lab for ML systems engineering.

## Architecture Overview
The system is designed in a modular, end-to-end pipeline:
1. **Core Modeling**: A PyTorch-based GPT-2 causal transformer built from scratch, alongside an identical JAX/Flax implementation.
2. **Distributed Training**: Using PyTorch's `DistributedDataParallel` with the Gloo backend, the system simulates multi-node training by synchronizing gradients across multiple processes via shared memory.
3. **Optimized Inference**: The generation engine implements a manual KV cache to reduce O(N^2) attention complexity to O(N) per step.
4. **Quantization & Decoding**: The PyTorch model is dynamically quantized to INT8. I implemented custom `greedy`, `top_k`, and `nucleus` decoding strategies without relying on high-level libraries.
5. **Distributed Serving**: A FastAPI load balancer routes requests to three separate worker processes. Each worker batches incoming requests within a 50ms window, runs inference, and returns the response.

## Results That Surprised Me
Working on a CPU exposed several fundamental truths about ML systems that are often abstracted away:

**DDP Efficiency**: When scaling Distributed Data Parallel to 4 processes using the Gloo backend, efficiency hovered around 60%. Because the model was tiny and running on CPU, the AllReduce gradient synchronization overhead completely dominated the computation time. On a massive GPU with NCCL, compute hides communication; on a CPU, communication is the bottleneck.

**JAX XLA Compilation**: JAX's `jit` compilation showed significant overhead on the first run. The compiled subsequent runs were faster, but the speedup wasn't as dramatic as I've seen on GPUs. XLA's biggest wins come from kernel fusion, which is profoundly impactful on GPU memory bandwidth but less revolutionary on a CPU.

**KV Cache Speedup**: The KV cache implementation yielded an aggressively quadratic speedup curve. At sequence length 256, generation latency plummeted. Unlike hardware-specific optimizations, algorithmic improvements like caching provide undeniable benefits regardless of whether you're on an H100 or a MacBook.

**INT8 Quantization Overhead**: While dynamic INT8 quantization halved the memory footprint, it actually *increased* inference latency slightly on the Mac. Without heavily optimized CPU kernels or hardware-specific instruction sets for INT8, the overhead of dynamic casting outweighed the reduced memory bandwidth requirements.

## What I'd Build Next With Real GPU Access
If given access to a cluster of NVIDIA GPUs, I would immediately port the DDP backend from Gloo to NCCL and implement Fully Sharded Data Parallel (FSDP) to train a 7B parameter model. I would replace the basic dynamic batching with an asynchronous Continuous Batching system like vLLM, incorporating PagedAttention to eliminate memory fragmentation in the KV cache. Finally, I would compile the inference engine using TensorRT-LLM and utilize FP8 quantization (if on Hopper architecture) to maximize hardware utilization and throughput.

## Where to Find the Code
All code for this project, including the from-scratch PyTorch transformer, the distributed serving infrastructure, and the evaluation scripts, is open-source and available on GitHub. The repository includes clear instructions on how to replicate the tests and run the entire pipeline on a standard laptop. Check out the implementation and let me know your thoughts or suggestions for further optimization!
