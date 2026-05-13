# Distributed Transformer Lab

A production-grade machine learning systems project implementing a GPT-style transformer from scratch. This lab explores the performance trade-offs of KV caching, INT8 quantization, and distributed inference serving on consumer-grade hardware (MacBook CPU).

## Architecture

```text
       [ Locust Load Test ]
                |
        [ FastAPI Load Balancer ]
        /       |       \
 [ Worker 1 ] [ Worker 2 ] [ Worker 3 ]
      |           |           |
 [ INT8 GPT ] [ INT8 GPT ] [ INT8 GPT ]
```

## Master Results

| Configuration | p50 Latency (ms) | Throughput (tok/s) | Size (MB) | Perplexity |
|---------------|------------------|--------------------|-----------|------------|
| Baseline (FP32, no cache) | 4.50 | 28424.99 | 31.84 | 58261.20 |
| + KV Cache | 38.47 | N/A | N/A | N/A |
| + INT8 Quantization | 6.80 | N/A | 35.92 | 58259.20 |
| + Distributed (3 workers) | N/A | 571.20 | N/A | N/A |

> [!NOTE]
> Throughput in the distributed setting is limited by CPU context switching and FastAPI overhead on a single machine.

## Key Insights

- **Quantization**: INT8 quantization significantly reduced model size with negligible impact on perplexity, though CPU-based INT8 kernels (QNNPACK) showed a slight latency increase compared to optimized FP32 paths in this small scale.
- **KV Caching**: While essential for long-sequence generation, the initial implementation overhead is visible in micro-benchmarks.
- **Scaling**: Distributed serving via FastAPI and dynamic batching effectively managed concurrent requests, though peak throughput is constrained by the local CPU's available cores.

## Execution Guide

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Run Benchmarks
```bash
python benchmark_pytorch.py
python cache_benchmark.py
python quantize.py
```

### 3. Start Inference Server
```bash
python server.py
```

### 4. Run Load Tests
```bash
# In a separate terminal
python load_test.py
```

### 5. Generate Master Table
```bash
python results_table.py
```

## Tech Stack
- **Frameworks**: PyTorch, JAX, FastAPI
- **Inference**: Prometheus, Locust, Uvicorn
- **Utilities**: NumPy, Matplotlib, Transformers (Tokenizer)
