import os
import time
import asyncio
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer
from model_pytorch import GPT, GPTConfig
from decoding import generate
from metrics import REQUEST_LATENCY, REQUESTS_TOTAL, BATCH_SIZE, metrics_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.model = load_model()
    state.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    state.tokenizer.pad_token = state.tokenizer.eos_token
    state.batch_task = asyncio.create_task(batch_processor())
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", metrics_app)

# Global state
class WorkerState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.worker_id = os.environ.get("WORKER_ID", "worker_0")
        
        # Batching state
        self.queue = []
        self.batch_window_ms = 50
        self.batch_task = None
        
        # Health tracking
        self.requests_served = 0
        self.total_latency_ms = 0
        self.error_count = 0

state = WorkerState()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20

def load_model():
    print(f"[{state.worker_id}] Loading model...")
    torch.backends.quantized.engine = 'qnnpack'
    
    config = GPTConfig(
        n_layer=4, n_head=4, d_model=128, d_ff=512, block_size=256, vocab_size=50257
    )
    model_fp32 = GPT(config)
    model_fp32.eval()
    
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    return model_int8

async def batch_processor():
    while True:
        await asyncio.sleep(state.batch_window_ms / 1000.0)
        
        if not state.queue:
            continue
            
        # Extract up to a max batch size (e.g., 8)
        batch = state.queue[:8]
        state.queue = state.queue[8:]
        
        BATCH_SIZE.labels(worker_id=state.worker_id).observe(len(batch))
        
        try:
            # Prepare inputs
            prompts = [item[0].prompt for item in batch]
            max_tokens = max(item[0].max_tokens for item in batch)
            
            # Tokenize with padding
            # On CPU, dynamic shapes can be slow, but for simplicity we pad to max length in batch
            encoded = state.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"]
            
            t0 = time.perf_counter()
            # Generate (using greedy for simplicity and speed in the worker)
            out_idx = generate(state.model, input_ids, max_tokens, strategy="greedy")
            t1 = time.perf_counter()
            
            latency_ms = (t1 - t0) * 1000
            
            # Decode and set results
            for i, (req, future) in enumerate(batch):
                # Only return the generated part
                gen_tokens = out_idx[i, input_ids.size(1):]
                text = state.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
                
                state.requests_served += 1
                state.total_latency_ms += latency_ms
                REQUEST_LATENCY.labels(worker_id=state.worker_id).observe((t1 - t0))
                REQUESTS_TOTAL.labels(worker_id=state.worker_id, status="success").inc()
                
                future.set_result({"text": text, "latency_ms": latency_ms})
                
        except Exception as e:
            state.error_count += len(batch)
            for req, future in batch:
                REQUESTS_TOTAL.labels(worker_id=state.worker_id, status="error").inc()
                future.set_exception(e)

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    future = asyncio.Future()
    state.queue.append((req, future))
    try:
        result = await future
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    avg_lat = (state.total_latency_ms / state.requests_served) if state.requests_served > 0 else 0
    return {
        "status": "healthy",
        "requests_served": state.requests_served,
        "avg_latency_ms": avg_lat,
        "error_count": state.error_count
    }
