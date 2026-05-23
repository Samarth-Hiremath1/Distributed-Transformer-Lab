import asyncio
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from metrics import ACTIVE_WORKERS, metrics_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_health()
    asyncio.create_task(health_check_loop())
    yield

app = FastAPI(lifespan=lifespan)

# Mount prometheus metrics endpoint
app.mount("/metrics", metrics_app)

WORKERS = [
    {"url": "http://127.0.0.1:8001", "id": "worker_1", "healthy": False, "avg_latency": float('inf'), "error_count": 0},
    {"url": "http://127.0.0.1:8002", "id": "worker_2", "healthy": False, "avg_latency": float('inf'), "error_count": 0},
    {"url": "http://127.0.0.1:8003", "id": "worker_3", "healthy": False, "avg_latency": float('inf'), "error_count": 0},
]

rr_index = 0
ACTIVE_WORKER_COUNT = 3

@app.post("/set_active_workers/{count}")
async def set_active_workers(count: int):
    global ACTIVE_WORKER_COUNT
    ACTIVE_WORKER_COUNT = min(max(count, 1), len(WORKERS))
    return {"status": "ok", "active_workers": ACTIVE_WORKER_COUNT}

async def check_health():
    async with httpx.AsyncClient(timeout=2.0) as client:
        for w in WORKERS:
            try:
                resp = await client.get(f"{w['url']}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data["error_count"] > 10:
                        w["healthy"] = False
                    else:
                        w["healthy"] = True
                        w["avg_latency"] = data["avg_latency_ms"]
                        w["error_count"] = data["error_count"]
                else:
                    w["healthy"] = False
            except Exception:
                w["healthy"] = False
                
    healthy_count = sum(1 for w in WORKERS[:ACTIVE_WORKER_COUNT] if w["healthy"])
    ACTIVE_WORKERS.set(healthy_count)

async def health_check_loop():
    while True:
        await check_health()
        await asyncio.sleep(5)

@app.post("/generate")
async def generate(request: Request, mode: str = "roundrobin"):
    global rr_index
    # Only consider workers up to ACTIVE_WORKER_COUNT
    available_pool = WORKERS[:ACTIVE_WORKER_COUNT]
    healthy_workers = [w for w in available_pool if w["healthy"]]
    
    if not healthy_workers:
        raise HTTPException(status_code=503, detail="No healthy workers available")
        
    if mode == "latency":
        target = min(healthy_workers, key=lambda w: w["avg_latency"])
    else:
        # Default to round-robin
        target = healthy_workers[rr_index % len(healthy_workers)]
        rr_index += 1

    body = await request.json()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{target['url']}/generate", json=body)
            return resp.json()
        except Exception as e:
            target["healthy"] = False # Optimistically mark unhealthy
            raise HTTPException(status_code=502, detail=f"Worker {target['id']} failed: {str(e)}")

@app.get("/status")
async def status():
    return {"workers": WORKERS, "routing_decisions": {"mode": "supported: roundrobin, latency"}}
