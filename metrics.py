from prometheus_client import Histogram, Counter, Gauge, make_asgi_app

# Define metrics
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['worker_id'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

REQUESTS_TOTAL = Counter(
    'requests_total',
    'Total number of requests',
    ['worker_id', 'status']
)

ACTIVE_WORKERS = Gauge(
    'active_workers',
    'Number of active, healthy workers'
)

BATCH_SIZE = Histogram(
    'batch_size',
    'Distribution of inference batch sizes',
    buckets=[1, 2, 4, 8, 16]
)

# Helper to mount metrics on FastAPI
# Usage: app.mount("/metrics", metrics_app)
metrics_app = make_asgi_app()
