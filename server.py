import sys
import time
import signal
import subprocess
import requests

def main():
    processes = []
    
    print("Starting workers and load balancer...")
    
    # Start Load Balancer
    lb_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "load_balancer:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    processes.append(lb_process)
    
    # Start Workers
    for i in range(1, 4):
        port = 8000 + i
        env = {"WORKER_ID": f"worker_{i}"}
        # inherit current env to get PYTHONPATH and other vars
        import os
        worker_env = os.environ.copy()
        worker_env.update(env)
        
        p = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "worker:app", "--port", str(port)],
            env=worker_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(p)
        
    def cleanup(signum, frame):
        print("\nShutting down server cluster...")
        for p in processes:
            p.terminate()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("Waiting for all workers to be healthy...")
    # Polling load balancer status
    ready = False
    for _ in range(30):
        try:
            resp = requests.get("http://127.0.0.1:8000/status")
            if resp.status_code == 200:
                data = resp.json()
                healthy_count = sum(1 for w in data.get("workers", []) if w.get("healthy"))
                if healthy_count == 3:
                    ready = True
                    break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
        
    if not ready:
        print("Timeout waiting for workers to be healthy. Shutting down.")
        cleanup(None, None)
        
    print("Server cluster is fully operational.")
    print(" - Load Balancer: http://127.0.0.1:8000")
    print(" - Workers: ports 8001, 8002, 8003")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None)

if __name__ == "__main__":
    main()
