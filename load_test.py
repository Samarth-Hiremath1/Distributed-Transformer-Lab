import os
import csv
import json
import requests
import subprocess
from locust import HttpUser, task, between

# Locust User
class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Fast requests
    
    @task(4)
    def short_prompt(self):
        self.client.post("/generate", json={"prompt": "Hello world", "max_tokens": 10})
        
    @task(1)
    def long_prompt(self):
        self.client.post("/generate", json={"prompt": "Once upon a time in a galaxy", "max_tokens": 50})

if __name__ == "__main__":
    results = []
    
    print("Starting Automated Load Tests (60s per scenario)...")
    
    scenarios = [
        {"name": "Scenario A", "workers": 1},
        {"name": "Scenario B", "workers": 2},
        {"name": "Scenario C", "workers": 3},
    ]
    
    for scenario in scenarios:
        workers = scenario["workers"]
        print(f"\n--- Running {scenario['name']} ({workers} active workers) ---")
        
        # Configure load balancer
        try:
            resp = requests.post(f"http://127.0.0.1:8000/set_active_workers/{workers}")
            if resp.status_code == 200:
                print(f"Configured load balancer for {workers} workers.")
            else:
                print(f"Failed to configure load balancer: HTTP {resp.status_code}")
        except Exception as e:
            print(f"Failed to connect to load balancer: {e}")
            print("Ensure server.py is running in another terminal!")
            exit(1)
            
        # Run locust headlessly
        cmd = [
            "locust",
            "-f", __file__,
            "--headless",
            "-u", "20",
            "-r", "20",
            "--run-time", "60s",
            "--host", "http://127.0.0.1:8000",
            "--csv", f"locust_results_{workers}"
        ]
        
        print("Running Locust test... please wait 60s.")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Read CSV using standard library
        stats_file = f"locust_results_{workers}_stats.csv"
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Name"] == "Aggregated":
                        reqs = float(row.get("Requests", 0))
                        fails = float(row.get("Failures", 0))
                        
                        def safe_float(val):
                            if val == 'N/A' or val == '': return 0.0
                            return float(val)
                            
                        rps = safe_float(row.get("Requests/s", 0))
                        p50 = safe_float(row.get("50%", 0))
                        p95 = safe_float(row.get("95%", 0))
                        error_rate = (fails / reqs) * 100 if reqs > 0 else 0
                        
                        print(f"Results: {rps:.2f} rps, p50: {p50}ms, p95: {p95}ms, error_rate: {error_rate:.2f}%")
                        
                        results.append({
                            "workers": workers,
                            "rps": rps,
                            "p50_ms": p50,
                            "p95_ms": p95,
                            "error_rate_%": error_rate
                        })
                        break
            
            # Cleanup CSVs
            for suffix in ["_stats.csv", "_stats_history.csv", "_failures.csv", "_exceptions.csv"]:
                f = f"locust_results_{workers}{suffix}"
                if os.path.exists(f):
                    os.remove(f)
        
    print("\n" + "="*60)
    print("LOAD TEST RESULTS SUMMARY")
    print("="*60)
    print(f"{'workers':<10} | {'rps':<10} | {'p50_ms':<10} | {'p95_ms':<10} | {'error_rate_%':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['workers']:<10} | {r['rps']:<10.2f} | {r['p50_ms']:<10.2f} | {r['p95_ms']:<10.2f} | {r['error_rate_%']:<15.2f}")
        
    with open("load_test_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaved load_test_results.json")
