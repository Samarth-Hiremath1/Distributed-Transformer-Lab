import os
import re
import json
import subprocess
import matplotlib.pyplot as plt

def run_ddp_training(nproc):
    print(f"\nRunning DDP Training with {nproc} processes...")
    # Run torchrun via subprocess
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "train_ddp.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running torchrun for {nproc} processes:")
        print(result.stderr)
        return None
        
    # Parse the throughput from the output
    throughput = None
    for line in result.stdout.split('\n'):
        if line.startswith("__SCALING_RESULT__"):
            parts = line.split(":")
            if len(parts) == 3 and int(parts[1]) == nproc:
                throughput = float(parts[2])
                
    if throughput is None:
        print(f"Could not find scaling result in output for {nproc} processes.")
        print(result.stdout)
        
    return throughput

def main():
    print("Starting DDP Scaling Analysis")
    
    nprocs_list = [1, 2, 4]
    throughputs = []
    
    for nproc in nprocs_list:
        throughput = run_ddp_training(nproc)
        if throughput is None:
            return
        throughputs.append(throughput)
        
    # Compute efficiencies
    baseline_tps = throughputs[0]
    efficiencies = []
    
    print("\n" + "="*60)
    print("DDP SCALING ANALYSIS RESULTS")
    print("="*60)
    print(f"{'Processes':<12} | {'Tokens/sec':<15} | {'Scaling Efficiency %':<20}")
    print("-" * 55)
    
    results_dict = {}
    
    for i, nproc in enumerate(nprocs_list):
        tps = throughputs[i]
        efficiency = (tps / (nproc * baseline_tps)) * 100
        efficiencies.append(efficiency)
        
        print(f"{nproc:<12} | {tps:<15.2f} | {efficiency:<20.2f}")
        
        results_dict[str(nproc)] = {
            "tokens_per_sec": tps,
            "scaling_efficiency_pct": efficiency
        }
        
    print("="*60)
    print("\nKEY INSIGHT ON MACBOOK CPU:")
    print("Notice the significant drop in scaling efficiency at 2 and 4 processes.")
    print("On a CPU, Gloo uses shared memory for intra-machine communication, but the")
    print("AllReduce algorithm still requires synchronization barriers. Because our")
    print("model is tiny, the computation per step finishes extremely quickly.")
    print("Consequently, the time spent synchronizing gradients across processes")
    print("dominates the total step time, destroying scaling efficiency.")
    print("On a GPU cluster with NCCL and a large model, computation time vastly")
    print("exceeds communication time, maintaining much higher efficiency.")
    
    # Save JSON
    with open("scaling_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
        
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Tokens / sec', color=color)
    ax1.plot(nprocs_list, throughputs, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(nprocs_list)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Scaling Efficiency (%)', color=color)  
    ax2.plot(nprocs_list, efficiencies, marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)

    plt.title('DDP Scaling Performance on CPU (Gloo Backend)')
    fig.tight_layout()  
    plt.savefig('scaling_curve.png', dpi=300)
    print("\nSaved scaling_curve.png and scaling_results.json")

if __name__ == "__main__":
    main()
