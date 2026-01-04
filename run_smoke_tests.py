import subprocess
import time
import sys
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[Executor] Running: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
    else:
        elapsed = time.time() - start
        print(f"✅ Success ({elapsed:.1f}s)")

def main():
    print("🚀 Starting SMOKE TEST Matrix Execution (All Models, 10 Steps)")
    
    # Scan for available partitions in data/wiki/graph
    base_dir = Path("data/wiki/graph")
    methods = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "embeddings" and (d / "full_graph.pt").exists()]
    
    if not methods:
        print("❌ No valid partition graphs found in data/wiki/graph/")
        return
        
    print(f"Found Partition Methods: {methods}")

    experiments = [
        # Baseline
        ("Experiment A (MLP + InfoNCE)", "python -m src.main train --model_type mlp --loss_type infonce --epochs 1"),
        # Multi-Label
        ("Experiment B (MLP + BCE)",     "python -m src.main train --model_type mlp --loss_type bce --epochs 1"),
        # GNN Variants (InfoNCE)
        ("Experiment C (GCN + InfoNCE)", "python -m src.main train --model_type gcn --loss_type infonce --epochs 1"),
        ("Experiment D (SAGE + InfoNCE)", "python -m src.main train --model_type sage --loss_type infonce --epochs 1"),
        ("Experiment E (GIN + InfoNCE)",  "python -m src.main train --model_type gin --loss_type infonce --epochs 1"),
        # GNN Variants (BCE)
        ("Experiment F (GCN + BCE)",     "python -m src.main train --model_type gcn --loss_type bce --epochs 1"),
        ("Experiment G (SAGE + BCE)",    "python -m src.main train --model_type sage --loss_type bce --epochs 1"),
        ("Experiment H (GIN + BCE)",     "python -m src.main train --model_type gin --loss_type bce --epochs 1"),
    ]
    
    for method in methods:
        print(f"\n============================================================")
        print(f"🎯 PARTITION METHOD: {method.upper()}")
        print(f"============================================================")
        
        graph_path = base_dir / method / "full_graph.pt"
        train_path = base_dir / method / "train.json"
        
        if not train_path.exists():
            print(f"⚠️  Skipping {method}: Missing train.json in {train_path}")
            continue
            
        extra_args = f"--graph_path {graph_path} --train_path {train_path} --max_steps 10 --dataset wiki"
        
        for name, cmd in experiments:
            full_cmd = f"{cmd} {extra_args}"
            print(f"\n--- {name} [{method}] ---")
            run_cmd(full_cmd)
        
    print("\n🎉 Smoke Tests Completed.")

if __name__ == "__main__":
    main()
