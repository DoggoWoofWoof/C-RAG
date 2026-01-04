import subprocess
import time
import sys
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[Executor] Running: {cmd}")
    start = time.time()
    # Using python -m src.main to rely on relative imports
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
    else:
        elapsed = time.time() - start
        print(f"✅ Success ({elapsed:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description="Run Experiment Matrix")
    parser.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"], help="Target Dataset")
    parser.add_argument("--method", type=str, default=None, help="Filter to specific partition method (e.g. 'hybrid')")
    args = parser.parse_args()

    print(f"🚀 Starting Experiment Matrix Execution (Dataset: {args.dataset.upper()})")
    
    # Scan for available partitions
    base_dir = Path(f"data/{args.dataset}/graph")
    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return

    methods = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "embeddings" and (d / "full_graph.pt").exists()]
    
    if args.method:
        if args.method in methods:
            methods = [args.method]
        else:
            print(f"❌ Method '{args.method}' not found in {methods}")
            return

    if not methods:
        print(f"❌ No valid partition graphs found in {base_dir}")
        return
        
    print(f"Found Partition Methods: {methods}")
    
    experiments = [
        # Baseline
        ("Experiment A (MLP + InfoNCE)", "python -m src.main train --model_type mlp --loss_type infonce --epochs 5"),
        
        # Multi-Label
        ("Experiment B (MLP + BCE)",     "python -m src.main train --model_type mlp --loss_type bce --epochs 5"),
        
        # GNN Variants (InfoNCE)
        ("Experiment C (GCN + InfoNCE)", "python -m src.main train --model_type gcn --loss_type infonce --epochs 5"),
        ("Experiment D (SAGE + InfoNCE)", "python -m src.main train --model_type sage --loss_type infonce --epochs 5"),
        ("Experiment E (GIN + InfoNCE)",  "python -m src.main train --model_type gin --loss_type infonce --epochs 5"),

        # GNN Variants (BCE)
        ("Experiment F (GCN + BCE)",     "python -m src.main train --model_type gcn --loss_type bce --epochs 5"),
        ("Experiment G (SAGE + BCE)",    "python -m src.main train --model_type sage --loss_type bce --epochs 5"),
        ("Experiment H (GIN + BCE)",     "python -m src.main train --model_type gin --loss_type bce --epochs 5"),
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
            
        # Explicitly pass dataset to ensure metadata/logging is correct
        extra_args = f"--graph_path {graph_path} --train_path {train_path} --dataset {args.dataset} --method {method}"
        
        for name, cmd in experiments:
             full_cmd = f"{cmd} {extra_args}"
             print(f"\n------------------------------------------------------------")
             print(f"▶ STARTING {name} [{method}]")
             print(f"------------------------------------------------------------")
             run_cmd(full_cmd)
        
    print("\n🎉 All Experiments Completed.")

if __name__ == "__main__":
    main()
