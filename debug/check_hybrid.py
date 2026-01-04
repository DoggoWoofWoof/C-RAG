
import torch
from collections import Counter
import matplotlib.pyplot as plt
import argparse

def check_hybrid():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiki")
    parser.add_argument("--method", type=str, default="hybrid")
    args = parser.parse_args()
    
    path = f"data/{args.dataset}/graph/{args.method}/full_graph.pt"
    print(f"Loading {path}...")
    try:
        data = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print("❌ File not found yet. Partitioning likely still running.")
        return

    part_id = data.part_id
    unique, counts = torch.unique(part_id, return_counts=True)
    
    print(f"Total Partitions: {len(unique)}")
    print(f"Total Nodes: {part_id.size(0)}")
    
    counts_list = counts.tolist()
    min_size = min(counts_list)
    max_size = max(counts_list)
    avg_size = sum(counts_list) / len(counts_list)
    
    print(f"Min Size: {min_size}")
    print(f"Max Size: {max_size}")
    print(f"Avg Size: {avg_size:.2f}")
    
    # Check Giants and Islands
    giants = [c for c in counts_list if c > 200]
    islands = [c for c in counts_list if c < 100]
    
    print(f"Giants (>200): {len(giants)}")
    print(f"Islands (<100): {len(islands)}")
    
    if len(giants) == 0 and len(islands) == 0:
        print("✅ SUCCESS: Hybrid Partitioning achieved target balance!")
    else:
        print("⚠️  WARNING: Balance Constraints not fully met.")
        if giants:
            print(f"   Largest Giants: {sorted(giants, reverse=True)[:5]}")
        if islands:
             print(f"   Smallest Islands: {sorted(islands)[:5]}")

if __name__ == "__main__":
    check_hybrid()
