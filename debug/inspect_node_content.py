
import torch
import argparse
import sys
import os
import json

sys.path.append(os.getcwd())

from src.graph.engine import GraphEngine

def inspect_partition_content():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiki")
    parser.add_argument("--method", type=str, default="leiden")
    parser.add_argument("--part_id", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading Graph ({args.dataset}/{args.method})...")
    path = f"data/{args.dataset}/graph/{args.method}/full_graph.pt"
    
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    data = torch.load(path, weights_only=False)
    part_id = data.part_id
    
    # 1. Get Node IDs for this partition
    mask = (part_id == args.part_id)
    node_indices = mask.nonzero().view(-1).tolist()
    count = len(node_indices)
    
    print(f"\nPartition {args.part_id}: {count} nodes.")
    
    if count == 0:
        return

    # 2. Load Node Text Map (from separate file or map)
    # The GraphEngine usually manages this, but let's load raw nodes.jsonl for speed/simplicity
    # mapping index -> title
    
    nodes_path = f"data/{args.dataset}/processed/nodes.jsonl"
    print(f"Reading {nodes_path} to look up titles...")
    
    # We need to map global index 'i' to the line in jsonl?
    # Provided Ingest maintains order: Line 0 = Node 0.
    
    # Sample random 20 indices
    import random
    if count > 20:
        sample_indices = sorted(random.sample(node_indices, 20))
    else:
        sample_indices = sorted(node_indices)
    
    found_titles = []
    
    # Scan file linearly (efficient enough for 380k lines if we just skip)
    current_idx = 0
    target_idx = 0
    
    with open(nodes_path, 'r', encoding='utf-8') as f:
        for idx in range(len(sample_indices)):
            target = sample_indices[idx]
            
            # Skip lines until target
            while current_idx < target:
                f.readline()
                current_idx += 1
            
            # Read target
            line = f.readline()
            if not line: break
            
            node = json.loads(line)
            found_titles.append(f"[{target}] {node.get('title', 'NO_TITLE')} (Len: {len(node.get('text', ''))})")
            current_idx += 1
            
    print("\n=== Sample Content from Partition 0 ===")
    for t in found_titles:
        print(t)

if __name__ == "__main__":
    inspect_partition_content()
