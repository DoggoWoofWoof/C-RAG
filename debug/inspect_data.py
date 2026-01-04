import torch
import json
import numpy as np

def inspect():
    # 1. Inspect generated data
    try:
        with open("data/train.json", "r") as f:
            data = json.load(f)
            print(f"Total Samples: {len(data)}")
            if len(data) > 0:
                print("\n--- Example 1 ---")
                print(json.dumps(data[0], indent=2))
                print("\n--- Explanation ---")
                print("Question: Synthetic query generated from anchor nodes.")
                print("Partition ID: The answer/target partition.")
                print("Anchor Nodes: The specific nodes in the partition that inspired the question.")
    except FileNotFoundError:
        print("data/train.json not found.")

    # 2. Inspect Partition Sizes
    print("\n--- Partition Stats ---")
    try:
        graph = torch.load("data/graph/full_graph.pt", weights_only=False)
        part_ids = graph.part_id.numpy()
        
        # Count nodes per partition
        unique, counts = np.unique(part_ids, return_counts=True)
        size_map = dict(zip(unique, counts))
        
        sizes = list(size_map.values())
        print(f"Total Partitions: {len(sizes)}")
        print(f"  - Max Size: {max(sizes)}")
        print(f"  - Min Size: {min(sizes)}")
        print(f"  - Avg Size: {np.mean(sizes):.2f}")
        print(f"  - Median Size: {np.median(sizes)}")
        
        less_than_5 = sum(1 for s in sizes if s < 5)
        print(f"  - Partitions with < 5 nodes: {less_than_5} ({less_than_5/len(sizes):.1%})")
        
    except Exception as e:
        print(f"Error loading graph: {e}")

if __name__ == "__main__":
    inspect()
