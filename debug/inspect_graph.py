import torch
import os

path = "data/corafull/graph/leiden/full_graph.pt"
print(f"Inspecting {path}...")
try:
    data = torch.load(path, weights_only=False)
    print("Keys:", data.keys)
    if hasattr(data, 'num_nodes'):
        print(f"Num Nodes: {data.num_nodes}")
except Exception as e:
    print(f"Error: {e}")
