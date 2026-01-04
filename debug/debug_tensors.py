import torch

def check():
    print("Loading graph...")
    # weights_only=False for PyG
    full_graph = torch.load("data/graph/full_graph.pt", weights_only=False)
    
    if hasattr(full_graph, 'part_centroids'):
        c = full_graph.part_centroids
        print(f"Centroids Shape: {c.shape}")
        print(f"Centroids Dtype: {c.dtype}")
        
    else:
        print("Attribute 'part_centroids' missing!")

if __name__ == "__main__":
    check()
