import torch
from src.graph.engine import GraphEngine
import argparse

def check_structure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiki")
    parser.add_argument("--method", type=str, default="leiden")
    args = parser.parse_args()

    path = f"data/{args.dataset}/graph/{args.method}/full_graph.pt"
    print(f"Loading {path}...")
    ge = GraphEngine()
    try:
        ge.load(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return
    
    data = ge.data
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    num_parts = data.part_id.max().item() + 1
    
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Partitions: {num_parts}")
    
    # Check Edge Index
    row, col = data.edge_index
    print(f"Self-loops: {(row == col).sum().item()}")
    
    # Check Partition Edge Counts
    p_row = data.part_id[row]
    p_col = data.part_id[col]
    
    mask_inter = (p_row != p_col)
    num_inter = mask_inter.sum().item()
    print(f"Inter-Partition Edges: {num_inter} ({num_inter/num_edges:.2%})")
    
    if num_edges > 0:
        # Get unique partitions participating in inter-edges
        p_src = p_row[mask_inter].unique()
        p_dst = p_col[mask_inter].unique()
        connected_parts = torch.cat([p_src, p_dst]).unique()
        print(f"Partitions with Inter-Edges: {connected_parts.size(0)}")
        print(f"Isolated Partitions: {num_parts - connected_parts.size(0)}")
        
        # Sample some isolated partitions
        all_parts = torch.arange(num_parts)
        is_connected = torch.isin(all_parts, connected_parts)
        isolated = all_parts[~is_connected]
        if isolated.numel() > 0:
            print(f"Sample Isolated Parts: {isolated[:10].tolist()}")
            # Check size of isolated partitions
            sizes = []
            for pid in isolated[:5]:
                size = (data.part_id == pid).sum().item()
                sizes.append(size)
            print(f"Sizes of first 5 isolated parts: {sizes}")

    print("\nAnalyzing Global Connectivity (Components)...")
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        import numpy as np
        
        # Build Adjacency Matrix
        row = row.numpy()
        col = col.numpy()
        data_ones = np.ones(row.shape[0])
        adj = sp.coo_matrix((data_ones, (row, col)), shape=(num_nodes, num_nodes))
        
        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        print(f"Total Connected Components: {n_components}")
        
        # Analyze Component Sizes
        unique, counts = np.unique(labels, return_counts=True)
        # Sort by size
        counts.sort()
        biggest = counts[-1]
        singletons = (counts == 1).sum()
        small_islands = (counts <= 5).sum()
        
        print(f"Largest Component Size (GCC): {biggest} ({biggest/num_nodes:.2%})")
        print(f"Number of Singleton Nodes (Size 1): {singletons}")
        print(f"Number of Small Islands (Size <= 5): {small_islands}")
        
        if num_parts - connected_parts.size(0) > 0:
            print(f"  -> This explains why so many partitions are isolated.")
        else:
            print(f"  -> Graph is well-connected (0 Isolated Partitions).")
            
    except ImportError:
        print("Scipy not installed. Skipping component analysis.")

if __name__ == "__main__":
    check_structure()
