import torch
from pathlib import Path
from tqdm import tqdm

def extract_coarse_edges(graph_path="data/graph/full_graph.pt", output_path="data/graph/partition_edges.pt"):
    graph_path = Path(graph_path)
    output_path = Path(output_path)
    
    print(f"Loading {graph_path}...")
    full_graph = torch.load(graph_path, weights_only=False)
    
    edge_index = full_graph.edge_index # [2, Num_Edges]
    part_id = full_graph.part_id       # [Num_Nodes]
    
    print("Mapping edges to partitions...")
    # 1. Look up partition IDs for source and target nodes
    # src, dst = edge_index[0], edge_index[1]
    # p_src = part_id[src]
    # p_dst = part_id[dst]
    
    # Efficient lookup using tensor indexing
    # Ensure all on same device (CPU is safer for memory)
    p_src = part_id[edge_index[0]]
    p_dst = part_id[edge_index[1]]
    
    print(f"  Node Edges: {edge_index.size(1)}")
    
    # 2. Stack them to shape [2, Num_Edges] pair list
    p_edges = torch.stack([p_src, p_dst], dim=0)
    
    # 3. Filter Self-Loops (p_src == p_dst)
    # We only care about connections BETWEEN partitions for the structural graph
    # (Actually, self-loops are fine for GNNs, but unique is key)
    mask = p_src != p_dst
    inter_edges = p_edges[:, mask]
    
    print(f"  Inter-Partition Node Edges: {inter_edges.size(1)}")
    
    # 4. Get Unique Edges
    # unique() on dim=1 is slow and tricky in old torch versions.
    # Transpose to [N, 2] -> unique(dim=0) -> Transpose back
    unique_p_edges = torch.unique(inter_edges.t(), dim=0).t()
    
    print(f"  Unique Partition Edges: {unique_p_edges.size(1)}")
    
    # 5. Save
    torch.save(unique_p_edges, output_path)
    print(f"Saved coarse edges to {output_path}")

if __name__ == "__main__":
    extract_coarse_edges()
