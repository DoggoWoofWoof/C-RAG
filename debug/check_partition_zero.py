
import torch
import argparse
import sys
import os

sys.path.append(os.getcwd())

from src.graph.engine import GraphEngine

def check_partition_zero():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiki")
    parser.add_argument("--method", type=str, default="leiden")
    args = parser.parse_args()

    print(f"Loading Graph ({args.dataset}/{args.method})...")
    path = f"data/{args.dataset}/graph/{args.method}/full_graph.pt"
    
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    data = torch.load(path, weights_only=False)
    
    part_id = data.part_id
    edge_index = data.edge_index
    
    # 1. Identify Partition 0
    mask_0 = (part_id == 0)
    nodes_0 = mask_0.nonzero().view(-1)
    count_0 = nodes_0.size(0)
    
    print(f"\nAnalyzing Partition 0...")
    print(f"Total Nodes: {count_0}")
    
    if count_0 == 0:
        print("Partition 0 is empty.")
        return

    # 2. Calculate Degree for these nodes
    # We need to see how many edges connect TO or FROM these nodes.
    # Degree = In-Degree + Out-Degree
    
    # Efficient degree calc:
    # Convert edge_index to sparse or just hist
    print("Computing degrees...")
    from torch_geometric.utils import degree
    
    # degree returns (num_nodes,) tensor
    # We check both row and col for undir/dir
    d_out = degree(edge_index[0], num_nodes=data.num_nodes)
    d_in = degree(edge_index[1], num_nodes=data.num_nodes)
    d_total = d_out + d_in
    
    # Get degrees for Part 0
    deg_0 = d_total[nodes_0]
    
    # Stats
    zero_deg = (deg_0 == 0).sum().item()
    avg_deg = deg_0.mean().item()
    max_deg = deg_0.max().item()
    
    print(f"Zero-Degree Nodes: {zero_deg} ({zero_deg/count_0:.2%})")
    print(f"Avg Degree: {avg_deg:.4f}")
    print(f"Max Degree: {max_deg}")
    
    # 3. Check connectivity WITHIN Partition 0
    # Do they connect to EACH OTHER?
    print("Checking internal edges...")
    
    # Creating a set is slow for 68k, use boolean mask
    # Src in P0 AND Dst in P0
    src, dst = edge_index
    mask_src = mask_0[src]
    mask_dst = mask_0[dst]
    internal_edges = (mask_src & mask_dst).sum().item()
    
    print(f"Internal Edges (P0 <-> P0): {internal_edges}")
    if internal_edges == 0:
        print("✅ confirmed: Partition 0 has NO internal structure.")
    else:
        print(f"❌ Partition 0 has {internal_edges} internal edges. It is not purely disconnected.")

if __name__ == "__main__":
    check_partition_zero()
