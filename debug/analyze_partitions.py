import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import degree

METHODS = ["leiden", "metis", "hybrid"]
BASE_PATH = "data/corafull/graph"

def analyze_method(method):
    if "/" in method:
        # manual override e.g. wiki/leiden
        tokens = method.split("/")
        dataset = tokens[0]
        m = tokens[1]
        path = f"data/{dataset}/graph/{m}/full_graph.pt"
    else:
        path = f"{BASE_PATH}/{method}/full_graph.pt"
        
    if not os.path.exists(path):
        print(f"[{method}] ❌ File not found: {path} (Checked {os.path.abspath(path)})")
        return None

    print(f"[{method}] Loading {path}...")
    try:
        data = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"[{method}] ❌ Error loading: {e}")
        return None

    # Assuming data.part_id is set
    attr_name = 'part_id' if hasattr(data, 'part_id') else 'partition_ids'
    
    if not hasattr(data, attr_name):
        print(f"[{method}] ❌ No '{attr_name}' in graph. Found keys: {data.keys}")
        return None
    
    p_ids = getattr(data, attr_name)
    if isinstance(p_ids, list):
        p_ids = torch.tensor(p_ids)
    
    if p_ids.dim() > 1:
        p_ids = p_ids.squeeze()

    unique, counts = torch.unique(p_ids, return_counts=True)
    num_parts = len(unique)
    min_nodes = counts.min().item()
    max_nodes = counts.max().item()
    avg_nodes = counts.float().mean().item()

    # 2. Semantic Similarity (Coherence) - Advanced Check
    avg_sim = 0.0
    lift = 0.0
    
    if hasattr(data, 'x') and data.x is not None:
        try:
            from torch_scatter import scatter_mean
            
            x = data.x
            # Handle part_id (N,)
            if p_ids.size(0) != x.size(0):
                 print(f"[{method}] ⚠️ Size mismatch: x {x.size(0)} vs part_id {p_ids.size(0)}. Skipping Sim.")
            else:
                max_id = p_ids.max().item()
                # 1. Compute Centroids
                centroids = scatter_mean(x, p_ids, dim=0, dim_size=max_id + 1)
                
                # Normalize
                x_norm = F.normalize(x, p=2, dim=1)
                centroids_norm = F.normalize(centroids, p=2, dim=1)
                
                # Handle NaNs from empty partitions
                x_norm = torch.nan_to_num(x_norm)
                centroids_norm = torch.nan_to_num(centroids_norm)
                
                # 2. Centroid per node
                node_centroids = centroids_norm[p_ids]
                
                # 3. Dot Product
                sims = (x_norm * node_centroids).sum(dim=1)
                
                # 4. Agg per partition
                part_avg_sim = scatter_mean(sims, p_ids, dim=0, dim_size=max_id + 1)
                active_sims = part_avg_sim[unique]
                
                avg_sim = active_sims.mean().item()
                
                # 5. Baseline (Random)
                perm = torch.randperm(x.size(0))
                x_shuffled = x_norm[perm]
                random_sims = (x_norm * x_shuffled).sum(dim=1)
                avg_random = random_sims.mean().item()
                
                lift = avg_sim - avg_random
                print(f"[{method}]    ↳ Coherence: {avg_sim:.4f} (Lift: +{lift:.4f} vs Rand {avg_random:.4f})")
                
        except ImportError:
            print(f"[{method}] ⚠️ torch_scatter not installed. Using simple sampling.")
            pass


    # 3. Edge Cut Ratio (Topology)
    # edges = data.edge_index
    # internal_edges = (p_ids[edges[0]] == p_ids[edges[1]])
    # cut_ratio = (~internal_edges).sum() / edges.size(1)

    print(f"[{method}] ✅ Parts: {num_parts} | Sizes: {min_nodes}-{max_nodes} (Avg {avg_nodes:.1f}) | Sim: {avg_sim:.4f}")
    
    return {
        "method": method.capitalize(),
        "partitions": num_parts,
        "min": min_nodes,
        "max": max_nodes,
        "avg": round(avg_nodes, 1),
        "sim": f"{avg_sim:.3f}",
        "lift": f"+{lift:.3f}"
    }

def main():
    # Define targets explicitly, including wiki/leiden and wiki/hybrid
    target_methods = ["leiden", "metis", "hybrid", "wiki/leiden", "wiki/hybrid", "wiki/metis"]
    
    print(f"Analyzing methods: {target_methods}...")
    
    results = []
    for m in target_methods:
        res = analyze_method(m)
        if res:
            results.append(res)
    
    if not results:
        print("No results found.")
        return

    # Dynamic column width for method name
    max_name = max(len(r['method']) for r in results)
    name_width = max(max_name + 2, 12)

    print("\n" + "="*100)
    print(f"{'Method':<{name_width}} | {'Parts':<6} | {'Min':<5} | {'Max':<5} | {'Avg':<6} | {'Sim':<6} | {'Lift'}")
    print("-" * 100)
    for r in results:
        print(f"{r['method']:<{name_width}} | {r['partitions']:<6} | {r['min']:<5} | {r['max']:<5} | {r['avg']:<6} | {r['sim']:<6} | {r['lift']}")
    print("="*100)

if __name__ == "__main__":
    main()
