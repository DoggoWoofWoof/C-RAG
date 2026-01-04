import os
import torch
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import community as community_louvain # python-louvain
import networkx as nx
from pathlib import Path
from .engine import GraphEngine

class GraphPartitioner:
    def __init__(self, data_dir=None, output_dir=None, embedding_dir=None):
        if data_dir is None or output_dir is None:
            raise ValueError("GraphPartitioner requires 'data_dir' and 'output_dir' to be specified.")
            
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Default embedding dir to sibling 'embeddings' folder if not specified
        if embedding_dir:
            self.embedding_dir = Path(embedding_dir)
        else:
            # If output_dir is "data/wiki/graph/leiden", parent is "data/wiki/graph"
            # So embeddings should comprise "data/wiki/graph/embeddings"
            parent = self.output_dir.parent
            if parent.name == "graph":
                self.embedding_dir = parent / "embeddings"
            else:
                 # Fallback for flat structure
                 self.embedding_dir = self.output_dir / "embeddings"
                 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.ge = GraphEngine()

    def run_pipeline(self, target_size=200, method="leiden"):
        # 1. Load Graph Structure
        nodes_path = self.data_dir / "nodes.jsonl"
        edges_path = self.data_dir / "edges.jsonl"
        
        if not nodes_path.exists():
            raise FileNotFoundError(f"Missing {nodes_path}. Run ingest first.")
            
        self.ge.load_graph(edges_path, nodes_path)
        
        # 2. Compute Embeddings (if not cached)
        self.compute_node_embeddings()
        
        # 3. Partition Graph
        self.compute_partitions(target_size=target_size, method=method)
        
        # 4. Save
        save_path = self.output_dir / "full_graph.pt"
        self.ge.save(save_path)
        print(f"Saved partitioned graph to {save_path}")

    def compute_node_embeddings(self, model_name="intfloat/e5-base-v2", batch_size=32):
        emb_path = self.embedding_dir / "embeddings.pt"
        
        # 1. Try to load complete checkpoint
        if emb_path.exists():
            print(f"  • Found complete cached embeddings at {emb_path}. Loading...")
            self.ge.data.x = torch.load(emb_path, map_location="cpu")
            return

        print("  • Computing Node Embeddings with Checkpointing...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        
        # Pre-allocate tensor to avoid concatenating huge lists
        num_nodes = self.ge.data.num_nodes
        # Use float16 to save RAM/Disk (380k * 768 * 2 bytes = 580MB vs 1.1GB)
        embedding_tensor = torch.zeros((num_nodes, 768), dtype=torch.float16)
        
        # Chunking strategy
        chunk_size = 10000 
        total_chunks = (num_nodes + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(total_chunks), desc="Embedding Chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_nodes)
            
            chunk_path = self.embedding_dir / f"emb_chunk_{chunk_idx}.pt"
            
            if chunk_path.exists():
                # Load cached chunk
                chunk_emb = torch.load(chunk_path)
                embedding_tensor[start_idx:end_idx] = chunk_emb
                continue
            
            # Prepare batch texts
            batch_texts = []
            for i in range(start_idx, end_idx):
                text = self.ge.node_text_map.get(i, "")
                batch_texts.append(f"passage: {text}")
            
            # Encode
            chunk_emb_np = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            chunk_emb = torch.tensor(chunk_emb_np).to(torch.float16)
            
            # Save Chunk
            torch.save(chunk_emb, chunk_path)
            
            # Assign to main tensor
            embedding_tensor[start_idx:end_idx] = chunk_emb
            
        print("    All chunks computed. Assembling full tensor...")
        self.ge.data.x = embedding_tensor
        
        print(f"    Saving full embeddings to {emb_path}...")
        torch.save(self.ge.data.x, emb_path)



    def compute_partitions(self, target_size=200, method="leiden"):
        print(f"  • Running Partitioning (Method: {method.upper()}, Target Size: {target_size})...")
        
        if method == "metis":
            num_nodes = self.ge.data.num_nodes
            # Dynamic K Calculation
            dynamic_k = max(1, num_nodes // target_size)
            print(f"  • Metis Dynamic K: {num_nodes} nodes / {target_size} = {dynamic_k} partitions.")
            self._partition_metis(dynamic_k)
        elif method == "hybrid":
            self._partition_hybrid(target_size=target_size)
        else:
            self._partition_leiden()

    def _partition_metis(self, num_parts):
        import pymetis
        
        print(f"    Running Metis (k={num_parts})...")
        
        # 1. Convert to Adjacency List for Metis
        # PyG edge_index is [2, E]. 
        # Metis expects a list of lists where adj[i] = [neighbors of i]
        
        num_nodes = self.ge.data.num_nodes
        edge_index = self.ge.data.edge_index
        
        # Determine strict adjacency
        # Pymetis requires undirected graph usually? 
        # Let's use nx for safe adjacency list generation if memory permits, or purely torch
        
        print("    Building Adjacency List...")
        # Torch efficient way
        row, col = edge_index
        # We need to sort by row to efficiently group
        # Actually, let's just use to_networkx if we already rely on it, but pymetis is faster without nx overhead?
        # Let's stick to list of lists for safety as pymetis crashes on bad input
        
        adj_list = [[] for _ in range(num_nodes)]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        
        for s, d in zip(src, dst):
            if s != d: # No self loops for metis
                adj_list[s].append(d)
                # adj_list[d].append(s) # Assuming edge_index is already undirected or we want directed? 
                # Metis handles undirected graphs. If our edge_index is directed, we might need to symmetrize.
                # In our case, wiki edges are directed. Metis ignores direction usually or treats as undirected.
                # Let's symmetrize to be safe for "clustering".
                adj_list[d].append(s)
        
        # Deduplicate
        adj_list = [list(set(l)) for l in adj_list]
        
        print("    Calling pymetis.part_graph...")
        # n_cuts, membership
        n_cuts, membership = pymetis.part_graph(num_parts, adjacency=adj_list)
        
        print(f"    Metis finished. Cuts: {n_cuts}")
        
        self.ge.data.part_id = torch.tensor(membership, dtype=torch.long)
        self._compute_centroids()

    def _partition_leiden(self):
        import community as community_louvain
        from torch_geometric.utils import to_networkx
        
        print("    Converting to NetworkX (for Louvain/Leiden)...")
        data_for_nx = self.ge.data.clone()
        data_for_nx.x = None
        G = to_networkx(data_for_nx, to_undirected=True)
        
        print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        print("    Optimizing Modularity...")
        
        partition_dict = community_louvain.best_partition(G, random_state=42)
        
        print("    Mapping partitions...")
        membership = [partition_dict[i] for i in range(self.ge.data.num_nodes)]
        self.ge.data.part_id = torch.tensor(membership, dtype=torch.long)
        
        num_found = self.ge.data.part_id.max().item() + 1
        print(f"    Louvain found {num_found} partitions.")
        
        self._compute_centroids()

    def _partition_hybrid(self, target_size=200):
        print("    Running Hybrid Partitioning (Recursive Leiden + Island Merge)...")
        # Step 0: Initial Leiden
        self._partition_leiden()
        
        min_s = target_size // 2
        max_s = target_size
        
        # Step 1: Semantic Merge (Islands < min_s)
        # We process merging FIRST so that if a merged group becomes a giant, 
        # the subsequent split step will catch it.
        print(f"    🏝️ Step 1: Merging Islands (< {min_s} nodes)...")
        self._merge_islands(min_size=min_s, max_capacity=max_s)
        
        # Step 2: Recursive Split (Giants > max_s)
        print(f"    🌀 Step 2: Recursively Splitting Giants (> {max_s} nodes)...")
        self._recursive_split_giants(max_size=max_s)
        
        # Step 3: Cleanup Merge (New Islands < min_s)
        # Splitting giants creates new tiny fragments. We absorb them now.
        print(f"    🧹 Step 3: Cleanup Merge (New Islands < {min_s} nodes)...")
        self._merge_islands(min_size=min_s, max_capacity=max_s)
        
        # Step 4: Cluster Orphans
        # Any remaining islands (that didn't fit) get grouped together.
        self._cluster_orphans(min_size=min_s, max_capacity=max_s)
        
        # Step 5: Re-index
        print("    Examples of final partitions:")
        unique, counts = torch.unique(self.ge.data.part_id, return_counts=True)
        print(f"    Total Partitions: {len(unique)}")
        print(f"    Largest: {counts.max().item()}, Smallest: {counts.min().item()}")
        
        self._compute_centroids()

    def _recursive_split_giants(self, max_size=2000):
        import community as community_louvain
        from torch_geometric.utils import to_networkx, subgraph
        import networkx as nx

        iteration = 0
        while True:
            iteration += 1
            part_id = self.ge.data.part_id.clone()
            unique, counts = torch.unique(part_id, return_counts=True)
            
            # Identify giants
            giant_mask = counts > max_size
            giant_ids = unique[giant_mask].tolist()
            
            if len(giant_ids) == 0:
                print(f"    ✓ No giants > {max_size} remaining.")
                break
                
            print(f"    [Iter {iteration}] Found {len(giant_ids)} giants to split (Max: {counts.max().item()})...")
            
            if iteration > 10:
                print("    ⚠️ Reached max split iterations (10). Stopping.")
                break
            
            next_id = part_id.max().item() + 1
            
            split_occured = False
            
            for g_id in tqdm(giant_ids, desc=f"Splitting Giants (Iter {iteration})"):
                # Get node indices for this giant
                nodes = (part_id == g_id).nonzero().view(-1)
                
                # Check for tiny giants (singletons or weirdness)
                if len(nodes) <= 1: 
                    continue

                subset_edge_index, _ = subgraph(nodes, self.ge.data.edge_index, relabel_nodes=True)
                
                G_sub = nx.Graph()
                G_sub.add_nodes_from(range(len(nodes)))
                G_sub.add_edges_from(subset_edge_index.t().tolist())
                
                # Run Louvain
                # Resolution can decrease to force more splits? Default 1.0
                sub_part = community_louvain.best_partition(G_sub)
                
                local_p_ids = set(sub_part.values())
                
                # If Louvain failed to split (only 1 community), force METIS or Randomized?
                # For now, let's just accept it and warn, or try Metis if len=1
                # But strictly, if it returns 1, we made no progress on this giant.
                if len(local_p_ids) < 2 and len(nodes) > max_size:
                     pass # Louvain refused to split.
                else:
                    split_occured = True
                    for local_n, local_p in sub_part.items():
                        original_n = nodes[local_n]
                        # Assign NEW global IDs
                        part_id[original_n] = next_id + local_p
                    
                    next_id += len(local_p_ids)
                
            self.ge.data.part_id = part_id
            
            if not split_occured:
                 print("    ⚠️ Louvain refused to split remaining giants. Stopping.")
                 break
        
    def _merge_islands(self, min_size=50, max_capacity=200):
        # We need embeddings
        if self.ge.data.x is None:
             self.compute_node_embeddings() # Ensure loaded
             
        # Compute interim centroids
        from torch_scatter import scatter_mean
        import torch.nn.functional as F
        
        part_id = self.ge.data.part_id
        x = self.ge.data.x
        
        # 1. Get stats
        unique_ids, counts = torch.unique(part_id, return_counts=True)
        id_to_count = {uid.item(): count.item() for uid, count in zip(unique_ids, counts)}
        
        # 2. Identify Islands vs Valid
        island_mask = counts < min_size
        valid_mask = ~island_mask
        
        island_ids = unique_ids[island_mask]
        valid_ids = unique_ids[valid_mask]
        
        if len(island_ids) == 0:
            print("    ✓ No islands found.")
            return

        print(f"    Merging {len(island_ids)} islands into {len(valid_ids)} valid cores (Max Cap: {max_capacity})...")
        
        max_id = part_id.max().item()
        
        # Optim: Relap IDs to contiguous range for scatter, then map back?
        # For now, relying on dimension expansion.
        if max_id > 20000:
             print(f"    ⚠️ High max_id ({max_id}). Scatter might be slow or OOM.")

        centroids = scatter_mean(x, part_id, dim=0, dim_size=max_id + 1)
        
        # Filter strictly valid centroids
        valid_centroids = centroids[valid_ids] # [N_valid, Dim]
        valid_centroids = F.normalize(valid_centroids, p=2, dim=1)
        
        # Iterate islands and merge
        island_centroids = centroids[island_ids] # [N_island, Dim]
        island_centroids = F.normalize(island_centroids, p=2, dim=1)
        
        # Similarity Matrix: [N_island, N_valid]
        sim = torch.mm(island_centroids, valid_centroids.t())
        
        # Sort candidates for each island: [N_island, N_valid] indices
        sorted_indices = torch.argsort(sim, dim=1, descending=True) 
        
        mapping = torch.arange(max_id + 1, device=part_id.device)
        
        island_ids_list = island_ids.tolist()
        valid_ids_list = valid_ids.tolist()
        
        merged_count = 0
        leftover_count = 0
        
        for i, old_id in enumerate(island_ids_list):
            island_size = id_to_count[old_id]
            merged = False
            
            # Candidates for this island
            candidates = sorted_indices[i] 
            
            for c_idx in candidates:
                real_id = valid_ids_list[c_idx]
                current_size = id_to_count[real_id]
                
                if current_size + island_size <= max_capacity:
                    # Merge!
                    mapping[old_id] = real_id
                    id_to_count[real_id] += island_size 
                    id_to_count[old_id] = 0
                    merged = True
                    merged_count += 1
                    break
            
            if not merged:
                leftover_count += 1
                # Do NOT force merge. Leave as is.
                
        print(f"    ✓ Merged {merged_count} islands. Leftover: {leftover_count}.")
        
        # Apply mapping
        self.ge.data.part_id = mapping[part_id]
        
    def _cluster_orphans(self, min_size=50, max_capacity=200):
        print("    📦 Clustering Leftover Orphans...")
        part_id = self.ge.data.part_id
        unique_ids, counts = torch.unique(part_id, return_counts=True)
        
        # Identify remaining islands
        island_mask = counts < min_size
        island_ids = unique_ids[island_mask].tolist()

        if not island_ids:
            print("    ✓ No orphans to cluster.")
            return

        print(f"    Found {len(island_ids)} orphans. Running Semantic Greedy Packing (Cosine)...")

        # 1. Compute Centroids for these Orphans
        # Since orphans are technically partitions, we need their avg embedding.
        # But wait! 'island_ids' are partition IDs.
        # We need the node indices belonging to each orphan.
        from torch_scatter import scatter_mean
        import torch.nn.functional as F

        # Grab all node embeddings (Frozen)
        try:
             # Try getting from ge first (if loaded)
             node_emb = self.ge.get_node_embeddings() 
        except:
             # If not loaded, self.ge.data.x might be it (if we passed it in)
            if self.ge.data.x is None:
                # Emergency load? Or assume it's there. Partitioning usually implies embeddings are ready.
                print("    ⚠️ Error: Node embeddings not found for semantic packing. Falling back to ID packing.")
                self._cluster_orphans_fallback(island_ids, max_capacity)
                return
            node_emb = self.ge.data.x

        # Calculate centroids for existing partitions
        # But we only care about the orphans.
        # Create a mask for nodes in orphan partitions
        # This is expensive. Let's do it smarter. Use scatter_mean on everything, then pick orphans.
        
        print("    Computing orphan centroids...")
        full_centroids = scatter_mean(node_emb, part_id, dim=0) # [Num_Parts, Dim]
        orphan_centroids = full_centroids[island_ids] # [Num_Orphans, Dim]
        orphan_centroids = F.normalize(orphan_centroids, p=2, dim=1) # Normalize for Cosine

        # 2. Semantic Greedy Packing
        # Strategy: Pick Seed -> Find Nearest -> Pack -> Repeat
        
        # Map: Old_ID -> New_ID
        # We'll construct a dictionary `orphan_to_target`
        orphan_to_target = {}
        processed_orphans = set()
        
        # Available Counts
        orphan_sizes = {oid: counts[unique_ids == oid].item() for oid in island_ids}
        
        # New Bins
        current_bin_id = part_id.max().item() + 1
        bins_created = 0
        
        # Determine loop order? Size desc? Or random?
        import numpy as np
        # Using numpy argsort for speed if needed, but python sort is fine for ~2000 orphans
        # Sort by size usually helps bin packing, but here we prioritize semantic density.
        
        orphan_list_idx = list(range(len(island_ids)))
        
        while len(processed_orphans) < len(island_ids):
             # Pick a seed (first unprocessed)
             seed_idx = -1
             for idx in orphan_list_idx:
                 if island_ids[idx] not in processed_orphans:
                     seed_idx = idx
                     break
             
             if seed_idx == -1: break # Done
             
             seed_oid = island_ids[seed_idx]
             seed_emb = orphan_centroids[seed_idx].unsqueeze(0) # [1, Dim]
             
             # Start a new bin with the Seed
             current_bin_id += 1
             bins_created += 1
             
             orphan_to_target[seed_oid] = current_bin_id
             processed_orphans.add(seed_oid)
             current_bin_load = orphan_sizes[seed_oid]
             
             # Find compatible neighbors
             # Filter unprocessed
             remaining_indices = [i for i in orphan_list_idx if island_ids[i] not in processed_orphans]
             
             if not remaining_indices: continue
             
             remaining_tensor = orphan_centroids[remaining_indices] # [K, Dim]
             
             # Compute scores (Dot product)
             scores = torch.mm(seed_emb, remaining_tensor.t()).squeeze(0) # [K]
             
             # Sort candidates by similarity
             sorted_score_indices = torch.argsort(scores, descending=True)
             
             # Attempt to pack
             for relative_idx in sorted_score_indices:
                 real_list_idx = remaining_indices[relative_idx.item()]
                 cand_oid = island_ids[real_list_idx]
                 cand_size = orphan_sizes[cand_oid]
                 
                 if current_bin_load + cand_size <= max_capacity:
                     # Add to bin
                     orphan_to_target[cand_oid] = current_bin_id
                     processed_orphans.add(cand_oid)
                     current_bin_load += cand_size
                 else:
                     # Bin full, stop trying this bin. 
                     # (Greedy: Assume if best match doesn't fit, maybe smaller ones do? 
                     # But for simplicity, we stop or skip. Strict packing is hard. Let's skip and try next best.)
                     continue
        
        print(f"    ✓ Packed {len(island_ids)} orphans into {bins_created} semantic clusters.")
        
        # Apply updates
        max_idx = max(part_id.max().item(), current_bin_id)
        full_map = torch.arange(max_idx + 1, device=part_id.device)
        
        for k, v in orphan_to_target.items():
            full_map[k] = v
            
        self.ge.data.part_id = full_map[part_id]

    def _cluster_orphans_fallback(self, island_ids, max_capacity):
        # The old ID-based logic as backup
        print("    Using Fallback (ID-based) Packing...")
        part_id = self.ge.data.part_id
        unique_ids, counts = torch.unique(part_id, return_counts=True)
        
        next_id = part_id.max().item() + 1
        current_bin_id = next_id
        current_bin_size = 0
        orphan_update = {} 
        
        for oid in island_ids:
            size_val = counts[unique_ids == oid].item()
            if current_bin_size + size_val > max_capacity:
                current_bin_id += 1
                current_bin_size = 0
            orphan_update[oid] = current_bin_id
            current_bin_size += size_val # Fixed bug: was 'size'
            
        max_idx = max(part_id.max().item(), current_bin_id)
        full_map = torch.arange(max_idx + 1, device=part_id.device)
        for k, v in orphan_update.items():
            full_map[k] = v
        self.ge.data.part_id = full_map[part_id]

    def _compute_centroids(self):
        # Compute Centroids
        from torch_scatter import scatter_mean
        
        print("    Computing Partition Centroids...")
        # Ensure x is loaded
        if self.ge.data.x is None:
             emb_path = self.output_dir / "embeddings.pt"
             if emb_path.exists():
                 self.ge.data.x = torch.load(emb_path, map_location="cpu")
             else:
                 print("    ⚠️ Warning: No embeddings found to compute centroids.")
                 return
             
        # Re-map IDs to contiguous 0..K before computing
        unique_ids, inverse_indices = torch.unique(self.ge.data.part_id, return_inverse=True)
        self.ge.data.part_id = inverse_indices # 0..K
        
        centroids = scatter_mean(self.ge.data.x, self.ge.data.part_id, dim=0)
        self.ge.data.part_centroids = centroids
        print(f"    Centroids shape: {centroids.shape}")


