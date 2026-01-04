import modal
import multiprocessing
import sys
import time
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import itertools
from collections import Counter, defaultdict, deque

# Fallback for local execution where torch might not be present or partial
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import gc # For explicit garbage collection
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

# --- HELPER FOR ROBUST PARTITIONING (Global for Multiprocessing) ---

def _partitioner_target(queue, n_parts, xadj, adjncy):
    import pymetis
    try:
        _, membership = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy)
        queue.put(membership)
    except Exception as e:
        print(f"      - [Subprocess] Error during partitioning: {e}", flush=True)
        queue.put(None)

def _extract_fragment_fast_rw(source_graph, target_size):
    from torch_sparse import SparseTensor
    if source_graph.num_nodes == 0 or source_graph.num_edges == 0: return None
    # Assuming source_graph is already on GPU if main process calls this
    device = source_graph.edge_index.device
    
    # On-the-fly sparse tensor creation is fast on GPU, but caching is better
    if hasattr(source_graph, 'adj_t') and source_graph.adj_t is not None:
        adj = source_graph.adj_t
    else:
        adj = SparseTensor(row=source_graph.edge_index[0], col=source_graph.edge_index[1], sparse_sizes=(source_graph.num_nodes, source_graph.num_nodes))
    
    row_ptr, _, _ = adj.csr()
    
    start_node = torch.randint(0, source_graph.num_nodes, (1,), device=device).item()
    
    # random_walk is extremely fast on CUDA
    walk = torch.ops.torch_sparse.random_walk(row_ptr, source_graph.edge_index[1], torch.tensor([start_node], device=device), target_size)[0]
    q_nodes = torch.unique(walk)
    
    if len(q_nodes) < target_size / 2: return None
    return q_nodes

def _extract_subgraph_from_adj(adj_t, node_indices, original_data):
    # Fast subgraph extraction using CSR slicing: O(num_subset_nodes * avg_degree)
    from torch_geometric.data import Data
    
    # 1. Slicing the SparseTensor to get the sub-adjacency
    # adj_t[rows, cols]
    
    # CRITICAL FIX: Ensure indices are on the correct devices for mixed CPU/GPU slicing
    
    # 1a. Slice STRUCTURE (GPU or CPU, depending on adj_t)
    adj_device = adj_t.device()
    if node_indices.device != adj_device:
        node_indices_struct = node_indices.to(adj_device)
    else:
        node_indices_struct = node_indices
        
    sub_adj = adj_t[node_indices_struct, node_indices_struct]
    row, col, _ = sub_adj.coo()
    edge_index = torch.stack([row, col], dim=0) # on adj_device
    
    # 1b. Slice FEATURES (CPU, as we moved data.x to CPU to save VRAM)
    feat_device = original_data.x.device
    if node_indices.device != feat_device:
        node_indices_feat = node_indices.to(feat_device)
    else:
        node_indices_feat = node_indices
        
    x = original_data.x[node_indices_feat]
    
    # 2. Return Data object on CPU (to avoid clogging GPU with batch queue)
    # DataLoader handles moving final batch to GPU
    data = Data(x=x.cpu(), edge_index=edge_index.cpu(), num_nodes=len(node_indices))
    
    # Note: node_indices are global IDs. The new edge_index is already re-indexed to 0..len(subset)-1 by the slicing!
    # data constructed above
    
    # Preserve necessary attributes if they exist
    if hasattr(original_data, 'node_type'):
        data.node_type = original_data.node_type[node_indices]
    if hasattr(original_data, 'global_id'):
        data.global_id = original_data.global_id[node_indices]
    else:
        # If original data doesn't have global_id, assign the absolute indices
        data.global_id = node_indices
        
    # CRITICAL: Preserve global metadata like node_types list to avoid KeyErrors during collation
    if hasattr(original_data, 'node_types'):
        data.node_types = original_data.node_types
    if hasattr(original_data, 'node_offset'):
        data.node_offset = original_data.node_offset
        
    return data

def _finalize_query_from_nodes(original_data, adj_t, global_node_indices, min_nodes):
    if not global_node_indices: return None, None
    
    # Ensure indices are a unique tensor on the correct device
    if isinstance(global_node_indices, list):
        q_global_nodes = torch.tensor(list(set(global_node_indices)), dtype=torch.long, device=original_data.x.device)
    else:
        q_global_nodes = torch.unique(global_node_indices)
        
    if len(q_global_nodes) < min_nodes: return None, None
    
    Gq = _extract_subgraph_from_adj(adj_t, q_global_nodes, original_data)
    return Gq, q_global_nodes

def are_partitions_neighbors_sparse(adj_t, nodes1, nodes2):
    """
    Checks neighbor connectivity using SparseTensor (CSR) slicing.
    adj_t: The global SparseTensor of the graph.
    nodes1, nodes2: Tensors of global node indices.
    """
    # Optimized: Use SparseTensor slicing which is implemented in C++ / CUDA
    sub_adj = adj_t[nodes1, nodes2]
    return sub_adj.nnz() > 0

def generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, coarse_to_fine_map, possible_start_edges, coarse_edge_to_fine_bridges=None, min_nodes=80, max_nodes=100):
    if coarse_part_graph.number_of_edges() == 0: raise RuntimeError("Coarse graph has no edges.")
    configurations = [(2, 2),(3, 2),(4, 2),(3, 3),(4, 3),(5, 2),(5, 3),(5, 4),(6, 3),(6, 4),(6, 5),(8, 4),(8, 5),(8, 6),(10, 5),(10, 6),(10, 7),(12, 6),(12, 7),(12, 8),(15, 7),(15, 8),(15, 9)]; random.shuffle(configurations)
    
    # Pre-computed maps passed as arguments
    
    import time
    t_start_search = time.time()
    
    for num_frags, min_coarse_parts in configurations:
        random.shuffle(possible_start_edges) # Shuffle to randomize search start
        
        # Optimization: Use pre-computed valid bridges if available
        if coarse_edge_to_fine_bridges is not None:
            # Iterate through shuffled coarse edges (c1, c2)
            # We assume possible_start_edges are already edges in the coarse graph
            # We can also just iterate keys of coarse_edge_to_fine_bridges if we want, but possible_start_edges is fine
            
            for c_idx1, c_idx2 in possible_start_edges:
                # Get valid bridges for this coarse edge
                # The map might store (c1, c2) or (c2, c1), or both if we made it symmetric.
                # Assuming we made it symmetric or cover both directions.
                bridges = coarse_edge_to_fine_bridges.get((c_idx1, c_idx2))
                if not bridges:
                     bridges = coarse_edge_to_fine_bridges.get((c_idx2, c_idx1))
                
                if not bridges: continue
                
                # Pick a random valid bridge (f1, f2)
                f1, f2 = random.choice(bridges)
                
                # Guaranteed connected!
                # checks += 1 # Technically 1 check
                
                q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
                # Standard BFS expansion to find connected fine partitions
                while queue and len(q_fine_indices) < num_frags:
                    current_fine_idx = queue.pop(0); current_c_idx = fine_to_coarse_map[current_fine_idx]
                    coarse_neighbors_and_self = list(coarse_part_graph.neighbors(current_c_idx)) + [current_c_idx]
                    potential_fine_neighbors = [fn for c_idx in coarse_neighbors_and_self for fn in coarse_to_fine_map.get(c_idx, [])]
                    random.shuffle(potential_fine_neighbors)
                    
                    for neighbor_idx in potential_fine_neighbors:
                        if neighbor_idx not in visited and are_partitions_neighbors_sparse(adj_t, fine_part_nodes_map[current_fine_idx], fine_part_nodes_map[neighbor_idx]):
                            visited.add(neighbor_idx); queue.append(neighbor_idx); q_fine_indices.append(neighbor_idx)
                            if len(q_fine_indices) >= num_frags: break
                            
                if len(q_fine_indices) < num_frags: continue
                true_coarse_indices = {fine_to_coarse_map[f_idx] for f_idx in q_fine_indices}
                if len(true_coarse_indices) < min_coarse_parts: continue
                
                nodes_per_frag = max_nodes // num_frags; all_query_nodes = []
                for fine_idx in q_fine_indices:
                    local_nodes = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
                    if local_nodes is not None: 
                        all_query_nodes.extend(fine_part_nodes_map[fine_idx][local_nodes].tolist()) 
                
                Gq, _ = _finalize_query_from_nodes(original_data, adj_t, all_query_nodes, min_nodes)
                
                # Construct Gpos (stitched fine partitions) for consistency
                stitched_nodes = torch.cat([fine_part_nodes_map[idx] for idx in q_fine_indices])
                G_stitched = _extract_subgraph_from_adj(adj_t, stitched_nodes, original_data)
                
                duration = time.time()-t_start_search
                if duration > 5.0 and random.random() < 0.001:
                    print(f"[PROFILE] multi-coarse (optimized) match found after {checks} checks and {duration:.4f}s.", file=sys.stderr)
                metadata = {'type': 'multi-coarse-opt', 'time': duration}
                return Gq, G_stitched, list(true_coarse_indices), metadata
        
        else:
             # Should not be reached if bridges are computed
             continue
                    
    raise RuntimeError("Failed to generate multi-coarse-partition query.")

def generate_hierarchical_sample(original_data, adj_t, coarse_graphs, fine_graphs, node_to_coarse_tensor, fine_to_coarse_map, coarse_to_fine_map, coarse_edges_list, fine_part_nodes_map, coarse_part_nodes_map, coarse_part_graph, k=3, q_size_min=20, q_size_max=120, prob_k_hop=0.2, prob_single_part=0.2, prob_multi_coarse=0.4, max_gpos_nodes=4000, coarse_edge_to_fine_bridges=None):
    from torch_geometric.utils import k_hop_subgraph
    import time
    
    t0 = time.time()
    rand_choice = random.random(); device = original_data.x.device; Gq, Gpos, G_coarse_pos = None, None, None
    sample_type = "unknown"

    if rand_choice < prob_k_hop:
        sample_type = "k-hop"
        t_0_khop = time.time()
        # 1. Anchor
        anchor = torch.randint(0, original_data.num_nodes, (1,), device=device).item()
        
        # 2. Positive Context Pool (k=6)
        # We keep k=6 to define the "pool" from which we *could* sample, and to define "Gpos" if we wanted the full neighborhood.
        # But crucially, we use this to ensure our query is "inside" this region.
        subset_k_hop, _, _, _ = k_hop_subgraph(anchor, k, original_data.edge_index, relabel_nodes=False)
        
        if len(subset_k_hop) < q_size_min:
             print(f"[DEBUG] k-hop failed: k-hop size {len(subset_k_hop)} < min {q_size_min}", file=sys.stderr)
             return None

        # 3. Query Sampling: Connected BFS Blob
        # Instead of random nodes, we want a *connected* blob of size ~100 starting from anchor.
        # We explicitly run a small BFS until we hit target size.
        current_q_size = random.randint(q_size_min, q_size_max)
        
        if len(subset_k_hop) > current_q_size:
            # We perform a local BFS to get exactly `current_q_size` connected nodes
            # k_hop_subgraph doesn't limit by count, so we do a quick BFS manually.
            # Using k_hop_subgraph with a small k is an approximation, but variable size.
            query_nodes_list = [anchor]
            visited = {anchor}
            queue = deque([anchor])
            
            # For efficiency, we can limit BFS to the subset_k_hop subgraph, or just original.
            # Original is fine since we explore small number of nodes.
            while len(query_nodes_list) < current_q_size and queue:
                u = queue.popleft()
                # Get neighbors of u
                # row, col = original_data.edge_index
                # neighbors = col[row == u] is slow. `adj_t` is faster if available.
                # Assuming adj_t (SparseTensor) is available and globally accessible or prompt passed it. It is 'adj_t'.
                
                # Helper to get neighbors from SparseTensor efficiently:
                row, col, _ = adj_t[u].coo() 
                neighbors = col # for symmetric/undirected this works well as neighbors
                
                # If adj_t is not symmetric, we might miss incoming edges, but for BFS 'out' it's fine.
                # OGBN-MAG is heterogeneous converted to homogeneous usually undirected or bi-directional.
                
                for v_tn in neighbors:
                    v = v_tn.item()
                    if v not in visited:
                        visited.add(v)
                        query_nodes_list.append(v)
                        queue.append(v)
                        if len(query_nodes_list) >= current_q_size:
                            break
            
            query_nodes = torch.tensor(query_nodes_list, device=device)
        else:
            query_nodes = subset_k_hop

        # 4. Identify Coarse Partitions (Positive Context)
        # "Gpos is the concatenation of all the coarse partition of the nodes in the query"
        subset_coarse_ids = node_to_coarse_tensor[query_nodes]
        mask = subset_coarse_ids >= 0
        if mask.sum() == 0: 
            print("[DEBUG] k-hop failed: no valid coarse IDs found in query", file=sys.stderr)
            return None
            
        unique_coarse_ids, counts = torch.unique(subset_coarse_ids[mask], return_counts=True)
        
        # Optimization: Limit to top-10 interacting partitions to prevent OOM
        # User requested: "take the top 10 partitions having the most nodes of the query"
        k_partitions = 10
        if len(unique_coarse_ids) > k_partitions:
            _, top_indices = torch.topk(counts, k_partitions)
            unique_coarse_ids = unique_coarse_ids[top_indices]

            # print(f"[DEBUG] Clamped k-hop Gpos partitions to {len(unique_coarse_ids)}", file=sys.stderr)
        
        # 5. Construct Gpos from these partitions
        # CRITICAL FIX: Ensure all fetched nodes are on the correct device (CPU/GPU)
        target_device = original_data.x.device 
        pos_nodes_list = [coarse_part_nodes_map[cid.item()].to(target_device) for cid in unique_coarse_ids]
        if not pos_nodes_list: return None
        all_pos_nodes = torch.cat(pos_nodes_list).unique() 
        
        t_extract = time.time()
        try:
            # Manageable size now
            Gpos = _extract_subgraph_from_adj(adj_t, all_pos_nodes, original_data)
        except RuntimeError as e:
            if random.random() < 0.01:
                 print(f"[ERROR] k-hop Error during Gpos extraction (size {len(all_pos_nodes)}): {e}", file=sys.stderr)
            return None
            
        dur_extract = time.time() - t_extract
        
        # 6. Extract Gq 
        Gq = _extract_subgraph_from_adj(adj_t, query_nodes, original_data)
        
        # Representative coarse graph (mode)
        mode_id = torch.mode(subset_coarse_ids[mask]).values.item()
        
        # reconstruct coarse graph from original data to get features (as coarse_graphs[mode_id] now lacks them)
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[mode_id], original_data)
        
        dur_khop = time.time() - t_0_khop
        # Updated profile log
        if dur_khop > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] k-hop({k}) total:{dur_khop:.4f}s (k_hop_pool:{len(subset_k_hop)}, query_size:{len(query_nodes)}, pos_size:{len(all_pos_nodes)}, parts:{len(unique_coarse_ids)})", file=sys.stderr)
        metadata = {'type': 'k-hop', 'time': dur_khop}

    elif rand_choice < prob_k_hop + prob_single_part:
        sample_type = "single-part"
        t_single = time.time()
        
        if not fine_graphs: return None
        fine_idx = random.choice(list(fine_to_coarse_map.keys())); Gpos = fine_graphs[fine_idx]
        
        # Relaxed check or kept? Keeping check for single-part as it should be small.
        if Gpos.num_nodes > max_gpos_nodes: 
             print(f"[DEBUG] single-part failed: size {Gpos.num_nodes} > max", file=sys.stderr)
             return None
        
        q_nodes_local = _extract_fragment_fast_rw(Gpos, random.randint(q_size_min, q_size_max))
        if q_nodes_local is None: return None
        
        q_mask = torch.zeros(Gpos.num_nodes, dtype=torch.bool, device=device); q_mask[q_nodes_local] = True
        Gq = Gpos.subgraph(q_mask)
        
        # If Gpos (fine graph) lacks features, Gq will too.
        # We need to reconstruct Gq and Gpos with features.
        # Gpos nodes are: fine_part_nodes_map[fine_idx]
        pos_nodes_global = fine_part_nodes_map[fine_idx]
        Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes_global, original_data)
        
        # Recalculate Gq from the NEW Gpos or just extract using global indices
        # q_nodes_local are indices into the OLD Gpos (without features). 
        # But indices should be same if edge_index is same.
        # Safer: use global indices of query
        q_nodes_global = pos_nodes_global[q_nodes_local]
        Gq = _extract_subgraph_from_adj(adj_t, q_nodes_global, original_data)
        
        coarse_parent_idx = fine_to_coarse_map.get(fine_idx)
        if coarse_parent_idx is None: return None
        # reconstruct coarse graph with features
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[coarse_parent_idx], original_data)
        
        duration = time.time() - t_single
        if duration > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] single-part took {duration:.4f}s", file=sys.stderr)
        metadata = {'type': 'single-part', 'time': duration}
        
    elif rand_choice < prob_k_hop + prob_single_part + prob_multi_coarse:
        sample_type = "multi-coarse"
        t_multi = time.time()
        try:
            res = generate_multi_coarse_partition_query(original_data, adj_t, coarse_part_graph, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, coarse_to_fine_map, coarse_edges_list, coarse_edge_to_fine_bridges=coarse_edge_to_fine_bridges, min_nodes=q_size_min, max_nodes=q_size_max)
            
            if res is None: return None
            Gq, Gpos, coarse_indices, meta_mc = res
            metadata = meta_mc # Already computed
            # Override checking time? No, keep logic simple
                 
            all_coarse_pos_nodes = torch.cat([coarse_part_nodes_map[c_idx] for c_idx in coarse_indices])
            G_coarse_pos = _extract_subgraph_from_adj(adj_t, all_coarse_pos_nodes, original_data)
            
            # Reconstruct Gpos from original_data to ensure features are present.
            # generate_multi_coarse... returns G_stitched which is built from fine_part_nodes_map.
            # Since original_data has features, the resulting subgraph will also have them.
            
            duration = time.time() - t_multi
            if duration > 5.0 and random.random() < 0.001:
                print(f"[PROFILE] multi-coarse took {duration:.4f}s", file=sys.stderr)
        except RuntimeError: return None
        
    else:
        sample_type = "sibling-walk"
        t_walk = time.time()
        if not fine_part_nodes_map or len(fine_part_nodes_map) < 2: return None
        
        # Retry loop for sibling walk
        Gpos = None
        source_part_indices = None
        
        for attempt in range(10):
            num_frags = random.randint(2, 3); start_fine_idx = random.choice(list(fine_part_nodes_map.keys())); coarse_parent_idx = fine_to_coarse_map.get(start_fine_idx)
            if coarse_parent_idx is None: continue
            
            siblings = [idx for idx, c_idx in fine_to_coarse_map.items() if c_idx == coarse_parent_idx]
            if len(siblings) < num_frags: continue

            source_part_indices = {start_fine_idx}; queue = [start_fine_idx]
            random.shuffle(siblings) # Shuffle once per attempt is fine, or shuffle in loop
            
            # Simple BFS on siblings
            # Note: siblings list includes self, but we handle it.
            
            # Optimization: Try to find neighbors among siblings actively
            potential_neighbors = [s for s in siblings if s != start_fine_idx]
            random.shuffle(potential_neighbors)
            
            current_cluster = [start_fine_idx]
            
            # Greedy expansion within siblings
            for candidate in potential_neighbors:
                # Check if candidate connects to any in current_cluster
                # This is O(cluster_size) check
                is_connected = False
                for node in current_cluster:
                     if are_partitions_neighbors_sparse(adj_t, fine_part_nodes_map[node], fine_part_nodes_map[candidate]):
                         is_connected = True
                         break
                
                if is_connected:
                    current_cluster.append(candidate)
                    if len(current_cluster) >= num_frags: break
            
            if len(current_cluster) >= num_frags:
                source_part_indices = set(current_cluster)
                pos_nodes = torch.cat([fine_part_nodes_map[i] for i in source_part_indices])
                if len(pos_nodes) <= max_gpos_nodes:
                     Gpos = _extract_subgraph_from_adj(adj_t, pos_nodes, original_data)
                     break
                # Else loop continues (try another start)

        if Gpos is None:
            # print("[DEBUG] sibling-walk failed after retries", file=sys.stderr)
            return None
        
        nodes_per_frag = (q_size_min + q_size_max) // (2 * num_frags); all_query_global_nodes = []
        for fine_idx in source_part_indices:
            local_indices = _extract_fragment_fast_rw(fine_graphs[fine_idx], nodes_per_frag)
            if local_indices is not None:
                all_query_global_nodes.extend(fine_part_nodes_map[fine_idx][local_indices].tolist())
            
        Gq, _ = _finalize_query_from_nodes(original_data, adj_t, all_query_global_nodes, min_nodes=q_size_min)
        if Gq is None: return None
        # reconstruct coarse graph
        G_coarse_pos = _extract_subgraph_from_adj(adj_t, coarse_part_nodes_map[coarse_parent_idx], original_data)
        
        duration = time.time() - t_walk
        if duration > 5.0 and random.random() < 0.001:
            print(f"[PROFILE] sibling-walk took {duration:.4f}s", file=sys.stderr)
        metadata = {'type': 'sibling-walk', 'time': duration}
        
    if Gq is None or Gpos is None or G_coarse_pos is None: return None
    # print(f"[DEBUG] Success: {sample_type}", file=sys.stderr)
    if 'metadata' not in locals(): metadata = {'type': sample_type, 'time': time.time()-t0}
    return Gq, Gpos, G_coarse_pos, metadata

class JigsawDataset(Dataset):
    def __init__(self, original_data, adj_t, hierarchies, batch_size, steps_per_epoch):
        self.original_data = original_data
        self.adj_t = adj_t # Full GPU SparseTensor
        self.hierarchies = hierarchies
        
        # Optimize: Pre-convert node_to_coarse_map to GPU tensor for each hierarchy
        self.node_to_coarse_tensors = []
        for h_data in hierarchies:
            node_map_dict = h_data['node_to_coarse_map']
            # Create a tensor initialized with -1 or a valid default
            # Assuming nodes are 0..num_nodes-1
            # Using int32 should be enough for coarse IDs
            mapper = torch.full((original_data.num_nodes,), -1, dtype=torch.long, device=original_data.x.device)
            
            # This creation is one-time but might be slow if loop. 
            # Ideally we construct from keys/values tensors.
            # node_map_dict keys are global IDs.
            keys = torch.tensor(list(node_map_dict.keys()), dtype=torch.long)
            values = torch.tensor(list(node_map_dict.values()), dtype=torch.long)
            # Move to GPU for assignment
            keys = keys.to(original_data.x.device)
            values = values.to(original_data.x.device)
            mapper[keys] = values
            self.node_to_coarse_tensors.append(mapper)
            
            # Pre-compute reverse map and edges for multi-coarse sampling optimization
            c2f = defaultdict(list)
            
            # The hierarchy dict usually contains 'fine_to_coarse_map'
            f2c = h_data['fine_to_coarse_map']
            for f, c in f2c.items():
                c2f[c].append(f)
            h_data['precomputed_coarse_to_fine'] = c2f
            
            # Pre-compute coarse edges list
            # We copy it to a list so we can shuffle a copy later without re-creating the list from graph
            h_data['precomputed_coarse_edges'] = list(h_data['coarse_part_graph'].edges())
            
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
    
    def __len__(self):
        return self.steps_per_epoch

    def generate_sample(self):
        # Everything happens in the same process, directly on GPU tensors
        # Loop until we get a valid sample (failed samples return None)
        while True:
            # Randomly select a hierarchy each time we retry
            h_idx = random.randint(0, len(self.hierarchies) - 1)
            h_data = self.hierarchies[h_idx]
            node_mapper = self.node_to_coarse_tensors[h_idx]
            
            # Unpack hierarchy data
            try:
                sample = generate_hierarchical_sample(
                    self.original_data, self.adj_t, 
                    h_data['coarse_graphs'], h_data['fine_graphs'], 
                    node_mapper, # Passing Tensor instead of dict
                    h_data['fine_to_coarse_map'],
                    h_data['precomputed_coarse_to_fine'],
                    list(h_data['precomputed_coarse_edges']), # Pass a copy or list to shuffle inside
                    h_data['fine_part_nodes_map'], 
                    h_data['coarse_part_nodes_map'],
                    h_data['coarse_part_graph'],
                    coarse_edge_to_fine_bridges=h_data.get('coarse_edge_to_fine_bridges')
                )
                if sample:
                    # Inject h_idx into metadata
                    sample[3]['hierarchy_idx'] = h_idx
                    return sample
            except RuntimeError:
                continue # Retry on error

    def __getitem__(self, idx):
        return self.generate_sample()

def jigsaw_collate_fn(batch_list):
    from torch_geometric.data import Batch
    gqs = []
    gpos = []
    gcs = []
    metadatas = []
    
    for b in batch_list:
        if b is None: continue 
        # tuple unpacking: (Gq, Gpos, G_coarse_pos, metadata)
        if len(b) >= 4:
            item = b[0]
            if hasattr(item, 'part_id'): delattr(item, 'part_id')
            gqs.append(item)

            item = b[1]
            if hasattr(item, 'part_id'): delattr(item, 'part_id')
            gpos.append(item)

            item = b[2]
            if hasattr(item, 'part_id'): delattr(item, 'part_id')
            gcs.append(item)

            metadatas.append(b[3])
        else:
            # Drop invalid batches
            continue

    return Batch.from_data_list(gqs), Batch.from_data_list(gpos), Batch.from_data_list(gcs), metadatas

# --- MODAL SETUP ---

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core packages
        "numpy<2.0", "networkx==3.2.1", "pymetis==2022.1", "torch==2.2.1",
        "torch_geometric==2.5.2", "torch-scatter==2.1.2", "torch-sparse==0.6.18",
        # OGB dependencies
        "ogb>=1.3.6", "torchdata==0.7.1", "pandas", "PyYAML", "pydantic", "tqdm",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    # Set the library path so C++ extensions can find Torch's CUDA libs.
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/torch/lib"})
)

app = modal.App("jigsaw-mag-training-full-graph", image=image)
cache_volume = modal.Volume.from_name("jigsaw-cache-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="l4", # We primarily need GPU for the Model, not for Graph Sampling (which is CPU bound)
    volumes={"/cache": cache_volume},
    timeout=86400, # 24 hours
    cpu=10.0, # Reserve high CPU for main thread + workers
    memory=133120, # 100GB RAM for graph
)
def train(epochs, steps_per_epoch, batch_size, num_hierarchies=1, num_workers=6, fallback_mode=1):
    # --- REMOTE-ONLY IMPORTS and DEFINITIONS ---
    import itertools
    import random
    from collections import Counter, defaultdict
    import networkx as nx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential, LayerNorm
    import sys
    import queue
    import threading
    import concurrent.futures
    
    # Configure stdout/stderr buffering
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.data import Batch, Data, HeteroData
    from torch_geometric.nn import GINConv, global_mean_pool, GATConv, global_max_pool, global_add_pool
    from torch_sparse import SparseTensor

    def run_pymetis_in_subprocess(n_parts, xadj, adjncy):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_partitioner_target, args=(q, n_parts, xadj, adjncy))
        p.start()
        try:
            # 10 minute timeout for partitioning
            result = q.get(timeout=600)
        except multiprocessing.queues.Empty:
            p.kill()
            raise RuntimeError(f"METIS partitioning timed out after 600s. Subprocess stuck?")
        p.join()
        if result is None: raise RuntimeError("Partitioning failed in subprocess.")
        return None, result

    # --- MAG CONVERSION & FEATURE AUGMENTATION HELPERS ---
    def convert_hetero_to_homo(hetero_data: "HeteroData") -> Data:
        """
        Convert OGBN-MAG HeteroData -> homogeneous Data
        """
        print("  - Converting heterogeneous graph to homogeneous...")
        node_types = list(hetero_data.num_nodes_dict.keys())
        node_offset, total_nodes = {}, 0
        for nt in node_types:
            node_offset[nt] = total_nodes
            total_nodes += hetero_data.num_nodes_dict[nt]

        feat_dim = hetero_data.x_dict["paper"].size(1)
        x = torch.zeros(total_nodes, feat_dim, dtype=torch.float)
        p_start, p_end = node_offset["paper"], node_offset["paper"] + hetero_data.num_nodes_dict["paper"]
        x[p_start:p_end] = hetero_data.x_dict["paper"]

        node_type_ids = torch.zeros(total_nodes, dtype=torch.long)
        for i, nt in enumerate(node_types):
            s, e = node_offset[nt], node_offset[nt] + hetero_data.num_nodes_dict[nt]
            node_type_ids[s:e] = i

        all_ei = []
        for (src_t, rel, dst_t), ei in hetero_data.edge_index_dict.items():
            gei = ei.clone(); gei[0] += node_offset[src_t]; gei[1] += node_offset[dst_t]
            all_ei.append(gei)

        edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.empty((2, 0), dtype=torch.long)

        homo = Data(x=x, edge_index=edge_index, num_nodes=total_nodes)
        homo.node_type = node_type_ids; homo.node_types = node_types
        homo.node_offset = node_offset; homo.global_id = torch.arange(total_nodes, dtype=torch.long)
        print(f"    - Converted to homogeneous: {homo.num_nodes} nodes, {homo.edge_index.size(1)} edges")
        return homo

    class NodeFeatureAugmentor(nn.Module):
        def __init__(self, num_nodes: int, num_types: int, type_dim: int = 16, node_dim: int = 0):
            super().__init__(); self.type_emb = nn.Embedding(num_types, type_dim); self.node_dim = node_dim
            self.node_emb = nn.Embedding(num_nodes, node_dim) if node_dim > 0 else None
        @property
        def added_dim(self) -> int: return self.type_emb.embedding_dim + (self.node_emb.embedding_dim if self.node_emb is not None else 0)
        def forward(self, data: Data) -> torch.Tensor:
            pieces = [data.x, self.type_emb(data.node_type)]
            if self.node_emb is not None:
                gid = data.global_id if hasattr(data, "global_id") else torch.arange(data.num_nodes, device=data.x.device)
                pieces.append(self.node_emb(gid))
            return torch.cat(pieces, dim=1)

    def make_undirected_fast(edge_index, num_nodes):
        # This part runs on CPU initially or we can move edge_index to GPU first
        # Since we are immediately moving to GPU after, let's keep this as is for robust conversion
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0)

    # --- MODEL ARCHITECTURE ---
    class ImprovedSubgraphEncoder(torch.nn.Module):
        def __init__(self, in_neurons, hidden_neurons, output_neurons, dropout=0.1, use_residual=True, use_attention=False):
            super().__init__()
            self.use_residual = use_residual
            self.use_attention = use_attention
            self.dropout = dropout

            if not use_attention:
                nn1 = Sequential(Linear(in_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv1 = GINConv(nn1)
                nn2 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv2 = GINConv(nn2)
                nn3 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv3 = GINConv(nn3)
                nn4 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv4 = GINConv(nn4)
                nn5 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv5 = GINConv(nn5)
                nn6 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv6 = GINConv(nn6)

            self.ln1 = LayerNorm(hidden_neurons); self.ln2 = LayerNorm(hidden_neurons); self.ln3 = LayerNorm(hidden_neurons)
            self.ln4 = LayerNorm(hidden_neurons); self.ln5 = LayerNorm(hidden_neurons); self.ln6 = LayerNorm(hidden_neurons)
            self.input_proj = Linear(in_neurons, hidden_neurons) if in_neurons != hidden_neurons else None
            self.use_multi_pool = True; readout_dim = hidden_neurons * 6 * 3

            self.readout_proj = Sequential(
                Linear(readout_dim, hidden_neurons * 2), ReLU(), Dropout(dropout),
                Linear(hidden_neurons * 2, hidden_neurons), ReLU(), Dropout(dropout),
                Linear(hidden_neurons, output_neurons)
            )
            self.readout_skip = Linear(readout_dim, output_neurons)

        def forward(self, x, edge_index, batch):
            layer_outputs = []
            x_res = self.input_proj(x) if self.input_proj is not None else x

            h1 = F.relu(self.ln1(self.conv1(x, edge_index) + (x_res if self.use_residual and self.conv1(x, edge_index).shape == x_res.shape else 0)))
            layer_outputs.append(h1)
            h2 = F.relu(self.ln2(self.conv2(h1, edge_index) + (h1 if self.use_residual else 0)))
            layer_outputs.append(h2)
            h3 = F.relu(self.ln3(self.conv3(h2, edge_index) + (h2 if self.use_residual else 0)))
            layer_outputs.append(h3)
            h4 = F.relu(self.ln4(self.conv4(h3, edge_index) + (h3 if self.use_residual else 0)))
            layer_outputs.append(h4)
            h5 = F.relu(self.ln5(self.conv5(h4, edge_index) + (h4 if self.use_residual else 0)))
            layer_outputs.append(h5)
            h6 = F.relu(self.ln6(self.conv6(h5, edge_index) + (h5 if self.use_residual else 0)))
            layer_outputs.append(h6)

            pooled_representations = []
            for layer_out in layer_outputs:
                pooled_representations.extend([global_mean_pool(layer_out, batch), global_max_pool(layer_out, batch), global_add_pool(layer_out, batch)])
            h_final = torch.cat(pooled_representations, dim=1)
            return F.normalize(self.readout_proj(h_final) + self.readout_skip(h_final), dim=1)

    # --- HIERARCHICAL LOSS ---
    def info_nce_loss(queries, positives, temperature=0.1):
        logits = torch.matmul(queries, positives.T) / temperature; labels = torch.arange(len(queries), device=queries.device)
        return F.cross_entropy(logits, labels)
    def hierarchical_info_nce_loss(zq, z_fine, z_coarse, temperature=0.1, alpha=0.5):
        loss_fine = info_nce_loss(zq, z_fine, temperature); loss_coarse = info_nce_loss(zq, z_coarse, temperature)
        return (alpha * loss_fine) + ((1 - alpha) * loss_coarse)

    # --- DATA PARTITIONING AND HIERARCHY HELPERS ---
    def make_partitions(dataset, num_parts, keep_features=True):
        from torch_geometric.utils import subgraph
        from torch_geometric.data import Data
        
        # --- SANITY CHECKS ---
        if dataset.num_nodes == 0: return [], {}
        # Ensure consistency
        if dataset.x is not None and dataset.x.size(0) != dataset.num_nodes:
             raise RuntimeError(f"Dataset x size {dataset.x.size(0)} != num_nodes {dataset.num_nodes}")
        
        # Check edge index bounds (expensive but necessary for debugging this crash)
        # We are already moving to CPU below, so we can check there
        
        if dataset.num_nodes < num_parts: num_parts = dataset.num_nodes
        if num_parts <= 1: 
            d = Data(edge_index=dataset.edge_index, num_nodes=dataset.num_nodes)
            if keep_features:
                 d.x = dataset.x
                 d.y = dataset.y
            d.part_id = 0
            return [d], {0: torch.arange(dataset.num_nodes, device=dataset.edge_index.device)}
        
        # Partitioning needs to happen on CPU (pymetis requirement usually)
        # Validate edge index bounds on GPU (fast fail)
        if dataset.edge_index.numel() > 0:
             max_idx = dataset.edge_index.max().item()
             if max_idx >= dataset.num_nodes:
                  raise RuntimeError(f"Edge index max {max_idx} >= num_nodes {dataset.num_nodes}")

        edge_index_cpu = dataset.edge_index.cpu()
        if edge_index_cpu.numel() > 0 and edge_index_cpu.max() >= dataset.num_nodes:
             raise RuntimeError(f"Edge index contains indices >= num_nodes ({dataset.num_nodes}). Max: {edge_index_cpu.max()}")

        adj = SparseTensor.from_edge_index(edge_index_cpu, sparse_sizes=(dataset.num_nodes, dataset.num_nodes))
        xadj_t, adjncy_t, _ = adj.csr(); xadj, adjncy = xadj_t.tolist(), adjncy_t.tolist()
        
        _, membership = run_pymetis_in_subprocess(num_parts, xadj=xadj, adjncy=adjncy)
        
        part_graphs, part_nodes_map = [], {}
        for part_id in range(num_parts):
            node_indices = [i for i, p in enumerate(membership) if p == part_id]
            if node_indices:
                # Sanity check indices
                if max(node_indices) >= dataset.num_nodes:
                     raise RuntimeError(f"Partition {part_id} has indices >= num_nodes ({dataset.num_nodes})")
                
                nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=dataset.edge_index.device)
                part_nodes_map[part_id] = nodes_tensor
                
                # Manual Data construction to avoid implicit subgraph issues and ensure correct relabeling
                torch.cuda.synchronize()
                relabeled_edge_index, _ = subgraph(nodes_tensor, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.num_nodes)
                torch.cuda.synchronize()
                

                # Verify relabeled edge index
                if relabeled_edge_index.numel() > 0 and relabeled_edge_index.max() >= len(node_indices):
                     raise RuntimeError(f"Relabeled edge index OOB: max {relabeled_edge_index.max()} >= num_sub_nodes {len(node_indices)}")
                
                part_data = Data(edge_index=relabeled_edge_index, num_nodes=len(nodes_tensor))
                part_data.part_id = part_id
                
                
                # Copy attributes manually to be safe
                if keep_features:
                    if dataset.x is not None:
                        part_data.x = dataset.x[nodes_tensor]
                    if dataset.y is not None:
                        part_data.y = dataset.y[nodes_tensor]
                    
                    # Copy masks if present
                    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                        if hasattr(dataset, mask_name):
                            setattr(part_data, mask_name, getattr(dataset, mask_name)[nodes_tensor])
                    
                    # Copy node_type if present (essential for heterogeneous-to-homogeneous graphs)
                    if hasattr(dataset, 'node_type') and dataset.node_type is not None:
                        part_data.node_type = dataset.node_type[nodes_tensor]
                
                # Copy global metadata that doesn't need slicing
                for global_attr in ['node_types', 'node_offset', 'edge_types', 'edge_offset']:
                    if hasattr(dataset, global_attr):
                        setattr(part_data, global_attr, getattr(dataset, global_attr))

                if hasattr(dataset, 'global_id') and dataset.global_id is not None:
                    part_data.global_id = dataset.global_id[nodes_tensor]
                else:
                    # If global_id is missing, use the indices into the current dataset (which might be global)
                    part_data.global_id = nodes_tensor
                
                # OPTIMIZATION: Cache adj_t for fast random walks later
                if part_data.edge_index.numel() > 0:
                     try:
                         # Ensure we are on the correct device
                         part_data.adj_t = SparseTensor(row=part_data.edge_index[0], col=part_data.edge_index[1], 
                                                        sparse_sizes=(part_data.num_nodes, part_data.num_nodes))
                         part_data.adj_t.csr() # Pre-compute CSR
                     except Exception:
                         part_data.adj_t = None
                
                part_graphs.append(part_data)
            else:
                part_graphs.append(None)
        return part_graphs, part_nodes_map

    def build_single_hierarchy(data, num_coarse, num_fine):
        print(f"\n  * Building hierarchy with {num_coarse} coarse partitions...")
        # OPTIMIZATION: Do not keep features in hierarchy graphs to save RAM
        coarse_graphs, coarse_part_nodes_map = make_partitions(data, num_coarse, keep_features=False)
        
        # Move map to CPU for networkx graph construction, or keep as is.
        # Constructing coarse_part_graph (networkx) happens on CPU.
        node_to_coarse_map = {node_idx.item(): coarse_id for coarse_id, nodes in coarse_part_nodes_map.items() for node_idx in nodes}
        coarse_part_graph = nx.Graph()
        
        # This loop over edges is SLOW on CPU if done node-by-node in Python for 20M edges.
        # But coarse graph has fewer edges. 
        # Wait, iterating over ALL data.edge_index is required to find coarse edges.
        # Optimization: Map edges to coarse IDs using tensors, then distinct.
        
        print("    - Constructing coarse graph efficiently...", flush=True)
        # Vectorized coarse edge construction
        src, dst = data.edge_index
        # We need a tensor map from node_id -> coarse_id
        # coarse_ids tensor
        coarse_ids = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device) - 1
        for cid, nodes in coarse_part_nodes_map.items():
            coarse_ids[nodes] = cid
            
        c_src = coarse_ids[src]
        c_dst = coarse_ids[dst]
        
        # Filter inter-partition edges
        mask = (c_src != c_dst) & (c_src != -1) & (c_dst != -1)
        c_edges = torch.stack([c_src[mask], c_dst[mask]], dim=1)
        
        # Unique edges
        c_edges = torch.unique(c_edges, dim=0).cpu().numpy()
        coarse_part_graph.add_edges_from(c_edges)
        
        fine_graphs, fine_part_nodes_map, fine_to_coarse_map = [], {}, {}; fine_global_idx = 0
        iterator = tqdm(enumerate(coarse_graphs), total=len(coarse_graphs), desc="    - Creating fine partitions", unit="coarse_part", ncols=100, mininterval=30.0)
        for coarse_list_idx, coarse_graph in iterator:
            # Fix for alignment: use original part_id if available
            coarse_idx = getattr(coarse_graph, 'part_id', coarse_list_idx)
            
            if coarse_idx not in coarse_part_nodes_map: continue
            global_nodes_of_this_coarse_part = coarse_part_nodes_map[coarse_idx]
            if coarse_graph.num_nodes < (num_fine * 2) or coarse_graph.num_edges == 0:
                finer_partitions, finer_nodes_map_local = [coarse_graph], {0: torch.arange(coarse_graph.num_nodes, device=data.x.device)}
            else: 
                # print(f"DEBUG: Processing coarse_idx {coarse_idx} with {coarse_graph.num_nodes} nodes, {coarse_graph.num_edges} edges", flush=True)
                finer_partitions, finer_nodes_map_local = make_partitions(coarse_graph, num_fine, keep_features=False)
            for fine_local_idx, fine_part in enumerate(finer_partitions):
                if fine_local_idx not in finer_nodes_map_local: continue
                local_indices_in_coarse = finer_nodes_map_local[fine_local_idx]
                global_indices_for_fine = global_nodes_of_this_coarse_part[local_indices_in_coarse]
                if fine_part.num_nodes > 10 and fine_part.num_edges > 0:
                    fine_graphs.append(fine_part); fine_part_nodes_map[fine_global_idx] = global_indices_for_fine
                    fine_to_coarse_map[fine_global_idx] = coarse_idx; fine_global_idx += 1
        # Pre-compute coarse_edge -> valid fine bridges
        # This requires mapping fine partitions back to nodes
        # 'fine_part_nodes_map' has global indices for each fine partition.
        # We need a global 'fine_id' tensor.
        fine_ids = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
        for fid, nodes in fine_part_nodes_map.items():
            fine_ids[nodes] = fid
        
        # We also need coarse IDs (already computed as coarse_ids in previous block, but let's assume it might not serve purely or we need to ensure consistency)
        # Re-using 'coarse_ids' from previous block (it was a local variable, so we might need to re-compute or it's gone if out of scope? It is in scope).
        
        # Check if coarse_ids is still available?
        # Python scoping: variables in loops leak, but coarse_ids was defined at 805. It should be available.
        
        f_src = fine_ids[src]
        f_dst = fine_ids[dst]
        
        # Filter edges where fine partitions differ (potential bridges)
        # And ensure valid fine mapping
        bridge_mask = (f_src != f_dst) & (f_src != -1) & (f_dst != -1)
        
        # Filter further: only edges between DIFFERENT coarse partitions
        # (We only optimize multi-coarse sampling between coarse neighbors)
        bridge_mask = bridge_mask & (c_src != c_dst) & (c_src != -1) & (c_dst != -1)
        # Note: c_src/c_dst were defined in previous block.
        
        # Extract bridge pairs
        b_c_src = c_src[bridge_mask]
        b_c_dst = c_dst[bridge_mask]
        b_f_src = f_src[bridge_mask]
        b_f_dst = f_dst[bridge_mask]
        
        # Stack into (N, 4) tensor: [c1, c2, f1, f2]
        bridges_tensor = torch.stack([b_c_src, b_c_dst, b_f_src, b_f_dst], dim=1)
        
        # Unique bridges
        bridges_tensor = torch.unique(bridges_tensor, dim=0)
        
        # Move to CPU to build dictionary
        bridges_np = bridges_tensor.cpu().numpy()
        
        coarse_edge_to_fine_bridges = defaultdict(list)
        for r in bridges_np:
            c1, c2, f1, f2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            # Add symmetric entries because we shuffle edges and might query (c1,c2) or (c2,c1)
            coarse_edge_to_fine_bridges[(c1, c2)].append((f1, f2))
            coarse_edge_to_fine_bridges[(c2, c1)].append((f2, f1))
             
        # print(f"    - Pre-computed {len(bridges_np)} fine bridges", flush=True)

        return {
            'coarse_graphs': coarse_graphs,
            'fine_graphs': fine_graphs,
            'node_to_coarse_map': node_to_coarse_map,
            'fine_to_coarse_map': fine_to_coarse_map,
            'fine_part_nodes_map': fine_part_nodes_map,
            'coarse_part_graph': coarse_part_graph,
            'coarse_part_nodes_map': coarse_part_nodes_map,
            'coarse_edge_to_fine_bridges': dict(coarse_edge_to_fine_bridges)
        }

    def build_multiple_hierarchies(data, n_hierarchies):
        print(f"[SETUP] Building {n_hierarchies} different hierarchies for Jigsaw training...")
        hierarchies = []; iterator = tqdm(range(n_hierarchies), desc="Building hierarchies", unit="hierarchy", mininterval=30.0)
        for i in iterator:
            num_coarse = random.randint(1900, 2000)
            num_fine = random.randint(5, 10)
            print(f"  - Hierarchy {i}: Coarse={num_coarse}, Fine={num_fine}", flush=True)
            hierarchy_data = build_single_hierarchy(data, num_coarse, num_fine)
            hierarchies.append(hierarchy_data)
        return hierarchies

    # --- CORE TRAINING LOGIC ---
    device = torch.device("cuda"); print(f"[REMOTE INFO] Using device: {device}", flush=True)
    print("[REMOTE INFO] Loading OGBN-MAG (heterogeneous)...", flush=True)
    dataset = PygNodePropPredDataset(name="ogbn-mag", root="/tmp/ogbn_mag_data")
    data = convert_hetero_to_homo(dataset[0])
    
    # Initialize global_id attribute (essential for tracking nodes across partitions)
    if not hasattr(data, 'global_id'):
        data.global_id = torch.arange(data.num_nodes)
        
    print("\n[INFO] Symmetrizing full graph with SparseTensor...", flush=True)
    data.edge_index = make_undirected_fast(data.edge_index, data.num_nodes)
    print(f"  - Undirected edges: {data.edge_index.size(1)}", flush=True)

    # print(f"[INFO] Moving entire graph to GPU: {device}...", flush=True)
    # data = data.to(device)
    print(f"[INFO] Graph loaded. Keeping structure on CPU to optimize memory usage.", flush=True)
    # OPTIMIZATION: Keep data on CPU to save VRAM and utilize system RAM.
    # GPU is only used for model forward/backward and small batch data.

    TYPE_DIM = 16; NODE_DIM = 16
    augmentor = NodeFeatureAugmentor(num_nodes=data.num_nodes, num_types=len(data.node_types), type_dim=TYPE_DIM, node_dim=NODE_DIM).to(device)
    base_feat_dim = data.x.size(1); augmented_feat_dim = base_feat_dim + augmentor.added_dim
    print(f"\n[INFO] Base features: {base_feat_dim}, Augmented features: {augmented_feat_dim}", flush=True)

    # --- OPTIMIZATION: USE SPARSETENSOR INSTEAD OF DICT ---
    print("[SETUP] Building SparseTensor adjacency for efficient slicing (on CPU)...", flush=True)
    # create sparse tensor and KEEP ON CPU
    adj_t = SparseTensor(
        row=data.edge_index[0], 
        col=data.edge_index[1], 
        sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    # Pre-process CSR for fast lookup
    adj_t.csr() 
    print("  - SparseTensor built and on CPU.", flush=True)

    encoder = ImprovedSubgraphEncoder(augmented_feat_dim, 256, 128, use_attention=False, dropout=0.1, use_residual=True).to(device)
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), augmentor.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    encoder.train(); augmentor.train()
    
    # Hierarchy building involves partitioning which might use CPU/METIS, but the resulting subgraphs will be on GPU.
    # We do this WHILE data is on GPU for fast slicing
    
    CACHE_PATH = "/cache/hierarchies_optimized_v5_cpu.pt" # Bump version for CPU-only experiment
    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Found cached hierarchies at {CACHE_PATH}. Loading...", flush=True)
        try:
            hierarchies = torch.load(CACHE_PATH)
            print(f"[CACHE] Successfully loaded {len(hierarchies)} hierarchies!", flush=True)
        except Exception as e:
            print(f"[CACHE] Failed to load cache: {e}. Re-building...", flush=True)
            hierarchies = build_multiple_hierarchies(data, num_hierarchies)
            print(f"[CACHE] Saving hierarchies to {CACHE_PATH}...", flush=True)
            torch.save(hierarchies, CACHE_PATH)
            try:
                # Force sync if using Modal Volume
                volume = modal.Volume.lookup("jigsaw-cache-vol")
                volume.commit() 
            except: pass
    else:
        print(f"[CACHE] No cache found at {CACHE_PATH}. Building from scratch...", flush=True)
        hierarchies = build_multiple_hierarchies(data, num_hierarchies)
        print(f"[CACHE] Saving hierarchies to {CACHE_PATH}...", flush=True)
        torch.save(hierarchies, CACHE_PATH)
        try:
             # Force sync if using Modal Volume
             volume = modal.Volume.lookup("jigsaw-cache-vol")
             volume.commit() 
        except: pass
    print("-" * 50)

    # NOW move to CPU for parallel worker processes to avoid VRAM hogging
    print("[INFO] Creating Graph and SparseTensor copies on CPU for parallel sampling...", flush=True)
    data_cpu = data.cpu()
    adj_t_cpu = adj_t.cpu()
    
    # Update hierarchy maps if they are tensors on GPU?
    # build_single_hierarchy leaves tensors on the device of `data` (GPU).
    # We need to move them to CPU too if we want workers to pickle them efficiently without CUDA context issues.
    # CRITICAL FIX: We must NOT modify 'hierarchies' in place because dataset_gpu needs them on GPU.
    # We create a deep copy for CPU
    import copy
    print("[INFO] Creating independent CPU hierarchy copy...", flush=True)
    hierarchies_cpu = []
    
    for h_gpu in hierarchies:
        h_cpu = {}
        for k, v in h_gpu.items():
            if isinstance(v, torch.Tensor):
                h_cpu[k] = v.cpu()
            elif isinstance(v, list):
                # Check list of Data objects
                if len(v) > 0:
                     if isinstance(v[0], Data):
                         # Data objects (coarse_graphs, fine_graphs)
                         # Explicitly handle attributes including adj_t
                         new_list = []
                         for item in v:
                             if item is None: 
                                 new_list.append(None)
                                 continue
                             item_cpu = item.cpu() # Moves standard attributes
                             # Manually move adj_t if present
                             if hasattr(item, 'adj_t') and item.adj_t is not None:
                                 # item.adj_t is SparseTensor
                                 # SparseTensor.cpu() is not in-place?
                                 # SparseTensor methods usually return new object.
                                 # We need to assign it.
                                 try:
                                     # Re-create or move? SparseTensor has .to()
                                     item_cpu.adj_t = item.adj_t.cpu()
                                 except Exception:
                                     pass
                             new_list.append(item_cpu)
                         h_cpu[k] = new_list
                     elif hasattr(v[0], 'cpu'):
                         h_cpu[k] = [item.cpu() for item in v]
                     else:
                         h_cpu[k] = v # Copy list of fallback
                else: 
                     h_cpu[k] = []
            elif isinstance(v, dict):
                 # part_nodes_map
                 new_dict = {}
                 for subk, subv in v.items():
                     if isinstance(subv, torch.Tensor):
                         new_dict[subk] = subv.cpu()
                     else:
                         new_dict[subk] = subv
                 h_cpu[k] = new_dict
            else:
                 h_cpu[k] = v # ints, floats, etc.
        hierarchies_cpu.append(h_cpu)
    
    
    # 3. Create dataset for CPU workers
    # Use the CPU copies we just created
    dataset_cpu = JigsawDataset(data_cpu, adj_t_cpu, hierarchies_cpu, batch_size, steps_per_epoch * batch_size)
    
    # 4. Create dataset for Main-Thread fallback (if workers are too slow)
    dataset_fallback = None
    if fallback_mode == 1:
        print("[INFO] Fallback Mode 1: CPU. Initializing fallback dataset using CPU data...", flush=True)
        # Use CPU data for main-thread generation to avoid VRAM usage
        dataset_fallback = JigsawDataset(data_cpu, adj_t_cpu, hierarchies_cpu, batch_size, steps_per_epoch * batch_size)
    elif fallback_mode == 2:
        print("[INFO] Fallback Mode 2: GPU. Initializing fallback dataset using GPU data...", flush=True)
        # Use GPU data for main-thread generation (Faster but uses VRAM)
        dataset_fallback = JigsawDataset(data, adj_t, hierarchies, batch_size, steps_per_epoch * batch_size)
    else:
        print("[INFO] Fallback Mode 0: Disabled. Main thread will purely wait for workers.", flush=True)

    # --- CHECKPOINT RESUME LOGIC ---
    CHECKPOINT_PATH = "/cache/checkpoint_jigsaw.pt"
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[RESUME] Found checkpoint at {CHECKPOINT_PATH}. Loading...", flush=True)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            augmentor.load_state_dict(checkpoint['augmentor_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[RESUME] Successfully resumed from Epoch {start_epoch} (Last completed: {checkpoint['epoch']})", flush=True)
        except Exception as e:
            print(f"[RESUME] Failed to load checkpoint: {e}. Starting from scratch.", flush=True)
    else:
        print("[RESUME] No checkpoint found. Starting from scratch.", flush=True)

    for epoch in range(start_epoch, epochs):
        total_loss = 0

        
        # --- PARALLEL DATALOADER (Multiprocessing on CPU) ---
        batch_loader = DataLoader(
            dataset_cpu, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=jigsaw_collate_fn,
            persistent_workers=False, # Disable persistence to free memory after iterator close
            prefetch_factor=2,
            worker_init_fn=lambda worker_id: torch.set_num_threads(1) # Prevent OpenMP contention
        )
        
        iterator = iter(batch_loader)
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="step", mininterval=30.0) 
        
        # ThreadPool for non-blocking fetch from CPU loader
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Helper to submit next fetch
        def fetch_next():
            try:
                return next(iterator)
            except StopIteration:
                return None
        
        current_future = executor.submit(fetch_next)
        
        for step in pbar:
            t_wait_start = time.time()
            
            # Try to get batch from CPU workers with short timeout
            try:
                # Unpack tuple from DataLoader: (gqs, gpos, gcs, metadata_list)
                res = current_future.result(timeout=0.1) 
                # If we get here, CPU was fast enough
                if res is None: # StopIteration
                     iterator = iter(batch_loader); current_future = executor.submit(fetch_next); continue
                
                batch_data = res[0], res[1], res[2]
                batch_metadata = res[3]

                # Submit next immediately
                current_future = executor.submit(fetch_next)
                
                wait_time = time.time() - t_wait_start
                avg_gen_time = sum(m['time'] for m in batch_metadata) / len(batch_metadata)
                type_counts = Counter(m['type'] for m in batch_metadata)
                
            except concurrent.futures.TimeoutError:
                # Workers are slow! Check fallback mode
                if fallback_mode == 0 or dataset_fallback is None:
                    # Mode 0: Just block and wait for the worker
                    # print("[WARN] Worker starvation. Waiting...", file=sys.stderr)
                    try:
                        res = current_future.result() # Infinite block
                        if res is None: 
                             iterator = iter(batch_loader); current_future = executor.submit(fetch_next); continue
                        batch_data = (res[0], res[1], res[2])
                        batch_metadata = res[3]
                        current_future = executor.submit(fetch_next)
                        
                        wait_time = time.time() - t_wait_start
                        avg_gen_time = sum(m['time'] for m in batch_metadata) / len(batch_metadata)
                        type_counts = Counter(m['type'] for m in batch_metadata)
                    except Exception as e:
                        print(f"[ERROR] Worker failed: {e}")
                        break
                else:
                    # Mode 1 (CPU) or Mode 2 (GPU) fallback
                    # print(f"[WARN] Worker starvation (>{time.time()-t_wait_start:.2f}s). Generating on Main Thread...", file=sys.stderr)
                    
                    fallback_samples = []
                    batch_metadata = []
                    t_fallback_start = time.time()
                    for _ in range(batch_size):
                        s = dataset_fallback.generate_sample()
                        if s: 
                             # Sanitize s[:3]
                             cleaned_parts = []
                             for item in s[:3]:
                                 if hasattr(item, 'part_id'): delattr(item, 'part_id')
                                 cleaned_parts.append(item)
                             
                             fallback_samples.append(tuple(cleaned_parts))
                             batch_metadata.append(s[3])
                    
                    if not fallback_samples: continue
                    # Collate manually
                    batch_data = (Batch.from_data_list([s[0] for s in fallback_samples]), 
                                  Batch.from_data_list([s[1] for s in fallback_samples]), 
                                  Batch.from_data_list([s[2] for s in fallback_samples]))
                    
                    # Mark source
                    source_tag = 'fallback_cpu' if fallback_mode == 1 else 'fallback_gpu'
                    for m in batch_metadata: m['source'] = source_tag
                    
                    wait_time = time.time() - t_wait_start 
                    avg_gen_time = (time.time() - t_fallback_start) / len(fallback_samples)
                    type_counts = Counter(m['type'] for m in batch_metadata)

            # Log summary every 10 steps or if starvation happened
            if step % 10 == 0 or wait_time > 0.5:
                # Format counts: "k-hop:12, single:4"
                counts_str = ", ".join([f"{k}:{v}" for k, v in type_counts.items()])
                if 'fallback_cpu' in str(batch_metadata): source = "Fallback-CPU"
                elif 'fallback_gpu' in str(batch_metadata): source = "Fallback-GPU"
                else: source = "Worker-CPU"
                
                tqdm.write(f"[Step {step}] {source} | Wait:{wait_time:.3f}s | GenAvg:{avg_gen_time:.3f}s | Types: {{ {counts_str} }}")
                
                # Verify hierarchy usage:
                h_counts = Counter(m.get('hierarchy_idx', -1) for m in batch_metadata)
                h_str = ", ".join([f"H{k}:{v}" for k, v in h_counts.items()])
                if step % 50 == 0: # Less frequent log
                     tqdm.write(f"          Hierarchy usage: {{ {h_str} }}")
            
            queries, positives, coarse_positives = batch_data
            # if t_wait > 1.0: print(f"[WARN] Waited {t_wait:.4f}s", file=sys.stderr)

            if batch_data is None: continue

            
            # Move batches to GPU here!
            query_batch, pos_batch, coarse_pos_batch = batch_data
            query_batch = query_batch.to(device)
            pos_batch = pos_batch.to(device)
            coarse_pos_batch = coarse_pos_batch.to(device)
            
            optimizer.zero_grad()
            try:
                # Batches are now on GPU
                xq = augmentor(query_batch); xp = augmentor(pos_batch); xc = augmentor(coarse_pos_batch)
                
                zq = encoder(xq, query_batch.edge_index, query_batch.batch)
                z_pos = encoder(xp, pos_batch.edge_index, pos_batch.batch)
                z_coarse = encoder(xc, coarse_pos_batch.edge_index, coarse_pos_batch.batch)
                
                loss = hierarchical_info_nce_loss(zq, z_pos, z_coarse)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(augmentor.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item(); pbar.set_postfix({"loss": loss.item()})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    tqdm.write(f"\n[OOM] Step {step} | Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB | Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
                    if 'query_batch' in locals():
                        tqdm.write(f"[OOM] Batch Size (Nodes): Query={query_batch.num_nodes}, Pos={pos_batch.num_nodes if 'pos_batch' in locals() else '?'}")
                    print(f"WARNING: OOM at step {step}. Skipping batch.")
                    torch.cuda.empty_cache(); continue
                else: raise e
            # Explicit cleanup
            del query_batch, pos_batch, coarse_pos_batch, xq, xp, xc, zq, z_pos, z_coarse, loss
            del batch_data # Crucial: release reference to the tuple holding CPU tensors
            
            # Aggressive cleanup every few steps or on OOM danger
            if step % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        # iterator.stop() # No stop needed for standard dataloader
        avg_loss = total_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        scheduler.step(avg_loss)
        
        # End of epoch cleanup
        gc.collect()
        torch.cuda.empty_cache()
        current_lr = optimizer.param_groups[0]["lr"]
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_res = torch.cuda.memory_reserved() / 1e9
        print(f"Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}, LR = {current_lr:.1e}, GPU Mem: {mem_alloc:.2f}/{mem_res:.2f} GB")

        # --- CHECKPOINT SAVE LOGIC ---
        if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
            print(f"[CHECKPOINT] Saving checkpoint to {CHECKPOINT_PATH}...", flush=True)
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'augmentor_state_dict': augmentor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            try:
                # Force sync to Modal Volume
                volume = modal.Volume.from_name("jigsaw-cache-vol")
                volume.commit()
                print("[CHECKPOINT] Volume committed successfully.", flush=True)
            except Exception as e:
                print(f"[CHECKPOINT] Volume commit failed (non-fatal): {e}", flush=True)

    print("\n[REMOTE INFO] Training finished.")
    return {'encoder': encoder.cpu().state_dict(), 'augmentor': augmentor.cpu().state_dict()}

# --- THE LOCAL ENTRYPOINT ---
@app.local_entrypoint()
def main():
    import torch
    print("🚀 Starting Jigsaw GNN training on Modal for OGBN-MAG...")
    model_state_dicts = train.remote(
        epochs=100,
        steps_per_epoch=50,
        batch_size=32,
        num_hierarchies=3,
        fallback_mode=1 # 0=Wait, 1=CPU(MainThread), 2=GPU(MainThread)
    )
    file_path = "mag-6_layer-model-jigsaw.pth"
    torch.save(model_state_dicts, file_path)
    print(f"✅ Model saved to '{file_path}'")