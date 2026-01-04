# %%writefile generate_jigsaw_samples.py
import os
import json
import random
import itertools
from collections import Counter, defaultdict
from pathlib import Path
import pickle
import multiprocessing
import warnings
import sys
import copy

# --- SILENCE USERWARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module='outdated')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pymetis')

import numpy as np
import torch

# Let the main process use a few threads for faster single-process ops.
_MAIN_TORCH_THREADS = min(8, os.cpu_count() or 8)
os.environ.setdefault("OMP_NUM_THREADS", str(_MAIN_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_MAIN_TORCH_THREADS))
try:
    torch.set_num_threads(_MAIN_TORCH_THREADS)
    torch.set_num_interop_threads(min(2, _MAIN_TORCH_THREADS))
except Exception:
    pass

import networkx as nx
from tqdm import tqdm
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
import torch.serialization

# ============================================================================
# PyTorch 2.6+ Serialization compatibility (OGB + PyG pickles on Windows/Py3.13)
# ============================================================================
try:
    from torch_geometric.data import Data as _PyGData, Batch as _PyGBatch
    from torch_geometric.data.data import DataTensorAttr as _DataTensorAttr, DataEdgeAttr as _DataEdgeAttr
    from torch_geometric.data.storage import GlobalStorage as _GlobalStorage, NodeStorage as _NodeStorage, EdgeStorage as _EdgeStorage
except Exception:
    _PyGData = globals().get("Data", None)
    _PyGBatch = globals().get("Batch", None)
    _DataTensorAttr = globals().get("DataTensorAttr", None)
    _DataEdgeAttr = globals().get("DataEdgeAttr", None)
    _GlobalStorage = globals().get("GlobalStorage", None)
    _NodeStorage = globals().get("NodeStorage", None)
    _EdgeStorage = globals().get("EdgeStorage", None)

_safe = [cls for cls in [
    _PyGData, _PyGBatch, _DataTensorAttr, _DataEdgeAttr, _GlobalStorage, _NodeStorage, _EdgeStorage
] if cls is not None]

try:
    if _safe:
        torch.serialization.add_safe_globals(_safe)
except Exception:
    pass

_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    try:
        return _orig_torch_load(*args, **kwargs)
    except Exception:
        try:
            from contextlib import nullcontext
            ctx = torch.serialization.safe_globals(_safe) if hasattr(torch.serialization, "safe_globals") and _safe else nullcontext()
            with ctx:
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)
        except Exception as e:
            raise e
torch.load = _compat_torch_load
# ============================================================================

# --- CONFIGURATION ---
EPOCHS = 150
STEPS_PER_EPOCH = 50
BATCH_SIZE = 64
TARGET_TOTAL_SAMPLES = EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE
SAVE_BATCH_SIZE = 64
HIERARCHY_CONFIGS = [(1900, 5), (1950, 8), (2000, 10)]
MULTI_COARSE_CONFIGS = [
    (2, 2),(3, 2),(4, 2),(3, 3),(4, 3),(5, 2),(5, 3),(5, 4),(6, 3),(6, 4),(6, 5),
    (8, 4),(8, 5),(8, 6),(10, 5),(10, 6),(10, 7),(12, 6),(12, 7),(12, 8),(15, 7),
    (15, 8),(15, 9)
]
LOCAL_DATA_DIR = Path("./jigsaw_dataset")
JACCARD_SIMILARITY_THRESHOLD = 0.8

# --- worker-style globals (also used for single-process path) ---
worker_dataset = None
worker_membership = None
worker_data = None
worker_hierarchies = None
worker_global_node_to_fine_maps = None
worker_sample_types = None

# CSR globals (used by are_partitions_neighbors and fragment sampling)
worker_row_ptr = None
worker_col = None

def _init_threads_only(n: int):
    import os as _os
    import torch as _torch
    _os.environ["OMP_NUM_THREADS"] = str(n)
    _os.environ["MKL_NUM_THREADS"] = str(n)
    _os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    try:
        _torch.set_num_threads(n)
        _torch.set_num_interop_threads(min(2, n))
    except Exception:
        pass

def init_coarse_worker(dataset_to_share, membership_to_share):
    _init_threads_only(1)
    global worker_dataset, worker_membership
    worker_dataset = dataset_to_share
    worker_membership = membership_to_share

def init_sample_worker(
    data_to_share,
    hierarchies_lite_to_share,
    _unused,
    node_maps_tensor_list,
    sample_types_to_share,
    row_ptr_shared=None,
    col_shared=None,
):
    """Initializer for sample generation (also used in single-process path)."""
    _init_threads_only(1)
    global worker_data, worker_hierarchies
    global worker_global_node_to_fine_maps, worker_sample_types
    global worker_row_ptr, worker_col

    worker_data = data_to_share
    worker_hierarchies = hierarchies_lite_to_share
    worker_global_node_to_fine_maps = node_maps_tensor_list
    worker_sample_types = sample_types_to_share

    worker_row_ptr = row_ptr_shared
    worker_col = col_shared

# --- HELPERS & DATA PREP ---

def _create_subgraph_worker(part_id):
    node_indices = [i for i, p in enumerate(worker_membership) if p == part_id]
    if not node_indices:
        return None
    nodes_tensor = torch.tensor(node_indices, dtype=torch.long)
    subgraph = worker_dataset.subgraph(nodes_tensor)  # index-based, no big mask
    return part_id, subgraph, nodes_tensor

def make_partitions(dataset, num_parts):
    if dataset.num_nodes < num_parts:
        num_parts = dataset.num_nodes
    if num_parts <= 1:
        return [dataset], {0: torch.arange(dataset.num_nodes)}, {0: dataset}

    part_graphs_map = {}
    part_nodes_map = {}
    NPROC = max(1, min(os.cpu_count() or 1, 6))

    with tqdm(total=3, desc=f"  Coarse Partitioning ({num_parts} parts)", ncols=100) as pbar:
        pbar.set_postfix_str("Stage: Preparing graph for PyMetis")
        adj = SparseTensor.from_edge_index(dataset.edge_index, sparse_sizes=(dataset.num_nodes, dataset.num_nodes))
        xadj_t, adjncy_t, _ = adj.csr()
        xadj, adjncy = xadj_t.tolist(), adjncy_t.tolist()
        pbar.update(1)

        pbar.set_postfix_str("Stage: Running PyMetis partitioning...")
        import pymetis
        try:
            _, membership = pymetis.part_graph(num_parts, xadj=xadj, adjncy=adjncy)
        except Exception as e:
            raise RuntimeError(f"PyMetis failed during coarse partitioning: {e}")
        pbar.update(1)

        pbar.set_postfix_str("Stage: Creating subgraphs in parallel...")
        with multiprocessing.Pool(
            processes=NPROC,
            initializer=init_coarse_worker,
            initargs=(dataset, membership),
            maxtasksperchild=500
        ) as pool:
            tasks = range(num_parts)
            for result in tqdm(
                pool.imap_unordered(_create_subgraph_worker, tasks, chunksize=4),
                total=len(tasks), desc="    Creating subgraphs", ncols=100, leave=False
            ):
                if result:
                    part_id, subgraph, nodes_tensor = result
                    part_graphs_map[part_id] = subgraph
                    part_nodes_map[part_id] = nodes_tensor
        pbar.update(1)
        pbar.set_postfix_str("Stage: Complete!")

    part_graphs_list = [part_graphs_map[i] for i in sorted(part_graphs_map.keys())]
    return part_graphs_list, part_nodes_map, part_graphs_map

def _fine_partition_worker(args):
    """
    Build fine partitions inside a coarse partition.
    Returns a list of dicts with (coarse_idx, fine_part, global_indices).
    """
    coarse_idx, coarse_graph, coarse_part_nodes, num_fine = args
    membership = []
    finer_nodes_map_local = {}

    if coarse_graph.num_nodes < (num_fine * 2) or coarse_graph.num_edges == 0:
        finer_partitions_map = {0: coarse_graph}
        finer_nodes_map_local[0] = torch.arange(coarse_graph.num_nodes)
        membership = [0] * coarse_graph.num_nodes
    else:
        import pymetis
        adj = SparseTensor.from_edge_index(coarse_graph.edge_index, sparse_sizes=(coarse_graph.num_nodes, coarse_graph.num_nodes))
        xadj_t, adjncy_t, _ = adj.csr()
        xadj, adjncy = xadj_t.tolist(), adjncy_t.tolist()
        try:
            _, membership = pymetis.part_graph(num_fine, xadj=xadj, adjncy=adjncy)
        except Exception:
            membership = [0] * coarse_graph.num_nodes

        finer_partitions_map = {}
        for part_id in range(num_fine):
            node_indices_local = [i for i, p in enumerate(membership) if p == part_id]
            if node_indices_local:
                nodes_tensor_local = torch.tensor(node_indices_local, dtype=torch.long)
                finer_nodes_map_local[part_id] = nodes_tensor_local
                finer_partitions_map[part_id] = coarse_graph.subgraph(nodes_tensor_local)

    worker_results = []
    if not membership:
        return worker_results

    for part_id, fine_part in finer_partitions_map.items():
        if fine_part is None:
            continue
        if fine_part.num_nodes > 10 and fine_part.num_edges > 0:
            local_nodes = finer_nodes_map_local.get(part_id, None)
            if local_nodes is None or local_nodes.numel() == 0:
                continue
            global_indices_for_fine = coarse_part_nodes[local_nodes]
            worker_results.append({
                'coarse_idx': coarse_idx,
                'fine_part': fine_part,
                'global_indices': global_indices_for_fine
            })

    return worker_results

def build_single_hierarchy(data, num_coarse, num_fine):
    print(f"\n  • Building hierarchy with {num_coarse} coarse and {num_fine} fine partitions...")
    data = data.cpu()
    try:
        coarse_graphs_list, coarse_part_nodes_map, coarse_graphs_map = make_partitions(data, num_coarse)
    except Exception as e:
        print(f"    ERROR in coarse partitioning: {e}")
        raise

    node_to_coarse_map = {node_idx.item(): coarse_id for coarse_id, nodes in coarse_part_nodes_map.items() for node_idx in nodes}
    coarse_part_graph = nx.Graph()
    for u, v in data.edge_index.t().tolist():
        c_u, c_v = node_to_coarse_map.get(u), node_to_coarse_map.get(v)
        if c_u is not None and c_v is not None and c_u != c_v:
            coarse_part_graph.add_edge(c_u, c_v)

    tasks = []
    for coarse_id, coarse_graph in coarse_graphs_map.items():
        if coarse_id in coarse_part_nodes_map:
            tasks.append((coarse_id, coarse_graph, coarse_part_nodes_map[coarse_id], num_fine))

    fine_graphs, fine_part_nodes_map, fine_to_coarse_map = [], {}, {}
    fine_global_idx = 0
    NPROC = max(1, min(os.cpu_count() or 1, 6))

    with multiprocessing.Pool(processes=NPROC, initializer=_init_threads_only, maxtasksperchild=500) as pool:
        for worker_results in tqdm(pool.imap_unordered(_fine_partition_worker, tasks, chunksize=4), total=len(tasks), desc="    Fine partitioning", ncols=100):
            for result in worker_results:
                fine_graphs.append(result['fine_part'])
                fine_part_nodes_map[fine_global_idx] = result['global_indices']
                fine_to_coarse_map[fine_global_idx] = result['coarse_idx']
                fine_global_idx += 1

    print(f"    ✓ Created {len(fine_graphs)} fine partitions from {len(coarse_graphs_list)} coarse partitions")
    return (coarse_graphs_list, fine_graphs, node_to_coarse_map, fine_to_coarse_map, fine_part_nodes_map, coarse_part_graph, coarse_part_nodes_map)

def convert_hetero_to_homo(hetero_data: "HeteroData"):
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
        gei = ei.clone()
        gei[0] += node_offset[src_t]
        gei[1] += node_offset[dst_t]
        all_ei.append(gei)
    edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.empty((2, 0), dtype=torch.long)
    homo = Data(x=x, edge_index=edge_index, num_nodes=total_nodes)
    homo.node_type = node_type_ids
    homo.node_types = node_types
    homo.node_offset = node_offset
    homo.global_id = torch.arange(total_nodes, dtype=torch.long)
    print(f"    - Converted to homogeneous: {homo.num_nodes} nodes, {homo.edge_index.size(1)} edges")
    return homo

def make_undirected_fast(edge_index, num_nodes):
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
    row, col, _ = adj.coo()
    return torch.stack([row, col], dim=0)

# --- CSR-bounded fragment sampler over a *node set* (global ids) ---
def _extract_fragment_from_node_set(node_indices_tensor: torch.Tensor, target_size: int):
    global worker_row_ptr, worker_col
    if worker_row_ptr is None or worker_col is None:
        return None
    if node_indices_tensor is None or node_indices_tensor.numel() == 0:
        return None

    allowed_list = node_indices_tensor.tolist()
    allowed_set = set(allowed_list)
    start = random.choice(allowed_list)
    visited = {start}
    queue = [start]

    target = min(target_size, len(allowed_set))
    budget = target_size * 8  # soft cap

    while queue and len(visited) < target and budget > 0:
        cur = queue.pop(0)
        s = int(worker_row_ptr[cur].item())
        e = int(worker_row_ptr[cur + 1].item())
        if e <= s:
            budget -= 1
            continue
        nbrs = worker_col[s:e].tolist()
        random.shuffle(nbrs)
        for nb in nbrs:
            if nb in allowed_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target:
                    break
        budget -= 1

    return torch.tensor(list(visited), dtype=torch.long)

def _finalize_query_from_nodes(original_data, global_node_indices, min_nodes):
    if not global_node_indices:
        return None, None
    q_global_nodes = torch.tensor(list(set(global_node_indices)), dtype=torch.long)
    if len(q_global_nodes) < min_nodes:
        return None, None
    return original_data.subgraph(q_global_nodes), q_global_nodes

# --- CSR-based neighbor check ---
def are_partitions_neighbors(_ignored, nodes1, nodes2):
    global worker_row_ptr, worker_col
    if worker_row_ptr is None or worker_col is None:
        return False
    if nodes1 is None or nodes2 is None or len(nodes1) == 0 or len(nodes2) == 0:
        return False
    if len(nodes1) < len(nodes2):
        small, large = nodes1, nodes2
    else:
        small, large = nodes2, nodes1
    small_set = set(int(x) for x in small.tolist())
    for node in large.tolist():
        start = int(worker_row_ptr[node].item())
        end = int(worker_row_ptr[node + 1].item())
        for nbr in worker_col[start:end].tolist():
            if nbr in small_set:
                return True
    return False

def generate_multi_coarse_partition_query(original_data, _adj_ignored, hierarchy_data, config, min_nodes=80, max_nodes=100):
    (_, _, _, fine_to_coarse_map, fine_part_nodes_map, coarse_part_graph, coarse_part_nodes_map) = hierarchy_data
    num_frags, min_coarse_parts = config
    if coarse_part_graph.number_of_edges() == 0:
        return None

    coarse_to_fine_map = defaultdict(list)
    for f_idx, c_idx in fine_to_coarse_map.items():
        coarse_to_fine_map[c_idx].append(f_idx)

    possible_start_edges = list(coarse_part_graph.edges())
    random.shuffle(possible_start_edges)

    for c_idx1, c_idx2 in possible_start_edges:
        fine_parts_in_c1 = coarse_to_fine_map.get(c_idx1, [])
        fine_parts_in_c2 = coarse_to_fine_map.get(c_idx2, [])
        all_fine_pairs = list(itertools.product(fine_parts_in_c1, fine_parts_in_c2))
        random.shuffle(all_fine_pairs)

        for f1, f2 in all_fine_pairs:
            if f1 not in fine_part_nodes_map or f2 not in fine_part_nodes_map:
                continue
            if not are_partitions_neighbors(None, fine_part_nodes_map[f1], fine_part_nodes_map[f2]):
                continue

            q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
            while queue and len(q_fine_indices) < num_frags:
                current_fine_idx = queue.pop(0)
                current_c_idx = fine_to_coarse_map[current_fine_idx]
                coarse_neighbors_and_self = list(coarse_part_graph.neighbors(current_c_idx)) + [current_c_idx]
                potential_fine_neighbors = [fn for cidx in coarse_neighbors_and_self for fn in coarse_to_fine_map.get(cidx, [])]
                random.shuffle(potential_fine_neighbors)
                for neighbor_idx in potential_fine_neighbors:
                    if neighbor_idx not in visited and neighbor_idx in fine_part_nodes_map and current_fine_idx in fine_part_nodes_map:
                        if are_partitions_neighbors(None, fine_part_nodes_map[current_fine_idx], fine_part_nodes_map[neighbor_idx]):
                            visited.add(neighbor_idx); queue.append(neighbor_idx); q_fine_indices.append(neighbor_idx)
                            if len(q_fine_indices) >= num_frags:
                                break

            if len(q_fine_indices) < num_frags:
                continue

            true_coarse_indices = {fine_to_coarse_map[f_idx] for f_idx in q_fine_indices}
            if len(true_coarse_indices) < min_coarse_parts:
                continue

            nodes_per_frag = max(5, max_nodes // num_frags)
            all_query_nodes = []
            for fine_idx in q_fine_indices:
                if fine_idx not in fine_part_nodes_map:
                    continue
                local_nodes_global = _extract_fragment_from_node_set(fine_part_nodes_map[fine_idx], nodes_per_frag)
                if local_nodes_global is not None and local_nodes_global.numel() > 0:
                    all_query_nodes.extend(local_nodes_global.tolist())

            if len(all_query_nodes) < min_nodes:
                continue

            Gq, _ = _finalize_query_from_nodes(original_data, all_query_nodes, min_nodes)
            if Gq is None:
                continue

            stitched_nodes = torch.cat([fine_part_nodes_map[idx] for idx in q_fine_indices if idx in fine_part_nodes_map])
            if stitched_nodes.numel() == 0:
                continue
            G_stitched = original_data.subgraph(stitched_nodes)

            all_coarse_pos_nodes = torch.cat([coarse_part_nodes_map[cidx] for cidx in true_coarse_indices if cidx in coarse_part_nodes_map])
            if all_coarse_pos_nodes.numel() == 0:
                continue
            G_coarse_pos = original_data.subgraph(all_coarse_pos_nodes)

            return Gq, G_stitched, G_coarse_pos, q_fine_indices

    return None

def _generate_one_stratified_sample(h_idx, c_idx, s_type_idx, return_attempts: bool = False):
    max_attempts = 30  # keep retries as requested
    sample_type = worker_sample_types[s_type_idx]

    for attempt in range(1, max_attempts + 1):
        hierarchy_data = worker_hierarchies[h_idx]
        (_, _, _, fine_to_coarse_map, fine_part_nodes_map, _coarse_part_graph, coarse_part_nodes_map) = hierarchy_data

        Gq, Gpos, G_coarse_pos, source_fine_indices = None, None, None, None
        q_size_min, q_size_max, max_gpos_nodes = 20, 120, 4000

        if sample_type == "k_hop":
            if c_idx not in coarse_part_nodes_map or len(coarse_part_nodes_map[c_idx]) == 0:
                continue

            hop_candidates = [2, 3, 4]
            random.shuffle(hop_candidates)
            subset_pos = None

            for h in hop_candidates:
                anchor_global = random.choice(coarse_part_nodes_map[c_idx]).item()
                subset_pos, _, _, _ = k_hop_subgraph(
                    anchor_global,
                    num_hops=h,
                    edge_index=worker_data.edge_index,
                    relabel_nodes=False
                )
                if len(subset_pos) < q_size_min:
                    subset_pos = None
                    continue
                if len(subset_pos) > max_gpos_nodes:
                    idx = torch.randperm(len(subset_pos))[:max_gpos_nodes]
                    subset_pos = subset_pos[idx]
                break

            if subset_pos is None or len(subset_pos) < q_size_min:
                continue

            Gpos = worker_data.subgraph(subset_pos)
            if Gpos.num_nodes < q_size_min or Gpos.num_edges == 0:
                continue

            q_upper = min(q_size_max, Gpos.num_nodes)
            q_lower = min(q_size_min, q_upper)
            if q_lower < 2:
                continue
            q_take = random.randint(q_lower, q_upper)
            q_nodes_local_indices = torch.randperm(Gpos.num_nodes)[:q_take]
            Gq = Gpos.subgraph(q_nodes_local_indices)
            if Gq.num_edges == 0:
                continue

            G_coarse_pos = worker_data.subgraph(coarse_part_nodes_map[c_idx])

            node_to_fine_tensor = worker_global_node_to_fine_maps[h_idx]
            fine_ids = node_to_fine_tensor[subset_pos].tolist()
            source_fine_indices = list({fid for fid in fine_ids if fid >= 0})

        elif sample_type == "single_fine_part":
            coarse_to_fine_map_local = defaultdict(list)
            for f_idx, coarse_id in fine_to_coarse_map.items():
                coarse_to_fine_map_local[coarse_id].append(f_idx)
            possible_fine_parts = coarse_to_fine_map_local.get(c_idx, [])
            if not possible_fine_parts:
                continue
            fine_idx = random.choice(possible_fine_parts)
            if fine_idx not in fine_part_nodes_map:
                continue

            pos_nodes = fine_part_nodes_map[fine_idx]
            if pos_nodes.numel() == 0:
                continue
            if pos_nodes.numel() > max_gpos_nodes:
                pos_nodes = pos_nodes[torch.randperm(pos_nodes.numel())[:max_gpos_nodes]]

            Gpos = worker_data.subgraph(pos_nodes)

            q_nodes_global = _extract_fragment_from_node_set(pos_nodes, random.randint(q_size_min, q_size_max))
            if q_nodes_global is None or q_nodes_global.numel() < q_size_min:
                continue
            Gq = worker_data.subgraph(q_nodes_global)

            G_coarse_pos = worker_data.subgraph(coarse_part_nodes_map[c_idx])
            source_fine_indices = [fine_idx]

        elif sample_type.startswith("multi_coarse"):
            parts = sample_type.split("_")
            num_frags = int(parts[-2])
            min_coarse = int(parts[-1])
            config = (num_frags, min_coarse)
            res = generate_multi_coarse_partition_query(worker_data, None, hierarchy_data, config)
            if res:
                Gq, Gpos, G_coarse_pos, source_fine_indices = res

        elif sample_type == "multi_fine_sibling":
            coarse_to_fine_map_local = defaultdict(list)
            for f_idx, coarse_id in fine_to_coarse_map.items():
                coarse_to_fine_map_local[coarse_id].append(f_idx)
            siblings = coarse_to_fine_map_local.get(c_idx, [])
            if len(siblings) < 2:
                continue

            num_frags = random.randint(2, min(4, len(siblings)))
            start_fine_idx = random.choice(siblings)
            source_part_indices = {start_fine_idx}
            q = [start_fine_idx]
            visited = {start_fine_idx}

            while q and len(source_part_indices) < num_frags:
                curr = q.pop(0)
                valid_siblings = [s for s in siblings if s in fine_part_nodes_map and curr in fine_part_nodes_map]
                random.shuffle(valid_siblings)
                for neighbor in valid_siblings:
                    if (neighbor not in visited and
                        are_partitions_neighbors(None, fine_part_nodes_map[curr], fine_part_nodes_map[neighbor])):
                        visited.add(neighbor)
                        source_part_indices.add(neighbor)
                        q.append(neighbor)
                        if len(source_part_indices) >= num_frags:
                            break

            if len(source_part_indices) < num_frags:
                continue

            valid_indices = [i for i in source_part_indices if i in fine_part_nodes_map]
            if len(valid_indices) < num_frags:
                continue

            pos_nodes = torch.cat([fine_part_nodes_map[i] for i in valid_indices])
            if pos_nodes.numel() > max_gpos_nodes:
                pos_nodes = pos_nodes[torch.randperm(pos_nodes.numel())[:max_gpos_nodes]]

            Gpos = worker_data.subgraph(pos_nodes)

            nodes_per_frag = max(5, (q_size_min + q_size_max) // (2 * num_frags))
            all_query_global_nodes = []
            for fine_idx in valid_indices:
                local_nodes_global = _extract_fragment_from_node_set(fine_part_nodes_map[fine_idx], nodes_per_frag)
                if local_nodes_global is not None and local_nodes_global.numel() > 0:
                    all_query_global_nodes.extend(local_nodes_global.tolist())

            if not all_query_global_nodes:
                continue

            Gq, _ = _finalize_query_from_nodes(worker_data, all_query_global_nodes, q_size_min)
            if Gq is None:
                continue

            G_coarse_pos = worker_data.subgraph(coarse_part_nodes_map[c_idx])
            source_fine_indices = list(valid_indices)

        # Success gate
        if (Gq and Gpos and G_coarse_pos and source_fine_indices and
            Gq.num_nodes > 1 and Gpos.num_nodes > 1 and G_coarse_pos.num_nodes > 1 and
            Gq.num_edges > 0 and Gpos.num_edges > 0):
            if return_attempts:
                return (Gq, Gpos, G_coarse_pos, source_fine_indices), attempt
            return (Gq, Gpos, G_coarse_pos, source_fine_indices)

    if return_attempts:
        return None, max_attempts
    return None

# (kept for completeness; not used in single-process loop)
def _sample_generation_worker(tasks):
    results = []
    for task in tasks:
        sample_data, attempts_used = _generate_one_stratified_sample(
            task['h_idx'], task['c_idx'], task['s_type_idx'], return_attempts=True
        )
        results.append({'task': task, 'result': sample_data, 'attempts': attempts_used})
    return results

class JigsawSampleGenerator:
    # Only warn if these *semantic* knobs change between runs
    RELEVANT_CFG_KEYS = ("HIERARCHY_CONFIGS", "MULTI_COARSE_CONFIGS", "JACCARD_SIMILARITY_THRESHOLD")

    def __init__(self, data, hierarchies, adj_resource, global_node_to_fine_maps, output_dir, state_file):
        self.data = data
        self.hierarchies = hierarchies  # LITE tuples
        self.adj_resource = adj_resource  # (row_ptr, col)
        if isinstance(adj_resource, tuple) and len(adj_resource) == 2:
            self.row_ptr, self.col = adj_resource
        else:
            self.row_ptr, self.col = None, None
        self.global_node_to_fine_maps = global_node_to_fine_maps  # list of TENSORS
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.state_file = Path(state_file)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.sample_types = ["k_hop", "single_fine_part", "multi_fine_sibling"] + [
            f"multi_coarse_{c[0]}_{c[1]}" for c in MULTI_COARSE_CONFIGS
        ]

        existing = sorted(self.samples_dir.glob("batch_*.pt"))
        if existing:
            last = existing[-1].stem
            try:
                last_idx = int(last.split("_")[-1])
                _next_idx_on_disk = last_idx + 1
            except Exception:
                _next_idx_on_disk = 0
        else:
            _next_idx_on_disk = 0
        self._next_idx_on_disk = _next_idx_on_disk

        self.metrics = {
            "duplicates_dropped": 0,
            "success_attempts_histogram": {i: 0 for i in range(1, 11)},
            "success_attempts_by_type_sum": defaultdict(int),
            "success_attempts_by_type_cnt": defaultdict(int),
            "success_by_hierarchy": defaultdict(int),
        }

        self.config_fingerprint = {
            # keep full fingerprint; comparison uses RELEVANT_CFG_KEYS only
            "HIERARCHY_CONFIGS": HIERARCHY_CONFIGS,
            "MULTI_COARSE_CONFIGS": MULTI_COARSE_CONFIGS,
            "JACCARD_SIMILARITY_THRESHOLD": JACCARD_SIMILARITY_THRESHOLD,
            "SAVE_BATCH_SIZE": SAVE_BATCH_SIZE,
        }

        self.state = self._load_state()

    # --------- helpers ----------
    @staticmethod
    def _atomic_write(path: Path, data_bytes: bytes):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data_bytes); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _json_compat(obj):
        """Recursively convert to JSON-safe types (fixes int64 errors)."""
        try:
            import numpy as _np
        except Exception:
            _np = None

        if obj is None or isinstance(obj, (str, bool, float, int)):
            return obj
        if _np is not None and isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (torch.Tensor,)):
            # Only expected for small index containers; convert to (python) ints
            if obj.numel() == 1:
                return int(obj.item())
            return [int(x) for x in obj.view(-1).tolist()]
        if isinstance(obj, (set, tuple, list)):
            return [JigsawSampleGenerator._json_compat(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): JigsawSampleGenerator._json_compat(v) for k, v in obj.items()}
        # Fallback to string for unexpected types
        return str(obj)

    def _configs_equivalent(self, old: dict, new: dict) -> bool:
        if not isinstance(old, dict) or not isinstance(new, dict):
            return False
        return all(old.get(k) == new.get(k) for k in self.RELEVANT_CFG_KEYS)

    # --------- state load / save ----------
    def _load_state(self):
        if self.state_file.exists():
            print(f"✅ Resuming from state file: {self.state_file}")
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            state.setdefault("scheduled_cycles", 0)

            # Warn only if the *relevant* knobs changed
            if not self._configs_equivalent(state.get("config_fingerprint", {}), self.config_fingerprint):
                print("⚠️ Config changed since last run. Continuing, but behavior may differ.")
            # Always update stored fingerprint to current to avoid repeated warnings
            state["config_fingerprint"] = self.config_fingerprint

            # Resume batch index safely
            state['current_batch_idx'] = int(max(state.get('current_batch_idx', 0), getattr(self, "_next_idx_on_disk", 0)))

            # Rebuild dedup sets
            self.source_to_nodes_sets = defaultdict(list)
            for key, list_of_node_lists in state.get('source_to_nodes', {}).items():
                self.source_to_nodes_sets[key] = [set(int(x) for x in nodes) for nodes in list_of_node_lists]

            # Coerce task fields to builtins (defensive)
            tasks = state.get('tasks', [])
            for t in tasks:
                if 'h_idx' in t: t['h_idx'] = int(t['h_idx'])
                if 'c_idx' in t: t['c_idx'] = int(t['c_idx'])
                if 's_type_idx' in t: t['s_type_idx'] = int(t['s_type_idx'])
                if 'cycle' in t: t['cycle'] = int(t['cycle'])
                if 'status' in t: t['status'] = str(t['status'])

            return state

        # Fresh state
        print("✨ Starting new generation: creating task list.")
        tasks = []
        total_possible_tasks_per_cycle = sum(len(h_data[6]) for h_data in self.hierarchies) * len(self.sample_types)
        num_cycles_needed = (TARGET_TOTAL_SAMPLES // total_possible_tasks_per_cycle) + 2

        for cycle in range(num_cycles_needed):
            for h_idx, h_data in enumerate(self.hierarchies):
                for c_idx in h_data[6].keys():  # coarse partitions
                    for s_type_idx in range(len(self.sample_types)):
                        task_id = f"cyc{cycle}-h{h_idx}-c{int(c_idx)}-s{s_type_idx}"
                        tasks.append({
                            "task_id": task_id,
                            "h_idx": int(h_idx),
                            "c_idx": int(c_idx),
                            "s_type_idx": int(s_type_idx),
                            "cycle": int(cycle),
                            "status": "pending"
                        })

        self.source_to_nodes_sets = defaultdict(list)
        return {
            "num_samples_generated": 0,
            "current_batch_idx": max(0, getattr(self, "_next_idx_on_disk", 0)),
            "tasks": tasks,
            "source_to_nodes": {},
            "scheduled_cycles": num_cycles_needed,
            "config_fingerprint": self.config_fingerprint,
        }

    def _save_state(self):
        # Convert the dedup structure back to lists
        state_copy = dict(self.state)
        state_copy['source_to_nodes'] = {k: [list(s) for s in self.source_to_nodes_sets.get(k, [])]
                                         for k in self.source_to_nodes_sets.keys()}

        # Ensure JSON-safe types across the whole state
        safe_state = self._json_compat(state_copy)
        payload = json.dumps(safe_state, indent=2).encode("utf-8")
        self._atomic_write(self.state_file, payload)

    # --------- dedup ----------
    def _is_near_duplicate(self, source_fingerprint, query_nodes_set):
        source_key = str(source_fingerprint)
        for existing_set in self.source_to_nodes_sets[source_key]:
            intersection = len(query_nodes_set.intersection(existing_set))
            union = len(query_nodes_set.union(existing_set))
            if union == 0:
                continue
            if (intersection / union) > JACCARD_SIMILARITY_THRESHOLD:
                return True
        self.source_to_nodes_sets[source_key].append(query_nodes_set)
        return False

    # --------- dynamic cycles ----------
    def _append_more_cycles(self, extra_cycles=2):
        current_max_cycle = int(self.state.get("scheduled_cycles", 0))
        start_cycle = current_max_cycle
        end_cycle = current_max_cycle + int(extra_cycles)
        print(f"➕ Adding {extra_cycles} extra cycle(s): {start_cycle} → {end_cycle-1}")

        for cycle in range(start_cycle, end_cycle):
            for h_idx, h_data in enumerate(self.hierarchies):
                for c_idx in h_data[6].keys():
                    for s_type_idx in range(len(self.sample_types)):
                        task_id = f"cyc{cycle}-h{h_idx}-c{int(c_idx)}-s{s_type_idx}"
                        self.state['tasks'].append({
                            "task_id": task_id,
                            "h_idx": int(h_idx),
                            "c_idx": int(c_idx),
                            "s_type_idx": int(s_type_idx),
                            "cycle": int(cycle),
                            "status": "pending"
                        })
        self.state["scheduled_cycles"] = end_cycle

    # --------- batch save ----------
    def _save_batch(self, batch_data):
        batch_idx = int(self.state['current_batch_idx'])
        filepath = self.samples_dir / f"batch_{batch_idx:05d}.pt"
        import io
        buf = io.BytesIO()
        torch.save(batch_data, buf, pickle_protocol=4)
        self._atomic_write(filepath, buf.getvalue())
        self.state['current_batch_idx'] = batch_idx + 1

    # --------- core generation ----------
    def generate(self):
        # single-process generation – no multiprocessing pools
        self._lock_file = self.output_dir / ".gen_lock"
        try:
            fd = os.open(self._lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
        except FileExistsError:
            print(f"🛑 Another generation appears to be running ({self._lock_file} exists).")
            return

        try:
            # Initialize "worker" globals in this process
            init_sample_worker(
                self.data,
                self.hierarchies,
                None,
                self.global_node_to_fine_maps,
                self.sample_types,
                row_ptr_shared=self.row_ptr,
                col_shared=self.col,
            )

            pbar = tqdm(total=TARGET_TOTAL_SAMPLES, initial=int(self.state['num_samples_generated']),
                        desc="Generating Samples (single process)", ncols=100)
            current_batch = []
            tasks = self.state['tasks']
            idx = 0

            while int(self.state['num_samples_generated']) < TARGET_TOTAL_SAMPLES:
                progressed = False

                # Single pass over tasks list
                for i in range(idx, len(tasks)):
                    t = tasks[i]
                    if int(self.state['num_samples_generated']) >= TARGET_TOTAL_SAMPLES:
                        break
                    if t['status'] not in ('pending', 'failed'):
                        continue

                    sample_data, attempts_used = _generate_one_stratified_sample(
                        t['h_idx'], t['c_idx'], t['s_type_idx'], return_attempts=True
                    )
                    if sample_data:
                        gq, gpos, g_coarse = sample_data[0], sample_data[1], sample_data[2]
                        source_fine_indices = sample_data[3]

                        query_nodes_set = set(int(x) for x in gq.global_id.numpy().tolist())
                        source_fingerprint = tuple(sorted(int(x) for x in source_fine_indices))

                        if not self._is_near_duplicate(source_fingerprint, query_nodes_set):
                            current_batch.append((gq, gpos, g_coarse))
                            self.state['num_samples_generated'] = int(self.state['num_samples_generated']) + 1
                            pbar.update(1)
                            t['status'] = 'success'
                            progressed = True

                            if attempts_used is not None:
                                attempts_used = int(max(1, min(10, int(attempts_used))))
                                self.metrics["success_attempts_histogram"][attempts_used] += 1
                                sname = self.sample_types[t['s_type_idx']]
                                self.metrics["success_attempts_by_type_sum"][sname] += attempts_used
                                self.metrics["success_attempts_by_type_cnt"][sname] += 1
                            self.metrics["success_by_hierarchy"][t['h_idx']] += 1

                            if len(current_batch) >= SAVE_BATCH_SIZE:
                                self._save_batch(current_batch)
                                current_batch = []
                        else:
                            t['status'] = 'failed'
                            self.metrics["duplicates_dropped"] += 1
                    else:
                        t['status'] = 'failed'

                self._save_state()
                idx = 0  # after each sweep, start from beginning to pick up newly appended tasks

                if not progressed:
                    # If nothing succeeded in a whole sweep, schedule more work
                    self._append_more_cycles(extra_cycles=2)

            if current_batch:
                self._save_batch(current_batch)

            self._save_state()
            pbar.close()
            print("\n🎉 Generation complete.")
            self.create_summary_report()

        except KeyboardInterrupt:
            print("\n🛑 Generation interrupted. Saving state and creating summary...")
            try:
                self._save_state()
            except Exception as e:
                print(f"⚠️ Failed to save state safely: {e}")
            try:
                self.create_summary_report()
            except Exception as e:
                print(f"⚠️ Failed to create summary: {e}")
        finally:
            try:
                if hasattr(self, "_lock_file") and self._lock_file.exists():
                    self._lock_file.unlink()
            except Exception:
                pass

    # --------- summary ----------
    def create_summary_report(self):
        print("\n📊 Creating generation summary report...")
        tasks = self.state['tasks']
        success_count = sum(1 for t in tasks if t['status'] == 'success')
        failed_count = sum(1 for t in tasks if t['status'] == 'failed')
        pending_count = sum(1 for t in tasks if t['status'] == 'pending')

        type_success = Counter()
        type_failure = Counter()
        partition_failure = Counter()

        for t in tasks:
            sample_type_name = self.sample_types[t['s_type_idx']]
            if t['status'] == 'success':
                type_success[sample_type_name] += 1
            elif t['status'] == 'failed':
                type_failure[sample_type_name] += 1
                partition_key = f"H{t['h_idx']}-C{t['c_idx']}"
                partition_failure[partition_key] += 1

        avg_attempts_by_type = {}
        for sname in self.sample_types:
            cnt = self.metrics["success_attempts_by_type_cnt"].get(sname, 0)
            sm = self.metrics["success_attempts_by_type_sum"].get(sname, 0)
            avg_attempts_by_type[sname] = (sm / cnt) if cnt > 0 else None

        success_by_hierarchy = {int(k): int(v) for k, v in self.metrics["success_by_hierarchy"].items()}

        summary = {
            "total_samples_target": int(TARGET_TOTAL_SAMPLES),
            "total_samples_generated": int(self.state['num_samples_generated']),
            "task_summary": {
                "succeeded": int(success_count),
                "failed": int(failed_count),
                "pending": int(pending_count),
                "total": int(len(tasks))
            },
            "success_by_type": {k: int(v) for k, v in dict(type_success).items()},
            "failures_by_type": {k: int(v) for k, v in dict(type_failure).items()},
            "top_20_failing_partitions": {k: int(v) for k, v in dict(partition_failure.most_common(20)).items()},
            "duplicates_dropped": int(self.metrics["duplicates_dropped"]),
            "success_attempts_histogram": {int(k): int(v) for k, v in self.metrics["success_attempts_histogram"].items()},
            "avg_success_attempts_by_type": {k: (float(v) if v is not None else None) for k, v in avg_attempts_by_type.items()},
            "success_by_hierarchy": success_by_hierarchy,
            "scheduled_cycles_total": int(self.state.get("scheduled_cycles", 0)),
            "config_fingerprint": self.config_fingerprint,
        }

        summary_file = self.output_dir / "generation_summary.json"
        safe_summary = self._json_compat(summary)
        with open(summary_file, 'w') as f:
            json.dump(safe_summary, f, indent=2)
        print(f"✅ Summary saved to {summary_file}")

def setup_data_and_helpers():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hierarchies_file = LOCAL_DATA_DIR / "hierarchies.pkl"

    if hierarchies_file.exists():
        print("💾 Loading pre-built hierarchies...")
        with open(hierarchies_file, 'rb') as f:
            hierarchies = pickle.load(f)
    else:
        hierarchies = None

    print("💾 Loading OGBN-MAG dataset...")
    dataset = PygNodePropPredDataset(name="ogbn-mag", root="/tmp/ogbn_mag_data")
    data = convert_hetero_to_homo(dataset[0])
    data.edge_index = make_undirected_fast(data.edge_index, data.num_nodes)

    if hierarchies is None:
        print("🏗️ Building hierarchies from scratch...")
        hierarchies = []
        for i, (num_coarse, num_fine) in enumerate(HIERARCHY_CONFIGS):
            print(f"\n--- Building Hierarchy {i+1}/{len(HIERARCHY_CONFIGS)} ---")
            hierarchies.append(build_single_hierarchy(data, num_coarse, num_fine))
        print(f"\n💾 Saving hierarchies to {hierarchies_file}...")
        with open(hierarchies_file, 'wb') as f:
            pickle.dump(hierarchies, f)

    print("\nBUILDING SHARED CSR ADJ...")
    adj = SparseTensor.from_edge_index(
        data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    row_ptr, col, _ = adj.csr()
    row_ptr = row_ptr.to(torch.int64).cpu().share_memory_()
    col = col.to(torch.int64).cpu().share_memory_()
    adj_csr = (row_ptr, col)

    # Share core tensors (harmless in single process; keeps options open)
    try:
        data.x.share_memory_()
        data.edge_index.share_memory_()
        if hasattr(data, "node_type"): data.node_type.share_memory_()
        if hasattr(data, "global_id"): data.global_id.share_memory_()
    except Exception:
        pass

    # Build LITE hierarchies + tensor node->fine maps
    hierarchies_lite = []
    node_to_fine_tensors = []

    for h in hierarchies:
        (_coarse_graphs_list, _fine_graphs, _node_to_coarse_map,
         fine_to_coarse_map, fine_part_nodes_map, coarse_part_graph, coarse_part_nodes_map) = h

        fine_part_nodes_map_shm = {int(fid): nodes.to(torch.int64).cpu().share_memory_()
                                   for fid, nodes in fine_part_nodes_map.items()}
        coarse_part_nodes_map_shm = {int(cid): nodes.to(torch.int64).cpu().share_memory_()
                                     for cid, nodes in coarse_part_nodes_map.items()}

        node_to_fine = torch.full((data.num_nodes,), -1, dtype=torch.int32)
        for fid, nodes in fine_part_nodes_map_shm.items():
            node_to_fine[nodes] = int(fid)
        node_to_fine_tensors.append(node_to_fine.share_memory_())

        hierarchies_lite.append((
            None,                        # 0: coarse_graphs_list (unused)
            None,                        # 1: fine_graphs (unused)
            None,                        # 2: node_to_coarse_map (unused)
            fine_to_coarse_map,          # 3
            fine_part_nodes_map_shm,     # 4
            coarse_part_graph,           # 5
            coarse_part_nodes_map_shm,   # 6
        ))

    print("  - Shared CSR and helper maps ready.")
    return data.cpu(), hierarchies_lite, adj_csr, node_to_fine_tensors

def main():
    data, hierarchies_lite, adj_csr, node_to_fine_tensors = setup_data_and_helpers()
    state_file = LOCAL_DATA_DIR / "generation_state.json"
    print("\n--- STARTING FULL SAMPLE GENERATION (single process) ---")
    generator = JigsawSampleGenerator(data, hierarchies_lite, adj_csr, node_to_fine_tensors, LOCAL_DATA_DIR, state_file)
    try:
        generator.generate()
    except KeyboardInterrupt:
        # This is a second layer; generate() already flushes & saves on Ctrl+C.
        print("\n🛑 Generation interrupted at top-level. Exiting gracefully.")

def run_test_validation(test_output_dir):
    print("\n--- RUNNING TEST DATA VALIDATION ---")
    from torch_geometric.nn import GINConv
    import torch.nn as nn
    class MinimalAugmentor(nn.Module):
        def __init__(self):
            super().__init__()
            self.type_emb = nn.Embedding(4, 16)
        def forward(self, data):
            return torch.cat([data.x, self.type_emb(data.node_type)], dim=1)
    class MinimalGNN(nn.Module):
        def __init__(self):
            super().__init__()
            mlp = nn.Sequential(nn.Linear(128 + 16, 32), nn.ReLU(), nn.Linear(32, 32))
            self.conv = GINConv(mlp)
        def forward(self, x, edge_index):
            return self.conv(x, edge_index)
    augmentor = MinimalAugmentor()
    model = MinimalGNN()
    print("✅ Minimal validation model created.")
    sample_files = list((test_output_dir / "samples").glob("*.pt"))
    if not sample_files:
        print("⚠️ No sample files found in test directory to validate.")
        return
    print(f"Found {len(sample_files)} sample file(s) to validate...")
    try:
        batch_data = torch.load(sample_files[0], weights_only=False)
        q_graphs, _, _ = zip(*batch_data)
        batch = Batch.from_data_list(list(q_graphs))
        print(f"Testing forward pass on a batch of {batch.num_graphs} query graphs...")
        augmented_x = augmentor(batch)
        output = model(augmented_x, batch.edge_index)
        assert output.shape[0] == batch.num_nodes
        assert output.shape[1] == 32
        print("✅ SUCCESS: Forward pass completed. Data format is valid for training.")
    except Exception as e:
        print(f"\n❌ FAILED: Validation forward pass error. Details: {e}")

def test_generate_one_partition(h_idx=0, c_idx=100):
    print(f"--- RUNNING TEST FOR HIERARCHY {h_idx}, COARSE PARTITION {c_idx} ---")
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 8))
    except Exception:
        pass
    data, hierarchies_lite, adj_csr, node_to_fine_tensors = setup_data_and_helpers()
    test_output_dir = LOCAL_DATA_DIR / "test_output"
    if test_output_dir.exists():
        import shutil
        print(f"Cleaning up old test directory: {test_output_dir}")
        shutil.rmtree(test_output_dir)
    (test_output_dir / "samples").mkdir(parents=True, exist_ok=True)

    sample_types = ["k_hop", "single_fine_part", "multi_fine_sibling"] + [f"multi_coarse_{c[0]}_{c[1]}" for c in MULTI_COARSE_CONFIGS]
    init_sample_worker(
        data, hierarchies_lite,
        None,
        node_to_fine_tensors, sample_types,
        row_ptr_shared=adj_csr[0],
        col_shared=adj_csr[1],
    )

    print(f"Targeting partition {c_idx}. Generating one of each sample type...")
    generated_count = 0
    test_samples = []

    for s_type_idx, sample_type in enumerate(tqdm(sample_types, desc="  Test Generation", ncols=100)):
        sample_data, attempts_used = _generate_one_stratified_sample(h_idx, c_idx, s_type_idx, return_attempts=True)
        if sample_data:
            gq, gpos, g_coarse, _ = sample_data
            test_samples.append((gq, gpos, g_coarse))
            generated_count += 1
            print(f"[OK] {sample_type} in {attempts_used} attempt(s)")
        else:
            print(f"[FAIL] {sample_type} after {attempts_used} attempt(s)")

    print(f"\n--- TEST GENERATION COMPLETE ---")
    print(f"Successfully generated {generated_count} / {len(sample_types)} sample types.")
    if test_samples:
        print(f"💾 Saving {len(test_samples)} generated test samples to '{test_output_dir / 'samples'}'...")
        for i in range(0, len(test_samples), SAVE_BATCH_SIZE):
            chunk = test_samples[i:i + SAVE_BATCH_SIZE]
            batch_idx = i // SAVE_BATCH_SIZE
            filepath = test_output_dir / "samples" / f"batch_{batch_idx:05d}.pt"
            torch.save(chunk, filepath, pickle_protocol=4)
        run_test_validation(test_output_dir)
    else:
        print("No samples were generated, skipping validation.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        h_idx_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        c_idx_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        test_generate_one_partition(h_idx=h_idx_arg, c_idx=c_idx_arg)
    else:
        main()