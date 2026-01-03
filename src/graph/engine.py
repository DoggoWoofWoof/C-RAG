import os
import torch
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import multiprocessing
import pymetis
from tqdm import tqdm

class GraphEngine:
    """
    Core engine for managing the large-scale graph, partitioning, and efficient subgraph extraction.
    """
    def __init__(self, edges_path=None, nodes_path=None, device='cpu'):
        self.device = device
        self.data = None  # PyG Data object
        self.adj_t = None # CSR SparseTensor
        self.node_text_map = {} # ID -> Text
        
        if edges_path:
            self.load_graph(edges_path, nodes_path)

    def load_graph(self, edges_path, nodes_path):
        """
        Loads graph from processed JSONL files.
        """
        print(f"  • Loading graph from {nodes_path} and {edges_path}...")
        import json
        
        # 1. Load Nodes
        x_list = [] # We might load embeddings here if they exist, else placeholder
        self.node_text_map = {}
        
        # We need to ensure ID mapping is consistent 0..N-1
        # The ingestion script aims to produce 0..N-1 IDs, but we verify.
        max_id = 0
        
        with open(nodes_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Nodes"):
                obj = json.loads(line)
                nid = obj['id']
                self.node_text_map[nid] = obj
                max_id = max(max_id, nid)
        
        num_nodes = max_id + 1
        print(f"    Found {num_nodes} nodes.")
        
        # 2. Load Edges
        src_list = []
        dst_list = []
        with open(edges_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Edges"):
                obj = json.loads(line)
                src_list.append(obj['src'])
                dst_list.append(obj['dst'])
                
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        # 3. Create PyG Data
        # We start with empty X, usually filled by Embedding step
        self.data = Data(num_nodes=num_nodes, edge_index=edge_index)
        
        # Initialize sparse structure
        self.from_pyg_data(self.data)

    def load_text_map(self, nodes_path):
        """
        Loads only the node text mapping from nodes.jsonl.
        Useful when loading a pre-saved graph .pt file that doesn't store the dict.
        """
        print(f"  • Loading text map from {nodes_path}...")
        import json
        self.node_text_map = {}
        with open(nodes_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Text Map"):
                obj = json.loads(line)
                self.node_text_map[obj['id']] = obj
        print(f"    Loaded {len(self.node_text_map)} text entries.")

    def from_pyg_data(self, data: Data):
        """
        Initialize from an existing PyG Data object (useful for testing/prototyping).
        """
        self.data = data
        self.data.to(self.device)
        print("  - Creating SparseTensor (CSR) for fast slicing...")
        self.adj_t = SparseTensor.from_edge_index(
            self.data.edge_index, 
            sparse_sizes=(self.data.num_nodes, self.data.num_nodes)
        ).to(self.device)
        print("  ✓ Graph initialized.")

    def partition_graph(self, num_parts):
        """
        Runs METIS to partition the graph.
        Returns:
            part_graphs (list): List of subgraph Data objects.
            part_nodes_map (dict): Map PartID -> Tensor of Global Node IDs.
        """
        print(f"  • Partitioning graph into {num_parts} parts...")
        
        # METIS requires CPU-based adjacency lists
        adj = self.adj_t.cpu()
        xadj_t, adjncy_t, _ = adj.csr()
        xadj, adjncy = xadj_t.tolist(), adjncy_t.tolist()
        
        _, membership = pymetis.part_graph(num_parts, xadj=xadj, adjncy=adjncy)
        
        part_graphs = []
        part_nodes_map = {}
        
        for part_id in range(num_parts):
            node_indices = [i for i, p in enumerate(membership) if p == part_id]
            if not node_indices:
                continue
                
            nodes_tensor = torch.tensor(node_indices, dtype=torch.long, device=self.device)
            part_nodes_map[part_id] = nodes_tensor
            
            # Extract subgraph
            # We use the efficient extraction logic
            subgraph = self.extract_subgraph(nodes_tensor)
            part_graphs.append(subgraph)
            
        return part_graphs, part_nodes_map

    def extract_subgraph(self, node_indices):
        """
        Efficiently extracts a subgraph for the given global node indices.
        """
        # This is the "Magic Slicing" logic from train_jigsaw_model.py
        
        # 1. Slice Structure (Adjacency)
        # adj_t[rows, cols] efficiently returns the sub-matrix
        sub_adj = self.adj_t[node_indices, node_indices]
        row, col, _ = sub_adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        
        # 2. Slice Features
        x = self.data.x[node_indices] if self.data.x is not None else None
        
        return Data(x=x, edge_index=edge_index, num_nodes=len(node_indices))

    def get_neighbors(self, node_idx):
        """
        Get neighbors of a specific node using CSR.
        """
        row, col, _ = self.adj_t[node_idx].coo()
        return col

    def save(self, path):
        torch.save(self.data, path)

    def load(self, path):
        # weights_only=False required for PyG Data objects
        self.data = torch.load(path, map_location=self.device, weights_only=False)
        self.from_pyg_data(self.data)
