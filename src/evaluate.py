import torch
import torch.nn.functional as F
from pathlib import Path
import json
import os
from tqdm import tqdm
from src.graph.engine import GraphEngine
from src.model.alignment_mlp import PartitionAlignerMLP
from src.model.alignment_gcn import PartitionAlignerGCN
from src.model.alignment_sage import PartitionAlignerSAGE
from src.model.alignment_gin import PartitionAlignerGIN
from src.data.ingest import AlignmentDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model_type, loss_type, graph_path):
    print(f"  Loading Model: {model_path} ({model_type.upper()})")
    
    # Needs Graph Engine to init model (centroids size)
    
    if not os.path.exists(graph_path):
         raise FileNotFoundError(f"Graph not found: {graph_path}")

    # Load Graph
    graph_data = torch.load(graph_path, map_location=DEVICE, weights_only=False)
    
    if hasattr(graph_data, "part_centroids"):
        centroids = graph_data.part_centroids
    elif hasattr(graph_data, "partition_centroids"):
        centroids = graph_data.partition_centroids
    else:
        # Fallback/Legacy
        print("  ⚠️ Warning: No centroids found. Using random (768).")
        centroids = torch.randn(10, 768).to(DEVICE) 
    
    # Init Model
    part_dim = centroids.size(1)
    if model_type == "mlp":
        model = PartitionAlignerMLP(partition_dim=part_dim, loss_type=loss_type)
    elif model_type == "gcn":
        model = PartitionAlignerGCN(partition_dim=part_dim, loss_type=loss_type)
    elif model_type == "sage":
        model = PartitionAlignerSAGE(partition_dim=part_dim, loss_type=loss_type)
    elif model_type == "gin":
        model = PartitionAlignerGIN(partition_dim=part_dim, loss_type=loss_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Load State
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # Force float32
    centroids = centroids.float()
    model.float()
    
    model.to(DEVICE)
    model.eval()
    
    return model, centroids

def evaluate(model, test_path, loss_type, centroids, graph_path=None):
    """
    Evaluates model on test_path jsonl.
    """
    dataset = AlignmentDataset(test_path, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # For GNNs, we might need partition edges
    partition_edge_index = None
    if isinstance(model, (PartitionAlignerGCN, PartitionAlignerSAGE, PartitionAlignerGIN)):
         # Try to find edges
         if graph_path:
             edge_path = Path(graph_path).parent / "partition_edges.pt"
             if edge_path.exists():
                 partition_edge_index = torch.load(edge_path, map_location=DEVICE, weights_only=False)
             else:
                 # Self loops
                 num_parts = centroids.size(0)
                 partition_edge_index = torch.arange(num_parts, device=DEVICE).unsqueeze(0).repeat(2, 1)

    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    mrr_sum = 0.0
    total = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE) 
            multi_labels = batch["multi_labels"].to(DEVICE)
            
            # Forward
            if partition_edge_index is not None:
                text_emb, part_emb = model(input_ids, attention_mask, centroids, partition_edge_index)
            else:
                text_emb, part_emb = model(input_ids, attention_mask, centroids)
            
            # Defensive: Ensure Float32
            text_emb = text_emb.float()
            part_emb = part_emb.float()
                
            # Logits Calculation
            if loss_type == "bce":
                logits = torch.matmul(text_emb, part_emb.t())
            else: # InfoNCE
                logits = torch.matmul(text_emb, part_emb.t()) * torch.exp(model.temperature)
                
            # Top-K
            _, topk = logits.topk(10, dim=1) # [Batch, 10]
            
            # Metrics
            for i in range(len(labels)):
                m_labels = multi_labels[i]
                true_set = set(m_labels[m_labels != -1].tolist())
                pred_list = topk[i].tolist()
                
                # P@1
                if pred_list[0] in true_set:
                    hits_1 += 1
                
                # R@5
                # Using standard Recall@K: |Intersection| / |True Set|
                
                def get_recall(k):
                    pred_set = set(pred_list[:k])
                    inter = true_set & pred_set
                    return len(inter) / len(true_set) if len(true_set) > 0 else 0
                
                hits_5 += get_recall(5)
                hits_10 += get_recall(10)

                # MRR Calculation (Mean Reciprocal Rank)
                # Find the first correct item in the predicted list
                mrr_score = 0.0
                for rank, pred_idx in enumerate(pred_list, start=1):
                     if pred_idx in true_set:
                         mrr_score = 1.0 / rank
                         break
                mrr_sum += mrr_score
                
                total += 1
                
    return {
        "P@1": hits_1 / total if total > 0 else 0,
        "R@5": hits_5 / total if total > 0 else 0,
        "R@10": hits_10 / total if total > 0 else 0,
        "MRR": mrr_sum / total if total > 0 else 0,
        "Total": total
    }
