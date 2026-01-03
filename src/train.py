import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import json
import os


from src.data.ingest import AlignmentDataset

def train_alignment_model(epochs=5, batch_size=32, lr=2e-5, model_type="mlp", loss_type="infonce", graph_path="data/wiki/graph/full_graph.pt", train_path="data/wiki/train.json", max_steps=None):
    """
    Generic Training Loop for Experiment Matrix.
    Args:
        model_type: 'mlp' (Baseline) or 'gcn' (Graph Tower).
        loss_type: 'infonce' (Contrastive) or 'bce' (Multi-Label).
        graph_path: Path to the partitioned graph (e.g., full_graph_leiden.pt).
        train_path: Path to the training data JSON.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  • Training {model_type.upper()} + {loss_type.upper()} on {device}")
    
    # 1. Load Data
    # train_path is passed as arg now
    
    if not os.path.exists(graph_path) or not os.path.exists(train_path):
        print(f"Missing {graph_path} or {train_path}. Cannot train.")
        return
        
    print(f"    Loading Graph: {graph_path}...")
    # weights_only=False required to load PyG Data object
    full_graph = torch.load(graph_path, map_location=device, weights_only=False)
    # Ensure Float32 for training (especially on CPU or for precision)
    centroids = full_graph.part_centroids.to(device).float() # [num_parts, dim]
    
    # 2. Init Model
    if model_type == "gcn":
        from .model.alignment_gcn import PartitionAlignerGCN
        model = PartitionAlignerGCN(loss_type=loss_type)
    elif model_type == "sage":
        from .model.alignment_sage import PartitionAlignerSAGE
        model = PartitionAlignerSAGE(loss_type=loss_type)
    elif model_type == "gin":
        from .model.alignment_gin import PartitionAlignerGIN
        model = PartitionAlignerGIN(loss_type=loss_type)
    else:
        # Default MLP (for InfoNCE or BCE generic)
        from .model.alignment_mlp import PartitionAlignerMLP
        model = PartitionAlignerMLP(loss_type=loss_type)
        
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    dataset = AlignmentDataset(train_path, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Training Loop
    model.train()
    
    partition_edge_index = None
    if model_type in ["gcn", "sage", "gin"]:
         # Deduce edges path from graph_path
         graph_dir = os.path.dirname(graph_path)
         edge_path = os.path.join(graph_dir, "partition_edges.pt")
         
         if os.path.exists(edge_path):
             print(f"    Loading Partition Edges from {edge_path}")
             partition_edge_index = torch.load(edge_path, map_location=device)
         else:
             print("    ⚠️ Using Self-Loops (Real edges not found)")
             num_parts = centroids.size(0)
             partition_edge_index = torch.arange(num_parts, device=device).unsqueeze(0).repeat(2, 1)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Accumulators
        epoch_p1 = 0
        epoch_r3_h = 0; epoch_r3_t = 0
        epoch_r5_h = 0; epoch_r5_t = 0
        epoch_r10_h = 0; epoch_r10_t = 0
        epoch_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            if max_steps and i >= max_steps:
                print(f"  🛑 Reached max_steps ({max_steps}). Stopping epoch.")
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device) # [Batch]
            if 'multi_labels' in batch:
                multi_labels = batch['multi_labels'].to(device)
            else:
                multi_labels = labels.unsqueeze(1) # Fallback
            
            optimizer.zero_grad()
            
            # Forward Pass Logic
            if model_type in ["gcn", "sage", "gin"]:
                 text_emb, part_emb = model(input_ids, attention_mask, centroids, partition_edge_index)
            else:
                 text_emb, part_emb = model(input_ids, attention_mask, centroids)
            
            # Loss Calculation
            if loss_type == "bce":
                # BCE expects multi-hot targets. 
                # Construct from padded multi_labels [Batch, 10]
                num_parts = part_emb.size(0)
                targets = torch.zeros(labels.size(0), num_parts, device=device)
                
                # Manual scatter loop to handle padding -1
                for i in range(labels.size(0)):
                    m_labels = multi_labels[i]
                    valid = m_labels[m_labels != -1]
                    if len(valid) > 0:
                        targets[i, valid] = 1.0
                
                logits = torch.matmul(text_emb, part_emb.t())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                
                # For metrics, we need logits
                if not 'logits' in locals():
                    logits = torch.matmul(text_emb, part_emb.t())
                    
            else: # InfoNCE (Default)
                # Cosine Similarity
                logits = torch.matmul(text_emb, part_emb.t()) * torch.exp(model.temperature)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # --- Shared Metric Calculation ---
            # We need top-10 to cover all metrics up to 10
            # For BCE, logits are raw scores. For InfoNCE, logits are scaled cosines. 
            # Ranking logic is identical (argmax).
            _, topk_indices = logits.topk(10, dim=1) # [Batch, 10]
            
            def calc_metrics(topk_indices, multi_labels):
                batch_p1 = 0 
                batch_r3_hits, batch_r3_total = 0, 0
                batch_r5_hits, batch_r5_total = 0, 0
                batch_r10_hits, batch_r10_total = 0, 0
                
                preds_to_print = []
                
                for i in range(len(multi_labels)):
                        m_labels = multi_labels[i]
                        true_set = m_labels[m_labels != -1].tolist()
                        
                        # P@1
                        if topk_indices[i, 0].item() in true_set:
                            batch_p1 += 1
                            
                        # Helper for R@K
                        def get_hits(k):
                            pred_set = topk_indices[i, :k].tolist()
                            return len(set(true_set) & set(pred_set))
                        
                        # Hits
                        hits_3 = get_hits(3)
                        hits_5 = get_hits(5)
                        hits_10 = get_hits(10)
                        
                        batch_r3_hits += hits_3; batch_r3_total += len(true_set)
                        batch_r5_hits += hits_5; batch_r5_total += len(true_set)
                        batch_r10_hits += hits_10; batch_r10_total += len(true_set)
                        
                        if i < 2:
                            preds_to_print.append(f"    Sample {i}: True={true_set} | P@10={topk_indices[i, :10].tolist()}")
                
                return batch_p1, batch_r3_hits, batch_r3_total, batch_r5_hits, batch_r5_total, batch_r10_hits, batch_r10_total, preds_to_print

            # Calculate
            p1, r3_h, r3_t, r5_h, r5_t, r10_h, r10_t, debug_strs = calc_metrics(topk_indices, multi_labels)
            
            # Accumulate
            epoch_p1 += p1
            epoch_r3_h += r3_h; epoch_r3_t += r3_t
            epoch_r5_h += r5_h; epoch_r5_t += r5_t
            epoch_r10_h += r10_h; epoch_r10_t += r10_t
            epoch_samples += len(labels)
            
            # Logging
            current_p1 = p1 / len(labels)
            current_r5 = r5_h / r5_t if r5_t > 0 else 0
            current_r10 = r10_h / r10_t if r10_t > 0 else 0
        
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update desc
            pbar.set_postfix({
                "L": f"{total_loss / (pbar.n + 1):.2f}", 
                "P1": f"{current_p1:.0%}", 
                "R5": f"{current_r5:.0%}"
            })
            
            if (pbar.n + 1) % 500 == 0:
                    print(f"\n[DEBUG Step {pbar.n+1}]")
                    for s in debug_strs:
                        print(s)
            
        # End of Epoch Summary
        print(f"  ✓ Epoch {epoch+1} finished.")
        
        # Calculate final batch metrics for display (approx)
        avg_loss = total_loss / (len(dataloader) if len(dataloader) > 0 else 1)
        avg_p1 = epoch_p1 / epoch_samples if epoch_samples > 0 else 0
        avg_r3 = epoch_r3_h / epoch_r3_t if epoch_r3_t > 0 else 0
        avg_r5 = epoch_r5_h / epoch_r5_t if epoch_r5_t > 0 else 0
        avg_r10 = epoch_r10_h / epoch_r10_t if epoch_r10_t > 0 else 0
        
        print(f"    (Epoch Avg)  Loss: {avg_loss:.4f} | P@1: {avg_p1:.2%} | R@3: {avg_r3:.2%} | R@5: {avg_r5:.2%} | R@10: {avg_r10:.2%}")
            

        
    # 4. Save
    save_name = f"alignment_{model_type}_{loss_type}.pt"
    
    # Save to data/wiki/graph/leiden/model or similar
    graph_dir = os.path.dirname(graph_path) # e.g. data/wiki/graph/leiden
    model_dir = os.path.join(graph_dir, "model")
    
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
