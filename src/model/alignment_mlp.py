import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class PartitionAlignerMLP(nn.Module):
    """
    CLIP-style Dual Encoder to align User Queries with Graph Partitions.
    """
    def __init__(self, model_name='intfloat/e5-base-v2', partition_dim=768, loss_type='infonce'):
        super().__init__()
        self.loss_type = loss_type
        
        # 1. Text Encoder (Query Side)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_dim = self.text_encoder.config.hidden_size
        
        # 2. Partition Encoder (Graph Side)
        # Input: Partition Centroid (Average of all node embeddings in partition)
        # We project this to the common latent space.
        self.partition_proj = nn.Sequential(
            nn.Linear(partition_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, input_ids, attention_mask, partition_centroids, partition_edge_index=None):
        # partition_edge_index is optional for MLP but needed for GNN signature compatibility
        
        # 1. Encode Text
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean Pooling for E5/BERT
        text_emb = self.mean_pooling(outputs, attention_mask)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # 2. Encode Partitions
        part_emb = self.partition_proj(partition_centroids)
        part_emb = F.normalize(part_emb, p=2, dim=1)
        
        return text_emb, part_emb

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_loss(self, text_emb, part_emb, target_indices=None):
        if self.loss_type == 'bce':
            # BCE Loss (Multi-Label)
            # Logits = Cosine Sim * Temp
            logits = torch.matmul(text_emb, part_emb.t()) * torch.exp(self.temperature)
            return F.binary_cross_entropy_with_logits(logits, target_indices)
        else:
            # InfoNCE Loss (Contrastive)
            logits = torch.matmul(text_emb, part_emb.t()) * torch.exp(self.temperature)
            labels = torch.arange(len(text_emb), device=text_emb.device)
            return F.cross_entropy(logits, labels)
