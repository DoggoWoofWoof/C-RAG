import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import SAGEConv

from .alignment_mlp import PartitionAlignerMLP

class PartitionAlignerSAGE(PartitionAlignerMLP):
    def __init__(self, model_name="intfloat/e5-base-v2", partition_dim=768, hidden_dim=256, loss_type='infonce'):
        # Pass loss_type to parent
        super().__init__(model_name, partition_dim, loss_type=loss_type)
        
        # SAGE Layers
        self.sage1 = SAGEConv(partition_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, partition_dim) # Output must match embedding_dim for dot product
        
        self.activation = nn.ReLU()
        # Temperature handled by parent
        
    def forward(self, input_ids, attention_mask, part_centroids, partition_edge_index):
        # 1. Text Tower
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean Pooling (Parent method)
        text_emb = self.mean_pooling(outputs, attention_mask)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
        
        # 2. Graph Tower (SAGE)
        x = self.sage1(part_centroids, partition_edge_index)
        x = self.activation(x)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        part_emb = self.sage2(x, partition_edge_index)
        
        part_emb = torch.nn.functional.normalize(part_emb, p=2, dim=1)
        
        return text_emb, part_emb
