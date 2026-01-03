import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GINConv

from .alignment_mlp import PartitionAlignerMLP
from torch_geometric.nn import GINConv

class PartitionAlignerGIN(PartitionAlignerMLP):
    def __init__(self, model_name="intfloat/e5-base-v2", partition_dim=768, hidden_dim=256, loss_type='infonce'):
        super().__init__(model_name, partition_dim, loss_type=loss_type)
        
        # GIN requires an MLP for aggregation
        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(partition_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, partition_dim)
            )
        )
        
    def forward(self, input_ids, attention_mask, part_centroids, partition_edge_index):
        # 1. Text Tower
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Parent Method
        text_emb = self.mean_pooling(outputs, attention_mask)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
        
        # 2. Graph Tower (GIN)
        x = self.gin1(part_centroids, partition_edge_index)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        part_emb = self.gin2(x, partition_edge_index)
        
        part_emb = torch.nn.functional.normalize(part_emb, p=2, dim=1)
        
        return text_emb, part_emb
