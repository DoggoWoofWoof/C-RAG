import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .alignment_mlp import PartitionAlignerMLP

class PartitionAlignerGCN(PartitionAlignerMLP):
    """
    Experiment C/D: GCN-based Graph Tower.
    Instead of projecting Centroids via MLP, we run message passing 
    on the 'Partition Graph' (Graph of Communities).
    """
    def __init__(self, model_name='intfloat/e5-base-v2', partition_dim=768, hidden_dim=768, loss_type='infonce'):
        super().__init__(model_name, partition_dim, loss_type=loss_type)
        
        # Replace MLP with GCN
        # We need to construct the Graph Tower carefully.
        # It takes [Num_Parts, Dim] and [Edges] -> [Num_Parts, Dim]
        
        self.partition_proj = nn.Identity() # Disable the definition from parent
        
        self.gcn1 = GCNConv(partition_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        
    def forward_graph_tower(self, centroids, edge_index):
        """
        Args:
            centroids: [Num_Parts, 768]
            edge_index: [2, Num_Inter_Partition_Edges]
        """
        x = self.gcn1(centroids, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gcn2(x, edge_index)
        return x

    def forward(self, input_ids, attention_mask, centroids, partition_edge_index):
        """
        Overridden forward to accept partition_edge_index.
        """
        # 1. Text Tower
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.mean_pooling(outputs, attention_mask)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # 2. Graph Tower (GCN)
        part_emb = self.forward_graph_tower(centroids, partition_edge_index)
        part_emb = F.normalize(part_emb, p=2, dim=1)
        
        return text_emb, part_emb
