import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


from .own_constants import N_NODE_ACTIONS

class GraphEncoder(nn.Module):
    def __init__(self, 
                 gnn_in_features: int, 
                 emb_size: int,
                 layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(gnn_in_features, emb_size)] +
                                   [GCNConv(emb_size, emb_size) for _ in range(layers-1)])
        self.norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            h = F.relu(h)
            if h.shape == x.shape:  # residual from 2nd layer onward
                h = h + x
            x = self.norms[i](self.dropout(h))
        return x


class PolicyValueNet(nn.Module):
    """
    GNN policy+value head that:
      - computes per-node logits (N_NODE_ACTIONS each),
      - right-pads all graphs in the batch to the same length,
      - returns (logits, values) where logits.shape == [B, A_max_in_batch].

    This ordering matches the environment’s action mask:
      [node0_actions ... node(N-1)_actions]
    """
    def __init__(self,
                 gnn_in_dim: int,
                 emb: int,
                 hid: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.enc = GraphEncoder(gnn_in_dim, emb)

        self.global_node_feat = nn.Parameter(torch.zeros(gnn_in_dim))
        nn.init.normal_(self.global_node_feat, mean=0.0, std=0.02)

        self.node_head = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, N_NODE_ACTIONS)
        )

        self.value_head = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, batch: Batch):
        # If a global node exists, replace its raw features with the learned token
        if hasattr(batch, "is_global"):
            x = batch.x.clone()
            x[batch.is_global] = self.global_node_feat
        else:
            x = batch.x

        # Encode nodes
        h = self.enc(x, batch.edge_index)                  # [sumN, emb]

        # Graph representation: prefer the global node embedding if present
        if hasattr(batch, "is_global"):
            # Exactly one True per graph; they appear in batch order
            g = h[batch.is_global]                         # [B, emb]
        else:
            g = global_mean_pool(h, batch.batch)           # [B, emb]

        # Policy head: per-node logits (including the global node; mask will kill them)
        node_logits_per_node = self.node_head(h)           # [sumN, N_NODE_ACTIONS]
        flat_node_logits = node_logits_per_node.flatten()  # [sumN * N_NODE_ACTIONS]
        node_action_batch_idx = batch.batch.repeat_interleave(N_NODE_ACTIONS)
        logits, _ = to_dense_batch(flat_node_logits, node_action_batch_idx, batch_size=g.size(0), fill_value=0)

        # Value head from the graph embedding
        values = self.value_head(g).squeeze(-1)            # [B]
        return logits, values