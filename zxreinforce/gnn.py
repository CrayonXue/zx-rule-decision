import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.data import Batch


from .own_constants import (N_NODE_ACTIONS, N_EDGE_ACTIONS)

class GraphEncoder(nn.Module):
    def __init__(self, 
                 gnn_in_features: int, 
                 emb_size: int):
        super().__init__()
        self.emb_size = emb_size

        self.gnn = nn.ModuleList([
            GCNConv(gnn_in_features, emb_size),
            GCNConv(emb_size, emb_size),
            GCNConv(emb_size, emb_size)
        ])

    def forward(self, x, edge_index):
        for conv in self.gnn:
            x = conv(x, edge_index).relu()
        return x


class PolicyValueNet(nn.Module):
    """
    GNN policy+value head that:
      - computes per-node logits (N_NODE_ACTIONS each),
      - computes per-edge logits (N_EDGE_ACTIONS each) for every column in edge_index,
      - concatenates [node_flat, edge_flat] per graph,
      - right-pads all graphs in the batch to the same length,
      - returns (logits, values) where logits.shape == [B, A_max_in_batch].

    This ordering matches the environment’s action mask:
      [node0_actions ... node(N-1)_actions, edge0_actions ... edge(E-1)_actions]
    """
    def __init__(self,
                 gnn_in_dim: int,
                 emb: int,
                 hid: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.enc = GraphEncoder(gnn_in_dim, emb)

        self.node_head = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, N_NODE_ACTIONS)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(emb * 2, hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, N_EDGE_ACTIONS)
        )
        self.value_head = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, batch: Batch):
        # Encode nodes
        h = self.enc(batch.x, batch.edge_index)          # [sumN, emb]
        g = global_mean_pool(h, batch.batch)             # [B, emb]
        B = g.size(0)
        device = h.device

        # Per-node logits
        node_logits_per_node = self.node_head(h)         # [sumN, N_NODE_ACTIONS]

        # Per-edge logits for EVERY edge in edge_index (no dedup; keep env order)
        E_total = batch.edge_index.size(1)
        if E_total > 0:
            src, dst = batch.edge_index[0], batch.edge_index[1]   # [sumE]
            h_u, h_v = h[src], h[dst]
            edge_logits_per_edge = self.edge_head(torch.cat([h_u, h_v], dim=-1))  # [sumE, N_EDGE_ACTIONS]
            # Assign each edge to a graph via its source node's graph id (src and dst must belong to same graph)
            edge_batch = batch.batch[src]                          # [sumE]
        else:
            edge_logits_per_edge = h.new_zeros((0, N_EDGE_ACTIONS))
            edge_batch = torch.zeros(0, dtype=torch.long, device=device)

        # Counts and offsets per graph
        node_counts = torch.bincount(batch.batch, minlength=B)     # [B]
        edge_counts = torch.bincount(edge_batch, minlength=B)      # [B]

        node_offsets = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long, device=device), node_counts[:-1]]), dim=0
        )  # [B]
        edge_offsets = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long, device=device), edge_counts[:-1]]), dim=0
        )  # [B]

        values = self.value_head(g).squeeze(-1)                    # [B]

        # Assemble per-graph action vectors
        logits_list = []
        for i in range(B):
            n_i = int(node_counts[i].item())
            e_i = int(edge_counts[i].item())
            n0 = int(node_offsets[i].item())
            e0 = int(edge_offsets[i].item())

            # Flatten node/edge chunks for this graph
            node_flat = node_logits_per_node[n0:n0 + n_i].reshape(-1)   # n_i * N_NODE_ACTIONS
            edge_flat = edge_logits_per_edge[e0:e0 + e_i].reshape(-1)   # e_i * N_EDGE_ACTIONS

            vec = torch.cat([node_flat, edge_flat], dim=0)  # [A_i]
            logits_list.append(vec)

        # Right-pad to batchwise max action length
        maxA = max((v.numel() for v in logits_list), default=1)
        padded = []
        for v in logits_list:
            if v.numel() < maxA:
                pad = torch.zeros(maxA - v.numel(), device=v.device, dtype=v.dtype)
                v = torch.cat([v, pad], dim=0)
            padded.append(v)
        logits = torch.stack(padded, dim=0)  # [B, maxA]

        return logits, values