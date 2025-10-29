# gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from .own_constants import N_NODE_ACTIONS

class GraphEncoder(nn.Module):
    def __init__(self, gnn_in_features: int, emb_size: int, edge_in_dim: int = 4, layers=3, dropout=0.1):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_in_dim, emb_size)
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(layers)])
        self.convs = nn.ModuleList()
        for l in range(layers):
            din = gnn_in_features if l == 0 else emb_size
            mlp = nn.Sequential(
                nn.Linear(din, emb_size), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_size, emb_size),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=emb_size))

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            E = edge_index.size(1)
            edge_attr = x.new_zeros((E, self.edge_encoder.out_features))
        else:
            edge_attr = self.edge_encoder(edge_attr)

        for l, conv in enumerate(self.convs):
            h = conv(x, edge_index, edge_attr)
            h = F.relu(h)
            if h.shape == x.shape:
                h = h + x
            x = self.norms[l](self.dropout(h))
        return x


class PolicyValueNet(nn.Module):
    """
    GNN + GRU policy/value:
      - Node encoder -> global token g_t
      - GRU(g_t, h_{t-1}) -> z_t
      - Policy: node logits from node embeddings + broadcasted z_t
      - Value: from z_t
    """
    def __init__(self,
                 gnn_in_dim: int,
                 emb: int,
                 hid: int = 128,
                 dropout: float = 0.1,
                 layers: int = 3,
                 edge_in_dim: int = 4,
                 rec_dim: int = 256):
        super().__init__()
        self.enc = GraphEncoder(gnn_in_dim, emb, edge_in_dim=edge_in_dim, dropout=dropout, layers=layers)

        self.global_node_feat = nn.Parameter(torch.zeros(gnn_in_dim))
        nn.init.normal_(self.global_node_feat, mean=0.0, std=0.02)

        # Recurrent core (batch_first for 1-step batched inputs)
        self.rnn = nn.GRU(input_size=emb, hidden_size=rec_dim, num_layers=1, batch_first=True)

        # Broadcast GRU state to nodes (FiLM-style additive context)
        self.ctx_proj = nn.Linear(rec_dim, emb)

        self.node_head = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, N_NODE_ACTIONS)
        )
        # Value reads the GRU state directly
        self.value_head = nn.Sequential(
            nn.Linear(rec_dim, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, batch: Batch, h_in: torch.Tensor):
        # Replace global node features with learned token
        if hasattr(batch, "is_global"):
            x = batch.x.clone()
            x[batch.is_global] = self.global_node_feat
        else:
            x = batch.x

        # Node encodings
        h = self.enc(x, batch.edge_index, getattr(batch, "edge_attr", None))  # [sumN, emb]

        # Per-graph embedding from the global token
        g = h[batch.is_global] if hasattr(batch, "is_global") else global_mean_pool(h, batch.batch)  # [B, emb]

        # One-step GRU
        z, h_out = self.rnn(g.unsqueeze(1), h_in)  # z: [B,1,rec_dim], h_out: [1,B,rec_dim]
        z = z[:, -1, :]  # [B, rec_dim]

        # Broadcast GRU context to nodes belonging to each graph
        ctx = self.ctx_proj(z)                     # [B, emb]
        ctx_node = ctx[batch.batch]                # [sumN, emb]
        node_logits_per_node = self.node_head(h + ctx_node)       # [sumN, N_NODE_ACTIONS]

        # Flatten per-node-action logits to [B, A_max] with padding
        flat_node_logits = node_logits_per_node.flatten()
        node_action_batch_idx = batch.batch.repeat_interleave(N_NODE_ACTIONS)
        logits, _ = to_dense_batch(flat_node_logits, node_action_batch_idx, batch_size=g.size(0), fill_value=0)

        values = self.value_head(z).squeeze(-1)    # [B]
        return logits, values, h_out