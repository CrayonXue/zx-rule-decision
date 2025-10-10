import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


from .own_constants import (N_NODE_ACTIONS, N_EDGE_ACTIONS)

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
            nn.Linear(2*emb, hid), nn.ReLU(),
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
        B = g.size(0) # number of environments
        device = h.device

        # Per-node logits
        node_logits_per_node = self.node_head(h)         # [sumN, N_NODE_ACTIONS]

        if N_EDGE_ACTIONS == 0:
            # Shortcut: no edge actions
            # Just pad node logits to [B, maxA]
            flat_node_logits = node_logits_per_node.flatten()  # [sum(N_i * N_NODE_ACTIONS)]
            node_action_batch_idx = batch.batch.repeat_interleave(N_NODE_ACTIONS)
            logits, _ = to_dense_batch(flat_node_logits, node_action_batch_idx, batch_size=B, fill_value=0)
            values = self.value_head(g).squeeze(-1)
            return logits, values
        
        else:
            # per-edge logits on action edges
            eia = batch.edge_index
            if eia is None or eia.numel() == 0:
                edge_logits = torch.empty(0, N_EDGE_ACTIONS, device=device, dtype=node_logits_per_node.dtype)
                edge_graph = torch.empty(0, dtype=torch.long, device=device)
            else:
                u_all, v_all = eia[0], eia[1]            # directed edges
                is_canon = u_all < v_all                 # keep only one direction per undirected edge
                u = u_all[is_canon]
                v = v_all[is_canon]
                edge_graph = batch.batch[u]              # graph id per canonical edge
                uv_feat = torch.cat([h[u], h[v]], dim=1) # [sumE_canon, 2*emb]
                edge_logits = self.edge_head(uv_feat)    # [sumE_canon, N_EDGE_ACTIONS]

            start_e = torch.zeros(B+1, dtype=torch.long, device=device)
            if edge_graph.numel() > 0:
                # prefix sums to slice edges by graph
                e_counts = torch.bincount(edge_graph, minlength=B)            # [B]
                start_e[1:] = torch.cumsum(e_counts, dim=0)

            # 1. Flatten all action logits into a single tensor
            flat_node_logits = node_logits_per_node.flatten()  # [sum(N_i * N_NODE_ACTIONS)]
            flat_edge_logits = edge_logits.flatten()            # [sum(E_i * N_EDGE_ACTIONS)]

            # 2. Create a batch index for each action logit
            # The batch index for each node is in batch.batch
            # Repeat it for each of the node's actions
            node_action_batch_idx = batch.batch.repeat_interleave(N_NODE_ACTIONS)

            # The batch index for each edge action is derived from edge_graph
            edge_action_batch_idx = edge_graph.repeat_interleave(N_EDGE_ACTIONS)

            # 3. Concatenate all logits and their corresponding batch indices
            all_logits = torch.cat([flat_node_logits, flat_edge_logits], dim=0)
            all_batch_indices = torch.cat([node_action_batch_idx, edge_action_batch_idx], dim=0)

            # Compute expected per-graph widths for verification
            with torch.no_grad():
                node_counts = torch.bincount(batch.batch, minlength=B)  # [B]
                edge_counts = torch.bincount(edge_graph, minlength=B) if edge_graph.numel() > 0 else torch.zeros(B, device=device, dtype=torch.long)
                expectedA = node_counts * N_NODE_ACTIONS + edge_counts * N_EDGE_ACTIONS  # [B]

            # 4. Use to_dense_batch to create the padded [B, maxA] tensor
            # It also conveniently returns the mask!
            logits, _ = to_dense_batch(all_logits, all_batch_indices, batch_size=B, fill_value=0)

           

            # # Sanity check: verify that to_dense_batch respected our expected widths
            # actualA = (all_batch_indices.unsqueeze(0) == torch.arange(B, device=device).unsqueeze(1)).sum(dim=1)
            # assert torch.equal(actualA, expectedA), f"Per-graph widths mismatch: actual={actualA.tolist()} expected={expectedA.tolist()}"

            # per-graph segment lengths
            with torch.no_grad():
                node_counts = torch.bincount(batch.batch, minlength=B)
                edge_counts = torch.bincount(edge_graph, minlength=B) if edge_graph.numel() > 0 else torch.zeros(B, device=device, dtype=torch.long)
                
                node_len = node_counts * N_NODE_ACTIONS
                edge_len = edge_counts * N_EDGE_ACTIONS

            # add per-family biases
            for i in range(B):
                nA = int(node_len[i])
                eA = int(edge_len[i])
                if nA > 0:
                    logits[i, :nA] += -math.log(nA)
                if eA > 0:
                    logits[i, nA:nA+eA] += -math.log(eA)


            # Get values (this part was already correct)
            values = self.value_head(g).squeeze(-1)


            return logits, values