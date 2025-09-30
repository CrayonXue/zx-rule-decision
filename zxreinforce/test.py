# candidate_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool  # or your pooling
# If you use torch_scatter, import it too:
import torch_scatter

class CandidatePolicyValueNet(nn.Module):
    def __init__(self, gnn_backbone, hid_dim, n_ops, rule_dim=32):
        """
        gnn_backbone: nn.Module that maps a PyG Batch -> (node_emb [sumN,hid], batch_index [sumN])
        hid_dim: node embedding size from your GNN
        n_ops: number of operation types (rules)
        rule_dim: dimension of rule embeddings
        """
        super().__init__()
        self.gnn = gnn_backbone
        self.rule_table = nn.Embedding(n_ops, rule_dim)
        # Scoring function s(i,r) = Bilinear(h_i, e_r)
        self.bilinear = nn.Bilinear(hid_dim, rule_dim, 1, bias=True)
        # Value head on pooled graph embedding
        self.value_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def encode_nodes(self, pyg_batch):
        """
        Returns:
          h: [sumN, H]
          batch_idx: [sumN] graph id per node (0..B-1)
          g_emb: [B, H] pooled graph embeddings
        """
        h, batch_idx = self.gnn(pyg_batch)  # adapt to your backbone's API
        g_emb = global_mean_pool(h, batch_idx)
        return h, batch_idx, g_emb

    def value(self, g_emb):
        return self.value_mlp(g_emb).squeeze(-1)  # [B]

    def score_candidates(self, h_nodes, cand_nodes, cand_ops):
        """
        Inputs:
          h_nodes: [sumN, H] node embeddings for the whole batch
          cand_nodes: LongTensor [M] flat node indices into h_nodes
          cand_ops:   LongTensor [M] op ids (0..n_ops-1), aligned with cand_nodes
        Output:
          scores: [M] unnormalized scores for each candidate
        """
        H = h_nodes[cand_nodes]               # [M, H]
        E = self.rule_table(cand_ops)         # [M, rule_dim]
        S = self.bilinear(H, E).squeeze(-1)   # [M]
        return S


# candidate_utils.py
import torch

def per_graph_node_offsets(batch_idx: torch.Tensor, B: int):
    """
    batch_idx: [sumN] node->graph ids
    Returns:
      offsets: [B] start flat node index for each graph (prefix sums)
      sizes:   [B] number of nodes per graph
    """
    # sizes per graph
    sizes = torch.bincount(batch_idx, minlength=B)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=batch_idx.device),
                         torch.cumsum(sizes[:-1], dim=0)], dim=0)
    return offsets, sizes

def build_candidates_from_mask(mask_nk: torch.Tensor):
    """
    mask_nk: BoolTensor [N, K] for a single graph
    Returns:
      cand_nodes_local: [M] local node indices (0..N-1)
      cand_ops:         [M] op ids (0..K-1)
    """
    nz = mask_nk.nonzero(as_tuple=False)  # [M, 2]
    if nz.numel() == 0:
        return (mask_nk.new_zeros((0,), dtype=torch.long),
                mask_nk.new_zeros((0,), dtype=torch.long))
    return nz[:, 0].long(), nz[:, 1].long()

def pack_batch_candidates(node_masks_list, batch_idx, B):
    """
    node_masks_list: list length B, each is BoolTensor [N_b, K]
    batch_idx: [sumN] node->graph ids from PyG Batch
    B: batch size (#graphs)
    Returns:
      cand_nodes_flat: [M_total] flat node indices into h_nodes
      cand_ops_flat:   [M_total] op ids
      seg_ids:         [M_total] graph id per candidate (0..B-1)
      per_graph_M:     [B] number of candidates per graph
    """
    device = batch_idx.device
    offsets, sizes = per_graph_node_offsets(batch_idx, B)
    all_nodes, all_ops, all_seg = [], [], []
    per_graph_M = []

    for g, mask_nk in enumerate(node_masks_list):
        cn_local, co = build_candidates_from_mask(mask_nk)  # [M_g], [M_g]
        M_g = cn_local.shape[0]
        per_graph_M.append(M_g)
        if M_g > 0:
            cn_flat = cn_local + offsets[g]  # map to flat indices in h_nodes
            all_nodes.append(cn_flat)
            all_ops.append(co.to(device))
            all_seg.append(torch.full((M_g,), g, dtype=torch.long, device=device))

    if len(all_nodes) == 0:
        # No candidates in the whole batch
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, empty, torch.zeros(B, dtype=torch.long, device=device)

    cand_nodes_flat = torch.cat(all_nodes, dim=0)
    cand_ops_flat = torch.cat(all_ops, dim=0)
    seg_ids = torch.cat(all_seg, dim=0)
    per_graph_M = torch.tensor(per_graph_M, dtype=torch.long, device=device)
    return cand_nodes_flat, cand_ops_flat, seg_ids, per_graph_M


# segment_math.py
import torch

def segment_logsumexp(scores: torch.Tensor, seg_ids: torch.Tensor, B: int):
    """
    scores: [M_total]
    seg_ids: [M_total] in 0..B-1
    Returns:
      lse_per_graph: [B]
    """
    # max per segment
    max_buf = torch.full((B,), -float("inf"), device=scores.device, dtype=scores.dtype)
    max_buf = max_buf.scatter_reduce(0, seg_ids, scores, reduce="amax", include_self=True)

    # subtract and exp-sum
    expsum = torch.zeros(B, device=scores.device, dtype=scores.dtype)
    expsum = expsum.scatter_add(0, seg_ids, torch.exp(scores - max_buf[seg_ids]))
    lse = torch.log(expsum) + max_buf
    return lse

def segment_log_softmax(scores: torch.Tensor, seg_ids: torch.Tensor, B: int):
    """
    Returns:
      log_probs: [M_total] log softmax per segment
    """
    lse = segment_logsumexp(scores, seg_ids, B)  # [B]
    return scores - lse[seg_ids]

def segment_entropy_from_logp(logp: torch.Tensor, seg_ids: torch.Tensor, B: int):
    """
    Entropy H = -sum p*logp per segment; return [B]
    """
    p = torch.exp(logp)
    contrib = -p * logp  # [M_total]
    H = torch.zeros(B, device=logp.device, dtype=logp.dtype)
    H = H.scatter_add(0, seg_ids, contrib)
    return H  # [B]


# candidate_policy_runtime.py
import torch
from torch.distributions.categorical import Categorical
from candidate_utils import pack_batch_candidates
from segment_math import segment_log_softmax, segment_entropy_from_logp

@torch.no_grad()
def sample_actions_from_candidates(net, pyg_batch, node_masks_list, device=None, greedy=False):
    """
    Inputs:
      net: CandidatePolicyValueNet
      pyg_batch: PyG Batch of B graphs
      node_masks_list: list length B, each BoolTensor [N_b, K] for that graph
    Returns:
      actions: LongTensor [B, 2] -> (node_local, op_id) per graph
      logp:    Tensor [B] log-prob of chosen action
      entropy: Tensor [B] entropy of the per-graph candidate distribution
      aux: dict with cand tensors for debugging
    """
    device = device or next(net.parameters()).device
    pyg_batch = pyg_batch.to(device)
    B = pyg_batch.num_graphs

    # Encode nodes
    h, batch_idx, g_emb = net.encode_nodes(pyg_batch)   # h: [sumN,H], batch_idx:[sumN]
    values = net.value(g_emb)                            # [B]

    # Pack candidates
    cand_nodes, cand_ops, seg_ids, per_graph_M = pack_batch_candidates(node_masks_list, batch_idx, B)
    if cand_nodes.numel() == 0:
        # No candidates anywhere: define a convention (e.g., dummy action and zero logp/entropy)
        actions = torch.zeros((B,2), dtype=torch.long, device=device)
        logp = torch.zeros(B, device=device)
        entropy = torch.zeros(B, device=device)
        return actions, logp, entropy, {"per_graph_M": per_graph_M, "values": values}

    # Score only valid candidates
    scores = net.score_candidates(h, cand_nodes, cand_ops)  # [M_total]
    logp_all = segment_log_softmax(scores, seg_ids, B)      # [M_total]
    entropy = segment_entropy_from_logp(logp_all, seg_ids, B)  # [B]

    # Build a per-graph categorical
    # Easiest: split by segment; if B is large use an index-based sampling to avoid python loops
    actions = torch.zeros((B,2), dtype=torch.long, device=device)
    logp_chosen = torch.zeros(B, device=device)

    # indices per graph in the flat list
    # Compute start/end by counting per segment
    # Gather flat indices for each graph
    for g in range(B):
        M_g = int(per_graph_M[g].item())
        if M_g == 0:
            continue
        idx_g = torch.nonzero(seg_ids == g, as_tuple=False).squeeze(1)  # [M_g]
        logits_g = scores[idx_g]
        logp_g = logp_all[idx_g]
        if greedy:
            m = torch.argmax(logits_g)
        else:
            dist = Categorical(logits=logits_g)
            m = dist.sample()
        flat_idx = idx_g[m]  # index into cand_* tensors

        # Decode chosen (node_local, op)
        # We need local node index; we can compute it as:
        # local = flat_node_idx - offset[g]; pass offsets from pack fn if you prefer
        # Quick way: stash local node index at rollout time; here we recompute using pyg_batch.ptr or offsets
        # For simplicity, precompute offsets like this:
        # (We can pass offsets from a helper; omitted here for brevity.)
        # Instead, store the flat node index and later map to local if needed.
        actions[g, 0] = -1  # fill later if you require local indices; see training path below
        actions[g, 1] = cand_ops[flat_idx]
        logp_chosen[g] = logp_g[m]

    return actions, logp_chosen, entropy, {"cand_nodes": cand_nodes, "cand_ops": cand_ops,
                                           "seg_ids": seg_ids, "per_graph_M": per_graph_M,
                                           "values": values}