# eval.py
# Deterministic/stochastic evaluation for a trained PPO-GNN+GRU agent on ZXCalculus

import argparse
import os
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

# Local imports
from zxreinforce.Resetters import Resetter_GraphBank
from zxreinforce.zx_env_circuit import ZXCalculus
from zxreinforce.subproc_vec_env import SubprocVecEnv
from zxreinforce.gnn import PolicyValueNet
from zxreinforce.own_constants import N_NODE_ACTIONS


# ---------- utils copied from training (no gradients) ----------

@torch.no_grad()
def _pad_1d(list_tensors: List[torch.Tensor], pad_value: float = 0.0, dtype=None, device=None) -> torch.Tensor:
    if len(list_tensors) == 0:
        return torch.zeros(0, 1, dtype=dtype, device=device)
    max_len = max(int(t.numel()) for t in list_tensors)
    out = []
    for t in list_tensors:
        if dtype is not None: t = t.to(dtype)
        if device is not None: t = t.to(device)
        if t.numel() < max_len:
            pad = torch.full((max_len - t.numel(),), pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
        out.append(t)
    return torch.stack(out, dim=0)

@torch.no_grad()
def _with_global_node(data: Data, mask: torch.Tensor) -> tuple[Data, torch.Tensor]:
    # data.x: [N, F], data.edge_index: [2, E]; mask: [N * N_NODE_ACTIONS] (bool)
    x = data.x
    device = x.device
    N, F = x.size(0), x.size(1)
    gfeat = torch.zeros(1, F, dtype=x.dtype, device=device)
    x_ext = torch.cat([x, gfeat], dim=0)  # [N+1, F]

    # edges
    if N > 0:
        nodes = torch.arange(N, dtype=torch.long, device=device)
        gidx  = torch.full((N,), N, dtype=torch.long, device=device)
        edge_g2n = torch.stack([gidx, nodes], dim=0)
        edge_n2g = torch.stack([nodes, gidx], dim=0)
        ei = data.edge_index if data.edge_index.numel() > 0 else torch.empty(2, 0, dtype=torch.long, device=device)
        edge_index_ext = torch.cat([ei, edge_g2n, edge_n2g], dim=1)
    else:
        edge_index_ext = torch.empty(2, 0, dtype=torch.long, device=device)

    # edge_attr
    edge_attr_ext = None
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        E, D = data.edge_attr.size()
        zeros = torch.zeros((2 * N, D), dtype=data.edge_attr.dtype, device=device)
        edge_attr_ext = torch.cat([data.edge_attr, zeros], dim=0)

    # mark global
    is_global = torch.zeros(N + 1, dtype=torch.bool, device=device)
    is_global[N] = True

    # mask
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    mask_ext = torch.cat([mask, torch.zeros(N_NODE_ACTIONS, dtype=torch.bool, device=mask.device)], dim=0)

    data_ext = Data(x=x_ext, edge_index=edge_index_ext, edge_attr=edge_attr_ext)
    data_ext.is_global = is_global
    return data_ext, mask_ext

@torch.no_grad()
def _obs_to_batch(obs_batch: List[dict], device: torch.device) -> Tuple[Batch, torch.Tensor]:
    datas, masks = [], []
    for o in obs_batch:
        d = o["data"]
        m = o["mask"]
        if isinstance(m, np.ndarray):
            m = torch.from_numpy(m)
        d, m = _with_global_node(d, m)
        datas.append(d)
        masks.append(m.to(torch.bool))

    batch = Batch.from_data_list(datas)
    mask = _pad_1d(masks, pad_value=0.0, dtype=torch.bool, device=device)
    return batch.to(device), mask

@torch.no_grad()
def masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits, mask: [B, Amax]
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    logits_masked = logits.masked_fill(~mask, neg_inf)
    return torch.argmax(logits_masked, dim=1)

@torch.no_grad()
def masked_sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    logits_t = logits / max(1e-6, float(temperature))
    logits_t = logits_t.masked_fill(~mask, neg_inf)
    probs = torch.softmax(logits_t, dim=1)
    # if any row becomes NaN/zero, fallback to uniform over valid
    bad = ~torch.isfinite(probs).all(dim=1)
    if bad.any():
        u = torch.rand_like(probs)
        probs = torch.where(mask, u, torch.zeros_like(u))
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return torch.multinomial(probs, num_samples=1).squeeze(1)

def build_env_fn(args, shuffle: bool) :
    def make_env():
        exp_name = f"GraphBank_nq[{args.num_qubits_min}-{args.num_qubits_max}]_gates[{args.min_gates}-{args.max_gates}]_T{args.p_t}_H{args.p_h}_S{getattr(args,'p_s',0.2)}_CX{args.p_cnot}_X{args.p_not}_length{args.data_length}"
        bank_path = os.path.join(args.data_dir, exp_name)
        resetter = Resetter_GraphBank(bank_path=bank_path, seed=args.seed if args.seed >= 0 else None, shuffle=shuffle)
        env = ZXCalculus(
            max_steps=args.env_max_steps,
            resetter=resetter,
            count_down_from=args.count_down_from,
            step_penalty=args.step_penalty,
            length_penalty=args.length_penalty,
            extra_state_info=False,
            adapted_reward=args.adapted_reward,
        )
        return env
    return make_env

def load_net_from_ckpt(ckpt_path: str, device: torch.device) -> tuple[PolicyValueNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    hps = ckpt.get("hyper_parameters", {})
    # Pull shapes from checkpoint (Lightning saved via save_hyperparameters)
    node_feat_dim = int(hps.get("node_feat_dim", 5 + 10 + 4))
    emb_dim = int(hps.get("emb_dim", 256))
    hid_dim = int(hps.get("hid_dim", 128))
    rec_dim = int(hps.get("rec_dim", 256))

    net = PolicyValueNet(gnn_in_dim=node_feat_dim, emb=emb_dim, hid=hid_dim, rec_dim=rec_dim).to(device)
    # Filter state_dict to "net." prefix and load into our net
    sd = ckpt["state_dict"]
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(net_sd, strict=False)
    if missing:
        print("Warning: missing keys:", missing)
    if unexpected:
        print("Warning: unexpected keys:", unexpected)
    net.eval()
    return net, {"node_feat_dim": node_feat_dim, "emb_dim": emb_dim, "hid_dim": hid_dim, "rec_dim": rec_dim}

@torch.no_grad()
def evaluate(args):
    torch.manual_seed(max(0, args.seed))
    np.random.seed(max(0, args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu")
    net, shapes = load_net_from_ckpt(args.ckpt_path, device)

    # Build eval environments (no shuffle => deterministic pass through bank)
    env_fn = build_env_fn(args, shuffle=False)
    vec = SubprocVecEnv([env_fn for _ in range(args.num_envs_eval)], start_method="spawn")

    # Reset
    obs = [{"data": d, "mask": m} for (d, m) in vec.reset()]
    B = args.num_envs_eval
    h = torch.zeros(1, B, shapes["rec_dim"], device=device)

    # Episode trackers
    ep_ret = np.zeros(B, dtype=np.float32)
    ep_len = np.zeros(B, dtype=np.int32)
    done_count = 0

    # Aggregates
    returns, lengths, successes, truncs = [], [], [], []
    T_reductions, CNOT_reductions = [], []

    def record_episode(i_env: int, info: dict):
        nonlocal done_count
        returns.append(float(ep_ret[i_env]))
        lengths.append(int(ep_len[i_env]))
        successes.append(int(bool(info.get("terminated", False))))
        truncs.append(int(bool(info.get("TimeLimit.truncated", False))))
        T_reductions.append(int(info.get("T_reduced", 0)))
        CNOT_reductions.append(int(info.get("CNOT_reduced", 0)))
        ep_ret[i_env] = 0.0
        ep_len[i_env] = 0
        done_count += 1

    # Roll until we collect desired number of finished episodes
    while done_count < args.eval_episodes:
        # forward
        batch, mask = _obs_to_batch(obs, device)
        logits, values, h_next = net(batch, h)

        if args.greedy:
            actions = masked_argmax(logits, mask)
        else:
            actions = masked_sample(logits, mask, temperature=args.temperature)

        # step envs
        next_obs, rewards, dones, infos = vec.step(actions.detach().cpu().numpy())

        # stats
        for i in range(B):
            ep_ret[i] += float(rewards[i])
            ep_len[i] += 1
            if bool(dones[i]):
                record_episode(i, infos[i])

        # manage hidden state
        h = h_next
        if np.any(dones):
            dmask = torch.from_numpy(dones.astype(np.bool_)).to(device)
            h[:, dmask, :] = 0

        # move to next obs
        obs = next_obs

        # stop early if we have enough
        if done_count >= args.eval_episodes:
            break

    vec.close()

    # Summaries
    def _meanstd(x):
        if len(x) == 0: return (0.0, 0.0)
        arr = np.asarray(x, dtype=np.float32)
        return float(arr.mean()), float(arr.std(ddof=0))

    ret_m, ret_s = _meanstd(returns)
    len_m, len_s = _meanstd(lengths)
    Tred_m, Tred_s = _meanstd(T_reductions)
    CNOT_m, CNOT_s = _meanstd(CNOT_reductions)
    succ_rate = float(np.mean(successes)) if successes else 0.0
    trunc_rate = float(np.mean(truncs)) if truncs else 0.0

    summary = {
        "episodes": int(len(returns)),
        "success_rate": succ_rate,
        "timeout_rate": trunc_rate,
        "return_mean": ret_m, "return_std": ret_s,
        "length_mean": len_m, "length_std": len_s,
        "T_reduced_mean": Tred_m, "T_reduced_std": Tred_s,
        "CNOT_reduced_mean": CNOT_m, "CNOT_reduced_std": CNOT_s,
        "greedy": bool(args.greedy),
        "temperature": float(args.temperature),
        "node_feat_dim_ckpt": shapes["node_feat_dim"],
        "rec_dim_ckpt": shapes["rec_dim"],
    }
    print("==== Evaluation summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO-GNN+GRU agent on ZXCalculus")
    # checkpoint
    parser.add_argument("--ckpt_path", type=str, required=True)

    # eval control
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--num_envs_eval", type=int, default=8)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--out_json", type=str, default="")

    # env/bank params (used to locate the bank file)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_length", type=int, default=1000)
    parser.add_argument("--num_qubits_min", type=int, default=2)
    parser.add_argument("--num_qubits_max", type=int, default=6)
    parser.add_argument("--min_gates", type=int, default=5)
    parser.add_argument("--max_gates", type=int, default=30)
    parser.add_argument("--p_t", type=float, default=0.2)
    parser.add_argument("--p_h", type=float, default=0.2)
    parser.add_argument("--p_s", type=float, default=0.2)
    parser.add_argument("--p_cnot", type=float, default=0.2)
    parser.add_argument("--p_not", type=float, default=0.2)

    # env dynamics
    parser.add_argument("--env_max_steps", type=int, default=100)
    parser.add_argument("--step_penalty", type=float, default=0.02)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--adapted_reward", action="store_true", default=True)
    parser.add_argument("--count_down_from", type=int, default=20)

    args = parser.parse_args()
    evaluate(args)