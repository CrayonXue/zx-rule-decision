import os
import torch
import numpy as np
from torch_geometric.data import Batch
import argparse

# Your project imports
from zxreinforce.gnn import PolicyValueNet
from zxreinforce.zx_env_circuit import ZXCalculus
from zxreinforce.Resetters import Resetter_GraphBank
from zxreinforce.own_constants import N_NODE_ACTIONS  # action decoding

# ----- Helpers -----

NODE_ACTION_NAMES = ["select_node", "unfuse_rule", "color_change_rule", "split_hadamard", "pi_rule"]

def load_net_from_ckpt(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Pull model dims from Lightning hyperparams if present
    hps = ckpt.get("hyper_parameters", {})
    node_feat_dim = hps.get("node_feat_dim", 5 + 10 + 1 + 1)
    emb_dim       = hps.get("emb_dim", 256)
    hid_dim       = hps.get("hid_dim", 128)

    net = PolicyValueNet(gnn_in_dim=node_feat_dim, emb=emb_dim, hid=hid_dim).to(device)

    # Figure out the actual state_dict:
    if "state_dict" in ckpt:      # typical PL checkpoint
        raw = ckpt["state_dict"]
        # keep only keys that belong to the inner net, strip 'net.' prefix
        sd = {}
        for k, v in raw.items():
            if k.startswith("net."):
                sd[k[len("net."):]] = v
            elif k in net.state_dict():
                sd[k] = v
    else:
        # If someone saved torch.save(net.state_dict(), ...)
        sd = ckpt

    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("Warning - missing keys:", missing)
        print("Warning - unexpected keys:", unexpected)

    net.eval()
    return net, hps

@torch.no_grad()
def policy_logits_on_single_graph(net, data, mask, device):
    batch = Batch.from_data_list([data]).to(device)   # B=1
    mask_t = mask.to(device).bool().unsqueeze(0)     # [1, A]
    logits, value = net(batch)                       # [1, A], [1]
    # apply mask the same way as in training
    masked_logits = logits.masked_fill(~mask_t, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)
    return masked_logits.squeeze(0), probs.squeeze(0), value.squeeze(0)

def decode_action(a_idx: int):
    node_idx = a_idx // N_NODE_ACTIONS
    a_local  = a_idx % N_NODE_ACTIONS
    a_name   = NODE_ACTION_NAMES[a_local] if a_local < len(NODE_ACTION_NAMES) else f"action_{a_local}"
    return node_idx, a_local, a_name

def print_topk(masked_logits, probs, k=10):
    A = masked_logits.shape[0]
    k = min(k, A)
    topk = torch.topk(masked_logits, k=k, dim=0)
    print(f"Top-{k} actions:")
    for rank in range(k):
        a = int(topk.indices[rank].item())
        node_idx, a_local, a_name = decode_action(a)
        print(f"  #{rank+1}: action_id={a:4d}  node={node_idx:3d}  op={a_name:15s}  "
              f"logit={masked_logits[a].item():8.3f}  p={probs[a].item():.4f}")

def build_one_env_graph(seed=123, data_dir='./data', num_qubits_min=3, num_qubits_max=5,
                        min_gates=8, max_gates=20, data_length=1000, p_t=0.2, p_h=0.2,
                        env_max_steps=100, step_penalty=0.01, length_penalty=0.01, adapted_reward=True, count_down_from=20):
    exp_name = f"GraphBank_nq[{num_qubits_min}-{num_qubits_max}]_gates[{min_gates}-{max_gates}]_length{data_length}.pkl"
    bank_path = os.path.join(data_dir, exp_name)
    resetter = Resetter_GraphBank(
        bank_path = bank_path,
        seed=seed,    # deterministic example graph
    )
    env = ZXCalculus(
        max_steps=env_max_steps,
        resetter=resetter,
        count_down_from=count_down_from,
        step_penalty=step_penalty,
        length_penalty=length_penalty,
        extra_state_info=False,
        adapted_reward=adapted_reward,
    )
    data, mask = env.reset()
    return env, data, mask

# Optional: roll one greedy episode on this single env
@torch.no_grad()
def greedy_rollout(env, net, device="cpu", max_steps=50, verbose=True):
    data, mask = env.reset()
    total_r, t = 0.0, 0
    while t < max_steps:
        logits, probs, _ = policy_logits_on_single_graph(net, data, mask, device)
        a = int(torch.argmax(logits).item())
        node_idx, a_local, a_name = decode_action(a)
        if verbose:
            print(f"[t={t}] choose a={a} -> node={node_idx}, op={a_name}, p={probs[a].item():.4f}")
        data, mask, r, d = env.step(a)
        total_r += float(r)
        t += 1
        if d:
            if verbose:
                print(f"Done at t={t}, return={total_r:.3f}")
            break
    return total_r, t

@torch.no_grad()
def run_until_done(env, net, device="cpu", sample=False, topk_to_print=5):
    """
    Runs one episode with the loaded net on a single env until done==1.
    'done' can be either terminal success or the env's time limit.
    Set sample=True to sample from the masked policy instead of greedy argmax.
    """
    from torch_geometric.data import Batch

    data, mask = env.reset()
    t, ep_ret = 0, 0.0

    while True:
        if env.is_terminal():
            print(f"Episode finished after {t} steps, return={ep_ret:.3f}")
            break
        # Forward pass on this one graph
        batch = Batch.from_data_list([data]).to(device)
        mask_t = mask.to(device).bool().unsqueeze(0)         # [1, A]
        logits, _ = net(batch)                               # [1, A]
        masked_logits = logits.masked_fill(~mask_t, -1e9).squeeze(0)
        probs = torch.softmax(masked_logits, dim=-1)

        # Pick action: greedy or sample
        if sample:
            a = int(torch.distributions.Categorical(probs=probs).sample().item())
        else:
            a = int(torch.argmax(masked_logits).item())

        # Optional: print a few top actions
        if topk_to_print > 0 and t == 0:
            k = min(topk_to_print, masked_logits.numel())
            topk = torch.topk(masked_logits, k)
            print(f"Top-{k} at t=0:")
            for rank in range(k):
                aid = int(topk.indices[rank])
                node_idx = aid // N_NODE_ACTIONS
                op_idx   = aid %  N_NODE_ACTIONS
                
                if op_idx < 2**5:
                    op_name = "unfuse_rule" 
                else:
                    op_name  = ["color_change_rule","split_hadamard","pi_rule"][op_idx%(2**5)]
                print(f"  #{rank+1}: action={aid:4d} node={node_idx:3d} op={op_name:15s} p={probs[aid].item():.4f}")


        # Decode for printing
        node_idx = a // N_NODE_ACTIONS
        op_idx   = a %  N_NODE_ACTIONS
        if op_idx < 2**5:
            op_name = "unfuse_rule" 
        else:
            op_name  = ["color_change_rule","split_hadamard","pi_rule"][op_idx%(2**5)]
        # Step the env
        data, mask, r, d = env.step(a)
        ep_ret += float(r); t += 1
        print(f"[t={t:3d}] action={a:4d} node={node_idx:3d} op={op_name:15s} reward={r:.3f} done={bool(d)}")

        if d:
            print(f"Episode finished after {t} steps, return={ep_ret:.3f}")
            break


def eval(args):
    ckpt_path = args.ckpt_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, hps = load_net_from_ckpt(ckpt_path, device=device)

    # Build one example graph
    env, data, mask = build_one_env_graph(
        seed=123,
        data_dir=args.data_dir,
        num_qubits_min=args.num_qubits_min,
        num_qubits_max=args.num_qubits_max,
        min_gates=args.min_gates,
        max_gates=args.max_gates,
        data_length=args.data_length,
        p_t=args.p_t,
        p_h=args.p_h,
        env_max_steps=args.env_max_steps,
        step_penalty=args.step_penalty,
        length_penalty=args.length_penalty,
        adapted_reward=args.adapted_reward,
        count_down_from=args.count_down_from,
    )

    # Single forward pass: get predictions for this one graph
    masked_logits, probs, value = policy_logits_on_single_graph(net, data, mask, device)
    print(f"Value head prediction: {value.item():.3f}")
    run_until_done(env, net, device=device, sample=True, topk_to_print=2)


    # # print_topk(masked_logits, probs, k=10)
    # # If you want to actually apply the top action once:
    # top_action = int(torch.argmax(masked_logits).item())
    # node_idx, a_local, a_name = decode_action(top_action)
    # print(f"\nApplying top action: id={top_action}, node={node_idx}, op={a_name}")
    # data2, mask2, r, d = env.step(top_action)
    # print(f"Step reward={r:.3f}, done={bool(d)}")

    # Or run a short greedy rollout:
    # greedy_rollout(env, net, device=device, max_steps=30, verbose=True)

# ----- Main demo -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-GNN evalution")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to a checkpoint to load.")

    # Environment (ZXCalculus)
    parser.add_argument("--env_max_steps", type=int, default=100)
    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--length_penalty", type=float, default=0.01)
    parser.add_argument("--adapted_reward", action="store_true", default=True)
    parser.add_argument("--count_down_from", type=int, default=20)

    # Resetter (random circuit generator)
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing graph bank pickle files.")
    parser.add_argument("--data_length", type=int, default=1000, help="Number of graphs in the graph bank pickle file.")
    parser.add_argument("--num_qubits_min", type=int, default=2)
    parser.add_argument("--num_qubits_max", type=int, default=6)
    parser.add_argument("--min_gates", type=int, default=5)
    parser.add_argument("--max_gates", type=int, default=30)
    parser.add_argument("--p_t", type=float, default=0.2)
    parser.add_argument("--p_h", type=float, default=0.2)

    args = parser.parse_args()
    for i in range(10):
        print(f"\n=== Eval run {i+1}/10 ===")
        eval(args)

