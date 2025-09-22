import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical
import numpy as np
from .own_constants import ( N_NODE_ACTIONS, N_EDGE_ACTIONS)

def masked_categorical(logits, mask, eps=1e-8):
    # mask: 0/1 shape [B, A]
    # Use -1e4 instead of -1e9 to avoid overflow in 16-bit precision
    logits_masked = logits.masked_fill(mask == 0, -1e4)
    return Categorical(logits=logits_masked)

class NodeEncoder(nn.Module):
    def __init__(self, in_dim=1+5+10+1+1, hid=128, out=128):  # Added 1 for node_ids
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out), nn.ReLU(),
        )
    def forward(self, node_ids,node_type, node_phase, node_sel, qubit_on):
        # Inputs: [B, N, ...]
        x = torch.cat([
            node_ids.float(),         # 1
            node_type.float(),         # 5
            node_phase.float(),        # 10
            node_sel.float(),          # 1
            qubit_on.float()           # 1
        ], dim=-1)
        return self.net(x)             # [B, N, D]

class EdgeEncoder(nn.Module):
    def __init__(self, node_dim=128, in_extra=1, hid=128, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*node_dim + in_extra, hid), nn.ReLU(),
            nn.Linear(hid, out), nn.ReLU(),
        )
    def forward(self, node_emb, edge_index, edge_sel):
        # node_emb: [B, N, D], edge_index: [B, E, 2] (positional indices, -1 padded), edge_sel: [B, E, 1]
        B, E, _ = edge_index.shape
        N = node_emb.size(1)

        # Clamp indices to valid range [0, N-1] to prevent CUDA assertion errors
        idx_u = edge_index[..., 0].clamp(min=0, max=N-1)
        idx_v = edge_index[..., 1].clamp(min=0, max=N-1)

        # Gather endpoints; for padded (-1) edges, this gathers node 0; we will mask later
        u = torch.gather(node_emb, 1, idx_u.unsqueeze(-1).expand(-1, -1, node_emb.size(-1)))
        v = torch.gather(node_emb, 1, idx_v.unsqueeze(-1).expand(-1, -1, node_emb.size(-1)))
        x = torch.cat([u, v, edge_sel.float()], dim=-1)
        return self.net(x)  # [B, E, D]

class PolicyNet(nn.Module):
    def __init__(self, max_nodes, max_edges, node_dim=128, edge_dim=128, hid=256):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_head = nn.Sequential(nn.Linear(node_dim, hid), nn.ReLU(), nn.Linear(hid, N_NODE_ACTIONS))
        self.edge_head = nn.Sequential(nn.Linear(edge_dim, hid), nn.ReLU(), nn.Linear(hid, N_EDGE_ACTIONS))
        self.stop_head = nn.Sequential(nn.Linear(node_dim, hid), nn.ReLU(), nn.Linear(hid, 1))  # use pooled node feat

    def forward(self, node_emb, edge_emb, node_mask, edge_mask):
        # node_emb: [B, N, D], edge_emb: [B, E, D]
        # node_mask: [B, N] 1=valid; edge_mask: [B, E] 1=valid
        B, N, Dn = node_emb.shape
        E = edge_emb.size(1)

        # Per-node action logits
        node_logits = self.node_head(node_emb)     # [B, N, A_n]
        node_logits = node_logits.view(B, N*N_NODE_ACTIONS)

        # Per-edge action logits
        edge_logits = self.edge_head(edge_emb)     # [B, E, A_e]
        edge_logits = edge_logits.view(B, E*N_EDGE_ACTIONS)

        # STOP logit from pooled node embedding
        pooled = (node_emb * node_mask.unsqueeze(-1)).sum(dim=1) / (node_mask.sum(dim=1, keepdim=True) + 1e-6)
        stop_logit = self.stop_head(pooled)        # [B, 1]

        logits = torch.cat([node_logits, edge_logits, stop_logit], dim=1)  # [B, A]
        return logits

class ValueNet(nn.Module):
    def __init__(self, node_dim=128, edge_dim=128, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + edge_dim + 3, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, node_emb, edge_emb, node_mask, edge_mask, context):
        # Global pooling
        n_pool = (node_emb * node_mask.unsqueeze(-1)).sum(dim=1) / (node_mask.sum(dim=1, keepdim=True) + 1e-6)
        e_pool = (edge_emb * edge_mask.unsqueeze(-1)).sum(dim=1) / (edge_mask.sum(dim=1, keepdim=True) + 1e-6)
        x = torch.cat([n_pool, e_pool, context], dim=-1)
        return self.net(x).squeeze(-1)

class PPOLightning(pl.LightningModule):
    def __init__(self, env_fn, max_nodes=256, max_edges=512, rollout_steps=2048, batch_size=256,
                 epochs=4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, lr=3e-4, num_envs=8, device="cpu"):
        super().__init__()
        self.save_hyperparameters(ignore=["env_fn"])
        self.envs = [env_fn() for _ in range(num_envs)]

        self.node_enc = NodeEncoder()
        self.edge_enc = EdgeEncoder()
        self.policy = PolicyNet(max_nodes, max_edges)
        self.value = ValueNet()

        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.lr = lr
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.num_envs = num_envs

        self.automatic_optimization = False

        # Buffers
        self.reset_envs()

    def reset_envs(self):
        self.obs = [env.reset() for env in self.envs]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def _obs_to_tensors(self, obs_batch):
        # obs_batch is list of dicts length B
        def stack(key):
            return torch.from_numpy(np.stack([o[key] for o in obs_batch], axis=0)).to(self.device)
        node_ids = stack("node_ids")
        node_type = stack("node_type")
        node_phase = stack("node_phase")
        node_sel = stack("node_selected")
        qubit_on = stack("qubit_on")
        edge_pairs = stack("edge_pairs")
        edge_sel = stack("edge_selected")
        context = stack("context").float()
        action_mask = stack("action_mask").bool()

        # Masks: valid nodes (if any of type row non-zero)
        node_mask = (node_type.abs().sum(dim=-1) > 0).float()
        edge_mask = (edge_pairs[..., 0] >= 0).float()

        node_emb = self.node_enc(node_ids, node_type, node_phase, node_sel, qubit_on)
        edge_emb = self.edge_enc(node_emb, edge_pairs, edge_sel)

        return node_emb, edge_emb, node_mask, edge_mask, context, action_mask

    def _select_action(self, obs_batch):
        node_emb, edge_emb, node_mask, edge_mask, context, action_mask = self._obs_to_tensors(obs_batch)
        logits = self.policy(node_emb, edge_emb, node_mask, edge_mask)
        dist = masked_categorical(logits, action_mask)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value(node_emb, edge_emb, node_mask, edge_mask, context)
        return action, logprob, value

    def _step_envs(self, actions):
        next_obs, rewards, dones = [], [], []
        for i, env in enumerate(self.envs):
            obs, r, d, _ = env.step(int(actions[i].item()))
            if d:
                obs = env.reset()
            next_obs.append(obs)
            rewards.append(r)
            dones.append(d)
        return next_obs, np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool_)

    def collect_rollout(self):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        for t in range(self.rollout_steps):
            action, logp, value = self._select_action(self.obs)
            obs_buf.append(self.obs)
            act_buf.append(action.cpu())
            logp_buf.append(logp.cpu())
            val_buf.append(value.cpu())

            self.obs, rewards, dones = self._step_envs(action)
            rew_buf.append(torch.from_numpy(rewards))
            done_buf.append(torch.from_numpy(dones.astype(np.float32)))

        # Last value for bootstrap
        with torch.no_grad():
            _, _, last_value = self._select_action(self.obs)

        # Stack buffers: [T, B, ...]
        act = torch.stack(act_buf)                            # [T, B]
        logp = torch.stack(logp_buf)                          # [T, B]
        val = torch.stack(val_buf).squeeze(-1)                # [T, B]
        rew = torch.stack(rew_buf)                            # [T, B]
        done = torch.stack(done_buf)                          # [T, B]
        return obs_buf, act, logp, val, rew, done, last_value.squeeze(-1)

    def compute_gae(self, rewards, values, dones, last_value):
        # rewards/values/dones: [T, B], last_value: [B]
        # Ensure all tensors are on the same device as the model
        device = self.device
        rewards = rewards.to(device)
        values = values.to(device)
        dones = dones.to(device)
        last_value = last_value.to(device)
        
        T, B = rewards.shape
        adv = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            nextnonterminal = 1.0 - (dones[t] > 0).float()
            nextvalues = last_value if t == T-1 else values[t+1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # 1) Rollout
        obs_traj, act, logp_old, val, rew, done, last_value = self.collect_rollout()

        # 2) Compute advantages and returns
        # Detach tensors to prevent gradient flow from rollout to PPO updates
        adv, ret = self.compute_gae(rew.detach(), val.detach(), done.detach(), last_value.detach())
        # Flatten [T, B] -> [TB]
        T, B = act.shape
        total = T * B
        act = act.reshape(total)
        logp_old = logp_old.reshape(total).detach()  # Detach old log probs
        adv = (adv.reshape(total) - adv.mean()) / (adv.std() + 1e-8)
        ret = ret.reshape(total)

        # Prepare per-time-step obs list flattened
        flat_obs = []
        for t in range(T):
            for b in range(B):
                flat_obs.append(obs_traj[t][b])

        # 3) Optimize policy for K epochs with minibatches
        idx = torch.randperm(total)
        mb = self.batch_size
        for _ in range(self.epochs):
            for start in range(0, total, mb):
                sl = idx[start:start+mb]
                obs_mb = [flat_obs[i] for i in sl]
                actions_mb = act[sl].to(self.device)
                logp_old_mb = logp_old[sl].to(self.device)
                adv_mb = adv[sl].to(self.device)
                ret_mb = ret[sl].to(self.device)

                node_emb, edge_emb, node_mask, edge_mask, context, action_mask = self._obs_to_tensors(obs_mb)
                logits = self.policy(node_emb, edge_emb, node_mask, edge_mask)
                dist = masked_categorical(logits, action_mask)
                logp_new = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = (logp_new - logp_old_mb).exp()
                pg_loss = -torch.min(ratio * adv_mb, torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb).mean()

                values = self.value(node_emb, edge_emb, node_mask, edge_mask, context)
                v_loss = F.mse_loss(values, ret_mb)

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                opt.step()

        # Logging
        self.log_dict({
            "loss/policy": pg_loss.detach(),
            "loss/value": v_loss.detach(),
            "loss/entropy": entropy.detach(),
            "stats/return_mean": ret.mean().detach(),
        }, prog_bar=True, on_step=True, on_epoch=False)