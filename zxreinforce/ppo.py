# ppo_gnn.py
import numpy as np
from typing import List, Tuple, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from .gnn import PolicyValueNet  # uses your GraphEncoder + Policy/Value heads
from .own_constants import N_NODE_ACTIONS, N_EDGE_ACTIONS

def masked_categorical(logits: torch.Tensor, mask: torch.Tensor, neg_large: float = -1e4) -> Categorical:
    # logits, mask: [B, A]
    logits = logits.masked_fill(mask == 0, neg_large)
    return Categorical(logits=logits)


def _pad_1d(list_tensors: List[torch.Tensor], pad_value: float = 0.0, dtype=None, device=None) -> torch.Tensor:
    # Pads a list of 1D tensors to [B, max_len]
    if len(list_tensors) == 0:
        return torch.zeros(0, 1, dtype=dtype, device=device)
    max_len = max(int(t.numel()) for t in list_tensors)
    out = []
    for t in list_tensors:
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        if t.numel() < max_len:
            pad = torch.full((max_len - t.numel(),), pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
        out.append(t)
    return torch.stack(out, dim=0)


def _normalize_adv(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


class PPOLightningGNN(pl.LightningModule):
    def __init__(
        self,
        env_fn,
        node_feat_dim: int = 5 + 10 + 1 + 1,   # type[5] + phase[10] + selected[1] + qubit_on[1]; add degree if you include it
        emb_dim: int = 128,
        hid_dim: int = 128,
        rollout_steps: int = 512,
        batch_size: int = 256,
        epochs: int = 4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        lr: float = 3e-4,
        num_envs: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env_fn"])
        self.envs = [env_fn() for _ in range(num_envs)]

        self.net = PolicyValueNet(gnn_in_dim=node_feat_dim, emb=emb_dim, hid=hid_dim)

        # PPO hparams
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.lr = lr
        self.num_envs = num_envs

        self.automatic_optimization = False

        # episodic stats
        self._ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int32)
        self._completed_returns: List[float] = []
        self._completed_lengths: List[int] = []

        self.register_buffer("_total_env_steps", torch.tensor(0, dtype=torch.long))

        self.reset_envs()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reset_envs(self):
        # Each env.reset() should return Data
        self.obs = []
        for env in self.envs:
            ret = env.reset()
            data, mask = ret
            self.obs.append({"data": data, "mask": mask})

    @torch.no_grad()
    def _obs_to_batch(self, obs_batch: List[dict]) -> Tuple[Batch, torch.Tensor]:
        # Build a PyG Batch from Data objects and pad masks to same width
        datas = [o["data"] for o in obs_batch]  # list of Data
        batch = Batch.from_data_list(datas)

        # masks are variable-length per graph, pad to [B, A_max]
        masks = []
        for o in obs_batch:
            m = o["mask"]
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            if m is None:
                raise RuntimeError("Missing action mask in observation.")
            masks.append(m.to(torch.bool))

        mask = _pad_1d(masks, pad_value=0.0, dtype=torch.bool, device=self.device)  # [B, A_max]
        return batch.to(self.device), mask

    @torch.no_grad()
    def _select_action(self, obs_batch: List[dict]):
        batch, mask = self._obs_to_batch(obs_batch)
        logits, values = self.net(batch)  # logits: [B, A_max], values: [B]
        assert logits.size(0) == mask.size(0) and logits.size(1) == mask.size(1)
        dist = masked_categorical(logits, mask)
        action = dist.sample()          # [B]
        logprob = dist.log_prob(action) # [B]
        return action, logprob, values

    def _step_envs(self, actions: torch.Tensor):
        next_obs, rewards, dones = [], [], []
        for i, env in enumerate(self.envs):
            a = int(actions[i].item())
            ret = env.step(a)
            # step may return (Data, mask, r, d) or (obs, r, d, info)
            if isinstance(ret, tuple) and len(ret) >= 4:
                if isinstance(ret[0], Data):
                    data, mask, r, d = ret[0], ret[1], ret[2], ret[3]
            else:
                raise RuntimeError("Unexpected env.step return format")

            if d:
                ret2 = env.reset()
                data, mask = ret2[0], ret2[1]


            next_obs.append({"data": data, "mask": mask})
            rewards.append(float(r))
            dones.append(bool(d))

            # episodic stats
            self._ep_ret[i] += float(r)
            self._ep_len[i] += 1
            if d:
                self._completed_returns.append(float(self._ep_ret[i]))
                self._completed_lengths.append(int(self._ep_len[i]))
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0

        return next_obs, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.bool_)

    def collect_rollout(self):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        for _ in range(self.rollout_steps):
            action, logp, value = self._select_action(self.obs)
            obs_buf.append(self.obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
            # act_buf.append(action.cpu())
            # logp_buf.append(logp.cpu())
            # val_buf.append(value.cpu())            
            self.obs, rewards, dones = self._step_envs(action)
            rew_buf.append(torch.from_numpy(rewards))
            done_buf.append(torch.from_numpy(dones.astype(np.float32)))

        with torch.no_grad():
            _, _, last_value = self._select_action(self.obs)  # [B]

        act = torch.stack(act_buf)                 # [T,B]
        logp = torch.stack(logp_buf)               # [T,B]
        val = torch.stack(val_buf).squeeze(-1)     # [T,B]
        rew = torch.stack(rew_buf)                 # [T,B]
        done = torch.stack(done_buf)               # [T,B]
        return obs_buf, act, logp, val, rew, done, last_value.squeeze(-1)

    @torch.no_grad()
    def compute_gae(self, rewards, values, dones, last_value):
        # rewards/values/dones: [T,B], last_value: [B]
        device = self.device
        rewards = rewards.to(device)
        values = values.to(device)
        dones = dones.to(device)
        last_value = last_value.to(device)

        T, B = rewards.shape
        adv = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - (dones[t] > 0).float()
            next_values = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    def training_step(self,batch, batch_idx):
        opt = self.optimizers()

        # 1) Rollout
        # torch.cuda.synchronize()
        # start_time = time.time()
        obs_traj, act, logp_old, val, rew, done, last_value = self.collect_rollout()
        # torch.cuda.synchronize()
        # end_time = time.time()
        # self.log("time/rollout", end_time - start_time, prog_bar=False, on_step=True, on_epoch=False)

        self._total_env_steps += torch.tensor(self.rollout_steps * self.num_envs, device=self.device)
        self.log("counters/total_env_steps", self._total_env_steps, on_step=True, on_epoch=False, prog_bar=True)
 


        # 2) GAE
        adv, ret = self.compute_gae(rew.detach(), val.detach(), done.detach(), last_value.detach())
        T, B = act.shape
        total = T * B
        act = act.reshape(total).to(self.device)
        logp_old = logp_old.reshape(total).to(self.device)
        adv = _normalize_adv(adv.reshape(total)).to(self.device)
        ret = ret.reshape(total).to(self.device)
        
        # act = act.reshape(total).cpu()
        # logp_old = logp_old.reshape(total).cpu()
        # adv = _normalize_adv(adv.reshape(total)).cpu()
        # ret = ret.reshape(total).cpu()


        with torch.no_grad():
                # rollout stats
                self.log("rollout/T", torch.tensor(self.rollout_steps, device=self.device), on_step=True, on_epoch=False)
                self.log("rollout/B", torch.tensor(self.num_envs, device=self.device), on_step=True, on_epoch=False)
                self.log("rollout/samples", torch.tensor(self.rollout_steps*self.num_envs, device=self.device), on_step=True, on_epoch=False)
                self.log("rollout/reward_mean", rew.float().mean(), on_step=True, on_epoch=False, prog_bar=True)
                self.log("rollout/reward_std", rew.float().std(unbiased=False), on_step=True, on_epoch=False)
                self.log("rollout/done_frac", done.float().mean(), on_step=True, on_epoch=False)

                # value explained variance
                v_flat = val.detach().reshape(-1).to(self.device)
                r_flat = ret.detach()
                var_y = torch.var(r_flat, unbiased=False)
                ev = 1.0 - torch.var(r_flat - v_flat, unbiased=False) / (var_y + 1e-8)
                self.log("value/explained_variance", ev, on_step=True, on_epoch=False)


        # 3) Flatten obs (list of lists) into length TB
        flat_obs: List[dict] = []
        for t in range(T):
            flat_obs.extend(obs_traj[t])

        # 4) PPO epochs/minibatches

        idx = torch.randperm(total, device=self.device)
        # idx = torch.randperm(total)
        mb = self.batch_size

        pg_losses, v_losses, entropies, kls, clipfracs = [], [], [], [], []

        for _ in range(self.epochs):
            for start in range(0, total, mb):
                sl = idx[start:start + mb]
                obs_mb = [flat_obs[i] for i in sl.tolist()]

                # slice
                actions_mb = act[sl].to(self.device)
                logp_old_mb = logp_old[sl].to(self.device)
                adv_mb = adv[sl].to(self.device)
                ret_mb = ret[sl].to(self.device)

                batch_mb, mask_mb = self._obs_to_batch(obs_mb)
                logits, values = self.net(batch_mb)
                dist = masked_categorical(logits, mask_mb)
                logp_new = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = (logp_new - logp_old_mb).exp()
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                pg_loss = -torch.min(unclipped, clipped).mean()

                v_loss = F.mse_loss(values, ret_mb)

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                opt.step()

                with torch.no_grad():
                    approx_kl = (logp_old_mb - logp_new).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
                    pg_losses.append(pg_loss.detach())
                    v_losses.append(v_loss.detach())
                    entropies.append(entropy.detach())
                    kls.append(approx_kl.detach())
                    clipfracs.append(clipfrac.detach())
        def _m(lst): return torch.stack(lst).mean() if len(lst) else torch.tensor(0.0, device=self.device)

        metrics = {
            "loss/policy": _m(pg_losses),
            "loss/value": _m(v_losses),
            "loss/entropy": _m(entropies),
            "ppo/approx_kl": _m(kls),
            "ppo/clipfrac": _m(clipfracs),
            "stats/return_mean": ret.mean().detach(),
            "stats/return_std": ret.std(unbiased=False).detach(),
        }
        if len(self._completed_returns) > 0:
            metrics.update({
                "stats/ep_return_mean": torch.tensor(float(np.mean(self._completed_returns)), device=self.device),
                "stats/ep_len_mean": torch.tensor(float(np.mean(self._completed_lengths)), device=self.device),
                "stats/episodes": torch.tensor(len(self._completed_returns), device=self.device),
            })
            self._completed_returns.clear()
            self._completed_lengths.clear()

        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False)

    def on_train_start(self):
        self.reset_envs()