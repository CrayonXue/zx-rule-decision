# ppo.py
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
from .own_constants import N_NODE_ACTIONS
from .subproc_vec_env import SubprocVecEnv

def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    # logits, mask: [B, A] 
    invalid = (mask.sum(dim=1) == 0)
    if invalid.any():
        # ensure at least one valid action
        mask = mask.clone()
        mask[invalid, 0] = True
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
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
        node_feat_dim: int = 5 + 10 + 1,   # type[5] + phase[10] + qubit_on[1]; add degree if you include it
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
        use_global_node: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env_fn"])
        # self.envs = [env_fn() for _ in range(num_envs)]
        self.vec = SubprocVecEnv([env_fn for _ in range(num_envs)], start_method="spawn")

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

        self.total_updates = None  # will set in on_train_start
        self._ent_coef_start = float(self.ent_coef)
        self._ent_coef_end = 0.25 * self._ent_coef_start  # tweak as desired

        self.automatic_optimization = False
        self.use_global_node = use_global_node

        # episodic stats
        self._ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int32)
        self._completed_returns: List[float] = []
        self._completed_lengths: List[int] = []

        self.register_buffer("_total_env_steps", torch.tensor(0.0, dtype=torch.float32))

        self.reset_envs()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # step-wise anneal
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: self.lr_anneal_factor())
        return [opt], [{"scheduler": sched, "interval": "step"}]

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reset_envs(self):
        results = self.vec.reset()
        self.obs = [{"data": d, "mask": m} for (d, m) in results]
        # # Each env.reset() should return Data
        # self.obs = []
        # for env in self.envs:
        #     ret = env.reset()
        #     data, mask = ret
        #     self.obs.append({"data": data, "mask": mask})

    def _with_global_node(self, data: Data, mask: torch.Tensor) -> tuple[Data, torch.Tensor]:
        # data.x: [N, F], data.edge_index: [2, E]; mask: [N * N_NODE_ACTIONS] (bool)
        x = data.x
        device = x.device
        N, F = x.size(0), x.size(1)
        # Placeholder features for the global node (will be replaced in the model by a learned parameter)
        gfeat = torch.zeros(1, F, dtype=x.dtype, device=device)
        x_ext = torch.cat([x, gfeat], dim=0)  # [N+1, F]

        # Connect the global node (index N) to all nodes (bidirectional)
        if N > 0:
            nodes = torch.arange(N, dtype=torch.long, device=device)
            gidx  = torch.full((N,), N, dtype=torch.long, device=device)
            edge_g2n = torch.stack([gidx, nodes], dim=0)
            edge_n2g = torch.stack([nodes, gidx], dim=0)
            ei = data.edge_index if data.edge_index.numel() > 0 else torch.empty(2, 0, dtype=torch.long, device=device)
            edge_index_ext = torch.cat([ei, edge_g2n, edge_n2g], dim=1)
        else:
            edge_index_ext = torch.empty(2, 0, dtype=torch.long, device=device)

        # Mark which node is global
        is_global = torch.zeros(N + 1, dtype=torch.bool, device=device)
        is_global[N] = True

        # Extend the action mask with an all-false block for the global node
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        mask_ext = torch.cat([mask, torch.zeros(N_NODE_ACTIONS, dtype=torch.bool, device=mask.device)], dim=0)

        data_ext = Data(x=x_ext, edge_index=edge_index_ext)
        data_ext.is_global = is_global  # carried through PyG Batch
        return data_ext, mask_ext

    def total_train_updates(self):
        # updates = epochs * PPO updates per epoch; 1 Lightning train step == 1 PPO update in this setup
        if self.total_updates is None:
            # estimate if not set; you can also pass via hparams if you prefer exact
            self.total_updates = max(1, getattr(self.hparams, "max_epochs", 1) * getattr(self.hparams, "ppo_updates_per_epoch", 1))
        return self.total_updates

    def lr_anneal_factor(self):
        # linear from 1 -> 0
        frac = min(1.0, float(self.global_step) / float(self.total_train_updates()))
        return max(0.0, 1.0 - frac)

    def curr_ent_coef(self):
        frac = min(1.0, float(self.global_step) / float(self.total_train_updates()))
        return self._ent_coef_start * (1.0 - frac) + self._ent_coef_end * frac


    @torch.no_grad()
    def _obs_to_batch(self, obs_batch: List[dict]) -> Tuple[Batch, torch.Tensor]:
        datas, masks = [], []
        for o in obs_batch:
            d = o["data"]
            m = o["mask"]
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            if m is None:
                raise RuntimeError("Missing action mask in observation.")
            if self.use_global_node:
                d, m = self._with_global_node(d, m)
            datas.append(d)
            masks.append(m.to(torch.bool))

        batch = Batch.from_data_list(datas)
        mask = _pad_1d(masks, pad_value=0.0, dtype=torch.bool, device=self.device)
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
        # actions: [B] on cuda; move to cpu ints
        a_np = actions.detach().to("cpu").numpy()
        next_obs, rewards, dones, infos = self.vec.step(a_np)
        # episodic stats
        for i in range(self.num_envs):
            r, d = float(rewards[i]), bool(dones[i])
            self._ep_ret[i] += r
            self._ep_len[i] += 1
            if d:
                self._completed_returns.append(self._ep_ret[i])
                self._completed_lengths.append(self._ep_len[i])
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0
        return next_obs, rewards, dones, infos

    def on_train_end(self):
        try:
            self.vec.close()
        except Exception:
            pass

    def collect_rollout(self):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, timeout_buf, terminated_buf, T_reduce_buf, CNOT_reduce_buf = [], [], [], [], [], [], [], [],[],[]
        for _ in range(self.rollout_steps):
            action, logp, value = self._select_action(self.obs)
            obs_buf.append(self.obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
    
            self.obs, rewards, dones, infos = self._step_envs(action)
            rew_buf.append(torch.from_numpy(rewards))
            done_buf.append(torch.from_numpy(dones.astype(np.float32)))
            timeout_buf.append(torch.tensor([int(info.get("TimeLimit.truncated", False)) for info in infos]))
            terminated_buf.append(torch.tensor([int(info.get("terminated", False)) for info in infos]))
            # print("ppo print info", infos[0])
            T_reduce_buf.append(torch.tensor([int(info.get("T_reduced")) for info in infos]))
            CNOT_reduce_buf.append(torch.tensor([int(info.get("CNOT_reduced")) for info in infos]))

        with torch.no_grad():
            _, _, last_value = self._select_action(self.obs)  # [B]

        act = torch.stack(act_buf)                 # [T,B]
        logp = torch.stack(logp_buf)               # [T,B]
        val = torch.stack(val_buf).squeeze(-1)     # [T,B]
        rew = torch.stack(rew_buf)                 # [T,B]
        done = torch.stack(done_buf)               # [T,B]
        timeouts = torch.stack(timeout_buf)        # [T,B]
        terminated = torch.stack(terminated_buf)   # [T,B]
        T_reduced = torch.stack(T_reduce_buf)     # [T,B]
        CNOT_reduced = torch.stack(CNOT_reduce_buf)   # [T,B]
        return obs_buf, act, logp, val, rew, done, last_value.squeeze(-1), timeouts, T_reduced, CNOT_reduced

    @torch.no_grad()
    def compute_gae(self, rewards, values, dones, last_value,timeouts):
        # rewards/values/dones: [T,B], last_value: [B]
        device = self.device
        rewards = rewards.to(device)
        values = values.to(device)
        dones = dones.to(device)
        last_value = last_value.to(device)
        timeouts = timeouts.to(device)

        T, B = rewards.shape
        adv = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            real_done = (dones[t] > 0) & ~(timeouts[t] > 0) # treat timeout as nonterminal
            next_nonterminal = 1.0 - real_done.float()
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
        obs_traj, act, logp_old, val, rew, done, last_value, timeouts,T_reduced, CNOT_reduced = self.collect_rollout()
        # torch.cuda.synchronize()
        # end_time = time.time()
        # self.log("time/rollout", end_time - start_time, prog_bar=False, on_step=True, on_epoch=False)

        self._total_env_steps += self.rollout_steps * self.num_envs
        self.log_counter("counters/total_env_steps", self._total_env_steps, prog_bar=True)
        self.log_counter("counters/train_step", self.global_step, prog_bar=True)
        self.log_scalar("sched/ent_coef", self.curr_ent_coef())
        self.log_scalar("sched/lr", self.optimizers().param_groups[0]["lr"])

        # 2) GAE
        adv, ret = self.compute_gae(rew.detach(), val.detach(), done.detach(), last_value.detach(), timeouts.detach())
        T, B = act.shape
        total = T * B
        act = act.reshape(total).to(self.device)
        logp_old = logp_old.reshape(total).to(self.device)
        adv = _normalize_adv(adv.reshape(total)).to(self.device)
        ret = ret.reshape(total).to(self.device)
        
        val_old_flat = val.detach().reshape(total).to(self.device)  # v_t from rollout
        target_kl = 0.02  # try 0.01–0.05       

        with torch.no_grad():
            # rollout constants
            self.log_scalar("rollout/T", self.rollout_steps)
            self.log_scalar("rollout/B", self.num_envs)
            self.log_scalar("rollout/samples", self.rollout_steps * self.num_envs)

            # reward stats
            self.log_scalar("rollout/reward_mean", rew.float().mean(), prog_bar=True)
            self.log_scalar("rollout/reward_std", rew.float().std(unbiased=False))
            self.log_scalar("rollout/done_frac", done.float().mean())

            # statics 
            self.log_scalar("rollout/T_reduced_mean", T_reduced.float().mean())
            self.log_scalar("rollout/T_reduced_std", T_reduced.float().std(unbiased=False))
            self.log_scalar("rollout/CNOT_reduced_mean", CNOT_reduced.float().mean())
            self.log_scalar("rollout/CNOT_reduced_std", CNOT_reduced.float().std(unbiased=False))

            # value explained variance
            v_flat = val.detach().reshape(-1)
            r_flat = ret.detach()
            var_y = torch.var(r_flat, unbiased=False)
            ev = 1.0 - torch.var(r_flat - v_flat, unbiased=False) / (var_y + 1e-8)
            self.log_scalar("value/explained_variance", ev)


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

                # value clipping
                v_old_mb = val_old_flat[sl]
                v_clipped = v_old_mb + (values - v_old_mb).clamp(-self.clip_eps, self.clip_eps)
                v_loss_unclipped = F.mse_loss(values, ret_mb, reduction="mean")
                v_loss_clipped = F.mse_loss(v_clipped, ret_mb, reduction="mean")
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
                loss = pg_loss + self.vf_coef * v_loss - self.curr_ent_coef() * entropy
                #
                # v_loss = F.mse_loss(values, ret_mb)
                # loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                opt.step()

                with torch.no_grad():
                    approx_kl = (logp_old_mb - logp_new).mean()
                    if approx_kl > 1.5 * target_kl:
                        break  # break minibatch loop for this epoch
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

        self.log_scalars(metrics, prog_bar_keys=("loss/policy", "stats/return_mean"))


    def on_train_start(self):
        self.reset_envs()


    # ---------------- Logging helpers ----------------
    def _to_float_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(torch.float32)
        return torch.tensor(float(x), dtype=torch.float32)

    def log_scalar(self, name, value, prog_bar=False, sync_dist=False):
        """
        Use for real-valued metrics that can be reduced (loss, KL, entropy, returns, etc.)
        """
        v = self._to_float_tensor(value)
        self.log(name, v, on_step=True, on_epoch=False, prog_bar=prog_bar, sync_dist=sync_dist)

    def log_scalars(self, metrics: dict, prog_bar_keys=(), sync_dist=False):
        """
        Batch log of scalar metrics; prog_bar_keys is an iterable of names to show on the progress bar.
        """
        for k, v in metrics.items():
            self.log_scalar(k, v, prog_bar=(k in prog_bar_keys), sync_dist=sync_dist)

    def log_counter(self, name, value, step=None, prog_bar=False):
        """
        Use for integer counters (no reduction). Sends directly to the logger.
        Optionally mirrors as a float to the progress bar.
        """
        # extract plain int
        if isinstance(value, torch.Tensor):
            value = int(value.detach().item())
        else:
            value = int(value)
        step = int(self.global_step) if step is None else int(step)

        if getattr(self, "logger", None) is not None:
            # avoids Lightning reduction path and the 'needs to be floating' warning
            self.logger.log_metrics({name: value}, step=step)

        if prog_bar:
            # mirror as float for the bar; still harmless to reduce
            self.log(name, float(value), on_step=True, on_epoch=False, prog_bar=True)
    # -------------------------------------------------