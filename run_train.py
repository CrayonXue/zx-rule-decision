# train_gnn.py
# End-to-end PPO training for ZXCalculus (GNN variant) with PyTorch Lightning + SwanLab.

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# Optional SwanLab
try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
    import swanlab
    HAS_SWAN = True
except Exception:
    HAS_SWAN = False

# Local modules
from zxreinforce.Resetters import Resetter_GraphBank
from zxreinforce.zx_env_circuit import ZXCalculus
from zxreinforce.ppo import PPOLightningGNN


class StepsDataset(torch.utils.data.Dataset):
    """Dummy dataset that triggers N PPO updates per epoch."""
    def __init__(self, ppo_updates_per_epoch: int):
        self.ppo_updates_per_epoch = int(ppo_updates_per_epoch)

    def __len__(self):
        return self.ppo_updates_per_epoch

    def __getitem__(self, idx):
        return 0  # unused


def build_env_fn(args):
    """Factory returning a fresh environment instance each time it's called."""
    def make_env():
        exp_name = f"GraphBank_nq[{args.num_qubits_min}-{args.num_qubits_max}]_gates[{args.min_gates}-{args.max_gates}]_length{args.data_length}"
        bank_path = os.path.join(args.data_dir, exp_name)
        resetter = Resetter_GraphBank(
            bank_path=bank_path,
            seed=None if args.resetter_seed < 0 else args.resetter_seed,
        )
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


def train(args):
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Experiment naming
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.exp_name or (
        f"ZX-PPO-GNN_nq[{args.num_qubits_min}-{args.num_qubits_max}]_"
        f"gates[{args.min_gates}-{args.max_gates}]_envs{args.num_envs}_"
        f"T{args.rollout_steps}_mb{args.minibatch_size}_ppo{args.ppo_epochs}_"
        f"lr{args.lr}_{timestamp}"
    )
    save_root = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_root, exist_ok=True)

    # Logger
    if args.logger_type == "swanlab" and HAS_SWAN:
        try:
            swanlab.login(save=True)
        except Exception:
            pass
        logger = SwanLabLogger(
            project=args.swan_project,
            experiment_name=exp_name,
            logdir=os.path.join(save_root, "swanlab"),
            log_model="all",
        )
        logger.log_hyperparams(vars(args))
    else:
        logger = TensorBoardLogger(
            save_dir=os.path.join(save_root, "tb"),
            name=None,
            default_hp_metric=False,
            version="."  # write directly into tb/
        )

    # Env factory and agent
    env_fn = build_env_fn(args)
    agent = PPOLightningGNN(
        env_fn=env_fn,
        node_feat_dim=args.node_feat_dim,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        rollout_steps=args.rollout_steps,
        batch_size=args.minibatch_size,
        epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lr=args.lr,
        num_envs=args.num_envs,
    )

    # Checkpointing
    ckpt_dir = os.path.join(save_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step={step}-retmean={stats_return_mean:.3f}",
        monitor="stats/return_mean",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        save_on_train_epoch_end=False,
        every_n_train_steps=args.ckpt_every_n_steps,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Strategy
    strategy = "auto"
    if args.devices and int(args.devices) > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_train_steps if args.max_train_steps > 0 else None,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        accumulate_grad_batches=1,   # PPO handles its own minibatching
        gradient_clip_val=None,      # clipping done inside module
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=strategy,
        log_every_n_steps=1,
        enable_checkpointing=True,
        default_root_dir=save_root,
        limit_train_batches=1.0,     # consume whole StepsDataset
    )

    # DataLoader that triggers training_step repeatedly
    steps_ds = StepsDataset(ppo_updates_per_epoch=args.ppo_updates_per_epoch)
    steps_loader = torch.utils.data.DataLoader(
        steps_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False
    )

    # Fit
    ckpt_path = args.resume_from if args.resume_from else None
    trainer.fit(agent, train_dataloaders=steps_loader, ckpt_path=ckpt_path)

    if args.logger_type == "swanlab" and HAS_SWAN:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="PPO-GNN training for ZXCalculus with PyTorch Lightning")

    # Repro
    parser.add_argument("--seed", type=int, default=42)

    # IO / Logging
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--logger_type", type=str, default="tensorboard", choices=["tensorboard", "swanlab"])
    parser.add_argument("--swan_project", type=str, default="ZX-rule")
    parser.add_argument("--resume_from", type=str, default="", help="Path to a checkpoint to resume from.")

    # Hardware
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")

    # Trainer schedule
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--ppo_updates_per_epoch", type=int, default=50, help="Number of PPO updates per epoch.")
    parser.add_argument("--max_train_steps", type=int, default=0, help="Optional hard stop on total trainer steps (0 = ignore).")
    parser.add_argument("--ckpt_every_n_steps", type=int, default=200)
    parser.add_argument("--save_top_k", type=int, default=3)

    # PPO hyperparameters
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Samples per env per update.")
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--vf_coef", type=float, default=0.5)

    # GNN model dims
    parser.add_argument("--node_feat_dim", type=int, default=5 + 10 + 1)  # matches env x: 5+10+1+1
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=128)

    # Environment (ZXCalculus)
    parser.add_argument("--env_max_steps", type=int, default=100)
    parser.add_argument("--step_penalty", type=float, default=0.02)
    parser.add_argument("--length_penalty", type=float, default=0.0)
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
    parser.add_argument("--resetter_seed", type=int, default=-1, help="-1 = random each reset")

    args = parser.parse_args()
    train(args)