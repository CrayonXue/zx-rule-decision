# train.py
# End-to-end PPO training script for the ZXCalculus environment using PyTorch Lightning.
# This follows a "trainer + logger + callbacks" style similar to your previous project.

import argparse
import os
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# Optional: SwanLab (Lightning integration)

from swanlab.integration.pytorch_lightning import SwanLabLogger
import swanlab
HAS_SWAN = True


# Local imports
from zxreinforce.Resetters import Resetter_Circuit
from zxreinforce.zx_env_circuit import ZXCalculus
from zxreinforce.zx_gym_wrapper import ZXGymWrapper
from zxreinforce.ppo import PPOLightning


class StepsDataset(torch.utils.data.Dataset):
    """A dummy dataset that yields a fixed number of steps per epoch.
    Each item triggers one PPO update via PPOLightning.training_step."""
    def __init__(self, steps_per_epoch: int):
        self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        return 0  # batch is ignored


def build_env_fn(args):
    # Closure that constructs a fresh wrapped environment (called inside each Lightning process)
    def make_env():
        resetter = Resetter_Circuit(
            num_qubits_min=args.num_qubits_min,
            num_qubits_max=args.num_qubits_max,
            min_gates=args.min_gates,
            max_gates=args.max_gates,
            p_t=args.p_t,
            p_h=args.p_h,
            seed=None if args.resetter_seed < 0 else args.resetter_seed,
        )
        env = ZXCalculus(
            max_steps=args.env_max_steps,
            add_reward_per_step=args.step_penalty,
            resetter=resetter,
            count_down_from=args.count_down_from,
            dont_allow_stop=args.dont_allow_stop,
            extra_state_info=False,
            adapted_reward=args.adapted_reward,
        )
        wrapped = ZXGymWrapper(
            env,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_qubits=args.max_qubits,
        )
        return wrapped
    return make_env


def train(args):
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Experiment naming
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.exp_name
    if not exp_name:
        exp_name = (
            f"ZX-PPO_nq[{args.num_qubits_min}-{args.num_qubits_max}]_"
            f"gates[{args.min_gates}-{args.max_gates}]_"
            f"N{args.max_nodes}_E{args.max_edges}_"
            f"envs{args.num_envs}_rs{args.rollout_steps}_mb{args.minibatch_size}_"
            f"pe{args.ppo_epochs}_lr{args.lr}_{timestamp}"
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
            version="."  # write directly under tb/
        )

    # Env factory and Lightning module
    env_fn = build_env_fn(args)
    agent = PPOLightning(
        env_fn=env_fn,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
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
        device="cuda" if args.accelerator in ("gpu", "auto") and torch.cuda.is_available() else "cpu",
    )

    # Checkpointing on training metric (logged each update in PPOLightning)
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
        max_steps=args.max_steps if args.max_steps > 0 else None,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        accumulate_grad_batches=1,  # PPO does its own minibatching
        gradient_clip_val=None,     # we clip inside the LightningModule
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=strategy,
        log_every_n_steps=1,
        enable_checkpointing=True,
        default_root_dir=save_root,
        limit_train_batches=1.0,    # consume the full StepsDataset each epoch
    )

    # Dummy dataloader to drive training_step calls
    steps_ds = StepsDataset(steps_per_epoch=args.steps_per_epoch)
    steps_loader = torch.utils.data.DataLoader(
        steps_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False
    )

    # Fit
    trainer.fit(agent, train_dataloaders=steps_loader)

    if args.logger_type == "swanlab" and HAS_SWAN:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for ZXCalculus with PyTorch Lightning")

    # Repro
    parser.add_argument("--seed", type=int, default=42)

    # Logging and IO
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--logger_type", type=str, default="swanlab", choices=["tensorboard", "swanlab"])
    parser.add_argument("--swan_project", type=str, default="ZX-rule")

    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")

    # Trainer schedule
    parser.add_argument("--max_epochs", type=int, default=1000, help="Number of outer epochs (each contains several PPO updates).")
    parser.add_argument("--steps_per_epoch", type=int, default=50, help="How many PPO updates per epoch (each update collects a rollout).")
    parser.add_argument("--max_steps", type=int, default=1000, help="Optional hard stop on total trainer steps (0 = ignore).")
    parser.add_argument("--ckpt_every_n_steps", type=int, default=200)
    parser.add_argument("--save_top_k", type=int, default=3)

    # PPO hyperparameters
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=2048, help="T in GAE; samples per env per update.")
    parser.add_argument("--minibatch_size", type=int, default=512, help="Minibatch size for PPO updates.")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Gradient epochs per update.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--vf_coef", type=float, default=0.5)

    # Wrapper (fixed shapes)
    parser.add_argument("--max_nodes", type=int, default=128)
    parser.add_argument("--max_edges", type=int, default=256)
    parser.add_argument("--max_qubits", type=int, default=64)

    # Environment (ZXCalculus)
    parser.add_argument("--env_max_steps", type=int, default=100)
    parser.add_argument("--step_penalty", type=float, default=-0.02, help="Added each step to incentivize shorter solutions.")
    parser.add_argument("--adapted_reward", action="store_true", default=True)
    parser.add_argument("--count_down_from", type=int, default=20)
    parser.add_argument("--dont_allow_stop", action="store_true", default=False)

    # Resetter (random circuit generator)
    parser.add_argument("--num_qubits_min", type=int, default=2)
    parser.add_argument("--num_qubits_max", type=int, default=6)
    parser.add_argument("--min_gates", type=int, default=5)
    parser.add_argument("--max_gates", type=int, default=30)
    parser.add_argument("--p_t", type=float, default=0.2)
    parser.add_argument("--p_h", type=float, default=0.2)
    parser.add_argument("--resetter_seed", type=int, default=-1, help="-1 = random each reset")

    args = parser.parse_args()
    train(args)