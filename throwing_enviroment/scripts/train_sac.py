#!/usr/bin/env python3
"""Train a SAC agent for the Gazebo-style throw primitive.

Uses stable-baselines3 SAC (matching the reference paper — Kasaei & Kasaei, ICRA 2023)
with DirectRLEnv for fast GPU-parallel training.

The agent learns 4 macro parameters:
  [initial_joint_value, final_joint_value, releasing_time, duration]

Each episode is a single throw (one outer step = full throw primitive).

Usage:
    source ~/env_isaaclab/bin/activate
    cd throwing_enviroment
    python scripts/train_sac.py --headless --num_envs=4096
    python scripts/train_sac.py --headless --num_envs=4096 --max_iterations=100000
    python scripts/train_sac.py --headless --num_envs=4096 --checkpoint path/to/model.zip
"""

import argparse
import os
import signal
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC agent for throw primitive.")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments (default: 4096).")
parser.add_argument("--max_iterations", type=int, default=100000,
                    help="Total training timesteps (default: 100000).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from SB3 checkpoint (.zip).")
parser.add_argument("--playing_arm_side", type=str, default="right",
                    choices=["right", "left"], help="Which arm throws.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg
from tasks.throwing_direct_env import ThrowingDirectEnv
from tasks.sb3_vec_env import DirectRLVecEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class LatestCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self._last_save = None

    def _on_step(self) -> bool:
        if self._last_save is None:
            self._last_save = self.num_timesteps
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(self.save_path, include=["replay_buffer"])
            self._last_save = self.num_timesteps
        return True


def main():
    cfg = ThrowingDirectEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.playing_arm_side = args_cli.playing_arm_side
    cfg.seed = args_cli.seed

    print(f"\n{'='*60}")
    print(f"  SAC Training — Throw Primitive (SB3 + DirectRLEnv)")
    print(f"  num_envs          : {cfg.scene.num_envs}")
    print(f"  playing_arm_side  : {cfg.playing_arm_side}")
    print(f"  max_iterations    : {args_cli.max_iterations}")
    print(f"  seed              : {cfg.seed}")
    print(f"  observation_dim   : {cfg.observation_space}")
    print(f"  action_dim        : {cfg.action_space}")
    print(f"{'='*60}\n")

    env = ThrowingDirectEnv(cfg=cfg)
    env_wrapped = DirectRLVecEnv(env)

    log_root = os.path.abspath(os.path.join("logs", "sac", "throwing_primitive"))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac_sb3")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        model = SAC.load(
            args_cli.checkpoint, env=env_wrapped,
            tensorboard_log=log_dir, seed=args_cli.seed,
        )
    else:
        model = SAC(
            "MlpPolicy", env_wrapped,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            target_entropy="auto",
            policy_kwargs={
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
            },
            tensorboard_log=log_dir,
            seed=args_cli.seed,
            verbose=1,
        )

    total_timesteps = args_cli.max_iterations * cfg.scene.num_envs
    ckpt_interval = 1000 * cfg.scene.num_envs
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_interval,
        save_path=ckpt_dir,
        name_prefix="agent",
        save_replay_buffer=False,
    )

    latest_ckpt_callback = LatestCheckpointCallback(
        save_freq=ckpt_interval,
        save_path=os.path.join(log_dir, "latest_checkpoint"),
    )

    print(f"[INFO] Total timesteps: {total_timesteps} ({args_cli.max_iterations} iterations × {cfg.scene.num_envs} envs)")
    print(f"[INFO] Checkpoint every: {ckpt_interval} timesteps ({ckpt_interval // cfg.scene.num_envs} iterations)")

    # Register signal handler to save final checkpoint on SIGTERM
    # (SIGUSR1 goes to bash only due to SBATCH --signal=B:USR1@120; SIGTERM reaches Python)
    latest_ckpt_path = os.path.join(log_dir, "latest_checkpoint")

    def _save_on_signal(signum, frame):
        print(f"\n[CKPT] Received signal {signum} — saving latest_checkpoint...")
        model.save(latest_ckpt_path, include=["replay_buffer"])
        print(f"[CKPT] Saved to: {latest_ckpt_path}.zip")
        print(f"[CKPT] Exiting to let bash cleanup trap fire.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _save_on_signal)

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=(args_cli.checkpoint is None),
        log_interval=10,
        progress_bar=True,
        callback=[checkpoint_callback, latest_ckpt_callback],
    )

    ckpt_path = os.path.join(ckpt_dir, "agent_final")
    model.save(ckpt_path, include=["replay_buffer"])
    print(f"[INFO] Final model saved to: {ckpt_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
