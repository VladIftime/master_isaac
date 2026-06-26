#!/usr/bin/env python3
"""Train SAC + HER agent for push primitive with DirectRLEnv.

One outer step = one complete push macro-action (72 substeps via cuRobo IK).
Uses SB3 SAC with HerReplayBuffer for sparse-goal relabeling.
"""

import argparse
import os
import signal
import sys
from datetime import datetime

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, ".."))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC+HER agent for push primitive.")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=2000,
                    help="Total training iterations (each iteration = num_envs timesteps).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from SB3 checkpoint (.zip).")
parser.add_argument("--exp_name", type=str, default="push_sac_her",
                    help="Experiment name for log dir.")
parser.add_argument("--rel_act", action="store_true", dest="rel_act",
                    help="Use object-relative push actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from tasks.push_direct_env_cfg import PushDirectEnvCfg
from tasks.push_direct_env import PushDirectEnv
from tasks.sb3_vec_env import DirectRLVecEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.her import HerReplayBuffer


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
    cfg = PushDirectEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.rel_act = args_cli.rel_act

    print(f"\n{'='*60}")
    print(f"  SAC + HER Training — Push Primitive (SB3 + DirectRLEnv)")
    print(f"  num_envs         : {cfg.scene.num_envs}")
    print(f"  max_iterations   : {args_cli.max_iterations}")
    print(f"  seed             : {cfg.seed}")
    print(f"  rel_obs          : {cfg.rel_obs}")
    print(f"  rel_act          : {cfg.rel_act}")
    print(f"  max_pushes/ep    : {cfg.max_pushes_per_episode}")
    print(f"  decimation       : {cfg.decimation} substeps")
    print(f"  observation_dim  : {cfg.observation_space} (flat)")
    print(f"  action_dim       : {cfg.action_space}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    print("Creating environment...")
    env = PushDirectEnv(cfg=cfg)
    env_wrapped = DirectRLVecEnv(env)

    log_root = os.path.abspath(os.path.join("runs", "sac", "push_primitive"))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac_her")
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
            "MultiInputPolicy",
            env_wrapped,
            learning_rate=3e-4,
            buffer_size=200000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            target_entropy="auto",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            policy_kwargs={
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
            },
            tensorboard_log=log_dir,
            seed=args_cli.seed,
            verbose=1,
        )

    total_timesteps = args_cli.max_iterations * cfg.scene.num_envs
    ckpt_interval = max(1, total_timesteps // 10)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_interval,
        save_path=ckpt_dir,
        name_prefix="agent",
        save_replay_buffer=False,
    )

    latest_ckpt_callback = LatestCheckpointCallback(
        save_freq=max(1, total_timesteps // 20),
        save_path=os.path.join(log_dir, "latest_checkpoint"),
    )

    print(f"[INFO] Total timesteps: {total_timesteps} "
          f"({args_cli.max_iterations} iters x {cfg.scene.num_envs} envs)")
    print(f"[INFO] Checkpoint interval: {ckpt_interval} timesteps")
    print(f"[INFO] Replay buffer size: {200000}")
    print(f"[INFO] HER n_sampled_goal: 4, strategy: future")
    sys.stdout.flush()

    latest_ckpt_path = os.path.join(log_dir, "latest_checkpoint")

    def _save_on_signal(signum, frame):
        print(f"\n[CKPT] Received signal {signum} - saving latest_checkpoint...")
        model.save(latest_ckpt_path, include=["replay_buffer"])
        print(f"[CKPT] Saved to: {latest_ckpt_path}.zip")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _save_on_signal)

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=(args_cli.checkpoint is None),
        log_interval=4,
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
