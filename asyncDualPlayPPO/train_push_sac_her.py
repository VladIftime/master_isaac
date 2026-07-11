#!/usr/bin/env python3
"""Train SAC + HER agent for push primitive with DirectRLEnv.

One outer step = one complete push macro-action (72 substeps via cuRobo IK).
Uses SB3 SAC with HerReplayBuffer for sparse-goal relabeling.
"""

import argparse
import os
import signal
import sys

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


def atomic_sac_save(model, base_path, with_buffer=True):
    """Atomically save an SB3 model (+ its HER replay buffer).

    Writes to ``*.tmp`` files first, then ``os.replace`` into place so a
    process killed mid-save can never leave a truncated/corrupt checkpoint.
    Produces ``<base>.zip`` and (if with_buffer) ``<base>_replay.pkl``.
    """
    zip_tmp = base_path + ".zip.tmp"
    model.save(zip_tmp)
    os.replace(zip_tmp, base_path + ".zip")
    if with_buffer and getattr(model, "replay_buffer", None) is not None:
        rb_tmp = base_path + "_replay.pkl.tmp"
        model.save_replay_buffer(rb_tmp)
        os.replace(rb_tmp, base_path + "_replay.pkl")


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
            try:
                atomic_sac_save(self.model, self.save_path, with_buffer=True)
            except Exception as e:
                print(f"[CKPT] save failed at step {self.num_timesteps}: {e} — continuing.", flush=True)
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

    log_root = os.path.abspath(os.path.join("runs", "sac"))
    log_dir = os.path.join(log_root, args_cli.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    resumed = False
    model = None
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        try:
            model = SAC.load(
                args_cli.checkpoint, env=env_wrapped,
                tensorboard_log=None, seed=args_cli.seed,
            )
            resumed = True
            print(f"[INFO] Policy restored. num_timesteps={model.num_timesteps}")
            # Restore the HER replay buffer saved alongside the .zip.
            ckpt = args_cli.checkpoint
            rb_path = (ckpt[:-4] if ckpt.endswith(".zip") else ckpt) + "_replay.pkl"
            if os.path.exists(rb_path):
                try:
                    model.load_replay_buffer(rb_path)
                    print(f"[INFO] Replay buffer restored: {rb_path} "
                          f"(size={model.replay_buffer.size()})")
                except Exception as e:
                    print(f"[WARN] Replay buffer load failed ({e}); "
                          "continuing with empty buffer.", flush=True)
            else:
                print(f"[WARN] No replay buffer at {rb_path}; "
                      "continuing with empty buffer.", flush=True)
        except Exception as e:
            print(f"[WARN] Checkpoint load failed ({e}); starting fresh SAC.", flush=True)
            model = None
            resumed = False

    if model is None:
        model = SAC(
            "MultiInputPolicy",
            env_wrapped,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.95,
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
            tensorboard_log=None,
            seed=args_cli.seed,
            verbose=1,
        )

    total_timesteps = args_cli.max_iterations * cfg.scene.num_envs
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=ckpt_dir,
        name_prefix="agent",
        save_replay_buffer=True,
    )

    latest_ckpt_callback = LatestCheckpointCallback(
        save_freq=200000,
        save_path=os.path.join(log_dir, "latest_checkpoint"),
    )

    print(f"[TEST] Dry-run model.save() to verify checkpoint path is writable...")
    _test_path = os.path.join(log_dir, "_save_test")
    try:
        model.save(_test_path)
        print(f"[TEST] Save OK — {_test_path}.zip created ({os.path.getsize(_test_path + '.zip')} bytes)")
        os.remove(_test_path + ".zip")
    except Exception as e:
        print(f"[FATAL] model.save() failed before any training: {e}")
        env.close()
        sys.exit(1)

    print(f"[INFO] Total timesteps: {total_timesteps} "
          f"({args_cli.max_iterations} iters x {cfg.scene.num_envs} envs)")
    print(f"[INFO] Checkpoint intervals: agent=200000 steps, latest=200000 timesteps (with replay buffer)")
    print(f"[INFO] Replay buffer size: {100000}")
    print(f"[INFO] HER n_sampled_goal: 4, strategy: future")
    print(f"[INFO] Resumed: {resumed}")
    sys.stdout.flush()

    latest_ckpt_path = os.path.join(log_dir, "latest_checkpoint")

    def _save_on_signal(signum, frame):
        print(f"\n[CKPT] Received signal {signum} - saving latest_checkpoint (+replay buffer)...")
        try:
            atomic_sac_save(model, latest_ckpt_path, with_buffer=True)
            print(f"[CKPT] Saved to: {latest_ckpt_path}.zip (+_replay.pkl)")
        except Exception as e:
            print(f"[CKPT] Save on signal failed: {e}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _save_on_signal)
    signal.signal(signal.SIGUSR1, _save_on_signal)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=(not resumed),
            log_interval=4,
            progress_bar=True,
            callback=[checkpoint_callback, latest_ckpt_callback],
        )
    except FileNotFoundError:
        print("[WARN] TensorBoard async writer lost file during shutdown — "
              "training completed, ignoring.", flush=True)

    ckpt_path = os.path.join(ckpt_dir, "agent_final")
    atomic_sac_save(model, ckpt_path, with_buffer=True)
    print(f"[INFO] Final model saved to: {ckpt_path}.zip (+_replay.pkl)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
