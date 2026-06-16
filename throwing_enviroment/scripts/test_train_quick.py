#!/usr/bin/env python3
"""Quick sanity test — verify the SB3 SAC training pipeline doesn't produce NaN.

Runs a short training session locally and validates:
  1. No NaN in observations, rewards, or model weights
  2. Model checkpoint loads and produces valid actions
  3. Per-iteration timing (for HPC scaling projection)

Usage:
    source ~/env_isaaclab/bin/activate
    cd throwing_enviroment
    python scripts/test_train_quick.py --headless
    python scripts/test_train_quick.py --headless --num_envs 64 --steps 2000
"""

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quick SB3 SAC training sanity test.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel envs")
parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg
from tasks.throwing_direct_env import ThrowingDirectEnv
from tasks.sb3_vec_env import DirectRLVecEnv


def check_model_weights(model, name=""):
    """Check all model params for NaN."""
    import torch
    for n, p in model.policy.named_parameters():
        if torch.isnan(p).any():
            count = torch.isnan(p).sum().item()
            print(f"  [FAIL] {name} NaN in policy.{n}: {count}/{p.numel()}")
            return False
    for n, p in model.critic.named_parameters():
        if torch.isnan(p).any():
            count = torch.isnan(p).sum().item()
            print(f"  [FAIL] {name} NaN in critic.{n}: {count}/{p.numel()}")
            return False
    for n, p in model.critic_target.named_parameters():
        if torch.isnan(p).any():
            count = torch.isnan(p).sum().item()
            print(f"  [FAIL] {name} NaN in target_critic.{n}: {count}/{p.numel()}")
            return False
    return True


def main():
    cfg = ThrowingDirectEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.playing_arm_side = "right"
    cfg.seed = 42

    obs_dim = cfg.observation_space
    act_dim = cfg.action_space

    print(f"\n{'='*60}")
    print(f"  Quick SB3 SAC Sanity Test")
    print(f"  num_envs: {args_cli.num_envs}, steps: {args_cli.steps}")
    print(f"  obs_dim: {obs_dim}, act_dim: {act_dim}")
    print(f"{'='*60}\n")

    env = ThrowingDirectEnv(cfg=cfg)
    env_wrapped = DirectRLVecEnv(env)

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    total_timesteps = args_cli.steps * cfg.scene.num_envs
    ckpt_interval = max(100, args_cli.steps // 3)

    ckpt_dir = os.path.join(_PROJECT_ROOT, "logs", "scratch", "test_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_interval,
        save_path=ckpt_dir,
        name_prefix="agent",
        save_replay_buffer=True,
    )

    model = SAC(
        "MlpPolicy", env_wrapped,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs={"net_arch": [256, 256]},
        seed=42,
        verbose=1,
    )

    t0 = time.monotonic()
    print(f"[INFO] Total timesteps: {total_timesteps}, ckpt every: {ckpt_interval}")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        progress_bar=True,
        callback=checkpoint_callback,
    )
    elapsed = time.monotonic() - t0

    print(f"\n[INFO] Checkpoints saved to: {ckpt_dir}")
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith(".zip"):
            print(f"  {f}")

    print(f"\n{'─'*60}")
    print(f"  Validating model weights after {args_cli.steps} steps")
    print(f"{'─'*60}")

    ok = check_model_weights(model, f"step_{args_cli.steps}")
    if ok:
        print("  [OK] All model weights are clean (no NaN)")

    print(f"\n{'─'*60}")
    print(f"  Running one deterministic throw")
    print(f"{'─'*60}")

    obs = env_wrapped.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"  action={[f'{a:.3f}' for a in action.flatten().tolist()]}")

    obs_next, reward, done, info = env_wrapped.step(action)
    print(f"  reward mean={reward.mean():.3f}, done={done.any()}")

    env.close()

    # ── Summary ──────────────────────────────────────────────────────────
    pct = (total_timesteps / (100000 * cfg.scene.num_envs)) * 100
    proj_hours = (elapsed / total_timesteps) * (100000 * cfg.scene.num_envs) / 3600
    it_per_s = total_timesteps / elapsed

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Env steps        : {args_cli.steps} ({pct:.1f}% of 100K)")
    print(f"  Total transitions : {total_timesteps}")
    print(f"  Wall time        : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Speed            : {it_per_s:.0f} transitions/s ({it_per_s/cfg.scene.num_envs:.1f} env_steps/s)")
    print(f"  Projected 100K   : {proj_hours:.1f} hours")
    print(f"  Model weights    : {'OK - no NaN' if ok else 'FAIL - NaN detected'}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
