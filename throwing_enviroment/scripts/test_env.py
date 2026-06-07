#!/usr/bin/env python3
"""Test script: launch the throwing environment and step through it.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/test_env.py --ik diffik --num_envs 4
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test throwing environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=2000, help="Number of steps to run.")
parser.add_argument(
    "--ik",
    type=str,
    default="diffik",
    choices=["diffik", "osc", "rmpflow", "curobo"],
    help="IK solver to use.",
)
parser.add_argument("--arm", type=str, default="right", choices=["left", "right"], help="Arm to control.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tasks.throwing_env_cfg import ThrowingEnvCfg
from tasks.throwing_env import ThrowingEnv

cfg = ThrowingEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.ik_solver = args_cli.ik
cfg.playing_arm_side = args_cli.arm

print(f"\n=== Testing Throwing Environment ===")
print(f"  Num envs: {args_cli.num_envs}")
print(f"  IK solver: {args_cli.ik}")
print(f"  Arm: {args_cli.arm}")
print(f"  Steps: {args_cli.steps}")
print()

env = ThrowingEnv(cfg=cfg)

print(f"Action space: {env.action_space}")
print(f"Observation space shape:")
for key, space in env.observation_space.items():
    print(f"  {key}: {space.shape}")

obs, info = env.reset()
print(f"Reset done. Starting rollout for {args_cli.steps} steps...")

for step in range(args_cli.steps):
    action = (
        torch.rand(args_cli.num_envs, env.action_space.shape[1], device=env.device)
        * 0.3
        - 0.15
    )
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 100 == 0:
        alive = (~terminated).sum().item()
        milk_z = env.scene["milk"].data.root_pos_w[0, 2].item()
        target_pos = env.scene["target"].data.root_pos_w[0, :3]
        print(
            f"  Step {step:4d}: alive={alive}, "
            f"milk_z={milk_z:.3f}, "
            f"target=({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}), "
            f"reward_mean={reward.mean().item():.4f}"
        )

    if terminated.any():
        print(f"  Step {step:4d}: {terminated.sum().item()} envs terminated, resetting...")

print("\n=== Test complete ===")
env.close()
simulation_app.close()
