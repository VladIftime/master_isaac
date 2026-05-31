#!/usr/bin/env python3
"""Test script: launch the ping pong dual-arm environment and step through it.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd pingpong_dual_arm
    python scripts/test_env.py --ik diffik --num_envs 4
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Lock torch submodules before AppLauncher — Isaac Sim's pip_prebundle
# contains incompatible torch builds that break imports otherwise.
import torch
import torch._dynamo     # noqa: F401
import torch._C          # noqa: F401
import torch.optim       # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test ping pong dual-arm environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run.")
parser.add_argument("--ik", type=str, default="diffik", choices=["diffik", "osc", "rmpflow", "curobo"],
                    help="IK solver to use.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg
from tasks.pingpong_env import PingPongEnv

cfg = PingPongDualArmEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.ik_solver = args_cli.ik

print(f"\n=== Testing Ping Pong Dual-Arm Environment ===")
print(f"  Num envs: {args_cli.num_envs}")
print(f"  IK solver: {args_cli.ik}")
print(f"  Steps: {args_cli.steps}")
print()

env = PingPongEnv(cfg=cfg)

print(f"Action space: {env.action_space}")
print(f"Observation space shape:")
for key, space in env.observation_space.items():
    print(f"  {key}: {space.shape}")

obs, info = env.reset()
print(f"Reset done. Starting rollout for {args_cli.steps} steps...")

for step in range(args_cli.steps):
    action = torch.rand(args_cli.num_envs, env.action_space.shape[1], device=env.device) * 0.3 - 0.15
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 100 == 0:
        alive = (~terminated).sum().item()
        ball_z = env.scene["ball"].data.root_pos_w[0, 2].item()
        print(f"  Step {step:4d}: alive={alive}, ball_z={ball_z:.3f}, "
              f"reward_mean={reward.mean().item():.4f}")

    if terminated.any():
        print(f"  Step {step:4d}: {terminated.sum().item()} envs terminated, resetting...")

print("\n=== Test complete ===")
env.close()
simulation_app.close()
