#!/usr/bin/env python3
"""Run the ping pong environment with a random policy for validation.

Usage:
    cd pingpong_dual_arm
    ../../isaaclab.sh -p scripts/random_policy.py -- --ik diffik --num_envs 4

Runs the environment with random delta-pose actions to verify physics,
robot rendering, and basic functionality.
"""

import argparse
import os
import sys
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random policy rollout for ping pong env.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=2000, help="Number of steps.")
parser.add_argument("--ik", type=str, default="diffik", choices=["diffik", "osc", "rmpflow"],
                    help="IK solver.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg
from tasks.pingpong_env import PingPongDualArmEnv

cfg = PingPongDualArmEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.ik_solver = args_cli.ik

print(f"Random policy with {args_cli.ik} IK, {args_cli.num_envs} envs")

env = PingPongDualArmEnv(cfg)

obs, info = env.reset()

action_scale = 0.05
total_reward = torch.zeros(args_cli.num_envs, device=env.device)
total_steps = torch.zeros(args_cli.num_envs, device=env.device)
rally_count = 0
points_won_A = 0
points_won_B = 0

for step in range(args_cli.steps):
    action = torch.randn(args_cli.num_envs, env.action_space.shape[1], device=env.device) * action_scale
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward.flatten()
    total_steps += 1

    if step % 200 == 0:
        ball_z = env.scene["ball"].data.root_pos_w[0, 2].item()
        ee_A = (env.scene["robot_A"].data.body_pos_w[0, :] - env.scene.env_origins[0]).mean(dim=0)
        ee_B = (env.scene["robot_B"].data.body_pos_w[0, :] - env.scene.env_origins[0]).mean(dim=0)
        print(f"Step {step:4d}: ball_z={ball_z:.3f}, "
              f"ee_A=({ee_A[0]:.3f},{ee_A[1]:.3f},{ee_A[2]:.3f}), "
              f"ee_B=({ee_B[0]:.3f},{ee_B[1]:.3f},{ee_B[2]:.3f})")

    if terminated.any():
        env_id = torch.where(terminated)[0][0].item()
        print(f"Step {step:4d}: term env {env_id}, "
              f"ball_z={env.scene['ball'].data.root_pos_w[env_id,2].item():.3f}")

print(f"\nDone. Avg reward: {(total_reward / total_steps).mean().item():.6f}")
env.close()
simulation_app.close()
