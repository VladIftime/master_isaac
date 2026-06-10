#!/usr/bin/env python3
"""Train a SAC agent for the Gazebo-style throw primitive.

The agent learns 4 macro parameters:
  [initial_joint_value, final_joint_value, releasing_time, duration]

Each episode is a single throw (one outer step = full IK grasping + throw).

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/train_sac.py --headless --num_envs=64
    python scripts/train_sac.py --headless --num_envs=128 --max_iterations=35000
    python scripts/train_sac.py --headless --num_envs=64 --checkpoint=logs/skrl/throwing_primitive/.../agent_5000.pt
"""

import argparse
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC agent for throw primitive.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=35000, help="Total training timesteps.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint.")
parser.add_argument("--playing_arm_side", type=str, default="right", help="Which arm throws.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import yaml
from datetime import datetime

import torch
import numpy as np

import skrl
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import wrap_env

from tasks.throwing_primitive_env_cfg import ThrowingPrimitiveEnvCfg
from tasks.throwing_primitive_env import ThrowingPrimitiveEnv


def main():
    cfg = ThrowingPrimitiveEnvCfg()
    cfg.num_envs = args_cli.num_envs
    cfg.playing_arm_side = args_cli.playing_arm_side
    cfg.seed = args_cli.seed

    print(f"\n{'='*60}")
    print(f"  SAC Training — Throw Primitive")
    print(f"  num_envs          : {cfg.num_envs}")
    print(f"  playing_arm_side  : {cfg.playing_arm_side}")
    print(f"  max_iterations    : {args_cli.max_iterations}")
    print(f"  seed              : {cfg.seed}")
    print(f"{'='*60}\n")

    env = ThrowingPrimitiveEnv(cfg=cfg)

    agent_cfg_path = os.path.join(
        _PROJECT_ROOT, "source", "Throwing", "Throwing", "tasks",
        "throwing", "agents", "skrl_sac_cfg.yaml",
    )
    with open(agent_cfg_path, "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["seed"] = args_cli.seed
    agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac_torch"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env_wrapped = wrap_env(env, wrapper="gymnasium")

    runner = Runner(env_wrapped, agent_cfg)

    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)

    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
