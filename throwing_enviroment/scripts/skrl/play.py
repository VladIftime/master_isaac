#!/usr/bin/env python3
"""Playback a trained RL agent for throwing with skrl.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/skrl/play.py --task=Throwing-Direct-v0 --checkpoint=path/to/agent.pt --num_envs=1
"""

import argparse
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Playback a trained throwing agent.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Throwing-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "IPPO", "MAPPO"])
parser.add_argument("--ik_solver", type=str, default=None, help="IK solver.")
parser.add_argument("--playing_arm_side", type=str, default="right", help="Arm side.")

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import random

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Throwing.tasks  # noqa: F401

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = (
    "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.ik_solver is not None:
        env_cfg.ik_solver = args_cli.ik_solver
    if args_cli.playing_arm_side:
        env_cfg.playing_arm_side = args_cli.playing_arm_side

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    runner = None
    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
        runner = Runner(env, agent_cfg)
    elif args_cli.ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner
        runner = Runner(env, agent_cfg)

    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    obs = env.reset()
    frames = []
    total_reward = 0.0

    while simulation_app.is_running():
        with torch.no_grad():
            actions = runner.agent.running_models["policy"].act(obs["states"], timestep=0, timesteps=1)
        actions = actions.get("mean_actions", actions)

        env.render()
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards.mean().item()

        if dones.any():
            print(f"Episode done, reward={total_reward:.3f}")
            total_reward = 0.0
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
