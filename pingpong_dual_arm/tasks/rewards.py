"""Reward functions for competitive ping pong — ported from Isaaclab-TableTennisRobot.

All heavy computation is done in PingPongEnv._compute_intermediate_values().
These functions only return the precomputed tensors.
"""

from __future__ import annotations

import torch

if "TYPE_CHECKING":
    from tasks.pingpong_env import PingPongEnv


def paddle_contact_reward_A(env: "PingPongEnv") -> torch.Tensor:
    return env._contact_A


def paddle_contact_reward_B(env: "PingPongEnv") -> torch.Tensor:
    return env._contact_B


def table_success_reward_A(env: "PingPongEnv") -> torch.Tensor:
    return env._table_success_A


def table_success_reward_B(env: "PingPongEnv") -> torch.Tensor:
    return env._table_success_B


def table_fail_reward_A(env: "PingPongEnv") -> torch.Tensor:
    return env._table_fail_A


def table_fail_reward_B(env: "PingPongEnv") -> torch.Tensor:
    return env._table_fail_B


def ball_floor_penalty(env: "PingPongEnv") -> torch.Tensor:
    return env._ball_floor.float()


def velocity_reward_A(env: "PingPongEnv") -> torch.Tensor:
    return env._velocity_A


def velocity_reward_B(env: "PingPongEnv") -> torch.Tensor:
    return env._velocity_B


def ball_pos_reward_A(env: "PingPongEnv") -> torch.Tensor:
    return env._ball_pos_rw_A


def ball_pos_reward_B(env: "PingPongEnv") -> torch.Tensor:
    return env._ball_pos_rw_B
