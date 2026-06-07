"""Reward functions for the throwing environment.

All heavy computation is done in ThrowingEnv._compute_rewards().
These functions are lightweight accessors returning precomputed reward tensors.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .throwing_env import ThrowingEnv


def dist_to_target_reward(env: "ThrowingEnv") -> torch.Tensor:
    """Gaussian distance reward: exp(-dist^2 / 0.1)."""
    return env._dist_reward


def success_bonus(env: "ThrowingEnv") -> torch.Tensor:
    """+2 one-time bonus when object lands within 0.15m of target."""
    return env._success_bonus


def ee_velocity_reward(env: "ThrowingEnv") -> torch.Tensor:
    """+0.5 reward proportional to EE velocity while holding object."""
    return env._ee_vel_reward
