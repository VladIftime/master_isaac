"""
Reward functions for asymmetric dual-play (Asymmetric Self-Play paper).

Alice's rewards are outcome-based: she receives feedback only at the end of
Bob's episode, based on whether Bob succeeded or failed.

Bob's rewards are computed in AsyncDualPlayEnvWrapper._compute_bob_sparse_rewards(),
not here, to keep them integer-valued and free of IsaacLab's dt-scaling.
"""

# Single source of truth for all Alice reward magnitudes.
# Import these constants in train.py — never hardcode the values there.
ALICE_BOB_FAIL_REWARD: float = 1.0       # Bob failed to reach the goal
ALICE_BOB_SUCCESS_REWARD: float = 0.0    # Bob succeeded
ALICE_VALID_GOAL_BONUS: float = 1.0      # Alice set a valid, in-zone goal
ALICE_OUT_OF_ZONE_PENALTY: float = -3.0  # Alice's goal was valid but outside the placement area

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def alice_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Alice's per-episode outcome reward based on whether Bob solved the goal.

    Returns ALICE_BOB_FAIL_REWARD when Bob fails and ALICE_BOB_SUCCESS_REWARD
    when Bob succeeds. The goal-setting bonus (ALICE_VALID_GOAL_BONUS) is applied
    separately by the wrapper at goal-validation time.

    Returns:
        Reward tensor (num_envs,)
    """
    batch_size = env.scene[object_cfg.name].data.root_pos_w.shape[0]
    rewards = torch.zeros(batch_size, device=env.device)

    if hasattr(env, "extras") and "bob_success" in env.extras:
        bob_success = env.extras["bob_success"]
        rewards = torch.where(
            bob_success,
            torch.full_like(rewards, ALICE_BOB_SUCCESS_REWARD),
            torch.full_like(rewards, ALICE_BOB_FAIL_REWARD),
        )

    return rewards


def valid_goal_bonus(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Alice's bonus for setting a valid, in-zone goal.

    Triggered once per goal when the wrapper calls mark_goal_valid().

    Returns:
        Reward tensor (num_envs,)
    """
    rewards = torch.zeros(env.scene.num_envs, device=env.device)

    if hasattr(env, "extras") and "goal_valid" in env.extras:
        goal_valid = env.extras["goal_valid"]
        rewards = torch.where(goal_valid, torch.full_like(rewards, ALICE_VALID_GOAL_BONUS), rewards)

    return rewards


def out_of_bounds_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    x_range: tuple = (-0.6, 0.6),
    y_range: tuple = (0.3, 0.9),
) -> torch.Tensor:
    """
    Alice's penalty for placing an object outside the valid placement area.

    The placement area (x_range, y_range) is smaller than the table bounds,
    so Alice is discouraged from setting goals at the table edges where Bob
    cannot reliably reach.

    Returns:
        Penalty tensor (num_envs,)
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w

    if pos.dim() == 2:
        pos = pos.unsqueeze(1)

    pos = pos - env.scene.env_origins.unsqueeze(1)

    x_out = (pos[..., 0] < x_range[0]) | (pos[..., 0] > x_range[1])
    y_out = (pos[..., 1] < y_range[0]) | (pos[..., 1] > y_range[1])
    any_out = (x_out | y_out).any(dim=1)

    return torch.where(
        any_out,
        torch.full_like(any_out, ALICE_OUT_OF_ZONE_PENALTY, dtype=torch.float),
        torch.zeros_like(any_out, dtype=torch.float),
    )
