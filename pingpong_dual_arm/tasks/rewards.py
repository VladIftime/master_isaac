"""Reward functions for competitive ping pong.

Scoring:
  - Rally time bonus: reward for keeping the ball in play
  - Ball contact: reward when the ball contacts the opponent's racket
  - Ball height: reward for keeping the ball above the net
  - Point won: terminal reward when opponent fails to return
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def rally_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Positive reward for each timestep the ball stays above the table.

    Encourage maintaining a rally.

    Returns:
        (num_envs,) — 1.0 if ball is in play, 0.0 otherwise
    """
    ball = env.scene[asset_cfg.name]
    dev = ball.data.root_pos_w.device
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2].to(dev)
    above_table = ball_z > 0.02
    in_play = torch.ones(env.num_envs, device=dev)
    return torch.where(above_table, in_play, torch.zeros_like(in_play))


def ball_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_height: float = 0.15,
) -> torch.Tensor:
    """Reward for keeping the ball at a playable height."""
    ball = env.scene[asset_cfg.name]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2].to(ball.data.root_pos_w.device)
    above_table = ball_z > 0.05
    height_diff = torch.abs(ball_z - target_height)
    reward = torch.exp(-height_diff * 5.0)
    return reward * above_table.float()


def ball_contact_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward when the ball is near a racket and has high speed.

    Proxy for contact detection without contact sensors.

    Returns:
        (num_envs,) — contact proxy reward
    """
    ball = env.scene[asset_cfg.name]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2].to(ball.data.root_pos_w.device)
    ball_speed = torch.norm(ball.data.root_lin_vel_w, dim=-1)
    in_play_zone = (ball_z > 0.05) & (ball_z < 0.5)
    has_speed = ball_speed > 0.5
    return (in_play_zone & has_speed).float() * 0.1


def point_won_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminal reward: +1 when you win the point, -1 when you lose.

    A point is won when the ball passes the opponent's end of the table
    or the opponent fails to return.

    This is a placeholder — actual scoring logic depends on the game rules
    and should be set via `env.extras` by a point-tracking system.

    Returns:
        (num_envs,) — point outcome reward
    """
    rewards = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "extras") and "point_outcome" in env.extras:
        outcome = env.extras["point_outcome"]  # 1 = won, -1 = lost, 0 = ongoing
        rewards = outcome.float()
    return rewards
