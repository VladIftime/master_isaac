"""Event handlers for ping pong environment resets.

Ported from Isaaclab-TableTennisRobot:
  - serve_ball_alternating: serve from alternating sides with randomized velocity
  - reset_robot_joints: reset both robots to default joint positions
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .pingpong_env import PingPongEnv


_serve_side = None


def reset_robot_joints(
    env: "PingPongEnv",
    env_ids: torch.Tensor,
) -> None:
    """Reset both robots to their default joint positions."""
    if len(env_ids) == 0:
        return

    for robot_name in ["robot_A", "robot_B"]:
        try:
            robot = env.scene[robot_name]
            robot.write_joint_state_to_sim(
                position=robot.data.default_joint_pos[env_ids],
                velocity=robot.data.default_joint_vel[env_ids],
                env_ids=env_ids,
            )
        except KeyError:
            pass

    env._reset_game_state(env_ids)


def serve_ball_alternating(
    env: "PingPongEnv",
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Serve the ball from alternating sides with randomized velocity.

    Velocity randomization matches Isaaclab-TableTennisRobot:
      X speed: -1 to +1 m/s
      Y speed: 3.5 to 5 m/s (toward opponent)
      Z speed: 2.0 to 2.2 m/s (upward arc)
      X position: -0.2 to +0.2 m noise
    """
    global _serve_side

    num_resets = len(env_ids)
    if num_resets == 0:
        return

    ball = env.scene[ball_cfg.name]
    env_origins = env.scene.env_origins[env_ids].to(env.device)

    cfg = env.cfg

    # Alternate serving side
    if _serve_side is None or _serve_side == "B":
        _serve_side = "A"
    else:
        _serve_side = "B"

    # Ball spawn position: near the server's side, OUTSIDE the table zones
    # (zones are y∈[-1.35, -0.1] and y∈[0, 1.36] — spawn outside these)
    # Serve from A side means ball at y=-1.5 (behind neg zone), velocity toward +Y (toward B)
    # Serve from B side means ball at y=+1.5 (behind pos zone), velocity toward -Y (toward A)
    if _serve_side == "A":
        y_spawn = -1.5
        vy_mult = 1.0  # velocity toward +Y (toward B)
    else:
        y_spawn = 1.5
        vy_mult = -1.0  # velocity toward -Y (toward A)

    x_noise = torch.empty(num_resets, 1, device=env.device).uniform_(
        *cfg.ball_pos_x_range
    )
    x_local = x_noise.squeeze(-1) + env_origins[:, 0]
    y_local = torch.full((num_resets,), y_spawn, device=env.device) + env_origins[:, 1]
    z_local = torch.full((num_resets,), 1.0, device=env.device) + env_origins[:, 2]

    pos_global = torch.stack([x_local, y_local, z_local], dim=1)
    identity_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        .unsqueeze(0)
        .expand(num_resets, -1)
    )

    v_x = (
        torch.empty(num_resets, 1, device=env.device)
        .uniform_(*cfg.ball_speed_x_range)
        .squeeze(-1)
    )
    v_y = (
        torch.empty(num_resets, 1, device=env.device)
        .uniform_(*cfg.ball_speed_y_range)
        .squeeze(-1)
        * vy_mult
    )
    v_z = (
        torch.empty(num_resets, 1, device=env.device)
        .uniform_(*cfg.ball_speed_z_range)
        .squeeze(-1)
    )

    lin_vel = torch.stack([v_x, v_y, v_z], dim=1)
    ang_vel = torch.zeros(num_resets, 3, device=env.device)

    ball.write_root_pose_to_sim(
        torch.cat([pos_global, identity_quat], dim=1), env_ids=env_ids
    )
    ball.write_root_velocity_to_sim(
        torch.cat([lin_vel, ang_vel], dim=1), env_ids=env_ids
    )
