"""Event handlers for ping pong environment resets."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def serve_ball_random(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Serve the ball from a random position near the center of the table."""
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    ball = env.scene[ball_cfg.name]
    env_origins = env.scene.env_origins[env_ids].to(env.device)

    x_local = (torch.rand(num_resets, device=env.device) - 0.5) * 0.3
    y_local = (torch.rand(num_resets, device=env.device) - 0.5) * 0.2
    z_local = torch.full((num_resets,), 0.28, device=env.device)

    pos_global = torch.stack([
        x_local + env_origins[:, 0],
        y_local + env_origins[:, 1],
        z_local + env_origins[:, 2],
    ], dim=1)

    yaw = torch.rand(num_resets, device=env.device) * 2 * torch.pi
    quat = torch.stack([
        torch.cos(yaw / 2.0),
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        torch.sin(yaw / 2.0),
    ], dim=1)

    vx = (torch.rand(num_resets, device=env.device) - 0.5) * 0.5
    vy = (torch.rand(num_resets, device=env.device) - 0.5) * 0.8
    vz = torch.rand(num_resets, device=env.device) * 0.3 + 0.3
    lin_vel = torch.stack([vx, vy, vz], dim=1)
    ang_vel = torch.zeros(num_resets, 3, device=env.device)

    ball.write_root_pose_to_sim(torch.cat([pos_global, quat], dim=1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=1), env_ids=env_ids)


def reset_robot_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset both robots to their ready positions."""
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


def reset_ball_to_serve(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    server: str = "A",
) -> None:
    """Reset the ball to a serving position for the specified robot."""
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    ball = env.scene["ball"]
    env_origins = env.scene.env_origins[env_ids].to(env.device)

    y_serve = -0.5 if server == "A" else 0.5

    pos_global = torch.stack([
        torch.zeros(num_resets, device=env.device) + env_origins[:, 0],
        torch.full((num_resets,), y_serve, device=env.device) + env_origins[:, 1],
        torch.full((num_resets,), 0.15, device=env.device) + env_origins[:, 2],
    ], dim=1)

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(num_resets, -1)
    zero_vel = torch.zeros(num_resets, 6, device=env.device)

    ball.write_root_pose_to_sim(torch.cat([pos_global, identity_quat], dim=1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
