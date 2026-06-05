"""Observation functions for the ping pong dual-arm environment.

Key observations for competitive ping pong:
  - Robot joint positions and velocities (12 DOF per robot, both arms)
  - End-effector poses (position + Euler angles)
  - Ball position and velocity
"""

import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion (w, x, y, z) to ZYX Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr, cosr)
    sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny, cosy)
    return torch.stack([roll, pitch, yaw], dim=-1)


def robot_joint_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_A"),
) -> torch.Tensor:
    """Joint positions for a specified robot articulation.

    Returns:
        (num_envs, num_joints)
    """
    robot = env.scene[robot_cfg.name]
    if robot_cfg.joint_names is None:
        raise ValueError("robot_joint_positions: robot_cfg must specify joint_names")
    joint_ids, _ = robot.find_joints(robot_cfg.joint_names)
    return robot.data.joint_pos[:, joint_ids].to(env.device)


def robot_joint_velocities(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_A"),
) -> torch.Tensor:
    """Joint velocities for a specified robot articulation.

    Returns:
        (num_envs, num_joints)
    """
    robot = env.scene[robot_cfg.name]
    if robot_cfg.joint_names is None:
        raise ValueError("robot_joint_velocities: robot_cfg must specify joint_names")
    joint_ids, _ = robot.find_joints(robot_cfg.joint_names)
    return robot.data.joint_vel[:, joint_ids].to(env.device)


def ee_poses(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot_A", body_names=["right_wrist_3_link"]
    ),
) -> torch.Tensor:
    """End-effector pose in environment-local coordinates.

    Returns:
        (num_envs, 6) — [pos(3), roll, pitch, yaw]
    """
    robot = env.scene[ee_cfg.name]
    body_ids, _ = robot.find_bodies(ee_cfg.body_names)
    body_pos = robot.data.body_pos_w[:, body_ids[0]]
    pos = body_pos - env.scene.env_origins.to(body_pos.device)
    quat = robot.data.body_quat_w[:, body_ids[0]]
    euler = _quat_to_euler_xyz(quat)
    return torch.cat([pos, euler], dim=-1)


def ball_state(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position and velocity in environment-local coordinates.

    Returns:
        (num_envs, 9) — [pos(3), lin_vel(3), ang_vel(3)]
    """
    ball = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w - env.scene.env_origins.to(ball.data.root_pos_w.device)
    lin_vel = ball.data.root_lin_vel_w
    ang_vel = ball.data.root_ang_vel_w
    return torch.cat([pos, lin_vel, ang_vel], dim=-1)


def ball_to_robot_relative(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot_A", body_names=["body_base_link"]
    ),
) -> torch.Tensor:
    """Ball position relative to a robot's base link.

    Returns:
        (num_envs, 3) — ball position relative to robot base
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_body_cfg.name]
    body_ids, _ = robot.find_bodies(robot_body_cfg.body_names)
    dev = ball.data.root_pos_w.device
    env_origins = env.scene.env_origins.to(dev)
    ball_pos = ball.data.root_pos_w - env_origins
    robot_base = robot.data.body_pos_w[:, body_ids[0]] - env_origins
    return ball_pos - robot_base


def ball_projected_state(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball state projected to a 2D table-plane view.

    Returns:
        (num_envs, 3) — [x, y, z_velocity]
    """
    ball = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w - env.scene.env_origins.to(ball.data.root_pos_w.device)
    vel = ball.data.root_lin_vel_w
    return torch.cat([pos[..., :2], vel[..., 2:3]], dim=-1)
