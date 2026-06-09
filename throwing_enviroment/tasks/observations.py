"""Observation functions for the throwing environment.

Key observations:
  - Throwing arm joint positions and velocities (6 DOF)
  - End-effector pose (position + Euler angles)
  - Object, target, and obstacle positions (env-local)
  - Distance vector from object to target
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    if robot_cfg.joint_names is None:
        raise ValueError("robot_joint_positions: robot_cfg must specify joint_names")
    joint_ids, _ = robot.find_joints(robot_cfg.joint_names)
    return robot.data.joint_pos[:, joint_ids].to(env.device)


def robot_joint_velocities(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    if robot_cfg.joint_names is None:
        raise ValueError("robot_joint_velocities: robot_cfg must specify joint_names")
    joint_ids, _ = robot.find_joints(robot_cfg.joint_names)
    return robot.data.joint_vel[:, joint_ids].to(env.device)


def ee_pose(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_wrist_3_link"]),
) -> torch.Tensor:
    robot = env.scene[ee_cfg.name]
    body_ids, _ = robot.find_bodies(ee_cfg.body_names)
    body_pos = robot.data.body_pos_w[:, body_ids[0]]
    pos = body_pos - env.scene.env_origins.to(body_pos.device)
    quat = robot.data.body_quat_w[:, body_ids[0]]
    euler = _quat_to_euler_xyz(quat)
    return torch.cat([pos, euler], dim=-1)


def object_position(
    env: ManagerBasedRLEnv,
    object_name: str = "milk",
) -> torch.Tensor:
    obj = env.scene[object_name]
    dev = obj.data.root_pos_w.device
    pos = obj.data.root_pos_w - env.scene.env_origins.to(dev)
    return pos


def dist_to_target(
    env: ManagerBasedRLEnv,
    object_name: str = "milk",
    target_name: str = "target",
) -> torch.Tensor:
    milk = env.scene[object_name]
    target = env.scene[target_name]
    dev = milk.data.root_pos_w.device
    env_origins = env.scene.env_origins.to(dev)
    milk_pos = milk.data.root_pos_w - env_origins
    target_pos = target.data.root_pos_w - env_origins
    return target_pos - milk_pos
