"""Event handlers for ping pong environment resets."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sim_device(env, tensor=None):
    """Get the simulation device for a tensor-aware operation."""
    if tensor is not None and hasattr(tensor, 'device'):
        return tensor.device
    return env.device


def _write_root_pose(rigid_object, root_pose, env_ids, env):
    """Write root pose with device handling for GPU pipeline."""
    dev = rigid_object.data.root_link_pose_w.device
    rigid_object.write_root_pose_to_sim(root_pose.to(dev), env_ids=env_ids)


def _write_root_vel(rigid_object, root_vel, env_ids, env):
    """Write root velocity with device handling for GPU pipeline."""
    dev = rigid_object.data.root_link_pose_w.device
    rigid_object.write_root_velocity_to_sim(root_vel.to(dev), env_ids=env_ids)


def _write_joints(robot, joint_pos, joint_vel, env_ids, env):
    """Write joint state with device handling for GPU pipeline."""
    dev = robot.data.root_link_pose_w.device
    robot.write_joint_state_to_sim(joint_pos.to(dev), joint_vel.to(dev), env_ids=env_ids)


def reset_all_scene(env, env_ids):
    """Reset all scene objects — pose only, no velocity (velocity set by serve)."""
    for rigid_object in env.scene.rigid_objects.values():
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        env_origins = env.scene.env_origins[env_ids].to(default_root_state.device)
        default_root_state[:, 0:3] += env_origins
        _write_root_pose(rigid_object, default_root_state[:, :7], env_ids, env)
    for articulation_asset in env.scene.articulations.values():
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        env_origins = env.scene.env_origins[env_ids].to(default_root_state.device)
        default_root_state[:, 0:3] += env_origins
        _write_root_pose(articulation_asset, default_root_state[:, :7], env_ids, env)
        _write_joints(articulation_asset,
                      articulation_asset.data.default_joint_pos[env_ids].clone(),
                      articulation_asset.data.default_joint_vel[env_ids].clone(),
                      env_ids, env)


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
    env_origins = env.scene.env_origins[env_ids]

    x_local = (torch.rand(num_resets, device=env.device) - 0.5) * 0.4
    y_local = (torch.rand(num_resets, device=env.device) - 0.5) * 0.3
    z_local = torch.full((num_resets,), 0.3, device=env.device)

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

    vx = (torch.rand(num_resets, device=env.device) - 0.5) * 2.0
    vy = (torch.rand(num_resets, device=env.device) - 0.5) * 3.0
    vz = torch.rand(num_resets, device=env.device) * 1.0 + 0.5
    lin_vel = torch.stack([vx, vy, vz], dim=1)
    ang_vel = torch.zeros(num_resets, 3, device=env.device)

    _write_root_pose(ball, torch.cat([pos_global, quat], dim=1), env_ids, env)
    _write_root_vel(ball, torch.cat([lin_vel, ang_vel], dim=1), env_ids, env)


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
            _write_joints(robot,
                          robot.data.default_joint_pos[env_ids].clone(),
                          robot.data.default_joint_vel[env_ids].clone(),
                          env_ids, env)
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
    env_origins = env.scene.env_origins[env_ids]

    y_serve = -0.5 if server == "A" else 0.5

    pos_global = torch.stack([
        torch.zeros(num_resets, device=env.device) + env_origins[:, 0],
        torch.full((num_resets,), y_serve, device=env.device) + env_origins[:, 1],
        torch.full((num_resets,), 0.15, device=env.device) + env_origins[:, 2],
    ], dim=1)

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(num_resets, -1)
    zero_vel = torch.zeros(num_resets, 6, device=env.device)

    _write_root_pose(ball, torch.cat([pos_global, identity_quat], dim=1), env_ids, env)
    _write_root_vel(ball, zero_vel, env_ids, env)
