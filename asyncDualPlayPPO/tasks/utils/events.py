"""Event handlers for dual-arm environment resets."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_object_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (-0.3, 0.3),
    y_range: tuple[float, float] = (0.55, 0.65),
    z_height: float = 0.03,
) -> None:
    """
    Randomize positions of all objects within the workspace on episode reset.

    Objects are placed within (x_range, y_range) with a fixed z_height and a
    random yaw orientation. Objects missing from the scene are silently skipped.

    Args:
        env: The RL environment.
        env_ids: Indices of environments to reset.
        x_range: Min/max local x-position for randomization.
        y_range: Min/max local y-position for randomization.
        z_height: Fixed local z-height for all objects.
    """
    object_names = ["target_object", "cube", "cylinder", "rect", "triangle"]
    num_resets = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    for obj_name in object_names:
        try:
            obj = env.scene[obj_name]

            x_local = torch.rand(num_resets, device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
            y_local = torch.rand(num_resets, device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
            z_local = torch.full((num_resets,), z_height, device=env.device)

            yaw = torch.rand(num_resets, device=env.device) * 2 * torch.pi
            quat = torch.stack([
                torch.cos(yaw / 2),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(yaw / 2),
            ], dim=1)

            pos_global = torch.stack([
                x_local + env_origins[:, 0],
                y_local + env_origins[:, 1],
                z_local + env_origins[:, 2],
            ], dim=1)

            obj.write_root_pose_to_sim(torch.cat([pos_global, quat], dim=1), env_ids=env_ids)

        except KeyError:
            pass


def reset_objects_to_fixed_safe_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """
    Reset objects to a fixed, collision-free resting position.

    Used between phases (Alice→Bob, Bob→Alice) and on episode failure to
    guarantee a clean starting state. Objects missing from the scene are
    silently skipped.

    Args:
        env: The RL environment.
        env_ids: Indices of environments to reset.
    """
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    print(f"[Reset] Resetting objects to safe pose for {num_resets} envs")

    env_origins = env.scene.env_origins[env_ids]
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    zero_vel = torch.zeros(num_resets, 6, device=env.device)

    placements = {
        "target_object": torch.tensor([-0.15, 0.5, 0.05], device=env.device),
        "cube":          torch.tensor([-0.25, 0.5, 0.05], device=env.device),
    }

    for obj_name, local_pos in placements.items():
        try:
            obj = env.scene[obj_name]
            pos = local_pos.unsqueeze(0).expand(num_resets, -1) + env_origins
            quat = identity_quat.unsqueeze(0).expand(num_resets, -1)
            obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        except KeyError:
            pass


def reset_robot_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """
    Reset both arms to their default joint configuration.

    Args:
        env: The RL environment.
        env_ids: Indices of environments to reset.
    """
    if len(env_ids) == 0:
        return

    print(f"[Reset] Resetting robot joints for {len(env_ids)} envs")

    try:
        robot = env.scene["robot"]
        robot.write_joint_state_to_sim(
            position=robot.data.default_joint_pos[env_ids],
            velocity=robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
    except KeyError:
        print("[Warning] 'robot' asset not found — skipping joint reset")
