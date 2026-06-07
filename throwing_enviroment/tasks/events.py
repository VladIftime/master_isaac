"""Event handlers for throwing environment resets.

  - reset_robot_joints: reset robot to default joint positions, open gripper
  - randomize_target_position: randomize target position on table
  - attach_milk_to_gripper: move milk to gripper, close gripper
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .throwing_env import ThrowingEnv


def reset_robot_joints(
    env: "ThrowingEnv",
    env_ids: torch.Tensor,
) -> None:
    """Reset robot to default joint positions with gripper open."""
    if len(env_ids) == 0:
        return

    robot = env.scene["robot"]
    robot.write_joint_state_to_sim(
        position=robot.data.default_joint_pos[env_ids],
        velocity=robot.data.default_joint_vel[env_ids],
        env_ids=env_ids,
    )

    env._reset_game_state(env_ids)


def randomize_target_position(
    env: "ThrowingEnv",
    env_ids: torch.Tensor,
) -> None:
    """Randomize target position on the table."""
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    target = env.scene["target"]
    env_origins = env.scene.env_origins[env_ids].to(env.device)

    cfg = env.cfg

    target_x = torch.empty(num_resets, device=env.device).uniform_(*cfg.target_x_range)
    target_y = torch.empty(num_resets, device=env.device).uniform_(*cfg.target_y_range)
    target_z = torch.full((num_resets,), cfg.target_z, device=env.device)
    target_pos = torch.stack([
        target_x + env_origins[:, 0],
        target_y + env_origins[:, 1],
        target_z + env_origins[:, 2],
    ], dim=-1)
    target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(num_resets, -1)
    target.write_root_pose_to_sim(torch.cat([target_pos, target_quat], dim=1), env_ids=env_ids)


def attach_milk_to_gripper(
    env: "ThrowingEnv",
    env_ids: torch.Tensor,
) -> None:
    """Spawn milk between closed-gripper fingers, close gripper, mark as holding.

    Uses physics gripping (Option B): bottle held by contact friction between
    finger pads, not by kinematic pose writes.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene["robot"]
    milk = env.scene["milk"]

    body_ids, _ = robot.find_bodies([env._ee_body])
    ee_pos = robot.data.body_pos_w[env_ids, body_ids[0]]
    ee_quat = robot.data.body_quat_w[env_ids, body_ids[0]]

    # Offset bottle root so its center (~z=0.107) sits at finger pinch point
    # Pinch point world offset from wrist at home pose: (-0.012, 0.129, -0.069)
    bottle_center_from_root = 0.107
    pinch_z = -0.069
    bottle_offset = torch.tensor(
        [-0.012, 0.129, pinch_z - bottle_center_from_root],
        device=env.device,
    )
    bottle_root = ee_pos + bottle_offset.unsqueeze(0)
    milk_pose = torch.cat([bottle_root, ee_quat], dim=-1)
    milk.write_root_pose_to_sim(milk_pose, env_ids=env_ids)
    milk.write_root_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )

    # Close gripper — fingers squeeze bottle from both sides
    gripper_joint_ids, _ = robot.find_joints([env._gripper_joint])
    gripper_pos_des = torch.full((len(env_ids), 1), 0.7, device=env.device)
    gripper_vel_des = torch.zeros(len(env_ids), 1, device=env.device)
    robot.write_joint_state_to_sim(
        position=gripper_pos_des,
        velocity=gripper_vel_des,
        joint_ids=gripper_joint_ids,
        env_ids=env_ids,
    )
