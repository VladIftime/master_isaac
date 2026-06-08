"""Event handlers for throwing environment resets.

  - reset_robot_joints: reset robot to default joint positions, open gripper
  - randomize_target_position: randomize target position on table
  - attach_milk_to_gripper: move milk to gripper, close gripper
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate

if TYPE_CHECKING:
    from .throwing_env import ThrowingEnv

# Right gripper: joint name patterns and their mimic multiplier relative to finger_joint.
# The Robotiq 2F-140 has 6 revolute joints; the finger_joint drives the left outer knuckle
# directly, and 5 mimic joints couple the other finger segments.  PhysX mimic constraints
# do not resolve instantly on write_joint_state_to_sim, so we must set all joints explicitly.
_GRIPPER_R_PATTERNS = [
    "rgripper_finger_joint",
    "rgripper_.*_knuckle_joint$",
    "rgripper_.*_inner_finger_joint$",
]
# Multipliers keyed by suffix to the full joint name
_GRIPPER_R_MULT = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": -1.0,
    "left_inner_finger_joint": 1.0,
    "right_outer_knuckle_joint": -1.0,
    "right_inner_knuckle_joint": -1.0,
    "right_inner_finger_joint": 1.0,
}


def _set_gripper_state(robot, finger_target: float, env_ids: torch.Tensor):
    """Write all right-gripper revolute joints to positions derived from finger_target."""
    n = len(env_ids)
    gripper_ids, gripper_names = robot.find_joints(_GRIPPER_R_PATTERNS)

    positions = torch.zeros(n, len(gripper_names), device=robot.device)
    for i, name in enumerate(gripper_names):
        suffix = name
        mult = next(
            (v for k, v in _GRIPPER_R_MULT.items() if name.endswith(k)), 1.0
        )
        positions[:, i] = finger_target * mult

    zeros_vel = torch.zeros_like(positions)
    robot.write_joint_state_to_sim(positions, zeros_vel, joint_ids=gripper_ids, env_ids=env_ids)
    robot.set_joint_position_target(positions, joint_ids=gripper_ids, env_ids=env_ids)


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

    if not env.cfg.randomize_target:
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
    if env.cfg.disable_attachment:
        return

    robot = env.scene["robot"]
    milk = env.scene["milk"]

    body_ids, _ = robot.find_bodies([env._ee_body])
    ee_pos = robot.data.body_pos_w[env_ids, body_ids[0]]
    ee_quat = robot.data.body_quat_w[env_ids, body_ids[0]]

    # Offset bottle root so its center sits at finger pinch point, in EE-local frame
    bottle_offset_local = torch.tensor(
        [-0.012, 0.129, -0.176], device=env.device,
    ).unsqueeze(0).expand(len(env_ids), -1)
    bottle_offset_world = quat_rotate(ee_quat, bottle_offset_local)
    bottle_root = ee_pos + bottle_offset_world
    milk_pose = torch.cat([bottle_root, ee_quat], dim=-1)
    milk.write_root_pose_to_sim(milk_pose, env_ids=env_ids)
    milk.write_root_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )

    # Close gripper — all 6 revolute joints set for both fingers
    _set_gripper_state(robot, 0.7, env_ids)
