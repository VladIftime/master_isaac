"""Termination functions for dual-arm environment boundary checks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg_left: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_3_link"),
    asset_cfg_right: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_3_link"),
    table_z: float = 0.0,
    margin: float = -0.05,
    x_range: tuple[float, float] = (-0.8, 0.8),
    y_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """
    Terminate when an end-effector leaves the safe workspace.

    The z margin is intentionally negative so that light contact with the table
    surface (Z ≈ 0) does not immediately terminate the episode — only genuine
    table penetration (Z < margin) does.  The x/y bounds are intentionally wider
    than the IK action limits so that natural wrist overshoot is absorbed before
    triggering termination.

    Args:
        env: The RL environment.
        asset_cfg_left: Config for the left wrist link used as the end-effector proxy.
        asset_cfg_right: Config for the right wrist link.
        table_z: Table surface Z coordinate in the local frame.
        margin: Termination fires when ee_z < table_z + margin.
        x_range: Safe local x-range for end-effectors.
        y_range: Safe local y-range for end-effectors.

    Returns:
        Boolean tensor (num_envs,) — True where the episode should end.
    """
    robot = env.scene[asset_cfg_left.name]

    left_body  = asset_cfg_left.body_names  or "left_wrist_3_link"
    right_body = asset_cfg_right.body_names or "right_wrist_3_link"

    left_ids,  _ = robot.find_bodies(left_body)
    right_ids, _ = env.scene[asset_cfg_right.name].find_bodies(right_body)

    left_ee  = robot.data.body_pos_w[:, left_ids[0],  :] - env.scene.env_origins
    right_ee = env.scene[asset_cfg_right.name].data.body_pos_w[:, right_ids[0], :] - env.scene.env_origins

    z_violation = (left_ee[:, 2] < table_z + margin) | (right_ee[:, 2] < table_z + margin)
    x_violation = (
        (left_ee[:, 0]  < x_range[0]) | (left_ee[:, 0]  > x_range[1]) |
        (right_ee[:, 0] < x_range[0]) | (right_ee[:, 0] > x_range[1])
    )
    y_violation = (
        (left_ee[:, 1]  < y_range[0]) | (left_ee[:, 1]  > y_range[1]) |
        (right_ee[:, 1] < y_range[0]) | (right_ee[:, 1] > y_range[1])
    )

    out_of_bounds = z_violation | x_violation | y_violation

    if out_of_bounds.any():
        print(f"[Termination] Robot out of bounds: {out_of_bounds.sum().item()} envs")

    return out_of_bounds


def objects_out_of_bounds(
    env: ManagerBasedRLEnv,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-0.5, 1.5),
    z_min: float = -0.2,
) -> torch.Tensor:
    """
    Terminate when any tracked object leaves the table workspace.

    Objects missing from the scene are silently skipped so the function
    is robust to different scene configurations.

    Args:
        env: The RL environment.
        x_range: Valid local x-range for objects.
        y_range: Valid local y-range for objects.
        z_min: Minimum world-z height; objects below this have fallen off the table.

    Returns:
        Boolean tensor (num_envs,) — True where the episode should end.
    """
    object_names = ["target_object", "cube", "cylinder", "rect", "triangle"]
    out_of_bounds = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for obj_name in object_names:
        try:
            obj = env.scene[obj_name]
            pos_w = obj.data.root_pos_w
            pos_local = pos_w - env.scene.env_origins

            out_of_bounds |= (
                (pos_local[:, 0] < x_range[0]) | (pos_local[:, 0] > x_range[1]) |
                (pos_local[:, 1] < y_range[0]) | (pos_local[:, 1] > y_range[1]) |
                (pos_w[:, 2] < z_min)
            )
        except KeyError:
            pass

    if out_of_bounds.any():
        print(f"[Termination] Objects out of bounds: {out_of_bounds.sum().item()} envs")

    return out_of_bounds
