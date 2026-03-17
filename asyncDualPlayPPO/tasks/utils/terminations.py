"""Termination functions for dual-arm environment boundary checks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_3_link"),
    table_z: float = 0.0,
    margin: float = -0.05,
    x_range: tuple[float, float] = (-0.8, 0.8),
    y_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """
    Terminate when the end-effector leaves the safe workspace.

    Args:
        env: The RL environment.
        asset_cfg: Config for the wrist link used as the end-effector proxy.
        table_z: Table surface Z coordinate in the local frame.
        margin: Termination fires when ee_z < table_z + margin.
        x_range: Safe local x-range for end-effector.
        y_range: Safe local y-range for end-effector.

    Returns:
        Boolean tensor (num_envs,) — True where the episode should end.
    """
    robot = env.scene[asset_cfg.name]
    body_names = asset_cfg.body_names or "wrist_3_link"
    body_ids, _ = robot.find_bodies(body_names)

    ee_pos = robot.data.body_pos_w[:, body_ids[0], :] - env.scene.env_origins

    z_violation = (ee_pos[:, 2] < table_z + margin)
    x_violation = (ee_pos[:, 0] < x_range[0]) | (ee_pos[:, 0] > x_range[1])
    y_violation = (ee_pos[:, 1] < y_range[0]) | (ee_pos[:, 1] > y_range[1])

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
