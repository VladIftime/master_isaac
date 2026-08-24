"""IK solver configurations for dual-arm ping pong robots.

Provides ready-to-use action config factories for controller types available
in Isaac Lab's controllers API:
  - DiffIK (Differential IK with Damped Least Squares)
  - OSC   (Operational Space Control)
  - RMPflow (Riemannian Motion Policies)
  - cuRobo (GPU nonlinear IK, direct solve_batch)

Usage in env cfg:
    from ik_solvers import build_ik_action

    arm_action_left = build_ik_action("diffik", asset="robot_A", side="right")
"""

from __future__ import annotations

from typing import Literal

from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg

IK_SOLVER_TYPE = Literal["diffik", "osc", "rmpflow", "curobo"]

_DEFAULT_JOINT_PATTERNS = {
    "left":  ["left_shoulder_.*",  "left_elbow_.*",  "left_wrist_.*"],
    "right": ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
}

_DEFAULT_JOINT_NAMES = {
    "left": [
        "left_shoulder_pan_joint", "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint", "left_wrist_2_joint", "left_wrist_3_joint",
    ],
    "right": [
        "right_shoulder_pan_joint", "right_shoulder_lift_joint",
        "right_elbow_joint",
        "right_wrist_1_joint", "right_wrist_2_joint", "right_wrist_3_joint",
    ],
}

_DEFAULT_BODY_NAMES = {
    "left": "left_wrist_3_link",
    "right": "right_wrist_3_link",
}


def get_joint_patterns(side: str) -> list[str]:
    return _DEFAULT_JOINT_NAMES[side]


def get_body_name(side: str) -> str:
    return _DEFAULT_BODY_NAMES[side]


def build_diffik_action(
    asset_name: str,
    joint_names: list[str] | None = None,
    body_name: str | None = None,
    side: str = "right",
    scale: float = 0.15,
    lambda_val: float = 0.1,
) -> ActionTerm:
    joint_names = joint_names or get_joint_patterns(side)
    body_name = body_name or get_body_name(side)
    return DifferentialInverseKinematicsActionCfg(
        asset_name=asset_name,
        joint_names=joint_names,
        body_name=body_name,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": lambda_val},
        ),
        scale=scale,
    )


def build_osc_action(
    asset_name: str,
    joint_names: list[str] | None = None,
    body_name: str | None = None,
    side: str = "right",
    position_scale: float = 0.15,
    orientation_scale: float = 0.15,
    use_relative: bool = True,
) -> ActionTerm:
    from isaaclab.controllers import OperationalSpaceControllerCfg
    from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg

    joint_names = joint_names or get_joint_patterns(side)
    body_name = body_name or get_body_name(side)

    target_type = "pose_rel" if use_relative else "pose_abs"

    return OperationalSpaceControllerActionCfg(
        asset_name=asset_name,
        joint_names=joint_names,
        body_name=body_name,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=[target_type],
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
            impedance_mode="variable_kp",
            motion_stiffness_task=[360, 360, 360, 360, 360, 360],
            motion_damping_ratio_task=1.0,
            nullspace_control="none",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        position_scale=position_scale,
        orientation_scale=orientation_scale,
        nullspace_joint_pos_target="none",
    )


def build_rmpflow_action(
    asset_name: str,
    joint_names: list[str] | None = None,
    body_name: str | None = None,
    side: str = "right",
    config_file: str | None = None,
    urdf_file: str | None = None,
    collision_file: str | None = None,
    articulation_prim_expr: str | None = None,
    scale: float = 0.8,
) -> ActionTerm:
    from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg
    from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
    import os

    joint_names = joint_names or get_joint_patterns(side)
    body_name = body_name or get_body_name(side)

    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if config_file is None:
        config_file = os.path.join(
            _pkg_root, "urdf", "cuMotion", f"rmpflow_config_{side}.yaml"
        )
    if urdf_file is None:
        urdf_file = os.path.join(
            _pkg_root, "assets", "urdf", "ur_robotics", "ur5e", "ur5e_robotiq_140.urdf"
        )
    if collision_file is None:
        collision_file = os.path.join(
            _pkg_root, "urdf", "cuMotion", f"lula_{side}.yaml"
        )
    if articulation_prim_expr is None:
        articulation_prim_expr = f"/World/envs/env_.*/{asset_name}"

    return RMPFlowActionCfg(
        asset_name=asset_name,
        joint_names=joint_names,
        body_name=body_name,
        articulation_prim_expr=articulation_prim_expr,
        use_relative_mode=True,
        controller=RmpFlowControllerCfg(
            config_file=config_file,
            urdf_file=urdf_file,
            collision_file=collision_file,
            frame_name=body_name,
            evaluations_per_frame=1.0,
        ),
        scale=scale,
    )


def build_curobo_action(
    asset_name: str,
    joint_names: list[str] | None = None,
    body_name: str | None = None,
    side: str = "right",
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.05,
    num_seeds: int = 10,
    **kwargs,
) -> ActionTerm:
    from ik_solvers.curobo_ik import CuroboIKActionCfg

    joint_names = joint_names or get_joint_patterns(side)
    body_name = body_name or get_body_name(side)

    return CuroboIKActionCfg(
        asset_name=asset_name,
        joint_names=joint_names,
        body_name=body_name,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
        num_seeds=num_seeds,
    )


IK_BUILDERS = {
    "diffik": build_diffik_action,
    "osc": build_osc_action,
    "rmpflow": build_rmpflow_action,
    "curobo": build_curobo_action,
}


def build_ik_action(
    solver_type: IK_SOLVER_TYPE,
    asset_name: str,
    side: str = "right",
    **kwargs,
) -> ActionTerm:
    if solver_type not in IK_BUILDERS:
        raise ValueError(
            f"Unknown IK solver '{solver_type}'. Options: {list(IK_BUILDERS)}"
        )
    return IK_BUILDERS[solver_type](asset_name=asset_name, side=side, **kwargs)
