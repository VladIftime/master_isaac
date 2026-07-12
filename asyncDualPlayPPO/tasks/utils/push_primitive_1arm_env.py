# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base scene and environment configuration for 1-arm push tasks.

Provides the shared robot, table, workspace border markers, objects,
terminations, events, and action skeleton used by both train_push.py
and train_curobo.py.  Each training script overrides the arm action
at runtime (JointPositionActionCfg for cuRobo).
"""

from dataclasses import MISSING
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UrdfFileCfg,
    UsdFileCfg,
)
from isaaclab.utils import configclass

from . import terminations
import isaaclab.envs.mdp as mdp

# Resolve the project root relative to this file.
_PROJ_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
ISAACLAB_DUAL_ARM_EXT_DIR = _PROJ_ROOT


##
# Robot configuration — single UR5e with Robotiq 140 gripper
##

UR5e_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/urdf/ur_robotics/ur5e/ur5e_robotiq_140.urdf",
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=50.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 1.57,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "left_inner_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
            stiffness=5000.0,
            damping=200.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=500.0,
            damping=100.0,
        ),
        "manual_mimics": ImplicitActuatorCfg(
            joint_names_expr=[".*knuckle_joint", ".*inner_finger_joint"],
            stiffness=500.0,
            damping=20.0,
        ),
    },
)


##
# Visual-only body (torso + head + LEFT arm).  The single push arm (RobotUnified)
# visually serves as this body's RIGHT arm.  The right arm + right gripper have been
# removed from the URDF and all collisions stripped, so this is purely decorative and
# never interacts with the push arm, objects, or IK.
#
# The LEFT arm pose is baked directly into the URDF as *fixed* joints (crane pose,
# base turned +90 deg CCW so it points left), so the body is a fully rigid static
# prop that cannot drift back to a default pose.  It is spawned as a plain AssetBase
# (not an Articulation) so nothing drives or manages its joints.
#
# Placement: the body's (removed) right-arm socket lands on the push arm base at the
# world origin (0,0,0).  From the URDF, right_base = body_root + (0.255, 0.05, 0.030)
# after the built-in Rz+90 body rotation, so the body root sits at the negative of that.
##

_BODY_RIGHT_SOCKET_OFFSET = (0.255, 0.05, 0.030)
_BODY_ROOT_POS = (
    -_BODY_RIGHT_SOCKET_OFFSET[0],
    -_BODY_RIGHT_SOCKET_OFFSET[1],
    -_BODY_RIGHT_SOCKET_OFFSET[2],
)

BODY_DISPLAY_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/BodyDisplay",
    init_state=AssetBaseCfg.InitialStateCfg(pos=_BODY_ROOT_POS),
    spawn=UrdfFileCfg(
        asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/dual_arm_body_display.urdf",
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=50.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        activate_contact_sensors=False,
    ),
)


##
# Scene definition
##

@configclass
class Push1ArmSceneCfg(InteractiveSceneCfg):
    """Scene for 1-arm push tasks: robot, table, objects, workspace borders."""

    robot: ArticulationCfg = UR5e_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotUnified",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=UR5e_CFG.init_state.joint_pos,
        ),
    )

    # ── Visual-only body (torso + head + left arm); the push arm is the right arm ──
    body_display: AssetBaseCfg = BODY_DISPLAY_CFG.replace(
        prim_path="{ENV_REGEX_NS}/BodyDisplay",
    )

    # ── Stand under the body (throwing-env style, visual only) ─────────────────
    body_stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BodyStand",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_BODY_ROOT_POS[0], _BODY_ROOT_POS[1], -0.33], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.32)),
        ),
    )

    # ── Table (kinematic, 2m×2m) ──────────────────────────────────────────────
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.40, -0.05], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.40, 1.00, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
    )

    # ── Workspace border lines (cuboids flat on table, matching EE ws limits) ──
    # EE workspace: X∈[-0.50,+0.50], Y∈[0.25,0.70]
    zone_border_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.70, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.02, 0.02, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.08, 0.08)
            ),
        ),
    )
    zone_border_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderBottom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.25, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.02, 0.02, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.08, 0.08)
            ),
        ),
    )
    zone_border_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.50, 0.475, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.47, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.08, 0.08)
            ),
        ),
    )
    zone_border_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.50, 0.475, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.47, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.08, 0.08)
            ),
        ),
    )

    # ── Task objects — physical T-block (overridden by training scripts) ──────
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.5, 0.05],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/t_shape.usda",
            scale=(2.0, 2.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10000.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.25, 0.7, 0.05],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cube.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
        ),
    )
    cylinder = None
    rect = None
    triangle = None

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action skeleton — arm_action is overridden at runtime for cuRobo.

    Total action dim = arm(6) + gripper(1) = 7.
    """

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=1.0,
        use_default_offset=False,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "finger_joint",
            "left_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint",
            "right_inner_finger_joint",
        ],
        open_command_expr={
            "finger_joint": 0.0,
            "left_inner_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
        },
        close_command_expr={
            "finger_joint": 0.8,
            "left_inner_knuckle_joint": 0.8,
            "left_inner_finger_joint": -0.8,
            "right_outer_knuckle_joint": 0.8,
            "right_inner_knuckle_joint": 0.8,
            "right_inner_finger_joint": -0.8,
        },
    )


@configclass
class EventCfg:
    """Default events — reset scene to defaults on episode start."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Default termination terms."""

    robot_through_table = DoneTerm(
        func=terminations.robot_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),
            "table_z": 0.0,
            "margin": -0.05,
            "x_range": (-1.0, 1.0),
            "y_range": (-1.0, 1.0),
        },
    )

    objects_off_table = DoneTerm(
        func=terminations.objects_out_of_bounds,
        params={
            "x_range": (-1.0, 1.0),
            "y_range": (-0.5, 1.5),
            "z_min": -0.2,
        },
    )


@configclass
class Push1ArmEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration for 1-arm push tasks."""

    scene: Push1ArmSceneCfg = Push1ArmSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 20.0
        self.sim.dt = 0.02
        self.sim.use_fabric = True
        self.sim.render_interval = self.decimation
        self.reset_settle_steps = 5
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4
