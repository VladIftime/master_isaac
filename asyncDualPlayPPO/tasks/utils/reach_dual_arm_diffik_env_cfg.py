# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# DifferentialIK variant of reach_dual_arm_env_cfg.py.
# Identical except ActionsCfg.arm_action uses DifferentialInverseKinematicsActionCfg
# instead of RMPFlowActionCfg.  No collision YAML files required.
# All other classes (scene, observations, events, terminations, base env cfg) are
# unchanged copies so that both variants stay in sync when the shared parts change.

from dataclasses import MISSING
import math
import os
import re
import torch
import numpy as np
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd

# Define block variety
BLOCK_FILES = ["concave.usd", "cube.usd", "cylinder.usd", "rect.usd", "triangle.usd"]

def spawn_random_block(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None):
    """
    Custom spawner that picks a USD file based on the environment index.
    
    Ensures that different environments have different shapes, providing variety 
    across the parallel training batch without breaking the physics tensor backend.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.stage import get_current_stage
    
    # Base directory for blocks (infer from existing path)
    base_dir = os.path.dirname(cfg.usd_path)
    
    # If the prim_path is a regex (e.g. contains .*), we need to spawn for all environments
    if ".*" in prim_path:
        stage = get_current_stage()
        # Find all environments already created by the cloner
        env_folder = stage.GetPrimAtPath("/World/envs")
        if not env_folder.IsValid():
            # Fallback if envs are not under /World/envs
            return spawn_from_usd(prim_path, cfg, translation, orientation)
            
        env_names = [child.GetName() for child in env_folder.GetChildren() if child.GetName().startswith("env_")]
        if not env_names:
            return spawn_from_usd(prim_path, cfg, translation, orientation)
            
        # Spawn one for each environment
        for env_name in env_names:
            # Extract index
            try:
                env_idx = int(env_name.split("_")[-1])
            except:
                env_idx = 0
            
            # Construct actual path for this env
            actual_path = prim_path.replace(".*", str(env_idx))
            
            # Choose block (different pattern for Cube vs TargetObject)
            offset = 2 if "Cube" in actual_path else 0
            usd_file = BLOCK_FILES[(env_idx + offset) % len(BLOCK_FILES)]
            cfg.usd_path = os.path.join(base_dir, usd_file)
            
            # Spawn
            spawn_from_usd(actual_path, cfg, translation, orientation)
        
        # Return the first one as representative
        return stage.GetPrimAtPath(prim_path.replace(".*", "0"))

    # Single path case (fallback or template)
    match = re.search(r"env_(\d+)", prim_path)
    env_idx = int(match.group(1)) if match else 0
    
    offset = 2 if "Cube" in prim_path else 0
    usd_file = BLOCK_FILES[(env_idx + offset) % len(BLOCK_FILES)]
    cfg.usd_path = os.path.join(base_dir, usd_file)
    
    return spawn_from_usd(prim_path, cfg, translation, orientation)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UrdfFileCfg,
)
from isaaclab.utils import configclass

from . import terminations
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg

_PROJ_ROOT = os.path.dirname(  # dual_arm_Isaacgym/
    os.path.dirname(  # asyncDualPlayPPO/
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tasks/  # utils/
    )
)
ISAACLAB_DUAL_ARM_EXT_DIR = _PROJ_ROOT


##
# Robot configuration  (identical to RMPflow variant)
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

UR5e_Dual_CFG = UR5e_CFG.replace(
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/dual_arm_robot_no_gripper_col.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Left arm
            "left_shoulder_pan_joint": 0.0,
            "left_shoulder_lift_joint": -1.57,
            "left_elbow_joint": -1.57,
            "left_wrist_1_joint": -1.57,
            "left_wrist_2_joint": 1.57,
            "left_wrist_3_joint": 1.57,
            "lgripper_finger_joint": 0.0,
            "lgripper_left_inner_finger_joint": 0.0,
            "lgripper_right_inner_finger_joint": 0.0,
            # Right arm
            "right_shoulder_pan_joint": 0.0,
            "right_shoulder_lift_joint": -1.57,
            "right_elbow_joint": 1.57,
            "right_wrist_1_joint": -1.57,
            "right_wrist_2_joint": -1.57,
            "right_wrist_3_joint": 1.57,
            "rgripper_finger_joint": 0.0,
            "rgripper_left_inner_finger_joint": 0.0,
            "rgripper_right_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm_left": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
            stiffness=5000.0,
            damping=200.0,
        ),
        "arm_right": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
            stiffness=5000.0,
            damping=200.0,
        ),
        "gripper_left": ImplicitActuatorCfg(
            joint_names_expr=["lgripper_finger_joint"],
            stiffness=500.0,
            damping=100.0,
        ),
        "gripper_right": ImplicitActuatorCfg(
            joint_names_expr=["rgripper_finger_joint"],
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
# Scene definition  (identical to RMPflow variant)
##


@configclass
class ReachDualArmSceneCfg(InteractiveSceneCfg):
    """Configuration for the dual-arm reach scene."""

    replicate_physics: bool = False

    robot: ArticulationCfg = UR5e_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotUnified",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=UR5e_CFG.init_state.joint_pos,
        ),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.5, -0.05], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.1),
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

    zone_border_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 1.0, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.52, 0.02, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05)
            ),
        ),
    )

    zone_border_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderBottom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.2, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.52, 0.02, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05)
            ),
        ),
    )

    zone_border_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.75, 0.6, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.82, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05)
            ),
        ),
    )

    zone_border_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.75, 0.6, 0.001], rot=[1.0, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.82, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05)
            ),
        ),
    )

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.7, 0.05],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=UsdFileCfg(
            func=spawn_random_block, # Use custom randomizer
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cylinder.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.15, 0.7, 0.05],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=UsdFileCfg(
            func=spawn_random_block, # Use custom randomizer
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cube.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
    )

    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.05, 0.5, 0.05], rot=[0.0, 0.0, 0.0, 1.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cylinder.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.4)),
        ),
    )

    rect = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rect",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.05, 0.5, 0.05], rot=[0.0, 0.0, 0.0, 1.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/rect.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.2)),
        ),
    )

    triangle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Triangle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.2, 0.6, 0.05], rot=[0.0, 0.0, 0.0, 1.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/triangle.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.8)),
        ),
    )

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
    """Action specifications for the MDP — DifferentialIK variant.

    Arm action uses DLS Jacobian IK in relative-pose mode:
      command_type="pose", use_relative_mode=True  →  6D [dx,dy,dz,droll,dpitch,dyaw]
    This matches RMPflow's 6D delta interface so the policy network shape is the same.

    scale=0.5 is conservative vs RMPflow's 0.8; tune upward if motion is too slow.
    No YAML collision files needed — DiffIK is purely Jacobian-based.

    Gripper action is identical to the RMPflow variant.
    """

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"],
        body_name="wrist_3_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        ),
        scale=0.15,
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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        object_pos = ObsTerm(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("target_object")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

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
class ReachDualArmDiffIKEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for the dual-arm reach environment — DifferentialIK variant."""

    scene: ReachDualArmSceneCfg = ReachDualArmSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
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
