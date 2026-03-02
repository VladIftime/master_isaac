
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math
import os

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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UrdfFileCfg, UsdFileCfg
from isaaclab.utils import configclass

# INTERNAL IMPORTS (Consolidated)
# INTERNAL IMPORTS (Consolidated)
from . import terminations
from . import events
from . import bounded_ik
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg

# Define Asset Path - Fallback to relative if package not found, or hardcoded
# Check for extension existence; fallback to dynamic path based on file location.
# try:
#     from isaaclab_dual_arm import ISAACLAB_DUAL_ARM_EXT_DIR
# except ImportError:
#     # Fallback to dynamic path resolution if extension not installed
#     # This file is at: asyncDualPlayPPO/tasks/utils/reach_dual_arm_env_cfg.py
#     # Project root is: ../../../ from here
import os
_current_file = os.path.abspath(__file__)
_utils_dir = os.path.dirname(_current_file)  # tasks/utils
_tasks_dir = os.path.dirname(_utils_dir)      # tasks
_async_dir = os.path.dirname(_tasks_dir)      # asyncDualPlayPPO
ISAACLAB_DUAL_ARM_EXT_DIR = os.path.dirname(_async_dir)  # dual_arm_Isaacgym






##
# Robot Configuration
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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        # Enable self-collisions to prevent arms from overlapping
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # Prevents arm folding and overlap
            solver_position_iteration_count=8,  # Increased for better stability
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
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
            "left_inner_finger_joint": 0.0,
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
            joint_names_expr=["finger_joint", ".*inner_finger_joint"],
            stiffness=1e6,
            damping=80.0,
        ),
        # [CRITICAL FIX] Add passive actuators for gripper knuckle joints
        # Robotiq has 12 joints total: 6 arm + 3 finger + 3 knuckle (mimics)
        # This fixes warning: "Not all actuators configured! 9 != 12"
        "passive_mimics": ImplicitActuatorCfg(
            joint_names_expr=[".*knuckle_joint"],  # Catches all knuckle joints
            stiffness=0.0,  # Passive (let physics/mimics drive them)
            damping=10.0,   # Small damping for stability
        ),
    },
)

# Dual Arm Configuration (Merged)
UR5e_Dual_CFG = UR5e_CFG.replace(
    spawn=UrdfFileCfg(
        asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/dual_arm_robot_no_gripper_col.urdf",
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=50.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Left Arm
            "left_shoulder_pan_joint": 0.0,
            "left_shoulder_lift_joint": -1.57,
            "left_elbow_joint": -1.57,
            "left_wrist_1_joint": -1.57,
            "left_wrist_2_joint": 1.57,
            "left_wrist_3_joint": 1.57,
            "lgripper_finger_joint": 0.0,
            "lgripper_left_inner_finger_joint": 0.0,
            "lgripper_right_inner_finger_joint": 0.0,
            # Right Arm
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
            stiffness=500.0, # Reduced stiffness
            damping=100.0,   # Increased damping for stability
        ),
        "gripper_right": ImplicitActuatorCfg(
            joint_names_expr=["rgripper_finger_joint"],
            stiffness=500.0,
            damping=100.0,
        ),
        # FIX: Give stiffness to mimics so manual commands work
        "manual_mimics": ImplicitActuatorCfg(
            joint_names_expr=[".*knuckle_joint", ".*inner_finger_joint"],
            stiffness=500.0, # Was 0.0, changed to 500.0 to enable manual driving
            damping=20.0,
        ),
    },
)

##
# Scene definition
##

@configclass
class ReachDualArmSceneCfg(InteractiveSceneCfg):
    """Configuration for the dual arm reach scene."""

    # Single unified robot spawn (includes body + both arms)
    robot: ArticulationCfg = UR5e_Dual_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotUnified",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), # Origin
            # Joint pos from merged config
            joint_pos=UR5e_Dual_CFG.init_state.joint_pos,
        ),
    )

    # Robot Body (Visual/Collision)
    # Robot Body (Visual/Collision)
    # TODO: Integrate robot body assets (head, body_front) and verify rotation conventions.
    




    # Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.5, -0.05], rot=[1.0, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.1),  # 10cm thick to prevent tunneling
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Table doesn't move
                disable_gravity=True,
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

    # Objects (Blocks)
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.5, 0.03], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UrdfFileCfg(
            asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/concave.urdf",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            fix_base=False,
            joint_drive=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),  # Red - Target
        ),
    )
    
    # Distractors
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.5, 0.03], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UrdfFileCfg(
            asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cube.urdf",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            fix_base=False,
            joint_drive=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),  # Blue
        ),
    )
    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.05, 0.5, 0.03], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UrdfFileCfg(
            asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cylinder.urdf",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            fix_base=False,
            joint_drive=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.4)),  # Green
        ),
    )
    rect = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rect",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.05, 0.5, 0.03], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UrdfFileCfg(
            asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/rect.urdf",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            fix_base=False,
            joint_drive=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.2)),  # Yellow
        ),
    )
    triangle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Triangle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.6, 0.03], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UrdfFileCfg(
            asset_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/triangle.urdf",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            fix_base=False,
            joint_drive=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.8)),  # Pink
        ),
    )

    # Camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.1, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.3, 2.8), rot=(0.5, 0.5, 0.5, 0.5), convention="ros"),
    )

    # Plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=GroundPlaneCfg(),
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # HYBRID IK SOLUTION: Position-focused control with minimal rotation
    # This gives Alice "pushing power" without the "exploding/folding" instability
    # Position scale (0.05): Robot can move 5cm per step - enough to push objects
    # Rotation scale (0.01): Nearly disables random twisting - prevents wrist spasms
    
    # LEFT ARM - RMPflow Control
    left_arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
        body_name="left_wrist_3_link",
        controller=RmpFlowControllerCfg(
            config_file=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/cuMotion/rmpflow_config_left.yaml",
            urdf_file=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/dual_arm_robot.urdf",
            collision_file=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/cuMotion/lula_left.yaml",
            frame_name="left_wrist_3_link",
            evaluations_per_frame=12.0,
        ),
        articulation_prim_expr="/World/envs/env_.*/RobotUnified",
        scale=1.0,
    )
    
    # RIGHT ARM - RMPflow Control
    right_arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
        body_name="right_wrist_3_link",
        controller=RmpFlowControllerCfg(
            config_file=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/cuMotion/rmpflow_config_right.yaml",
            urdf_file=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/dual_arm_robot.urdf",
            collision_file=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/urdf/cuMotion/lula_right.yaml",
            frame_name="right_wrist_3_link",
            evaluations_per_frame=12.0,
        ),
        articulation_prim_expr="/World/envs/env_.*/RobotUnified",
        scale=1.0,
    )
    
    # Unified Gripper Action (Controls both grippers with 1 signal)
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        # Target BOTH main finger joints AND MIMICS
        # This explicit control prevents physics engine from failing to drive passive mimics
        joint_names=[
            # Drivers
            "lgripper_finger_joint", 
            "rgripper_finger_joint",
            # Left Mimics
            "lgripper_left_inner_knuckle_joint",
            "lgripper_left_inner_finger_joint",
            "lgripper_right_outer_knuckle_joint",
            "lgripper_right_inner_knuckle_joint",
            "lgripper_right_inner_finger_joint",
            # Right Mimics
            "rgripper_left_inner_knuckle_joint",
            "rgripper_left_inner_finger_joint",
            "rgripper_right_outer_knuckle_joint",
            "rgripper_right_inner_knuckle_joint",
            "rgripper_right_inner_finger_joint",
        ],
        
        # When command is "Open" (approx -1), set all to 0.0
        open_command_expr={
            "lgripper_finger_joint": 0.0,
            "rgripper_finger_joint": 0.0,
            # Left Mimics
            "lgripper_left_inner_knuckle_joint": 0.0,
            "lgripper_left_inner_finger_joint": 0.0,
            "lgripper_right_outer_knuckle_joint": 0.0,
            "lgripper_right_inner_knuckle_joint": 0.0,
            "lgripper_right_inner_finger_joint": 0.0,
            # Right Mimics
            "rgripper_left_inner_knuckle_joint": 0.0,
            "rgripper_left_inner_finger_joint": 0.0,
            "rgripper_right_outer_knuckle_joint": 0.0,
            "rgripper_right_inner_knuckle_joint": 0.0,
            "rgripper_right_inner_finger_joint": 0.0,
        },
        
        # When command is "Close" (approx +1), set driven to 0.8, others based on multiplier
        # Multipliers:
        # inner_knuckle, right_outer_knuckle, right_inner_knuckle -> -1
        # inner_finger -> +1
        close_command_expr={
            "lgripper_finger_joint": 0.8,
            "rgripper_finger_joint": 0.8,
            # Left Mimics
            "lgripper_left_inner_knuckle_joint": -0.8,  # -1
            "lgripper_left_inner_finger_joint": 0.8,    # +1
            "lgripper_right_outer_knuckle_joint": -0.8, # -1
            "lgripper_right_inner_knuckle_joint": -0.8, # -1
            "lgripper_right_inner_finger_joint": 0.8,   # +1
            # Right Mimics
            "rgripper_left_inner_knuckle_joint": -0.8,  # -1
            "rgripper_left_inner_finger_joint": 0.8,    # +1
            "rgripper_right_outer_knuckle_joint": -0.8, # -1
            "rgripper_right_inner_knuckle_joint": -0.8, # -1
            "rgripper_right_inner_finger_joint": 0.8,   # +1
        },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos_left = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_.*"])})
        joint_pos_right = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_.*"])})
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("target_object")})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    

    
    # Fixed start positions to prevent invalid initialization
    reset_fixed_objects = EventTerm(
        func=events.reset_objects_to_fixed_safe_pose,
        mode="reset",
    )
    
    # Old Randomization (Disabled)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    
    robot_through_table = DoneTerm(
        func=terminations.robot_out_of_bounds,
        params={
            "asset_cfg_left": SceneEntityCfg("robot", body_names=["left_.*"]),
            "asset_cfg_right": SceneEntityCfg("robot", body_names=["right_.*"]),
            "table_z": 0.0,
            
            # [CRITICAL FIX]
            # 1. Z-Buffer: -0.05 means terminate if Z < -0.05m
            # This allows robot to touch table (Z=0.0) without dying
            "margin": -0.05,
            
            # 2. X/Y Buffer: [-0.8, 0.8]
            # IK Action limits are [-0.6, 0.6]
            # This gives 20cm of "Overshoot Room" for momentum
            "x_range": (-0.8, 0.8),
            "y_range": (-0.8, 0.8),
        }
    )
    
    objects_off_table = DoneTerm(
        func=terminations.objects_out_of_bounds,
        params={
            "x_range": (-1.0, 1.0),
            "y_range": (-0.5, 1.5),
            "z_min": -0.2,
        }
    )

@configclass
class ReachDualArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ReachDualArm environment (Base)."""
    scene: ReachDualArmSceneCfg = ReachDualArmSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # No rewards defined here as we don't need them for the base class in this context
    # (Or we could define them if we wanted to support the original task too)
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.reset_settle_steps = 200
