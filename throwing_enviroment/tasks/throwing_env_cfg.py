"""Throwing environment configuration.

Single dual-arm robot throws an object toward a target basket,
avoiding an obstacle.  One arm is controlled via a configurable IK solver.
"""

from dataclasses import MISSING
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UrdfFileCfg,
    UsdFileCfg,
)
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from . import terminations as term_mod
from . import observations as obs_mod
from . import rewards as rew_mod
from . import events as evt_mod

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PINGPONG_ROOT = os.path.join(_PKG_ROOT, "..", "pingpong_dual_arm")

import sys

if _PINGPONG_ROOT not in sys.path:
    sys.path.insert(0, _PINGPONG_ROOT)

from ik_solvers import build_ik_action, IK_SOLVER_TYPE

##
# Robot configuration (reuses pingpong's DualArm_CFG pattern)
##

UR5e_SINGLE_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=f"{_PINGPONG_ROOT}/assets/urdf/ur_robotics/ur5e/ur5e_robotiq_140.urdf",
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
    },
)

DualArm_CFG = UR5e_SINGLE_CFG.replace(
    spawn=UrdfFileCfg(
        asset_path=f"{_PINGPONG_ROOT}/urdf/dual_arm_robot.urdf",
        fix_base=False,
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
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_shoulder_pan_joint": 0.0,
            "left_shoulder_lift_joint": -1.57,
            "left_elbow_joint": -1.57,
            "left_wrist_1_joint": -1.57,
            "left_wrist_2_joint": 1.57,
            "left_wrist_3_joint": 0.0,
            "right_shoulder_pan_joint": 0.0,
            "right_shoulder_lift_joint": -1.57,
            "right_elbow_joint": 1.57,
            "right_wrist_1_joint": -1.57,
            "right_wrist_2_joint": -1.57,
            "right_wrist_3_joint": 0.0,
            "lgripper_finger_joint": 0.0,
            "rgripper_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm_left": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
            stiffness=8000.0,
            damping=500.0,
        ),
        "arm_right": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
            stiffness=8000.0,
            damping=500.0,
        ),
        "gripper_left": ImplicitActuatorCfg(
            joint_names_expr=["lgripper_finger_joint"],
            stiffness=5000.0,
            damping=500.0,
        ),
        "gripper_right": ImplicitActuatorCfg(
            joint_names_expr=[
                "rgripper_finger_joint",
                "rgripper_.*_knuckle_joint$",
                "rgripper_.*_inner_finger_joint$",
            ],
            stiffness=5000.0,
            damping=500.0,
        ),
    },
)

ARM_JOINTS_LEFT = ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"]
ARM_JOINTS_RIGHT = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]

##
# Scene geometry constants
##

STAND_Z = 0.6
ROBOT_POS = (0.0, 0.0, STAND_Z)

TABLE_Z = STAND_Z                        # table surface height
TABLE_CENTER_POS = (0.0, 1.0, TABLE_Z)   # table center forward of robot
TABLE_SIZE = (2.0, 1.2, 0.05)            # tabletop dimensions (x, y, z)

##
# Scene definition
##


@configclass
class ThrowingSceneCfg(InteractiveSceneCfg):
    """Scene with one dual-arm robot on a stand and throwable objects."""

    replicate_physics: bool = False

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(
            size=(6.0, 6.0),
            color=(0.15, 0.15, 0.15),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[ROBOT_POS[0], ROBOT_POS[1], STAND_Z / 2.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, STAND_Z),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.35),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[TABLE_CENTER_POS[0], TABLE_CENTER_POS[1], TABLE_Z - TABLE_SIZE[2] / 2.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.65),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )

    robot = DualArm_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=DualArm_CFG.init_state.replace(
            pos=ROBOT_POS,
        ),
    )

    milk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Milk",
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/assets/new_usds/drink001/drink_target.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
        ),
    )

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/assets/new_usds/shopping basket002/basket_target.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=2.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_CENTER_POS[0], TABLE_CENTER_POS[1], TABLE_Z + 0.1),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """6-D relative pose delta action for the throwing arm."""

    arm: ActionTerm = MISSING

    def __post_init__(self):
        if self.arm is MISSING:
            self.arm = build_ik_action("diffik", asset_name="robot", side="right")


@configclass
class ObservationsCfg:
    """Observation specifications: throwing arm joints/EE + object states."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=obs_mod.robot_joint_positions,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINTS_RIGHT
                )
            },
        )
        joint_vel = ObsTerm(
            func=obs_mod.robot_joint_velocities,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINTS_RIGHT
                )
            },
        )
        ee_pose = ObsTerm(
            func=obs_mod.ee_pose,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["right_wrist_3_link"])
            },
        )
        object_pos = ObsTerm(func=obs_mod.object_position, params={"object_name": "milk"})
        target_pos = ObsTerm(func=obs_mod.object_position, params={"object_name": "target"})
        dist_to_target = ObsTerm(func=obs_mod.dist_to_target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: reset robot, randomize objects, attach object."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_robot = EventTerm(
        func=evt_mod.reset_robot_joints,
        mode="reset",
    )
    randomize_target = EventTerm(
        func=evt_mod.randomize_target_position,
        mode="reset",
    )
    attach_object = EventTerm(
        func=evt_mod.attach_milk_to_gripper,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms: distance-based throwing reward, success bonus."""

    dist_reward = RewTerm(func=rew_mod.dist_to_target_reward, weight=1.0)
    success_bonus = RewTerm(func=rew_mod.success_bonus, weight=2.0)
    ee_velocity_reward = RewTerm(func=rew_mod.ee_velocity_reward, weight=0.5)


@configclass
class TerminationsCfg:
    """Termination conditions — episode ends on time or object settled."""

    time_limit = DoneTerm(func=mdp.time_out, time_out="truncated")
    object_out_of_bounds = DoneTerm(
        func=term_mod.object_out_of_bounds, time_out="terminated"
    )
    object_settled = DoneTerm(
        func=term_mod.object_settled, time_out="terminated"
    )


##
# Top-level environment config
##


@configclass
class ThrowingEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config for single-arm throwing with IK-based control."""

    ik_solver: IK_SOLVER_TYPE = "diffik"
    playing_arm_side: str = "right"

    release_min_steps: int = 10
    release_vel_threshold: float = 2.0
    release_at_step: int = 0
    randomize_target: bool = True
    disable_attachment: bool = False

    grip_settle_steps: int = 40

    target_x_range: tuple = (-0.45, 0.45)
    target_y_range: tuple = (0.7, 1.3)
    target_z: float = TABLE_Z + 0.1

    contact_threshold: float = 0.06

    scene: ThrowingSceneCfg = ThrowingSceneCfg(num_envs=4, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 5.0
        self.sim.dt = 1.0 / 120.0
        self.sim.use_fabric = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.3,
        )
        self.reset_settle_steps = 10

        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4

        self.actions.arm = build_ik_action(
            self.ik_solver,
            asset_name="robot",
            side=self.playing_arm_side,
        )
