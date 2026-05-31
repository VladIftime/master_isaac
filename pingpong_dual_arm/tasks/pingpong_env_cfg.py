"""Ping pong dual-arm environment configuration.

Two identical dual-arm robot systems face each other across a ping pong table.
Each robot uses one arm for the racket, controlled via a configurable IK solver.
The environment provides a standard Gymnasium interface for plugging in any RL algorithm.
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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UrdfFileCfg, UsdFileCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from . import terminations as term_mod
from . import observations as obs_mod
from . import rewards as rew_mod
from . import events as evt_mod

from ik_solvers import build_ik_action, IK_SOLVER_TYPE

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


##
# Robot configuration
##

UR5e_SINGLE_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=f"{_PKG_ROOT}/assets/urdf/ur_robotics/ur5e/ur5e_robotiq_140.urdf",
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
    spawn=UsdFileCfg(
        usd_path=f"{_PKG_ROOT}/urdf/dual_arm_robot_no_gripper_col.usd",
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
            "left_wrist_3_joint": 1.57,
            "right_shoulder_pan_joint": 0.0,
            "right_shoulder_lift_joint": -1.57,
            "right_elbow_joint": 1.57,
            "right_wrist_1_joint": -1.57,
            "right_wrist_2_joint": -1.57,
            "right_wrist_3_joint": 1.57,
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
    },
)

ARM_JOINTS_LEFT = ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"]
ARM_JOINTS_RIGHT = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]


##
# Scene definition
##

TABLE_WIDTH = 1.525
TABLE_LENGTH = 2.74
TABLE_HEIGHT = 0.76
TABLE_THICKNESS = 0.02
NET_HEIGHT = 0.1525
BALL_RADIUS = 0.02
BALL_MASS = 0.0027


@configclass
class PingPongSceneCfg(InteractiveSceneCfg):
    """Scene with two dual-arm robots, a ping pong table, ball, and rackets."""

    replicate_physics: bool = False

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=GroundPlaneCfg(
            size=(6.0, 6.0),
            color=(0.15, 0.15, 0.15),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, TABLE_HEIGHT / 2.0],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_WIDTH, TABLE_LENGTH, TABLE_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.877,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.4, 0.75),
            ),
        ),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, TABLE_HEIGHT + 0.3],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.95,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
            ),
        ),
    )

    net = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Net",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, TABLE_HEIGHT + NET_HEIGHT / 2.0],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_WIDTH, 0.015, NET_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9),
            ),
        ),
    )

    racket_A = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RacketA",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -TABLE_LENGTH / 2.0 + 0.15, TABLE_HEIGHT + 0.15],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.16, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
    )

    racket_B = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RacketB",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, TABLE_LENGTH / 2.0 - 0.15, TABLE_HEIGHT + 0.15],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.16, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.8),
            ),
        ),
    )

    # Robot A: positioned at one end of the table (-Y side)
    robot_A = DualArm_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotA",
        init_state=DualArm_CFG.init_state.replace(
            pos=(0.0, -TABLE_LENGTH / 2.0 - 0.15, 0.0),
        ),
    )

    # Robot B: positioned at the other end of the table (+Y side), facing inward
    robot_B = DualArm_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotB",
        init_state=DualArm_CFG.init_state.replace(
            pos=(0.0, TABLE_LENGTH / 2.0 + 0.15, 0.0),
            rot=(0.0, 0.0, 1.0, 0.0),  # Yaw 180 (facing -Y)
        ),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications.

    Each arm receives a 6D relative pose delta command.
    """

    arm_A: ActionTerm = MISSING
    arm_B: ActionTerm = MISSING

    def __post_init__(self):
        if self.arm_A is MISSING:
            self.arm_A = build_ik_action("diffik", asset_name="robot_A", side="right")
        if self.arm_B is MISSING:
            self.arm_B = build_ik_action("diffik", asset_name="robot_B", side="right")


@configclass
class ObservationsCfg:
    """Observation specifications for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_A = ObsTerm(
            func=obs_mod.robot_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot_A", joint_names=ARM_JOINTS_LEFT + ARM_JOINTS_RIGHT)},
        )
        ee_pose_A = ObsTerm(
            func=obs_mod.ee_poses,
            params={"ee_cfg": SceneEntityCfg("robot_A", body_names=["right_wrist_3_link"])},
        )
        joint_pos_B = ObsTerm(
            func=obs_mod.robot_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot_B", joint_names=ARM_JOINTS_LEFT + ARM_JOINTS_RIGHT)},
        )
        ee_pose_B = ObsTerm(
            func=obs_mod.ee_poses,
            params={"ee_cfg": SceneEntityCfg("robot_B", body_names=["right_wrist_3_link"])},
        )
        ball_state = ObsTerm(
            func=obs_mod.ball_state,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events (resets)."""

    reset_robot_joints = EventTerm(func=evt_mod.reset_robot_joints, mode="reset")
    serve_ball = EventTerm(func=evt_mod.serve_ball_random, mode="reset")


@configclass
class TerminationsCfg:
    """Termination terms."""

    ball_fell_off = DoneTerm(
        func=term_mod.ball_out_of_bounds,
        params={"z_min": 0.0, "x_range": (-1.2, 1.2), "y_range": (-1.4, 1.4)},
    )

    robot_out_of_bounds = DoneTerm(
        func=term_mod.robot_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot_A", body_names=["right_wrist_3_link"]),
            "table_z": 0.0,
            "margin": -0.1,
            "x_range": (-1.2, 1.2),
            "y_range": (-1.4, 1.4),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms."""

    rally = RewTerm(func=rew_mod.rally_time_reward, weight=0.001)
    ball_height = RewTerm(func=rew_mod.ball_height_reward, weight=0.1)
    ball_contact = RewTerm(func=rew_mod.ball_contact_reward, weight=1.0)
    point_won = RewTerm(func=rew_mod.point_won_reward, weight=1.0)


@configclass
class PingPongDualArmEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config for competitive ping pong with two dual-arm robots."""

    ik_solver: IK_SOLVER_TYPE = "diffik"
    playing_arm_side: str = "right"

    scene: PingPongSceneCfg = PingPongSceneCfg(num_envs=4, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 256
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4
        self.sim.physx.gpu_heap_capacity = 64 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 16 * 1024 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1024 * 1024

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.0

        self.actions.arm_A = build_ik_action(
            self.ik_solver, asset_name="robot_A", side=self.playing_arm_side,
        )
        self.actions.arm_B = build_ik_action(
            self.ik_solver, asset_name="robot_B", side=self.playing_arm_side,
        )
