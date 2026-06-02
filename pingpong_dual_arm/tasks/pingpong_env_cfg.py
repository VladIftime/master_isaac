"""Ping pong dual-arm environment configuration.

Two identical dual-arm robot systems face each other across a ping pong table.
Each robot uses one arm for the racket, controlled via a configurable IK solver.
Full table-tennis game logic: virtual paddle contact, table zone scoring,
randomized ball serves, reward shaping (ported from Isaaclab-TableTennisRobot).
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
    spawn=UrdfFileCfg(
        asset_path=f"{_PKG_ROOT}/urdf/dual_arm_robot_no_gripper_col.urdf",
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
            "left_shoulder_pan_joint": -0.29,
            "left_shoulder_lift_joint": -1.212,
            "left_elbow_joint": 1.712,
            "left_wrist_1_joint": 0.0,
            "left_wrist_2_joint": -0.33,
            "left_wrist_3_joint": 1.39,
            "right_shoulder_pan_joint": -0.29,
            "right_shoulder_lift_joint": -1.212,
            "right_elbow_joint": 1.712,
            "right_wrist_1_joint": 0.0,
            "right_wrist_2_joint": -0.33,
            "right_wrist_3_joint": 1.39,
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
# Table geometry constants
##

TABLE_WIDTH = 1.525
TABLE_LENGTH = 2.74
TABLE_HEIGHT = 0.78
STAND_Z = 0.6  # robot base height matches UR10 in TableTennisRobot

STAND_A_POS = (0.0, -2.7, STAND_Z)
STAND_B_POS = (0.0, 2.7, STAND_Z)


##
# Scene definition
##

@configclass
class PingPongSceneCfg(InteractiveSceneCfg):
    """Scene with two dual-arm robots on stands, table, ball, and rackets."""

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

    # Robot stands — matching UR10 base height from TableTennisRobot (z=0.6)
    # Cuboid from z=0 to z=STAND_Z, centered at half height
    stand_A = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StandA",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[STAND_A_POS[0], STAND_A_POS[1], STAND_Z / 2.0],
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

    stand_B = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StandB",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[STAND_B_POS[0], STAND_B_POS[1], STAND_Z / 2.0],
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

    # Table — custom USD from TableTennisRobot (kinematic, fixed in place)
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/meshes/custom_usd_pingpong/Table_tennis.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # Ball — custom USD from TableTennisRobot
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/meshes/custom_usd_pingpong/Ping_pong_ball.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT + 0.15),
        ),
    )

    # Kinematic rackets — tracked to wrist_3_link each step
    racket_A = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RacketA",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -TABLE_LENGTH / 2.0 + 0.15, TABLE_HEIGHT + 0.15],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/assets/pingpong/racket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    racket_B = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RacketB",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, TABLE_LENGTH / 2.0 - 0.15, TABLE_HEIGHT + 0.15],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/assets/pingpong/racket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Robot A: -Y side, facing +Y, body base at z=STAND_Z on stand
    robot_A = DualArm_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotA",
        init_state=DualArm_CFG.init_state.replace(
            pos=STAND_A_POS,
        ),
    )

    # Robot B: +Y side, facing -Y (180° yaw), body base at z=STAND_Z on stand
    robot_B = DualArm_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotB",
        init_state=DualArm_CFG.init_state.replace(
            pos=STAND_B_POS,
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """12-D relative pose delta actions: 6-D per robot's playing arm."""

    arm_A: ActionTerm = MISSING
    arm_B: ActionTerm = MISSING

    def __post_init__(self):
        if self.arm_A is MISSING:
            self.arm_A = build_ik_action("diffik", asset_name="robot_A", side="right")
        if self.arm_B is MISSING:
            self.arm_B = build_ik_action("diffik", asset_name="robot_B", side="right")


@configclass
class ObservationsCfg:
    """Observation specifications: both robots' joints/EE + ball state."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_A = ObsTerm(
            func=obs_mod.robot_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot_A", joint_names=ARM_JOINTS_LEFT + ARM_JOINTS_RIGHT)},
        )
        joint_vel_A = ObsTerm(
            func=obs_mod.robot_joint_velocities,
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
        joint_vel_B = ObsTerm(
            func=obs_mod.robot_joint_velocities,
            params={"robot_cfg": SceneEntityCfg("robot_B", joint_names=ARM_JOINTS_LEFT + ARM_JOINTS_RIGHT)},
        )
        ee_pose_B = ObsTerm(
            func=obs_mod.ee_poses,
            params={"ee_cfg": SceneEntityCfg("robot_B", body_names=["right_wrist_3_link"])},
        )
        ball_state = ObsTerm(func=obs_mod.ball_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: serve ball with randomization, reset robots."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_robots = EventTerm(
        func=evt_mod.reset_robot_joints,
        mode="reset",
    )
    serve_ball = EventTerm(
        func=evt_mod.serve_ball_alternating,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms: ported from TableTennisRobot, adapted for dual-robot self-play."""

    contact_A = RewTerm(func=rew_mod.paddle_contact_reward_A, weight=1.0)
    contact_B = RewTerm(func=rew_mod.paddle_contact_reward_B, weight=1.0)
    table_success_A = RewTerm(func=rew_mod.table_success_reward_A, weight=5.0)
    table_success_B = RewTerm(func=rew_mod.table_success_reward_B, weight=5.0)
    table_fail_A = RewTerm(func=rew_mod.table_fail_reward_A, weight=-2.0)
    table_fail_B = RewTerm(func=rew_mod.table_fail_reward_B, weight=-2.0)
    ball_floor = RewTerm(func=rew_mod.ball_floor_penalty, weight=-3.5)
    velocity_A = RewTerm(func=rew_mod.velocity_reward_A, weight=0.5)
    velocity_B = RewTerm(func=rew_mod.velocity_reward_B, weight=0.5)
    ball_pos_A = RewTerm(func=rew_mod.ball_pos_reward_A, weight=2.0)
    ball_pos_B = RewTerm(func=rew_mod.ball_pos_reward_B, weight=2.0)


@configclass
class TerminationsCfg:
    """Termination conditions — episode ends on point scored or ball out."""

    time_limit = DoneTerm(func=mdp.time_out, time_out="truncated")
    table_success_A = DoneTerm(func=term_mod.round_end_success_A, time_out="terminated")
    table_success_B = DoneTerm(func=term_mod.round_end_success_B, time_out="terminated")
    table_fail_A = DoneTerm(func=term_mod.round_end_fail_A, time_out="terminated")
    table_fail_B = DoneTerm(func=term_mod.round_end_fail_B, time_out="terminated")
    ball_floor = DoneTerm(func=term_mod.ball_to_floor, time_out="terminated")
    ball_out_of_bounds = DoneTerm(func=term_mod.ball_out_of_bounds, time_out="terminated")


##
# Top-level environment config
##

@configclass
class PingPongDualArmEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config for competitive ping pong with two dual-arm robots.

    Ported game logic from Isaaclab-TableTennisRobot:
      - Virtual paddle contact detection (distance threshold)
      - Table zone scoring (opponent vs own halves)
      - Randomized ball serves with alternating sides
      - Reward shaping: contact, velocity, table success/fail, floor penalty
    """

    ik_solver: IK_SOLVER_TYPE = "diffik"
    playing_arm_side: str = "right"

    # Game parameters — same ranges as TableTennisRobot
    ball_speed_x_range: tuple = (-1.0, 1.0)
    ball_speed_y_range: tuple = (3.5, 5.0)
    ball_speed_z_range: tuple = (2.0, 2.2)
    ball_pos_x_range: tuple = (-0.2, 0.2)
    ball_pos_y_range: tuple = (-1.4, -1.0)

    # Table contact zone definitions (in env-local coords, table centre at origin)
    # Negative Y zone: y ∈ [-1.35, -0.1] -> Robot A's own zone, Robot B's opponent zone
    # Positive Y zone: y ∈ [0, 1.36]   -> Robot B's own zone, Robot A's opponent zone
    table_zone_neg_y: tuple = (-1.35, -0.1)
    table_zone_pos_y: tuple = (0.0, 1.36)
    table_zone_x: tuple = (-0.74, 0.74)
    table_zone_z: tuple = (0.68, 0.735)

    contact_threshold: float = 0.06

    scene: PingPongSceneCfg = PingPongSceneCfg(num_envs=4, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 5.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,
        )
        self.reset_settle_steps = 5

        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4

        self.actions.arm_A = build_ik_action(
            self.ik_solver, asset_name="robot_A", side=self.playing_arm_side,
        )
        self.actions.arm_B = build_ik_action(
            self.ik_solver, asset_name="robot_B", side=self.playing_arm_side,
        )
