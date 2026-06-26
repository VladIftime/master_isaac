"""DirectRLEnv config for push primitive with cuRobo IK.

One outer step = one complete push macro-action (72 physics substeps).
No ManagerBasedRLEnv overhead — direct physics access + state-machine _apply_action().
"""

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg, UsdFileCfg
from isaaclab.utils import configclass

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE_X = 0.0
TABLE_Y = 0.40
TABLE_Z = -0.05
TABLE_SIZE = (1.40, 1.00, 0.1)

ROBOT_POS = (0.0, 0.0, 0.0)

PHASE_APPROACH = 12
PHASE_DESCEND = 16
PHASE_PUSH = 20
PHASE_RETRACT = 16
PHASE_RETURN = 8
TOTAL_DECIMATION = PHASE_APPROACH + PHASE_DESCEND + PHASE_PUSH + PHASE_RETRACT + PHASE_RETURN

PUSH_APPROACH_HEIGHT = 0.50

OBS_ROBOT_DIM = 6
OBS_OBJ_STATE_DIM = 14
OBS_GOAL_DIM = 6
OBS_DIST_DIM = 2
OBS_REL_DIM = 2


@configclass
class PushDirectSceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = True

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.005]),
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TABLE_X, TABLE_Y, TABLE_Z],
                                                  rot=[1.0, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                          disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.5,
            ),
        ),
    )

    zone_border_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderTop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.70, 0.001],
                                                  rot=[1.0, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(1.02, 0.02, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
        ),
    )
    zone_border_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderBottom",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.25, 0.001],
                                                  rot=[1.0, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(1.02, 0.02, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
        ),
    )
    zone_border_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.50, 0.475, 0.001],
                                                  rot=[1.0, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.47, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
        ),
    )
    zone_border_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ZoneBorderRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.50, 0.475, 0.001],
                                                  rot=[1.0, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.47, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RobotUnified",
        spawn=UrdfFileCfg(
            asset_path=f"{PROJ_ROOT}/assets/urdf/ur_robotics/ur5e/ur5e_robotiq_140.urdf",
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=1000.0, damping=50.0,
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
            pos=ROBOT_POS,
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
                stiffness=5000.0, damping=200.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=500.0, damping=100.0,
            ),
            "manual_mimics": ImplicitActuatorCfg(
                joint_names_expr=[".*knuckle_joint", ".*inner_finger_joint"],
                stiffness=500.0, damping=20.0,
            ),
        },
    )

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.5, 0.05], rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{PROJ_ROOT}/assets/blocks/t_shape.usda",
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

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class PushDirectEnvCfg(DirectRLEnvCfg):
    decimation: int = TOTAL_DECIMATION
    episode_length_s: float = 20.0
    action_space: int = 4
    observation_space: int = 22
    state_space: int = 0

    scene: PushDirectSceneCfg = PushDirectSceneCfg(num_envs=4096, env_spacing=2.5)

    max_pushes_per_episode: int = 5
    rel_obs: bool = True

    push_success_threshold_pos: float = 0.05
    push_success_threshold_rot: float = 0.2
    dense_alpha: float = 3.0
    dense_beta: float = 0.5
    dense_rot_beta: float = 0.25
    completion_bonus: float = 5.0
    rotation_sub_bonus: float = 2.0
    tip_penalty: float = -5.0
    catastrophe_penalty: float = -10.0
    tip_over_threshold: float = 0.3

    goal_x_range: tuple = (-0.40, 0.40)
    goal_y_range: tuple = (0.30, 0.70)
    goal_z: float = 0.02
    spawn_x_range: tuple = (-0.40, 0.40)
    spawn_y_range: tuple = (0.30, 0.70)
    spawn_z: float = 0.05
    goal_min_dist: float = 0.05
    goal_max_dist: float = 0.45

    ws_x: tuple = (-0.50, 0.50)
    ws_y: tuple = (0.25, 0.70)
    ws_z: tuple = (0.25, 0.55)

    rel_act: bool = False
    action_min_r: float = 0.02
    action_max_r: float = 0.08
    action_max_len: float = 0.20

    ik_n_iters: int = 30
    ik_inner_iters: int = 10

    def __post_init__(self):
        self.decimation = TOTAL_DECIMATION
        self.sim.dt = 0.02
        self.sim.use_fabric = True
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4
