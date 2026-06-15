"""DirectRLEnv configuration for the throw primitive.

No ManagerBasedRLEnv overhead — bare-metal physics stepping with a
state-machine _apply_action(). Designed for maximum training throughput.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg, UsdFileCfg
from isaaclab.utils import configclass

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAND_Z = 0.6
ROBOT_POS = (0.0, 0.0, STAND_Z)
TABLE_Z = STAND_Z - 0.1

PHASE_STABILIZE = 10
PHASE_GO_TO_INIT = 20
PHASE_GO_TO_INITIAL = 20
PHASE_THROW_MAX = 120
PHASE_FLIGHT = 150
TOTAL_DECIMATION = PHASE_STABILIZE + PHASE_GO_TO_INIT + PHASE_GO_TO_INITIAL + PHASE_THROW_MAX + PHASE_FLIGHT

DRINK_HOLD_Z_OFFSET = -0.25
DRINK_BELOW_TABLE_Z = 0.45
SUCCESS_THRESHOLD = 0.15

_ROBOT_USD = os.path.join(_PKG_ROOT, "assets", "robot", "dual_arm_robot.usd")
_ROBOT_URDF = os.path.join(_PKG_ROOT, "urdf", "dual_arm_robot.urdf")


def _robot_spawn_cfg():
    if os.path.exists(_ROBOT_USD):
        return UsdFileCfg(
            usd_path=_ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
            activate_contact_sensors=False,
        )
    return UrdfFileCfg(
        asset_path=_ROBOT_URDF,
        fix_base=False,
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
            fix_root_link=True,
        ),
        activate_contact_sensors=False,
    )


@configclass
class ThrowingDirectSceneCfg(InteractiveSceneCfg):
    """Minimal scene for throw primitive — optimized for GPU parallelism."""

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 1.0, TABLE_Z - 0.025]),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.7, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.65)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, STAND_Z / 2.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, STAND_Z),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=_robot_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_POS,
            joint_pos={
                "left_shoulder_pan_joint": 0.0,
                "left_shoulder_lift_joint": -1.57,
                "left_elbow_joint": -1.57,
                "left_wrist_1_joint": -1.57,
                "left_wrist_2_joint": 1.57,
                "left_wrist_3_joint": 1.5708,
                "right_shoulder_pan_joint": 0.0,
                "right_shoulder_lift_joint": -1.57,
                "right_elbow_joint": 1.57,
                "right_wrist_1_joint": -1.57,
                "right_wrist_2_joint": -1.57,
                "right_wrist_3_joint": 1.5708,
                "lgripper_finger_joint": 0.0,
                "rgripper_finger_joint": 0.0,
            },
        ),
        actuators={
            "arm_left": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
                stiffness=8000.0, damping=500.0,
            ),
            "arm_right": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
                stiffness=8000.0, damping=500.0,
            ),
            "gripper_right": ImplicitActuatorCfg(
                joint_names_expr=[
                    "rgripper_finger_joint",
                    "rgripper_.*_knuckle_joint$",
                    "rgripper_.*_inner_finger_joint$",
                ],
                stiffness=5000.0, damping=500.0,
            ),
            "gripper_left": ImplicitActuatorCfg(
                joint_names_expr=["lgripper_finger_joint"],
                stiffness=5000.0, damping=500.0,
            ),
        },
    )

    milk: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Milk",
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/assets/new_usds/drink001/drink_target.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=8,
                sleep_threshold=0.005, stabilization_threshold=0.0025,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=UsdFileCfg(
            usd_path=f"{_PKG_ROOT}/assets/new_usds/basket_02/model_basket1.usd",
            scale=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, TABLE_Z + 0.001)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ThrowingDirectEnvCfg(DirectRLEnvCfg):
    """Config for DirectRLEnv throw primitive — one outer step = one full throw."""

    decimation: int = TOTAL_DECIMATION
    episode_length_s: float = 4.0
    action_space: int = 4
    observation_space: int = 10
    state_space: int = 10

    scene: ThrowingDirectSceneCfg = ThrowingDirectSceneCfg(num_envs=2048, env_spacing=3.0)

    playing_arm_side: str = "right"
    target_x_range: tuple = (0.0, 0.5)
    target_y_range: tuple = (1.0, 1.6)
    target_z: float = TABLE_Z + 0.001

    success_threshold: float = SUCCESS_THRESHOLD
    grasp_strength: float = 0.48

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.use_fabric = True
        self.sim.render_interval = 2
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.3,
        )
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4
