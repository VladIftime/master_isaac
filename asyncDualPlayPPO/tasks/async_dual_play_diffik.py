"""
Environment configuration for asymmetric dual-play — DifferentialIK variant.

Identical to async_dual_play.py except it inherits scene/actions/terminations/events
from reach_dual_arm_diffik_env_cfg (DiffIK) instead of reach_dual_arm_env_cfg (RMPflow).
Observations and reward structure are unchanged.
"""

from dataclasses import MISSING
import sys
import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, patterns
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from .utils.push_primitive_1arm_env import (
    Push1ArmEnvCfg,
    Push1ArmSceneCfg,
    ActionsCfg,
    EventCfg,
    TerminationsCfg,
    ISAACLAB_DUAL_ARM_EXT_DIR,
)
from .utils import observations, rewards


@configclass
class AsyncDualPlayObservationsCfg:
    """
    Asymmetric observations for Alice and Bob.

    Alice sees robot state and object state(s).
    Bob sees the same, plus goal state and per-object distances.

    cube_state / cube_goal_state / cube_goal_distance are defined for the
    2-object task mode.  In 1-object mode train_curobo.py sets them to None.
    In distractor mode only cube_goal_state / cube_goal_distance are nulled.
    """

    @configclass
    class AlicePolicyCfg(ObsGroup):
        """Alice's observations — no goal information."""

        ee_pose = ObsTerm(
            func=observations.ee_poses,
            params={"ee_cfg": SceneEntityCfg("robot", body_names="wrist_3_link")},
        )
        gripper_pos = ObsTerm(
            func=observations.gripper_positions,
            params={"arm_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        object_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": None,
            },
        )
        cube_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": None,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BobPolicyCfg(ObsGroup):
        """Bob's observations — extends Alice's with goal state and distance."""

        ee_pose = ObsTerm(
            func=observations.ee_poses,
            params={"ee_cfg": SceneEntityCfg("robot", body_names="wrist_3_link")},
        )
        gripper_pos = ObsTerm(
            func=observations.gripper_positions,
            params={"arm_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        object_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": None,
            },
        )
        goal_state = ObsTerm(
            func=observations.goal_states,
            params={"object_cfg": SceneEntityCfg("target_object")},
        )
        goal_distance = ObsTerm(
            func=observations.goal_distance,
            params={"object_cfg": SceneEntityCfg("target_object")},
        )
        cube_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": None,
            },
        )
        cube_goal_state = ObsTerm(
            func=observations.goal_states,
            params={"object_cfg": SceneEntityCfg("cube")},
        )
        cube_goal_distance = ObsTerm(
            func=observations.goal_distance,
            params={"object_cfg": SceneEntityCfg("cube")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    alice_policy: AlicePolicyCfg = AlicePolicyCfg()
    bob_policy: BobPolicyCfg = BobPolicyCfg()


@configclass
class AsyncDualPlayRewardsCfg:
    """Reward specifications — handled manually in wrapper.py."""
    pass


@configclass
class AsyncDualPlayDiffIKEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment configuration — DifferentialIK variant."""

    @configclass
    class AsyncDualPlaySceneCfg(Push1ArmSceneCfg):
        """Scene: target_object (T-block) + cube (second task or distractor)."""

        cylinder = None
        rect = None
        triangle = None
        camera = None
        contact_forces = None

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
            ),
        )

        goal_ghost = None
        goal_marker = None  # handled via VisualizationMarkers

    scene: AsyncDualPlaySceneCfg = AsyncDualPlaySceneCfg(num_envs=4, env_spacing=2.5)
    observations: AsyncDualPlayObservationsCfg = AsyncDualPlayObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: AsyncDualPlayRewardsCfg = AsyncDualPlayRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 10000.0
        self.sim.dt = 0.02
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 81920 * 4
