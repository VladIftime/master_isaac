"""
Environment configuration for single-agent push task — cuRobo variant.

Minimal scene: one UR5e robot, one table, one object (random shape from pool,
assigned per environment index — same pattern as train_curobo.py).
Observations: single 29D vector (robot + object state + goal).
Actions: JointPositionActionCfg (cuRobo computes joint positions externally).
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
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from .utils.reach_dual_arm_diffik_env_cfg import (
    ReachDualArmDiffIKEnvCfg,
    ReachDualArmSceneCfg,
    ActionsCfg,
    EventCfg,
    TerminationsCfg,
    ISAACLAB_DUAL_ARM_EXT_DIR,
)
from .utils import observations


@configclass
class PushTaskObservationsCfg:
    """
    Single-agent observation for push-PPO baseline.

    Layout (flat, 29D for 1 object):
      [ee_pose(6) | gripper(1) | obj_state(14) | goal_pose(6) | goal_dist(2)]
    """

    @configclass
    class PushPolicyCfg(ObsGroup):
        """Push agent observations — robot state + object state + goal."""

        ee_pose = ObsTerm(
            func=observations.ee_poses,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            },
        )
        gripper_pos = ObsTerm(
            func=observations.gripper_positions,
            params={
                "arm_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            },
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    push_policy: PushPolicyCfg = PushPolicyCfg()


@configclass
class PushTaskRewardsCfg:
    """
    Reward config for push task — all reward computation happens in the
    wrapper, not via Isaac Lab's reward terms.  This class exists only to
    satisfy the ManagerBasedRLEnv interface.
    """
    pass


@configclass
class PushTaskCuRoboEnvCfg(ManagerBasedRLEnvCfg):
    """
    Full environment configuration for the push-PPO baseline.

    Inherits scene/actions/terminations/events from ReachDualArmDiffIKEnvCfg.
    Uses JointPositionActionCfg for cuRobo-driven joint control.
    """

    @configclass
    class PushTaskSceneCfg(ReachDualArmSceneCfg):
        """Scene with all block shapes; each scenario activates one, others hidden."""
        camera = None
        contact_forces = None
        goal_ghost = None

        cube = None
        cylinder = None
        rect = None
        triangle = None

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

    scene: PushTaskSceneCfg = PushTaskSceneCfg(num_envs=4, env_spacing=2.5)
    observations: PushTaskObservationsCfg = PushTaskObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: PushTaskRewardsCfg = PushTaskRewardsCfg()
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
        # Disable terminations (they cause auto-resets mid-trajectory)
        self.terminations.robot_through_table = None
        self.terminations.objects_off_table = None
