"""
Environment configuration for asymmetric dual-play.

Extends ReachDualArmEnvCfg with the two-agent (Alice/Bob) observation and
reward structure needed for the Asymmetric Self-Play paper.
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

from .utils.reach_dual_arm_env_cfg import (
    ReachDualArmEnvCfg,
    ReachDualArmSceneCfg,
    ActionsCfg,
    EventCfg,
    TerminationsCfg,
)
from .utils import observations, rewards


@configclass
class AsyncDualPlayObservationsCfg:
    """
    Asymmetric observations for Alice and Bob.

    Alice sees robot state and object state.
    Bob sees the same, plus the goal state and per-object distances to the goal.
    """

    @configclass
    class AlicePolicyCfg(ObsGroup):
        """Alice's observations — no goal information."""

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
                "object_cfg":  SceneEntityCfg("target_object"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": SceneEntityCfg("contact_forces"),
            },
        )
        cube_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg":  SceneEntityCfg("cube"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": SceneEntityCfg("contact_forces"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BobPolicyCfg(ObsGroup):
        """Bob's observations — extends Alice's with goal state and distance to goal."""

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
                "object_cfg":  SceneEntityCfg("target_object"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": SceneEntityCfg("contact_forces"),
            },
        )
        cube_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg":  SceneEntityCfg("cube"),
                "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "contact_cfg": SceneEntityCfg("contact_forces"),
            },
        )
        goal_state = ObsTerm(
            func=observations.goal_states,
            params={"object_cfg": SceneEntityCfg("target_object")},
        )
        cube_goal_state = ObsTerm(
            func=observations.goal_states,
            params={"object_cfg": SceneEntityCfg("cube")},
        )
        goal_distance = ObsTerm(
            func=observations.goal_distance,
            params={"object_cfg": SceneEntityCfg("target_object")},
        )
        cube_goal_distance = ObsTerm(
            func=observations.goal_distance,
            params={"object_cfg": SceneEntityCfg("cube")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    alice_policy: AlicePolicyCfg = AlicePolicyCfg()
    bob_policy:   BobPolicyCfg   = BobPolicyCfg()


@configclass
class AsyncDualPlayRewardsCfg:
    """
    Reward specifications for asymmetric dual-play.

    Alice's rewards are applied manually in train.py (outcome reward) and
    wrapper.py (goal-validity bonus) to guarantee one-time triggering.

    Bob's sparse rewards (+1 per object placed, +5 completion bonus) are
    computed in AsyncDualPlayEnvWrapper._compute_bob_sparse_rewards() to
    avoid IsaacLab's dt-scaling, which would produce fractional values.
    """
    pass


@configclass
class AsyncDualPlayEnvCfg(ManagerBasedRLEnvCfg):
    """
    Full environment configuration for asymmetric dual-play.

    Inherits scene, actions, terminations, and events from ReachDualArmEnvCfg.
    Overrides observations and rewards for the two-agent Alice/Bob structure,
    and narrows the scene to two objects (target + cube).
    """

    @configclass
    class AsyncDualPlaySceneCfg(ReachDualArmSceneCfg):
        """Scene restricted to two objects and augmented with gripper contact sensors."""
        cylinder = None
        rect     = None
        triangle = None
        camera   = None

        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/RobotUnified/.*finger.*",
            history_length=3,
            track_air_time=False,
        )

    scene:        AsyncDualPlaySceneCfg        = AsyncDualPlaySceneCfg(num_envs=4, env_spacing=2.5)
    observations: AsyncDualPlayObservationsCfg = AsyncDualPlayObservationsCfg()
    actions:      ActionsCfg                   = ActionsCfg()
    rewards:      AsyncDualPlayRewardsCfg      = AsyncDualPlayRewardsCfg()
    terminations: TerminationsCfg              = TerminationsCfg()
    events:       EventCfg                     = EventCfg()

    def __post_init__(self):
        self.decimation          = 2
        self.episode_length_s    = 65.0  # Alice(250) + 5 × Bob(600) steps at dt=0.01, decimation=2
        self.sim.dt              = 0.01
        self.sim.render_interval = self.decimation
