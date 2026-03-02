
"""
Environment configuration for asymmetric dual-play.

Extends ReachDualArmEnvCfg with paper-specific modifications for Alice and Bob agents.
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

# INTERNAL IMPORT
from .utils.reach_dual_arm_env_cfg import (
    ReachDualArmEnvCfg,
    ReachDualArmSceneCfg,
    ActionsCfg,
    EventCfg,
    TerminationsCfg,
)

# Import our custom observation and reward functions
from .utils import observations, rewards
from isaaclab.sensors import ContactSensorCfg, patterns



@configclass
class AsyncDualPlayObservationsCfg:
    """
    Observation specifications for asymmetric dual-play.
    
    Alice sees: joints, grippers, objects
    Bob sees: joints, grippers, objects, goals, goal_distances
    """
    
    @configclass
    class AlicePolicyCfg(ObsGroup):
        """Observations for Alice (no goal information)"""
        
        # Robot state
        joint_pos = ObsTerm(
            func=observations.robot_joint_positions,
            params={
                "left_arm_cfg": SceneEntityCfg("robot", joint_names=["left_.*"]),
                "right_arm_cfg": SceneEntityCfg("robot", joint_names=["right_.*"]),
            }
        )
        
        gripper_pos = ObsTerm(
            func=observations.gripper_positions,
            params={
                "left_arm_cfg": SceneEntityCfg("robot", joint_names=["lgripper_finger_joint"]),
                "right_arm_cfg": SceneEntityCfg("robot", joint_names=["rgripper_finger_joint"]),
            }
        )
        
        # Object state

        object_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "left_gripper_cfg": SceneEntityCfg("robot", body_names="left_wrist_3_link"),
                "right_gripper_cfg": SceneEntityCfg("robot", body_names="right_wrist_3_link"),
                "left_contact_cfg": SceneEntityCfg("contact_forces_left"),
                "right_contact_cfg": SceneEntityCfg("contact_forces_right"),
            }
        )
        
        cube_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "left_gripper_cfg": SceneEntityCfg("robot", body_names="left_wrist_3_link"),
                "right_gripper_cfg": SceneEntityCfg("robot", body_names="right_wrist_3_link"),
                "left_contact_cfg": SceneEntityCfg("contact_forces_left"),
                "right_contact_cfg": SceneEntityCfg("contact_forces_right"),
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class BobPolicyCfg(ObsGroup):
        """Observations for Bob (includes goal information)"""
        
        # Robot state (same as Alice)
        joint_pos = ObsTerm(
            func=observations.robot_joint_positions,
            params={
                "left_arm_cfg": SceneEntityCfg("robot", joint_names=["left_.*"]),
                "right_arm_cfg": SceneEntityCfg("robot", joint_names=["right_.*"]),
            }
        )
        
        gripper_pos = ObsTerm(
            func=observations.gripper_positions,
            params={
                "left_arm_cfg": SceneEntityCfg("robot", joint_names=["lgripper_finger_joint"]),
                "right_arm_cfg": SceneEntityCfg("robot", joint_names=["rgripper_finger_joint"]),
            }
        )
        
        # Object state

        object_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "left_gripper_cfg": SceneEntityCfg("robot", body_names="left_wrist_3_link"),
                "right_gripper_cfg": SceneEntityCfg("robot", body_names="right_wrist_3_link"),
                "left_contact_cfg": SceneEntityCfg("contact_forces_left"),
                "right_contact_cfg": SceneEntityCfg("contact_forces_right"),
            }
        )
        
        cube_state = ObsTerm(
            func=observations.object_states,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "left_gripper_cfg": SceneEntityCfg("robot", body_names="left_wrist_3_link"),
                "right_gripper_cfg": SceneEntityCfg("robot", body_names="right_wrist_3_link"),
                "left_contact_cfg": SceneEntityCfg("contact_forces_left"),
                "right_contact_cfg": SceneEntityCfg("contact_forces_right"),
            }
        )
        
        # Goal state (Bob only)
        goal_state = ObsTerm(
            func=observations.goal_states,
            params={"object_cfg": SceneEntityCfg("target_object")}
        )
        
        cube_goal_state = ObsTerm(
            func=observations.goal_states,
            params={"object_cfg": SceneEntityCfg("cube")}
        )
        
        # Distance to goal
        goal_distance = ObsTerm(
            func=observations.goal_distance,
            params={"object_cfg": SceneEntityCfg("target_object")}
        )
        
        cube_goal_distance = ObsTerm(
            func=observations.goal_distance,
            params={"object_cfg": SceneEntityCfg("cube")}
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    alice_policy: AlicePolicyCfg = AlicePolicyCfg()
    bob_policy: BobPolicyCfg = BobPolicyCfg()


@configclass
class AsyncDualPlayRewardsCfg:
    """Reward specifications for asymmetric dual-play."""
    
    # Alice rewards
    # Handled manually in train.py (outcome) and wrapper (validity) to ensure one-time triggering
    
    # REMOVED: alice_out_of_bounds RewTerm
    # This was causing MASSIVE penalties because it was applied EVERY simulation step.
    # If Alice takes 100 steps and is out of bounds for 50, she'd get -150 total penalty,
    # which dwarfs the +1/-5 outcome rewards and causes policy collapse.
    # The -3.0 penalty is now ONLY applied once in wrapper.py at goal validation.
    
    # alice_out_of_bounds = RewTerm(
    #     func=rewards.out_of_bounds_penalty,
    #     weight=1.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("target_object"),
    #         "x_range": (-0.6, 0.6),
    #         "y_range": (0.3, 0.9),
    #     }
    # )
    
    # Bob rewards
    # NOTE: Bob's sparse rewards (+1 per object placed, +5 completion bonus) are computed
    # directly in AsyncDualPlayEnvWrapper._compute_bob_sparse_rewards() to avoid the
    # dt-scaling applied by RewardManager.compute() which would produce fractional values.
    # See implementation_plan.md for details on the reward calculation issue.
    pass


@configclass
class AsyncDualPlayEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the asymmetric dual-play environment.
    
    This extends the base RL environment configuration. We mostly reuse
    components from the existing ReachDualArmEnvCfg but override Observations
    and Rewards to support the minimal Alice/Bob logic.
    """
    

    # Scene settings (reuse from existing dual-arm environment)
    # We define a custom scene config inside the class or externally to limit objects
    @configclass
    class AsyncDualPlaySceneCfg(ReachDualArmSceneCfg):
        # Limit to 2 objects (Target + Cube) as per user request
        cylinder = None
        rect = None
        triangle = None
        
        # Disable camera to avoid spawning it when cameras are disabled (saves memory)
        camera = None

        # Contact Sensors
        contact_forces_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/RobotUnified/lgripper_.*finger.*",
            history_length=3,
            track_air_time=False,
        )
        contact_forces_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/RobotUnified/rgripper_.*finger.*",
            history_length=3,
            track_air_time=False,
        )
        
    scene: AsyncDualPlaySceneCfg = AsyncDualPlaySceneCfg(num_envs=4, env_spacing=2.5)
    
    # Observations (asymmetric for Alice and Bob)
    observations: AsyncDualPlayObservationsCfg = AsyncDualPlayObservationsCfg()
    
    # Actions (same as base environment)
    actions: ActionsCfg = ActionsCfg()
    
    # Rewards (separate for Alice and Bob)
    rewards: AsyncDualPlayRewardsCfg = AsyncDualPlayRewardsCfg()
    
    # Terminations and events (reuse directly from base env config)
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        # Episode length for Alice (250 steps) and Bob (600 steps)
        # Total max episode: 250 + 5 * 600 = 3250 steps
        # At dt=0.01 and decimation=2: 3250 * 0.02 = 65 seconds
        self.episode_length_s = 65.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
