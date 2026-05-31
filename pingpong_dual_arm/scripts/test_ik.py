#!/usr/bin/env python3
"""Test script: compare different IK solvers on a simple reach task.

Usage:
    cd pingpong_dual_arm
    ../../isaaclab.sh -p scripts/test_ik.py -- --solver osc

This script creates a minimal environment with one dual-arm robot and tests
the selected IK solver by commanding the end-effector to target poses.
"""

import argparse
import os
import sys
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test IK solvers for dual-arm robot.")
parser.add_argument("--solver", type=str, default="diffik", choices=["diffik", "osc", "rmpflow"],
                    help="IK solver to test.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print(f"\n=== Testing IK Solver: {args_cli.solver} ===")

from ik_solvers import build_ik_action

action_cfg = build_ik_action(args_cli.solver, asset_name="robot", side="right")

print(f"Action config created:")
print(f"  Type: {type(action_cfg).__name__}")
print(f"  Joint names: {action_cfg.joint_names}")
print(f"  Body name: {action_cfg.body_name}")

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from tasks.pingpong_env import PingPongDualArmEnv
import os

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@configclass
class TestIKEnvCfg(ManagerBasedRLEnvCfg):
    @configclass
    class TestSceneCfg(InteractiveSceneCfg):
        replicate_physics: bool = False
        plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
            spawn=GroundPlaneCfg(),
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

    @configclass
    class TestObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
            ee_pos = ObsTerm(func=mdp.body_pos_w, params={"asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_3_link")})
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        policy: PolicyCfg = PolicyCfg()

    @configclass
    class TestEventCfg:
        reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    scene: TestSceneCfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    observations: TestObservationsCfg = TestObservationsCfg()
    events: TestEventCfg = TestEventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.01


cfg = TestIKEnvCfg()
cfg.scene.robot = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{_PKG_ROOT}/urdf/dual_arm_robot_no_gripper_col.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        joint_pos={
            "left_shoulder_pan_joint": 0.0,
            "left_shoulder_lift_joint": -1.57,
            "left_elbow_joint": -1.57,
            "left_wrist_1_joint": -1.57,
            "left_wrist_2_joint": 1.57,
            "left_wrist_3_joint": 1.57,
            "right_shoulder_pan_joint": 0.0,
            "right_shoulder_lift_joint": -0.8,
            "right_elbow_joint": 1.57,
            "right_wrist_1_joint": -1.57,
            "right_wrist_2_joint": -1.57,
            "right_wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm_left": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
            stiffness=5000.0, damping=200.0,
        ),
        "arm_right": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
            stiffness=5000.0, damping=200.0,
        ),
    },
)

cfg.actions = type("ActionsCfg", (), {"arm": action_cfg})()

env = PingPongDualArmEnv(cfg)
obs, info = env.reset()
print(f"Environment initialized. Action dim: {env.action_space.shape}")

target_idx = 0
targets = [
    torch.tensor([[0.3, 0.0, 0.5, 0.0, 0.0, 0.0]]),
    torch.tensor([[0.3, 0.2, 0.3, 0.0, 0.5, 0.0]]),
    torch.tensor([[0.3, -0.2, 0.4, 0.0, -0.5, 0.0]]),
    torch.tensor([[0.0, -0.3, 0.5, 0.7, 0.0, 0.0]]),
    torch.tensor([[0.0, 0.3, 0.3, -0.7, 0.0, 0.0]]),
    torch.tensor([[0.3, 0.0, 0.2, 0.0, 0.0, 0.0]]),
]

print(f"\nTesting {len(targets)} reach targets...")

for step in range(500):
    if step % 100 == 0:
        target_idx = (target_idx + 1) % len(targets)
        print(f"  Step {step}: switching to target {target_idx}")
    action = targets[target_idx].to(env.device)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 10 == 0:
        ee_pos = env.scene["robot"].data.body_pos_w[:, 0] - env.scene.env_origins
        print(f"  EE pos: x={ee_pos[0,0]:.3f}, y={ee_pos[0,1]:.3f}, z={ee_pos[0,2]:.3f}")

print(f"\n=== IK test complete with solver '{args_cli.solver}' ===")
env.close()
simulation_app.close()
