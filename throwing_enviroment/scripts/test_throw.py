#!/usr/bin/env python3
"""Test the throw primitive — same pipeline as SAC training.

Runs execute_primitive_batched() with configurable action parameters.
On --loop, the pre-position cache kicks in after the first throw,
skipping SETTLE+APPROACH+DESCEND (~220 steps).

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/test_throw.py --ik diffik
    python scripts/test_throw.py --ik diffik --loop
    python scripts/test_throw.py --ik diffik --loop --initial_jv 0.5 --final_jv -0.3
"""

import argparse
import os
import sys

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test throw primitive (same pipeline as SAC).")
parser.add_argument(
    "--ik", type=str, default="diffik", choices=["diffik", "osc", "rmpflow", "curobo"]
)
parser.add_argument("--loop", action="store_true", help="Run throws indefinitely")
parser.add_argument("--initial_jv", type=float, default=0.0,
                    help="Raw action[0] for initial shoulder_pan [-1, 1]")
parser.add_argument("--final_jv", type=float, default=0.0,
                    help="Raw action[1] for final shoulder_pan [-1, 1]")
parser.add_argument("--releasing_time", type=float, default=0.4,
                    help="Fraction of throw duration at which to release [0.05, 1.0]")
parser.add_argument("--duration", type=float, default=0.3,
                    help="Throw trajectory time in seconds [0.1, 1.0]")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument("--playing_arm_side", type=str, default="right",
                    choices=["right", "left"], help="Which arm throws")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from tasks.throwing_env_cfg import ThrowingEnvCfg
from tasks.throwing_env import ThrowingEnv
from tasks.throw_primitive import execute_primitive_batched, map_action_to_params
from tasks.events import _set_gripper_state

SIDE = args_cli.playing_arm_side
EE_BODY = f"{SIDE}_wrist_3_link"

cfg = ThrowingEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.ik_solver = args_cli.ik
cfg.playing_arm_side = SIDE
cfg.disable_attachment = True
cfg.release_vel_threshold = float("inf")
cfg.release_at_step = 0
cfg.__post_init__()
cfg.episode_length_s = 60.0

from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

@configclass
class _NoTerminations:
    time_limit = DoneTerm(func=mdp.time_out, time_out="truncated")

cfg.terminations = _NoTerminations()

env = ThrowingEnv(cfg=cfg)
device = env.device
env.reset()
env._holding[:] = False
env._released[:] = False

robot = env.scene["robot"]
milk = env.scene["milk"]
target_obj = env.scene["target"]
origin = env.scene.env_origins[0].to(device)

if SIDE == "right":
    arm_patterns = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
else:
    arm_patterns = ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"]
arm_ids, _ = robot.find_joints(arm_patterns)

target_center_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/targetCenter",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)
target_center_marker = VisualizationMarkers(target_center_cfg)

action = torch.tensor(
    [[args_cli.initial_jv, args_cli.final_jv,
      args_cli.releasing_time, args_cli.duration]],
    device=device,
).expand(args_cli.num_envs, -1)

params = map_action_to_params(action[0:1], side=SIDE)

print(f"\n=== Throw Primitive Test ===")
print(f"  IK solver     : {args_cli.ik}")
print(f"  Arm            : {SIDE}")
print(f"  Num envs       : {args_cli.num_envs}")
print(f"  Loop           : {args_cli.loop}")
print(f"  Raw action     : [{args_cli.initial_jv:.3f}, {args_cli.final_jv:.3f}, "
      f"{args_cli.releasing_time:.3f}, {args_cli.duration:.3f}]")
print(f"  Mapped params  : initial_jv={params[0,0]:.3f} rad  final_jv={params[0,1]:.3f} rad  "
      f"rel_time={params[0,2]:.3f}  duration={params[0,3]:.3f} s")
print()

grasp_cache = {}
throw_number = 0

try:
    while simulation_app.is_running():
        throw_number += 1

        env.reset()
        env._holding[:] = False
        env._released[:] = False

        tgt_pos = target_obj.data.root_pos_w[0, :3]
        target_center_marker.visualize(
            translations=tgt_pos.unsqueeze(0).cpu().numpy()
        )

        cached = "prepos" in grasp_cache
        tag = "PRE-POSITIONED" if cached else "FULL SEQUENCE"
        print(f"[Throw #{throw_number}] [{tag}]  target=("
              f"{(tgt_pos[0]-origin[0]):.3f}, {(tgt_pos[1]-origin[1]):.3f}, "
              f"{(tgt_pos[2]-origin[2]):.3f})")

        result = execute_primitive_batched(
            env=env,
            actions=action,
            arm_joint_ids=arm_ids,
            ee_body_name=EE_BODY,
            gripper_set_fn=_set_gripper_state,
            side=SIDE,
            grasp_cache=grasp_cache,
        )

        for i in range(min(args_cli.num_envs, 10)):
            d = result["distances"][i].item()
            drop = result["dropped"][i].item()
            mp = result["milk_final_pos"][i]
            tp = result["target_pos"][i]
            status = "DROPPED" if drop else ("SUCCESS" if d < 0.15 else f"dist={d:.3f}m")
            print(f"  env[{i}]: milk=({mp[0]:+.3f},{mp[1]:+.3f},{mp[2]:+.3f})  "
                  f"target=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f})  {status}")

        mean_d = result["distances"].mean().item()
        print(f"  Mean distance: {mean_d:.3f}m\n")

        if not args_cli.loop:
            break

except KeyboardInterrupt:
    print(f"\n[Interrupted after {throw_number} throws]")

env.close()
simulation_app.close()
