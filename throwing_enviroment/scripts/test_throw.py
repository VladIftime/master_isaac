#!/usr/bin/env python3
"""Single throw test — kinematic hold, ballistic release, observe landing.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/test_throw.py --ik diffik
    python scripts/test_throw.py --ik diffik --headless
"""

import argparse
import math
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

parser = argparse.ArgumentParser(description="Single throw test.")
parser.add_argument("--ik", type=str, default="diffik", choices=["diffik", "osc", "rmpflow", "curobo"])
parser.add_argument("--amp", type=float, default=0.3, help="Throwing arc amplitude (m)")
parser.add_argument("--period", type=int, default=60, help="Steps for full throwing arc")
parser.add_argument("--release-at", type=int, default=30, help="Step to release (0=auto: period/2)")
parser.add_argument("--loop", action="store_true", help="Run throws indefinitely, resetting after each")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import compute_pose_error
from tasks.throwing_env_cfg import ThrowingEnvCfg
from tasks.throwing_env import ThrowingEnv

SIDE = "right"
EE_BODY = f"{SIDE}_wrist_3_link"

cfg = ThrowingEnvCfg()
cfg.scene.num_envs = 1
cfg.ik_solver = args_cli.ik
cfg.playing_arm_side = SIDE
cfg.release_min_steps = 1
cfg.release_vel_threshold = 999.0  # manual release only
cfg.__post_init__()

print(f"\n=== Single Throw Test ===")
print(f"  IK: {args_cli.ik}  Amp: {args_cli.amp}m  Period: {args_cli.period}")
print(f"  Release at: {args_cli.release_at}")
print()

env = ThrowingEnv(cfg=cfg)
device = env.device
env.reset()

robot = env.scene["robot"]
milk = env.scene["milk"]
target = env.scene["target"]
origin = env.scene.env_origins[0].to(device)

body_ids, _ = robot.find_bodies([EE_BODY])
init_pos = (robot.data.body_pos_w[:, body_ids[0]] - origin).clone()
init_quat = robot.data.body_quat_w[:, body_ids[0]].clone()
x0, y0, z0 = init_pos[0, 0].item(), init_pos[0, 1].item(), init_pos[0, 2].item()

print(f"  EE:   ({x0:.3f}, {y0:.3f}, {z0:.3f})")
print(f"  Target: {target.data.root_pos_w[0,:3].tolist()}")
print(f"  Bottle: {milk.data.root_pos_w[0,:3].tolist()}")
print()

release_step = args_cli.release_at if args_cli.release_at > 0 else args_cli.period // 2

print(f"{'step':>5}  {'ee_y':>7} {'ee_z':>7}  {'obj_y':>7} {'obj_z':>7}  {'v_obj':>6}  {'dist3d':>7}  state")
print(f"{'-----':>5}  {'-----':>7} {'-----':>7}  {'-----':>7} {'-----':>7}  {'-----':>6}  {'-----':>7}  -----")

throw_number = 0

def run_single_throw():
    global step, released, throw_number
    step = 0
    released = False
    throw_number += 1

    while simulation_app.is_running():
        step += 1

        curr_pos = robot.data.body_pos_w[:, body_ids[0]] - origin
        curr_quat = robot.data.body_quat_w[:, body_ids[0]]

        progress = step / args_cli.period
        target_y = y0 + 0.40 * min(progress, 1.0)
        target_z = z0 + args_cli.amp * math.sin(math.pi * min(progress, 1.0))
        target_x = x0

        target_pos_t = torch.tensor([[target_x, target_y, target_z]], device=device)
        pos_err, rot_err = compute_pose_error(
            curr_pos, curr_quat, target_pos_t, curr_quat.clone(), rot_error_type="axis_angle",
        )
        action = torch.cat([pos_err[0], rot_err[0]], dim=-1).unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated[0].item() or truncated[0].item():
            milk_final = milk.data.root_pos_w[0, :3]
            tgt_final = target.data.root_pos_w[0, :3]
            dist_3d = torch.norm(milk_final - tgt_final).item()
            print(f"\n  Throw #{throw_number} complete. 3D distance: {dist_3d:.3f}m\n", flush=True)
            if args_cli.loop:
                env.reset()
                return True
            return False

        # Release
        if step >= release_step and not released:
            env._holding[0] = False
            env._released[0] = True
            gripper_ids, _ = robot.find_joints(["rgripper_finger_joint"])
            robot.set_joint_position_target(
                torch.zeros(1, 1, device=device), joint_ids=gripper_ids,
            )
            released = True
            print("  >>> RELEASED <<<", flush=True)

        # Read state
        milk_pos = milk.data.root_pos_w[0, :3] - origin
        milk_vel = milk.data.root_lin_vel_w[0]
        dist_3d = torch.norm(milk.data.root_pos_w[0, :3] - target.data.root_pos_w[0, :3]).item()
        ee_pos = curr_pos[0]
        obj_speed = torch.norm(milk_vel).item()

        state_str = "HOLD" if not released else "FLY"

        if step % 5 == 0 or step == 1:
            print(
                f"  {step:>5}  {ee_pos[1]:+7.3f} {ee_pos[2]:+7.3f}  "
                f"{milk_pos[1]:+7.3f} {milk_pos[2]:+7.3f}  "
                f"{obj_speed:+6.3f}  {dist_3d:+7.3f}  {state_str}",
                flush=True,
            )

        if step > 2000:
            print("\n  Timeout.", flush=True)
            return False

try:
    if args_cli.loop:
        while simulation_app.is_running():
            run_single_throw()
    else:
        run_single_throw()
except KeyboardInterrupt:
    print(f"\n[Interrupted after {throw_number} throws]", flush=True)

env.close()
simulation_app.close()
