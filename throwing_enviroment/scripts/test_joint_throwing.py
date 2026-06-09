#!/usr/bin/env python3
"""Joint-space pick-and-throw benchmark (Gazebo-style).

Uses task-space IK (via env.step) for reaching the drink, then switches to
Gazebo-style all-6-joints simultaneous interpolation for the throw. Object
is held by physics gripper friction (proven reliable).

Phases:
  SETTLE    — spawn drink on table, arm at crane pose, gripper open
  APPROACH  — EE moves XY above drink (task-space IK via env.step)
  DESCEND   — EE lowers Z to grasp height (task-space IK via env.step)
  GRASP     — close gripper gradually
  LIFT      — EE returns to crane pose (task-space IK via env.step)
  THROW     — ALL 6 joints interpolate simultaneously from lift-end to
              computed throw_end_joints (aimed at target, power-scaled).
              Gripper opens at RELEASE_PROGRESS. (Gazebo directlySetAllJoints)
  FLIGHT    — watch drink fly/land, report distance to target
  RETURN    — arm returns to crane pose

Usage:
  source ~/env_isaaclab/bin/activate
  cd throwing_enviroment
  python scripts/test_joint_throwing.py
  python scripts/test_joint_throwing.py --headless --num_throws 5
  python scripts/test_joint_throwing.py --throw_steps 30 --release_progress 0.5
  python scripts/test_joint_throwing.py --target_x 0.0 --target_y 1.0
"""

import argparse
import math
import os
import sys
import time

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Joint-space pick-and-throw (Gazebo-style)")
parser.add_argument("--num_throws", type=int, default=0, help="Number of throws (0=infinite)")
parser.add_argument("--throw_steps", type=int, default=40, help="Steps for THROW phase")
parser.add_argument("--release_progress", type=float, default=0.55, help="Fraction through THROW to release")
parser.add_argument("--target_x", type=float, default=None, help="Fixed target X (None=randomize)")
parser.add_argument("--target_y", type=float, default=None, help="Fixed target Y (None=randomize)")
parser.add_argument("--step-delay", type=float, default=0.0, help="Sleep between steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from isaaclab.utils.math import compute_pose_error
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from tasks.throwing_env_cfg import ThrowingEnvCfg, TABLE_Z
from tasks.throwing_env import ThrowingEnv
from tasks.events import _set_gripper_state

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PLAYING_SIDE = "right"
EE_BODY = "right_wrist_3_link"
GRIPPER_JOINT = "rgripper_finger_joint"
ARM_JOINT_PATTERNS = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]

DRINK_WORLD_X = 0.65
DRINK_WORLD_Y = 0.50
DRINK_WORLD_Z = 0.72

IK_DEFAULT_SCALE = 0.8
GRASP_Z_OFFSET = 0.3

NOMINAL_DIST = 1.0

SHOULDER_LIFT_DELTA = 0.44
ELBOW_DELTA = -1.47

PHASE_STEPS = {
    "SETTLE": 60,
    "APPROACH": 60,
    "DESCEND": 100,
    "GRASP": 20,
    "LIFT": 60,
    "THROW": 40,
    "FLIGHT": 300,
    "RETURN": 60,
}

RELEASE_PROGRESS = 0.40

ARM_THROW_DIRECTION_OFFSET = -math.pi / 2

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _ee_state(env):
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies([EE_BODY])
    pos = robot.data.body_pos_w[:, body_ids[0]]
    quat = robot.data.body_quat_w[:, body_ids[0]]
    return pos, quat


def _drink_pos_to_table(env):
    milk = env.scene["milk"]
    origin = env.scene.env_origins[0].to(env.device)
    drink_pos_w = torch.tensor(
        [[DRINK_WORLD_X, DRINK_WORLD_Y, DRINK_WORLD_Z]], device=env.device,
    ) + origin
    drink_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    milk.write_root_pose_to_sim(
        torch.cat([drink_pos_w, drink_quat], dim=-1),
        env_ids=torch.tensor([0], device=env.device),
    )
    milk.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=env.device),
        env_ids=torch.tensor([0], device=env.device),
    )


def _gripper_pos(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints([GRIPPER_JOINT])
    return robot.data.joint_pos[0, ids[0]].item()


def compute_throw_waypoints(throw_start_joints, target_local, robot_local, device):
    dx = target_local[0] - robot_local[0]
    dy = target_local[1] - robot_local[1]

    aim_angle = math.atan2(dx.item() if torch.is_tensor(dx) else dx,
                           dy.item() if torch.is_tensor(dy) else dy)

    dist = math.sqrt((dx.item() if torch.is_tensor(dx) else dx)**2 +
                     (dy.item() if torch.is_tensor(dy) else dy)**2)
    power = max(0.6, min(1.5, dist / NOMINAL_DIST))

    throw_end = throw_start_joints.clone()
    throw_end[0] = aim_angle - ARM_THROW_DIRECTION_OFFSET
    throw_end[1] = throw_start_joints[1] + SHOULDER_LIFT_DELTA * power
    throw_end[2] = throw_start_joints[2] + ELBOW_DELTA * power

    return throw_end, aim_angle, power


def _setup_markers():
    ee_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/eeCurrent",
        markers={
            "sphere_ee": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 0.0),
                ),
            ),
        },
    )
    target_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/targetMarker",
        markers={
            "sphere_target": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            ),
        },
    )
    return (
        VisualizationMarkers(ee_cfg),
        VisualizationMarkers(target_cfg),
    )


def _setup_spawn_area(origin, x_range: tuple, y_range: tuple):
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_width = x_max - x_min
    y_width = y_max - y_min
    z_surface = TABLE_Z + 0.001

    spawn_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/spawnArea",
        markers={
            "area_floor": sim_utils.CuboidCfg(
                size=(x_width, y_width, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.4, 1.0),
                    opacity=0.3,
                ),
            ),
            "corner": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.8, 1.0),
                ),
            ),
        },
    )
    spawn_marker = VisualizationMarkers(spawn_cfg)

    translations = np.array([
        [x_center, y_center, z_surface + 0.003],
        [x_min, y_min, z_surface],
        [x_min, y_max, z_surface],
        [x_max, y_min, z_surface],
        [x_max, y_max, z_surface],
    ]) + origin.unsqueeze(0).cpu().numpy()
    marker_indices = [0, 1, 1, 1, 1]

    spawn_marker.visualize(translations=translations, marker_indices=marker_indices)
    return spawn_marker


def _print_header():
    hdr = (
        f"  {'step':>5}  {'phase':>8}"
        f"  {'ee_x':>7} {'ee_y':>7} {'ee_z':>7}"
        f"  {'obj_x':>7} {'obj_y':>7} {'obj_z':>7}"
        f"  {'grip':>6}"
    )
    sep = (
        f"  {'-----':>5}  {'--------':>8}"
        f"  {'-----':>7} {'-----':>7} {'-----':>7}"
        f"  {'-----':>7} {'-----':>7} {'-----':>7}"
        f"  {'-----':>6}"
    )
    print(hdr, flush=True)
    print(sep, flush=True)


def _print_step(step, phase_name, ee_local, drink_local, grip):
    print(
        f"  {step:<5}  {phase_name:>8}"
        f"  {ee_local[0]:+7.3f} {ee_local[1]:+7.3f} {ee_local[2]:+7.3f}"
        f"  {drink_local[0]:+7.3f} {drink_local[1]:+7.3f} {drink_local[2]:+7.3f}"
        f"  {grip:+6.3f}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def run(num_throws: int = 0):
    PHASE_STEPS["THROW"] = args_cli.throw_steps
    release_progress = args_cli.release_progress
    headless = args_cli.headless

    print(f"\n{'='*60}")
    print(f"  Joint-Space Throwing (Gazebo-style)")
    print(f"  Throw steps    : {PHASE_STEPS['THROW']}")
    print(f"  Release @      : {release_progress:.0%}")
    print(f"  Drink at       : ({DRINK_WORLD_X:.2f}, {DRINK_WORLD_Y:.2f}, {DRINK_WORLD_Z:.2f})")
    print(f"  Throw deltas   : shoulder_lift={SHOULDER_LIFT_DELTA}, elbow={ELBOW_DELTA}")
    print(f"{'='*60}\n")

    cfg = ThrowingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.ik_solver = "diffik"
    cfg.playing_arm_side = PLAYING_SIDE
    cfg.release_at_step = 0
    cfg.release_vel_threshold = float("inf")
    cfg.disable_attachment = True

    if args_cli.target_x is not None and args_cli.target_y is not None:
        cfg.randomize_target = False
        cfg.target_x_range = (args_cli.target_x, args_cli.target_x)
        cfg.target_y_range = (args_cli.target_y, args_cli.target_y)
    else:
        cfg.randomize_target = True

    cfg.__post_init__()

    if hasattr(cfg.actions.arm, "scale"):
        cfg.actions.arm.scale = IK_DEFAULT_SCALE
    if hasattr(cfg.actions.arm, "position_scale"):
        cfg.actions.arm.position_scale = IK_DEFAULT_SCALE
        cfg.actions.arm.orientation_scale = IK_DEFAULT_SCALE

    env = ThrowingEnv(cfg=cfg)
    device = env.device
    env_ids = torch.tensor([0], device=device)
    env.reset()

    robot = env.scene["robot"]
    milk = env.scene["milk"]
    target_obj = env.scene["target"]
    origin = env.scene.env_origins[0].to(device)

    env._holding[:] = False
    env._released[:] = False

    arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)

    ee_marker, target_marker = _setup_markers()
    spawn_marker = _setup_spawn_area(origin, cfg.target_x_range, cfg.target_y_range)

    throw_distances = []
    throw_number = 0

    try:
        while simulation_app.is_running():
            throw_number += 1
            if num_throws > 0 and throw_number > num_throws:
                break

            print(f"\n{'─'*60}")
            print(f"  Throw #{throw_number}")
            print(f"{'─'*60}\n")

            # ── SETTLE ──────────────────────────────────────────────────
            if throw_number > 1:
                env.reset()
                env._holding[:] = False
                env._released[:] = False

            _drink_pos_to_table(env)
            _set_gripper_state(robot, 0.0, env_ids)
            for _ in range(PHASE_STEPS["SETTLE"]):
                env.step(torch.zeros(1, 6, device=device))

            drink_actual_w = milk.data.root_pos_w[0, :3].clone()
            drink_actual_local = drink_actual_w - origin
            dx = drink_actual_local[0].item()
            dy = drink_actual_local[1].item()
            dz = drink_actual_local[2].item()

            crane_pos, crane_quat = _ee_state(env)
            crane_pos_local = crane_pos[0] - origin
            cx = crane_pos_local[0].item()
            cy = crane_pos_local[1].item()
            cz = crane_pos_local[2].item()

            tgt_w = target_obj.data.root_pos_w[0, :3]
            tgt_local = tgt_w - origin

            print(f"  Crane EE  : ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            print(f"  Drink @   : ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            print(f"  Target    : ({tgt_local[0]:.3f}, {tgt_local[1]:.3f}, {tgt_local[2]:.3f})")
            print()

            _print_header()
            total_steps = 0
            released = False

            # ── APPROACH (task-space IK: XY above drink at crane Z) ─────
            phase_name = "APPROACH"
            target_approach = torch.tensor([[dx, dy, cz]], device=device)
            for i in range(PHASE_STEPS["APPROACH"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, _ = compute_pose_error(
                    ee_pos, ee_quat,
                    target_approach + origin.unsqueeze(0), ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                action = torch.cat([pos_err[0], torch.zeros(3, device=device)], dim=-1).unsqueeze(0)
                env.step(action)
                if total_steps % 5 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(total_steps, phase_name, ee_local, milk_local, _gripper_pos(env))
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ── DESCEND (task-space IK: lower to grasp height) ──────────
            phase_name = "DESCEND"
            target_descend = torch.tensor([[dx, dy, dz + GRASP_Z_OFFSET]], device=device)
            for i in range(PHASE_STEPS["DESCEND"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, _ = compute_pose_error(
                    ee_pos, ee_quat,
                    target_descend + origin.unsqueeze(0), ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                action = torch.cat([pos_err[0], torch.zeros(3, device=device)], dim=-1).unsqueeze(0)
                env.step(action)
                if total_steps % 5 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(total_steps, phase_name, ee_local, milk_local, _gripper_pos(env))
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ── GRASP (close gripper gradually) ─────────────────────────
            phase_name = "GRASP"
            print(f"  >>> GRASP at step {total_steps} <<<", flush=True)
            grasp_steps = PHASE_STEPS["GRASP"]
            for i in range(grasp_steps):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                progress = i / (grasp_steps - 1) if grasp_steps > 1 else 1.0
                _set_gripper_state(robot, 0.7 * progress, env_ids)
                env.step(torch.zeros(1, 6, device=device))
                if total_steps % 5 == 0:
                    ee_pos, _ = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(total_steps, phase_name, ee_local, milk_local, _gripper_pos(env))
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ── LIFT (task-space IK: return to crane pose) ──────────────
            phase_name = "LIFT"
            target_lift = torch.tensor([[cx, cy, cz]], device=device)
            for i in range(PHASE_STEPS["LIFT"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, _ = compute_pose_error(
                    ee_pos, ee_quat,
                    target_lift + origin.unsqueeze(0), ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                action = torch.cat([pos_err[0], torch.zeros(3, device=device)], dim=-1).unsqueeze(0)
                env.step(action)
                if total_steps % 5 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(total_steps, phase_name, ee_local, milk_local, _gripper_pos(env))
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ── THROW (Gazebo-style: all 6 joints simultaneous) ─────────
            phase_name = "THROW"
            print(f"  >>> THROW at step {total_steps} <<<", flush=True)

            throw_start_joints = robot.data.joint_pos[0, arm_ids].clone()

            robot_local = torch.tensor([0.0, 0.0, 0.0], device=device)
            throw_end_joints, aim_angle, power = compute_throw_waypoints(
                throw_start_joints, tgt_local, robot_local, device)

            print(f"      aim={math.degrees(aim_angle):.1f}° power={power:.2f}", flush=True)
            print(f"      start: {throw_start_joints.cpu().tolist()}", flush=True)
            print(f"      end  : {throw_end_joints.cpu().tolist()}", flush=True)

            throw_phase_steps = PHASE_STEPS["THROW"]
            release_step = int(release_progress * throw_phase_steps)
            for i in range(throw_phase_steps):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                robot.set_joint_position_target(
                    throw_end_joints.unsqueeze(0), joint_ids=arm_ids, env_ids=env_ids)
                robot.write_data_to_sim()
                env.sim.step(render=not headless)

                if i >= release_step and not released:
                    _set_gripper_state(robot, 0.0, env_ids)
                    released = True
                    print(f"      >>> RELEASED at step {i}/{throw_phase_steps} <<<", flush=True)

                if total_steps % 5 == 0:
                    ee_pos, _ = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(total_steps, phase_name, ee_local, milk_local, _gripper_pos(env))

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            if not released:
                _set_gripper_state(robot, 0.0, env_ids)
                released = True
                print(f"      >>> RELEASED (end of throw) <<<", flush=True)

            env.step(torch.zeros(1, 6, device=device))

            # ── FLIGHT ──────────────────────────────────────────────────
            phase_name = "FLIGHT"
            milk_settled = False
            settle_count = 0
            distance = float("inf")

            for i in range(PHASE_STEPS["FLIGHT"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                env.step(torch.zeros(1, 6, device=device))

                milk_vel = milk.data.root_lin_vel_w[0]
                milk_vel_norm = torch.norm(milk_vel).item()

                if not milk_settled:
                    if milk_vel_norm < 0.05:
                        settle_count += 1
                    else:
                        settle_count = 0

                    if settle_count >= 30:
                        milk_settled = True
                        milk_final = milk.data.root_pos_w[0, :3]
                        tgt_final = target_obj.data.root_pos_w[0, :3]
                        distance = torch.norm(milk_final - tgt_final).item()
                        print(f"\n  >>> LANDED at step {total_steps}: 3D dist = {distance:.3f}m <<<\n", flush=True)

                if total_steps % 5 == 0:
                    ee_pos, _ = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(total_steps, phase_name, ee_local, milk_local, _gripper_pos(env))

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

                if milk_settled and i >= 60:
                    break

            throw_distances.append(distance)

            # ── RETURN (back to crane) ──────────────────────────────────
            phase_name = "RETURN"
            crane_target = torch.tensor([[cx, cy, cz]], device=device)
            for i in range(PHASE_STEPS["RETURN"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, _ = compute_pose_error(
                    ee_pos, ee_quat,
                    crane_target + origin.unsqueeze(0), ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                action = torch.cat([pos_err[0], torch.zeros(3, device=device)], dim=-1).unsqueeze(0)
                env.step(action)

            if num_throws > 0 and throw_number >= num_throws:
                break

    except KeyboardInterrupt:
        print(f"\n[Interrupted after {throw_number} throws]", flush=True)

    env.close()

    # ── SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS ({len(throw_distances)} throws)")
    print(f"  {'Throw':>6}  {'Dist(m)':>10}")
    print(f"  {'-'*6}  {'-'*10}")
    for i, d in enumerate(throw_distances):
        print(f"  {i+1:>6}  {d:>10.3f}")
    if throw_distances:
        valid = [d for d in throw_distances if d != float("inf")]
        if valid:
            print(f"\n  Mean: {sum(valid)/len(valid):.3f}m")
            print(f"  Best: {min(valid):.3f}m")
    print(f"{'='*60}")


run(num_throws=args_cli.num_throws)
simulation_app.close()
