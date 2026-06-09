#!/usr/bin/env python3
"""Multi-phase pick-and-throw IK benchmark.

Phases:
  SETUP    — spawn drink on table, arm at crane pose, gripper open
  APPROACH — EE moves XY above drink (Z stays at crane height)
  DESCEND  — EE lowers Z to grasp height
  GRASP    — close gripper, start kinematic attachment
  LIFT     — EE returns to crane pose (drink follows via attach)
  RAISE    — EE lifts Z-only by THROW_EXTEND_Z_OFFSET (preserves orientation)
  EXTEND   — EE moves to target XY at raised height (preserves orientation)
  SETTLE   — pause for object to settle in gripper after EXTEND
  ORIENT   — rotate gripper to aim at target (yaw-only), then lock orientation
  THROW    — wrist_2 snap from extended position,
              open gripper + cut attachment at peak snap
  FLIGHT   — watch drink fly/land, report distance to target, reset cycle

Usage:
  source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
  cd throwing_enviroment
  python scripts/test_ik_throwing.py --ik diffik
  python scripts/test_ik_throwing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-phase pick-and-throw IK benchmark")
parser.add_argument(
    "--ik",
    type=str,
    default="diffik",
    choices=["diffik", "osc", "rmpflow", "curobo"],
    help="IK solver",
)
parser.add_argument(
    "--compare",
    type=str,
    default=None,
    help="Run multiple solvers, e.g. --compare diffik:osc:curobo",
)
parser.add_argument("--output", type=str, default=None, help="CSV output path")
parser.add_argument("--step-delay", type=float, default=0.0, help="Sleep between steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import (
    compute_pose_error,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    quat_rotate,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from tasks.throwing_env_cfg import ThrowingEnvCfg, TABLE_Z
from tasks.throwing_env import ThrowingEnv
from tasks.events import _set_gripper_state

PLAYING_SIDE = "right"
EE_BODY = "right_wrist_3_link"
GRIPPER_JOINT = "rgripper_finger_joint"

DRINK_WORLD_X = 0.65
DRINK_WORLD_Y = 0.50
DRINK_WORLD_Z = 0.72

TARGET_WIDTH = 0.38
TARGET_LENGTH = 0.51
TARGET_HEIGHT = 0.27

BOTTLE_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.0])

IK_DEFAULT_SCALE = 0.8

GRASP_Z_OFFSET = 0.3

THROW_SNAP_RAD = 12.0
THROW_RELEASE_PROGRESS = 0.60
THROW_EXTEND_Z_OFFSET = 0.20
EXTEND_RATIO = 0.6

PHASE_STEPS = {
    "APPROACH": 60,
    "DESCEND": 100,
    "GRASP": 20,
    "LIFT": 60,
    "RAISE": 40,
    "EXTEND": 65,
    "SETTLE": 15,
    "ORIENT": 40,
    "THROW": 40,
    "FLIGHT": 300,
}


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


def _ee_state(env):
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies([EE_BODY])
    pos = robot.data.body_pos_w[:, body_ids[0]]
    quat = robot.data.body_quat_w[:, body_ids[0]]
    return pos, quat


def _drink_pos_to_table(env):
    """Reposition the drink bottle onto the table at the fixed spawn location."""
    milk = env.scene["milk"]
    origin = env.scene.env_origins[0].to(env.device)
    drink_pos_w = (
        torch.tensor(
            [[DRINK_WORLD_X, DRINK_WORLD_Y, DRINK_WORLD_Z]],
            device=env.device,
        )
        + origin
    )
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


def _throw_angle(progress):
    """Wrist_2 angle (rad) for throw: forward-only linear ramp.

    Tuned by THROW_SNAP_RAD. Constant angular velocity throughout.
    """
    return THROW_SNAP_RAD * progress


def _print_header():
    hdr = (
        f"  {'step':>5}  {'phase':>8}"
        f"  {'ee_x':>7} {'ee_y':>7} {'ee_z':>7}"
        f"  {'obj_x':>7} {'obj_y':>7} {'obj_z':>7}"
        f"  {'grip':>6}  {'ik_cm':>5}"
    )
    sep = (
        f"  {'-----':>5}  {'--------':>8}"
        f"  {'-----':>7} {'-----':>7} {'-----':>7}"
        f"  {'-----':>7} {'-----':>7} {'-----':>7}"
        f"  {'-----':>6}  {'-----':>5}"
    )
    print(hdr, flush=True)
    print(sep, flush=True)


def _print_step(step, phase_name, ee_pos_local, drink_pos_local, grip, ik_err_cm):
    print(
        f"  {step:<5}  {phase_name:>8}"
        f"  {ee_pos_local[0]:+7.3f} {ee_pos_local[1]:+7.3f} {ee_pos_local[2]:+7.3f}"
        f"  {drink_pos_local[0]:+7.3f} {drink_pos_local[1]:+7.3f} {drink_pos_local[2]:+7.3f}"
        f"  {grip:+6.3f}  {ik_err_cm:5.1f}",
        flush=True,
    )


def run_benchmark(
    solver_name: str, snap_rad: float = None, num_throws: int = None
) -> dict:
    _snap = snap_rad if snap_rad is not None else THROW_SNAP_RAD
    print(f"\n{'='*60}")
    print(f"  IK Solver  : {solver_name}")
    print(f"  Snap rad   : {_snap}")
    print(
        f"  Drink at   : ({DRINK_WORLD_X:.2f}, {DRINK_WORLD_Y:.2f}, {DRINK_WORLD_Z:.2f})"
    )
    print(
        f"  Phases     : APPROACH({PHASE_STEPS['APPROACH']}) DESCEND({PHASE_STEPS['DESCEND']})"
        f" GRASP({PHASE_STEPS['GRASP']}) LIFT({PHASE_STEPS['LIFT']})"
        f" RAISE({PHASE_STEPS['RAISE']})"
        f" EXTEND({PHASE_STEPS['EXTEND']}) SETTLE({PHASE_STEPS['SETTLE']})"
        f" ORIENT({PHASE_STEPS['ORIENT']})"
        f" THROW({PHASE_STEPS['THROW']})"
    )
    print(f"{'='*60}\n")

    cfg = ThrowingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.ik_solver = solver_name
    cfg.playing_arm_side = PLAYING_SIDE
    cfg.randomize_target = True
    cfg.release_at_step = 0
    cfg.release_vel_threshold = float("inf")
    cfg.disable_attachment = True
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
    target = env.scene["target"]
    origin = env.scene.env_origins[0].to(device)

    # Disable env's built-in attachment/release — script manages it manually
    env._holding[:] = False
    env._released[:] = False

    # Place drink on table, open gripper, let it settle
    _drink_pos_to_table(env)
    _set_gripper_state(robot, 0.0, env_ids)
    for _ in range(60):
        env.step(torch.zeros(1, 6, device=device))

    # Read ACTUAL settled drink position on the table (not hardcoded Z)
    drink_actual_w = milk.data.root_pos_w[0, :3].clone()
    drink_actual_local = drink_actual_w - origin
    dx, dy, dz = (
        drink_actual_local[0].item(),
        drink_actual_local[1].item(),
        drink_actual_local[2].item(),
    )

    # Read crane-pose EE position and orientation
    crane_pos, crane_quat = _ee_state(env)
    crane_quat = crane_quat.clone()
    crane_pos_local = crane_pos[0] - origin
    cx, cy, cz = (
        crane_pos_local[0].item(),
        crane_pos_local[1].item(),
        crane_pos_local[2].item(),
    )

    # EE targets directly above the drink, grasping above center
    ex, ey, ez = dx, dy, dz

    print(f"  Crane EE  : ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"  Grasp @   : ({ex:.3f}, {ey:.3f}, {ez + GRASP_Z_OFFSET:.3f})")
    print(f"  Target    : {target.data.root_pos_w[0, :3].tolist()}")
    print(f"  Drink @   : ({dx:.3f}, {dy:.3f}, {dz:.3f})")
    print()

    ee_marker, target_marker = _setup_markers()
    spawn_marker = _setup_spawn_area(origin, cfg.target_x_range, cfg.target_y_range)

    pos_errors_all = []
    rot_errors_all = []
    total_steps = 0
    throw_distances = []

    released = False
    throw_number = 0

    try:
        while simulation_app.is_running():
            total_steps = 0
            released = False
            throw_number += 1

            if throw_number > 1:
                env.reset()
                env._holding[:] = False
                env._released[:] = False
                _drink_pos_to_table(env)
                _set_gripper_state(robot, 0.0, env_ids)
                for _ in range(30):
                    env.step(torch.zeros(1, 6, device=device))
                drink_actual_w = milk.data.root_pos_w[0, :3].clone()
                drink_actual_local = drink_actual_w - origin
                dx, dy, dz = (
                    drink_actual_local[0].item(),
                    drink_actual_local[1].item(),
                    drink_actual_local[2].item(),
                )
                crane_pos, crane_quat = _ee_state(env)
                crane_quat = crane_quat.clone()
                crane_pos_local = crane_pos[0] - origin
                cx, cy, cz = (
                    crane_pos_local[0].item(),
                    crane_pos_local[1].item(),
                    crane_pos_local[2].item(),
                )
                ex, ey, ez = dx, dy, dz
                print(f"\n--- Throw #{throw_number} ---\n", flush=True)
                print(f"  Drink settled @ ({dx:.3f}, {dy:.3f}, {dz:.3f})", flush=True)

            _print_header()
            # ---- APPROACH (top-down: XY above drink at crane height) ----
            phase_name = "APPROACH"
            target_approach = torch.tensor(
                [[ex, ey, cz]],
                device=device,
            )
            for i in range(PHASE_STEPS["APPROACH"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                ee_pos, ee_quat = _ee_state(env)
                pos_err, rot_err = compute_pose_error(
                    ee_pos,
                    ee_quat,
                    target_approach + origin.unsqueeze(0),
                    ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                pos_errors_all.append(pos_err.norm(dim=-1).item())
                rot_errors_all.append(rot_err.norm(dim=-1).item())

                action = torch.cat([pos_err[0], rot_err[0] * 0.0], dim=-1).unsqueeze(0)
                env.step(action)

                if total_steps % 10 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    ik_err = (
                        torch.norm(ee_pos[0] + origin - target_approach - origin).item()
                        * 100
                    )
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        ik_err,
                    )

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- DESCEND (top-down: lower to above drink center) ----
            phase_name = "DESCEND"
            target_descend = torch.tensor(
                [[ex, ey, ez + GRASP_Z_OFFSET]],
                device=device,
            )
            for i in range(PHASE_STEPS["DESCEND"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                ee_pos, ee_quat = _ee_state(env)
                pos_err, rot_err = compute_pose_error(
                    ee_pos,
                    ee_quat,
                    target_descend + origin.unsqueeze(0),
                    ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                pos_errors_all.append(pos_err.norm(dim=-1).item())
                rot_errors_all.append(rot_err.norm(dim=-1).item())

                action = torch.cat([pos_err[0], rot_err[0] * 0.0], dim=-1).unsqueeze(0)
                env.step(action)

                if total_steps % 10 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    ik_err = (
                        torch.norm(ee_pos[0] + origin - target_descend - origin).item()
                        * 100
                    )
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        ik_err,
                    )

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- GRASP ----
            phase_name = "GRASP"
            print(f"  >>> CLOSING GRIPPER at step {total_steps} <<<", flush=True)
            grasp_steps = PHASE_STEPS["GRASP"]
            for i in range(grasp_steps):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                progress = i / (grasp_steps - 1) if grasp_steps > 1 else 1.0
                grip_target = 0.7 * progress
                _set_gripper_state(robot, grip_target, env_ids)

                action = torch.zeros(1, 6, device=device)
                env.step(action)

                if total_steps % 10 == 0:
                    ee_pos, ee_quat = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        0.0,
                    )

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- LIFT (return to crane pose) ----
            phase_name = "LIFT"
            target_lift = torch.tensor([[cx, cy, cz]], device=device)
            for i in range(PHASE_STEPS["LIFT"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                ee_pos, ee_quat = _ee_state(env)
                pos_err, rot_err = compute_pose_error(
                    ee_pos,
                    ee_quat,
                    target_lift + origin.unsqueeze(0),
                    ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                pos_errors_all.append(pos_err.norm(dim=-1).item())
                rot_errors_all.append(rot_err.norm(dim=-1).item())

                action = torch.cat([pos_err[0], rot_err[0] * 0.0], dim=-1).unsqueeze(0)
                env.step(action)

                if total_steps % 10 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    ik_err = (
                        torch.norm(ee_pos[0] + origin - target_lift - origin).item()
                        * 100
                    )
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        ik_err,
                    )

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- RAISE (Z-only lift to throw height) ----
            phase_name = "RAISE"
            print(f"  >>> RAISE at step {total_steps} <<<", flush=True)
            ee_pos_before, ee_quat = _ee_state(env)
            ee_local_before = ee_pos_before[0] - origin
            raise_target_local = torch.tensor(
                [ee_local_before[0].item(), ee_local_before[1].item(),
                 ee_local_before[2].item() + THROW_EXTEND_Z_OFFSET],
                device=device,
            )
            raise_target_w = raise_target_local + origin

            for i in range(PHASE_STEPS["RAISE"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, _ = compute_pose_error(
                    ee_pos,
                    ee_quat,
                    raise_target_w.unsqueeze(0),
                    ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                action = torch.cat(
                    [pos_err[0], torch.zeros(3, device=device)], dim=-1
                ).unsqueeze(0)
                env.step(action)
                if total_steps % 10 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    ik_err = torch.norm(ee_pos[0] - raise_target_w).item() * 100
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        ik_err,
                    )
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- EXTEND (move EE toward target in XY at raised height) ----
            phase_name = "EXTEND"
            print(f"  >>> EXTEND at step {total_steps} <<<", flush=True)
            ee_pos_before, ee_quat = _ee_state(env)
            ee_local_before = ee_pos_before[0] - origin
            tgt_w = target.data.root_pos_w[0, :3]
            tgt_local = tgt_w - origin
            extend_xy = ee_local_before[:2] + (tgt_local[:2] - ee_local_before[:2]) * EXTEND_RATIO
            extend_z = ee_local_before[2]
            extend_target_local = torch.tensor(
                [extend_xy[0].item(), extend_xy[1].item(), extend_z.item()],
                device=device,
            )
            extend_target_w = extend_target_local + origin

            for i in range(PHASE_STEPS["EXTEND"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, _ = compute_pose_error(
                    ee_pos,
                    ee_quat,
                    extend_target_w.unsqueeze(0),
                    ee_quat.clone(),
                    rot_error_type="axis_angle",
                )
                action = torch.cat(
                    [pos_err[0], torch.zeros(3, device=device)], dim=-1
                ).unsqueeze(0)
                env.step(action)
                if total_steps % 10 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    ik_err = torch.norm(ee_pos[0] - extend_target_w).item() * 100
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        ik_err,
                    )
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- SETTLE (pause for object to settle in gripper) ----
            phase_name = "SETTLE"
            print(f"  >>> SETTLE at step {total_steps} <<<", flush=True)
            for i in range(PHASE_STEPS["SETTLE"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                env.step(torch.zeros(1, 6, device=device))
                if total_steps % 10 == 0:
                    ee_pos, _ = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        0.0,
                    )
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- ORIENT (aim gripper at target and lock orientation) ----
            phase_name = "ORIENT"
            print(f"  >>> ORIENT at step {total_steps} <<<", flush=True)
            ee_orient_pos, _ = _ee_state(env)
            orient_target_pos = ee_orient_pos.clone()
            ee_orient_local = ee_orient_pos[0] - origin
            tgt_w = target.data.root_pos_w[0, :3]
            tgt_local = tgt_w - origin
            aim_yaw = torch.atan2(
                tgt_local[1] - ee_orient_local[1],
                tgt_local[0] - ee_orient_local[0],
            ) - math.pi / 2
            z_rot_quat = quat_from_euler_xyz(
                torch.tensor([0.0], device=device),
                torch.tensor([0.0], device=device),
                aim_yaw.unsqueeze(0),
            )
            orient_target_quat = quat_mul(z_rot_quat, crane_quat)

            for i in range(PHASE_STEPS["ORIENT"]):
                total_steps += 1
                if not simulation_app.is_running():
                    break
                ee_pos, ee_quat = _ee_state(env)
                pos_err, rot_err = compute_pose_error(
                    ee_pos,
                    ee_quat,
                    orient_target_pos,
                    orient_target_quat,
                    rot_error_type="axis_angle",
                )
                action = torch.cat([pos_err[0], rot_err[0]], dim=-1).unsqueeze(0)
                env.step(action)
                if total_steps % 10 == 0:
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        0.0,
                    )
                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # ---- THROW (wrist_2 snap from extended position) ----
            # Direct wrist_2 joint control: bypass IK, write joint targets directly
            phase_name = "THROW"
            print(f"  >>> THROW at step {total_steps} <<<", flush=True)

            arm_ids, arm_names = robot.find_joints(
                ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
            )
            wrist2_ids, _ = robot.find_joints(["right_wrist_2_joint"])
            wrist2_global = wrist2_ids[0]
            wrist2_local = next(i for i, g in enumerate(arm_ids) if g == wrist2_global)

            extend_end_joints = robot.data.joint_pos[0, arm_ids].clone().cpu()
            wrist2_start = extend_end_joints[wrist2_local].item()

            throw_phase_steps = PHASE_STEPS["THROW"]
            for i in range(throw_phase_steps):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                progress = i / (throw_phase_steps - 1) if throw_phase_steps > 1 else 1.0
                raw_angle = _snap * progress
                target_wrist2 = wrist2_start + raw_angle

                targets = extend_end_joints.clone()
                targets[wrist2_local] = target_wrist2
                robot.set_joint_position_target(
                    targets.unsqueeze(0).to(device), joint_ids=arm_ids, env_ids=env_ids
                )
                robot.write_data_to_sim()
                env.sim.step(
                    render=(
                        not args_cli.headless if hasattr(args_cli, "headless") else True
                    )
                )

                if progress >= THROW_RELEASE_PROGRESS and not released:
                    _set_gripper_state(robot, 0.0, env_ids)
                    released = True
                    print(
                        f"  >>> RELEASED at step {total_steps} (progress {progress:.2f}) <<<",
                        flush=True,
                    )

                if total_steps % 10 == 0:
                    ee_pos, ee_quat = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        0.0,
                    )

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

            # Re-sync IK controller after manual THROW stepping
            env.step(torch.zeros(1, 6, device=device))

            # ---- FLIGHT ----
            phase_name = "FLIGHT"
            flight_plan_steps = PHASE_STEPS["FLIGHT"]
            milk_settled = False
            settle_steps = 0

            for i in range(flight_plan_steps):
                total_steps += 1
                if not simulation_app.is_running():
                    break

                action = torch.zeros(1, 6, device=device)
                env.step(action)

                milk_vel = milk.data.root_lin_vel_w[0]
                milk_vel_norm = torch.norm(milk_vel).item()

                if not milk_settled:
                    if milk_vel_norm < 0.05:
                        settle_steps += 1
                    else:
                        settle_steps = 0

                    if settle_steps >= 30:
                        milk_settled = True
                        milk_final = milk.data.root_pos_w[0, :3]
                        tgt_final = target.data.root_pos_w[0, :3]
                        dist3d = torch.norm(milk_final - tgt_final).item()
                        throw_distances.append(dist3d)
                        print(
                            f"\n  >>> LANDED at step {total_steps}: 3D dist to target = {dist3d:.3f}m <<<\n",
                            flush=True,
                        )

                if total_steps % 10 == 0:
                    milk_local = milk.data.root_pos_w[0, :3] - origin
                    ee_pos, _ = _ee_state(env)
                    ee_local = ee_pos[0] - origin
                    _print_step(
                        total_steps,
                        phase_name,
                        ee_local,
                        milk_local,
                        _gripper_pos(env),
                        0.0,
                    )

                if args_cli.step_delay > 0:
                    time.sleep(args_cli.step_delay)

                if milk_settled and i >= 60:
                    break

            # Cycle ends — go back to SETUP for next throw
            if num_throws is not None and throw_number >= num_throws:
                break

    except KeyboardInterrupt:
        print(f"\n[Interrupted after {throw_number} throws]", flush=True)

    env.close()

    n = max(len(pos_errors_all), 1)
    return {
        "solver": solver_name,
        "snap_rad": _snap,
        "total_steps": total_steps,
        "throws": throw_number,
        "mean_pos_error_cm": sum(pos_errors_all) / n * 100,
        "max_pos_error_cm": max(pos_errors_all) * 100 if pos_errors_all else 0.0,
        "mean_rot_error_deg": math.degrees(sum(rot_errors_all) / n),
        "max_rot_error_deg": (
            math.degrees(max(rot_errors_all)) if rot_errors_all else 0.0
        ),
        "mean_throw_dist_m": sum(throw_distances) / max(len(throw_distances), 1),
        "best_throw_dist_m": min(throw_distances) if throw_distances else float("inf"),
    }


def print_metrics(metrics: dict):
    print(f"\n  {'Metric':<30} {'Value'}")
    print(f"  {'-'*30} {'-'*15}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.3f}")
        else:
            print(f"  {k:<30} {v}")


if args_cli.compare:
    solvers = args_cli.compare.split(":")
    all_metrics = []
    for solver in solvers:
        metrics = run_benchmark(solver)
        print_metrics(metrics)
        all_metrics.append(metrics)

    if args_cli.output and all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(args_cli.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[Metrics] Saved to {args_cli.output}")
elif args_cli.output:
    metrics = run_benchmark(args_cli.ik)
    print_metrics(metrics)
    with open(args_cli.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"\n[Metrics] Saved to {args_cli.output}")
else:
    metrics = run_benchmark(args_cli.ik)
    print_metrics(metrics)

simulation_app.close()
