#!/usr/bin/env python3
"""Throwing trajectory benchmark — compare IK solvers on throwing motions.

Generates three benchmark trajectories (arc, linear punch, sinusoidal lob)
anchored at the initial EE position. Only the throwing arm moves.

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

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Throwing IK solver benchmark")
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
parser.add_argument("--trajectory", type=str, default="arc",
                    choices=["arc", "linear", "lob"],
                    help="Benchmark trajectory type")
parser.add_argument("--amp", type=float, default=0.15, help="Amplitude (m)")
parser.add_argument("--period", type=int, default=60, help="Steps per cycle")
parser.add_argument("--step-delay", type=float, default=0.0, help="Sleep between steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import compute_pose_error
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from tasks.throwing_env_cfg import ThrowingEnvCfg
from tasks.throwing_env import ThrowingEnv

PLAYING_SIDE = "right"
BODY_TRACK = f"{PLAYING_SIDE}_wrist_3_link"
NUM_PATH_MARKERS = 31


def _setup_markers():
    path_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/throwPath",
        markers={
            "sphere_path": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.8, 1.0),
                ),
            ),
        },
    )
    release_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/releasePoint",
        markers={
            "sphere_release": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
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
    desired_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/desiredEE",
        markers={
            "sphere_desired": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.5, 0.0),
                ),
            ),
        },
    )
    return (
        VisualizationMarkers(path_cfg),
        VisualizationMarkers(release_cfg),
        VisualizationMarkers(target_cfg),
        VisualizationMarkers(ee_cfg),
        VisualizationMarkers(desired_cfg),
    )


def _compute_path_world(initial_pos, origin, traj_type, amp, period, device):
    x0, y0, z0 = initial_pos[0, 0].item(), initial_pos[0, 1].item(), initial_pos[0, 2].item()
    half = NUM_PATH_MARKERS // 2
    if traj_type == "arc":
        t = torch.linspace(0, period, NUM_PATH_MARKERS, device=device)
        x = torch.full((NUM_PATH_MARKERS,), x0, device=device)
        y = y0 + 0.15 * (t / period)
        sin_vals = torch.sin(math.pi * t / period)
        z = z0 + amp * sin_vals
    elif traj_type == "linear":
        t = torch.linspace(0, 1, NUM_PATH_MARKERS, device=device)
        x = torch.full((NUM_PATH_MARKERS,), x0, device=device)
        y = y0 + 0.25 * t
        z = z0 + 0.12 * t * (1.0 - t) * 4.0
    elif traj_type == "lob":
        t = torch.linspace(0, 1, NUM_PATH_MARKERS, device=device)
        x = torch.full((NUM_PATH_MARKERS,), x0, device=device)
        y = y0 + 0.2 * t
        phase = torch.linspace(0, math.pi, NUM_PATH_MARKERS, device=device)
        z = z0 + amp * torch.sin(phase)
    local = torch.stack([x, y, z], dim=-1)
    world = local + origin.unsqueeze(0)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(NUM_PATH_MARKERS, 1)
    return world, quat


def _ee_local(env, robot_name):
    robot = env.scene[robot_name]
    body_ids, _ = robot.find_bodies([BODY_TRACK])
    pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, body_ids[0]]
    return pos, quat


def _trajectory_target(step, initial_pos, traj_type, amp, period):
    """Return (x, y, z) target position for the given step.  Arc is in YZ plane."""
    progress = (step % period) / period
    x0, y0, z0 = initial_pos[0, 0].item(), initial_pos[0, 1].item(), initial_pos[0, 2].item()

    if traj_type == "arc":
        x = x0
        y = y0 + 0.15 * progress
        z = z0 + amp * math.sin(math.pi * progress)
    elif traj_type == "linear":
        x = x0
        y = y0 + 0.25 * progress
        z = z0 + 0.12 * progress * (1.0 - progress) * 4.0
    elif traj_type == "lob":
        x = x0
        y = y0 + 0.2 * progress
        z = z0 + amp * math.sin(math.pi * progress)
    return x, y, z


def _gripper_joint_name(side):
    return f"{'l' if side == 'left' else 'r'}gripper_finger_joint"


def run_benchmark(solver_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  IK Solver : {solver_name}")
    print(f"  Trajectory: {args_cli.trajectory}")
    print(f"  Amplitude : {args_cli.amp:.2f} m")
    print(f"  Period    : {args_cli.period} steps")
    print(f"  Loop mode : repeating indefinitely, Ctrl+C to stop")
    print(f"{'='*60}\n")

    cfg = ThrowingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.ik_solver = solver_name
    cfg.playing_arm_side = PLAYING_SIDE
    cfg.__post_init__()
    if hasattr(cfg.actions.arm, "scale"):
        cfg.actions.arm.scale = 1.0
    if hasattr(cfg.actions.arm, "position_scale"):
        cfg.actions.arm.position_scale = 1.0
        cfg.actions.arm.orientation_scale = 1.0

    env = ThrowingEnv(cfg=cfg)
    device = env.device
    env.reset()

    origin = env.scene.env_origins[0].to(device)

    init_pos, init_quat = _ee_local(env, "robot")
    print(f"  EE start pos (local): x={init_pos[0,0]:.3f} y={init_pos[0,1]:.3f} z={init_pos[0,2]:.3f}\n")

    path_markers, release_marker, target_marker, ee_marker, desired_marker = _setup_markers()

    path_world, path_quat = _compute_path_world(
        init_pos, origin, args_cli.trajectory, args_cli.amp, args_cli.period, device,
    )
    path_markers.visualize(path_world, path_quat)
    print("  [Markers] Path spheres (cyan), release (red), target (green), EE (yellow), desired (orange)", flush=True)

    release_idx = NUM_PATH_MARKERS // 2
    release_pos = path_world[release_idx : release_idx + 1]
    release_quat = path_quat[release_idx : release_idx + 1]
    release_marker.visualize(release_pos, release_quat)

    pos_errors_all = []
    rot_errors_all = []
    joint_jerks_all = []
    ee_positions = []
    gripper_states = []

    gripper_joint = _gripper_joint_name(PLAYING_SIDE)

    prev_joint_vel = None
    prev_joint_accel = None
    total_steps = 0
    t_last = time.time()

    COL_HDR = f"  {'step':>5}  {'ee_x':>7} {'ee_y':>7} {'ee_z':>7}  {'obj_x':>7} {'obj_y':>7} {'obj_z':>7}  {'grip':>6}  ik_cm"
    print(COL_HDR, flush=True)
    print(f"  {'-'*5}  {'-'*7} {'-'*7} {'-'*7}  {'-'*7} {'-'*7} {'-'*7}  {'-'*6}  {'-'*4}", flush=True)

    try:
        while simulation_app.is_running():
            total_steps += 1

            curr_pos, curr_quat = _ee_local(env, "robot")
            milk_pos = env.scene["milk"].data.root_pos_w[0, :3] - origin

            target_x, target_y, target_z = _trajectory_target(
                total_steps, init_pos, args_cli.trajectory, args_cli.amp, args_cli.period
            )

            target_pos_t = torch.tensor(
                [[target_x, target_y, target_z]], device=device,
            )
            pos_err, rot_err = compute_pose_error(
                curr_pos, curr_quat,
                target_pos_t, curr_quat.clone(),
                rot_error_type="axis_angle",
            )
            pos_errors_all.append(pos_err.norm(dim=-1).item())
            rot_errors_all.append(rot_err.norm(dim=-1).item())

            action = torch.cat([pos_err[0], rot_err[0]], dim=-1).unsqueeze(0)
            env.step(action)

            robot = env.scene["robot"]
            gripper_ids, _ = robot.find_joints([gripper_joint])
            gripper_pos = robot.data.joint_pos[0, gripper_ids[0]].item()

            ee_positions.append((curr_pos[0, 0].item(), curr_pos[0, 1].item(), curr_pos[0, 2].item()))
            gripper_states.append(gripper_pos)

            joint_vel = robot.data.joint_vel[0, :6]
            if prev_joint_vel is not None and prev_joint_accel is not None:
                joint_accel = (joint_vel - prev_joint_vel) / env.dt
                joint_jerk = (joint_accel - prev_joint_accel) / env.dt
                joint_jerks_all.append(joint_jerk.abs().sum().item())
                prev_joint_accel = joint_accel
            elif prev_joint_vel is not None:
                prev_joint_accel = (joint_vel - prev_joint_vel) / env.dt
            prev_joint_vel = joint_vel

            if total_steps % 10 == 0:
                target_pos_t = torch.tensor(
                    [[target_x, target_y, target_z]], device=device,
                )
                world_ee = curr_pos + origin.unsqueeze(0)
                world_desired = target_pos_t + origin.unsqueeze(0)
                ee_marker.visualize(world_ee, curr_quat)
                desired_marker.visualize(world_desired, curr_quat)

            if total_steps % 50 == 0:
                cp = curr_pos[0]
                mp = milk_pos
                ik_err_cm = torch.norm(curr_pos[0] - target_pos_t[0]).item() * 100
                print(
                    f"  {total_steps:<5}  "
                    f"{cp[0]:+7.3f} {cp[1]:+7.3f} {cp[2]:+7.3f}  "
                    f"{mp[0]:+7.3f} {mp[1]:+7.3f} {mp[2]:+7.3f}  "
                    f"{gripper_pos:+6.3f}  {ik_err_cm:4.1f}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[Interrupted by user]", flush=True)

    env.close()

    n = max(len(pos_errors_all), 1)
    return {
        "solver": solver_name,
        "trajectory": args_cli.trajectory,
        "steps": total_steps,
        "mean_pos_error_cm": sum(pos_errors_all) / n * 100,
        "max_pos_error_cm": max(pos_errors_all) * 100,
        "mean_rot_error_deg": math.degrees(sum(rot_errors_all) / n),
        "max_rot_error_deg": math.degrees(max(rot_errors_all)),
        "mean_joint_jerk": sum(joint_jerks_all) / max(len(joint_jerks_all), 1),
        "total_steps": n,
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
    run_benchmark(args_cli.ik)

simulation_app.close()
