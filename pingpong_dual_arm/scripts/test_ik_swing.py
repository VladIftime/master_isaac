#!/usr/bin/env python3
"""Swing trajectory benchmark — compare IK solvers on a table-tennis stroke.

Generates a sinusoidal XY sweep at fixed Z — the racket oscillates left-to-right
across the table.  Only robot A's playing arm moves; robot B sits still.

The simulation runs indefinitely.  Press Ctrl+C to stop and see aggregate metrics.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd pingpong_dual_arm
    python scripts/test_ik_swing.py --ik diffik
    python scripts/test_ik_swing.py --ik curobo
    python scripts/test_ik_swing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv
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

parser = argparse.ArgumentParser(description="Swing IK solver benchmark")
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
    help="Run multiple solvers, e.g. --compare diffik:osc:rmpflow:curobo",
)
parser.add_argument("--output", type=str, default=None, help="CSV output path")
parser.add_argument("--amp", type=float, default=0.15, help="Swing amplitude (m) on X")
parser.add_argument(
    "--depth", type=float, default=0.12, help="Swing arc depth (m) on Y"
)
parser.add_argument("--period", type=int, default=60, help="Swing steps per cycle")
parser.add_argument("--step-delay", type=float, default=0.0, help="Sleep between steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import compute_pose_error

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg, TABLE_HEIGHT
from tasks.pingpong_env import PingPongEnv

PLAYING_SIDE = "right"
BODY_TRACK = f"{PLAYING_SIDE}_wrist_3_link"
NUM_PATH_MARKERS = 31


def _setup_path_markers():
    cfg_a = VisualizationMarkersCfg(
        prim_path="/Visuals/swingPathA",
        markers={
            "sphere_A": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.6, 1.0),
                ),
            ),
        },
    )
    cfg_b = VisualizationMarkersCfg(
        prim_path="/Visuals/swingPathB",
        markers={
            "sphere_B": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.3, 0.0),
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg_a), VisualizationMarkers(cfg_b)


def _compute_path_world(
    ee_x_local: float,
    ee_y_local: float,
    ee_z_local: float,
    origin: torch.Tensor,
    amp: float,
    depth: float,
    y_direction: float = 1.0,
    device: str = "cuda",
):
    xs = torch.linspace(-amp, amp, NUM_PATH_MARKERS, device=device)
    x_local = ee_x_local + xs
    y_local = ee_y_local + y_direction * depth * (1.0 - (xs / amp) ** 2)
    z_local = torch.full((NUM_PATH_MARKERS,), ee_z_local, device=device)
    local = torch.stack([x_local, y_local, z_local], dim=-1)
    world = local + origin.unsqueeze(0)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(NUM_PATH_MARKERS, 1)
    return world, quat


def _ee_pos(env, robot_name, wrist_link):
    robot = env.scene[robot_name]
    body_ids, _ = robot.find_bodies([wrist_link])
    pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    return pos


def _log_all_ee(env):
    al = _ee_pos(env, "robot_A", "left_wrist_3_link")
    ar = _ee_pos(env, "robot_A", "right_wrist_3_link")
    bl = _ee_pos(env, "robot_B", "left_wrist_3_link")
    br = _ee_pos(env, "robot_B", "right_wrist_3_link")
    return (
        f"A_left=({al[0,0]:+.3f},{al[0,1]:+.3f},{al[0,2]:+.3f})  "
        f"A_right=({ar[0,0]:+.3f},{ar[0,1]:+.3f},{ar[0,2]:+.3f})  "
        f"B_left=({bl[0,0]:+.3f},{bl[0,1]:+.3f},{bl[0,2]:+.3f})  "
        f"B_right=({br[0,0]:+.3f},{br[0,1]:+.3f},{br[0,2]:+.3f})"
    )


def _ee_local(env, robot_name):
    robot = env.scene[robot_name]
    body_ids, _ = robot.find_bodies([BODY_TRACK])
    pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, body_ids[0]]
    return pos, quat


def _joint_order(env, robot_name, action_cfg):
    robot = env.scene[robot_name]
    jids, jnames = robot.find_joints(action_cfg.joint_names)
    return jids, jnames


def run_indefinite(solver_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  IK Solver : {solver_name}")
    print(f"  Side      : {PLAYING_SIDE}")
    print(f"  Amplitude : {args_cli.amp:.2f} m  (X axis)")
    print(f"  Depth     : {args_cli.depth:.2f} m  (Y arc)")
    print(f"  Period    : {args_cli.period} steps")
    print(f"  Ctrl+C to stop")
    print(f"{'='*60}\n")

    cfg = PingPongDualArmEnvCfg()
    cfg.scene.num_envs = 1
    cfg.ik_solver = solver_name
    cfg.playing_arm_side = PLAYING_SIDE
    cfg.__post_init__()
    if hasattr(cfg.actions.arm_A, "scale"):
        cfg.actions.arm_A.scale = 1.0
        cfg.actions.arm_B.scale = 1.0
    if hasattr(cfg.actions.arm_A, "position_scale"):
        cfg.actions.arm_A.position_scale = 1.0
        cfg.actions.arm_A.orientation_scale = 1.0
        cfg.actions.arm_B.position_scale = 1.0
        cfg.actions.arm_B.orientation_scale = 1.0

    env = PingPongEnv(cfg=cfg)
    device = env.device

    env.reset()

    markers_a, markers_b = _setup_path_markers()
    origin_a = env.scene.env_origins[0]
    origin_b = env.scene.env_origins[0].clone()
    a_pos, _ = _ee_local(env, "robot_A")
    b_pos, _ = _ee_local(env, "robot_B")

    path_a, quat_a = _compute_path_world(
        a_pos[0, 0].item(),
        a_pos[0, 1].item(),
        a_pos[0, 2].item(),
        origin_a,
        args_cli.amp,
        args_cli.depth,
        y_direction=1.0,
        device=device,
    )
    path_b, quat_b = _compute_path_world(
        b_pos[0, 0].item(),
        b_pos[0, 1].item(),
        b_pos[0, 2].item(),
        origin_b,
        args_cli.amp,
        args_cli.depth,
        y_direction=-1.0,
        device=device,
    )
    markers_a.visualize(path_a, quat_a)
    markers_b.visualize(path_b, quat_b)
    print(
        f"  Path markers: {NUM_PATH_MARKERS} blue spheres for A, "
        f"{NUM_PATH_MARKERS} red spheres for B"
    )
    print(
        f"    A ee_y={a_pos[0,1]:.3f} ee_z={a_pos[0,2]:.3f} origin={origin_a.tolist()}"
    )
    print(
        f"    A path X range: [{path_a[:,0].min():.3f}, {path_a[:,0].max():.3f}]  "
        f"Y range: [{path_a[:,1].min():.3f}, {path_a[:,1].max():.3f}]  "
        f"Z range: [{path_a[:,2].min():.3f}, {path_a[:,2].max():.3f}]"
    )
    print(f"    A first 5: {path_a[:5].tolist()}")
    print(
        f"    B ee_y={b_pos[0,1]:.3f} ee_z={b_pos[0,2]:.3f} origin={origin_b.tolist()}"
    )
    print(
        f"    B path X range: [{path_b[:,0].min():.3f}, {path_b[:,0].max():.3f}]  "
        f"Y range: [{path_b[:,1].min():.3f}, {path_b[:,1].max():.3f}]  "
        f"Z range: [{path_b[:,2].min():.3f}, {path_b[:,2].max():.3f}]"
    )
    print(f"    B first 5: {path_b[:5].tolist()}")
    print()

    jids_a, jnames_a = _joint_order(env, "robot_A", cfg.actions.arm_A)
    jids_b, jnames_b = _joint_order(env, "robot_B", cfg.actions.arm_B)
    print(f"  robot_A targeted joints ({len(jnames_a)}): {jnames_a}")
    print(f"  robot_B targeted joints ({len(jnames_b)}): {jnames_b}")
    print()

    ee_pos, ee_quat = _ee_local(env, "robot_A")
    print(
        f"  robot_A EE start pos (local): "
        f"x={ee_pos[0,0]:.3f} y={ee_pos[0,1]:.3f} z={ee_pos[0,2]:.3f}\n"
    )

    pos_errors_all = []
    rot_errors_all = []
    joint_jerks_all = []

    prev_joint_vel = None
    prev_joint_accel = None

    swing_step = 0
    step_count = 0
    t_last = time.time()

    try:
        while simulation_app.is_running():
            swing_step += 1
            step_count += 1

            curr_a_pos, curr_a_quat = _ee_local(env, "robot_A")
            curr_b_pos, curr_b_quat = _ee_local(env, "robot_B")

            sin_a = math.sin(2.0 * math.pi * swing_step / args_cli.period)
            sin_b = math.sin(2.0 * math.pi * swing_step / args_cli.period + math.pi)

            # Fixed arc path — targets relative to initial EE position
            target_x_a = a_pos[0, 0].item() + args_cli.amp * sin_a
            target_y_a = a_pos[0, 1].item() + args_cli.depth * (1.0 - sin_a**2)
            target_x_b = b_pos[0, 0].item() + args_cli.amp * sin_b
            target_y_b = b_pos[0, 1].item() - args_cli.depth * (1.0 - sin_b**2)

            target_a = torch.tensor(
                [[target_x_a, target_y_a, a_pos[0, 2].item()]],
                device=device,
            )
            pos_a_err, rot_a_err = compute_pose_error(
                curr_a_pos,
                curr_a_quat,
                target_a,
                curr_a_quat.clone(),
                rot_error_type="axis_angle",
            )
            pos_errors_all.append(pos_a_err.norm(dim=-1).item())
            rot_errors_all.append(rot_a_err.norm(dim=-1).item())

            target_b = torch.tensor(
                [[target_x_b, target_y_b, b_pos[0, 2].item()]],
                device=device,
            )
            pos_b_err, rot_b_err = compute_pose_error(
                curr_b_pos,
                curr_b_quat,
                target_b,
                curr_b_quat.clone(),
                rot_error_type="axis_angle",
            )

            action = torch.zeros(1, env.action_space.shape[1], device=device)
            action[0, 0:6] = torch.cat([pos_a_err[0], rot_a_err[0]], dim=-1)
            action[0, 6:12] = torch.cat([pos_b_err[0], rot_b_err[0]], dim=-1)

            if solver_name == "osc":
                ra = env.scene["robot_A"]
                rb = env.scene["robot_B"]
                ra.set_joint_position_target(
                    ra.data.joint_pos[:, jids_a].clone(),
                    joint_ids=jids_a,
                )
                rb.set_joint_position_target(
                    rb.data.joint_pos[:, jids_b].clone(),
                    joint_ids=jids_b,
                )

            env.step(action)

            robot = env.scene["robot_A"]
            joint_vel = robot.data.joint_vel[0, 0:6]
            if prev_joint_vel is not None and prev_joint_accel is not None:
                joint_accel = (joint_vel - prev_joint_vel) / env.dt
                joint_jerk = (joint_accel - prev_joint_accel) / env.dt
                joint_jerks_all.append(joint_jerk.abs().sum().item())
                prev_joint_accel = joint_accel
            elif prev_joint_vel is not None:
                prev_joint_accel = (joint_vel - prev_joint_vel) / env.dt
            prev_joint_vel = joint_vel

            if args_cli.step_delay > 0:
                time.sleep(args_cli.step_delay)

            if step_count % (args_cli.period * 10) == 0:
                n = len(pos_errors_all)
                p_mean = sum(pos_errors_all[-1000:]) / min(1000, n) * 100
                r_mean = sum(rot_errors_all[-1000:]) / min(1000, n) * 180 / math.pi
                print(
                    f"  step {step_count:<6}  "
                    f"pos_err={p_mean:.2f}cm  "
                    f"rot_err={r_mean:.1f}deg  "
                    f"tgt_A_x={target_x_a:+.3f}  tgt_B_x={target_x_b:+.3f}",
                    flush=True,
                )

            if step_count % 10 == 0:
                now = time.time()
                elapsed = now - t_last
                t_last = now
                a_err = pos_a_err.norm(dim=-1).item() * 100
                b_err = pos_b_err.norm(dim=-1).item() * 100
                a_rot = rot_a_err.norm(dim=-1).item() * 180 / math.pi
                b_rot = rot_b_err.norm(dim=-1).item() * 180 / math.pi
                print(
                    f"  step {step_count:<6}  "
                    f"A_err={a_err:5.1f}cm/{a_rot:4.1f}deg  "
                    f"B_err={b_err:5.1f}cm/{b_rot:4.1f}deg  "
                    f"[{elapsed:.2f}s/10steps]  "
                    f"|  {_log_all_ee(env)}",
                    flush=True,
                )
                markers_a.visualize(path_a, quat_a)
                markers_b.visualize(path_b, quat_b)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    env.close()

    n = max(len(pos_errors_all), 1)
    return {
        "solver": solver_name,
        "steps": step_count,
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
    print(f"  {'Solver':<30} {metrics['solver']}")
    print(f"  {'Steps completed':<30} {metrics['steps']}")
    print(f"  {'Mean position error (cm)':<30} {metrics['mean_pos_error_cm']:.2f}")
    print(f"  {'Max position error (cm)':<30} {metrics['max_pos_error_cm']:.2f}")
    print(f"  {'Mean orientation error (deg)':<30} {metrics['mean_rot_error_deg']:.1f}")
    print(f"  {'Max orientation error (deg)':<30} {metrics['max_rot_error_deg']:.1f}")
    print(f"  {'Mean joint jerk (rad/s^3)':<30} {metrics['mean_joint_jerk']:.1f}")
    print(f"  {'Total steps':<30} {metrics['total_steps']}")
    print()


def run_compare(compare_str: str):
    solvers = compare_str.split(":")
    all_metrics = []
    for solver in solvers:
        metrics = run_indefinite(solver)
        print_metrics(metrics)
        all_metrics.append(metrics)

    if args_cli.output and all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(args_cli.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"[Metrics] Saved to {args_cli.output}")


if args_cli.compare:
    run_compare(args_cli.compare)
else:
    metrics = run_indefinite(args_cli.ik)
    print_metrics(metrics)
    if args_cli.output:
        with open(args_cli.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"[Metrics] Saved to {args_cli.output}")

simulation_app.close()
