#!/usr/bin/env python3
"""Ping pong swing demo — two robots alternate swinging with rackets.

Each robot's playing arm follows a sinusoidal X-axis swing motion.
The ball spawns at table centre + 30 cm, launched ±Y at 10 m/s (50/50).
When the ball falls below the table it respawns — no episode resets.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd pingpong_dual_arm
    python scripts/pingpong_swing.py --ik diffik
    python scripts/pingpong_swing.py --ik osc
"""

import math
import os
import sys
import time
import torch
import torch._dynamo  # noqa: F401 — lock torch before AppLauncher
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="Ping pong swing demo")
parser.add_argument(
    "--ik",
    type=str,
    default="diffik",
    choices=["diffik", "osc", "rmpflow", "curobo"],
    help="IK solver",
)
parser.add_argument("--steps", type=int, default=2000, help="Simulation steps")
parser.add_argument(
    "--step-delay",
    type=float,
    default=0.0,
    help="Sleep between steps for visual inspection",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.math import euler_xyz_from_quat
from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg
from tasks.pingpong_env import PingPongEnv

# ---------------------------------------------------------------------------
# Swing macro primitive
# ---------------------------------------------------------------------------
# Two robots swing their rackets on the X axis (left-right across the table)
# with a sinusoidal motion.  Robot A and B are 180° out of phase.

SWING_AMP = 0.15  # ±0.15 m  on X axis
SWING_PERIOD = 60  # steps per cycle (1.2 s at dt=0.02)
SWING_Y_A = -0.35  # nominal Y for robot A (over table)
SWING_Y_B = +0.35  # nominal Y for robot B (over table)
SWING_Z = 0.35  # nominal Z (above table top at ~0.235)
DIFFIK_GAIN = 0.3  # proportional gain for delta

TABLE_HEIGHT = 0.45  # matches env cfg TABLE_HEIGHT
BALL_SPAWN_Z = TABLE_HEIGHT + 0.15  # spawn just above table
BALL_SPEED = 4.0  # launch speed (m/s)
BALL_UP = 4.0  # upward velocity for arc
TABLE_SURFACE = 0.05  # ball respawn trigger z
BODY_TRACK = "right_wrist_3_link"


# ---------------------------------------------------------------------------
# Helper — read EE pose in local frame (position + Euler ZYX)
# ---------------------------------------------------------------------------
def _ee_local(env, robot_name):
    """Return EE position [x,y,z] and Euler angles [roll,pitch,yaw] in local frame."""
    robot = env.scene[robot_name]
    body_ids, _ = robot.find_bodies([BODY_TRACK])
    pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, body_ids[0]]
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    return pos, euler


# ---------------------------------------------------------------------------
# Helper — spawn ball at centre with random ±Y velocity
# ---------------------------------------------------------------------------
def _serve_ball(env):
    """Place ball at table centre + 30 cm, launch ±Y at 10 m/s."""
    ball = env.scene["ball"]
    origins = env.scene.env_origins
    N = 1

    x = torch.zeros(N, device=env.device)
    y = torch.zeros(N, device=env.device)
    z = torch.full((N,), BALL_SPAWN_Z, device=env.device)
    pos_global = torch.stack(
        [x + origins[:, 0], y + origins[:, 1], z + origins[:, 2]], dim=1
    )

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)

    # 50/50 left or right (±Y), slight upward arc to stay visible
    sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
    vy = torch.full((N,), sign * BALL_SPEED, device=env.device)
    vx = torch.zeros(N, device=env.device)
    vz = torch.full((N,), BALL_UP, device=env.device)
    lin_vel = torch.stack([vx, vy, vz], dim=1)
    ang_vel = torch.zeros(N, 3, device=env.device)

    ball.write_root_pose_to_sim(torch.cat([pos_global, q], dim=1))
    ball.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("  Ping Pong Swing Demo")
print(f"  IK solver : {args.ik}")
print(f"  Steps     : {args.steps}")
print("  Swing     : X-axis sine  ±0.30 m, alternating A/B")
print(f"  Ball      : {BALL_SPEED} m/s  ±Y, respawn on fall")
print("=" * 64 + "\n")

# --- Environment ---
print("[Setup] Creating environment (num_envs=1)...")
cfg = PingPongDualArmEnvCfg()
cfg.scene.num_envs = 1
cfg.ik_solver = args.ik

env = PingPongEnv(cfg=cfg)
device = env.device
print(f"  Action dim : {env.action_space.shape[1]}")
print(f"  Obs policy : {env.observation_space['policy'].shape}")

# Initial reset
print("[Setup] Initial scene reset...")
obs = env.reset()

# Initial ball serve
_serve_ball(env)

# --- Swing loop ---
swing_step = 0

print(f"\n[Loop] Starting {args.steps} steps.\n")

try:
    while swing_step < args.steps and simulation_app.is_running():
        # Read current EE positions
        pos_A, _ = _ee_local(env, "robot_A")
        pos_B, _ = _ee_local(env, "robot_B")

        # Compute sinusoidal targets (phase offset π between A and B)
        phase_A = 2.0 * math.pi * swing_step / SWING_PERIOD
        phase_B = phase_A + math.pi

        target_A_x = SWING_AMP * math.sin(phase_A)
        target_B_x = SWING_AMP * math.sin(phase_B)

        # Delta actions for each robot (X, Y, Z, roll, pitch, yaw)
        dx_A = (target_A_x - pos_A[0, 0].item()) * DIFFIK_GAIN
        dy_A = (SWING_Y_A - pos_A[0, 1].item()) * DIFFIK_GAIN
        dz_A = (SWING_Z - pos_A[0, 2].item()) * DIFFIK_GAIN

        dx_B = (target_B_x - pos_B[0, 0].item()) * DIFFIK_GAIN
        dy_B = (SWING_Y_B - pos_B[0, 1].item()) * DIFFIK_GAIN
        dz_B = (SWING_Z - pos_B[0, 2].item()) * DIFFIK_GAIN

        action = torch.zeros(1, env.action_space.shape[1], device=device)
        # arm_A: 6D delta [dx, dy, dz, droll, dpitch, dyaw]
        action[0, 0] = dx_A
        action[0, 1] = dy_A
        action[0, 2] = dz_A
        action[0, 3] = 0.0
        action[0, 4] = 0.0
        action[0, 5] = 0.0
        # arm_B: 6D delta
        action[0, 6] = dx_B
        action[0, 7] = dy_B
        action[0, 8] = dz_B
        action[0, 9] = 0.0
        action[0, 10] = 0.0
        action[0, 11] = 0.0

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Ball tracking & respawn
        ball = env.scene["ball"]
        ball_z_local = (ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).item()
        if ball_z_local < TABLE_SURFACE:
            _serve_ball(env)

        # Print status every 100 steps
        if swing_step % 100 == 0:
            bx = ball.data.root_pos_w[0, 0].item()
            by = ball.data.root_pos_w[0, 1].item()
            bv = torch.norm(ball.data.root_lin_vel_w[0]).item()
            print(
                f"  Step {swing_step:4d}  "
                f"ee_A=({pos_A[0,0]:+.2f},{pos_A[0,1]:+.2f},{pos_A[0,2]:+.2f})  "
                f"tgt_A=({target_A_x:+.2f})  "
                f"ee_B=({pos_B[0,0]:+.2f},{pos_B[0,1]:+.2f},{pos_B[0,2]:+.2f})  "
                f"tgt_B=({target_B_x:+.2f})  "
                f"ball=({bx:+.2f},{by:+.2f},{ball_z_local:+.2f})  vel={bv:.1f}",
                flush=True,
            )

        if args.step_delay > 0:
            time.sleep(args.step_delay)

        swing_step += 1

except KeyboardInterrupt:
    print("\n[Loop] Interrupted by user.")

print(f"\n=== Done ({swing_step} steps) ===")
env.close()
simulation_app.close()
