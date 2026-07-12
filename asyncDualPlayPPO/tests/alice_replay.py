#!/usr/bin/env python3
"""
Alice-Replay Diagnostic — demo achievability under contact-rich pushing.

ASP's theoretical guarantee (Sukhbaatar 2018, Plappert 2021) is that Alice's
own action sequence, replayed from Bob's clean reset, should reach the goal she
proposed — every goal is achievable by construction. In contact-rich pushing
with discrete macro-actions and stochastic PhysX contact, this does NOT hold.

This diagnostic loads a trained ASP checkpoint, reads Alice's demonstrated
trajectories from the ABC buffer (saved to ``abc_buffer.pt``), replays each
action sequence from a fresh reset, and reports the **Alice-replay success
rate**: how often replaying the goal-setting agent's actions from the
goal-solving agent's start state actually reaches the proposed goal.

A low replay rate shows the ASP premise itself breaks in this domain — a
failure at the level of the assumption, not the outcome.

Usage (Isaac container, from the project root):
  isaaclab.sh -p tests/alice_replay.py \
      --chkpt-bob runs/<exp>/bob/model_best.pt \
      --num-replays 50 --headless [--csv results.csv]

Requires: the ABC buffer at ``.../bob/abc_buffer.pt`` (written during training).
Works for T-block (E, G, I) and disc (F, H) ASP models; char_length is
auto-detected from the checkpoint path (``dpose`` → 0.07, ``disc`` → 0.0).

Output:
  - Per-replay: push count, pos/rot error, stop reason (success, max_pushes, …).
  - Summary: reach rate (Alice-replay SR) and metrics.
  - Optional CSV: one row per replay.
"""

import argparse
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, ".."))

# ── cuRobo MUST be imported before AppLauncher on HPC ────────────────────────
os.environ["CUROBO_LOG_LEVEL"] = "ERROR"
import curobo.util_file
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.math import Pose as CuroboPose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Alice-Replay Diagnostic for ASP models.")
parser.add_argument("--chkpt_bob", type=str, required=True,
                    help="Path to Bob checkpoint (e.g. runs/.../bob/model_best.pt)")
parser.add_argument("--chkpt_alice", type=str, default=None,
                    help="Path to Alice checkpoint (auto: sibling of chkpt_bob)")
parser.add_argument("--num_replays", type=int, default=50,
                    help="Number of ABC-buffer trajectories to replay")
parser.add_argument("--char_length", type=float, default=None,
                    help="Characteristic length L (auto-detect from path: dpose→0.07, disc→0.0)")
parser.add_argument("--rot_threshold", type=float, default=0.2,
                    help="Rotation success threshold in radians")
parser.add_argument("--csv", type=str, default=None,
                    help="Save per-replay results to CSV")
parser.add_argument("--no_recalc", action="store_false", dest="recalc",
                    help="Disable per-step d_pose observation transform")
parser.set_defaults(recalc=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import copy
import random as _py_random
import numpy as np
import torch
import yaml

from asyncDualPlayPPO.tasks.utils.wrapper_push_asp import PushASPEnvWrapper, _OBS_ROBOT_DIM
from asyncDualPlayPPO.tasks.utils.action_push_relative import (
    decode_push_action_relative,
    TBLOCK_MIN_R, TBLOCK_MAX_R,
    DISC_MIN_R, DISC_MAX_R,
)
from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints
from asyncDualPlayPPO.tasks.utils.validation_configs import get_test_config
from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
from asyncDualPlayPPO.tasks.push_task_curobo_disc import PushTaskCuRoboDiscEnvCfg


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff[..., 0] = diff[..., 0].clamp(max=torch.pi).unsqueeze(-1)
    return diff.max(dim=-1)[0]


def _euler_xyz_to_quat_local(euler):
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)


def main():
    # ── auto-detect char_length + object type ─────────────────────────────────
    _ckpt_lower = args.chkpt_bob.lower()
    _is_disc = "disc" in _ckpt_lower or "taspf" in _ckpt_lower
    if args.char_length is None:
        args.char_length = 0.0 if _is_disc else 0.07
    print(f"[Diag] auto-detected char_length={args.char_length} ({'disc' if _is_disc else 'T-block'}) "
          f"from path {os.path.basename(args.chkpt_bob)}")

    # ── PPO config ─────────────────────────────────────────────────────────────
    ppo_cfg_path = os.path.join(PROJ_ROOT, "cfg/ppo/ppo_continuous.yaml")
    with open(ppo_cfg_path) as _f:
        ppo_cfg = yaml.safe_load(_f)

    num_cat_dims = 4
    num_bins = 21

    # ── Env ────────────────────────────────────────────────────────────────────
    _EnvCfg = PushTaskCuRoboDiscEnvCfg if _is_disc else PushTaskCuRoboEnvCfg
    env_cfg = _EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ], scale=1.0, use_default_offset=False,
    )
    if env_cfg.scene.cube is not None:
        env_cfg.scene.cube = None
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushASPEnvWrapper(
        env=base_env, alice_pushes=5, bob_pushes=30,
        max_goals_per_episode=1, num_objects=1,
        dpose_obs=True, char_length=args.char_length,
        dpose_threshold=0.055 if not _is_disc else 0.05,
        obj_spawn_z=0.03 if _is_disc else 0.05,
        obj_settled_z=0.03 if _is_disc else 0.05,
        device=device,
    )
    print("[Diag] Environment ready.")

    # ── cuRobo IK ──────────────────────────────────────────────────────────────
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(_robot_cfg, world_model=None, tensor_args=_tensor_args)
    _ik_config.solver.newton_optimizer.n_iters = 30
    _ik_config.solver.newton_optimizer.inner_iters = 10
    ik_solver = IKSolver(_ik_config)
    print("[Diag] IK solver ready.")

    # ── Scene handles ──────────────────────────────────────────────────────────
    _robot_scene = base_env.scene["robot"]
    _arm_jids, _arm_jnames = _robot_scene.find_joints("shoulder_pan_joint")
    _calib_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
    _ee_cfg = SceneEntityCfg("ee_frame", body_names=["robotiq_arg2f_base_link"])
    _lf_id = base_env.scene.body_ids["robot"]["robotiq_arg2f_base_link"]
    _WS_X, _WS_Y = (-0.50, 0.50), (0.25, 0.70)
    _WS_Z = (0.25, 0.55)
    _QUAT_TOOL_DOWN = torch.tensor([[1, 0, 0, 0]], device=device)
    _TOTAL_IK_ERROR = torch.tensor([[0, 0, 0.003]], device=device)

    def _tcp_pos_local():
        return base_env.scene["robot"].data.body_pos_w[:, _lf_id] - base_env.scene.env_origins

    # ── Load Bob ───────────────────────────────────────────────────────────────
    _chkpt = torch.load(args.chkpt_bob, map_location="cpu", weights_only=False)
    _chkpt_state = _chkpt.get("model_state_dict", _chkpt)
    _pi_w0 = _chkpt_state.get("pi_encoder.obj_encoder.0.weight")
    _is_noge = _pi_w0 is not None and _pi_w0.shape[1] == 22
    _has_goal_encoder = not _is_noge

    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_pi_encoder"] = True
    bob_cfg["policy"]["use_multicategorical"] = True
    bob_cfg["policy"]["use_lstm"] = True
    bob_cfg["policy"]["use_goal_encoder"] = _has_goal_encoder
    bob_cfg["policy"]["num_cat_dims"] = num_cat_dims
    bob_cfg["policy"]["num_bins"] = num_bins
    bob_cfg["policy"]["robot_state_dim"] = 6
    if _has_goal_encoder:
        bob_cfg["policy"]["num_objects"] = 1
        bob_cfg["policy"]["goal_embed_dim"] = 8
    else:
        bob_cfg["policy"]["pi_obj_dim"] = 22

    bob_ppo = PPOABC(vec_env=env, cfg_train=bob_cfg, device=device,
                     sampler="sequential", log_dir="/tmp/alice_replay_bob",
                     asymmetric=False)
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space
    bob_ppo.action_space = bob_ppo.action_space
    bob_ppo.desired_kl = None
    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape, bob_ppo.state_space.shape,
        bob_ppo.action_space.shape, bob_ppo.init_noise_std,
        bob_ppo.model_cfg, asymmetric=False,
    ).to(device)
    bob_ppo.load(args.chkpt_bob)
    bob_ppo.actor_critic.eval()
    print(f"[Diag] Loaded Bob {'(no GE)' if _is_noge else ''}")

    # ── Load Alice + ABC buffer ────────────────────────────────────────────────
    _bob_dir = os.path.dirname(args.chkpt_bob)
    if args.chkpt_alice is None:
        args.chkpt_alice = os.path.join(os.path.dirname(_bob_dir), "alice", "model_best.pt")
        if not os.path.isfile(args.chkpt_alice):
            args.chkpt_alice = os.path.join(os.path.dirname(_bob_dir), "alice", "latest_checkpoint.pt")

    alice_cfg = copy.deepcopy(ppo_cfg["params"])
    alice_cfg["policy"]["use_pi_encoder"] = False
    alice_cfg["policy"]["use_multicategorical"] = True
    alice_cfg["policy"]["use_lstm"] = True
    alice_cfg["policy"]["use_goal_encoder"] = False
    alice_cfg["policy"]["num_cat_dims"] = num_cat_dims
    alice_cfg["policy"]["num_bins"] = num_bins
    alice_cfg["policy"]["robot_state_dim"] = 6

    alice_ppo = PPO(vec_env=env, cfg_train=alice_cfg, device=device,
                    sampler="sequential", log_dir="/tmp/alice_replay_alice",
                    asymmetric=False)
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space
    alice_ppo.action_space = alice_ppo.action_space
    alice_ppo.desired_kl = None
    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape, alice_ppo.state_space.shape,
        alice_ppo.action_space.shape, alice_ppo.init_noise_std,
        alice_ppo.model_cfg, asymmetric=False,
    ).to(device)
    if os.path.isfile(args.chkpt_alice):
        alice_ppo.load(args.chkpt_alice)
        alice_ppo.actor_critic.eval()
        print(f"[Diag] Loaded Alice from {args.chkpt_alice}")
    else:
        print(f"[WARN] Alice checkpoint not found at {args.chkpt_alice}; "
              "Alice weights not restored (analysis continues).")

    # ── ABC buffer ─────────────────────────────────────────────────────────────
    _abc_path = os.path.join(_bob_dir, "abc_buffer.pt")
    bob_ppo.abc_buffer = None
    n_replays = 0
    if os.path.isfile(_abc_path):
        import random as _rand
        from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer
        bob_ppo.abc_buffer = GPUDemonstrationBuffer(
            capacity=2000, obs_shape=(env.bob_obs_dim,),
            states_shape=(env.num_envs,), actions_shape=(num_cat_dims,),
            device=device, traj_maxlen=5000,
        )
        bob_ppo.abc_buffer.load(_abc_path)
        n_replays = min(args.num_replays, bob_ppo.abc_buffer.size)
        print(f"[Diag] ABC buffer loaded: {bob_ppo.abc_buffer.size} trajectories, "
              f"will replay {n_replays}")
    else:
        print(f"[ERROR] ABC buffer not found at {_abc_path} — nothing to replay.")
        env.close()
        simulation_app.close()
        sys.exit(1)

    # ── Replays ────────────────────────────────────────────────────────────────
    results = []
    _py_random.seed(42)
    trajs = bob_ppo.abc_buffer.sample_trajectories(n_replays)
    _obj_dim = _OBS_ROBOT_DIM + 14  # offset of goal block
    _min_r = DISC_MIN_R if _is_disc else TBLOCK_MIN_R
    _max_r = DISC_MAX_R if _is_disc else TBLOCK_MAX_R

    for r_idx, traj in enumerate(trajs):
        T = traj["obs"].shape[0]
        acts = traj["acts"].long()
        goal_obs = traj["obs"][-1]
        goal_pos = goal_obs[_obj_dim:_obj_dim + 3].clone()
        goal_euler = torch.zeros(3, device=device)
        goal_euler[2] = goal_obs[_obj_dim + 5]

        print(f"\n[Replay {r_idx + 1}/{n_replays}] T={T} pushes, "
              f"goal=({goal_pos[0]:.3f},{goal_pos[1]:.3f}) yaw={goal_euler[2]:.3f}")

        # ── Reset Bob-side ────────────────────────────────────────────────────
        obs, _info = env.reset()
        _reset_jpos = _robot_scene.data.joint_pos.clone()
        _reset_jpos[:, _arm_jids] = _calib_cmd
        _robot_scene.write_joint_state_to_sim(_reset_jpos, torch.zeros_like(_reset_jpos))
        env.episode_manager.current_phase[:] = 1
        env.episode_manager.goal_valid[:] = True
        env.episode_manager.goal_count[:] = 1
        env.episode_manager.completion_given[:] = False
        _gs = torch.zeros(1, 6, device=device)
        _gs[0, :3] = goal_pos
        _gs[0, 3:6] = goal_euler
        env.episode_manager.goal_states = _gs
        env._update_goal_in_extras()

        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _calib_cmd.clone()

        pushes_used = 0
        stop_reason = "max_pushes"
        pos_err_prev = 0.0
        rot_err_prev = 0.0
        obj_pos = obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
        obj_euler = obs[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
        pos_err_prev = (obj_pos[:, :2] - goal_pos[:2]).norm().item()
        rot_err_prev = _rot_distance_rad(obj_euler[:, :3], goal_euler.unsqueeze(0)[:, :3]).item()

        for t in range(min(T, 30)):
            a = acts[t:t + 1]
            obj_x = float(obs[0, _OBS_ROBOT_DIM])
            obj_y = float(obs[0, _OBS_ROBOT_DIM + 1])
            obj_yaw = float(obs[0, _OBS_ROBOT_DIM + 5])
            Xs, Ys, length, theta = decode_push_action_relative(
                a, torch.tensor([[obj_x, obj_y]], device=device),
                torch.tensor([obj_yaw], device=device),
                num_bins=num_bins, min_r=_min_r, max_r=_max_r,
            )
            Xf = Xs + length * torch.cos(theta)
            Yf = Ys + length * torch.sin(theta)

            waypoints = compute_push_waypoints(
                Xs=Xs, Ys=Ys, length=length, theta=theta,
                current_ee_pos=ee_pos_local, current_ee_quat=ee_quat_w, device=device,
            )
            terminated = torch.zeros(1, dtype=torch.bool, device=device)
            for wp_idx, (wp_pos, wp_quat, _wp_grip) in enumerate(waypoints):
                ik_target = wp_pos - _TOTAL_IK_ERROR
                ik_target[:, 0].clamp_(_WS_X[0], _WS_X[1])
                ik_target[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
                ik_target[:, 2].clamp_(_WS_Z[0], _WS_Z[1])
                result = ik_solver.solve_batch(
                    CuroboPose(position=ik_target, quaternion=wp_quat),
                    seed_config=prev_joint_cmd.unsqueeze(1),
                    retract_config=prev_joint_cmd,
                )
                ik_ok = result.success.squeeze(-1)
                cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
                solved = result.solution.view(1, 6)
                elbow_bad = solved[:, 2] < 0.0
                if elbow_bad.any():
                    ik_ok[elbow_bad] = False
                raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, prev_joint_cmd)
                if terminated.any():
                    raw_cmd[terminated] = cur_joints[terminated]
                prev_joint_cmd = raw_cmd.detach().clone()
                env_full = torch.zeros(1, base_env.action_space.shape[0], device=device)
                env_full[:, :6] = raw_cmd
                env_full[:, 6] = -1.0
                obs, _, step_terminated, _, _ = base_env.step(env_full)
                terminated |= step_terminated
                _tcp_z_check = _tcp_pos_local()[:, 2]
                terminated |= (_tcp_z_check < -0.01)

            pushes_used += 1

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

            obj_pos = obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            obj_euler = obs[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
            pos_err = (obj_pos[:, :2] - goal_pos[:2]).norm().item()
            rot_err = _rot_distance_rad(obj_euler[:, :3], goal_euler.unsqueeze(0)[:, :3]).item()
            obj_z = obj_pos[0, 2].item()
            launched = (obj_z > 0.10)
            tipped = (abs(obj_euler[0, 0]) > 0.3 or abs(obj_euler[0, 1]) > 0.3)

            at_goal = pos_err < 0.05 and rot_err < args.rot_threshold
            if _is_disc:
                at_goal = pos_err < 0.05

            if at_goal:
                stop_reason = "success"
                break
            if launched:
                stop_reason = "launched"
                break
            if tipped:
                stop_reason = "tipped"
                break
            if terminated.any():
                stop_reason = "physics"
                break
            if pushes_used >= 30:
                break

            pos_err_prev = pos_err
            rot_err_prev = rot_err

        results.append({"pushes": pushes_used, "pos_err": pos_err,
                        "rot_err": rot_err, "stop_reason": stop_reason,
                        "success": 1 if stop_reason == "success" else 0})
        print(f"  -> {stop_reason} | pushes={pushes_used} "
              f"pos_err={pos_err:.4f}m rot_err={rot_err:.3f}rad")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_ok = sum(r["success"] for r in results)
    rate = n_ok / n_replays if n_replays else 0.0
    pushes = [r["pushes"] for r in results]
    avg_p = np.mean(pushes) if pushes else 0.0
    print(f"\n{'='*55}")
    print(f"  Alice-replay success rate: {rate:.3f} ({n_ok}/{n_replays})")
    print(f"  Average pushes: {avg_p:.1f}")
    print(f"  Model: {os.path.basename(args.chkpt_bob)}")
    print(f"  char_length={args.char_length}  {'disc' if _is_disc else 'T-block'} "
          f" gate={'pos<0.05' if _is_disc else 'pos<0.05 ∧ rot<0.2'}")
    print(f"{'='*55}")

    if args.csv and results:
        import csv as _csv
        with open(args.csv, "w", newline="") as _f:
            w = _csv.writer(_f)
            w.writerow(["replay", "pushes", "pos_err", "rot_err", "stop_reason", "success"])
            for i, r in enumerate(results):
                w.writerow([i + 1, r["pushes"], r["pos_err"], r["rot_err"],
                           r["stop_reason"], r["success"]])
        print(f"[CSV] {args.csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
