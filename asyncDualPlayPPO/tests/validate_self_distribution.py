"""
Validate an ASP model (Alice + Bob) *under its own distribution*.

Freezes both checkpoints (alice/model_best.pt, bob/model_best.pt) and replays the exact
Alice->Bob loop without any reward updates:

  1. Reset the object to a start pose drawn from the training start distribution
     (tight spawn box + random yaw, via ``env._rand_reset_objs``).
  2. Run frozen Alice for H_A = 5 pushes (sampling from her policy, LSTM hidden state
     carried across pushes).  The object pose at phase end becomes the goal g.
  3. Apply the same validity check as training (``validate_goal``: moved enough, on
     table, in workspace) plus the training "too-easy" filter (Bob must not start
     within the d_pose threshold of the goal).  Discard invalid / too-easy goals.
  4. Run frozen Bob on each valid goal with the standard eval protocol
     (``--bob-tries`` attempts x up to ``--bob-pushes`` pushes, combined gate
     pos < 0.05 m and rot < 0.2 rad; disc = position-only), sampling actions.
  5. Report Alice's validity rate + goal spread (x/y/yaw/displacement) and Bob's
     within-distribution scene SR + trial SR, and overlay Alice's goal support against
     the 30 held-out validation goals.

This is the "Path B" within-distribution validation — the apples-to-apples version of the
held-out evaluation, and it also measures Alice herself.

Usage (ASP-disc, seed s42):
  python -m asyncDualPlayPPO.tests.validate_self_distribution \
      --chkpt_bob final_results_thesis/discF_e528_i3000_s42/bob/model_best.pt \
      --chkpt_alice final_results_thesis/discF_e528_i3000_s42/alice/model_best.pt \
      --char-length 0.0 --dpose-threshold 0.05 --num-alice-goals 30 \
      --bob-tries 3 --bob-pushes 10 --headless --out-dir /tmp/selfdist_discF

Usage (ASP-dPose T-block, seed s42):
  python -m asyncDualPlayPPO.tests.validate_self_distribution \
      --chkpt_bob final_results_thesis/pbrsE_e528_i3000_s42/bob/model_best.pt \
      --chkpt_alice final_results_thesis/pbrsE_e528_i3000_s42/alice/model_best.pt \
      --char-length 0.07 --dpose-threshold 0.055 --num-alice-goals 30 \
      --bob-tries 3 --bob-pushes 10 --headless --out-dir /tmp/selfdist_pbrsE

Notes:
  * Object type, geometry, spawn/settled z, and approach radius are selected automatically
    from ``--char-length`` (0.0 => disc, else T-block) — matching each training run.
  * ``dpose_obs`` is always enabled (both discF and pbrsE are d_pose models).
  * Outcome-based Alice (skip_shallow_penalty=False) is assumed; both models here use it.
"""

import argparse
import csv
import json
import math
import os
import signal
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# cuRobo must be imported before AppLauncher
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print("[ERROR] cuRobo not found.")
    sys.exit(1)

import torch
import torch._dynamo    # noqa
import torch._C         # noqa
import torch.optim      # noqa

from isaaclab.app import AppLauncher

from asyncDualPlayPPO.tasks.utils.validation_configs import (
    get_test_config, get_test_count, set_test_set,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)
_WS_Z = (0.25, 0.55)

_ALICE_PUSHES = 5


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def _wilson(p, n, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    parser = argparse.ArgumentParser(description="Validate ASP model under its own distribution")
    parser.add_argument("--chkpt_bob", type=str, required=True, help="Bob checkpoint (or run dir with bob/)")
    parser.add_argument("--chkpt_alice", type=str, required=True, help="Alice checkpoint (or run dir with alice/)")
    parser.add_argument("--char-length", type=float, default=0.07,
                        help="SE(2) characteristic length L (0.0 => disc, else T-block)")
    parser.add_argument("--dpose-threshold", type=float, default=0.055,
                        help="d_pose success threshold in metres")
    parser.add_argument("--num-alice-goals", type=int, default=30,
                        help="Number of Alice goals to sample (each = 5 Alice pushes + Bob eval)")
    parser.add_argument("--bob-tries", type=int, default=3, help="Bob attempts per goal")
    parser.add_argument("--bob-pushes", type=int, default=10, help="Max Bob pushes per attempt")
    parser.add_argument("--rot-threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians (combined gate)")
    parser.add_argument("--argmax", action="store_true", dest="argmax",
                        help="Use argmax (deterministic) actions for Bob instead of sampling")
    parser.add_argument("--out-dir", type=str, default="/tmp/self_distribution",
                        help="Directory for CSV/JSON/summary/plot output")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    is_disc = args.char_length < 1e-6
    set_test_set("disc" if is_disc else "all")

    def _resolve(path, sub):
        if os.path.isdir(path):
            cand = os.path.join(path, sub, "model_best.pt")
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(path, sub, "latest_checkpoint.pt")
            if os.path.isfile(cand):
                return cand
            raise SystemExit(f"[ERROR] no checkpoint in {path}/{sub}/")
        return path

    args.chkpt_bob = _resolve(args.chkpt_bob, "bob")
    args.chkpt_alice = _resolve(args.chkpt_alice, "alice")
    os.makedirs(args.out_dir, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    signal.signal(signal.SIGINT, lambda *_: (simulation_app.close(), os._exit(1)))

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.push_task_curobo_disc import PushTaskCuRoboDiscEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push_asp import PushASPEnvWrapper, _OBS_ROBOT_DIM
    from asyncDualPlayPPO.tasks.utils.reward_pbrs import compute_dpose
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative, TBLOCK_MIN_R, TBLOCK_MAX_R, DISC_MIN_R, DISC_MAX_R,
    )
    from asyncDualPlayPPO.tasks.utils.observations import _euler_xyz_to_quat
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.module import MultiCategorical
    from asyncDualPlayPPO.utils.goal_validator import validate_goal
    import copy
    import gymnasium as gym_mc
    import yaml
    import numpy as np

    ppo_cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg/ppo/ppo_continuous.yaml")
    with open(ppo_cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    num_cat_dims = 4
    num_bins = 21

    if is_disc:
        env_cfg = PushTaskCuRoboDiscEnvCfg()
        obj_spawn_z, obj_settled_z = 0.03, 0.03
        min_r, max_r = DISC_MIN_R, DISC_MAX_R
    else:
        env_cfg = PushTaskCuRoboEnvCfg()
        obj_spawn_z, obj_settled_z = 0.05, 0.023
        min_r, max_r = TBLOCK_MIN_R, TBLOCK_MAX_R
    env_cfg.scene.num_envs = 1
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES, scale=1.0, use_default_offset=False,
    )

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushASPEnvWrapper(
        env=base_env, alice_pushes=_ALICE_PUSHES, bob_pushes=args.bob_pushes,
        max_goals_per_episode=1, num_objects=1,
        dpose_obs=True, char_length=args.char_length, dpose_threshold=args.dpose_threshold,
        obj_spawn_z=obj_spawn_z, obj_settled_z=obj_settled_z,
        device=device,
    )

    _blk_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/blocks")
    _goal_viz = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarker",
            markers={"tblock": UsdFileCfg(
                usd_path=os.path.join(_blk_dir, "t_shape.usda"), scale=(2.0, 2.0, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)))},
        )
    )
    _goal_viz_disc = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkerDisc",
            markers={"disc": sim_utils.CylinderCfg(
                radius=0.05, height=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)))},
        )
    )
    _ident_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    def _update_goal_marker(gx, gy, gyaw=0.0):
        origins = env.env.scene.env_origins
        pos = torch.tensor([[gx, gy, 0.001]], device=device) + origins
        euler = torch.zeros(1, 3, device=device)
        euler[0, 2] = gyaw
        quat = _euler_xyz_to_quat(euler)
        hide_pos = torch.tensor([[0.0, 0.0, -1.0]], device=device) + origins
        hide_quat = _QUAT_TOOL_DOWN.expand(1, 4)
        if is_disc:
            _goal_viz_disc.visualize(translations=pos, orientations=quat)
            _goal_viz.visualize(translations=hide_pos, orientations=hide_quat)
        else:
            _goal_viz.visualize(translations=pos, orientations=quat)
            _goal_viz_disc.visualize(translations=hide_pos, orientations=hide_quat)

    # ── cuRobo IK ─────────────────────────────────────────────────────────────
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    _ik_config.solver.newton_optimizer.n_iters = 30
    _ik_config.solver.newton_optimizer.inner_iters = 10
    ik_solver = IKSolver(_ik_config)

    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    print("[Setup] Calibrating IK→physics error...")
    _calib_pos = torch.zeros(1, 3, device=device)
    _calib_pos[:, 1] = 0.60
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN.expand(1, 4)),
        seed_config=_calib_cur.unsqueeze(1), retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(1, 6)
    _calib_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
    _TOTAL_IK_ERROR = (_tcp_pos_local() - _calib_pos).clone()
    print(f"[Setup] IK error = ({float(_TOTAL_IK_ERROR[0,0]):+.3f}, "
          f"{float(_TOTAL_IK_ERROR[0,1]):+.3f}, {float(_TOTAL_IK_ERROR[0,2]):+.3f})")

    def _reset_robot():
        jpos = _robot_scene.data.joint_pos.clone()
        jpos[:, _arm_jids] = _calib_cmd
        _robot_scene.write_joint_state_to_sim(jpos, torch.zeros_like(jpos))

    # ── Detect GoalEncoder from Bob checkpoint ────────────────────────────────
    _chkpt = torch.load(args.chkpt_bob, map_location="cpu", weights_only=False)
    _chkpt_state = _chkpt.get("model_state_dict", _chkpt)
    _pi_w0 = _chkpt_state.get("pi_encoder.obj_encoder.0.weight")
    _is_noge = _pi_w0 is not None and _pi_w0.shape[1] == 22
    _has_goal_encoder = not _is_noge
    del _chkpt, _chkpt_state
    torch.cuda.empty_cache()
    print(f"[Detect] GoalEncoder={'ON' if _has_goal_encoder else 'OFF'}")

    # ── Bob ───────────────────────────────────────────────────────────────────
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

    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    bob_ppo = PPOABC(vec_env=env, cfg_train=bob_cfg, device=device,
                     sampler="sequential", log_dir="/tmp/validate_self_dist_bob", asymmetric=False)
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space
    bob_ppo.action_space = _mc_space
    bob_ppo.desired_kl = None
    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape, bob_ppo.state_space.shape, bob_ppo.action_space.shape,
        bob_ppo.init_noise_std, bob_ppo.model_cfg, asymmetric=False,
    ).to(device)
    if hasattr(bob_ppo.actor_critic, "_goal_proj") and bob_ppo.actor_critic._goal_proj is not None:
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.1)
    bob_ppo.load(args.chkpt_bob)
    bob_ppo.actor_critic.eval()
    _bob_lsz = bob_ppo.actor_critic.lstm_hidden_size
    print(f"[Load] Bob from {args.chkpt_bob}")

    # ── Alice ─────────────────────────────────────────────────────────────────
    alice_cfg = copy.deepcopy(ppo_cfg["params"])
    alice_cfg["policy"]["use_pi_encoder"] = True
    alice_cfg["policy"]["use_multicategorical"] = True
    alice_cfg["policy"]["use_lstm"] = True
    alice_cfg["policy"]["use_goal_encoder"] = False
    alice_cfg["policy"]["num_cat_dims"] = num_cat_dims
    alice_cfg["policy"]["num_bins"] = num_bins
    alice_cfg["policy"]["robot_state_dim"] = 6

    alice_ppo = PPO(vec_env=env, cfg_train=alice_cfg, device=device,
                    sampler="sequential", log_dir="/tmp/validate_self_dist_alice", asymmetric=False)
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space
    alice_ppo.action_space = _mc_space
    alice_ppo.desired_kl = None
    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape, alice_ppo.state_space.shape, alice_ppo.action_space.shape,
        alice_ppo.init_noise_std, alice_ppo.model_cfg, asymmetric=False,
    ).to(device)
    alice_ppo.load(args.chkpt_alice)
    alice_ppo.actor_critic.eval()
    _alice_lsz = alice_ppo.actor_critic.lstm_hidden_size
    print(f"[Load] Alice from {args.chkpt_alice}")

    print(f"[Config] object={'disc' if is_disc else 'tblock'} char_length={args.char_length} "
          f"dpose_threshold={args.dpose_threshold} n_goals={args.num_alice_goals} "
          f"bob_tries={args.bob_tries} bob_pushes={args.bob_pushes}")

    # ── Init env ──────────────────────────────────────────────────────────────
    env.reset()
    env.env.sim.step()
    _close_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _close_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
    _close_act[:, 6] = -1.0
    env.step(_close_act)

    env.episode_manager.initial_states = torch.zeros(1, 6, device=device)
    _zero_ids = torch.tensor([0], device=device, dtype=torch.long)

    # ── Execute a push trajectory (shared by Alice + Bob) ─────────────────────
    def _execute_push(Xs, Ys, length, theta, ee_pos_local, ee_quat_w, prev_joint_cmd):
        waypoints = compute_push_waypoints(
            Xs=Xs, Ys=Ys, length=length, theta=theta,
            current_ee_pos=ee_pos_local, current_ee_quat=ee_quat_w, device=device,
        )
        terminated = torch.zeros(1, dtype=torch.bool, device=device)
        for (wp_pos, wp_quat, _wp_grip) in waypoints:
            ik_target = wp_pos - _TOTAL_IK_ERROR
            ik_target[:, 0].clamp_(_WS_X[0], _WS_X[1])
            ik_target[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
            ik_target[:, 2].clamp_(_WS_Z[0], _WS_Z[1])
            result = ik_solver.solve_batch(
                CuroboPose(position=ik_target, quaternion=wp_quat),
                seed_config=prev_joint_cmd.unsqueeze(1), retract_config=prev_joint_cmd,
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
            env_full = torch.zeros(1, env.action_space.shape[0], device=device)
            env_full[:, :6] = raw_cmd
            env_full[:, 6] = -1.0
            _, _, step_terminated, _, _ = env.step(env_full)
            terminated |= step_terminated
            terminated |= (_tcp_pos_local()[:, 2] < -0.01)
        return terminated

    def _sample_action(actor, obs, hidden):
        with torch.no_grad():
            raw, (h, c) = actor._actor_forward(obs, (hidden[0], hidden[1]))
        dist = MultiCategorical(raw.view(1, num_cat_dims, num_bins))
        act = dist.mode() if args.argmax else dist.sample()
        return act, [h, c]

    # ── Alice -> goal ─────────────────────────────────────────────────────────
    def _run_alice():
        _reset_robot()
        _sp = env._rand_reset_objs(_zero_ids)
        env.env.scene.write_data_to_sim()
        env.episode_manager.initial_states[0] = env._initial_states_from_spawn(_sp, 1)[0]

        full_obs = env._get_push_obs()
        alice_obs = env._get_alice_obs(full_obs)
        alice_hidden = [torch.zeros(1, _alice_lsz, device=device),
                        torch.zeros(1, _alice_lsz, device=device)]
        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

        terminated = torch.zeros(1, dtype=torch.bool, device=device)
        for _ in range(_ALICE_PUSHES):
            act, alice_hidden = _sample_action(alice_ppo.actor_critic, alice_obs, alice_hidden)
            obj_x = full_obs[0, _OBS_ROBOT_DIM].item()
            obj_y = full_obs[0, _OBS_ROBOT_DIM + 1].item()
            obj_yaw = float(full_obs[0, _OBS_ROBOT_DIM + 5].item())
            Xs, Ys, length, theta = decode_push_action_relative(
                act, torch.tensor([[obj_x, obj_y]], device=device),
                torch.tensor([obj_yaw], device=device),
                num_bins=num_bins, min_r=min_r, max_r=max_r, max_len=0.20,
            )
            terminated |= _execute_push(Xs, Ys, length, theta, ee_pos_local, ee_quat_w, prev_joint_cmd)
            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
            full_obs = env._get_push_obs()
            alice_obs = env._get_alice_obs(full_obs)

        goal_state = env._extract_object_states(full_obs)      # (1,6)
        initial_state = env.episode_manager.initial_states     # (1,6)
        valid, _, reasons = validate_goal(
            initial_state, goal_state,
            env.table_bounds, env.placement_bounds,
            pos_threshold=0.05, rot_threshold=0.25,
            min_meaningful_disp=0.10,
            require_all_moved=False,
            skip_shallow_penalty=False,
        )
        # too-easy filter (matches training _transition_to_bob)
        dp = compute_dpose(
            initial_state[:, 0:3], goal_state[:, 0:3],
            initial_state[:, 5], goal_state[:, 5], args.char_length,
        )
        too_easy = dp < args.dpose_threshold

        disp_3d = float((goal_state[0, :3] - initial_state[0, :3]).norm().item())
        disp_2d = float((goal_state[0, :2] - initial_state[0, :2]).norm().item())

        info = {
            "valid": bool(valid[0]) and not bool(too_easy[0]),
            "reason": ("too_easy" if bool(too_easy[0]) else (reasons[0] if valid[0] else reasons[0])),
            "goal_x": float(goal_state[0, 0]), "goal_y": float(goal_state[0, 1]),
            "goal_yaw": float(goal_state[0, 5]),
            "init_x": float(initial_state[0, 0]), "init_y": float(initial_state[0, 1]),
            "init_yaw": float(initial_state[0, 5]),
            "disp_3d": disp_3d, "disp_2d": disp_2d,
            "terminated": bool(terminated[0]),
        }
        if not info["valid"]:
            return info, None
        return info, (goal_state, initial_state)

    # ── Bob attempt on a fixed goal ───────────────────────────────────────────
    def _bob_attempt(goal_state, initial_state):
        _reset_robot()
        # object to start pose
        start = initial_state[0]
        t_quat = _euler_xyz_to_quat(start[3:6].unsqueeze(0))
        pos_global = start[:3].unsqueeze(0) + env.env.scene.env_origins[0]
        env.env.scene["target_object"].write_root_pose_to_sim(torch.cat([pos_global, t_quat], dim=-1))
        env.env.scene["target_object"].write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        env.env.sim.step()

        env.episode_manager.goal_states = goal_state.clone()
        env.episode_manager.current_phase[:] = 1
        env.episode_manager.phase_step[:] = 0
        env._update_goal_in_extras()
        _update_goal_marker(goal_state[0, 0], goal_state[0, 1], goal_state[0, 5])

        full_obs = env._get_push_obs()
        bob_obs = full_obs.clone()
        bob_hidden = [torch.zeros(1, _bob_lsz, device=device),
                      torch.zeros(1, _bob_lsz, device=device)]
        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

        pos_err = 0.0
        rot_err = 0.0
        success = False
        for _ in range(args.bob_pushes):
            act, bob_hidden = _sample_action(bob_ppo.actor_critic, bob_obs, bob_hidden)
            obj_x = full_obs[0, _OBS_ROBOT_DIM].item()
            obj_y = full_obs[0, _OBS_ROBOT_DIM + 1].item()
            obj_yaw = float(full_obs[0, _OBS_ROBOT_DIM + 5].item())
            Xs, Ys, length, theta = decode_push_action_relative(
                act, torch.tensor([[obj_x, obj_y]], device=device),
                torch.tensor([obj_yaw], device=device),
                num_bins=num_bins, min_r=min_r, max_r=max_r, max_len=0.20,
            )
            terminated = _execute_push(Xs, Ys, length, theta, ee_pos_local, ee_quat_w, prev_joint_cmd)
            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
            full_obs = env._get_push_obs()
            bob_obs = full_obs.clone()

            cur_obj_pos = full_obs[0, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            cur_obj_euler = full_obs[0, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
            goal_pos = full_obs[0, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 17]
            goal_euler = full_obs[0, _OBS_ROBOT_DIM + 17:_OBS_ROBOT_DIM + 20]
            pos_err = (cur_obj_pos - goal_pos).norm().item()
            rot_err = _rot_distance_rad(cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)).item()
            if is_disc:
                rot_err = 0.0
            if is_disc:
                success = pos_err < 0.05
            else:
                success = pos_err < 0.05 and rot_err < args.rot_threshold
            if success or terminated[0]:
                break
        return success, pos_err, rot_err

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = []
    n_valid = 0
    n_too_easy = 0
    n_invalid = 0
    n_terminated = 0
    bob_scene_success = 0
    bob_trial_success = 0
    bob_trial_total = 0

    for gi in range(args.num_alice_goals):
        info, goal_tuple = _run_alice()
        if info["terminated"]:
            n_terminated += 1
        rec = {
            "goal_index": gi + 1,
            "valid": int(info["valid"]),
            "reason": info["reason"],
            "goal_x": round(info["goal_x"], 5), "goal_y": round(info["goal_y"], 5),
            "goal_yaw": round(info["goal_yaw"], 5),
            "init_x": round(info["init_x"], 5), "init_y": round(info["init_y"], 5),
            "init_yaw": round(info["init_yaw"], 5),
            "disp_3d": round(info["disp_3d"], 5), "disp_2d": round(info["disp_2d"], 5),
            "bob_scene_success": 0, "bob_trial_success": 0, "bob_trial_total": 0,
            "final_pos_err": float("nan"), "final_rot_err": float("nan"),
        }
        if info["valid"]:
            n_valid += 1
            goal_state, initial_state = goal_tuple
            trial_ok = 0
            best_pos = float("inf")
            best_rot = float("inf")
            for _ in range(args.bob_tries):
                ok, pos_err, rot_err = _bob_attempt(goal_state, initial_state)
                if ok:
                    trial_ok += 1
                if pos_err < best_pos:
                    best_pos, best_rot = pos_err, rot_err
            rec["bob_trial_success"] = trial_ok
            rec["bob_trial_total"] = args.bob_tries
            rec["bob_scene_success"] = 1 if trial_ok > 0 else 0
            rec["final_pos_err"] = round(best_pos, 5)
            rec["final_rot_err"] = round(best_rot, 5)
            bob_scene_success += rec["bob_scene_success"]
            bob_trial_success += trial_ok
            bob_trial_total += args.bob_tries
            print(f"[Goal {gi+1:3d}] valid  goal=({rec['goal_x']:+.3f},{rec['goal_y']:+.3f}) "
                  f"disp={rec['disp_3d']:.3f}m  Bob {trial_ok}/{args.bob_tries}")
        else:
            if info["reason"] == "too_easy":
                n_too_easy += 1
            else:
                n_invalid += 1
            print(f"[Goal {gi+1:3d}] INVALID({info['reason']})  "
                  f"goal=({rec['goal_x']:+.3f},{rec['goal_y']:+.3f}) disp={rec['disp_3d']:.3f}m")
        results.append(rec)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    gx = np.array([r["goal_x"] for r in results])
    gy = np.array([r["goal_y"] for r in results])
    gyaw = np.array([r["goal_yaw"] for r in results])
    disp = np.array([r["disp_3d"] for r in results])

    scene_sr = bob_scene_success / n_valid * 100 if n_valid else 0.0
    trial_sr = bob_trial_success / bob_trial_total * 100 if bob_trial_total else 0.0
    ci_lo, ci_hi = _wilson(bob_trial_success / bob_trial_total if bob_trial_total else 0.0,
                           bob_trial_total)

    def _spread(a):
        if len(a) == 0:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
        return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

    summary = {
        "object": "disc" if is_disc else "tblock",
        "char_length": args.char_length,
        "dpose_threshold": args.dpose_threshold,
        "num_alice_goals": args.num_alice_goals,
        "alice_valid": n_valid,
        "alice_too_easy": n_too_easy,
        "alice_invalid": n_invalid,
        "alice_terminated": n_terminated,
        "alice_validity_rate": n_valid / args.num_alice_goals * 100,
        "alice_goal_spread": {
            "x": _spread(gx), "y": _spread(gy), "yaw": _spread(gyaw), "disp_3d": _spread(disp),
        },
        "bob_goals_evaluated": n_valid,
        "bob_scene_success": bob_scene_success,
        "bob_scene_sr": scene_sr,
        "bob_trial_success": bob_trial_success,
        "bob_trial_total": bob_trial_total,
        "bob_trial_sr": trial_sr,
        "bob_trial_sr_ci95": [ci_lo * 100, ci_hi * 100],
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "alice_goals.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["goal_index", "valid", "reason", "goal_x", "goal_y", "goal_yaw",
                    "init_x", "init_y", "init_yaw", "disp_3d", "disp_2d",
                    "bob_scene_success", "bob_trial_success", "bob_trial_total",
                    "final_pos_err", "final_rot_err"])
        for r in results:
            w.writerow([r["goal_index"], r["valid"], r["reason"], r["goal_x"], r["goal_y"],
                        r["goal_yaw"], r["init_x"], r["init_y"], r["init_yaw"],
                        r["disp_3d"], r["disp_2d"], r["bob_scene_success"],
                        r["bob_trial_success"], r["bob_trial_total"],
                        r["final_pos_err"], r["final_rot_err"]])

    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Self-Distribution Validation — {'disc' if is_disc else 'T-block'} ASP\n\n")
        f.write(f"- Object: **{'disc' if is_disc else 'T-block'}**, char_length={args.char_length}, "
                f"dpose_threshold={args.dpose_threshold}\n")
        f.write(f"- Alice goals sampled: {args.num_alice_goals}; Bob protocol: {args.bob_tries} tries x {args.bob_pushes} pushes\n\n")
        f.write(f"## Alice\n")
        f.write(f"- Validity rate: **{summary['alice_validity_rate']:.1f}%** "
                f"({n_valid} valid, {n_too_easy} too-easy, {n_invalid} invalid, {n_terminated} terminated)\n")
        for k, v in summary["alice_goal_spread"].items():
            f.write(f"- goal {k}: mean={v['mean']:+.4f} std={v['std']:.4f} [{v['min']:+.4f}, {v['max']:+.4f}]\n")
        f.write(f"\n## Bob (within-distribution)\n")
        f.write(f"- Scene SR: **{scene_sr:.1f}%** ({bob_scene_success}/{n_valid} goals)\n")
        f.write(f"- Trial SR: **{trial_sr:.1f}%** ({bob_trial_success}/{bob_trial_total}) "
                f"95% CI [{ci_lo*100:.1f}, {ci_hi*100:.1f}]\n")

    # ── Plot: Alice goal support vs held-out goals ────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        held_x, held_y, held_yaw = [], [], []
        for i in range(1, get_test_count() + 1):
            c = get_test_config(i)
            if c is None:
                continue
            held_x.append(c.main_goal_x)
            held_y.append(c.main_goal_y)
            held_yaw.append(c.main_goal_yaw)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(held_x, held_y, c="0.7", marker="s", s=50, label="held-out goals")
        axes[0].scatter(gx, gy, c="#d62728", s=20, alpha=0.7, label="Alice goals")
        axes[0].set_xlabel("goal x (m)")
        axes[0].set_ylabel("goal y (m)")
        axes[0].set_title("Alice goal support vs held-out goals (XY)")
        axes[0].legend()
        axes[0].set_aspect("equal", adjustable="box")
        axes[1].hist(gyaw, bins=36, range=(-np.pi, np.pi), alpha=0.7, color="#d62728", label="Alice")
        axes[1].hist(held_yaw, bins=36, range=(-np.pi, np.pi), alpha=0.4, color="0.5", label="held-out")
        axes[1].set_xlabel("goal yaw (rad)")
        axes[1].set_ylabel("count")
        axes[1].set_title("Goal yaw distribution")
        axes[1].legend()
        fig.tight_layout()
        plot_path = os.path.join(args.out_dir, "alice_support_overlay.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {plot_path}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    print(f"\n{summary['alice_validity_rate']:.1f}% Alice validity | "
          f"scene SR {scene_sr:.1f}% | trial SR {trial_sr:.1f}% [{ci_lo*100:.1f},{ci_hi*100:.1f}]")
    print(f"Outputs in {args.out_dir}")

    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
