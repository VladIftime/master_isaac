"""
Validate a trained push-PPO model against test configurations.

Uses the pre-defined test scenes from validation_configs.py to measure push
success rate, steps-to-solve, and push count across multiple scenarios.
Visual markers show goal position and push arrows.  Airborne/tipped/OOB
objects are detected and terminate the test early.

Usage:
  # Original Push-PPO (absolute actions):
  python -m asyncDualPlayPPO.tests.validate_push \
      --chkpt runs/push_ppo_baseline/agent/model_best.pt \
      --num_tests 10 --headless

  # PBRS Models A/B (object-relative obs+actions):
  python -m asyncDualPlayPPO.tests.validate_push \
      --chkpt runs/long_runs/hpc_pbrs_simp_528env/agent/model_best.pt \
      --rel-obs --rel-act --num_tests 10 --headless --csv results.csv
"""

import argparse
import os
import signal
import sys
import time
import math
from dataclasses import dataclass
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    ALL_TESTS, get_test_config, get_test_count, PushTestConfig,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


@dataclass
class ValidationResult:
    test_index: int
    test_name: str
    test_type: str
    success: bool
    pushes_used: int
    final_pos_error: float
    final_rot_error: float
    area_coverage: float
    trial_count: int = 1
    success_count: int = 0


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def _area_coverage(pos_err, rot_err):
    pc = max(0.0, 1.0 - pos_err / 0.10)
    rc = max(0.0, 1.0 - rot_err / 0.40)
    return pc * rc * 100.0


def main():
    parser = argparse.ArgumentParser(description="Validate Push-PPO Model")
    parser.add_argument("--chkpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--num_tests", type=int, default=10, help="Number of test scenes to run")
    parser.add_argument("--max_pushes", type=int, default=15, help="Max pushes per test")
    parser.add_argument("--max_tries", type=int, default=3, help="Max retries per test")
    parser.add_argument("--rot_threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians (default 0.2)")
    parser.add_argument("--rel-obs", action="store_true", dest="rel_obs",
                        help="Use object-relative observation (30D instead of 28D)")
    parser.add_argument("--rel-act", action="store_true", dest="rel_act",
                        help="Decode actions as object-relative (r, phi, len, theta)")
    parser.add_argument("--argmax", action="store_true", dest="argmax",
                        help="Use argmax (deterministic) actions instead of sampling")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save validation results to CSV file")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    signal.signal(signal.SIGINT, lambda *_: (simulation_app.close(), os._exit(1)))

    chkpt_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.chkpt)))
    if args.csv is None:
        args.csv = os.path.join(chkpt_run_dir, "validation_results.csv")
    elif not os.path.isabs(args.csv):
        args.csv = os.path.join(chkpt_run_dir, args.csv)

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper, _euler_to_quat
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints,
    )
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
        TBLOCK_MIN_R, TBLOCK_MAX_R,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import RolloutStorage
    import gymnasium as gym_mc
    import numpy as np

    # ── Environment ────────────────────────────────────────────────────────────
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1  # single env for validation

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushEnvWrapper(
        env=base_env, device=device, num_objects=1,
        max_pushes_per_episode=args.max_pushes,
        rel_obs=args.rel_obs,
    )

    # ── Visual markers ─────────────────────────────────────────────────────────
    _goal_viz = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkers",
            markers={
                "tblock": UsdFileCfg(
                    usd_path=os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), "assets/blocks/t_shape.usda"),
                    scale=(2.0, 2.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),
                ),
            },
        )
    )
    _push_viz_start = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushStart",
            markers={"sphere": sim_utils.SphereCfg(radius=0.015,
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))},
        )
    )
    _push_viz_end = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushEnd",
            markers={"sphere": sim_utils.SphereCfg(radius=0.015,
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
        )
    )
    _push_viz_arrow = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushArrow",
            markers={"cylinder": sim_utils.CylinderCfg(radius=0.005, height=0.30,
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)))},
        )
    )
    _ident_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    def _update_goal_marker(gx, gy, gyaw=0.0):
        origins = env.env.scene.env_origins
        pos = torch.tensor([[gx, gy, 0.001]], device=device) + origins
        euler = torch.zeros(1, 3, device=device)
        euler[0, 2] = gyaw
        quat = _euler_to_quat(euler)
        _goal_viz.visualize(translations=pos, orientations=quat)

    def _update_push_markers(Xs, Ys, Xf, Yf, angle):
        N = 1
        origins = env.env.scene.env_origins
        z_t = 0.002
        ident = _ident_quat.to(device).expand(N, 4)
        sp = torch.stack([Xs.float(), Ys.float(), torch.full((N,), z_t, device=device)], dim=-1) + origins
        ep = torch.stack([Xf.float(), Yf.float(), torch.full((N,), z_t, device=device)], dim=-1) + origins
        _push_viz_start.visualize(translations=sp, orientations=ident)
        _push_viz_end.visualize(translations=ep, orientations=ident)
        mid = torch.stack([(Xs + Xf) / 2, (Ys + Yf) / 2,
                           torch.full((N,), z_t, device=device)], dim=-1) + origins
        half = math.pi / 4
        ch, sh = math.cos(half), math.sin(half)
        aq = torch.stack([torch.full((N,), ch, device=device),
                          -sh * torch.sin(angle), sh * torch.cos(angle),
                          torch.zeros(N, device=device)], dim=-1)
        _push_viz_arrow.visualize(translations=mid, orientations=aq)

    # ── cuRobo IK ─────────────────────────────────────────────────────────────
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    ik_solver = IKSolver(_ik_config)

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    # ── IK→physics calibration ────────────────────────────────────────────────
    _WS_X = (-0.50, 0.50)
    _WS_Y = (0.25, 0.70)
    _WS_Z = (0.25, 0.55)
    print("[Setup] Calibrating IK→physics error...")
    _calib_pos = torch.zeros(1, 3, device=device)
    _calib_pos[:, 1] = 0.60
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN.expand(1, 4)),
        seed_config=_calib_cur.unsqueeze(1),
        retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(1, 6)
    _calib_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
    _finger_after = _tcp_pos_local()
    _TOTAL_IK_ERROR = (_finger_after - _calib_pos).clone()
    print(f"[Setup] IK error = ({float(_TOTAL_IK_ERROR[0,0]):+.3f}, "
          f"{float(_TOTAL_IK_ERROR[0,1]):+.3f}, {float(_TOTAL_IK_ERROR[0,2]):+.3f})")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    num_cat_dims = 4
    num_bins = 21

    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    agent_cfg = {
        "learn": {
            "nsteps": 32, "noptepochs": 3, "nminibatches": 4,
            "cliprange": 0.2, "ent_coef": 0.01, "gamma": 0.998, "lam": 0.95,
            "optim_stepsize": 3e-4, "init_noise_std": 0.3,
            "value_loss_coef": 1.0, "max_grad_norm": 1.0,
        },
        "policy": {
            "use_multicategorical": True, "num_cat_dims": 4, "num_bins": 21,
            "use_lstm": True, "lstm_hidden_size": 256,
            "pi_hid_sizes": [512, 256, 128],
            "vf_hid_sizes": [512, 256, 128],
            "activation": "relu",
        },
    }

    agent = PPO(
        vec_env=env, cfg_train=agent_cfg, device=device,
        sampler="sequential", log_dir="/tmp/validate_push",
        asymmetric=False,
    )
    agent.observation_space = env.observation_space
    agent.state_space = env.state_space
    agent.action_space = _mc_space
    agent.desired_kl = None

    agent.actor_critic = ActorCriticPush(
        agent.observation_space.shape, agent.state_space.shape,
        agent.action_space.shape, agent.init_noise_std, agent.model_cfg,
        asymmetric=False,
    ).to(device)

    agent.load(args.chkpt)
    agent.actor_critic.eval()
    print(f"[Validate] Loaded model from {args.chkpt}")
    print(f"[Validate] Mode: rel_obs={args.rel_obs}, rel_act={args.rel_act}, "
          f"rot_threshold={args.rot_threshold:.3f} rad, max_pushes={args.max_pushes}")

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: List[ValidationResult] = []
    test_cfgs_data: List[dict] = []
    n_tests = min(args.num_tests, get_test_count())

    for test_idx in range(1, n_tests + 1):
        cfg = get_test_config(test_idx)
        if cfg is None:
            continue

        _obj_type = getattr(cfg, "object_type", "tblock")

        print(f"\n[Test {test_idx}/{n_tests}] {cfg.name} #{cfg.test_id}")
        print(f"  [{cfg.test_type}] goal=({cfg.main_goal_x:+.3f},{cfg.main_goal_y:+.3f}) yaw={cfg.main_goal_yaw:+.3f}  "
              f"start=({cfg.main_start.x:+.3f},{cfg.main_start.y:+.3f})")

        TRIAL_COUNT = args.max_tries
        trial_successes = 0
        trial_pushes = []
        best_pos_err = float('inf')
        best_rot_err = float('inf')

        for trial in range(TRIAL_COUNT):
            obs = env.reset()
            env.goal_pos_euler[0, 0] = cfg.main_goal_x
            env.goal_pos_euler[0, 1] = cfg.main_goal_y
            env.goal_pos_euler[0, 2] = 0.0
            env.goal_pos_euler[0, 3:5] = 0.0
            env.goal_pos_euler[0, 5] = cfg.main_goal_yaw
            env._update_goal_in_extras()
            _update_goal_marker(cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw)

            obj = env.env.scene["target_object"]
            obj.write_root_pose_to_sim(torch.tensor([[
                cfg.main_start.x, cfg.main_start.y, 0.02, 1.0, 0.0, 0.0, 0.0
            ]], device=device))
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

            env.env.sim.step()
            obs = env._get_push_obs()
            env._capture_prev_obj(obs)

            _init_obj_pos = obs[0, env.robot_dim:env.robot_dim + 3]
            _init_obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
            _init_goal_pos = obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
            _init_goal_euler = obs[0, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]
            _init_pos_err = (_init_obj_pos - _init_goal_pos).norm().item()
            _init_rot_err = _rot_distance_rad(_init_obj_euler.unsqueeze(0), _init_goal_euler.unsqueeze(0)).item()
            if _obj_type == "disc":
                _init_rot_err = 0.0
            _init_oob_2d = float((_init_obj_pos[:2] - _init_goal_pos[:2]).norm().item())

            hidden = [
                torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
                torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
            ]

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

            trial_ok = False
            pushes_used = 0
            stop_reason = "max_pushes"
            pos_err = 0.0
            rot_err = 0.0
            prev_pos_err = _init_pos_err
            prev_rot_err = _init_rot_err

            for push_i in range(args.max_pushes):
                with torch.no_grad():
                    actions, _, _, _, _, _, new_h = agent.actor_critic.act_with_hidden(
                        obs, None, (hidden[0], hidden[1]),
                        deterministic=args.argmax,
                    )
                    if new_h is not None:
                        hidden[0] = new_h[0]
                        hidden[1] = new_h[1]

                if args.rel_act:
                    obj_x = obs[0, env.robot_dim]
                    obj_y = obs[0, env.robot_dim + 1]
                    obj_yaw = obs[0, env.robot_dim + 5]
                    Xs, Ys, length, theta = decode_push_action_relative(
                        actions,
                        torch.stack([obj_x, obj_y]).unsqueeze(0),
                        obj_yaw.unsqueeze(0),
                        num_bins=num_bins,
                        min_r=TBLOCK_MIN_R, max_r=TBLOCK_MAX_R,
                    )
                else:
                    Xs, Ys, length, theta = decode_push_action(actions, num_bins=num_bins)

                Xf = Xs + length * torch.cos(theta)
                Yf = Ys + length * torch.sin(theta)

                waypoints = compute_push_waypoints(
                    Xs=Xs, Ys=Ys, length=length, theta=theta,
                    current_ee_pos=ee_pos_local,
                    current_ee_quat=ee_quat_w, device=device,
                )

                _update_push_markers(Xs, Ys, Xf, Yf, theta)

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

                    env_full = torch.zeros(1, env.action_space.shape[0], device=device)
                    env_full[:, :6] = raw_cmd
                    env_full[:, 6] = -1.0  # always closed
                    obs, _, step_terminated, _, _ = env.step(env_full)
                    terminated |= step_terminated

                    _tcp_z_check = _tcp_pos_local()[:, 2]
                    terminated |= (_tcp_z_check < -0.01)

                env.push_count[0] += 1
                pushes_used = int(env.push_count[0].item())

                ee_pos_local = _tcp_pos_local()
                ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
                prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
                obs = env._get_push_obs()

                cur_obj_pos = obs[0, env.robot_dim:env.robot_dim + 3]
                cur_obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
                goal_pos = obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
                goal_euler = obs[0, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]

                pos_err = (cur_obj_pos - goal_pos).norm().item()
                rot_err = _rot_distance_rad(cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)).item()
                if _obj_type == "disc":
                    rot_err = 0.0

                obj_z = float(obs[0, env.robot_dim + 2])
                tipped = (abs(float(cur_obj_euler[0])) > 0.3 or abs(float(cur_obj_euler[1])) > 0.3)
                oob_2d = float((cur_obj_pos[:2] - goal_pos[:2]).norm())

                r_dec = float(torch.sqrt((Xs - obj_x.detach())**2 + (Ys - obj_y.detach())**2).item()) if args.rel_act else float(Xs.item())
                _cov = _area_coverage(pos_err, rot_err)
                print(f"  push {push_i:2d}: bins=({', '.join(f'{int(actions[0,i].item()):2d}' for i in range(4))})  "
                      f"r={r_dec:.3f} len={float(length.item()):.3f} θ={math.degrees(float(theta.item())):.0f}°  "
                      f"pos={pos_err:.4f}m rot={rot_err:.3f}rad z={obj_z:.3f} cov={_cov:.1f}%")

                if terminated[0]:
                    stop_reason = "physics"
                    break

                if _obj_type == "disc":
                    _success_check = pos_err < 0.05
                else:
                    _success_check = pos_err < 0.05 and rot_err < args.rot_threshold
                if _success_check:
                    trial_ok = True

                if obj_z > 0.10:
                    stop_reason = "launched"
                    pos_err = prev_pos_err
                    rot_err = prev_rot_err
                    break
                if tipped:
                    stop_reason = "tipped"
                    pos_err = prev_pos_err
                    rot_err = prev_rot_err
                    break
                if oob_2d > _init_oob_2d + 0.20:
                    stop_reason = "oob"
                    break

                prev_pos_err = pos_err
                prev_rot_err = rot_err
                env.capture_pre_push(obs)

            if trial_ok:
                trial_successes += 1

            trial_pushes.append(pushes_used)
            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_rot_err = rot_err

        avg_pushes = int(np.mean(trial_pushes)) if trial_pushes else 0
        sr_pct = trial_successes / TRIAL_COUNT * 100

        result = ValidationResult(
            test_index=test_idx,
            test_name=f"{cfg.name} #{cfg.test_id}",
            test_type=cfg.test_type,
            success=(trial_successes > 0),
            pushes_used=avg_pushes,
            final_pos_error=best_pos_err,
            final_rot_error=best_rot_err,
            area_coverage=_area_coverage(best_pos_err, best_rot_err),
            trial_count=TRIAL_COUNT,
            success_count=trial_successes,
        )
        results.append(result)
        test_cfgs_data.append({
            "start_x": cfg.main_start.x, "start_y": cfg.main_start.y,
            "goal_x": cfg.main_goal_x, "goal_y": cfg.main_goal_y,
            "goal_yaw": cfg.main_goal_yaw,
            "object_type": getattr(cfg, "object_type", "tblock"),
        })
        status = "PASS" if trial_successes > 0 else "FAIL"
        print(f"  {status} | {trial_successes}/{TRIAL_COUNT} = {sr_pct:.0f}% | avg_pushes: {avg_pushes} | "
              f"best_pos_err: {best_pos_err:.4f} | best_rot_err: {best_rot_err:.4f} | cov: {_area_coverage(best_pos_err, best_rot_err):.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_trials = sum(r.trial_count for r in results)
    total_successes = sum(r.success_count for r in results)
    sr = total_successes / total_trials * 100 if total_trials > 0 else 0
    n_tests_passed = sum(1 for r in results if r.success_count > 0)
    avg_pushes = np.mean([r.pushes_used for r in results]) if results else 0

    pos_only = [r for r in results if r.test_type == "pos_only"]
    pos_rot  = [r for r in results if r.test_type == "pos_rot"]
    po_trials = sum(r.trial_count for r in pos_only)
    pr_trials = sum(r.trial_count for r in pos_rot)
    po_successes = sum(r.success_count for r in pos_only)
    pr_successes = sum(r.success_count for r in pos_rot)
    sr_po = po_successes / po_trials * 100 if po_trials > 0 else 0
    sr_pr = pr_successes / pr_trials * 100 if pr_trials > 0 else 0

    avg_cov = np.mean([r.area_coverage for r in results]) if results else 0

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Test configs:  {len(results)}")
    print(f"  Trials:        {total_trials}")
    print(f"  Successes:     {total_successes}")
    print(f"  Success rate:  {sr:.1f}%")
    print(f"  Tests passed:  {n_tests_passed}/{len(results)} ({n_tests_passed/len(results)*100:.0f}% of configs)")
    print(f"  Pos-only SR:   {sr_po:.1f}% ({po_successes}/{po_trials} trials)")
    print(f"  Pos+rot SR:    {sr_pr:.1f}% ({pr_successes}/{pr_trials} trials)")
    print(f"  Avg pushes:    {avg_pushes:.1f}")
    print(f"  Avg coverage:  {avg_cov:.1f}%")
    print(f"{'='*60}")

    for r in results:
        sr_pct = r.success_count / r.trial_count * 100 if r.trial_count > 0 else 0
        status = "PASS" if r.success_count > 0 else "FAIL"
        print(f"  {status:5s} | Test {r.test_index:2d} | {r.test_name:30s} | {r.success_count}/{r.trial_count} ({sr_pct:.0f}%) pushes={r.pushes_used:2d}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as _f:
            writer = _csv.writer(_f)
            writer.writerow(["test_index", "test_name", "test_type", "success", "pushes_used",
                             "pos_err", "rot_err", "area_coverage", "trial_count", "success_count"])
            for r in results:
                writer.writerow([r.test_index, r.test_name, r.test_type, int(r.success),
                                 r.pushes_used, r.final_pos_error, r.final_rot_error, r.area_coverage,
                                 r.trial_count, r.success_count])
        print(f"\n[CSV] Results saved to {args.csv}")

    if args.csv and results:
        try:
            from asyncDualPlayPPO.tests.plot_validation import generate_single_run_plot
            plot_data = [{"test_index": r.test_index, "test_name": r.test_name,
                          "success": r.success, "final_pos_error": r.final_pos_error,
                          "final_rot_error": r.final_rot_error} for r in results]
            plot_path = os.path.splitext(args.csv)[0] + ".png"
            generate_single_run_plot(plot_data, test_cfgs_data, plot_path,
                                    rot_threshold_rad=args.rot_threshold)
        except Exception as _e:
            print(f"[WARN] Plot generation failed: {_e}")

    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
