"""
cuRobo step-based Bob evaluation suite.

Evaluates ASP Bob (from train_curobo.py) on 10 standardized tests.
Bob uses per-step EE delta actions (6D x 11 bins) with cuRobo IK.

Run:
  python -m asyncDualPlayPPO.tests.eval_suite_curobo
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
MODELS = [
    {
        "name": "ASP Bob (cuRobo step)",
        "checkpoint": "runs/curobo_v1/bob/model_best.pt",
    },
]

BOB_TIMESTEPS = 100
MASTER_SEED = 42
HEADLESS = False
SUCCESS_POS_THRESH = 0.05
SUCCESS_ROT_THRESH = 0.2
OUTPUT_CSV = "results/eval_curobo_results.csv"
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print("[ERROR] cuRobo not found.")
    import sys
    sys.exit(1)

import torch
import torch._dynamo  # noqa
import torch._C      # noqa
import torch.optim   # noqa

from isaaclab.app import AppLauncher

import os
import sys
import csv
import math
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)
_WS_Z = (0.00, 0.55)

_EE_HOME_X_OFFSET = 0.02
_EE_HOME_Y = 0.50
_EE_HOME_Z = 0.05


def main():
    parser = argparse.ArgumentParser(description="cuRobo Step-Based Bob Evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = HEADLESS

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    import yaml
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    from asyncDualPlayPPO.tasks.async_dual_play_curobo import AsyncDualPlayCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
    from asyncDualPlayPPO.tasks.utils import terminations
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic
    from asyncDualPlayPPO.tests.eval_test_defs import (
        TESTS, generate_episodes, get_test_def,
    )

    ppo_cfg_path = os.path.join(os.path.dirname(__file__), "..", "cfg/ppo/ppo_continuous.yaml")
    with open(ppo_cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    num_cat_dims = 6
    num_bins = 11
    _max_delta_m = 0.02
    _max_delta_rot = 0.10

    # ── Environment ───────────────────────────────────────────────────────────
    print("[Eval] Creating environment (num_envs=1)...")
    env_cfg = AsyncDualPlayCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.terminations.objects_off_table.func = terminations.objects_out_of_bounds
    env_cfg.terminations.objects_off_table.params = {
        "x_range": (-0.75, 0.75),
        "y_range": (0.2, 1.0),
        "z_min": -0.2,
    }
    env_cfg.scene.cube = None
    env_cfg.observations.alice_policy.cube_state = None
    env_cfg.observations.bob_policy.cube_state = None
    env_cfg.observations.bob_policy.cube_goal_state = None
    env_cfg.observations.bob_policy.cube_goal_distance = None

    class SuppressAllOutput:
        def __enter__(self):
            self.stdout_fd = sys.stdout.fileno()
            self.stderr_fd = sys.stderr.fileno()
            self.saved_stdout = os.dup(self.stdout_fd)
            self.saved_stderr = os.dup(self.stderr_fd)
            self.devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(self.devnull, self.stdout_fd)
            os.dup2(self.devnull, self.stderr_fd)
        def __exit__(self, *a):
            os.dup2(self.saved_stdout, self.stdout_fd)
            os.dup2(self.saved_stderr, self.stderr_fd)
            os.close(self.saved_stdout)
            os.close(self.saved_stderr)
            os.close(self.devnull)

    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    alice_timesteps = 100
    bob_timesteps = BOB_TIMESTEPS
    env = AsyncDualPlayEnvWrapper(
        env=base_env,
        alice_timesteps=alice_timesteps,
        bob_timesteps=bob_timesteps,
        max_goals_per_episode=5,
        num_objects=1,
        device=device,
        arm_config="default",
    )
    print("[Eval] Environment ready.")

    # ── cuRobo IK solver ──────────────────────────────────────────────────────
    print("[cuRobo] Initialising IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    ik_solver = IKSolver(_ik_config)

    _wup_pos = torch.zeros(1, 3, device=device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(1, 1, 6, device=device),
        retract_config=torch.zeros(1, 6, device=device),
    )
    print("[cuRobo] IK solver ready.")

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    # ── Visualization markers ─────────────────────────────────────────────────
    _blk_dir = os.path.join(os.path.dirname(__file__), "..", "assets/blocks")
    _goal_viz = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkers",
            markers={
                "tblock": UsdFileCfg(
                    usd_path=os.path.join(_blk_dir, "t_shape.usda"),
                    scale=(2.0, 2.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),
                ),
            },
        )
    )

    def _euler_xyz_to_quat(euler):
        roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack([w, x, y, z], dim=-1)

    def _update_goal_markers(goal_state):
        origins = env.env.scene.env_origins
        pos = torch.zeros(1, 3, device=device)
        pos[0, :2] = goal_state[:2]
        pos[0, 2] = 0.001
        euler = goal_state[3:6].unsqueeze(0).clone()
        euler[:, 0] = 0.0
        euler[:, 1] = 0.0
        quat = _euler_xyz_to_quat(euler)
        _goal_viz.visualize(translations=pos + origins, orientations=quat)

    # ── Robot body/joint indices ──────────────────────────────────────────────
    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    def _quat_mul(a, b):
        w = a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3]
        x = a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,3] - a[:,3]*b[:,2]
        y = a[:,0]*b[:,2] - a[:,1]*b[:,3] + a[:,2]*b[:,0] + a[:,3]*b[:,1]
        z = a[:,0]*b[:,3] + a[:,1]*b[:,2] - a[:,2]*b[:,1] + a[:,3]*b[:,0]
        return torch.stack([w, x, y, z], dim=1)

    def _bins_to_xyz_rxy_gripper(bin_indices, gripper_state):
        center = (num_bins - 1) / 2.0
        threshold = 2.0
        normalized = (bin_indices.float() - center) / center
        xyz = normalized[:, :3] * _max_delta_m
        rxry = normalized[:, 3:5] * _max_delta_rot
        rxry = rxry.clamp(-0.10, 0.10)
        g_bin = bin_indices[:, 5].float()
        new_gs = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0
        new_gs[g_bin > center + threshold - 1] = 1.0
        return xyz, rxry, new_gs

    def _rot_distance_rad(euler_a, euler_b):
        diff = (euler_a - euler_b) % (2.0 * torch.pi)
        diff = torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)
        return diff.max(dim=-1)[0]

    def _yaw_to_quat(yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return torch.tensor([[cy, 0.0, 0.0, sy]], device=device, dtype=torch.float32)

    # ── Build model ───────────────────────────────────────────────────────────
    bob_obs_dim = env.bob_observation_space.shape[0]
    policy_cfg = ppo_cfg["params"]["policy"].copy()
    policy_cfg["use_goal_encoder"] = True
    policy_cfg["num_objects"] = 1
    policy_cfg["num_cat_dims"] = num_cat_dims
    policy_cfg["num_bins"] = num_bins

    # ── Initialize environment ────────────────────────────────────────────────
    print("[Eval] Initialising environment...")
    with SuppressAllOutput():
        env.reset()
    print("[Eval] Ready.")

    # ── Results ───────────────────────────────────────────────────────────────
    all_results = []

    for model_idx, model_cfg in enumerate(MODELS):
        model_name = model_cfg["name"]
        checkpoint = model_cfg["checkpoint"]

        print(f"\n{'='*70}")
        print(f"MODEL {model_idx+1}/{len(MODELS)}: {model_name}")
        print(f"  checkpoint={checkpoint}")
        print(f"{'='*70}")

        actor_critic = ActorCritic(
            obs_shape=(bob_obs_dim,),
            states_shape=(bob_obs_dim,),
            actions_shape=(num_cat_dims,),
            initial_std=1.0,
            model_cfg=policy_cfg,
            asymmetric=False,
        ).to(device)

        if not os.path.isfile(checkpoint):
            print(f"[ERROR] Checkpoint not found: {checkpoint}")
            continue
        ckpt = torch.load(checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        actor_critic.load_state_dict(state_dict, strict=True)
        actor_critic.eval()
        print(f"[Eval] Model loaded. obs_dim={bob_obs_dim}")

        lstm_hidden_size = actor_critic.lstm_hidden_size if hasattr(actor_critic, "lstm_hidden_size") else 256
        model_results = []

        for test_def in TESTS:
            test_id = test_def.test_id
            test_name = test_def.name
            n_episodes = test_def.n_episodes

            episodes = generate_episodes(test_id, MASTER_SEED)
            successes = 0
            pos_errors = []
            rot_errors = []
            steps_list = []

            print(f"\n  [Test {test_id:2d}] {test_name} ({n_episodes} episodes)")

            for ep_idx, ep_cfg in enumerate(episodes):
                with SuppressAllOutput():
                    env.reset()

                obj = env.env.scene["target_object"]
                quat = _yaw_to_quat(ep_cfg.start_yaw)
                pose = torch.tensor([[
                    ep_cfg.start_x, ep_cfg.start_y, 0.02,
                    quat[0, 0].item(), quat[0, 1].item(), quat[0, 2].item(), quat[0, 3].item(),
                ]], device=device)
                obj.write_root_pose_to_sim(pose)
                env.env.scene.write_data_to_sim()

                goal_state = torch.tensor([
                    ep_cfg.goal_x, ep_cfg.goal_y, 0.0, 0.0, 0.0, ep_cfg.goal_yaw,
                ], device=device)

                env.episode_manager.current_phase[:] = 1
                env.episode_manager.phase_step[:] = 0
                env.episode_manager.goal_valid[:] = True
                env.episode_manager.goal_states[0] = goal_state
                env.episode_manager.initial_states[0] = torch.tensor([
                    ep_cfg.start_x, ep_cfg.start_y, 0.02, 0.0, 0.0, ep_cfg.start_yaw,
                ], device=device)

                for _ in range(5):
                    hold_act = torch.zeros(1, base_env.action_space.shape[0], device=device)
                    hold_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
                    hold_act[:, 6] = -1.0
                    base_env.step(hold_act)

                _update_goal_markers(goal_state)

                ee_target_local = _tcp_pos_local()
                ee_target_local[:, 0] += _EE_HOME_X_OFFSET
                ee_target_local[:, 1] = _EE_HOME_Y
                ee_target_local[:, 2] = _EE_HOME_Z
                ee_target_quat_w = _QUAT_TOOL_DOWN.clone()
                gripper_state = torch.ones(1, 1, device=device)
                prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

                hidden = [
                    torch.zeros(1, lstm_hidden_size, device=device),
                    torch.zeros(1, lstm_hidden_size, device=device),
                ]

                ep_success = False
                ep_steps = 0
                ep_pos_err = 999.0
                ep_rot_err = 999.0

                for step_i in range(BOB_TIMESTEPS):
                    obs_dict = env.env.observation_manager.compute()
                    bob_obs = obs_dict["bob_policy"]

                    with torch.no_grad():
                        h_in = (hidden[0], hidden[1])
                        actions, _, _, _, _, new_h = actor_critic.act_with_hidden(
                            bob_obs, None, h_in,
                        )
                        if new_h is not None:
                            hidden[0] = new_h[0]
                            hidden[1] = new_h[1]

                    bin_indices = actions.long()
                    xyz_delta, rxry_delta, gripper_state = _bins_to_xyz_rxy_gripper(
                        bin_indices, gripper_state,
                    )

                    ee_target_local = ee_target_local + xyz_delta
                    ee_target_local[:, 0].clamp_(_WS_X[0], _WS_X[1])
                    ee_target_local[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
                    ee_target_local[:, 2].clamp_(_WS_Z[0], _WS_Z[1])

                    rx, ry = rxry_delta[:, 0:1], rxry_delta[:, 1:2]
                    dq_x = torch.zeros(1, 4, device=device)
                    dq_x[:, 0] = torch.cos(rx[:, 0] * 0.5)
                    dq_x[:, 1] = torch.sin(rx[:, 0] * 0.5)
                    dq_y = torch.zeros(1, 4, device=device)
                    dq_y[:, 0] = torch.cos(ry[:, 0] * 0.5)
                    dq_y[:, 2] = torch.sin(ry[:, 0] * 0.5)
                    ee_target_quat_w = _quat_mul(_quat_mul(ee_target_quat_w, dq_x), dq_y)
                    ee_target_quat_w = ee_target_quat_w / ee_target_quat_w.norm(dim=-1, keepdim=True)

                    ik_pos = ee_target_local + env.env.scene.env_origins
                    result = ik_solver.solve_batch(
                        CuroboPose(position=ik_pos, quaternion=ee_target_quat_w),
                        seed_config=prev_joint_cmd.unsqueeze(1),
                        retract_config=prev_joint_cmd,
                    )
                    ik_ok = result.success.squeeze(-1)
                    cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
                    solved = result.solution.view(1, 6)

                    if not ik_ok[0]:
                        ee_target_local = _tcp_pos_local()
                        raw_cmd = cur_joints
                    else:
                        raw_cmd = solved

                    prev_joint_cmd = raw_cmd.detach().clone()

                    env_full = torch.zeros(1, base_env.action_space.shape[0], device=device)
                    env_full[:, :6] = raw_cmd
                    env_full[:, 6] = gripper_state[0, 0]
                    base_env.step(env_full)

                    ep_steps += 1

                    obj_data = env.env.scene["target_object"].data
                    obj_pos_w = obj_data.root_pos_w[0] - env.env.scene.env_origins[0]
                    obj_quat_w = obj_data.root_quat_w[0]

                    w, x, y, z = obj_quat_w[0], obj_quat_w[1], obj_quat_w[2], obj_quat_w[3]
                    sinr = 2.0 * (w * x + y * z)
                    cosr = 1.0 - 2.0 * (x * x + y * y)
                    roll = torch.atan2(sinr, cosr)
                    sinp = 2.0 * (w * y - z * x)
                    pitch = torch.asin(sinp.clamp(-1, 1))
                    siny = 2.0 * (w * z + x * y)
                    cosy = 1.0 - 2.0 * (y * y + z * z)
                    yaw = torch.atan2(siny, cosy)

                    obj_euler = torch.stack([roll, pitch, yaw])
                    goal_euler = goal_state[3:6]

                    ep_pos_err = (obj_pos_w[:2] - goal_state[:2]).norm().item()
                    ep_rot_err = _rot_distance_rad(
                        obj_euler.unsqueeze(0), goal_euler.unsqueeze(0),
                    ).item()

                    if ep_pos_err < SUCCESS_POS_THRESH and ep_rot_err < SUCCESS_ROT_THRESH:
                        ep_success = True
                        break

                if ep_success:
                    successes += 1
                pos_errors.append(ep_pos_err)
                rot_errors.append(ep_rot_err)
                steps_list.append(ep_steps)

                seed_used = MASTER_SEED * 1000 + test_id * 100 + ep_idx
                all_results.append({
                    "model_name": model_name,
                    "model_type": "asp_curobo_bob",
                    "checkpoint": checkpoint,
                    "test_id": test_id,
                    "test_name": test_name,
                    "episode_idx": ep_idx,
                    "seed": seed_used,
                    "success": int(ep_success),
                    "pos_error": round(ep_pos_err, 5),
                    "rot_error": round(ep_rot_err, 5),
                    "pushes_used": ep_steps,
                })

                if (ep_idx + 1) % 10 == 0 or ep_idx == n_episodes - 1:
                    sr_so_far = successes / (ep_idx + 1) * 100
                    print(f"    ep {ep_idx+1:3d}/{n_episodes}  SR={sr_so_far:5.1f}%  "
                          f"pos_err={np.mean(pos_errors[-10:]):.4f}  "
                          f"rot_err={np.mean(rot_errors[-10:]):.4f}  "
                          f"avg_steps={np.mean(steps_list[-10:]):.1f}")

            sr = successes / n_episodes * 100
            avg_pe = np.mean(pos_errors)
            avg_re = np.mean(rot_errors)
            avg_s = np.mean(steps_list)
            print(f"    RESULT: SR={sr:.1f}%  PosErr={avg_pe:.4f}m  "
                  f"RotErr={avg_re:.4f}rad  AvgSteps={avg_s:.1f}")
            model_results.append((test_id, test_name, sr, avg_pe, avg_re, avg_s))

        print(f"\n  {'─'*60}")
        print(f"  SUMMARY: {model_name}")
        print(f"  {'─'*60}")
        print(f"  {'Test':<25s} {'SR%':>6s} {'PosErr':>8s} {'RotErr':>8s} {'Steps':>7s}")
        for tid, tname, sr, pe, re, ap in model_results:
            print(f"  {tname:<25s} {sr:5.1f}% {pe:8.4f} {re:8.4f} {ap:6.1f}")
        overall_sr = np.mean([r[2] for r in model_results])
        overall_pe = np.mean([r[3] for r in model_results])
        overall_re = np.mean([r[4] for r in model_results])
        overall_ap = np.mean([r[5] for r in model_results])
        print(f"  {'OVERALL':<25s} {overall_sr:5.1f}% {overall_pe:8.4f} {overall_re:8.4f} {overall_ap:6.1f}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "model_type", "checkpoint", "test_id", "test_name",
            "episode_idx", "seed", "success", "pos_error", "rot_error", "pushes_used",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[Eval] Results written to {OUTPUT_CSV} ({len(all_results)} rows)")

    simulation_app.close()


if __name__ == "__main__":
    main()
