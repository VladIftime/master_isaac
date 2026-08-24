"""
Push-T model comparison evaluation suite.

Evaluates push-PPO and push-ASP Bob models on 10 standardized tests.
All models face identical (start, goal) pairs for fair comparison.

Run:
  python -m asyncDualPlayPPO.tests.eval_suite
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this section to select models and settings
# ═══════════════════════════════════════════════════════════════════════════════
# Add/remove entries freely. A single entry evaluates one model.
# Types:
#   "push_ppo"     — ActorCriticPush (module_push.py), no GoalEncoder
#   "push_asp_bob" — ActorCritic (module.py), GoalEncoder + PI encoder
#
# rel_act: decode actions as object-relative (r, phi, len, theta)
# rel_obs: for push_ppo appends [rel_dx, rel_dy] (28→30D);
#           for push_asp_bob replaces goal_dist(2) with [rel_dx, rel_dy] (28D)

MODELS = [
    {
        "name": "Push-PPO (abs)",
        "type": "push_ppo",
        "rel_act": False,
        "rel_obs": False,
        "checkpoint": "/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/runs/26.06.09/hpc_push_2048env/agent/model_best.pt",
    },
    {
        "name": "Push-PPO (rel_full)",
        "type": "push_ppo",
        "rel_act": True,
        "rel_obs": True,
        "checkpoint": "/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/runs/26.06.09/hpc_push_2048env_rel_full/agent/model_best.pt",
    },
    {
        "name": "Push-ASP Bob",
        "type": "push_asp_bob",
        "rel_act": True,
        "rel_obs": True,
        "checkpoint": "/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/runs/26.06.09/hpc_push_asp_2048env/bob/model_best.pt",
    },
]

MAX_PUSHES = 10
MASTER_SEED = 42
HEADLESS = False
SUCCESS_POS_THRESH = 0.05
SUCCESS_ROT_THRESH = 0.2
OUTPUT_CSV = "results/eval_push_results.csv"
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
_WS_Z = (0.25, 0.55)


def main():
    parser = argparse.ArgumentParser(description="Push-T Evaluation Suite")
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

    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper, _euler_to_quat
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints,
    )
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
        TBLOCK_MIN_R, TBLOCK_MAX_R,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic
    from asyncDualPlayPPO.tests.eval_test_defs import (
        TESTS, generate_episodes, get_test_def,
    )

    ppo_cfg_path = os.path.join(os.path.dirname(__file__), "..", "cfg/ppo/ppo_continuous.yaml")
    with open(ppo_cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    num_cat_dims = 4
    num_bins = 21

    # ── Environment setup (single env, shared across all models) ──────────────
    print("[Eval] Creating environment (num_envs=1)...")
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.scene.cube = None

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

    env = PushEnvWrapper(
        env=base_env,
        device=device,
        num_objects=1,
        max_pushes_per_episode=MAX_PUSHES,
        headless=HEADLESS,
        rel_obs=False,
    )
    print(f"[Eval] Environment ready (obs_dim={env.obs_dim}D).")

    # ── cuRobo IK solver ──────────────────────────────────────────────────────
    print("[cuRobo] Initialising IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    _ik_config.solver.newton_optimizer.n_iters = 30
    _ik_config.solver.newton_optimizer.inner_iters = 10
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
    _push_viz_start = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushStart",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
    )
    _push_viz_end = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushEnd",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )
    _push_viz_arrow = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushArrow",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=0.005, height=0.30,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
            },
        )
    )

    def _update_goal_markers():
        goals = env.goal_pos_euler
        origins = env.env.scene.env_origins
        pos = goals[:, :3].clone()
        pos[:, 2] = 0.001
        euler = goals[:, 3:6].clone()
        euler[:, 0] = 0.0
        euler[:, 1] = 0.0
        quat = _euler_to_quat(euler)
        _goal_viz.visualize(translations=pos + origins, orientations=quat)

    _ident_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    def _update_push_markers(Xs, Ys, Xf, Yf, theta):
        try:
            N = Xs.shape[0]
            origins = env.env.scene.env_origins
            z_table = 0.002
            ident = _ident_quat.expand(N, 4)
            start_pos = torch.stack([Xs, Ys, torch.full((N,), z_table, device=device)], dim=-1) + origins
            end_pos = torch.stack([Xf, Yf, torch.full((N,), z_table, device=device)], dim=-1) + origins
            _push_viz_start.visualize(translations=start_pos, orientations=ident)
            _push_viz_end.visualize(translations=end_pos, orientations=ident)
            mid_w = torch.stack([(Xs + Xf) / 2, (Ys + Yf) / 2,
                                 torch.full((N,), z_table, device=device)], dim=-1) + origins
            half = math.pi / 4
            ch, sh = math.cos(half), math.sin(half)
            arrow_quat = torch.stack([
                torch.full((N,), ch, device=device),
                -sh * torch.sin(theta),
                sh * torch.cos(theta),
                torch.zeros(N, device=device),
            ], dim=-1)
            _push_viz_arrow.visualize(translations=mid_w, orientations=arrow_quat)
        except Exception:
            pass

    # ── Robot body/joint indices ──────────────────────────────────────────────
    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    # ── IK-physics calibration ────────────────────────────────────────────────
    print("[Setup] Calibrating IK-physics error...")
    _calib_pos = torch.zeros(1, 3, device=device)
    _calib_pos[:, 1] = 0.60
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN),
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

    # ── Helper: yaw to quaternion (wxyz) ──────────────────────────────────────
    def _yaw_to_quat(yaw: float) -> torch.Tensor:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return torch.tensor([[cy, 0.0, 0.0, sy]], device=device, dtype=torch.float32)

    # ── Helper: rotation distance ─────────────────────────────────────────────
    def _rot_distance_rad(euler_a, euler_b):
        diff = (euler_a - euler_b) % (2.0 * torch.pi)
        diff = torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)
        return diff.max(dim=-1)[0]

    # ── Helper: post-process observation for model ────────────────────────────
    def _adapt_obs(obs_28d, model_cfg):
        if model_cfg["type"] == "push_ppo" and model_cfg.get("rel_obs", False):
            obj_x = obs_28d[:, 6]
            obj_y = obs_28d[:, 7]
            goal_x = obs_28d[:, 20]
            goal_y = obs_28d[:, 21]
            rel_dx = goal_x - obj_x
            rel_dy = goal_y - obj_y
            return torch.cat([obs_28d, rel_dx.unsqueeze(-1), rel_dy.unsqueeze(-1)], dim=-1)
        elif model_cfg["type"] == "push_asp_bob" and model_cfg.get("rel_obs", False):
            obs = obs_28d.clone()
            obj_x = obs[:, 6]
            obj_y = obs[:, 7]
            goal_x = obs[:, 20]
            goal_y = obs[:, 21]
            obs[:, 26] = goal_x - obj_x
            obs[:, 27] = goal_y - obj_y
            return obs
        return obs_28d

    # ── Helper: determine obs dimension for model ─────────────────────────────
    def _obs_dim_for_model(model_cfg):
        if model_cfg["type"] == "push_ppo" and model_cfg.get("rel_obs", False):
            return 30
        return 28

    # ── Model config builders ─────────────────────────────────────────────────
    def _build_push_ppo_model(obs_dim):
        policy_cfg = ppo_cfg["params"]["policy"].copy()
        policy_cfg["num_cat_dims"] = num_cat_dims
        policy_cfg["num_bins"] = num_bins
        model = ActorCriticPush(
            obs_shape=(obs_dim,),
            states_shape=(obs_dim,),
            actions_shape=(num_cat_dims,),
            init_noise_std=1.0,
            model_cfg=policy_cfg,
            asymmetric=False,
        ).to(device)
        return model

    def _build_push_asp_bob_model(obs_dim):
        policy_cfg = ppo_cfg["params"]["policy"].copy()
        policy_cfg["use_pi_encoder"] = True
        policy_cfg["use_multicategorical"] = True
        policy_cfg["use_lstm"] = True
        policy_cfg["use_goal_encoder"] = True
        policy_cfg["num_cat_dims"] = num_cat_dims
        policy_cfg["num_bins"] = num_bins
        policy_cfg["num_objects"] = 1
        policy_cfg["robot_state_dim"] = 6
        policy_cfg["goal_embed_dim"] = 8
        model = ActorCritic(
            obs_shape=(obs_dim,),
            states_shape=(obs_dim,),
            actions_shape=(num_cat_dims,),
            initial_std=1.0,
            model_cfg=policy_cfg,
            asymmetric=False,
        ).to(device)
        return model

    # ── Load model weights ────────────────────────────────────────────────────
    def _load_model(model, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            return False
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return True

    # ── Initialize environment ────────────────────────────────────────────────
    print("[Eval] Initialising environment...")
    with SuppressAllOutput():
        env.reset()

    _close_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _close_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
    _close_act[:, 6] = -1.0
    env.step(_close_act)

    # ── Results storage ───────────────────────────────────────────────────────
    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION LOOP
    # ══════════════════════════════════════════════════════════════════════════
    for model_idx, model_cfg in enumerate(MODELS):
        model_name = model_cfg["name"]
        model_type = model_cfg["type"]
        checkpoint = model_cfg["checkpoint"]
        rel_act = model_cfg.get("rel_act", False)

        print(f"\n{'='*70}")
        print(f"MODEL {model_idx+1}/{len(MODELS)}: {model_name}")
        print(f"  type={model_type}  rel_act={rel_act}  checkpoint={checkpoint}")
        print(f"{'='*70}")

        obs_dim = _obs_dim_for_model(model_cfg)
        if model_type == "push_ppo":
            actor_critic = _build_push_ppo_model(obs_dim)
        elif model_type == "push_asp_bob":
            actor_critic = _build_push_asp_bob_model(obs_dim)
        else:
            print(f"[ERROR] Unknown model type: {model_type}")
            continue

        if not _load_model(actor_critic, checkpoint):
            continue

        lstm_hidden_size = actor_critic.lstm_hidden_size if hasattr(actor_critic, "lstm_hidden_size") else 256
        print(f"[Eval] Model loaded. obs_dim={obs_dim}, LSTM={lstm_hidden_size}")

        model_results = []

        for test_def in TESTS:
            test_id = test_def.test_id
            test_name = test_def.name
            n_episodes = test_def.n_episodes

            episodes = generate_episodes(test_id, MASTER_SEED)
            successes = 0
            pos_errors = []
            rot_errors = []
            pushes_list = []

            print(f"\n  [Test {test_id:2d}] {test_name} ({n_episodes} episodes)")

            for ep_idx, ep_cfg in enumerate(episodes):
                env.reset()

                obj = env.env.scene["target_object"]
                quat = _yaw_to_quat(ep_cfg.start_yaw)
                pose = torch.tensor([[
                    ep_cfg.start_x, ep_cfg.start_y, 0.02,
                    quat[0, 0].item(), quat[0, 1].item(), quat[0, 2].item(), quat[0, 3].item(),
                ]], device=device)
                obj.write_root_pose_to_sim(pose)
                env.env.scene.write_data_to_sim()

                env.goal_pos_euler[0, 0] = ep_cfg.goal_x
                env.goal_pos_euler[0, 1] = ep_cfg.goal_y
                env.goal_pos_euler[0, 2] = 0.0
                env.goal_pos_euler[0, 3] = 0.0
                env.goal_pos_euler[0, 4] = 0.0
                env.goal_pos_euler[0, 5] = ep_cfg.goal_yaw
                env._update_goal_in_extras()

                for _ in range(5):
                    hold_act = torch.zeros(1, env.action_space.shape[0], device=device)
                    hold_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
                    hold_act[:, 6] = -1.0
                    env.step(hold_act)

                obs = env._get_push_obs()
                env._capture_prev_obj(obs)
                _update_goal_markers()

                hidden = [
                    torch.zeros(1, lstm_hidden_size, device=device),
                    torch.zeros(1, lstm_hidden_size, device=device),
                ]
                ee_pos_local = _tcp_pos_local()
                ee_quat_w = _QUAT_TOOL_DOWN.clone()
                prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

                ep_success = False
                ep_pushes = 0
                ep_pos_err = 999.0
                ep_rot_err = 999.0
                ep_end_reason = "max_pushes"

                for push_i in range(MAX_PUSHES):
                    obs_model = _adapt_obs(obs, model_cfg)

                    with torch.no_grad():
                        h_in = (hidden[0], hidden[1])
                        if model_type == "push_ppo":
                            actions, _, _, _, _, _, new_h = actor_critic.act_with_hidden(
                                obs_model, None, h_in,
                            )
                        else:
                            actions, _, _, _, _, new_h = actor_critic.act_with_hidden(
                                obs_model, None, h_in,
                            )
                        if new_h is not None:
                            hidden[0] = new_h[0]
                            hidden[1] = new_h[1]

                    if rel_act:
                        obj_x = obs[:, 6]
                        obj_y = obs[:, 7]
                        obj_yaw = obs[:, 11]
                        obj_xy = torch.stack([obj_x, obj_y], dim=-1)
                        Xs, Ys, length, theta = decode_push_action_relative(
                            actions, obj_xy, obj_yaw, num_bins=num_bins,
                            min_r=TBLOCK_MIN_R, max_r=TBLOCK_MAX_R,
                        )
                    else:
                        Xs, Ys, length, theta = decode_push_action(actions, num_bins=num_bins)

                    Xf = Xs + length * torch.cos(theta)
                    Yf = Ys + length * torch.sin(theta)

                    waypoints = compute_push_waypoints(
                        Xs=Xs, Ys=Ys, length=length, theta=theta,
                        current_ee_pos=ee_pos_local,
                        current_ee_quat=ee_quat_w,
                        device=device,
                    )

                    _update_push_markers(Xs, Ys, Xf, Yf, theta)

                    terminated = torch.zeros(1, dtype=torch.bool, device=device)
                    for wp_pos, wp_quat, _wp_grip in waypoints:
                        if terminated[0]:
                            break
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
                        raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, cur_joints)
                        prev_joint_cmd = raw_cmd.detach().clone()

                        env_full = torch.zeros(1, env.action_space.shape[0], device=device)
                        env_full[:, :6] = raw_cmd
                        env_full[:, 6] = -1.0
                        obs, _, step_term, _, _ = env.step(env_full)
                        terminated |= step_term

                        tcp_z = _tcp_pos_local()[0, 2].item()
                        if tcp_z < 0.01:
                            terminated[0] = True

                        obj_z = obs[0, 8].item()
                        if obj_z > 0.08:
                            terminated[0] = True

                    ep_pushes += 1

                    cur_obj_pos = obs[0, 6:9]
                    cur_obj_euler = obs[0, 9:12]
                    goal_pos = obs[0, 20:23]
                    goal_euler = obs[0, 23:26]

                    ep_pos_err = (cur_obj_pos - goal_pos).norm().item()
                    ep_rot_err = _rot_distance_rad(
                        cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)
                    ).item()

                    if ep_pos_err < SUCCESS_POS_THRESH and ep_rot_err < SUCCESS_ROT_THRESH:
                        ep_success = True
                        ep_end_reason = "success"
                        break

                    _ox = cur_obj_pos[0].item()
                    _oy = cur_obj_pos[1].item()
                    _oz = cur_obj_pos[2].item()
                    if abs(_ox) > 0.75 or _oy < 0.1 or _oy > 1.0 or _oz < -0.1 or _oz > 0.5:
                        ep_end_reason = "off_table"
                        break

                    if terminated[0]:
                        ep_end_reason = "terminated"
                        break

                    ee_pos_local = _tcp_pos_local()
                    ee_quat_w = _QUAT_TOOL_DOWN.clone()
                    env._capture_prev_obj(obs)

                if ep_success:
                    successes += 1
                pos_errors.append(ep_pos_err)
                rot_errors.append(ep_rot_err)
                pushes_list.append(ep_pushes)

                seed_used = MASTER_SEED * 1000 + test_id * 100 + ep_idx
                all_results.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "checkpoint": checkpoint,
                    "test_id": test_id,
                    "test_name": test_name,
                    "episode_idx": ep_idx,
                    "seed": seed_used,
                    "success": int(ep_success),
                    "pos_error": round(ep_pos_err, 5),
                    "rot_error": round(ep_rot_err, 5),
                    "pushes_used": ep_pushes,
                })

                tag = "SUCCESS" if ep_success else "FAIL   "
                final_obj = obs[0, 6:9]
                final_yaw = obs[0, 11].item()
                print(
                    f"    ep {ep_idx+1:3d}/{n_episodes}  {tag}  "
                    f"pushes={ep_pushes:2d}  "
                    f"start=({ep_cfg.start_x:+.3f},{ep_cfg.start_y:+.3f}) yaw={math.degrees(ep_cfg.start_yaw):+6.1f}°  "
                    f"goal=({ep_cfg.goal_x:+.3f},{ep_cfg.goal_y:+.3f}) yaw={math.degrees(ep_cfg.goal_yaw):+6.1f}°  "
                    f"final=({final_obj[0]:+.3f},{final_obj[1]:+.3f}) yaw={math.degrees(final_yaw):+6.1f}°  "
                    f"pos_err={ep_pos_err:.4f}m  rot_err={ep_rot_err:.4f}rad  "
                    f"end={ep_end_reason}"
                )

            sr = successes / n_episodes * 100
            avg_pe = np.mean(pos_errors)
            avg_re = np.mean(rot_errors)
            avg_p = np.mean(pushes_list)
            print(f"    RESULT: SR={sr:.1f}%  PosErr={avg_pe:.4f}m  "
                  f"RotErr={avg_re:.4f}rad  AvgPushes={avg_p:.1f}")
            model_results.append((test_id, test_name, sr, avg_pe, avg_re, avg_p))

        print(f"\n  {'─'*60}")
        print(f"  SUMMARY: {model_name}")
        print(f"  {'─'*60}")
        print(f"  {'Test':<25s} {'SR%':>6s} {'PosErr':>8s} {'RotErr':>8s} {'Pushes':>7s}")
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

    # ── Final comparison table ────────────────────────────────────────────────
    if len(MODELS) > 1:
        print(f"\n{'═'*70}")
        print(f"COMPARISON TABLE")
        print(f"{'═'*70}")
        print(f"{'Model':<25s} {'SR%':>6s} {'PosErr':>8s} {'RotErr':>8s} {'Pushes':>7s}")
        print(f"{'─'*25} {'─'*6} {'─'*8} {'─'*8} {'─'*7}")
        for mcfg in MODELS:
            mname = mcfg["name"]
            m_rows = [r for r in all_results if r["model_name"] == mname]
            if not m_rows:
                continue
            sr_val = np.mean([r["success"] for r in m_rows]) * 100
            pe_val = np.mean([r["pos_error"] for r in m_rows])
            re_val = np.mean([r["rot_error"] for r in m_rows])
            ap_val = np.mean([r["pushes_used"] for r in m_rows])
            print(f"{mname:<25s} {sr_val:5.1f}% {pe_val:8.4f} {re_val:8.4f} {ap_val:6.1f}")
        print(f"{'═'*70}")

    simulation_app.close()


if __name__ == "__main__":
    main()
