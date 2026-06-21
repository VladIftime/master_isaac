"""
Collect push demonstrations from a trained Push-PPO expert into a
diffusion_policy zarr ReplayBuffer for offline behavior cloning.

For every push the expert performs, one ``(obs, action)`` pair is recorded:
  - obs:    28D push observation (ee_pose|obj_state|goal_pose|goal_dist)
  - action: the push primitive, by default encoded as
            ``[Xs, Ys, length, sin(theta), cos(theta)]`` (5D), matching the
            default decoding in ``validate_push_diffusion.py``.

Pushes that launch / tip / push out-of-bounds the object are discarded (the
episode ends without recording that final bad push).  The resulting zarr is fed
to ``train_diffusion_unet_lowdim_push_workspace.yaml``.

Usage:
  # Absolute Push-PPO expert:
  python -m asyncDualPlayPPO.tests.collect_push_demos \
      --chkpt runs/push_ppo_baseline/agent/model_best.pt \
      --num_episodes 500 --max_pushes 15 --headless \
      --out diffusion_policy/data/push/push_demos.zarr

  # PBRS rel-obs/rel-act expert (e.g. hpc_pbrs_simp_528env): the expert reads 30D
  # rel-obs and emits relative actions, but we still record the 28D absolute obs
  # and the decoded absolute primitive, so the DP pipeline stays 28D/absolute.
  python -m asyncDualPlayPPO.tests.collect_push_demos \
      --chkpt asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/agent/model_best.pt \
      --rel-obs --rel-act --num_episodes 500 --headless \
      --out diffusion_policy/data/push/push_demos.zarr
"""

import argparse
import os
import sys
import math
from typing import List

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

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def main():
    parser = argparse.ArgumentParser(description="Collect Push-PPO demos for Diffusion Policy")
    parser.add_argument("--chkpt", type=str, required=True, help="Path to trained Push-PPO checkpoint")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of demonstration episodes")
    parser.add_argument("--max_pushes", type=int, default=15, help="Max pushes per episode")
    parser.add_argument("--rot_threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians (default 0.2)")
    parser.add_argument("--rel-obs", action="store_true", dest="rel_obs",
                        help="Expert uses object-relative observation (30D); required for PBRS rel models")
    parser.add_argument("--rel-act", action="store_true", dest="rel_act",
                        help="Decode expert actions as object-relative (r, phi, len, theta)")
    parser.add_argument("--theta_encoding", type=str, default="sincos",
                        choices=["sincos", "raw"],
                        help="Action theta encoding: sincos (5D) or raw (4D)")
    parser.add_argument("--keep_failed", action="store_true",
                        help="Also record the final push that launched/tipped/OOB the object")
    parser.add_argument("--out", type=str, default=None,
                        help="Output zarr path (default: diffusion_policy/data/push/push_demos.zarr)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.out is None:
        _master = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        args.out = os.path.join(_master, "diffusion_policy", "data", "push", "push_demos.zarr")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints,
    )
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
    import gymnasium as gym_mc

    # ReplayBuffer lives in the diffusion_policy package.
    _dp_repo = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "diffusion_policy")
    if _dp_repo not in sys.path:
        sys.path.insert(0, _dp_repo)
    from diffusion_policy.common.replay_buffer import ReplayBuffer

    # ── Environment ────────────────────────────────────────────────────────────
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushEnvWrapper(
        env=base_env, device=device, num_objects=1,
        max_pushes_per_episode=args.max_pushes, rel_obs=args.rel_obs,
    )
    # Always record the 28D absolute base obs (rel_obs only appends rel_dx/dy),
    # so the diffusion policy / validator stay in absolute 28D space.
    OBS_RECORD_DIM = env.robot_dim + env.obj_state_dim + 8  # ee(6)+obj(14)+goal(6)+dist(2)

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
        seed_config=_calib_cur.unsqueeze(1), retract_config=_calib_cur,
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

    # ── Load PPO expert ────────────────────────────────────────────────────────
    num_cat_dims, num_bins = 4, 21
    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32)
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
            "pi_hid_sizes": [512, 256, 128], "vf_hid_sizes": [512, 256, 128],
            "activation": "relu",
        },
    }
    agent = PPO(vec_env=env, cfg_train=agent_cfg, device=device,
                sampler="sequential", log_dir="/tmp/collect_push", asymmetric=False)
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
    print(f"[Collect] Loaded expert from {args.chkpt}")
    print(f"[Collect] episodes={args.num_episodes} max_pushes={args.max_pushes} "
          f"rel_obs={args.rel_obs} rel_act={args.rel_act} "
          f"theta_encoding={args.theta_encoding} record_obs_dim={OBS_RECORD_DIM} out={args.out}")

    def _encode_action(Xs, Ys, length, theta):
        xs = float(Xs.item()); ys = float(Ys.item())
        ln = float(length.item()); th = float(theta.item())
        if args.theta_encoding == "sincos":
            return np.array([xs, ys, ln, math.sin(th), math.cos(th)], dtype=np.float32)
        return np.array([xs, ys, ln, th], dtype=np.float32)

    replay_buffer = ReplayBuffer.create_from_path(args.out, mode='a')

    total_pairs = 0
    n_recorded_eps = 0
    for ep in range(args.num_episodes):
        obs = env.reset()
        env._capture_prev_obj(obs)

        hidden = [
            torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
            torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
        ]
        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

        ep_obs: List[np.ndarray] = []
        ep_act: List[np.ndarray] = []

        prev_pos_err = (obs[0, env.robot_dim:env.robot_dim + 3]
                        - obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
                        ).norm().item()
        init_oob = float((obs[0, env.robot_dim:env.robot_dim + 2]
                          - obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 2]
                          ).norm().item())

        for push_i in range(args.max_pushes):
            obs_before = obs[0, :OBS_RECORD_DIM].detach().cpu().numpy().astype(np.float32).copy()

            with torch.no_grad():
                actions, _, _, _, _, _, new_h = agent.actor_critic.act_with_hidden(
                    obs, None, (hidden[0], hidden[1]))
                if new_h is not None:
                    hidden[0], hidden[1] = new_h[0], new_h[1]

            if args.rel_act:
                obj_x = obs[0, env.robot_dim]
                obj_y = obs[0, env.robot_dim + 1]
                obj_yaw = obs[0, env.robot_dim + 5]
                Xs, Ys, length, theta = decode_push_action_relative(
                    actions,
                    torch.stack([obj_x, obj_y]).unsqueeze(0),
                    obj_yaw.unsqueeze(0),
                    num_bins=num_bins,
                )
            else:
                Xs, Ys, length, theta = decode_push_action(actions, num_bins=num_bins)
            action_vec = _encode_action(Xs, Ys, length, theta)

            waypoints = compute_push_waypoints(
                Xs=Xs, Ys=Ys, length=length, theta=theta,
                current_ee_pos=ee_pos_local, current_ee_quat=ee_quat_w, device=device)

            terminated = torch.zeros(1, dtype=torch.bool, device=device)
            for (wp_pos, wp_quat, _grip) in waypoints:
                ik_target = wp_pos - _TOTAL_IK_ERROR
                ik_target[:, 0].clamp_(_WS_X[0], _WS_X[1])
                ik_target[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
                ik_target[:, 2].clamp_(_WS_Z[0], _WS_Z[1])
                result = ik_solver.solve_batch(
                    CuroboPose(position=ik_target, quaternion=wp_quat),
                    seed_config=prev_joint_cmd.unsqueeze(1), retract_config=prev_joint_cmd)
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
                obs, _, step_terminated, _, _ = env.step(env_full)
                terminated |= step_terminated
                terminated |= (_tcp_pos_local()[:, 2] < -0.01)

            env.push_count[0] += 1
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
            obj_z = float(obs[0, env.robot_dim + 2])
            tipped = (abs(float(cur_obj_euler[0])) > 0.3 or abs(float(cur_obj_euler[1])) > 0.3)
            oob_2d = float((cur_obj_pos[:2] - goal_pos[:2]).norm())

            catastrophic = (bool(terminated[0]) or obj_z > 0.10 or tipped
                            or oob_2d > init_oob + 0.20)
            success = pos_err < 0.05 and rot_err < args.rot_threshold

            if catastrophic and not args.keep_failed:
                break  # discard this bad push

            ep_obs.append(obs_before)
            ep_act.append(action_vec)

            env.capture_pre_push(obs)
            prev_pos_err = pos_err

            if success or catastrophic:
                break

        if len(ep_obs) > 0:
            replay_buffer.add_episode({
                "obs": np.stack(ep_obs, axis=0),
                "action": np.stack(ep_act, axis=0),
            })
            n_recorded_eps += 1
            total_pairs += len(ep_obs)

        if (ep + 1) % 10 == 0:
            print(f"[Collect] ep {ep+1}/{args.num_episodes}  "
                  f"recorded_eps={n_recorded_eps}  total_pairs={total_pairs}")

    print(f"\n[Done] Wrote {n_recorded_eps} episodes / {total_pairs} pairs to {args.out}")
    print(f"        obs_dim={replay_buffer['obs'].shape[1]}  "
          f"action_dim={replay_buffer['action'].shape[1]}")

    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
