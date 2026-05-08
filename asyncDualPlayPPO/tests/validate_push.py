"""
Validate a trained push-PPO model against test configurations.

Uses the pre-defined test scenes from validation_configs.py to measure push
success rate, steps-to-solve, and push count across multiple scenarios.

Usage:
  python -m asyncDualPlayPPO.tests.validate_push \
      --chkpt runs/push_ppo_baseline/agent/model_best.pt \
      --num_tests 10 --headless
"""

import argparse
import os
import sys
import time
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

import isaaclab.envs.mdp as mdp

from asyncDualPlayPPO.tasks.utils.validation_configs import (
    ALL_TESTS, get_test_config, get_test_count, PushTestConfig,
)

from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
from asyncDualPlayPPO.tasks.utils.action_push import (
    decode_push_action, compute_push_waypoints,
)
from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
from asyncDualPlayPPO.algorithms.rl.ppo.storage import RolloutStorage

import copy
import gymnasium as gym_mc

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


@dataclass
class ValidationResult:
    test_index: int
    test_name: str
    success: bool
    pushes_used: int
    final_pos_error: float
    final_rot_error: float


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def main():
    parser = argparse.ArgumentParser(description="Validate Push-PPO Model")
    parser.add_argument("--chkpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--num_tests", type=int, default=10, help="Number of test scenes to run")
    parser.add_argument("--max_pushes", type=int, default=30, help="Max pushes per test")
    parser.add_argument("--headless", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
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
    )

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
    _wrist3_ids, _ = _robot_scene.find_bodies("wrist_3_link")
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    num_cat_dims = 6
    num_bins = 11

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
            "use_multicategorical": True, "num_cat_dims": 6, "num_bins": 11,
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

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: List[ValidationResult] = []
    n_tests = min(args.num_tests, get_test_count())

    for test_idx in range(1, n_tests + 1):
        cfg = get_test_config(test_idx)
        if cfg is None:
            continue

        print(f"\n[Test {test_idx}/{n_tests}] {cfg.name} #{cfg.test_id}")

        obs = env.reset()
        env.goal_pos_euler[0, 0] = cfg.main_goal_x
        env.goal_pos_euler[0, 1] = cfg.main_goal_y
        env.goal_pos_euler[0, 2] = 0.0
        env.goal_pos_euler[0, 3:] = 0.0
        env._update_goal_in_extras()

        # Place object at start position
        obj = env.env.scene["target_object"]
        obj.write_root_pose_to_sim(torch.tensor([[
            cfg.main_start.x, cfg.main_start.y, 0.02, 1.0, 0.0, 0.0, 0.0
        ]], device=device))

        env.env.sim.step()
        obs_dict = env.env.observation_manager.compute()
        obs = env._build_obs(obs_dict)
        env._capture_prev_obj(obs)

        hidden = [
            torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
            torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
        ]

        def _tcp_pos_local():
            lfw = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
            rfw = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
            return ((lfw + rfw) / 2.0 - env.env.scene.env_origins).clone()

        def _tcp_offset():
            lfw = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
            rfw = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
            w3w = _robot_scene.data.body_pos_w[:, _wrist3_ids[0]]
            return (lfw + rfw) / 2.0 - w3w

        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

        test_success = False
        pushes_used = 0

        for push_i in range(args.max_pushes):
            with torch.no_grad():
                actions, _, _, _, _, new_h = agent.actor_critic.act_with_hidden(
                    obs, None, (hidden[0], hidden[1]),
                )
                hidden[0] = new_h[0]
                hidden[1] = new_h[1]

            push_params = decode_push_action(actions, num_bins=num_bins)
            obj_pos = env._get_obj_pos(obs)

            waypoints = compute_push_waypoints(
                offset_x=push_params[:, 0], offset_y=push_params[:, 1],
                push_dx=push_params[:, 2], push_dy=push_params[:, 3],
                yaw=push_params[:, 4], push_dz=push_params[:, 5],
                obj_pos=obj_pos, current_ee_pos=ee_pos_local,
                current_ee_quat=ee_quat_w, device=device,
            )

            terminated = torch.zeros(1, dtype=torch.bool, device=device)
            for wp_pos, wp_quat, wp_grip in waypoints:
                tcp_offs = _tcp_offset()
                ik_target = wp_pos - tcp_offs
                result = ik_solver.solve_batch(
                    CuroboPose(position=ik_target, quaternion=wp_quat),
                    seed_config=prev_joint_cmd.unsqueeze(1),
                    retract_config=prev_joint_cmd,
                )
                ik_ok = result.success.squeeze(-1)
                cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
                solved = result.solution.view(1, 6)
                raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, cur_joints)
                prev_joint_cmd = raw_cmd.detach().clone()

                env_full = torch.zeros(1, env.action_space.shape[0], device=device)
                env_full[:, :6] = raw_cmd
                env_full[:, 6] = wp_grip
                obs, _, terminated, _, _ = env.step(env_full)

            env.push_count[0] += 1
            pushes_used = int(env.push_count[0].item())

            cur_obj_pos = obs[0, env.robot_dim:env.robot_dim + 3]
            cur_obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
            goal_pos = obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
            goal_euler = obs[0, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]

            pos_err = (cur_obj_pos - goal_pos).norm().item()
            rot_err = _rot_distance_rad(cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)).item()

            if pos_err < 0.05 and rot_err < 0.035:
                test_success = True
                break

            if terminated[0]:
                break

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            env._capture_prev_obj(obs)

        result = ValidationResult(
            test_index=test_idx,
            test_name=f"{cfg.name} #{cfg.test_id}",
            success=test_success,
            pushes_used=pushes_used,
            final_pos_error=pos_err if not test_success else 0.0,
            final_rot_error=rot_err if not test_success else 0.0,
        )
        results.append(result)
        status = "PASS" if test_success else "FAIL"
        print(f"  {status} | pushes: {pushes_used} | pos_err: {pos_err:.4f} | rot_err: {rot_err:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success = sum(1 for r in results if r.success)
    sr = n_success / len(results) * 100 if results else 0
    avg_pushes = np.mean([r.pushes_used for r in results]) if results else 0

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total tests:   {len(results)}")
    print(f"  Successes:     {n_success}")
    print(f"  Success rate:  {sr:.1f}%")
    print(f"  Avg pushes:    {avg_pushes:.1f}")
    print(f"{'='*60}")

    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  {status:5s} | Test {r.test_index:2d} | {r.test_name:30s} | pushes={r.pushes_used:2d}")

    simulation_app.close()


if __name__ == "__main__":
    main()
