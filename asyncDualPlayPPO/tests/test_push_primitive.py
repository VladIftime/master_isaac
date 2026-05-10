"""
test_push_primitive.py  —  Interactive push loop
=================================================
Continuously executes push macro-actions with randomised parameters.

Each push randomises approach offset, push displacement, and gripper yaw.
The block accumulates pushes (not teleported).  Every 3 pushes the
environment is fully reset, placing the block back at its initial position.

Loops forever until you close the viewport window or press Ctrl+C.

Usage:
    python -m asyncDualPlayPPO.tests.test_push_primitive
    python -m asyncDualPlayPPO.tests.test_push_primitive --step-delay 0.05
"""

import math
import os
import sys
import time
import torch
import torch._dynamo   # noqa: F401
import torch._C        # noqa: F401
import torch.optim     # noqa: F401

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# cuRobo must be imported before AppLauncher
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print("[ERROR] cuRobo not found. Install it before running this test.")
    sys.exit(1)

from isaaclab.app import AppLauncher

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# ── Randomisation ranges (env-local frame, metres / radians) ─────────────────
_OFFSET_X_RANGE = (-0.05,  0.05)           # gripper lateral approach offset
_OFFSET_Y_RANGE = (-0.10, -0.02)           # gripper depth behind object
_PUSH_DX_RANGE  = (-0.12,  0.12)           # push X component
_PUSH_DY_RANGE  = ( 0.10,  0.28)           # push Y component (forward)
_YAW_RANGE      = (-math.pi / 4, math.pi / 4)   # gripper yaw

_PUSHES_PER_ROUND = 10
_PAUSE_STEPS      = 75    # ~1.5 s at 50 Hz between pushes
_RESET_EVERY_N    = 3     # full env reset every N pushes to return block to start


def _rnd(lo: float, hi: float, device) -> float:
    return lo + (hi - lo) * torch.rand(1, device=device).item()


def _gen_round(n: int, device) -> list:
    return [
        {
            "offset_x": _rnd(*_OFFSET_X_RANGE, device),
            "offset_y": _rnd(*_OFFSET_Y_RANGE, device),
            "push_dx":  _rnd(*_PUSH_DX_RANGE, device),
            "push_dy":  _rnd(*_PUSH_DY_RANGE, device),
            "yaw":      _rnd(*_YAW_RANGE, device),
        }
        for _ in range(n)
    ]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Push primitive interactive loop")
    parser.add_argument(
        "--step-delay", type=float, default=0.0,
        help="Seconds to sleep after each waypoint step (0=fast, 0.05 for visual debugging)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    headless = getattr(args, "headless", False)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints

    # ── Environment ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Push Primitive — Interactive Loop")
    print(f"  {_PUSHES_PER_ROUND} pushes / round  |  reset every {_RESET_EVERY_N} pushes")
    if not headless and args.step_delay == 0.0:
        print("  Tip: --step-delay 0.05 slows down for visual inspection.")
    print("=" * 64)

    print("\n[Setup] Creating environment (num_envs=1)...")
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device   = base_env.device

    env = PushEnvWrapper(env=base_env, device=device, num_objects=1, max_pushes_per_episode=5)

    # ── cuRobo IK ─────────────────────────────────────────────────────────────
    print("[Setup] Initialising cuRobo IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml   = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg   = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config   = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    ik_solver = IKSolver(_ik_config)
    ik_solver.solve_batch(
        CuroboPose(
            position=torch.zeros(1, 3, device=device),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device),
        ),
        seed_config=torch.zeros(1, 1, 6, device=device),
        retract_config=torch.zeros(1, 6, device=device),
    )
    print("[Setup] IK warm-up done.")

    _QUAT_DOWN    = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    _robot_scene  = env.env.scene["robot"]
    _arm_jids, _  = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _    = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _    = _robot_scene.find_bodies("right_inner_finger")
    _w3_ids, _    = _robot_scene.find_bodies("wrist_3_link")

    def _tcp_local() -> torch.Tensor:
        lf = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return (lf + rf) / 2.0 - env.env.scene.env_origins

    def _tcp_offset() -> torch.Tensor:
        lf = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        w3 = _robot_scene.data.body_pos_w[:, _w3_ids[0]]
        return (lf + rf) / 2.0 - w3

    def _viewer_step():
        if not headless:
            simulation_app.update()

    def _pause(n_steps: int):
        hold = torch.zeros(1, env.action_space.shape[0], device=device)
        hold[0, :6] = _robot_scene.data.joint_pos[0, _arm_jids]
        for _ in range(n_steps):
            if not simulation_app.is_running():
                return
            env.step(hold)
            _viewer_step()
            time.sleep(0.02)

    # ── Initial reset ──────────────────────────────────────────────────────────
    print("[Setup] Resetting environment...")
    env.reset()
    _viewer_step()

    # Warm-up hold so the viewer initialises before the first push
    if not headless:
        _pause(20)

    # ── Main loop ─────────────────────────────────────────────────────────────
    round_num = 0
    pushes_since_reset = 0
    print("\n[Loop] Starting — close the viewport window to exit.\n")

    try:
        while simulation_app.is_running():
            round_num += 1
            configs = _gen_round(_PUSHES_PER_ROUND, device)

            print(f"{'='*64}")
            print(f"  Round {round_num}  ({_PUSHES_PER_ROUND} pushes  |  "
                  f"reset every {_RESET_EVERY_N})")
            print(f"{'='*64}")

            for push_num, cfg in enumerate(configs, 1):
                if not simulation_app.is_running():
                    break

                offset_x = cfg["offset_x"]
                offset_y = cfg["offset_y"]
                push_dx  = cfg["push_dx"]
                push_dy  = cfg["push_dy"]
                yaw      = cfg["yaw"]

                # ── Every _RESET_EVERY_N pushes, fully reset the environment ──
                if pushes_since_reset >= _RESET_EVERY_N:
                    print(f"\n  --- Environment reset after {_RESET_EVERY_N} pushes ---")
                    env.reset()
                    _viewer_step()
                    _pause(10)
                    pushes_since_reset = 0

                # ── Get current object position from observation ──────────────
                obs_dict    = env.env.observation_manager.compute()
                obs         = env._build_obs(obs_dict)
                obj_pos_obs = obs[:, env.robot_dim:env.robot_dim + 3].clone()

                # ── Move visual ghost only every reset cycle (reduce physics disturbance) ──
                if pushes_since_reset == 0:
                    goal_local = torch.tensor(
                        [obj_pos_obs[0, 0] + push_dx, obj_pos_obs[0, 1] + push_dy, 0.02],
                        device=device, dtype=torch.float32,
                    )
                    env.goal_pos_euler[0, :3] = goal_local
                    env.goal_pos_euler[0, 3:] = 0.0
                    env._update_goal_in_extras()
                    env._move_goal_ghost(torch.tensor([0], device=device))
                    _viewer_step()

                # ── Compute waypoints ─────────────────────────────────────────
                prev_jcmd  = _robot_scene.data.joint_pos[:, _arm_jids].clone()
                current_ee = _tcp_local()

                waypoints = compute_push_waypoints(
                    offset_x=torch.tensor([offset_x], device=device),
                    offset_y=torch.tensor([offset_y], device=device),
                    push_dx =torch.tensor([push_dx],  device=device),
                    push_dy =torch.tensor([push_dy],  device=device),
                    yaw     =torch.tensor([yaw],       device=device),
                    push_dz =torch.tensor([0.0],       device=device),
                    obj_pos =obj_pos_obs,
                    current_ee_pos =current_ee,
                    current_ee_quat=_QUAT_DOWN.expand(1, 4).clone(),
                    device=device,
                )

                print(
                    f"\n  [{round_num}.{push_num:02d}] "
                    f"obj=({float(obj_pos_obs[0, 0]):+.2f},{float(obj_pos_obs[0, 1]):+.2f})  "
                    f"off=({offset_x:+.2f},{offset_y:+.2f})  "
                    f"push=({push_dx:+.2f},{push_dy:+.2f})  "
                    f"yaw={math.degrees(yaw):+.0f}°",
                    flush=True,
                )

                # ── Execute push ──────────────────────────────────────────────
                ik_ok = 0
                for wp_pos, wp_quat, wp_grip in waypoints:
                    if not simulation_app.is_running():
                        break

                    ik_target = wp_pos - _tcp_offset()
                    result    = ik_solver.solve_batch(
                        CuroboPose(position=ik_target, quaternion=wp_quat),
                        seed_config=prev_jcmd.unsqueeze(1),
                        retract_config=prev_jcmd,
                    )
                    success = result.success.squeeze(-1)
                    if success.any():
                        ik_ok += 1

                    cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
                    raw_cmd    = torch.where(
                        success.unsqueeze(-1), result.solution.view(1, 6), cur_joints
                    )
                    prev_jcmd = raw_cmd.detach().clone()

                    env_full        = torch.zeros(1, env.action_space.shape[0], device=device)
                    env_full[:, :6] = raw_cmd
                    env_full[:, 6]  = wp_grip
                    obs, _, _, _, _ = env.step(env_full)

                    _viewer_step()
                    if args.step_delay > 0.0:
                        time.sleep(args.step_delay)

                # ── Result ────────────────────────────────────────────────────
                n_wp = len(waypoints)
                obj_after = obs[0, env.robot_dim:env.robot_dim + 3]
                disp = obj_after - obj_pos_obs[0]
                print(
                    f"         IK {ik_ok}/{n_wp}  "
                    f"disp=({float(disp[0]):+.3f},{float(disp[1]):+.3f}) m",
                    flush=True,
                )

                pushes_since_reset += 1
                _pause(_PAUSE_STEPS)

    except KeyboardInterrupt:
        print("\n[Loop] Interrupted by user.")

    simulation_app.close()


if __name__ == "__main__":
    main()
