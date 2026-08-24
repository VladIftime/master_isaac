"""
test_push_primitive.py  —  Interactive push loop
=================================================
Cycles through push scenarios. Each scenario is a 3-push sequence:
  1. Push forward/backward 10 cm (offset ~2 cm behind object center)
  2. Drag left/right 10 cm
  3. Spin 45° with gripper closed

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

# Workspace clamp limits
_WS_X = (-0.50, 0.50)
_WS_Y = (0.25,  0.70)
_WS_Z = ( 0.232, 0.55)  # floor = 0.093 TCP + ~0.139 tool0-to-TCP offset

# ── Scenarios: each is a 3-push sequence ─────────────────────────────────────
# Each push: {Xs, Ys, length, theta}
#   Xs, Ys   = push start position in world coords
#   length    = push length
#   theta     = push orientation angle (rad)
SCENARIOS = [
    # S0: push forward-left from right side
    [
        {"Xs": -0.10, "Ys": 0.55, "length": 0.10, "theta": -1.571},
        {"Xs":  0.10, "Ys": 0.50, "length": 0.15, "theta":  3.142},
        {"Xs":  0.05, "Ys": 0.45, "length": 0.12, "theta":  2.356},
    ],
    # S1: push right from left side
    [
        {"Xs": -0.15, "Ys": 0.55, "length": 0.10, "theta": 0.0},
        {"Xs": -0.25, "Ys": 0.45, "length": 0.12, "theta": 0.0},
        {"Xs": -0.35, "Ys": 0.55, "length": 0.10, "theta": 0.0},
    ],
    # S2: push left from right side
    [
        {"Xs": 0.15, "Ys": 0.50, "length": 0.10, "theta": 3.142},
        {"Xs": 0.25, "Ys": 0.60, "length": 0.12, "theta": 3.142},
        {"Xs": 0.35, "Ys": 0.40, "length": 0.10, "theta": 3.142},
    ],
    # S3: push forward from behind
    [
        {"Xs": -0.05, "Ys": 0.35, "length": 0.12, "theta": 1.571},
        {"Xs":  0.05, "Ys": 0.30, "length": 0.12, "theta": 1.571},
        {"Xs": -0.10, "Ys": 0.30, "length": 0.10, "theta": 1.571},
    ],
    # S4: diagonal push
    [
        {"Xs": -0.20, "Ys": 0.60, "length": 0.12, "theta": 0.785},
        {"Xs": -0.10, "Ys": 0.45, "length": 0.12, "theta": 2.356},
        {"Xs":  0.05, "Ys": 0.65, "length": 0.12, "theta": -0.785},
    ],
    # S5: varied approaches at same object
    [
        {"Xs": -0.10, "Ys": 0.55, "length": 0.10, "theta": -1.571},
        {"Xs":  0.10, "Ys": 0.45, "length": 0.08, "theta":  1.571},
        {"Xs": -0.10, "Ys": 0.50, "length": 0.14, "theta":  0.0},
    ],
]

_PAUSE_STEPS = 60  # ~1.2 s between pushes inside a scenario


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
    from isaaclab.utils.math import euler_xyz_from_quat
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints

    # ── Environment ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Push Primitive — Scenario Loop")
    print(f"  {len(SCENARIOS)} scenarios  |  3 pushes per scenario")
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

    def _obj_state_local(name: str):
        """Return (pos, euler_xyz, linvel, angvel) tensors in local frame."""
        obj = env.env.scene[name]
        origins = env.env.scene.env_origins
        pos = obj.data.root_pos_w - origins        # (1, 3)
        roll, pitch, yaw = euler_xyz_from_quat(obj.data.root_quat_w)
        euler = torch.stack([roll, pitch, yaw], dim=-1)  # (1, 3)
        linvel = obj.data.root_lin_vel_w            # (1, 3)
        angvel = obj.data.root_ang_vel_w            # (1, 3)
        return pos, euler, linvel, angvel

    # ── Initial reset ──────────────────────────────────────────────────────────
    print("[Setup] Resetting environment...")
    env.reset()
    _viewer_step()

    # Calibrate the IK→TCP error: cuRobo targets tool0 (its ee_link), but
    # the physics gripper is mounted below tool0 by ~16 cm.  Measuring
    # (actual_TCP − cuRobo_target) gives the constant offset needed to
    # correct every waypoint: ik_target = wp_pos − _FIXED_TCP_OFFSET.
    print("[Setup] Calibrating IK→TCP error (tool0 vs finger midpoint)...")
    _calib_pos = torch.tensor([[0.0, 0.60, 0.25]], device=device)
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_DOWN),
        seed_config=_calib_cur.unsqueeze(1),
        retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(1, 6)
    _calib_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
        _viewer_step()
    _FIXED_TCP_OFFSET = (_tcp_local() - _calib_pos).clone()   # actual_TCP - tool0_target
    print(
        f"[Setup] IK→TCP error = ({float(_FIXED_TCP_OFFSET[0,0]):+.3f}, "
        f"{float(_FIXED_TCP_OFFSET[0,1]):+.3f}, {float(_FIXED_TCP_OFFSET[0,2]):+.3f})"
    )

    # Warm-up hold so the viewer initialises before the first push
    if not headless:
        _pause(20)

    # ── Main loop ─────────────────────────────────────────────────────────────
    scenario_idx = 0
    print("\n[Loop] Starting — runs all 6 scenarios once.\n")

    try:
        while simulation_app.is_running() and scenario_idx < len(SCENARIOS):
            s_mod = scenario_idx % len(SCENARIOS)
            scenario = SCENARIOS[s_mod]
            active_obj_name = "target_object"
            scenario_idx += 1

            _pause(40)  # let object settle on table

            print(f"{'='*64}")
            print(f"  Scenario {scenario_idx}/{len(SCENARIOS)}")
            print(f"{'='*64}")

            for push_i, cfg in enumerate(scenario):
                if not simulation_app.is_running():
                    break

                Xs     = cfg["Xs"]
                Ys     = cfg["Ys"]
                length = cfg["length"]
                theta  = cfg["theta"]

                # ── Compute waypoints ─────────────────────────────────────────
                prev_jcmd  = _robot_scene.data.joint_pos[:, _arm_jids].clone()
                current_ee = _tcp_local()

                waypoints = compute_push_waypoints(
                    Xs=torch.tensor([Xs], device=device),
                    Ys=torch.tensor([Ys], device=device),
                    length=torch.tensor([length], device=device),
                    theta=torch.tensor([theta], device=device),
                    current_ee_pos=current_ee,
                    current_ee_quat=_QUAT_DOWN.expand(1, 4).clone(),
                    device=device,
                )

                obj_pos_pre, _, _, _ = _obj_state_local(active_obj_name)

                label = f"Push {push_i + 1}  len={length:.2f} theta={theta:.2f}"

                print(
                    f"\n  [{scenario_idx}.{push_i + 1}] "
                    f"Xs=({Xs:+.2f},{Ys:+.2f})  "
                    f"{label}",
                    flush=True,
                )

                # ── Execute push (gripper always closed) ──────────────────────
                ik_ok = 0
                last_good_joints = prev_jcmd.clone()
                for wp_i, (wp_pos, wp_quat, _wp_grip) in enumerate(waypoints):
                    if not simulation_app.is_running():
                        break

                    cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]

                    ik_target = wp_pos - _FIXED_TCP_OFFSET
                    ik_target[0, 0].clamp_(_WS_X[0], _WS_X[1])
                    ik_target[0, 1].clamp_(_WS_Y[0], _WS_Y[1])
                    ik_target[0, 2].clamp_(_WS_Z[0], _WS_Z[1])

                    result    = ik_solver.solve_batch(
                        CuroboPose(position=ik_target, quaternion=wp_quat),
                        seed_config=cur_joints.unsqueeze(1),
                        retract_config=cur_joints,
                    )
                    success = result.success.squeeze(-1)
                    if success.any():
                        ik_ok += 1
                        last_good_joints = result.solution.view(1, 6).clone()

                    raw_cmd = last_good_joints.clone()

                    if wp_i % 3 == 0:
                        ee_pos = _tcp_local()
                        _cur_pos, _cur_euler, _, _ = _obj_state_local(active_obj_name)
                        print(
                            f"    wp {wp_i}: "
                            f"ee=({float(ee_pos[0,0]):+.3f},{float(ee_pos[0,1]):+.3f},{float(ee_pos[0,2]):+.3f})  "
                            f"tgt=({float(wp_pos[0,0]):+.3f},{float(wp_pos[0,1]):+.3f},{float(wp_pos[0,2]):+.3f})  "
                            f"obj=({float(_cur_pos[0,0]):+.3f},{float(_cur_pos[0,1]):+.3f},{float(_cur_pos[0,2]):+.3f})  "
                            f"euler=({float(math.degrees(_cur_euler[0,0])):+.0f},{float(math.degrees(_cur_euler[0,1])):+.0f},{float(math.degrees(_cur_euler[0,2])):+.0f})\u00b0",
                            flush=True,
                        )
                    prev_jcmd = raw_cmd.detach().clone()

                    env_full        = torch.zeros(1, env.action_space.shape[0], device=device)
                    env_full[:, :6] = raw_cmd
                    env_full[:, 6]  = -1.0  # always closed
                    obs, _, _, _, _ = env.step(env_full)

                    _viewer_step()
                    if args.step_delay > 0.0:
                        time.sleep(args.step_delay)

                # ── Result ────────────────────────────────────────────────────
                n_wp = len(waypoints)
                obj_after, obj_euler_after, obj_linvel_after, obj_angvel_after = _obj_state_local(active_obj_name)
                disp = obj_after[0] - obj_pos_pre[0]
                print(
                    f"         IK {ik_ok}/{n_wp}  "
                    f"disp=({float(disp[0]):+.3f},{float(disp[1]):+.3f})m  "
                    f"obj=({float(obj_after[0,0]):+.3f},{float(obj_after[0,1]):+.3f},{float(obj_after[0,2]):+.3f})  "
                    f"euler=({math.degrees(float(obj_euler_after[0,0])):+.0f},{math.degrees(float(obj_euler_after[0,1])):+.0f},{math.degrees(float(obj_euler_after[0,2])):+.0f})°  "
                    f"vel=({float(obj_linvel_after[0,0]):+.3f},{float(obj_linvel_after[0,1]):+.3f},{float(obj_linvel_after[0,2]):+.3f})m/s",
                    flush=True,
                )

                _pause(_PAUSE_STEPS)

            # ── End of scenario ───────────────────────────────────────────────
            print(f"\n  --- End of scenario --- resetting environment ---\n")
            env.reset()
            _viewer_step()
            # Let object settle on table — hold arm at safe retracted pose
            _pause(80)

    except KeyboardInterrupt:
        print("\n[Loop] Interrupted by user.")

    simulation_app.close()


if __name__ == "__main__":
    main()
