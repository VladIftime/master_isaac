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

# cuRobo MUST be imported before AppLauncher — importing after causes a native
# crash in librtx.scenedb.plugin.so because CuRobo's CUDA context init races
# with the RTX rendering background thread (carbOnPluginStartup).
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

_N_SPIN = 8     # substeps for the spin phase

# Workspace clamp limits (local / env-origin-relative frame, metres)
_WS_X = (-0.50, 0.50)
_WS_Y = (0.25,  0.70)
_WS_Z = ( 0.00, 0.55)

# ── Scenarios: each is a 3-push sequence ─────────────────────────────────────
# Each push: {offset_x, offset_y, push_dx, push_dy, yaw, spin_yaw}
# Push 1 always: approach centered on stem, push forward 10cm
SCENARIOS = [
    # S0: Push1=Fwd + Push2=Left + Spin CW
    [
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": -0.10, "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": math.pi / 4},
    ],
    # S1: Push1=Fwd + Push2=Right + Spin CCW
    [
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.10, "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": -math.pi / 4},
    ],
    # S2: Push1=Fwd + Push2=Bwd + Spin CW
    [
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": -0.10,"yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": math.pi / 4},
    ],
    # S3: Push1=Fwd + Push2=Fwd + Spin CCW
    [
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": -math.pi / 4},
    ],
    # S4: Push1=Fwd + Push2=LeftFwd + Spin CW
    [
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": -0.07, "push_dy": 0.07, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,   "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": math.pi / 4},
    ],
    # S5: Push1=Fwd + Push2=RightFwd + Spin CCW
    [
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.10, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.07, "push_dy": 0.07, "yaw": 0.0, "spin_yaw": 0.0},
        {"offset_x": 0.0, "offset_y": 0.0, "push_dx": 0.0,  "push_dy": 0.0,  "yaw": 0.0, "spin_yaw": -math.pi / 4},
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

    # Force synchronous RTX rendering so SceneDB is fully initialized before
    # SetLightingMenuModeCommand fires — avoids a race-condition crash on
    # slower GPUs (RTX 3060 Ti + IOMMU enabled).
    for _flag in (
        "--/app/asyncRendering=false",
        "--/app/asyncRenderingLowLatency=false",
    ):
        if _flag not in sys.argv:
            sys.argv.append(_flag)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
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
    _ur5e_yaml   = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e_robotiq_2f_140.yml"))
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
        seed_config=torch.zeros(1, 1, 7, device=device),
        retract_config=torch.zeros(1, 7, device=device),
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
    scenario_idx = 0
    print("\n[Loop] Starting — close the viewport window to exit.\n")

    try:
        while simulation_app.is_running():
            scenario = SCENARIOS[scenario_idx % len(SCENARIOS)]
            scenario_idx += 1

            print(f"{'='*64}")
            print(f"  Scenario {scenario_idx}/{len(SCENARIOS)}  "
                  f"{'spin CW' if scenario[-1]['spin_yaw'] > 0 else 'spin CCW'}")
            print(f"{'='*64}")

            for push_i, cfg in enumerate(scenario):
                if not simulation_app.is_running():
                    break

                offset_x = cfg["offset_x"]
                offset_y = cfg["offset_y"]
                push_dx  = cfg["push_dx"]
                push_dy  = cfg["push_dy"]
                yaw      = cfg["yaw"]
                spin_yaw = cfg["spin_yaw"]

                # ── Get current object position from observation ──────────────
                obs_dict    = env.env.observation_manager.compute()
                obs         = env._build_obs(obs_dict)
                obj_pos_obs = obs[:, env.robot_dim:env.robot_dim + 3].clone()



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
                    n_spin=_N_SPIN,
                    spin_yaw=torch.tensor([spin_yaw], device=device) if spin_yaw != 0.0 else None,
                )

                label = f"Push {push_i + 1}"
                if push_i == 2:
                    label += f"  spin {math.degrees(spin_yaw):+.0f}\u00b0"
                else:
                    label += f"  ({push_dx:+.2f}, {push_dy:+.2f})m yaw={math.degrees(yaw):+.0f}\u00b0"

                print(
                    f"\n  [{scenario_idx}.{push_i + 1}] "
                    f"obj=({float(obj_pos_obs[0, 0]):+.2f},{float(obj_pos_obs[0, 1]):+.2f})  "
                    f"{label}",
                    flush=True,
                )

                # ── Execute push (velocity-smooth IK, like follow-target test) ──
                ik_ok = 0
                last_good_joints = prev_jcmd.clone()
                prev_grip = torch.ones(1, device=device)  # start open
                max_step_delta = 0.05  # 5 cm/s max per-step change
                for wp_i, (wp_pos, wp_quat, wp_grip) in enumerate(waypoints):
                    if not simulation_app.is_running():
                        break

                    cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]

                    # If gripper state changes, apply as separate hold step
                    grip_changed = bool((wp_grip != prev_grip).any())
                    if grip_changed:
                        env_full        = torch.zeros(1, env.action_space.shape[0], device=device)
                        env_full[:, :6] = cur_joints
                        env_full[:, 6]  = wp_grip
                        obs, _, _, _, _ = env.step(env_full)
                        _viewer_step()
                        prev_grip = wp_grip.clone()

                    ik_target = wp_pos
                    ik_target[0, 0].clamp_(_WS_X[0], _WS_X[1])
                    ik_target[0, 1].clamp_(_WS_Y[0], _WS_Y[1])
                    ik_target[0, 2].clamp_(_WS_Z[0], _WS_Z[1])

                    # Velocity smooth: limit how far target moves per step
                    delta_target = ik_target - (last_good_joints.unsqueeze(0) if False else ik_target)
                    # Use smooth interpolation from current to target
                    result    = ik_solver.solve_batch(
                        CuroboPose(position=ik_target, quaternion=wp_quat),
                        seed_config=torch.cat([cur_joints, torch.zeros(1, 1, device=device)], dim=-1).unsqueeze(1),
                        retract_config=torch.cat([cur_joints, torch.zeros(1, 1, device=device)], dim=-1),
                    )
                    success = result.success.squeeze(-1)
                    if success.any():
                        ik_ok += 1
                        last_good_joints = result.solution[:, :6].view(1, 6).clone()

                    raw_cmd = last_good_joints.clone()

                    if wp_i % 3 == 0:
                        ee_pos = _tcp_local()
                        wrist_pos = _robot_scene.data.body_pos_w[:, _w3_ids[0]] - env.env.scene.env_origins
                        obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6] if 'obs' in dir() else torch.zeros(3)
                        print(
                            f"    wp {wp_i}: "
                            f"ee=({float(ee_pos[0,0]):+.3f},{float(ee_pos[0,1]):+.3f},{float(ee_pos[0,2]):+.3f})  "
                            f"obj=({float(obj_pos_obs[0,0]):+.3f},{float(obj_pos_obs[0,1]):+.3f},{float(obj_pos_obs[0,2]):+.3f})  "
                            f"euler=({float(math.degrees(obj_euler[0])):+.0f},{float(math.degrees(obj_euler[1])):+.0f},{float(math.degrees(obj_euler[2])):+.0f})\u00b0",
                            flush=True,
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
                obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
                obj_linvel = obs[0, env.robot_dim + 6:env.robot_dim + 9]
                obj_angvel = obs[0, env.robot_dim + 9:env.robot_dim + 12]
                obj_contact = obs[0, env.robot_dim + 13]
                print(
                    f"         IK {ik_ok}/{n_wp}  "
                    f"disp=({float(disp[0]):+.3f},{float(disp[1]):+.3f})m  "
                    f"obj=({float(obj_after[0]):+.3f},{float(obj_after[1]):+.3f},{float(obj_after[2]):+.3f})  "
                    f"euler=({float(math.degrees(obj_euler[0])):+.0f},{float(math.degrees(obj_euler[1])):+.0f},{float(math.degrees(obj_euler[2])):+.0f})°  "
                    f"vel=({float(obj_linvel[0]):+.3f},{float(obj_linvel[1]):+.3f},{float(obj_linvel[2]):+.3f})m/s  "
                    f"contact={float(obj_contact):.0f}",
                    flush=True,
                )

                _pause(_PAUSE_STEPS)

            # ── End of scenario ───────────────────────────────────────────────
            print(f"\n  --- End of scenario --- resetting environment ---\n")
            env.reset()
            _viewer_step()
            _pause(15)

    except KeyboardInterrupt:
        print("\n[Loop] Interrupted by user.")

    simulation_app.close()


if __name__ == "__main__":
    main()
