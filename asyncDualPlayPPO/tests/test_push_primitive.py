"""
test_push_primitive.py
======================
Smoke-test for the push macro-action primitive.

Spawns a single environment, places the object at a known local-frame position,
fires one push action with a fixed yaw, and asserts:

  1. Waypoint count is correct — all 8 phases including the Phase-8 return.
  2. cuRobo IK succeeds for ≥ 50 % of waypoints.
  3. Object moved ≥ 2 cm in the intended push direction (+Y).
  4. After the return phase the arm TCP XY is within 15 cm of its pre-push XY
     (verifies Phase 8 actually brings the arm back).

Usage (headless / CI):
    python -m asyncDualPlayPPO.tests.test_push_primitive --headless

Usage (visual, with viewer — recommended for debugging):
    python -m asyncDualPlayPPO.tests.test_push_primitive
    python -m asyncDualPlayPPO.tests.test_push_primitive --step-delay 0.1
"""

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

# ── Push scenario (all values in env-local frame) ─────────────────────────────
_OBJ_LOCAL  = [0.00, 0.50, 0.02]   # object start XYZ (local frame)
_OFFSET_X   =  0.00                # approach laterally centred on object
_OFFSET_Y   = -0.04                # approach ~4 cm behind object center ≈ back face contact
_PUSH_DX    =  0.00                # no X component
_PUSH_DY    =  0.28                # push 28 cm in +Y (near max_push_delta=0.30)
_PUSH_YAW   =  0.00                # gripper yaw = 0
_PUSH_DZ    =  0.00                # no vertical component

# ── Assertion thresholds ──────────────────────────────────────────────────────
_EXPECTED_WP     = 5 + 3 + 3 + 1 + 12 + 1 + 3 + 5   # 33 substeps (all 8 phases)
_IK_OK_MIN       = 0.50
_OBJ_DISP_MIN_M  = 0.05   # object must move ≥ 5 cm in push direction
_PUSH_AXIS       = 1       # Y axis
_RETURN_TOL_M    = 0.15    # TCP XY must return within 15 cm of pre-push XY

# Phase boundaries (cumulative waypoint indices, 0-based start of each phase)
_PHASE_STARTS = {
    "1-Approach": 0,
    "2-Orient":   5,
    "3-Descend":  8,
    "4-Engage":   11,
    "5-Push":     12,
    "6-Release":  24,
    "7-Retract":  25,
    "8-Return":   28,
}
_PHASE_BY_WP = {}
for _ph, _start in _PHASE_STARTS.items():
    _end = _EXPECTED_WP
    for _other_start in sorted(_PHASE_STARTS.values()):
        if _other_start > _start:
            _end = _other_start
            break
    for _i in range(_start, _end):
        _PHASE_BY_WP[_i] = _ph


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"\n  [FAIL] {msg}\n")
        raise AssertionError(msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Push primitive smoke test")
    parser.add_argument(
        "--step-delay", type=float, default=0.0,
        help="Seconds to sleep after each waypoint step (0=fast, 0.05–0.1 for visual debugging)",
    )
    parser.add_argument(
        "--hold-secs", type=float, default=5.0,
        help="Seconds to hold the final pose in the viewer before closing (non-headless only)",
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
    from asyncDualPlayPPO.tasks.utils.action_push import (
        compute_push_waypoints, total_push_substeps, PushConfig,
    )

    failures = []

    def _viewer_step():
        """Pump the app event loop so the viewport refreshes each physics step."""
        if not headless:
            simulation_app.update()

    # ── Environment ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Push Primitive Smoke Test")
    print("=" * 64)
    if not headless and args.step_delay == 0.0:
        print("  Tip: run with --step-delay 0.05 to slow down for visual inspection.")
    print()

    print("[Setup] Creating environment (num_envs=1)...")
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

    env = PushEnvWrapper(
        env=base_env, device=device,
        num_objects=1, max_pushes_per_episode=5,
    )

    # ── cuRobo IK ─────────────────────────────────────────────────────────────
    print("[Setup] Initialising cuRobo IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml   = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg   = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config   = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    ik_solver = IKSolver(_ik_config)

    _wup_pos  = torch.zeros(1, 3, device=device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(1, 1, 6, device=device),
        retract_config=torch.zeros(1, 6, device=device),
    )
    print("[Setup] IK warm-up done.")

    _QUAT_DOWN    = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    _robot_scene  = env.env.scene["robot"]
    _arm_jids, _  = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _wrist3_ids, _ = _robot_scene.find_bodies("wrist_3_link")
    _lf_ids, _    = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _    = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_local() -> torch.Tensor:
        lf = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf + rf) / 2.0 - env.env.scene.env_origins).clone()

    def _tcp_offset() -> torch.Tensor:
        lf = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        w3 = _robot_scene.data.body_pos_w[:, _wrist3_ids[0]]
        return (lf + rf) / 2.0 - w3

    # ── Reset & place object ──────────────────────────────────────────────────
    print("[Setup] Resetting environment and placing object...")
    obs = env.reset()
    _viewer_step()

    env_origin = env.env.scene.env_origins[0]
    obj_local  = torch.tensor(_OBJ_LOCAL, device=device, dtype=torch.float32)
    obj_world  = obj_local + env_origin
    obj_pose   = torch.cat([
        obj_world.unsqueeze(0),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32),
    ], dim=-1)

    try:
        target_obj = env.env.scene["target_object"]
        target_obj.write_root_pose_to_sim(obj_pose)
    except KeyError:
        available = list(env.env.scene.rigid_objects.keys())
        print(f"  [WARN] 'target_object' not found. Available: {available}")
        if available:
            target_obj = env.env.scene[available[0]]
            target_obj.write_root_pose_to_sim(obj_pose)
        else:
            print("  [WARN] No rigid objects — object placement skipped.")
            target_obj = None

    env.env.sim.step()
    obs_dict = env.env.observation_manager.compute()
    obs = env._build_obs(obs_dict)
    env._capture_prev_obj(obs)

    goal_local = obj_local.clone()
    goal_local[_PUSH_AXIS] += _PUSH_DY
    env.goal_pos_euler[0, :3] = goal_local
    env.goal_pos_euler[0, 3:] = 0.0
    env._update_goal_in_extras()
    env._move_goal_ghost(torch.tensor([0], device=device))

    print(f"  Object local:  {_OBJ_LOCAL}")
    print(f"  Goal local:    {goal_local.cpu().tolist()}")
    print(f"  Push:  offset_y={_OFFSET_Y:+.2f}  push_dy={_PUSH_DY:+.2f}  yaw={_PUSH_YAW:.2f} rad")

    # Settle viewer — let rendering initialise before the push starts
    if not headless:
        print("[Setup] Settling viewer (20 hold steps)...")
        hold_cmd = torch.zeros(1, env.action_space.shape[0], device=device)
        hold_cmd[0, :6] = _robot_scene.data.joint_pos[0, _arm_jids]
        for _ in range(20):
            env.step(hold_cmd)
            simulation_app.update()

    # ── CHECK 1 — Waypoint count ──────────────────────────────────────────────
    print("\n[Check 1] Waypoint count (8 phases expected)...")
    ee_pos_before = _tcp_local()
    ee_quat_w     = _QUAT_DOWN.expand(1, 4).clone()
    prev_jcmd     = _robot_scene.data.joint_pos[:, _arm_jids].clone()
    obj_pos_obs   = obs[:, env.robot_dim:env.robot_dim + 3]

    waypoints = compute_push_waypoints(
        offset_x=torch.tensor([_OFFSET_X], device=device),
        offset_y=torch.tensor([_OFFSET_Y], device=device),
        push_dx=torch.tensor([_PUSH_DX],   device=device),
        push_dy=torch.tensor([_PUSH_DY],   device=device),
        yaw=torch.tensor([_PUSH_YAW],      device=device),
        push_dz=torch.tensor([_PUSH_DZ],   device=device),
        obj_pos=obj_pos_obs,
        current_ee_pos=ee_pos_before,
        current_ee_quat=ee_quat_w,
        device=device,
    )

    n_wp     = len(waypoints)
    expected = total_push_substeps(PushConfig())
    print(f"  Generated: {n_wp}  |  Expected: {expected}  "
          f"({'OK' if n_wp == expected else 'MISMATCH'})")

    try:
        _assert(n_wp == expected,
                f"waypoint count {n_wp} ≠ {expected} (PushConfig mismatch)")
        _assert(n_wp == _EXPECTED_WP,
                f"waypoint count {n_wp} ≠ {_EXPECTED_WP} — Phase 8 (return) missing?")
        print(f"  PASS  {n_wp} waypoints — "
              "approach(5)+orient(3)+descend(3)+engage(1)+push(12)+release(1)+retract(3)+return(5)")
    except AssertionError as e:
        failures.append(str(e))

    # ── CHECK 2+3 — Execute push, track IK & object motion ───────────────────
    print("\n[Check 2+3] Executing push trajectory...")
    obj_pos_before = obs[0, env.robot_dim:env.robot_dim + 3].clone()

    ik_total    = 0
    ik_ok_count = 0
    prev_phase  = ""

    for wp_idx, (wp_pos, wp_quat, wp_grip) in enumerate(waypoints):
        # Print phase transition
        phase = _PHASE_BY_WP.get(wp_idx, "?")
        if phase != prev_phase:
            print(f"    Phase {phase}  (wp {wp_idx})")
            prev_phase = phase

        tcp_offs  = _tcp_offset()
        ik_target = wp_pos - tcp_offs

        result = ik_solver.solve_batch(
            CuroboPose(position=ik_target, quaternion=wp_quat),
            seed_config=prev_jcmd.unsqueeze(1),
            retract_config=prev_jcmd,
        )
        ik_ok = result.success.squeeze(-1)
        ik_total += 1
        if ik_ok.any():
            ik_ok_count += 1

        cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
        solved     = result.solution.view(1, 6)
        raw_cmd    = torch.where(ik_ok.unsqueeze(-1), solved, cur_joints)
        prev_jcmd  = raw_cmd.detach().clone()

        env_full        = torch.zeros(1, env.action_space.shape[0], device=device)
        env_full[:, :6] = raw_cmd
        env_full[:, 6]  = wp_grip
        obs, _, _, _, _ = env.step(env_full)

        _viewer_step()

        if args.step_delay > 0.0:
            time.sleep(args.step_delay)

    # Post-push measurements
    obj_pos_after = obs[0, env.robot_dim:env.robot_dim + 3].clone()
    ee_pos_after  = _tcp_local()[0].clone()
    obj_disp      = (obj_pos_after - obj_pos_before).cpu()
    ik_frac       = ik_ok_count / max(1, ik_total)

    print(f"\n  IK:  {ik_ok_count}/{ik_total} succeeded ({ik_frac:.0%})")
    print(f"  Object displacement:  dx={obj_disp[0]:.3f}  dy={obj_disp[1]:.3f}  dz={obj_disp[2]:.3f}")

    try:
        _assert(ik_frac >= _IK_OK_MIN,
                f"IK success rate {ik_frac:.2f} < {_IK_OK_MIN}")
        print(f"  PASS  IK success ≥ {_IK_OK_MIN:.0%}")
    except AssertionError as e:
        failures.append(str(e))

    try:
        _assert(float(obj_disp[_PUSH_AXIS]) >= _OBJ_DISP_MIN_M,
                f"object moved only {float(obj_disp[_PUSH_AXIS]):.3f} m in push axis "
                f"(need ≥ {_OBJ_DISP_MIN_M} m)")
        print(f"  PASS  object moved {float(obj_disp[_PUSH_AXIS]):.3f} m in +Y")
    except AssertionError as e:
        failures.append(str(e))

    # ── CHECK 4 — Arm return (Phase 8) ───────────────────────────────────────
    print("\n[Check 4] Arm return position after Phase 8...")
    before_xy  = ee_pos_before[0, :2].cpu()
    after_xy   = ee_pos_after[:2].cpu()
    return_err = (before_xy - after_xy).norm().item()

    print(f"  Pre-push  TCP XY: ({before_xy[0]:.3f}, {before_xy[1]:.3f})")
    print(f"  Post-push TCP XY: ({after_xy[0]:.3f}, {after_xy[1]:.3f})")
    print(f"  Return error:     {return_err:.3f} m  (need ≤ {_RETURN_TOL_M} m)")
    print(f"  [Without Phase 8 the arm stays above the push endpoint ~25 cm away.]")

    try:
        _assert(return_err <= _RETURN_TOL_M,
                f"arm XY return error {return_err:.3f} m > {_RETURN_TOL_M} m "
                f"— Phase 8 may be missing or IK failed during return")
        print(f"  PASS  arm returned within {_RETURN_TOL_M} m of pre-push XY")
    except AssertionError as e:
        failures.append(str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    if not failures:
        print("  ALL 4 CHECKS PASSED")
        print(f"    1. Waypoints:   {n_wp} (8 phases)")
        print(f"    2. IK success:  {ik_ok_count}/{ik_total} ({ik_frac:.0%})")
        print(f"    3. Object push: {float(obj_disp[_PUSH_AXIS]):.3f} m in +Y")
        print(f"    4. Arm return:  {return_err:.3f} m XY error")
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for i, f in enumerate(failures, 1):
            print(f"    {i}. {f}")
    print("=" * 64)

    # Hold final pose in viewer so you can inspect the result
    if not headless and simulation_app.is_running():
        hold_secs = args.hold_secs
        print(f"\n  Holding final pose for {hold_secs:.0f} s — inspect the viewport.")
        print("  Press Ctrl+C to quit early.\n")
        hold_cmd = torch.zeros(1, env.action_space.shape[0], device=device)
        hold_cmd[0, :6] = _robot_scene.data.joint_pos[0, _arm_jids]
        t0 = time.time()
        try:
            while simulation_app.is_running() and (time.time() - t0) < hold_secs:
                env.step(hold_cmd)
                simulation_app.update()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass

    simulation_app.close()

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
