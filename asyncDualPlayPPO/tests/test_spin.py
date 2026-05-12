"""
Quick test: can the push primitive spin objects?

Tests large-offset pushes to verify torque induction for yaw rotation.
Run: source ~/IsaacLab/master_isaac/.master_venv/bin/activate
      python -m asyncDualPlayPPO.tests.test_spin --headless
"""
import os
import sys
import math
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print("[ERROR] cuRobo not found.")
    sys.exit(1)

from isaaclab.app import AppLauncher

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25,  0.70)
_WS_Z = ( 0.00, 0.55)


# ── Spin-focused scenarios: large offset + tangential push = max torque ────
# Torque = offset_x * push_dy - offset_y * push_dx
SPIN_SCENARIOS = [
    # S0: offset far right, push up → CCW spin
    {"label": "offset right 0.15, push up 0.25 → CCW spin (torque ~0.0375)",
     "offset_x": 0.15, "offset_y": 0.0, "push_dx": 0.0, "push_dy": 0.25},
    # S1: offset far up, push left → CW spin
    {"label": "offset up 0.15, push left 0.25 → CW spin (torque ~0.0375)",
     "offset_x": 0.0, "offset_y": 0.15, "push_dx": -0.25, "push_dy": 0.0},
    # S2: offset diagonal, push perpendicular → spin
    {"label": "offset (0.1,0.1), push (-0.15,0.15) → spin (torque ~0.03)",
     "offset_x": 0.10, "offset_y": 0.10, "push_dx": -0.15, "push_dy": 0.15},
    # S3: pure forward offset → no torque (control test)
    {"label": "offset y=0.15, push y=0.25 → NO spin (torque=0)",
     "offset_x": 0.0, "offset_y": 0.15, "push_dx": 0.0, "push_dy": 0.25},
    # S4: offset far left, push up → CW spin
    {"label": "offset left 0.15, push up 0.25 → CW spin (torque ~0.0375)",
     "offset_x": -0.15, "offset_y": 0.0, "push_dx": 0.0, "push_dy": 0.25},
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints

    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushEnvWrapper(env=base_env, device=device, num_objects=1, max_pushes_per_episode=5)

    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(_robot_cfg, world_model=None, tensor_args=_tensor_args)
    ik_solver = IKSolver(_ik_config)
    ik_solver.solve_batch(
        CuroboPose(
            position=torch.zeros(1, 3, device=device),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device),
        ),
        seed_config=torch.zeros(1, 1, 6, device=device),
        retract_config=torch.zeros(1, 6, device=device),
    )

    _QUAT_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")
    _w3_ids, _ = _robot_scene.find_bodies("wrist_3_link")

    def _tcp_local():
        lf = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return (lf + rf) / 2.0 - env.env.scene.env_origins

    def _tcp_offset():
        lf = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        w3 = _robot_scene.data.body_pos_w[:, _w3_ids[0]]
        return (lf + rf) / 2.0 - w3

    env.reset()

    # Calibrate TCP offset
    _calib_pos = torch.tensor([[0.0, 0.50, 0.25]], device=device)
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_DOWN),
        seed_config=_calib_cur.unsqueeze(1), retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(1, 6)
    _calib_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
    _FIXED_TCP_OFFSET = _tcp_offset().clone()

    # ── Test each scenario ─────────────────────────────────────────────────
    for si, sc in enumerate(SPIN_SCENARIOS):
        print(f"\n{'='*70}")
        print(f"  Spin Test {si}: {sc['label']}")
        print(f"{'='*70}")

        # Reset env for clean state
        env.reset()

        # Get object pose before push
        obs_dict = env.env.observation_manager.compute()
        obs = env._build_obs(obs_dict)
        obj_before = obs[0, env.robot_dim:env.robot_dim + 6].clone()
        yaw_before_deg = math.degrees(float(obj_before[5]))

        # Compute torque prediction
        torque = sc["offset_x"] * sc["push_dy"] - sc["offset_y"] * sc["push_dx"]
        print(f"  Predicted yaw torque: {torque:+.4f}  (offset × push)")

        # Execute push
        prev_jcmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
        current_ee = _tcp_local()

        waypoints = compute_push_waypoints(
            offset_x=torch.tensor([sc["offset_x"]], device=device),
            offset_y=torch.tensor([sc["offset_y"]], device=device),
            push_dx=torch.tensor([sc["push_dx"]], device=device),
            push_dy=torch.tensor([sc["push_dy"]], device=device),
            push_dz=torch.tensor([0.0], device=device),
            obj_pos=obs[:, env.robot_dim:env.robot_dim + 3],
            current_ee_pos=current_ee,
            current_ee_quat=_QUAT_DOWN.expand(1, 4).clone(),
            device=device,
        )

        last_good_joints = prev_jcmd.clone()
        prev_grip = torch.ones(1, device=device)
        max_angvel_seen = 0.0
        ik_ok = 0

        for wp_i, (wp_pos, wp_quat, wp_grip) in enumerate(waypoints):
            cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
            if (wp_grip != prev_grip).any():
                grip_hold = torch.zeros(1, env.action_space.shape[0], device=device)
                grip_hold[:, :6] = cur_joints
                grip_hold[:, 6] = wp_grip
                obs, _, _, _, _ = env.step(grip_hold)
                prev_grip = wp_grip.clone()

            ik_target = wp_pos - _FIXED_TCP_OFFSET
            ik_target[0, 0].clamp_(_WS_X[0], _WS_X[1])
            ik_target[0, 1].clamp_(_WS_Y[0], _WS_Y[1])
            ik_target[0, 2].clamp_(_WS_Z[0], _WS_Z[1])

            result = ik_solver.solve_batch(
                CuroboPose(position=ik_target, quaternion=wp_quat),
                seed_config=cur_joints.unsqueeze(1),
                retract_config=cur_joints,
            )
            success = result.success.squeeze(-1)
            if success.any():
                ik_ok += 1
                last_good_joints = result.solution.view(1, 6).clone()

            raw_cmd = last_good_joints.clone()
            prev_jcmd = raw_cmd.detach().clone()

            env_full = torch.zeros(1, env.action_space.shape[0], device=device)
            env_full[:, :6] = raw_cmd
            env_full[:, 6] = wp_grip
            obs, _, _, _, _ = env.step(env_full)

            angvel = float(obs[0, env.robot_dim + 9:env.robot_dim + 12].norm().item())
            if angvel > max_angvel_seen:
                max_angvel_seen = angvel

        # Result
        obj_after = obs[0, env.robot_dim:env.robot_dim + 6]
        disp = obj_after[:3] - obj_before[:3]
        yaw_change = float(obj_after[5] - obj_before[5])

        # Normalize yaw change to [-π, π]
        while yaw_change > math.pi:
            yaw_change -= 2 * math.pi
        while yaw_change < -math.pi:
            yaw_change += 2 * math.pi

        obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
        print(
            f"  IK: {ik_ok}/{len(waypoints)}  "
            f"disp=({float(disp[0]):+.3f},{float(disp[1]):+.3f},{float(disp[2]):+.3f})m  "
            f"yaw_change={math.degrees(yaw_change):+.1f}°  "
            f"final_euler=({math.degrees(float(obj_euler[0])):+.0f}°,"
            f"{math.degrees(float(obj_euler[1])):+.0f}°,"
            f"{math.degrees(float(obj_euler[2])):+.0f}°)  "
            f"max_angvel={max_angvel_seen:.3f}rad/s"
        )

    print("\nDone.")
    simulation_app.close()


if __name__ == "__main__":
    main()
