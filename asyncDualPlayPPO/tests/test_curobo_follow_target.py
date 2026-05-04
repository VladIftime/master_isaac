"""
CuRobo Interactive Follow-Target Test (IsaacLab)
=================================================

Same environment as test_abc_goal_encoder.py but drives the arm with CuRobo IK
tracking an interactive red sphere instead of an ABC-trained policy.

CuRoboKinematicsSolver wraps IKSolver following the Isaac Sim KinematicsSolver
interface pattern. In CuRobo's ur5e.yml the ee_link="tool0" is co-located with
wrist_3_link (zero position offset). The actual TCP (grasp_convenient_link) is
~0.225 m further along the tool axis, so each step the offset is measured from
live FK and subtracted from the IK target so the gripper centre tracks the ball.

Usage:
    python tests/test_curobo_follow_target.py --max_vel 0.5

Controls:
    1. Click the Red Ball in the viewport.
    2. Press 'W' to activate the translation gizmo.
    3. Drag the ball — the arm follows in real-time.
"""

import argparse
import os
import sys
import torch

# ==============================================================================
# CuRobo MUST be imported before AppLauncher (prevents library conflicts).
# ==============================================================================
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
except ModuleNotFoundError:
    print("\n[ERROR] CuRobo not found in .master_venv. Ensure it is installed.")
    sys.exit(1)
# ==============================================================================

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class CuRoboKinematicsSolver:
    """
    Wraps CuRobo's IKSolver as a kinematic solver (mirrors the Isaac Sim
    ArticulationKinematicsSolver interface).

    In CuRobo's ur5e.yml, ee_link="tool0" shares the same position as
    wrist_3_link (zero XYZ offset). Pass the desired wrist_3_link world
    position minus the robot base position as target_pos.

    Usage:
        solver = CuRoboKinematicsSolver(device="cuda:0")
        solver.reset(seed_joints=reset_joints)
        joint_pos, ok = solver.compute_inverse_kinematics(
            target_pos, target_quat, current_joints
        )
    """

    MAX_JOINT_DELTA = 0.25  # rad per control step (~14 deg); rejects arm flips

    def __init__(self, robot_yaml: str = "ur5e.yml", device: str = "cuda:0", num_seeds: int = 20):
        tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
        ur_yaml = load_yaml(join_path(get_robot_configs_path(), robot_yaml))
        robot_cfg = RobotConfig.from_dict(ur_yaml["robot_cfg"], tensor_args)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg, world_model=None, tensor_args=tensor_args, num_seeds=num_seeds
        )
        self._solver = IKSolver(ik_cfg)
        self._device = device
        self._last_joints: torch.Tensor | None = None

    def compute_inverse_kinematics(
        self,
        target_pos: torch.Tensor,     # (3,) target in robot-base frame
        target_quat: torch.Tensor,    # (4,) (w, x, y, z)
        current_joints: torch.Tensor, # (6,) current arm joint positions
    ):
        """
        Returns:
            joint_positions: (6,) solution tensor, or None if IK failed/flipped
            success: bool
        """
        seed = self._last_joints if self._last_joints is not None else current_joints
        goal = Pose(
            position=target_pos.unsqueeze(0),
            quaternion=target_quat.unsqueeze(0),
        )
        result = self._solver.solve_single(
            goal,
            seed_config=seed.unsqueeze(0).unsqueeze(0),  # (1, 1, 6)
            retract_config=seed.unsqueeze(0),             # (1, 6)
        )
        if not result.success.any():
            return None, False

        # js_solution.position shape: (batch=1, return_seeds=1, dof) → [0,0] for 1D tensor
        candidate = result.js_solution.position[0, 0]  # (dof,)

        # Reject solutions that would cause an instantaneous arm flip.
        if self._last_joints is not None:
            max_delta = torch.max(torch.abs(candidate - self._last_joints)).item()
            if max_delta > self.MAX_JOINT_DELTA:
                return None, False

        self._last_joints = candidate.clone()
        return candidate, True

    def reset(self, seed_joints: torch.Tensor | None = None) -> None:
        self._last_joints = seed_joints.clone() if seed_joints is not None else None


def main():
    parser = argparse.ArgumentParser(description="CuRobo Interactive Follow Target")
    parser.add_argument("--max_vel", type=float, default=0.5, help="Max EE velocity in m/s")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── Post-launch imports ────────────────────────────────────────────────────
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import UsdGeom, Usd, Gf

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # ── Environment ────────────────────────────────────────────────────────────
    print("\nCreating Environment...")
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = 1

    # Override arm action: raw absolute joint angles, no offset.
    # JointPositionActionCfg + use_default_offset=False lets env.step() drive
    # the ImplicitActuator PD controller with the angles CuRobo outputs directly.
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )
    # Disable all terminations — we never want the env to auto-reset.
    env_cfg.terminations.robot_through_table = None
    env_cfg.terminations.objects_off_table = None
    # Remove the manipulation objects so only the robot and table remain.
    try:
        env_cfg.scene.cube = None
    except AttributeError:
        pass
    try:
        env_cfg.scene.target_object = None
    except AttributeError:
        pass

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # ── CuRobo kinematic solver ────────────────────────────────────────────────
    print("\nInitializing CuRoboKinematicsSolver...")
    solver = CuRoboKinematicsSolver(device=device)

    # ── Reset and warm-up ──────────────────────────────────────────────────────
    env.reset()
    robot = env.scene["robot"]

    wrist3_ids, _ = robot.find_bodies("wrist_3_link")
    wrist3_id = wrist3_ids[0]

    # preserve_order=True ensures indices align 1-to-1 with ARM_JOINT_NAMES.
    joint_indices, found_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    print(f"  Joint index mapping: {list(zip(found_names, joint_indices))}")

    reset_joints = robot.data.joint_pos[:, joint_indices].clone()  # (1, 6)
    solver.reset(seed_joints=reset_joints[0])
    print(f"  Init joints: {reset_joints[0].cpu().numpy().round(3)}")

    # Hold reset pose for 10 steps so PhysX transforms settle.
    hold_action = torch.zeros((1, env.action_space.shape[-1]), device=device)
    hold_action[0, :6] = reset_joints[0]
    for _ in range(10):
        env.step(hold_action)
    robot.update(env.step_dt)

    # ── Find TCP (grasp_convenient_link) body ─────────────────────────────────
    # grasp_convenient_link is 0.225 m above robotiq_arg2f_base_link (Z-axis) —
    # the actual grasp centre between the fingers.  CuRobo's tool0 sits at the
    # same position as wrist_3_link (zero offset in CuRobo's ur5e URDF), so we
    # compensate: ik_target = sphere_pos_local - (tcp_pos_w - wrist3_pos_w).
    try:
        tcp_ids, _ = robot.find_bodies(["grasp_convenient_link"])
        tcp_id = tcp_ids[0]
        print(f"  TCP body: grasp_convenient_link (id={tcp_id})")
    except Exception:
        tcp_ids, _ = robot.find_bodies(["robotiq_arg2f_base_link"])
        tcp_id = tcp_ids[0]
        print(f"  TCP body: robotiq_arg2f_base_link (id={tcp_id}) [fallback]")

    # ── Spawn red target sphere at TCP position ────────────────────────────────
    tcp_pos_w = robot.data.body_pos_w[0, tcp_id].cpu().numpy()

    target_path = "/World/interactive_target"
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.05,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    sphere_cfg.func(target_path, sphere_cfg, translation=tcp_pos_w)

    import omni.kit.app

    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath(target_path)

    # Workspace bounds for the draggable ball (world frame, matches scene border cuboids).
    _WS_X_MIN, _WS_X_MAX = -0.75, 0.75
    _WS_Y_MIN, _WS_Y_MAX =  0.20, 1.00
    _WS_Z_MIN, _WS_Z_MAX =  0.02, 0.70

    _session_layer = stage.GetSessionLayer()
    _xform_api = UsdGeom.XformCommonAPI(target_prim)

    # Shared mutable slot so the update-subscription and main loop share one value.
    _ball_pos = list(tcp_pos_w)   # [x, y, z], updated every frame by subscription

    def _enforce_bounds(_event):
        """Kit pre-render callback: fires every frame inside simulation_app.update(),
        after the gizmo writes its drag position but before the scene is rendered.
        We write the clamped value back to the session layer so the render and the
        IK loop both see a position that is always inside the workspace."""
        t, *_ = _xform_api.GetXformVectors(Usd.TimeCode.Default())
        x = float(max(_WS_X_MIN, min(_WS_X_MAX, t[0])))
        y = float(max(_WS_Y_MIN, min(_WS_Y_MAX, t[1])))
        z = float(max(_WS_Z_MIN, min(_WS_Z_MAX, t[2])))
        _ball_pos[0], _ball_pos[1], _ball_pos[2] = x, y, z
        if x != t[0] or y != t[1] or z != t[2]:
            with Usd.EditContext(stage, _session_layer):
                _xform_api.SetTranslate(Gf.Vec3d(x, y, z))

    _bounds_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(  # noqa: F841
        _enforce_bounds, name="ball_workspace_clamp", order=-100
    )

    def _read_ball_pos() -> torch.Tensor:
        return torch.tensor(_ball_pos, device=device, dtype=torch.float32)

    # ── Control loop state ─────────────────────────────────────────────────────
    robot_base_pos_w = robot.data.root_pos_w[0].clone()
    # Hold orientation fixed at the reset pose wrist_3_link orientation.
    target_quat = robot.data.body_quat_w[0, wrist3_id].clone()  # (w, x, y, z)

    current_ik_target_w = torch.tensor(tcp_pos_w, device=device, dtype=torch.float32)
    max_step_delta = args.max_vel * env.step_dt
    MAX_REACH = 0.78  # conservative UR5e workspace radius in metres
    MIN_Z = 0.02      # keep above table

    action = torch.zeros((1, env.action_space.shape[-1]), device=device)
    action[0, :6] = reset_joints[0]

    print("\n" + "=" * 70)
    print("  [Interactive Mode Ready]")
    print("  1. Click the Red Ball in the viewport.")
    print("  2. Press 'W' to activate the translation gizmo.")
    print("  3. Drag the ball — the arm follows in real-time.")
    print(f"  Max velocity : {args.max_vel} m/s")
    print(f"  Robot base   : {robot_base_pos_w.cpu().numpy().round(3)}")
    print(f"  Ball spawn   : {tcp_pos_w.round(3)} (grasp_convenient_link / TCP)")
    print("=" * 70 + "\n")

    step_count = 0
    ik_ok_count = 0

    try:
        while simulation_app.is_running():
            # Flush USD gizmo writes so _clamp_and_read_ball() sees the latest position.
            simulation_app.update()

            ball_w = _read_ball_pos()

            # Velocity-clamp: advance IK target toward ball at max_vel m/s.
            delta = ball_w - current_ik_target_w
            dist = delta.norm()
            if dist > max_step_delta:
                current_ik_target_w = current_ik_target_w + (delta / dist) * max_step_delta
            else:
                current_ik_target_w = ball_w.clone()

            # Workspace clamp (robot-base frame).
            local_target = current_ik_target_w - robot_base_pos_w
            reach = local_target.norm()
            if reach > MAX_REACH:
                local_target = local_target * (MAX_REACH / reach)
                current_ik_target_w = local_target + robot_base_pos_w
            if local_target[2] < MIN_Z:
                local_target[2] = MIN_Z
                current_ik_target_w[2] = robot_base_pos_w[2] + MIN_Z

            # TCP offset (world frame): how much tcp overshoots wrist_3_link.
            # CuRobo places tool0 (= wrist_3_link) at the IK target, so subtract
            # the offset so that TCP ends up at local_target.
            tcp_pos_cur = robot.data.body_pos_w[0, tcp_id]
            wrist3_pos_cur = robot.data.body_pos_w[0, wrist3_id]
            tcp_offset_w = tcp_pos_cur - wrist3_pos_cur  # world frame, changes with wrist rotation
            tcp_offset_local = tcp_offset_w  # robot base at world origin, so frames align

            # Solve IK — target_pos is in robot-base frame, adjusted for TCP offset.
            cur_joints = robot.data.joint_pos[:, joint_indices][0]
            ik_local_target = local_target - tcp_offset_local
            sol, ok = solver.compute_inverse_kinematics(ik_local_target, target_quat, cur_joints)
            if ok:
                ik_ok_count += 1
                action[0, :6] = sol

            env.step(action)
            step_count += 1

            if step_count % 60 == 0:
                tcp_local = tcp_pos_cur - robot_base_pos_w
                err = (tcp_local - local_target).norm().item()
                print(
                    f"[{step_count:5d}] "
                    f"target=({local_target[0]:.3f},{local_target[1]:.3f},{local_target[2]:.3f}) "
                    f"tcp=({tcp_local[0]:.3f},{tcp_local[1]:.3f},{tcp_local[2]:.3f}) "
                    f"err={err:.4f}m  IK={'OK  ' if ok else 'FAIL'} "
                    f"({ik_ok_count}/{step_count})"
                )

    except KeyboardInterrupt:
        print("\nExiting.")

    simulation_app.close()


if __name__ == "__main__":
    main()
