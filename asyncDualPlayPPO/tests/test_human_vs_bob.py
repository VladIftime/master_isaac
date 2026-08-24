"""
Human vs Bob — Side-by-Side Evaluation
=======================================

Two identical arenas share the same randomly generated goal.

  Arena 0 (LEFT)  — human-controlled via gamepad + cuRobo IK
  Arena 1 (RIGHT) — loaded Bob checkpoint + cuRobo IK

Goal generation: Alice's checkpoint runs on arena 0 for alice_timesteps steps.
The final object configuration becomes the shared goal for both arenas.
If no Alice checkpoint is supplied, a random object placement is used instead.

Usage:
    python tests/test_human_vs_bob.py \\
        --chkpt_bob  runs/my_run/bob/model_500.pt \\
        --chkpt_alice runs/my_run/alice/model_500.pt   # optional
        --num_objects 2
        --max_vel 1.0

Controls (gamepad, arena 0):
    Left  stick  ↑↓←→ → EE +X/-X, +Y/-Y
    Right stick  ↑↓   → EE +Z/-Z
    X button          → toggle gripper
    R key             → regenerate goal and reset both arenas
"""

import argparse
import os
import sys
import time
import torch

# cuRobo MUST be imported before AppLauncher.
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
except ModuleNotFoundError:
    print("\n[ERROR] cuRobo not found. Ensure it is installed in the active venv.")
    sys.exit(1)

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Bob's observation layout (51D for 2 objects, 29D for 1 object):
#   robot_state(7) | [obj_state(14) | obj_goal(6) | obj_dist(2)] × num_objects
_OBS_DIM_1OBJ = 7 + (14 + 6 + 2)      # 29
_OBS_DIM_2OBJ = 7 + 2 * (14 + 6 + 2)  # 51

# Success thresholds (match training wrapper)
_POS_THRESH = 0.05   # metres
_ROT_THRESH = 0.20   # radians (~11°)


# ---------------------------------------------------------------------------
# Observation helpers (mirror observations.py without importing it)
# ---------------------------------------------------------------------------

def _quat_to_euler(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz → (N,3) roll/pitch/yaw."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = torch.asin((2*(w*y - z*x)).clamp(-1, 1))
    yaw   = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _build_bob_obs(
    robot, env, env_origins,
    goal_states: torch.Tensor,        # (2, 12) for 2 objects
    left_finger_ids, right_finger_ids,
    gripper_joint_ids,
    arm_joint_ids,
    target_obj, cube_obj,
    num_objects: int,
    device: str,
) -> torch.Tensor:
    """
    Builds Bob's 51D (or 29D for 1 object) observation for both envs.

    Returns (2, obs_dim) tensor.
    """
    N = 2  # always 2 envs in this script

    # --- Robot state (7D) ---
    # EE pose from wrist_3_link
    wrist_ids, _ = robot.find_bodies("wrist_3_link")
    ee_pos_w   = robot.data.body_pos_w[:, wrist_ids[0]]           # (2,3)
    ee_quat_w  = robot.data.body_quat_w[:, wrist_ids[0]]          # (2,4)
    ee_pos_loc = ee_pos_w - env_origins                            # (2,3) local
    ee_euler   = _quat_to_euler(ee_quat_w)                         # (2,3)
    finger_pos = robot.data.joint_pos[:, gripper_joint_ids[:1]]   # (2,1)
    robot_state = torch.cat([ee_pos_loc, ee_euler, finger_pos], dim=-1)  # (2,7)

    chunks = [robot_state]

    for obj_idx in range(num_objects):
        obj = target_obj if obj_idx == 0 else cube_obj

        # Object state (14D)
        obj_pos_w  = obj.data.root_pos_w    # (2,3)
        obj_quat_w = obj.data.root_quat_w   # (2,4)
        obj_lv     = obj.data.root_lin_vel_w.clamp(-5, 5)   # (2,3)
        obj_av     = obj.data.root_ang_vel_w.clamp(-5, 5)   # (2,3)
        obj_pos_loc = obj_pos_w - env_origins                # (2,3)
        obj_euler   = _quat_to_euler(obj_quat_w)             # (2,3)

        # Distance to gripper midpoint
        lf_pos = robot.data.body_pos_w[:, left_finger_ids[0]]
        rf_pos = robot.data.body_pos_w[:, right_finger_ids[0]]
        grip_pos = ((lf_pos + rf_pos) / 2.0) - env_origins
        dist_to_grip = (obj_pos_loc - grip_pos).norm(dim=-1, keepdim=True)   # (2,1)
        contact = torch.zeros(N, 1, device=device)

        obj_state = torch.cat([
            obj_pos_loc, obj_euler, obj_lv, obj_av, dist_to_grip, contact
        ], dim=-1)  # (2,14)

        # Goal state (6D)
        g_start = obj_idx * 6
        obj_goal = goal_states[:, g_start:g_start + 6]  # (2,6)

        # Goal distance (2D)
        pos_dist = (obj_pos_loc - obj_goal[:, :3]).norm(dim=-1, keepdim=True)  # (2,1)
        euler_diff = obj_euler - obj_goal[:, 3:6]
        # Wrap Euler diff to [-π, π]
        euler_diff = (euler_diff + torch.pi) % (2 * torch.pi) - torch.pi
        rot_dist = euler_diff.abs().max(dim=-1, keepdim=True).values  # (2,1)

        chunks.extend([obj_state, obj_goal, pos_dist, rot_dist])

    return torch.cat(chunks, dim=-1)  # (2, obs_dim)


def _check_success(
    env, env_origins,
    goal_states: torch.Tensor,  # (2,12)
    target_obj, cube_obj,
    num_objects: int,
) -> torch.Tensor:
    """Returns (2,) bool tensor: True if ALL objects within threshold."""
    N = 2
    all_ok = torch.ones(N, dtype=torch.bool, device=goal_states.device)

    for obj_idx in range(num_objects):
        obj = target_obj if obj_idx == 0 else cube_obj
        obj_pos_loc = obj.data.root_pos_w - env_origins
        g_start = obj_idx * 6
        goal_pos  = goal_states[:, g_start:g_start + 3]
        goal_eul  = goal_states[:, g_start + 3:g_start + 6]
        obj_quat  = obj.data.root_quat_w
        obj_eul   = _quat_to_euler(obj_quat)

        pos_ok = (obj_pos_loc - goal_pos).norm(dim=-1) < _POS_THRESH
        eul_diff = (obj_eul - goal_eul + torch.pi) % (2 * torch.pi) - torch.pi
        rot_ok = eul_diff.abs().max(dim=-1).values < _ROT_THRESH

        all_ok = all_ok & pos_ok & rot_ok

    return all_ok


# ---------------------------------------------------------------------------
# Goal visualisation helpers (ghost semi-transparent objects via USD)
# ---------------------------------------------------------------------------

def _spawn_goal_ghost(stage, name: str, pos_world, color=(0.2, 0.8, 0.2)):
    """Spawn a small semi-transparent cube at pos_world as a goal marker."""
    from pxr import UsdGeom, UsdShade, Gf, Sdf
    prim_path = f"/World/GoalMarker_{name}"
    # Remove previous if it exists
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(0.06)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(
        Gf.Vec3d(float(pos_world[0]), float(pos_world[1]), float(pos_world[2]))
    )
    mat    = UsdShade.Material.Define(stage, prim_path + "/Mat")
    shader = UsdShade.Shader.Define(stage,   prim_path + "/Mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)
    return prim_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Human vs Bob evaluation")
    parser.add_argument("--chkpt_bob",   required=True,  help="Path to Bob checkpoint .pt")
    parser.add_argument("--chkpt_alice", default=None,   help="Path to Alice checkpoint for goal gen (optional)")
    parser.add_argument("--num_objects", type=int, default=2, choices=[1, 2])
    parser.add_argument("--max_vel",     type=float, default=1.0, help="Human EE max velocity m/s")
    parser.add_argument("--alice_steps", type=int, default=100,   help="Steps to run Alice for goal gen")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── post-launch imports ────────────────────────────────────────────────
    import isaaclab.sim as sim_utils
    import isaaclab.envs.mdp as mdp
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.devices import Se3Gamepad, Se3GamepadCfg
    import omni.usd
    import carb
    import numpy as np
    from pxr import UsdGeom, Gf, Usd

    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic
    from asyncDualPlayPPO.tasks.utils.observations import _quat_to_euler_xyz

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── environment (num_envs=2: left=human, right=Bob) ───────────────────
    print("\n[Setup] Creating environment (num_envs=2)…")
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = 2

    # Use JointPosition actions (like DiffIK / cuRobo test script)
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )

    # Disable observation groups that depend on the wrapper's goal store —
    # we manage goals manually and build Bob's obs ourselves.
    for _attr in ("goal_state", "goal_distance", "cube_goal_state", "cube_goal_distance"):
        for _grp in (env_cfg.observations.alice_policy, env_cfg.observations.bob_policy):
            try:
                setattr(_grp, _attr, None)
            except AttributeError:
                pass

    if args.num_objects == 1:
        env_cfg.scene.cube = None
        env_cfg.observations.alice_policy.cube_state = None
        env_cfg.observations.bob_policy.cube_state   = None

    # Disable terminations so episodes don't interrupt the comparison.
    env_cfg.terminations.robot_through_table = None
    env_cfg.terminations.objects_off_table   = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    env_origins = env.scene.env_origins  # (2,3) world positions of each arena

    robot      = env.scene["robot"]
    target_obj = env.scene["target_object"]
    cube_obj   = env.scene["cube"] if args.num_objects == 2 else None

    # Find joint / body indices once
    joint_indices,    _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    gripper_joint_ids,_ = robot.find_joints(["finger_joint"], preserve_order=True)
    left_finger_ids,  _ = robot.find_bodies("left_inner_finger")
    right_finger_ids, _ = robot.find_bodies("right_inner_finger")
    wrist_ids,        _ = robot.find_bodies("wrist_3_link")

    reset_joints = robot.data.joint_pos[:, joint_indices].clone()  # (2,6)

    # ── cuRobo IK solver (shared for both arenas) ──────────────────────────
    print("[Setup] Initialising cuRobo IKSolver…")
    ur5e_yaml  = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    robot_cfg  = RobotConfig.from_dict(ur5e_yaml["robot_cfg"], tensor_args)
    ik_config  = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_model=None, tensor_args=tensor_args
    )
    ik_solver  = IKSolver(ik_config)

    # Warm-up CUDA graph
    _warmup_pos  = torch.zeros(2, 3, device=device)
    _warmup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).expand(2, 4)
    ik_solver.solve_batch(
        Pose(position=_warmup_pos, quaternion=_warmup_quat),
        seed_config=torch.zeros(2, 1, 6, device=device),
        retract_config=torch.zeros(2, 6, device=device),
    )
    print("[Setup] cuRobo warm-up done.")

    # Fixed "tool down" orientation for IK (matches test_curobo_follow_target.py)
    fixed_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).expand(2, 4)

    # ── load Bob checkpoint ────────────────────────────────────────────────
    print(f"[Setup] Loading Bob checkpoint: {args.chkpt_bob}")
    import yaml
    _cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "cfg", "ppo", "ppo_continuous.yaml"
    )
    with open(_cfg_path) as _f:
        _ppo_cfg = yaml.safe_load(_f)
    _model_cfg = _ppo_cfg["params"]["policy"]
    _model_cfg["use_goal_encoder"] = True
    _model_cfg["num_objects"] = args.num_objects

    obs_dim = _OBS_DIM_2OBJ if args.num_objects == 2 else _OBS_DIM_1OBJ
    num_cat_dims = _model_cfg.get("num_cat_dims", 4)
    num_bins     = _model_cfg.get("num_bins", 11)

    bob_ac = ActorCritic(
        obs_shape=(obs_dim,),
        states_shape=(obs_dim,),
        actions_shape=(num_cat_dims,),
        initial_std=0.3,
        model_cfg=_model_cfg,
        asymmetric=False,
    ).to(device)
    bob_ac.load_state_dict(torch.load(args.chkpt_bob, map_location=device), strict=False)
    bob_ac.eval()
    print(f"[Setup] Bob loaded ({obs_dim}D obs, {num_cat_dims}D action, {num_bins} bins).")

    # ── load Alice checkpoint for goal generation (optional) ──────────────
    alice_ac = None
    if args.chkpt_alice:
        print(f"[Setup] Loading Alice checkpoint: {args.chkpt_alice}")
        _alice_model_cfg = dict(_ppo_cfg["params"]["policy"])
        _alice_model_cfg["use_goal_encoder"] = False
        alice_obs_dim = 7 + args.num_objects * 14  # 35D (2obj) or 21D (1obj)
        alice_ac = ActorCritic(
            obs_shape=(alice_obs_dim,),
            states_shape=(alice_obs_dim,),
            actions_shape=(num_cat_dims,),
            initial_std=0.3,
            model_cfg=_alice_model_cfg,
            asymmetric=False,
        ).to(device)
        alice_ac.load_state_dict(
            torch.load(args.chkpt_alice, map_location=device), strict=False
        )
        alice_ac.eval()
        print("[Setup] Alice loaded for goal generation.")

    # ── gamepad ────────────────────────────────────────────────────────────
    gamepad = Se3Gamepad(Se3GamepadCfg(
        pos_sensitivity=0.02, rot_sensitivity=0.0, gripper_term=True
    ))
    gamepad.reset()

    # ── USD stage for goal markers and cameras ─────────────────────────────
    stage = omni.usd.get_context().get_stage()
    import omni.kit.viewport.utility as vp_util

    def _add_camera(path, eye, target, focal=22.0):
        from pxr import UsdGeom, Gf
        cam = UsdGeom.Camera.Define(stage, path)
        cam.GetFocalLengthAttr().Set(focal)
        e, t = Gf.Vec3d(*eye), Gf.Vec3d(*target)
        fwd = (t - e).GetNormalized()
        up_hint = Gf.Vec3d(0, 1, 0) if abs(fwd[2]) > 0.9 else Gf.Vec3d(0, 0, 1)
        right = Gf.Cross(fwd, up_hint).GetNormalized()
        up    = Gf.Cross(right, fwd)
        nfwd  = -fwd
        UsdGeom.Xformable(cam).MakeMatrixXform().Set(Gf.Matrix4d(
            right[0], right[1], right[2], 0,
            up[0],    up[1],    up[2],    0,
            nfwd[0],  nfwd[1],  nfwd[2],  0,
            e[0],     e[1],     e[2],     1,
        ))
        return path

    # Per-arena top-down cameras
    for i, origin in enumerate(env_origins):
        ox, oy = origin[0].item(), origin[1].item()
        _add_camera(f"/World/CamTop{i}",
                    eye=(ox, oy + 0.47, 1.3), target=(ox, oy + 0.47, 0.0), focal=18.0)
        _vp = vp_util.create_viewport_window(
            f"Arena {i} ({'Human' if i == 0 else 'Bob'}) — Top",
            width=420, height=280, position_x=0, position_y=i * 290,
        )
        _vp.viewport_api.camera_path = f"/World/CamTop{i}"

    # ── keyboard (R = regenerate goal) ────────────────────────────────────
    import omni.appwindow
    _regen = [False]

    def _on_key(event, *_a, **_kw):
        if (event.type == carb.input.KeyboardEventType.KEY_PRESS and
                event.input == carb.input.KeyboardInput.R):
            _regen[0] = True
        return True

    _appwindow  = omni.appwindow.get_default_app_window()
    _input_iface = carb.input.acquire_input_interface()
    _input_iface.subscribe_to_keyboard_events(_appwindow.get_keyboard(), _on_key)  # noqa: F841

    # ── IK + action helpers ────────────────────────────────────────────────
    max_step_delta = args.max_vel * env.step_dt

    def _ik_batch(
        ee_targets_local: torch.Tensor,   # (2,3) local-frame EE targets
        last_good_joints: torch.Tensor,   # (2,6) fallback
    ) -> torch.Tensor:
        """Solve IK for both envs simultaneously. Returns (2,6) joint targets."""
        # Convert local target to world frame (cuRobo works in robot base frame)
        ee_world = ee_targets_local + env_origins

        # TCP offset: cuRobo targets wrist_3_link; compensate for wrist→finger-mid offset
        lf_w  = robot.data.body_pos_w[:, left_finger_ids[0]]
        rf_w  = robot.data.body_pos_w[:, right_finger_ids[0]]
        w3_w  = robot.data.body_pos_w[:, wrist_ids[0]]
        tcp_offset = (lf_w + rf_w) / 2.0 - w3_w       # (2,3)
        ik_target_local = ee_targets_local - tcp_offset  # (2,3)

        cur_joints = robot.data.joint_pos[:, joint_indices]  # (2,6)
        result = ik_solver.solve_batch(
            Pose(position=ik_target_local, quaternion=fixed_quat),
            seed_config=cur_joints.unsqueeze(1),    # (2,1,6)
            retract_config=cur_joints,               # (2,6)
        )
        solved = result.solution.view(2, 6)
        success = result.success.squeeze(-1)         # (2,) bool
        return torch.where(success.unsqueeze(-1), solved, last_good_joints)

    def _decode_bins(bin_indices: torch.Tensor) -> torch.Tensor:
        """(N, num_cat_dims) int → (N, 3) EE XYZ delta (orientation fixed)."""
        center = (num_bins - 1) / 2.0
        norm   = (bin_indices.float() - center) / center         # [-1,1]
        xyz    = norm[:, :3] * bob_ac.max_delta if hasattr(bob_ac, "max_delta") else norm[:, :3] * 0.05
        return xyz  # only XYZ; orientation is fixed to "down"

    # ── goal state storage ─────────────────────────────────────────────────
    # goal_states: (2, num_objects*6)  —  [pos(3)+euler(3)] per object, local frame
    goal_states   = torch.zeros(2, args.num_objects * 6, device=device)
    goal_markers  = []  # list of USD prim paths for current goal markers
    goal_ready    = [False]

    def _place_goal_markers():
        """Spawn translucent green cubes at each goal position (both arenas)."""
        nonlocal goal_markers
        for p in goal_markers:
            if stage.GetPrimAtPath(p).IsValid():
                stage.RemovePrim(p)
        goal_markers.clear()
        colors = [(0.2, 0.8, 0.2), (0.2, 0.5, 1.0)]  # green=target_obj, blue=cube
        for obj_idx in range(args.num_objects):
            g_start = obj_idx * 6
            for arena in range(2):
                pos_local = goal_states[arena, g_start:g_start + 3]
                pos_world = pos_local + env_origins[arena]
                p = _spawn_goal_ghost(
                    stage,
                    f"obj{obj_idx}_arena{arena}",
                    pos_world.cpu().numpy(),
                    color=colors[obj_idx],
                )
                goal_markers.append(p)

    def _generate_goal():
        """
        Run Alice (or random actions) on arena 0 for alice_steps to produce a goal.
        The same goal is assigned to both arenas.
        """
        print(f"\n[Goal] Generating goal using "
              f"{'Alice checkpoint' if alice_ac else 'random walk'} "
              f"({args.alice_steps} steps)…")
        env.reset()
        robot.update(env.step_dt)

        # Hold reset pose for 10 steps so PhysX settles
        hold_act = torch.zeros(2, env.action_space.shape[-1], device=device)
        hold_act[:, :6] = robot.data.joint_pos[:, joint_indices].clone()
        for _ in range(10):
            env.step(hold_act)
        robot.update(env.step_dt)

        # Record initial object positions (before Alice moves them)
        act = hold_act.clone()
        alice_hidden = None

        for step in range(args.alice_steps):
            # Only drive arena 0 with Alice; arena 1 holds still
            if alice_ac is not None:
                # Build Alice's obs (35D): robot_state(7) + obj_states(14 × N)
                wrist_pos = robot.data.body_pos_w[:, wrist_ids[0]] - env_origins
                wrist_quat = robot.data.body_quat_w[:, wrist_ids[0]]
                wrist_euler = _quat_to_euler_xyz(wrist_quat)
                finger = robot.data.joint_pos[:, gripper_joint_ids[:1]]
                rs = torch.cat([wrist_pos, wrist_euler, finger], dim=-1)  # (2,7)

                obj_parts = [rs]
                for oi in range(args.num_objects):
                    o = target_obj if oi == 0 else cube_obj
                    op  = o.data.root_pos_w - env_origins
                    oe  = _quat_to_euler_xyz(o.data.root_quat_w)
                    olv = o.data.root_lin_vel_w.clamp(-5, 5)
                    oav = o.data.root_ang_vel_w.clamp(-5, 5)
                    lf  = robot.data.body_pos_w[:, left_finger_ids[0]] - env_origins
                    rf  = robot.data.body_pos_w[:, right_finger_ids[0]] - env_origins
                    grip_loc = (lf + rf) / 2.0
                    dist = (op - grip_loc).norm(dim=-1, keepdim=True)
                    obj_parts.append(torch.cat([op, oe, olv, oav, dist,
                                                torch.zeros(2, 1, device=device)], dim=-1))
                alice_obs = torch.cat(obj_parts, dim=-1)  # (2,35 or 21)

                with torch.no_grad():
                    bins, _, _, _, _, alice_hidden = alice_ac.act_with_hidden(
                        alice_obs, None, alice_hidden
                    )
                # Apply Alice's action to arena 0 only; arena 1 holds
                ee_delta_a = _decode_bins(bins[:1])  # (1,3)
                cur_ee_loc_a = (robot.data.body_pos_w[:1, wrist_ids[0]]
                                - env_origins[:1])
                ee_target_a = cur_ee_loc_a + ee_delta_a
                lgj = robot.data.joint_pos[:1, joint_indices].clone()
                new_joints_a = _ik_batch(
                    torch.cat([ee_target_a, cur_ee_loc_a], dim=0),  # batch both
                    torch.cat([lgj, lgj], dim=0),
                )[:1]  # take only arena 0 result
                act[0, :6] = new_joints_a[0]
                act[1, :6] = robot.data.joint_pos[1, joint_indices]  # hold
                # Gripper: bins[:, 3] if num_cat_dims>3
                if num_cat_dims > 3:
                    center = (num_bins - 1) / 2.0
                    g_norm = (bins[0, 3].float() - center) / center
                    act[0, 6] = torch.sign(g_norm)
            else:
                # Random walk: small random EE deltas on arena 0
                rand_delta = (torch.rand(3, device=device) - 0.5) * 0.03
                cur_ee_loc = (robot.data.body_pos_w[0, wrist_ids[0]]
                              - env_origins[0])
                ee_target = (cur_ee_loc + rand_delta).unsqueeze(0)
                lgj = robot.data.joint_pos[:1, joint_indices].clone()
                new_joints = _ik_batch(
                    torch.cat([ee_target, ee_target], dim=0),
                    torch.cat([lgj, lgj], dim=0),
                )[:1]
                act[0, :6] = new_joints[0]
                act[1, :6] = robot.data.joint_pos[1, joint_indices]

            env.step(act)
            robot.update(env.step_dt)

        # Capture arena 0's final object poses as the shared goal
        for oi in range(args.num_objects):
            o = target_obj if oi == 0 else cube_obj
            pos_loc  = o.data.root_pos_w[0] - env_origins[0]   # (3,)
            quat     = o.data.root_quat_w[0]                    # (4,)
            euler    = _quat_to_euler_xyz(quat.unsqueeze(0))[0]  # (3,)
            g = torch.cat([pos_loc, euler])                      # (6,)
            goal_states[:, oi*6:(oi+1)*6] = g.unsqueeze(0).expand(2, -1)

        goal_ready[0] = True
        _place_goal_markers()
        print(f"[Goal] Goal set! Object positions (local, arena 0):")
        for oi in range(args.num_objects):
            print(f"  obj{oi}: pos={goal_states[0, oi*6:oi*6+3].cpu().numpy().round(3)}")

        # Reset BOTH arenas to the same initial object state (pre-Alice = env reset)
        env.reset()
        robot.update(env.step_dt)
        hold_act[:, :6] = robot.data.joint_pos[:, joint_indices].clone()
        for _ in range(15):
            env.step(hold_act)
        robot.update(env.step_dt)
        print("[Goal] Both arenas reset to start. Competition begins!\n")

    # ── competition state ──────────────────────────────────────────────────
    # last_good_joints: holds last successful IK solution per arena
    last_good_joints = robot.data.joint_pos[:, joint_indices].clone()  # (2,6)

    # IK EE targets tracked separately per arena for smooth velocity clamping
    cur_ee_target_local = (
        robot.data.body_pos_w[:, wrist_ids[0]] - env_origins
    ).clone()  # (2,3)

    bob_hidden = None
    action = torch.zeros(2, env.action_space.shape[-1], device=device)
    action[:, :6] = last_good_joints

    step_count       = torch.zeros(2, dtype=torch.long)
    solve_time       = [None, None]   # None = not yet solved; float = steps when solved
    solve_time_sec   = [None, None]
    start_wall_time  = None

    winner_announced = [False]

    def _announce_winner():
        if winner_announced[0]:
            return
        winner_announced[0] = True
        h_solved = solve_time[0] is not None
        b_solved = solve_time[1] is not None
        if h_solved and b_solved:
            if solve_time[0] < solve_time[1]:
                print("\n🏆  HUMAN WINS! "
                      f"Human: {solve_time_sec[0]:.1f}s  |  Bob: {solve_time_sec[1]:.1f}s")
            elif solve_time[1] < solve_time[0]:
                print("\n🤖  BOB WINS!  "
                      f"Bob: {solve_time_sec[1]:.1f}s  |  Human: {solve_time_sec[0]:.1f}s")
            else:
                print(f"\n⚡  TIE!  Both solved in {solve_time_sec[0]:.1f}s")
        elif h_solved:
            print(f"\n🏆  HUMAN solved it in {solve_time_sec[0]:.1f}s  (Bob still going…)")
        elif b_solved:
            print(f"\n🤖  BOB solved it in {solve_time_sec[1]:.1f}s  (human still going…)")

    # ── generate first goal ────────────────────────────────────────────────
    _generate_goal()
    start_wall_time = time.time()

    # ── main loop ─────────────────────────────────────────────────────────
    print("=" * 68)
    print("  Arena 0 (LEFT)  — YOU via gamepad")
    print("  Arena 1 (RIGHT) — Bob (loaded model)")
    print("  Press R to regenerate a new goal and reset.")
    print("=" * 68)

    try:
        while simulation_app.is_running():
            simulation_app.update()

            # ── goal regeneration ──────────────────────────────────────────
            if _regen[0]:
                _regen[0] = False
                _generate_goal()
                last_good_joints = robot.data.joint_pos[:, joint_indices].clone()
                cur_ee_target_local = (
                    robot.data.body_pos_w[:, wrist_ids[0]] - env_origins
                ).clone()
                step_count.zero_()
                solve_time       = [None, None]
                solve_time_sec   = [None, None]
                winner_announced[0] = False
                bob_hidden = None
                start_wall_time = time.time()
                action[:, :6] = last_good_joints
                gamepad.reset()
                continue

            # ── Arena 0: gamepad → EE delta → cuRobo ──────────────────────
            gp = gamepad.advance()
            # Remap: left-stick UP=+Y, RIGHT=-X; right-stick UP=+Z
            dx_gp, dy_gp = gp[0].item(), gp[1].item()
            dz_gp = gp[2].item()
            delta0 = torch.tensor([dy_gp, dx_gp, dz_gp], device=device) * max_step_delta
            cur_ee_target_local[0] += delta0
            # Clamp to workspace
            cur_ee_target_local[0, 0] = cur_ee_target_local[0, 0].clamp(-0.65, 0.65)
            cur_ee_target_local[0, 1] = cur_ee_target_local[0, 1].clamp(0.20, 0.75)
            cur_ee_target_local[0, 2] = cur_ee_target_local[0, 2].clamp(-0.02, 0.80)
            # Gripper
            action[0, 6] = gp[6]

            # ── Arena 1: Bob policy → EE delta → cuRobo ───────────────────
            bob_obs = _build_bob_obs(
                robot=robot, env=env, env_origins=env_origins,
                goal_states=goal_states,
                left_finger_ids=left_finger_ids,
                right_finger_ids=right_finger_ids,
                gripper_joint_ids=gripper_joint_ids,
                arm_joint_ids=joint_indices,
                target_obj=target_obj,
                cube_obj=cube_obj,
                num_objects=args.num_objects,
                device=device,
            )
            with torch.no_grad():
                bins, _, _, _, _, bob_hidden = bob_ac.act_with_hidden(
                    bob_obs, None, bob_hidden
                )
            ee_delta_bob = _decode_bins(bins[1:2])     # (1,3)
            cur_ee_target_local[1] = (cur_ee_target_local[1] + ee_delta_bob[0])
            cur_ee_target_local[1, 0] = cur_ee_target_local[1, 0].clamp(-0.65, 0.65)
            cur_ee_target_local[1, 1] = cur_ee_target_local[1, 1].clamp(0.20, 0.75)
            cur_ee_target_local[1, 2] = cur_ee_target_local[1, 2].clamp(-0.02, 0.80)
            # Gripper: dim 3 if num_cat_dims > 3
            if num_cat_dims > 3:
                ctr = (num_bins - 1) / 2.0
                g_norm = (bins[1, 3].float() - ctr) / ctr
                action[1, 6] = torch.sign(g_norm)

            # ── Batch IK for both arenas ───────────────────────────────────
            new_joints = _ik_batch(cur_ee_target_local, last_good_joints)
            last_good_joints = new_joints
            action[:, :6] = new_joints

            env.step(action)
            robot.update(env.step_dt)
            step_count += 1

            # ── Success check ──────────────────────────────────────────────
            if goal_ready[0]:
                success = _check_success(
                    env=env,
                    env_origins=env_origins,
                    goal_states=goal_states,
                    target_obj=target_obj,
                    cube_obj=cube_obj,
                    num_objects=args.num_objects,
                )
                elapsed = time.time() - start_wall_time
                for arena in range(2):
                    if success[arena] and solve_time[arena] is None:
                        solve_time[arena]     = step_count[arena].item()
                        solve_time_sec[arena] = elapsed
                        label = "HUMAN" if arena == 0 else "BOB  "
                        print(
                            f"[{label}] ✓ Solved in {solve_time_sec[arena]:.1f}s "
                            f"({solve_time[arena]} steps)"
                        )
                        _announce_winner()

            # ── periodic status ────────────────────────────────────────────
            if step_count[0] % 50 == 0:
                elapsed = time.time() - (start_wall_time or time.time())
                lines = []
                for arena, label in [(0, "Human"), (1, "Bob  ")]:
                    # Distance of each object to goal
                    dists = []
                    for oi in range(args.num_objects):
                        o = target_obj if oi == 0 else cube_obj
                        pos_l = o.data.root_pos_w[arena] - env_origins[arena]
                        g_pos = goal_states[arena, oi*6:oi*6+3]
                        dists.append(f"{(pos_l - g_pos).norm().item():.3f}m")
                    solved = "✓" if solve_time[arena] is not None else " "
                    lines.append(
                        f"  [{label}{solved}] step={step_count[arena]:4d} "
                        f"t={elapsed:.1f}s  dist={' '.join(dists)}"
                    )
                print("\n".join(lines))

    except KeyboardInterrupt:
        print("\nInterrupted.")

    simulation_app.close()


if __name__ == "__main__":
    main()
