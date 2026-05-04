"""
CuRobo Interactive Follow-Target Test (IsaacLab Native)
=======================================================

Spawns a red visual sphere between the grippers that you can move with the mouse.
CuRobo computes the IK in real-time. A velocity clamp ensures the robot
smoothly chases the ball without jumps.

Usage:
    python tests/test_curobo_follow_target.py --max_vel 0.5
"""

import argparse
import os
import sys
import torch
import numpy as np

# ==============================================================================
# IMPORT SHIELD: Import CuRobo FIRST before modifying sys.path
# This prevents your local directories from shadowing the installed library.
# ==============================================================================
try:
    import curobo
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
except ModuleNotFoundError:
    print("\n[ERROR] CuRobo not found in .master_venv. Ensure it is installed.")
    sys.exit(1)
# ==============================================================================

import isaaclab.app
from isaaclab.app import AppLauncher

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    parser = argparse.ArgumentParser(description="CuRobo Interactive Follow Target")
    parser.add_argument("--max_vel", type=float, default=0.5, help="Max ball velocity in m/s")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ─── Isaac Lab & Config Imports ───────────────────────────────────────────────
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import UsdGeom, Usd

    # 1. Define Joint Names in Kinematic Chain Order
    cu_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    # 2. Environment Setup
    print("\nCreating Environment...")
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Override arm_action to use absolute Joint Positions for direct CuRobo control[cite: 3]
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=cu_joint_names, 
        scale=1.0,
        use_default_offset=False 
    )
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # 3. CuRobo IK Solver Setup (v0.7.8 Compatible)
    print("\nInitializing CuRobo IKSolver...")
    tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    
    # Load UR5e config from CuRobo registry[cite: 8]
    ur5e_yaml = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    robot_cfg = RobotConfig.from_dict(ur5e_yaml["robot_cfg"], tensor_args) 
    
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_model=None, 
        tensor_args=tensor_args
    )
    ik_solver = IKSolver(ik_config)

    # 4. Initialize Pose and Spawn Ball
    env.reset()
    robot_asset = env.scene["robot"]
    robot_asset.update(env.step_dt)

    # grasp_convenient_link is massless and dropped by PhysX.
    # Use the midpoint of the two inner fingers as the TCP instead.
    wrist3_id, _       = robot_asset.find_bodies("wrist_3_link")
    left_finger_id, _  = robot_asset.find_bodies("left_inner_finger")
    right_finger_id, _ = robot_asset.find_bodies("right_inner_finger")
    joint_indices, _   = robot_asset.find_joints(cu_joint_names)

    # Spawn ball at the midpoint between the two finger pads.
    left_pos  = robot_asset.data.body_pos_w[0, left_finger_id[0]]
    right_pos = robot_asset.data.body_pos_w[0, right_finger_id[0]]
    ball_initial_pos = ((left_pos + right_pos) / 2.0).cpu().numpy()

    target_path = "/World/interactive_target"
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.03, # Slightly smaller for better fit
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
    )
    sphere_cfg.func(target_path, sphere_cfg, translation=ball_initial_pos)

    stage = omni.usd.get_context().get_stage()
    target_xform = UsdGeom.Xformable(stage.GetPrimAtPath(target_path))

    # 5. Control Parameters
    max_step_delta = args.max_vel * env.step_dt
    # Sync tracking variable to the EXACT spawn position to avoid startup spasming
    current_ik_target_pos = torch.tensor(ball_initial_pos, device=device, dtype=torch.float32)
    
    # Tool orientation: pointing straight down[cite: 1]
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    print("\n" + "="*70)
    print(f"  [Interactive Mode Ready]")
    print(f"  1. Click the Red Ball in the viewport.")
    print(f"  2. Press 'W' to activate the translation gizmo.")
    print(f"  3. Drag the ball to guide the robot.")
    print(f"  Max velocity clamped to: {args.max_vel} m/s")
    print("="*70 + "\n")

    try:
        while simulation_app.is_running():
            # A. Read User's Mouse Position from USD API[cite: 1]
            transform = target_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = transform.ExtractTranslation()
            mouse_pos = torch.tensor(list(translation), device=device, dtype=torch.float32)

            # B. Apply Velocity Clamping[cite: 1]
            delta = mouse_pos - current_ik_target_pos
            distance = torch.norm(delta)
            
            if distance > max_step_delta:
                current_ik_target_pos += (delta / distance) * max_step_delta
            else:
                current_ik_target_pos = mouse_pos

            # C. Compute live TCP offset and convert to IK target for wrist_3_link.
            # CuRobo drives wrist_3_link; grasp_convenient_link is ~17cm past it.
            # Subtracting the live offset ensures the TCP (not the flange) reaches the ball.
            left_pos_w   = robot_asset.data.body_pos_w[0, left_finger_id[0]]
            right_pos_w  = robot_asset.data.body_pos_w[0, right_finger_id[0]]
            tcp_pos_w    = (left_pos_w + right_pos_w) / 2.0
            wrist3_pos_w = robot_asset.data.body_pos_w[0, wrist3_id[0]]
            tcp_offset_world = tcp_pos_w - wrist3_pos_w

            env_origin = env.scene.env_origins[0]
            ik_target_world = current_ik_target_pos - tcp_offset_world
            local_target_pos = ik_target_world - env_origin

            goal_pose = Pose(
                position=local_target_pos.unsqueeze(0),
                quaternion=target_quat
            )

            # D. Get current joint positions in correct order
            current_joint_pos = robot_asset.data.joint_pos[:, joint_indices]
            
            # E. Solve IK (Seeded with current joints to prevent flipping)[cite: 1]
            ik_result = ik_solver.solve_single(
                goal_pose, 
                seed_config=current_joint_pos.unsqueeze(1),
                retract_config=current_joint_pos
            )

            # F. Apply Actions[cite: 1, 3]
            action = torch.zeros((1, env.action_space.shape[-1]), device=device)
            
            if ik_result.success.any():
                # IK succeeded: Update the 6 arm joints
                action[0, :6] = ik_result.solution.view(-1)[:6] 
            else:
                # IK failed/Unreachable: Hold current position
                action[0, :6] = current_joint_pos.view(-1)
                
            env.step(action)

    except KeyboardInterrupt:
        print("\nExiting Interactive Mode.")

    simulation_app.close()

if __name__ == "__main__":
    main()