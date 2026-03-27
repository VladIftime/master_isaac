"""
Meta-ASP environment wrapper for Phase 2 (Charlie Architecture).

Translates meta-level commands (g_left, g_right) ∈ R^K into worker actions
across C atomic steps, manages physics snapshot/restore for exact S0 reset,
and exposes global state observations to the meta agents.
"""

import torch
from typing import Optional


class MetaASPWrapper:
    """
    Two-tier wrapper for Phase 2 Meta-ASP.

    Meta level operates at 1/C the frequency of the physics engine.
    Workers (frozen Phase 1.5 Bob policies) execute sub-goals for C steps.

    Episode structure:
        1. Reset to S0
        2. Meta-Alice phase: TA_meta meta-steps → produces S* (master goal)
        3. Reset to S0
        4. Meta-Bob phase: TB_meta meta-steps → attempts to reach S*
        5. Evaluate D(S_final, S*) → success/failure
    """

    def __init__(
        self,
        env,                        # AsyncDualPlayEnvWrapper instance
        worker_left,                # Phase 1.5 ActorCritic, frozen
        worker_right,               # Phase 1.5 ActorCritic, frozen
        K: int,                     # Goal embedding dimension from Phase 1
        C: int = 30,                # Atomic steps per meta-step
        TA_meta: int = 10,          # Meta-Alice's meta-step budget
        TB_meta: int = 15,          # Meta-Bob's meta-step budget
        success_threshold: float = 0.04,
        device: str = "cuda",
    ):
        self.env = env
        self.worker_left = worker_left
        self.worker_right = worker_right
        self.K = K
        self.C = C
        self.TA_meta = TA_meta
        self.TB_meta = TB_meta
        self.success_threshold = success_threshold
        self.device = device
        self.num_envs = env.num_envs

        # Worker LSTM hidden states — persist across C steps within one meta-step,
        # reset at episode boundaries via reset_worker_hidden()
        self.hidden_left = None
        self.hidden_right = None

    # ------------------------------------------------------------------
    # Worker Execution
    # ------------------------------------------------------------------

    def execute_workers(
        self,
        g_left: torch.Tensor,   # (num_envs, K)
        g_right: torch.Tensor,  # (num_envs, K)
    ) -> torch.Tensor:
        """
        Run both workers for C atomic steps with fixed sub-goals.

        g is injected directly into the worker's _goal_proj layer via the
        goal_override parameter, bypassing GoalEncoder entirely
        (Charlie paper Section 2.3).

        LSTM hidden states are carried across all C steps to retain temporal
        context (the worker's recurrent memory of the ongoing manipulation).
        Hidden states are only reset at episode boundaries.

        Returns:
            Global state after C steps: (num_envs, 30)
        """
        for _ in range(self.C):
            obs_left = self._get_left_arm_obs()    # (num_envs, 59)
            obs_right = self._get_right_arm_obs()  # (num_envs, 59)

            with torch.no_grad():
                act_left, *_, self.hidden_left = self.worker_left.act_with_hidden(
                    obs_left, None, self.hidden_left, goal_override=g_left
                )
                act_right, *_, self.hidden_right = self.worker_right.act_with_hidden(
                    obs_right, None, self.hidden_right, goal_override=g_right
                )

            env_action = self._merge_arm_actions(act_left, act_right)
            self.env.step(env_action)

        return self.get_global_state()

    def reset_worker_hidden(self):
        """Reset worker LSTM hidden states at meta-episode boundaries."""
        self.hidden_left = None
        self.hidden_right = None

    # ------------------------------------------------------------------
    # Global State Observations
    # ------------------------------------------------------------------

    def get_global_state(self) -> torch.Tensor:
        """
        Full global state for Meta-Alice and Meta-Bob.

        Layout (per env):
            left_ee_pose(7)     — EE pos(3) + quat(4), LOCAL frame
            left_gripper(1)     — gripper state
            right_ee_pose(7)    — EE pos(3) + quat(4), LOCAL frame
            right_gripper(1)    — gripper state
            target_obj(7)       — pos(3) + quat(4), LOCAL frame
            cube(7)             — pos(3) + quat(4), LOCAL frame
        Total: 30 dims

        Meta-Bob additionally receives S* appended (done in train_meta.py):
            global_state(30) + master_goal(14) = 44 dims
        """
        scene = self.env.env.scene
        origins = scene.env_origins

        robot = scene["robot"]

        # Left arm EE (using wrist_3_link)
        body_ids, _ = robot.find_bodies("wrist_3_link")
        ee_pos = robot.data.body_pos_w[:, body_ids[0]] - origins
        ee_quat = robot.data.body_quat_w[:, body_ids[0]]
        left_ee = torch.cat([ee_pos, ee_quat], dim=-1)  # (N, 7)

        # Left gripper
        joint_ids, _ = robot.find_joints(["finger_joint"])
        left_gripper = robot.data.joint_pos[:, joint_ids]  # (N, 1)

        # Right arm EE — same robot in current single-arm-per-env setup
        # In dual-arm, this would reference a second robot asset
        right_ee = left_ee.clone()      # Placeholder: updated when dual arm exists
        right_gripper = left_gripper.clone()

        # Object states (pos + quat only, in local frame)
        target = scene["target_object"]
        target_pos = target.data.root_pos_w - origins
        target_quat = target.data.root_quat_w
        if target_pos.dim() == 3:
            target_pos = target_pos[:, 0, :]
            target_quat = target_quat[:, 0, :]
        target_state = torch.cat([target_pos, target_quat], dim=-1)  # (N, 7)

        cube = scene["cube"]
        cube_pos = cube.data.root_pos_w - origins
        cube_quat = cube.data.root_quat_w
        if cube_pos.dim() == 3:
            cube_pos = cube_pos[:, 0, :]
            cube_quat = cube_quat[:, 0, :]
        cube_state = torch.cat([cube_pos, cube_quat], dim=-1)  # (N, 7)

        return torch.cat([
            left_ee, left_gripper,
            right_ee, right_gripper,
            target_state, cube_state,
        ], dim=-1)  # (N, 30)

    def get_object_state(self) -> torch.Tensor:
        """
        Object-only state for master goal recording.
        Returns: (num_envs, 14) — [target_pos(3), target_quat(4), cube_pos(3), cube_quat(4)]
        """
        scene = self.env.env.scene
        origins = scene.env_origins

        target = scene["target_object"]
        target_pos = target.data.root_pos_w - origins
        target_quat = target.data.root_quat_w
        if target_pos.dim() == 3:
            target_pos = target_pos[:, 0, :]
            target_quat = target_quat[:, 0, :]

        cube = scene["cube"]
        cube_pos = cube.data.root_pos_w - origins
        cube_quat = cube.data.root_quat_w
        if cube_pos.dim() == 3:
            cube_pos = cube_pos[:, 0, :]
            cube_quat = cube_quat[:, 0, :]

        return torch.cat([target_pos, target_quat, cube_pos, cube_quat], dim=-1)

    # ------------------------------------------------------------------
    # Physics Snapshot / Restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Snapshot full physics state for exact S0 restoration after Meta-Alice."""
        scene = self.env.env.scene
        robot = scene["robot"]
        target = scene["target_object"]
        cube = scene["cube"]

        return {
            "robot_joint_pos": robot.data.joint_pos.clone(),
            "robot_joint_vel": robot.data.joint_vel.clone(),
            "target_pos_w": target.data.root_pos_w.clone(),
            "target_quat_w": target.data.root_quat_w.clone(),
            "target_lin_vel_w": target.data.root_lin_vel_w.clone(),
            "target_ang_vel_w": target.data.root_ang_vel_w.clone(),
            "cube_pos_w": cube.data.root_pos_w.clone(),
            "cube_quat_w": cube.data.root_quat_w.clone(),
            "cube_lin_vel_w": cube.data.root_lin_vel_w.clone(),
            "cube_ang_vel_w": cube.data.root_ang_vel_w.clone(),
        }

    def restore(self, snap: dict, env_ids: Optional[torch.Tensor] = None):
        """Restore physics to exact snapshot state."""
        scene = self.env.env.scene
        robot = scene["robot"]
        target = scene["target_object"]
        cube = scene["cube"]

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Robot
        robot.write_joint_state_to_sim(
            snap["robot_joint_pos"][env_ids],
            snap["robot_joint_vel"][env_ids],
            env_ids=env_ids,
        )

        # Target object
        target_pose = torch.cat([
            snap["target_pos_w"][env_ids],
            snap["target_quat_w"][env_ids],
        ], dim=-1)
        target_vel = torch.cat([
            snap["target_lin_vel_w"][env_ids],
            snap["target_ang_vel_w"][env_ids],
        ], dim=-1)
        target.write_root_pose_to_sim(target_pose, env_ids=env_ids)
        target.write_root_velocity_to_sim(target_vel, env_ids=env_ids)

        # Cube
        cube_pose = torch.cat([
            snap["cube_pos_w"][env_ids],
            snap["cube_quat_w"][env_ids],
        ], dim=-1)
        cube_vel = torch.cat([
            snap["cube_lin_vel_w"][env_ids],
            snap["cube_ang_vel_w"][env_ids],
        ], dim=-1)
        cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim(cube_vel, env_ids=env_ids)

        scene.write_data_to_sim()

    # ------------------------------------------------------------------
    # Success Evaluation
    # ------------------------------------------------------------------

    def compute_success(
        self,
        S_final: torch.Tensor,  # (num_envs, 14)
        S_star: torch.Tensor,   # (num_envs, 14)
    ) -> torch.Tensor:
        """
        Evaluate whether Meta-Bob successfully reached S*.

        Uses per-object positional distance; success if both objects
        are within self.success_threshold meters.

        Returns: (num_envs,) bool tensor
        """
        target_dist = torch.norm(S_final[:, :3] - S_star[:, :3], dim=-1)
        cube_dist = torch.norm(S_final[:, 7:10] - S_star[:, 7:10], dim=-1)
        return (target_dist < self.success_threshold) & (cube_dist < self.success_threshold)

    # ------------------------------------------------------------------
    # Internal helpers (to be implemented per-environment)
    # ------------------------------------------------------------------

    def _get_left_arm_obs(self) -> torch.Tensor:
        """Get current observation vector for left arm worker. (num_envs, 59)"""
        # In single-arm Phase 1 setup, this returns the full bob observation
        return self.env.get_bob_obs()

    def _get_right_arm_obs(self) -> torch.Tensor:
        """Get current observation vector for right arm worker. (num_envs, 59)"""
        # Placeholder: in dual-arm, this reads from the second arm's observation group
        return self.env.get_bob_obs()

    def _merge_arm_actions(
        self,
        act_left: torch.Tensor,
        act_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge left and right arm actions into full environment action.

        In single-arm: returns act_left directly.
        In dual-arm: concatenates along the action dimension.
        """
        # Single-arm placeholder — dual-arm will concatenate
        return act_left
