"""
Custom environment wrapper for asymmetric dual-play.

This wrapper implements the Asymmetric Self-Play (ASP) framework for dual-arm
robot manipulation training. It manages:

1. Two-phase episode structure (Alice → Bob)
2. Asymmetric observations for each agent
3. Goal state capture and validation
4. Phase transitions based on timesteps and success
5. **SPARSE REWARDS ONLY** (integer-valued, per ASP paper philosophy)

Key Design Decision: Bob uses sparse binary rewards (+1/-1/+5) without dense
distance-based guidance. This follows the ASP paper's unsupervised goal discovery
approach where the curriculum emerges from adversarial self-play.

See ASP_IMPLEMENTATION.md for detailed documentation.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ...utils.episode_manager import EpisodeManager, Phase
from ...utils.goal_validator import validate_goal
from . import observations
from . import rewards as reward_utils
from .events import reset_objects_to_fixed_safe_pose, reset_robot_joints


class AsyncDualPlayEnvWrapper:
    """
    Wrapper for asymmetric dual-play environment.

    This manages the two-phase episodic structure where Alice sets goals
    and Bob attempts to solve them.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        alice_timesteps: int = 100,
        bob_timesteps: int = 200,
        max_goals_per_episode: int = 5,
        num_objects: int = 2,  # target_object and cube
        device: str = "cuda",
        arm_config: str = "default",
    ):
        self.env = env
        self.device = device
        self.num_objects = num_objects
        self.arm_config = arm_config

        # Safe reset positions (LOCAL frame) for initial_states tracking.
        # Objects are physically spawned at z=0.05 (events.py) but gravity settles
        # them to z≈0.023 on the table surface within the first few sim steps.
        # We record the SETTLED height here so the movement metric isn't inflated
        # by the ~2.7cm gravity drop. This preserves Z in the distance check so
        # Alice is still rewarded for lifting objects in the future.
        _id_euler = torch.tensor(
            [0.0, 0.0, 0.0], device=device
        )  # identity rotation (ZYX Euler)
        _t_pos = torch.tensor([-0.15, 0.7, 0.023], device=device)
        _c_pos = torch.tensor([-0.25, 0.7, 0.023], device=device)
        # Shape (12,): [t_pos(3), t_euler(3), c_pos(3), c_euler(3)] — Euler format matches _extract_object_states
        self._safe_reset_state = torch.cat([_t_pos, _id_euler, _c_pos, _id_euler])

        # Bob's success threshold (5cm)
        self.goal_tolerance = 0.05

        # Create episode manager
        self.episode_manager = EpisodeManager(
            num_envs=env.num_envs,
            device=device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            max_goals_per_episode=max_goals_per_episode,
        )
        # Attach manager to env so reward functions can access it
        self.env.episode_manager = self.episode_manager

        # Robot state dimensions (EE pose: 6 euler + 1 gripper = 7D)
        self.robot_state_dim = 7

        # Fetch the dimensions directly from the observation manager
        alice_dim_info = self.env.unwrapped.observation_manager.group_obs_dim[
            "alice_policy"
        ]
        bob_dim_info = self.env.unwrapped.observation_manager.group_obs_dim[
            "bob_policy"
        ]

        # Extract the integer dimension (handling whether Isaac Lab returns a tuple or raw int)
        self.alice_obs_dim = (
            alice_dim_info[0]
            if isinstance(alice_dim_info, (tuple, list))
            else alice_dim_info
        )
        self.bob_obs_dim = (
            bob_dim_info[0] if isinstance(bob_dim_info, (tuple, list)) else bob_dim_info
        )

        print(
            f"[Wrapper] Dynamically set Alice Obs Dim: {self.alice_obs_dim}, Bob Obs Dim: {self.bob_obs_dim}"
        )

        # Re-create the spaces as properly-shaped Gymnasium boxes (no batch dim) for training compatibility
        self.alice_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.alice_obs_dim,), dtype=np.float32
        )
        self.bob_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.bob_obs_dim,), dtype=np.float32
        )

        # Action and observation space (defaults to Bob's since it's larger)
        # However, for the training loop consistency, we define a combined space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.alice_obs_dim + self.bob_obs_dim,),
            dtype=np.float32,
        )
        self.state_space = self.observation_space

        # Handle Action Space (Strip batch dimension (num_envs) for PPO compatibility)
        # env.action_space is (num_envs, action_dim)
        if len(env.action_space.shape) > 1:
            self.action_space = gym.spaces.Box(
                low=env.action_space.low[0],
                high=env.action_space.high[0],
                shape=env.action_space.shape[1:],
                dtype=env.action_space.dtype,
            )
        else:
            self.action_space = env.action_space

        # Table and placement bounds for goal validation
        self.table_bounds = {
            "x_range": (-1.0, 1.0),
            "y_range": (-0.5, 1.5),
            "z_min": -0.2,
        }
        self.placement_bounds = {
            "x_range": (-0.6, 0.6),
            "y_range": (0.3, 0.9),
        }

        # Buffer to hold Alice's rewards until her cycle ends (either Alice failed or Bob finished)
        self.delayed_alice_reward = torch.zeros(env.num_envs, device=self.device)

        # Per-iteration aggregate stats (reset each iteration via reset_iter_stats())
        self._iter_stats = self._make_iter_stats()

        print(f"[AsyncDualPlayEnvWrapper] Initialized (task-space mode)")
        print(f"  Alice obs: {self.alice_obs_dim}, Bob obs: {self.bob_obs_dim}")
        print(f"  Robot state: {self.robot_state_dim} (EE pose 6 euler + gripper 1)")

    # ------------------------------------------------------------------
    # Per-iteration stats helpers
    # ------------------------------------------------------------------

    def _make_iter_stats(self) -> dict:
        return {
            "invalid_goals": 0,
            "valid_goals": 0,
            "bob_successes": 0,
            "bob_failures": 0,
            "terminations": {},        # reason -> count
        }

    def reset_iter_stats(self):
        self._iter_stats = self._make_iter_stats()

    def get_iter_stats(self) -> dict:
        return dict(self._iter_stats)

    @property
    def num_envs(self):
        return self.env.num_envs

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Reset environment and episode manager"""
        obs_dict, info = self.env.reset(seed=seed, options=options)

        # Reset episode manager
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.episode_manager.reset_episode(env_ids, reason="Global Manual Reset")
        self.delayed_alice_reward[env_ids] = 0.0

        # Place objects at safe starting positions (Isaac Lab's default reset may use
        # different spawn locations). Set initial_states directly from the known safe
        # pose rather than reading physics cache (avoids PhysX readback timing issues).
        reset_objects_to_fixed_safe_pose(self.env, env_ids)
        reset_robot_joints(self.env, env_ids)
        self.env.scene.write_data_to_sim()
        self.episode_manager.initial_states = (
            self._safe_reset_state.unsqueeze(0).expand(self.num_envs, -1).clone()
        )

        # Return concatenated observations so train.py can slice them
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        return obs, info

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step environment and handle phase transitions.

        Returns:
            observations: Tensor with appropriate obs for current phase
            rewards: Rewards for current agent
            terminated: Episode termination flags
            truncated: Episode truncation flags
            info: Additional info including phase transitions
        """
        # Action scaling for RMPFlow (use_relative_mode=True):
        # Now handled in reach_dual_arm_env_cfg.py via scale=(0.015, 0.05)
        scaled_action = action.clone()

        # Step base environment
        obs_dict, rewards, terminated, truncated, extras = self.env.step(scaled_action)

        # Capture phase state BEFORE any resets so reward/success logic sees the
        # correct phase even for envs that terminate this step.
        is_alice_before = self.episode_manager.is_alice_phase().clone()
        is_bob_before = self.episode_manager.is_bob_phase().clone()

        # Sync EpisodeManager with Env Resets
        dones = terminated | truncated
        if dones.any():
            reset_ids = torch.where(dones)[0]
            is_alice = self.episode_manager.is_alice_phase()

            # Build a reason string per env from the termination manager.
            # IsaacLab stores _term_dones as (num_envs, num_conditions) bool tensor
            # and condition names in _term_names list.
            tm = self.env.termination_manager
            term_dones = tm._term_dones        # (num_envs, num_conditions)
            term_names = tm._term_names        # list[str]

            for env_id in reset_ids:
                is_term = terminated[env_id].item()

                # Collect names of conditions that fired for this env.
                fired = [
                    term_names[i]
                    for i in range(len(term_names))
                    if term_dones[env_id, i].item()
                ]

                if fired:
                    reason = " | ".join(fired)
                elif is_term:
                    reason = "physics_termination"
                else:
                    reason = "timeout"

                # Accumulate termination counts for the iteration summary
                self._iter_stats["terminations"][reason] = (
                    self._iter_stats["terminations"].get(reason, 0) + 1
                )

                # Only penalize hard physics terminations (not timeouts)
                if not is_term:
                    self.episode_manager.reset_episode(
                        env_id.unsqueeze(0), reason=reason
                    )
                    continue

                if is_alice[env_id]:
                    if not hasattr(self, "_early_alice_failures"):
                        self._early_alice_failures = torch.zeros(
                            self.num_envs, dtype=torch.bool, device=self.device
                        )
                    self._early_alice_failures[env_id] = True
                    self.delayed_alice_reward[env_id] = -3.0

                self.episode_manager.reset_episode(env_id.unsqueeze(0), reason=reason)

        # Advance episode manager
        phase_info = self.episode_manager.step()
        any_reset = False

        # Prepare extras for Alice's total reward
        extras["alice_total_reward"] = torch.zeros(self.num_envs, device=self.device)
        extras["alice_failed_this_step"] = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Pull in early failures if any
        if hasattr(self, "_early_alice_failures"):
            extras["alice_failed_this_step"] |= self._early_alice_failures
            extras["alice_total_reward"] += torch.where(
                self._early_alice_failures,
                self.delayed_alice_reward,
                torch.zeros_like(self.delayed_alice_reward),
            )
            early_fail_ids = torch.where(self._early_alice_failures)[0]
            if len(early_fail_ids) > 0:
                self.delayed_alice_reward[early_fail_ids] = 0.0
            self._early_alice_failures.fill_(False)

        # Handle Alice phase completion
        if phase_info["alice_done"].any():
            alice_done_ids = torch.where(phase_info["alice_done"])[0]
            valid, invalid_ids = self._handle_alice_completion(obs_dict, alice_done_ids)

            if len(invalid_ids) > 0:
                extras["alice_total_reward"][invalid_ids] = self.delayed_alice_reward[
                    invalid_ids
                ]
                extras["alice_failed_this_step"][invalid_ids] = True
                terminated[invalid_ids] = True
                self.delayed_alice_reward[invalid_ids] = 0.0

            any_reset = True

        # Track Bob's success
        step_bob_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        step_bob_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        step_pos_err = torch.zeros(self.num_envs, device=self.device)
        step_rot_err = torch.zeros(self.num_envs, device=self.device)

        # Handle Bob phase completion
        if phase_info["bob_done"].any():
            bob_done_ids = torch.where(phase_info["bob_done"])[0]
            success, pos_err, rot_err = self._handle_bob_completion(
                obs_dict, bob_done_ids
            )
            step_bob_success[bob_done_ids] = success
            step_bob_done[bob_done_ids] = True
            step_pos_err[bob_done_ids] = pos_err
            step_rot_err[bob_done_ids] = rot_err

            extras["alice_total_reward"][bob_done_ids] = self.delayed_alice_reward[
                bob_done_ids
            ]
            self.delayed_alice_reward[bob_done_ids] = 0.0
            any_reset = True

        if any_reset:
            obs_dict = self.env.observation_manager.compute()

        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_rewards, bob_achieved_completion = self._get_current_rewards(
            obs_dict, rewards, is_alice_before, is_bob_before, action
        )

        # Handle Bob's EARLY SUCCESS
        if bob_achieved_completion.any():
            completion_ids = torch.where(bob_achieved_completion)[0]
            self.episode_manager.bob_success[completion_ids] = True
            step_bob_success[completion_ids] = True
            step_bob_done[completion_ids] = True
            self.episode_manager.completion_given[completion_ids] = True

            success, pos_err, rot_err = self._check_bob_success(
                obs_dict, completion_ids
            )
            step_pos_err[completion_ids] = pos_err
            step_rot_err[completion_ids] = rot_err

            completion_steps = self.episode_manager.phase_step[completion_ids].clone()
            completion_goals = self.episode_manager.goal_count[completion_ids].clone()

            self._handle_bob_success_transition(completion_ids)

            extras["alice_total_reward"][completion_ids] = self.delayed_alice_reward[
                completion_ids
            ]
            self.delayed_alice_reward[completion_ids] = 0.0


            can_continue = (
                self.episode_manager.goal_count[completion_ids]
                < self.episode_manager.max_goals
            )
            terminated[completion_ids] = ~can_continue
            any_reset = True

        if any_reset:
            obs_dict = self.env.observation_manager.compute()
            current_obs = self._get_current_observations(obs_dict)

        # Track contact forces
        if hasattr(self.env.scene, "sensors"):
            bob_mask = self.episode_manager.is_bob_phase()
            if bob_mask.any():
                forces = []
                if "contact_forces" in self.env.scene.sensors:
                    f = self.env.scene.sensors["contact_forces"].data.net_forces_w
                    forces.append(torch.max(torch.norm(f, dim=-1), dim=-1)[0])
                if forces:
                    max_f = torch.max(torch.stack(forces, dim=0), dim=0)[0]
                    self.episode_manager.max_contact_force = torch.where(
                        bob_mask,
                        torch.max(self.episode_manager.max_contact_force, max_f),
                        self.episode_manager.max_contact_force,
                    )

        extras["episode_manager"] = {
            "phase": self.episode_manager.current_phase.clone(),
            "goal_count": self.episode_manager.goal_count.clone(),
            "bob_success": self.episode_manager.bob_success.clone(),
            "bob_success_this_step": step_bob_success,
            "bob_done_this_step": step_bob_done,
            "bob_pos_err": step_pos_err,
            "bob_rot_err": step_rot_err,
            "goal_valid": self.episode_manager.goal_valid.clone(),
            "goal_states": (
                self.episode_manager.goal_states.clone()
                if self.episode_manager.goal_states is not None
                else torch.zeros((self.num_envs, 12), device=self.device)
            ),
            "max_contact_force": self.episode_manager.max_contact_force.clone(),
        }

        extras["goal_valid"] = self.episode_manager.goal_valid.clone()
        extras["bob_success"] = self.episode_manager.bob_success.clone()
        self.previous_actions = action.clone()

        return obs, current_rewards, terminated, truncated, extras

    def _handle_alice_completion(
        self, obs_dict: Dict, env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle completion of Alice's phase."""
        goal_state = self._extract_object_states(obs_dict)
        initial_state = self.episode_manager.initial_states

        active_goal = goal_state[env_ids]  # (N, 12) Euler local
        active_initial = initial_state[env_ids]  # (N, 12) Euler local

        # Alice must move objects MORE than Bob's success threshold (0.04m),
        # otherwise Bob starts already within the goal zone → instant win.
        # 0.05m gives a 1cm safety margin above Bob's 0.04m threshold.
        alice_pos_req = 0.05
        alice_rot_req = 0.25

        # 2. Validate Goal (Local vs Local)
        valid, val_reward, reasons = validate_goal(
            active_initial,
            active_goal,
            self.table_bounds,
            self.placement_bounds,
            pos_threshold=alice_pos_req,
            rot_threshold=alice_rot_req,
        )

        # 2b. Minimum XY-displacement filter.
        # validate_goal passes rotation-only goals (obj spins in place, no XY movement).
        # When that happens Bob starts the phase with the object already at its goal XY
        # position → instant success in 1-2 steps with zero learning signal.
        # Require at least one object to move >7cm in XY (above Bob's 5cm success threshold).
        _MIN_XY_DISP = 0.07  # 7cm — 2cm margin above Bob's goal_tolerance
        target_xy_disp = torch.norm(active_goal[:, 0:2] - active_initial[:, 0:2], dim=-1)
        cube_xy_disp   = torch.norm(active_goal[:, 6:8]  - active_initial[:, 6:8],  dim=-1)
        sufficient_xy  = (target_xy_disp > _MIN_XY_DISP) | (cube_xy_disp > _MIN_XY_DISP)
        xy_fail = valid & ~sufficient_xy
        val_reward = val_reward.clone()
        val_reward[xy_fail] = 0.0
        valid = valid & sufficient_xy
        for i in range(len(env_ids)):
            if xy_fail[i].item():
                reasons[i] = "XY Disp Too Small (0.0)"

        self.delayed_alice_reward[env_ids] = val_reward

        # 3. Per-env Alice reward events + aggregate goal counts
        start_pos = active_initial[:, 0:3]
        final_pos = active_goal[:, 0:3]
        dist_xy = torch.norm(final_pos[:, :2] - start_pos[:, :2], dim=-1)
        dist_z = (final_pos[:, 2] - start_pos[:, 2]).abs()
        for i, env_id in enumerate(env_ids):
            rew = val_reward[i].item()
            if valid[i].item():
                self._iter_stats["valid_goals"] += 1
                print(
                    f"  [AliceRew] Env {env_id.item()}: {rew:+.1f} valid goal | "
                    f"XY={dist_xy[i]:.3f}m Z={dist_z[i]:.3f}m",
                    flush=True,
                )
            else:
                self._iter_stats["invalid_goals"] += 1
                print(
                    f"  [AliceRew] Env {env_id.item()}: {rew:+.1f} invalid goal "
                    f"({reasons[i]}) | XY={dist_xy[i]:.3f}m",
                    flush=True,
                )

        # 4. Storage for Bob (LOCAL)
        self.episode_manager.store_goal_state(active_goal, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)

        # 5. Transition logic
        successful_goal = valid
        valid_env_ids = env_ids[successful_goal]

        if len(valid_env_ids) > 0:
            self.episode_manager.transition_to_bob(valid_env_ids)
            start_states = self.episode_manager.initial_states[valid_env_ids]
            origins = self.env.scene.env_origins[valid_env_ids]

            # Reset Target (Local -> World) — 12D Euler layout: [t_pos(3)|t_euler(3)|c_pos(3)|c_euler(3)]
            # write_root_pose_to_sim expects 7D (pos3 + quat4), so convert euler→quat first.
            from .observations import _euler_xyz_to_quat

            t_pos_local = start_states[:, 0:3]
            t_quat = _euler_xyz_to_quat(start_states[:, 3:6])
            pos1_global = t_pos_local + origins
            self.env.scene["target_object"].write_root_pose_to_sim(
                torch.cat([pos1_global, t_quat], dim=-1),
                env_ids=valid_env_ids,
            )

            # Reset Cube (Local -> World) — cube starts at indices 6:12 in 12D layout
            c_pos_local = start_states[:, 6:9]
            c_quat = _euler_xyz_to_quat(start_states[:, 9:12])
            pos2_global = c_pos_local + origins
            self.env.scene["cube"].write_root_pose_to_sim(
                torch.cat([pos2_global, c_quat], dim=-1),
                env_ids=valid_env_ids,
            )

            reset_robot_joints(self.env, valid_env_ids)
            self.env.scene.write_data_to_sim()

        # 6. Handle Invalid
        invalid_env_ids = env_ids[~successful_goal]
        if len(invalid_env_ids) > 0:
            self.episode_manager.reset_episode(
                invalid_env_ids, reason="Alice Invalid Goal"
            )
            reset_objects_to_fixed_safe_pose(self.env, invalid_env_ids)
            reset_robot_joints(self.env, invalid_env_ids)
            self.episode_manager.initial_states[invalid_env_ids] = (
                self._safe_reset_state.unsqueeze(0)
                .expand(len(invalid_env_ids), -1)
                .clone()
            )
            self.env.scene.write_data_to_sim()

        return valid, invalid_env_ids

    def _handle_bob_completion(
        self, obs_dict: Dict, env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle completion of Bob's phase"""
        # --- 1. Success check ---
        success, pos_err, rot_err = self._check_bob_success(obs_dict, env_ids)
        self.episode_manager.mark_bob_success(env_ids, success)

        # --- 2. Alice Outcome Reward (+5 if Bob fails, 0 if he succeeds) ---
        outcome_rewards = torch.where(
            success,
            torch.tensor(reward_utils.ALICE_BOB_SUCCESS_REWARD, device=self.device),
            torch.tensor(reward_utils.ALICE_BOB_FAIL_REWARD, device=self.device),
        )
        self.delayed_alice_reward[env_ids] += outcome_rewards

        # --- 3. Aggregate success/failure counts ---
        n_success = int(success.sum().item())
        n_failure = len(env_ids) - n_success
        self._iter_stats["bob_successes"] += n_success
        self._iter_stats["bob_failures"] += n_failure

        # --- 4. Transition Logic ---
        # Paper: Alice continues for all max_goals_per_episode goals even if Bob fails some.
        # `bob_ever_failed` was preventing Alice from proposing further goals after the first failure,
        # which starves Alice of reward signal and breaks the adversarial curriculum.
        can_continue = (
            self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        )

        # Bob -> Alice (next goal)
        continue_ids = env_ids[can_continue]
        if len(continue_ids) > 0:
            self.episode_manager.transition_to_alice(continue_ids)
            reset_objects_to_fixed_safe_pose(self.env, continue_ids)
            reset_robot_joints(self.env, continue_ids)
            # Set initial_states directly from known safe reset positions to avoid
            # PhysX readback timing issues (data.root_pos_w is stale until next env.step()).
            self.episode_manager.initial_states[continue_ids] = (
                self._safe_reset_state.unsqueeze(0)
                .expand(len(continue_ids), -1)
                .clone()
            )

        # Episode End (Bob failed or max goals)
        reset_ids = env_ids[~can_continue]
        if len(reset_ids) > 0:
            succeeded = self.episode_manager.bob_success[reset_ids]
            reason = "Bob Succeeded" if succeeded.any() else "Bob Failed"
            self.episode_manager.reset_episode(reset_ids, reason=reason)

            reset_objects_to_fixed_safe_pose(self.env, reset_ids)
            reset_robot_joints(self.env, reset_ids)
            self.episode_manager.initial_states[reset_ids] = (
                self._safe_reset_state.unsqueeze(0).expand(len(reset_ids), -1).clone()
            )

        # Commit all physics writes so the next observation read reflects the reset.
        self.env.scene.write_data_to_sim()

        return success, pos_err, rot_err

    def _handle_bob_success_transition(self, env_ids: torch.Tensor):
        """Handle transition when Bob achieves early success (completion bonus triggered).

        Per rewards.txt: If Bob achieves the completion reward (+5), his turn ends immediately.
        """
        # All these envs succeeded, so check if they can continue to next goal
        can_continue = (
            self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        )

        # Environments that can continue get new Alice phase
        continue_ids = env_ids[can_continue]
        if len(continue_ids) > 0:
            self.episode_manager.transition_to_alice(continue_ids)
            reset_objects_to_fixed_safe_pose(self.env, continue_ids)
            reset_robot_joints(self.env, continue_ids)
            self.episode_manager.initial_states[continue_ids] = (
                self._safe_reset_state.unsqueeze(0)
                .expand(len(continue_ids), -1)
                .clone()
            )

        # Others reset (reached max goals with success)
        reset_ids = env_ids[~can_continue]
        if len(reset_ids) > 0:
            self.episode_manager.reset_episode(reset_ids, reason="Episode Complete")

            reset_objects_to_fixed_safe_pose(self.env, reset_ids)
            reset_robot_joints(self.env, reset_ids)
            self.episode_manager.initial_states[reset_ids] = (
                self._safe_reset_state.unsqueeze(0).expand(len(reset_ids), -1).clone()
            )

        # Commit all physics writes so the next observation read reflects the reset.
        self.env.scene.write_data_to_sim()

    def _check_bob_success(self, obs_dict: Dict, env_ids: torch.Tensor) -> torch.Tensor:
        """Check if Bob successfully reached the goal.

        Both current_state and goal_state are 12D LOCAL Euler:
        [pos(3)+euler(3)] × num_objects.
        """
        current_state = self._extract_object_states(
            obs_dict
        )  # (num_envs, num_objects * 6)
        goal_state = self.episode_manager.goal_states  # (num_envs, num_objects * 6)

        # Slice to active environments
        current_state = current_state[env_ids]
        goal_state = goal_state[env_ids]

        import math

        # _extract_object_states returns [pos(3), euler(3)] x num_objects = 6D/object
        curr_reshaped = current_state.view(-1, self.num_objects, 6)
        goal_reshaped = goal_state.view(-1, self.num_objects, 6)

        # Position Distance
        pos_curr = curr_reshaped[:, :, :3]
        pos_goal = goal_reshaped[:, :, :3]
        pos_dist = torch.norm(
            pos_curr - pos_goal, dim=-1
        )  # (batch_subset, num_objects)

        # Rotation Distance: max abs Euler diff with wraparound (matches paper Appendix A.2)
        euler_curr = curr_reshaped[:, :, 3:6]
        euler_goal = goal_reshaped[:, :, 3:6]
        euler_diff = euler_curr - euler_goal
        euler_diff = (euler_diff + math.pi) % (2 * math.pi) - math.pi
        rot_dist = euler_diff.abs().max(dim=-1)[0]  # (batch_subset, num_objects)

        # Get threshold values dynamically
        pos_threshold = self.episode_manager.pos_threshold
        rot_threshold = self.episode_manager.rot_threshold

        # Success condition per object (MUST be strictly less than threshold to leave a gap for Alice's > threshold)
        obj_success = (pos_dist < pos_threshold) & (rot_dist < rot_threshold)

        # Episode success if ALL objects are successful
        # We need to return Per-Env success boolean
        success = torch.all(obj_success, dim=-1)

        # Return metrics for debugging
        # Max error across objects
        max_pos_err, _ = torch.max(pos_dist, dim=-1)
        max_rot_err, _ = torch.max(rot_dist, dim=-1)

        return success, max_pos_err, max_rot_err

    def _extract_object_states(self, obs_dict: Dict) -> torch.Tensor:
        """Extract object states in the LOCAL frame as Euler poses (12 dims total).

        Returns [pos(3)+euler(3)] per object = 6D per object, 12D total.
        Matching the paper's rotation representation (ZYX Euler angles).
        """
        from .observations import _quat_to_euler_xyz

        # 1. Pull RAW world data from the scene
        t_pos_w = self.env.scene["target_object"].data.root_pos_w
        t_quat_w = self.env.scene["target_object"].data.root_quat_w
        c_pos_w = self.env.scene["cube"].data.root_pos_w
        c_quat_w = self.env.scene["cube"].data.root_quat_w

        env_origins = self.env.scene.env_origins

        # 2. TRANSFORM: World -> Local
        t_pos_l = t_pos_w - env_origins
        c_pos_l = c_pos_w - env_origins

        # 3. Convert quaternion -> Euler angles (paper representation)
        t_euler = _quat_to_euler_xyz(t_quat_w)  # (N, 3)
        c_euler = _quat_to_euler_xyz(c_quat_w)  # (N, 3)

        # 4. Concatenate: [Target_Pos(3), Target_Euler(3), Cube_Pos(3), Cube_Euler(3)]
        # Total = 12 dims per environment
        return torch.cat([t_pos_l, t_euler, c_pos_l, c_euler], dim=-1)

    def _get_alice_observations(self, obs_dict: Dict) -> torch.Tensor:
        """Get Alice's observations from policy group"""
        return obs_dict["alice_policy"]

    def construct_bob_observation(
        self, alice_obs: torch.Tensor, goal_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct Bob's full observation (Alice Obs + Goal States + Goal Distances).

        Assumes alice_obs and goal_states are already in the LOCAL environment frame.
        goal_states must be in Euler format: [pos(3)+euler(3)] per object = 6D each.
        """
        # 0. Ensure batch dimensions
        if alice_obs.dim() == 1:
            alice_obs = alice_obs.unsqueeze(0)
        if goal_states.dim() == 1:
            goal_states = goal_states.unsqueeze(0)

        import math

        # 1. Reshape object states from Alice's observation
        # Alice obs layout: [RobotState(7), Obj1(14), Obj2(14)] -> Total 35 dims
        current_obj_reshaped = alice_obs[:, self.robot_state_dim :].view(
            -1, self.num_objects, 14
        )

        # 2. Reshape goal states
        # goal_states layout: [Obj1_Goal(6), Obj2_Goal(6)] -> Total 12 dims (Euler format)
        goal_states_reshaped = goal_states.view(-1, self.num_objects, 6)

        # 3. Compute Position Distances (Both are in LOCAL frame)
        pos_current = current_obj_reshaped[:, :, :3]
        pos_goal = goal_states_reshaped[:, :, :3]
        pos_dist = torch.norm(
            pos_current - pos_goal, dim=-1, keepdim=True
        )  # Result: (batch, num_objs, 1)

        # 4. Compute Rotation Distances (Euler max-diff, matches paper)
        euler_current = current_obj_reshaped[:, :, 3:6]
        euler_goal = goal_states_reshaped[:, :, 3:6]
        euler_diff = euler_current - euler_goal
        euler_diff = (euler_diff + math.pi) % (2 * math.pi) - math.pi
        euler_dist = euler_diff.abs().max(dim=-1, keepdim=True)[
            0
        ]  # (batch, num_objs, 1)

        # Final concatenation for Bob:
        # The PI encoder expects objects to be contiguous.
        # Interleaved: [Obj1(14), Goal1(6), Dist1(2), Obj2(14), Goal2(6), Dist2(2)]
        distances_per_obj = torch.cat(
            [pos_dist, euler_dist], dim=-1
        )  # (batch, num_objs, 2)

        # Concatenate per object: (batch, num_objs, 22)
        bob_objs = torch.cat(
            [current_obj_reshaped, goal_states_reshaped, distances_per_obj], dim=-1
        )

        # Flatten to (batch, num_objs * 22)
        bob_objs_flat = bob_objs.view(-1, self.num_objects * 22)

        # Final output: Robot(7) + Bob_Objs(44) = 51
        robot_state = alice_obs[:, : self.robot_state_dim]
        return torch.cat([robot_state, bob_objs_flat], dim=-1)

    def _get_bob_observations(self, obs_dict: Dict) -> torch.Tensor:
        """Get Bob's observations from policy group"""
        return obs_dict["bob_policy"]

    def _get_current_observations(self, obs_dict: Dict) -> torch.Tensor:
        """Get concatenated observations for both Alice and Bob"""
        return torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)

    def _get_current_rewards(
        self,
        obs_dict: Dict,
        base_rewards: torch.Tensor,
        is_alice: torch.Tensor,
        is_bob: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get rewards based on the phase state at START of step (before transitions).

        Args:
            obs_dict: Observation dictionary
            base_rewards: Rewards from RewardManager (dt-scaled)
            is_alice: Boolean mask of envs that were in Alice phase at start of step
            is_bob: Boolean mask of envs that were in Bob phase at start of step

        Alice: NO step rewards during phase - only gets outcome rewards at goal validation
        Bob: Gets sparse integer rewards computed directly (no dt-scaling)

        This ensures Alice doesn't accumulate massive penalties and Bob gets clean +1/-1/+5 rewards.
        """
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Alice: NO rewards during her phase
        # She only receives rewards at goal validation in _handle_alice_completion:
        # - Valid goal bonus: +1
        # - Out-of-zone penalty: -3
        # - Outcome reward (if Bob fails): +5 (applied in train.py)
        # This prevents per-step penalty accumulation (×50 steps = massive penalties)
        #
        # UPDATE: We MUST pass through base_rewards (penalties) for Alice to learn!
        # Otherwise she gets 0.0 for OOB/Collisions and never learns to avoid them.
        if is_alice.any():
            rewards[is_alice] = base_rewards[is_alice]

        # Track which envs just achieved completion (for early termination)
        bob_achieved_completion = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Bob: Compute sparse rewards directly (bypasses dt-scaling)
        if is_bob.any():
            bob_rewards, achieved_completion = self._compute_bob_sparse_rewards(
                obs_dict, action
            )
            rewards[is_bob] = bob_rewards[is_bob]
            bob_achieved_completion[is_bob] = achieved_completion[is_bob]

        return rewards, bob_achieved_completion

    def _compute_bob_sparse_rewards(
        self, obs_dict: Dict, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Bob's sparse rewards without dt-scaling, combined with dense and smoothing shaping.

        IMPORTANT: This implementation follows the Asymmetric Self-Play (ASP) paper's
        philosophy of SPARSE REWARDS ONLY. No dense distance-based rewards are used.

        Why Sparse Rewards?
        -------------------
        The ASP paper relies on the adversarial curriculum (Alice vs Bob) to drive
        learning, not hand-crafted reward shaping. Dense rewards (e.g., -distance_to_goal)
        would "short-circuit" the curriculum by guiding Bob too much, preventing Alice
        from finding true blind spots in Bob's policy.

        Sparse rewards force Bob to discover solutions without hints, making Alice's
        role as adversary more meaningful and enabling true unsupervised goal discovery.

        Reward Structure (per ASP paper):
        ----------------------------------
        - +1: Per object successfully placed at goal (threshold crossed once)
        - -1: Object leaves goal threshold (dropped, once)
        - +5: Completion bonus when ALL objects FIRST reach goal simultaneously
        - Turn ends immediately when completion bonus is achieved

        All rewards are INTEGER values (not dt-scaled), ensuring clean RL signals.

        Returns:
            Tuple of:
            - Reward tensor (num_envs,) with integer values
            - Boolean tensor (num_envs,) indicating which envs just achieved completion
        """
        from .observations import goal_distance
        from isaaclab.managers import SceneEntityCfg

        rewards = torch.zeros(self.num_envs, device=self.device)

        # # Debug: print goal_states BEFORE any computation
        # if self.episode_manager.goal_states is not None:
        #     print(f"[DEBUG bob_sparse entry] goal_states[0,0:3]={self.episode_manager.goal_states[0, 0:3].tolist()}", flush=True)

        # Get goal distances for both objects
        # target_object distances
        target_dists = goal_distance(
            self.env, object_cfg=SceneEntityCfg("target_object")
        )
        # cube distances
        cube_dists = goal_distance(self.env, object_cfg=SceneEntityCfg("cube"))

        # Concatenate: (num_envs, 2) + (num_envs, 2) -> (num_envs, 4)
        # Each is [pos_dist, rot_dist] so we have [target_pos, target_rot, cube_pos, cube_rot]
        all_dists = torch.cat([target_dists, cube_dists], dim=1)

        batch_size = all_dists.shape[0]
        num_objects = 2  # target_object and cube

        # Reshape: (num_envs, 2, 2) -> [obj_idx, (pos, rot)]
        dists = all_dists.view(batch_size, num_objects, 2)

        pos_dists = dists[..., 0]  # (num_envs, num_objects)
        rot_dists = dists[..., 1]

        # Success thresholds
        pos_threshold = self.episode_manager.pos_threshold
        rot_threshold = self.episode_manager.rot_threshold

        # Per-object success check
        obj_success = (pos_dists < pos_threshold) & (rot_dists < rot_threshold)

        # Get previous success state
        prev_success = self.episode_manager.prev_obj_success
        if prev_success is None:
            # First step: initialize with correct shape
            prev_success = torch.zeros(
                batch_size, num_objects, dtype=torch.bool, device=self.device
            )
            self.episode_manager.prev_obj_success = prev_success

        # Ensure shapes match (in case of env count mismatch)
        if prev_success.shape != obj_success.shape:
            prev_success = torch.zeros_like(obj_success, dtype=torch.bool)
            self.episode_manager.prev_obj_success = (
                prev_success  # Persist corrected state
            )

        # Prevent spurious rewards on the very first step of Bob's phase.
        # If an object (like the secondary cube) is untouched by Alice, its start position is its goal position.
        # It will be evaluated as "True" on step 1, but we don't want to give Bob a free +1 reward for it.
        # Restrict to Bob-phase envs only (Alice-phase envs at step 1 must not interfere).
        bob_phase_steps = self.episode_manager.phase_step
        is_bob_phase = self.episode_manager.is_bob_phase()
        is_first_step = (bob_phase_steps == 1) & is_bob_phase
        if is_first_step.any():
            first_step_mask = is_first_step.unsqueeze(1).expand_as(prev_success)
            prev_success = torch.where(first_step_mask, obj_success, prev_success)

        # Calculate delta rewards: +1 for newly placed, -1 for dropped
        # False -> True: +1
        # True -> False: -1
        # Same: 0
        # NOTE: delta.sum() correctly tracks NET progress toward goal
        # Example: obj1 enters goal (+1) + obj2 leaves goal (-1) = 0 total
        # This is correct per paper: Bob gets reward for progress, penalty for regress
        delta = obj_success.float() - prev_success.float()
        step_rewards = delta.sum(dim=1)  # Sum across objects

        # Completion bonus: +5 if ALL objects at goal AND bonus not yet given
        all_success = torch.all(obj_success, dim=1)
        completion_not_given = ~self.episode_manager.completion_given

        # Only allow completion if this is not the transition step (step 0) AND env is in Bob phase
        should_give_completion = (
            all_success & completion_not_given & (bob_phase_steps > 0) & is_bob_phase
        )

        # Per-env reward event logging (after delta and completion are known)
        _OBJ_NAMES = ["target", "cube"]

        completion_bonus = torch.where(
            should_give_completion,
            torch.ones(batch_size, device=self.device) * 5.0,
            torch.zeros(batch_size, device=self.device),
        )

        # Mark completion as given for environments that just received it
        self.episode_manager.completion_given = (
            self.episode_manager.completion_given | should_give_completion
        )

        rewards = step_rewards + completion_bonus

        # Per-env reward event logging — print every non-zero Bob reward
        env_origins = self.env.scene.env_origins
        for env_id in torch.where(rewards != 0)[0]:
            eid = env_id.item()
            step = bob_phase_steps[eid].item()
            rew = rewards[eid].item()
            if should_give_completion[eid]:
                t_curr_l = (
                    self.env.scene["target_object"].data.root_pos_w[eid]
                    - env_origins[eid]
                ).tolist()
                t_goal = self.episode_manager.goal_states[eid, 0:3].tolist()
                c_curr_l = (
                    self.env.scene["cube"].data.root_pos_w[eid] - env_origins[eid]
                ).tolist()
                c_goal = self.episode_manager.goal_states[eid, 6:9].tolist()
                print(
                    f"  [BobRew] Env {eid} step={step}: {rew:+.0f} COMPLETION | "
                    f"target dist={pos_dists[eid,0]:.3f} "
                    f"(curr={[round(x,3) for x in t_curr_l]} goal={[round(x,3) for x in t_goal]}) | "
                    f"cube dist={pos_dists[eid,1]:.3f} "
                    f"(curr={[round(x,3) for x in c_curr_l]} goal={[round(x,3) for x in c_goal]})",
                    flush=True,
                )
            else:
                parts = []
                for obj_idx, obj_name in enumerate(_OBJ_NAMES):
                    d = delta[eid, obj_idx].item()
                    if d > 0:
                        parts.append(f"+1 {obj_name} at goal (dist={pos_dists[eid,obj_idx]:.3f}m)")
                    elif d < 0:
                        parts.append(f"-1 {obj_name} left goal (dist={pos_dists[eid,obj_idx]:.3f}m)")
                print(
                    f"  [BobRew] Env {eid} step={step}: {rew:+.0f} | {' | '.join(parts)}",
                    flush=True,
                )

        # Update prev_obj_success only for Bob-phase envs to avoid Alice-phase
        # envs carrying stale success state into their first Bob step.
        new_prev = (
            self.episode_manager.prev_obj_success.clone()
            if self.episode_manager.prev_obj_success is not None
            else obj_success.clone()
        )
        new_prev[is_bob_phase] = obj_success[is_bob_phase]
        self.episode_manager.prev_obj_success = new_prev

        # Return rewards and which envs just achieved completion
        return rewards, should_give_completion

    def close(self):
        """Close wrapped environment"""
        self.env.close()
