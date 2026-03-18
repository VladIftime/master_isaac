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
        
        # Robot state dimensions (EE pose: 7 + 1 gripper)
        self.robot_state_dim = 8
        
        # Fetch the dimensions directly from the observation manager
        alice_dim_info = self.env.unwrapped.observation_manager.group_obs_dim["alice_policy"]
        bob_dim_info = self.env.unwrapped.observation_manager.group_obs_dim["bob_policy"]
        
        # Extract the integer dimension (handling whether Isaac Lab returns a tuple or raw int)
        self.alice_obs_dim = alice_dim_info[0] if isinstance(alice_dim_info, (tuple, list)) else alice_dim_info
        self.bob_obs_dim = bob_dim_info[0] if isinstance(bob_dim_info, (tuple, list)) else bob_dim_info
        
        print(f"[Wrapper] Dynamically set Alice Obs Dim: {self.alice_obs_dim}, Bob Obs Dim: {self.bob_obs_dim}")

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
            low=-np.inf, high=np.inf, shape=(self.alice_obs_dim + self.bob_obs_dim,), dtype=np.float32
        )
        self.state_space = self.observation_space
        
        # Handle Action Space (Strip batch dimension (num_envs) for PPO compatibility)
        # env.action_space is (num_envs, action_dim)
        if len(env.action_space.shape) > 1:
            self.action_space = gym.spaces.Box(
                low=env.action_space.low[0], 
                high=env.action_space.high[0], 
                shape=env.action_space.shape[1:], 
                dtype=env.action_space.dtype
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
        
        print(f"[AsyncDualPlayEnvWrapper] Initialized (task-space mode)")
        print(f"  Alice obs: {self.alice_obs_dim}, Bob obs: {self.bob_obs_dim}")
        print(f"  Robot state: {self.robot_state_dim} (EE pose 7 + gripper 1)")
        
    @property
    def num_envs(self):
        return self.env.num_envs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict]:
        """Reset environment and episode manager"""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        
        # Reset episode manager
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.episode_manager.reset_episode(env_ids, reason="Global Manual Reset")
        self.delayed_alice_reward[env_ids] = 0.0
        
        # Extract and store initial object states
        initial_state = self._extract_object_states(obs_dict)
        self.episode_manager.store_initial_state(initial_state)
        
        # Return concatenated observations so train.py can slice them
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        return obs, info
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
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
        
        # Sync EpisodeManager with Env Resets
        dones = terminated | truncated
        if dones.any():
            reset_ids = torch.where(dones)[0]
            
            # NOTE: extras["log"] contains aggregated scalar means for TensorBoard,
            # NOT per-env tensors. We use `terminated[env_id]` as the OOB proxy:
            # terminated=True means physics termination (OOB/collision),
            # truncated=True (without terminated) means timeout — no Alice penalty.
            is_alice = self.episode_manager.is_alice_phase()

            # the active boolean tensors for each termination condition.
            term_dones = self.env.termination_manager.term_dones
            
            for env_id in reset_ids:
                # Skip timeout resets — only penalize hard terminations
                if not terminated[env_id]:
                    continue
                
                # Determine specific reason from term_dones
                reasons = [name for name, val in term_dones.items() if val[env_id]]
                reason = " | ".join(reasons) if reasons else "OOB/Termination"
                
                if is_alice[env_id]:
                    if not hasattr(self, "_early_alice_failures"):
                        self._early_alice_failures = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._early_alice_failures[env_id] = True
                    self.delayed_alice_reward[env_id] = -3.0
                    print(f"[Reset] Env {env_id.item()}: Alice FAILED early ({reason}) | Penalty: -3.0", flush=True)

                self.episode_manager.reset_episode(env_id.unsqueeze(0), reason=reason)
        
        # Capture phase state BEFORE transitions
        is_alice_before = self.episode_manager.is_alice_phase().clone()
        is_bob_before = self.episode_manager.is_bob_phase().clone()
        
        # Advance episode manager
        phase_info = self.episode_manager.step()
        any_reset = False
        
        # Prepare extras for Alice's total reward
        extras["alice_total_reward"] = torch.zeros(self.num_envs, device=self.device)
        extras["alice_failed_this_step"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Pull in early failures if any
        if hasattr(self, "_early_alice_failures"):
            extras["alice_failed_this_step"] |= self._early_alice_failures
            extras["alice_total_reward"] += torch.where(self._early_alice_failures, self.delayed_alice_reward, torch.zeros_like(self.delayed_alice_reward))
            early_fail_ids = torch.where(self._early_alice_failures)[0]
            if len(early_fail_ids) > 0:
                self.delayed_alice_reward[early_fail_ids] = 0.0
            self._early_alice_failures.fill_(False)

        # Handle Alice phase completion
        if phase_info["alice_done"].any():
            alice_done_ids = torch.where(phase_info["alice_done"])[0]
            valid, invalid_ids = self._handle_alice_completion(obs_dict, alice_done_ids)
            
            if len(invalid_ids) > 0:
                  extras["alice_total_reward"][invalid_ids] = self.delayed_alice_reward[invalid_ids]
                  extras["alice_failed_this_step"][invalid_ids] = True
                  terminated[invalid_ids] = True
                  self.delayed_alice_reward[invalid_ids] = 0.0

            any_reset = True
        
        # Track Bob's success
        step_bob_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        step_bob_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        step_pos_err = torch.zeros(self.num_envs, device=self.device)
        step_rot_err = torch.zeros(self.num_envs, device=self.device)

        # Handle Bob phase completion
        if phase_info["bob_done"].any():
            bob_done_ids = torch.where(phase_info["bob_done"])[0]
            success, pos_err, rot_err = self._handle_bob_completion(obs_dict, bob_done_ids)
            step_bob_success[bob_done_ids] = success
            step_bob_done[bob_done_ids] = True
            step_pos_err[bob_done_ids] = pos_err
            step_rot_err[bob_done_ids] = rot_err
            
            extras["alice_total_reward"][bob_done_ids] = self.delayed_alice_reward[bob_done_ids]
            self.delayed_alice_reward[bob_done_ids] = 0.0 
            any_reset = True
        
        if any_reset:
            obs_dict = self.env.observation_manager.compute()
        
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_rewards, bob_achieved_completion = self._get_current_rewards(obs_dict, rewards, is_alice_before, is_bob_before, action)
        
        # Handle Bob's EARLY SUCCESS
        if bob_achieved_completion.any():
            completion_ids = torch.where(bob_achieved_completion)[0]
            self.episode_manager.bob_success[completion_ids] = True
            step_bob_success[completion_ids] = True
            step_bob_done[completion_ids] = True
            self.episode_manager.completion_given[completion_ids] = True
            
            success, pos_err, rot_err = self._check_bob_success(obs_dict, completion_ids)
            step_pos_err[completion_ids] = pos_err
            step_rot_err[completion_ids] = rot_err
            
            completion_steps = self.episode_manager.phase_step[completion_ids].clone()
            completion_goals = self.episode_manager.goal_count[completion_ids].clone()
            
            self._handle_bob_success_transition(completion_ids)
            
            extras["alice_total_reward"][completion_ids] = self.delayed_alice_reward[completion_ids]
            self.delayed_alice_reward[completion_ids] = 0.0

            for idx, env_id in enumerate(completion_ids):
                print(f"[BobComplete] Env {env_id.item()}: Goal {completion_goals[idx].item()}/5 @ step {completion_steps[idx].item()}/600")
            
            can_continue = (self.episode_manager.goal_count[completion_ids] < self.episode_manager.max_goals)
            terminated[completion_ids] = ~can_continue
            any_reset = True
        
        if any_reset:
            obs_dict = self.env.observation_manager.compute()
            current_obs = self._get_current_observations(obs_dict)
            
            new_alice_mask = self.episode_manager.is_alice_phase() & (self.episode_manager.phase_step == 0)
            if new_alice_mask.any() and self.episode_manager.initial_states is not None:
                new_alice_ids = torch.where(new_alice_mask)[0]
                fresh_states = self._extract_object_states(obs_dict)
                self.episode_manager.initial_states[new_alice_ids] = fresh_states[new_alice_ids].clone()
        
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
                    self.episode_manager.max_contact_force = torch.where(bob_mask, torch.max(self.episode_manager.max_contact_force, max_f), self.episode_manager.max_contact_force)
                    
        extras["episode_manager"] = {
            "phase": self.episode_manager.current_phase.clone(),
            "goal_count": self.episode_manager.goal_count.clone(),
            "bob_success": self.episode_manager.bob_success.clone(),
            "bob_success_this_step": step_bob_success,
            "bob_done_this_step": step_bob_done,
            "bob_pos_err": step_pos_err,
            "bob_rot_err": step_rot_err,
            "goal_valid": self.episode_manager.goal_valid.clone(), 
            "goal_states": self.episode_manager.goal_states.clone() if self.episode_manager.goal_states is not None else torch.zeros((self.num_envs, 14), device=self.device),
            "max_contact_force": self.episode_manager.max_contact_force.clone(),
        }
        
        extras["goal_valid"] = self.episode_manager.goal_valid.clone()
        extras["bob_success"] = self.episode_manager.bob_success.clone()
        self.previous_actions = action.clone()
        
        return obs, current_rewards, terminated, truncated, extras
    
    def _handle_alice_completion(self, obs_dict: Dict, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle completion of Alice's phase"""
        goal_state = self._extract_object_states(obs_dict)
        initial_state = self.episode_manager.initial_states
        alice_pos_req = 0.04 # Threshold for Alice to get reward. 0.04m (4cm)
        alice_rot_req = 0.10 # Reduced for easier exploration.
        
        # New Validator Return: (valid, reward, reason)
        valid, val_reward, reasons = validate_goal(
            initial_state[env_ids],
            goal_state[env_ids],
            self.table_bounds,
            self.placement_bounds,
            pos_threshold=alice_pos_req,
            rot_threshold=alice_rot_req
        )
        
        # Store immediate validation reward (+1.0, -3.0, or 0.0)
        self.delayed_alice_reward[env_ids] = val_reward
        
        # Snapshot for logs
        start_pos = initial_state[env_ids][:, 0:3] 
        final_pos = goal_state[env_ids][:, 0:3]
        dist_moved = torch.norm(final_pos - start_pos, dim=-1)
        
        for i, env_id in enumerate(env_ids):
            print(f"[Alice Reward] Env {env_id.item()}: {reasons[i]} | Moved: {dist_moved[i]:.3f}m", flush=True)
        
        # Transition/Storage logic
        # --- CHANGED: 17 changed to 15 to match Isaac Lab's actual object state output ---
        goal_state_storage = goal_state.view(-1, 2, 15)[:, :, :7].reshape(-1, 14)
        self.episode_manager.store_goal_state(goal_state_storage, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)
        
        # In this implementation, OOB goals (val_reward == -3.0) are considered INVALID for transition
        # This aligns with Step 1: "if oob -> return False, -3.0"
        successful_goal = valid
        valid_env_ids = env_ids[successful_goal]
        if len(valid_env_ids) > 0:
            self.episode_manager.transition_to_bob(valid_env_ids)
            
            # Reset objects to start states for Bob
            start_states = self.episode_manager.initial_states[valid_env_ids]
            env_origins = self.env.scene.env_origins[valid_env_ids]
            
            pos1_global = start_states[:, 0:3] + env_origins
            self.env.scene['target_object'].write_root_pose_to_sim(torch.cat([pos1_global, start_states[:, 3:7]], dim=-1), env_ids=valid_env_ids)
            self.env.scene['target_object'].write_root_velocity_to_sim(torch.zeros(len(valid_env_ids), 6, device=self.device), env_ids=valid_env_ids)
            
            pos2_global = start_states[:, 15:18] + env_origins
            self.env.scene['cube'].write_root_pose_to_sim(torch.cat([pos2_global, start_states[:, 18:22]], dim=-1), env_ids=valid_env_ids)
            self.env.scene['cube'].write_root_velocity_to_sim(torch.zeros(len(valid_env_ids), 6, device=self.device), env_ids=valid_env_ids)
            
            reset_robot_joints(self.env, valid_env_ids)
            print(f"[Reset] Alice->Bob Transition: Resetting Objects (Initial) & Robot (Default) for {len(valid_env_ids)} envs", flush=True)
            self.env.scene.write_data_to_sim()
        
        invalid_env_ids = env_ids[~successful_goal]
        if len(invalid_env_ids) > 0:
            self.episode_manager.reset_episode(invalid_env_ids, reason="Alice Invalid Goal")
            reset_objects_to_fixed_safe_pose(self.env, invalid_env_ids)
            reset_robot_joints(self.env, invalid_env_ids)
            self.env.scene.write_data_to_sim()
            
        return valid, invalid_env_ids
    
    
    def _handle_bob_completion(self, obs_dict: Dict, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle completion of Bob's phase"""
        # --- 1. Success check ---
        success, pos_err, rot_err = self._check_bob_success(obs_dict, env_ids)
        self.episode_manager.mark_bob_success(env_ids, success)

        # --- 2. Alice Outcome Reward (+5 if Bob fails, 0 if he succeeds) ---
        outcome_rewards = torch.where(success, 
                                     torch.tensor(reward_utils.ALICE_BOB_SUCCESS_REWARD, device=self.device), 
                                     torch.tensor(reward_utils.ALICE_BOB_FAIL_REWARD, device=self.device))
        self.delayed_alice_reward[env_ids] += outcome_rewards

        # --- 3. Logging ---
        if len(env_ids) > 0:
            first_env = env_ids[0].item()
            steps_used = self.episode_manager.phase_step[first_env].item()
            status = "SUCCESS" if success[0].item() else "FAILURE"
            print(f"[Phase] Env {first_env}: Bob {status} | Steps: {steps_used}/800 | Alice Outcome Reward: {outcome_rewards[0].item():.1f}", flush=True)
        
        # --- 4. Transition Logic ---
        # Paper: Alice continues for all max_goals_per_episode goals even if Bob fails some.
        # `bob_ever_failed` was preventing Alice from proposing further goals after the first failure,
        # which starves Alice of reward signal and breaks the adversarial curriculum.
        can_continue = self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        
        # Bob -> Alice (next goal)
        continue_ids = env_ids[can_continue]
        if len(continue_ids) > 0:
            print(f"[Phase] Bob->Alice Transition for {len(continue_ids)} envs")
            self.episode_manager.transition_to_alice(continue_ids)
            reset_objects_to_fixed_safe_pose(self.env, continue_ids)
            reset_robot_joints(self.env, continue_ids)
        
        # Episode End (Bob failed or max goals)
        reset_ids = env_ids[~can_continue]
        if len(reset_ids) > 0:
            succeeded = self.episode_manager.bob_success[reset_ids]
            reason = "Bob Succeeded" if succeeded.any() else "Bob Failed"
            self.episode_manager.reset_episode(reset_ids, reason=reason)
            
            reset_objects_to_fixed_safe_pose(self.env, reset_ids)
            reset_robot_joints(self.env, reset_ids)

        return success, pos_err, rot_err
    
    def _handle_bob_success_transition(self, env_ids: torch.Tensor):
        """Handle transition when Bob achieves early success (completion bonus triggered).
        
        Per rewards.txt: If Bob achieves the completion reward (+5), his turn ends immediately.
        """
        # All these envs succeeded, so check if they can continue to next goal
        can_continue = self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        
        # Environments that can continue get new Alice phase
        continue_ids = env_ids[can_continue]
        if len(continue_ids) > 0:
            print(f"[Phase] Bob->Alice Transition (Early Success) for {len(continue_ids)} envs")
            self.episode_manager.transition_to_alice(continue_ids)
            # FIX: Ensure Alice starts from clean state too!
            reset_objects_to_fixed_safe_pose(self.env, continue_ids)
            reset_robot_joints(self.env, continue_ids)
        
        # Others reset (reached max goals with success)
        reset_ids = env_ids[~can_continue]
        if len(reset_ids) > 0:
            self.episode_manager.reset_episode(reset_ids, reason="Episode Complete")

            reset_objects_to_fixed_safe_pose(self.env, reset_ids)
            reset_robot_joints(self.env, reset_ids)
    
    def _check_bob_success(self, obs_dict: Dict, env_ids: torch.Tensor) -> torch.Tensor:
        """Check if Bob successfully reached the goal"""
        # Get current object states
        current_state = self._extract_object_states(obs_dict)
        
        # Get goal states
        goal_state = self.episode_manager.goal_states
        
        # Compute distances
        pos_current = current_state[env_ids, :3]  # Simplified: first object only
        pos_goal = goal_state[env_ids, :3]
        

        
        # Need to reshape for rotation calc
        # current_state: (num_envs, num_objects * 7)
        # goal_state: (num_envs, num_objects * 7)
        
        # Filter for the relevant environments FIRST
        current_state = current_state[env_ids]
        goal_state = goal_state[env_ids]
        
        curr_reshaped = current_state.view(-1, self.num_objects, 15)
        goal_reshaped = goal_state.view(-1, self.num_objects, 7)
        
        # Position Distance
        pos_curr = curr_reshaped[:, :, :3]
        pos_goal = goal_reshaped[:, :, :3]
        pos_dist = torch.norm(pos_curr - pos_goal, dim=-1) # (batch_subset, num_objects)
        
        # Rotation Distance
        quat_curr = curr_reshaped[:, :, 3:7]
        quat_goal = goal_reshaped[:, :, 3:7]
        
        # Handle quaternion dot product (ensure positive)
        quat_dot = torch.abs(torch.sum(quat_curr * quat_goal, dim=-1))
        rot_dist = 1.0 - quat_dot # (batch_subset, num_objects)
        
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
        """Extract object states from observation dictionary.
        
        Returns:
            Tensor (num_envs, num_objects * 15) with states for target_object and cube
        """
        # Params matching single-arm env config
        params = {
            "gripper_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "contact_cfg": SceneEntityCfg("contact_forces"),
        }

        # Get states for each tracked object
        target_state = observations.object_states(self.env, SceneEntityCfg("target_object"), **params)  # (num_envs, 15)
        cube_state = observations.object_states(self.env, SceneEntityCfg("cube"), **params)  # (num_envs, 15)
        
        # Concatenate: (num_envs, 30)
        return torch.cat([target_state, cube_state], dim=1)
    
    # ... Rest of file is unchanged ...
    def _get_alice_observations(self, obs_dict: Dict) -> torch.Tensor:
        """Get Alice's observations from policy group"""
        return obs_dict["alice_policy"]
    
    def construct_bob_observation(self, alice_obs: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        """
        Construct Bob's full observation from Alice's observation and a Goal state.
        
        Args:
            alice_obs: Alice's observation (N, 48) or similar
            goal_states: Goal states (N, 14) [pos(3), rot(4) per object]
            
        Returns:
            Bob's observation (N, 66)
        """
        # Ensure shapes
        if alice_obs.dim() == 1:
            alice_obs = alice_obs.unsqueeze(0)
        if goal_states.dim() == 1:
            goal_states = goal_states.unsqueeze(0)
            
        # 1. Alice observations (28 or 48 depending on config, here assumed 48 based on self.alice_obs_dim)
        # Note: In __init__, alice_obs_dim is 48.
        # But _get_alice_observations returns (N, 28) in the code I read earlier?
        # Let's check _get_alice_observations:
        # return torch.cat([joints, grippers, obj_states], dim=-1)
        # joints (12) + grippers (2) + obj_states (34) = 48. Correct.
        
        # 2. Goal states (14)
        
        # 3. Goal distances (2 * 2 = 4)
        # Compute distances pairwise for each object
        
        # Identify where object states are in Alice obs
        # Alice obs: [Joints(12), Gripper(2), Obj1(15), Obj2(15)]
        # Object states start at index robot_state_dim
        
        # Reshape for computation: (batch, num_objects, 15)
        # We use self.num_objects which is 2
        
        current_obj_reshaped = alice_obs[:, self.robot_state_dim:].view(-1, self.num_objects, 15)
        goal_states_reshaped = goal_states.view(-1, self.num_objects, 7)
        
        # Position distance
        pos_current = current_obj_reshaped[:, :, :3]
        pos_goal = goal_states_reshaped[:, :, :3]
        pos_dist = torch.norm(pos_current - pos_goal, dim=-1, keepdim=True) # (batch, num_objs, 1)
        
        # Rotation distance
        quat_current = current_obj_reshaped[:, :, 3:7]
        quat_goal = goal_states_reshaped[:, :, 3:7]
        # Ensure dot is positive (quaternion double cover)
        quat_dot = torch.abs(torch.sum(quat_current * quat_goal, dim=-1, keepdim=True))
        quat_dist = 1.0 - quat_dot # (batch, num_objs, 1)
        
        distances = torch.cat([pos_dist, quat_dist], dim=-1) # (batch, num_objs, 2)
        distances_flat = distances.view(-1, self.num_objects * 2)
        
        # Concatenate ALL: 38 + 14 + 4 = 56
        return torch.cat([alice_obs, goal_states, distances_flat], dim=-1)

    def _get_bob_observations(self, obs_dict: Dict) -> torch.Tensor:
        """Get Bob's observations from policy group"""
        return obs_dict["bob_policy"]
    
    def _get_current_observations(self, obs_dict: Dict) -> torch.Tensor:
        """Get concatenated observations for both Alice and Bob"""
        return torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
    
    def _get_current_rewards(self, obs_dict: Dict, base_rewards: torch.Tensor, 
                              is_alice: torch.Tensor, is_bob: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        bob_achieved_completion = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Bob: Compute sparse rewards directly (bypasses dt-scaling)
        if is_bob.any():
            bob_rewards, achieved_completion = self._compute_bob_sparse_rewards(obs_dict, action)
            rewards[is_bob] = bob_rewards[is_bob]
            bob_achieved_completion[is_bob] = achieved_completion[is_bob]
        
        return rewards, bob_achieved_completion
    
    def _compute_bob_sparse_rewards(self, obs_dict: Dict, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # Get goal distances for both objects
        # target_object distances
        target_dists = goal_distance(self.env, object_cfg=SceneEntityCfg("target_object"))
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
            prev_success = torch.zeros(batch_size, num_objects, dtype=torch.bool, device=self.device)
            self.episode_manager.prev_obj_success = prev_success
        
        # Ensure shapes match (in case of env count mismatch)
        if prev_success.shape != obj_success.shape:
            prev_success = torch.zeros_like(obj_success, dtype=torch.bool)
            self.episode_manager.prev_obj_success = prev_success  # Persist corrected state
            
        # Prevent spurious rewards on the very first step of Bob's phase.
        # If an object (like the secondary cube) is untouched by Alice, its start position is its goal position.
        # It will be evaluated as "True" on step 1, but we don't want to give Bob a free +1 reward for it.
        bob_phase_steps = self.episode_manager.phase_step
        is_first_step = bob_phase_steps == 1
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
        
        # Prevent success on the very first step of Bob's phase (stale observations)
        # episode_manager.phase_step is 0 immediately after transition
        bob_phase_steps = self.episode_manager.phase_step
        
        # Only allow completion if this is not the transition step (step 0)
        should_give_completion = all_success & completion_not_given & (bob_phase_steps > 0)
        
        completion_bonus = torch.where(
            should_give_completion,
            torch.ones(batch_size, device=self.device) * 5.0,
            torch.zeros(batch_size, device=self.device)
        )
        
        
        
        # Mark completion as given for environments that just received it
        self.episode_manager.completion_given = self.episode_manager.completion_given | should_give_completion
        
        rewards = step_rewards + completion_bonus
        
        # Total Bob Reward (Sparse Only)
        total_rewards = rewards
        
        # Update state for next step
        self.episode_manager.prev_obj_success = obj_success.clone()
        
        # Return rewards and which envs just achieved completion
        return total_rewards, should_give_completion
    
    def close(self):
        """Close wrapped environment"""
        self.env.close()