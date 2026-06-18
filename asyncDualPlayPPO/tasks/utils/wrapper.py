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
import math
from typing import Optional, Dict, Any, Tuple
import gymnasium as gym




from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ...utils.episode_manager import EpisodeManager, Phase
from ...utils.goal_validator import validate_goal
from . import observations
from . import rewards as reward_utils
from .events import reset_objects_to_random_safe_pose, reset_robot_joints


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
        shaping_gamma: float = 0.99,
        shaping_coef: float = 1.0,
        profiler=None,
    ):
        self.env = env
        self.device = device
        self.num_objects = num_objects
        self.arm_config = arm_config
        # Potential-based shaping: F(s,a,s') = γ·Φ(s') − Φ(s), Φ(s) = −Σ pos_dist_i(s)
        # shaping_gamma must match PPO gamma for the theorem to hold exactly.
        self.shaping_gamma = shaping_gamma
        self.shaping_coef = shaping_coef
        self._profiler = profiler
        # Stores pos_dists from the *previous* Bob step per env; None until Bob phase starts.
        self.prev_pos_dists: Optional[torch.Tensor] = None

        # Safe reset positions (LOCAL frame) for initial_states tracking.
        # Objects are physically spawned at z=0.05 (events.py) but gravity settles
        # them to z≈0.023 on the table surface within the first few sim steps.
        # We record the SETTLED height here so the movement metric isn't inflated
        # by the ~2.7cm gravity drop. This preserves Z in the distance check so
        # Alice is still rewarded for lifting objects in the future.
        _id_euler = torch.tensor(
            [0.0, 0.0, 0.0], device=device
        )  # identity rotation (ZYX Euler)
        _t_pos = torch.tensor([0.0, 0.5, 0.023], device=device)
        _t2_pos = torch.tensor([-0.25, 0.7, 0.023], device=device)
        # Shape (6,) for 1 object, (12,) for 2 objects.
        # Layout: [t_pos(3), t_euler(3)] or [t_pos(3), t_euler(3), t2_pos(3), t2_euler(3)]
        if num_objects == 1:
            self._safe_reset_state = torch.cat([_t_pos, _id_euler])
        else:
            self._safe_reset_state = torch.cat([_t_pos, _id_euler, _t2_pos, _id_euler])

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
            "x_range": (-0.70, 0.70),
            "y_range": (-0.10, 0.90),
            "z_min": -0.2,
            "z_max": 0.15,
        }
        self.placement_bounds = {
            "x_range": (-0.50, 0.50),
            "y_range": (0.25, 0.70),
        }

        # Buffer to hold Alice's rewards until her cycle ends (either Alice failed or Bob finished)
        self.delayed_alice_reward = torch.zeros(env.num_envs, device=self.device)

        # Potential-based dense reward state for Alice.
        # alice_start_pos_xy: world-frame XY of target_object at start of Alice's phase.
        # alice_prev_dist: distance from start at the previous step (Φ(s_{t-1})).
        # _alice_phase_initialized: True once start pos has been captured for this phase.
        self.alice_start_pos_xy = torch.zeros(env.num_envs, 2, device=self.device)
        self.alice_prev_dist = torch.zeros(env.num_envs, device=self.device)
        self._alice_phase_initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=self.device
        )

        # Phase-end progress reward for Bob (episodic, not per-step — no GAE noise).
        # Captured on Bob's first step, paid on Bob's last step.
        # r_progress = clamp(w_pos·Δpos/init_pos + w_rot·Δrot/init_rot, −1, +1)
        self.bob_init_pos_err = torch.zeros(self.num_envs, device=self.device)
        self.bob_init_rot_err = torch.zeros(self.num_envs, device=self.device)
        self._bob_progress_captured = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Phase-end progress reward logging (printed every _BOB_PROGRESS_LOG_INTERVAL completions)
        self._bob_progress_log = []          # list of (progress, pos_progress, rot_progress) floats
        self._BOB_PROGRESS_LOG_INTERVAL = 50  # print summary every N Bob completions



        # Per-iteration aggregate stats (reset each iteration via reset_iter_stats())
        self._iter_stats = self._make_iter_stats()

        print(f"[AsyncDualPlayEnvWrapper] Initialized (task-space mode)")
        print(f"  Alice obs: {self.alice_obs_dim}, Bob obs: {self.bob_obs_dim}")
        print(f"  Robot state: {self.robot_state_dim} (EE pose 6 euler + gripper 1)")

    def set_table_color(self, env_ids: torch.Tensor, color: Tuple[float, float, float]):
        """Update table color for specific environments (non-headless only)."""
        if not self.env.sim.has_gui():
            return
        
        import omni.usd
        from pxr import Gf
        
        stage = omni.usd.get_context().get_stage()
        for i in env_ids.tolist():
            # Path for Isaac Lab CuboidSpawner + PreviewSurface material
            # Standard structure: {prim_path}/VisualMaterial/Shader
            shader_path = f"/World/envs/env_{i}/Table/VisualMaterial/Shader"
            prim = stage.GetPrimAtPath(shader_path)
            if prim.IsValid():
                # Set the diffuseColor input on the shader (UsdPreviewSurface)
                color_vec = Gf.Vec3f(color[0], color[1], color[2])
                attr = prim.GetAttribute("inputs:diffuseColor")
                if attr.IsValid():
                    attr.Set(color_vec)
                else:
                    print(f"[Visuals] Warning: Found prim at {shader_path} but no 'inputs:diffuseColor' attribute.")
            else:
                # Try fallback path for newer Isaac Lab versions
                fallback_path = f"/World/envs/env_{i}/Table/Visuals/mesh/material/Shader"
                prim = stage.GetPrimAtPath(fallback_path)
                if prim.IsValid():
                    color_vec = Gf.Vec3f(color[0], color[1], color[2])
                    attr = prim.GetAttribute("inputs:diffuseColor")
                    if attr.IsValid():
                        attr.Set(color_vec)
                    else:
                        print(f"[Visuals] Warning: Found prim at {fallback_path} but no 'inputs:diffuseColor' attribute.")
                else:
                    # Final attempt: search for any Shader under the Table prim
                    from pxr import Usd, UsdShade
                    table_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
                    if table_prim.IsValid():
                        found = False
                        # Use Usd.PrimRange to traverse the subtree
                        for p in Usd.PrimRange(table_prim):
                            if p.IsA(UsdShade.Shader):
                                attr = p.GetAttribute("inputs:diffuseColor")
                                if attr.IsValid():
                                    attr.Set(Gf.Vec3f(color[0], color[1], color[2]))
                                    found = True
                                    break
                        if not found:
                             print(f"[Visuals] Error: Could not find any Shader with diffuseColor under /World/envs/env_{i}/Table")
                    else:
                        print(f"[Visuals] Error: Table prim not found at /World/envs/env_{i}/Table")

    # ------------------------------------------------------------------
    # Per-iteration stats helpers
    # ------------------------------------------------------------------

    def _make_iter_stats(self) -> dict:
        return {
            "invalid_goals": 0,
            "valid_goals": 0,
            "bob_successes": 0,
            "bob_failures": 0,
            "terminations": {},  # reason -> count
            # Alice displacement accumulators — summed across all _handle_alice_completion
            # calls in the iteration; printed as a single [AliceDisp] line in train.py.
            "alice_total": 0,
            "alice_disp_3d_sum": 0.0,
            "alice_disp_xy_sum": 0.0,
            "alice_disp_xy_max": 0.0,
            "alice_disp_y_sum": 0.0,
            "alice_disp_z_sum": 0.0,
            "alice_not_moved": 0,
        }

    def reset_iter_stats(self):
        self._iter_stats = self._make_iter_stats()

    def get_iter_stats(self) -> dict:
        return dict(self._iter_stats)

    def _initial_states_from_spawn(self, spawn_info: dict, n: int) -> torch.Tensor:
        """Build initial_states tensor (n, 6) or (n, 12) from reset_objects_to_random_safe_pose output."""
        _id_euler = torch.zeros(n, 3, device=self.device)
        t_local = spawn_info.get("target_local", torch.zeros(n, 3, device=self.device))
        if self.num_objects == 1:
            return torch.cat([t_local, _id_euler], dim=-1)
        t2_local = spawn_info.get("cube_local")
        if t2_local is None:
            t2_local = t_local.clone()
            t2_local[:, 0] -= 0.10
        return torch.cat([t_local, _id_euler, t2_local, _id_euler], dim=-1)

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
        self._alice_phase_initialized[env_ids] = False
        
        # Set table to Red for Alice phase
        self.set_table_color(env_ids, (0.8, 0.1, 0.1))

        # Place objects at random positions so Alice cannot trivially push the block
        # to a fixed goal with her home-position arm sweep.  The returned spawn_info
        # carries the per-env local positions used (Z=0.023 settled height) so we can
        # set initial_states without a PhysX readback (avoids timing issues).
        spawn_info = reset_objects_to_random_safe_pose(self.env, env_ids)
        reset_robot_joints(self.env, env_ids)
        self.env.scene.write_data_to_sim()
        self.episode_manager.initial_states = self._initial_states_from_spawn(
            spawn_info, self.num_envs
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

        _prof = self._profiler
        _null = _prof is None

        # Step base environment
        if not _null:
            _prof.mark_start("isaac_step")
        obs_dict, rewards, terminated, truncated, extras = self.env.step(scaled_action)
        if not _null:
            _prof.mark_stop("isaac_step")

        # Capture phase state BEFORE any resets so reward/success logic sees the
        # correct phase even for envs that terminate this step.
        is_alice_before = self.episode_manager.is_alice_phase().clone()
        is_bob_before = self.episode_manager.is_bob_phase().clone()

        # Sync EpisodeManager with Env Resets — fully vectorised, no per-env .item() calls.
        if not _null:
            _prof.mark_start("wrapper_reset")
        dones = terminated | truncated
        if dones.any():
            reset_ids = torch.where(dones)[0]
            is_alice = self.episode_manager.is_alice_phase()

            tm = self.env.termination_manager
            term_dones = tm._term_dones  # (num_envs, num_conditions)
            term_names = tm._term_names

            is_term_vec = terminated[reset_ids]   # (num_reset,) bool
            is_trunc_vec = ~is_term_vec

            # Termination stats: count per condition with a single .item() each
            if is_term_vec.any():
                term_env_ids = reset_ids[is_term_vec]
                fired_mat = term_dones[term_env_ids]  # (num_term, num_conditions)
                for i, name in enumerate(term_names):
                    cnt = int(fired_mat[:, i].sum().item())
                    if cnt:
                        self._iter_stats["terminations"][name] = (
                            self._iter_stats["terminations"].get(name, 0) + cnt
                        )
                no_fired_cnt = int((~fired_mat.any(dim=1)).sum().item())
                if no_fired_cnt:
                    self._iter_stats["terminations"]["physics_termination"] = (
                        self._iter_stats["terminations"].get("physics_termination", 0)
                        + no_fired_cnt
                    )

            if is_trunc_vec.any():
                self._iter_stats["terminations"]["timeout"] = (
                    self._iter_stats["terminations"].get("timeout", 0)
                    + int(is_trunc_vec.sum().item())
                )

            # Handle Alice early failures (vectorised)
            alice_term_ids = reset_ids[is_term_vec & is_alice[reset_ids]]
            if len(alice_term_ids) > 0:
                if not hasattr(self, "_early_alice_failures"):
                    self._early_alice_failures = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                self._early_alice_failures[alice_term_ids] = True
                self.delayed_alice_reward[alice_term_ids] = -3.0
                self._alice_phase_initialized[alice_term_ids] = False

            # Handle Bob early terminations (e.g. pushes object off table):
            # pay Alice +5 for Bob failure, hide ghost, reset objects, update initial_states.
            bob_term_ids = reset_ids[is_term_vec & ~is_alice[reset_ids]]
            if len(bob_term_ids) > 0:
                self.delayed_alice_reward[bob_term_ids] += reward_utils.ALICE_BOB_FAIL_REWARD
                self.hide_goal_ghost(bob_term_ids)
                _sp = reset_objects_to_random_safe_pose(self.env, bob_term_ids)
                reset_robot_joints(self.env, bob_term_ids)
                self.env.scene.write_data_to_sim()
                self.episode_manager.initial_states[bob_term_ids] = (
                    self._initial_states_from_spawn(_sp, len(bob_term_ids))
                )
                self._alice_phase_initialized[bob_term_ids] = False

            # Batch reset — one call for all done envs
            self.episode_manager.reset_episode(reset_ids)
            self.hide_goal_ghost(reset_ids)
        if not _null:
            _prof.mark_stop("wrapper_reset")

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
        # Per-object errors for multi-object logging: (num_objects, num_envs)
        step_pos_per_obj = torch.zeros(self.num_objects, self.num_envs, device=self.device)
        step_rot_per_obj = torch.zeros(self.num_objects, self.num_envs, device=self.device)

        # Handle Bob phase completion
        bob_progress_reward = None
        if phase_info["bob_done"].any():
            bob_done_ids = torch.where(phase_info["bob_done"])[0]
            success, pos_err, rot_err, pos_po, rot_po, prog_rew = self._handle_bob_completion(
                obs_dict, bob_done_ids
            )
            step_bob_success[bob_done_ids] = success
            step_bob_done[bob_done_ids] = True
            step_pos_err[bob_done_ids] = pos_err
            step_rot_err[bob_done_ids] = rot_err
            if self.num_objects >= 2:
                step_pos_per_obj[:, bob_done_ids] = pos_po.T
                step_rot_per_obj[:, bob_done_ids] = rot_po.T
            # Phase-end progress reward: add to the last step's reward for these Bob envs
            bob_progress_reward = prog_rew  # (num_envs,) non-zero only at bob_done_ids

            extras["alice_total_reward"][bob_done_ids] = self.delayed_alice_reward[
                bob_done_ids
            ]
            self.delayed_alice_reward[bob_done_ids] = 0.0
            any_reset = True

        if any_reset:
            self.env.scene.write_data_to_sim()
            obs_dict = self.env.observation_manager.compute()

        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        if not _null:
            _prof.mark_start("wrapper_rewards")
        current_rewards, bob_achieved_completion = self._get_current_rewards(
            obs_dict, rewards, is_alice_before, is_bob_before, action
        )
        # Inject phase-end progress reward for Bob envs that just completed (normal path).
        # bob_progress_reward is a (num_envs,) tensor set in the bob_done branch above.
        if bob_progress_reward is not None:
            current_rewards += bob_progress_reward
        if not _null:
            _prof.mark_stop("wrapper_rewards")

        # Handle Bob's EARLY SUCCESS
        if bob_achieved_completion.any():
            completion_ids = torch.where(bob_achieved_completion)[0]
            self.episode_manager.bob_success[completion_ids] = True
            step_bob_success[completion_ids] = True
            step_bob_done[completion_ids] = True
            self.episode_manager.completion_given[completion_ids] = True

            success, pos_err, rot_err, pos_per_obj, rot_per_obj = self._check_bob_success(
                obs_dict, completion_ids
            )
            step_pos_err[completion_ids] = pos_err
            step_rot_err[completion_ids] = rot_err
            if self.num_objects >= 2:
                step_pos_per_obj[:, completion_ids] = pos_per_obj.T
                step_rot_per_obj[:, completion_ids] = rot_per_obj.T

            # Phase-end progress reward for early success
            w_pos, w_rot = 0.6, 0.4
            init_pos = self.bob_init_pos_err[completion_ids]
            init_rot = self.bob_init_rot_err[completion_ids]
            pos_prog = (init_pos - pos_err) / (init_pos + 1e-6)
            rot_prog = (init_rot - rot_err) / (init_rot + 1e-6)
            r_prog_early = (w_pos * pos_prog + w_rot * rot_prog).clamp(-1.0, 1.0)
            current_rewards[completion_ids] += r_prog_early
            self._bob_progress_captured[completion_ids] = False

            self._log_bob_progress(pos_prog, rot_prog, r_prog_early)

            completion_steps = self.episode_manager.phase_step[completion_ids].clone()
            completion_goals = self.episode_manager.goal_count[completion_ids].clone()

            self._handle_bob_success_transition(completion_ids)
            self._iter_stats["bob_successes"] += len(completion_ids)

            alice_success_penalty = torch.full(
                (len(completion_ids),),
                reward_utils.ALICE_BOB_SUCCESS_REWARD,
                device=self.device,
            )
            self.delayed_alice_reward[completion_ids] += alice_success_penalty

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
            self.env.scene.write_data_to_sim()
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
            "phase": self.episode_manager.current_phase,
            "goal_count": self.episode_manager.goal_count,
            "bob_success": self.episode_manager.bob_success,
            "bob_success_this_step": step_bob_success,
            "bob_done_this_step": step_bob_done,
            "bob_pos_err": step_pos_err,
            "bob_rot_err": step_rot_err,
            "bob_pos_per_obj": step_pos_per_obj,
            "bob_rot_per_obj": step_rot_per_obj,
            "goal_valid": self.episode_manager.goal_valid,
            "goal_states": (
                self.episode_manager.goal_states
                if self.episode_manager.goal_states is not None
                else torch.zeros(
                    (self.num_envs, self.num_objects * 6), device=self.device
                )
            ),
            "initial_states": self.episode_manager.initial_states.clone(),
            "max_contact_force": self.episode_manager.max_contact_force,
        }
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
        _ALICE_POS_REQ = 0.05
        alice_pos_req = _ALICE_POS_REQ
        alice_rot_req = 0.25

        # 2. Validate Goal (Local vs Local)
        valid, val_reward, reasons = validate_goal(
            active_initial,
            active_goal,
            self.table_bounds,
            self.placement_bounds,
            pos_threshold=alice_pos_req,
            rot_threshold=alice_rot_req,
            min_meaningful_disp=0.10,
            require_all_moved=(self.num_objects >= 2),
        )

        # Fix 10: removed 7cm min-XY filter — paper accepts rotation-only and short-range goals
        self.delayed_alice_reward[env_ids] = val_reward

        # 3. Aggregate stats (skip per-env debug prints to prevent CPU-GPU stall on HPC)
        start_pos = active_initial[:, 0:3]
        final_pos = active_goal[:, 0:3]
        dist_3d = torch.norm(final_pos - start_pos, dim=-1)
        dist_xy = torch.norm(final_pos[:, :2] - start_pos[:, :2], dim=-1)
        dist_y = (final_pos[:, 1] - start_pos[:, 1]).abs()
        dist_z = (final_pos[:, 2] - start_pos[:, 2]).abs()

        self._alice_phase_initialized[env_ids] = False

        n = len(env_ids)
        self._iter_stats["valid_goals"] += int(valid.sum().item())
        self._iter_stats["invalid_goals"] += int((~valid).sum().item())
        self._iter_stats["alice_total"] += n
        self._iter_stats["alice_disp_3d_sum"] += dist_3d.sum().item()
        self._iter_stats["alice_disp_xy_sum"] += dist_xy.sum().item()
        self._iter_stats["alice_disp_xy_max"] = max(
            self._iter_stats["alice_disp_xy_max"], dist_xy.max().item()
        )
        self._iter_stats["alice_disp_y_sum"] += dist_y.sum().item()
        self._iter_stats["alice_disp_z_sum"] += dist_z.sum().item()
        self._iter_stats["alice_not_moved"] += int(
            (dist_3d <= _ALICE_POS_REQ).sum().item()
        )

        # 4. Storage for Bob (LOCAL)
        self.episode_manager.store_goal_state(active_goal, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)
        self.episode_manager.mark_alice_base_reward(env_ids, val_reward)

        # 5. Transition logic
        successful_goal = valid
        valid_env_ids = env_ids[successful_goal]

        if len(valid_env_ids) > 0:
            self.episode_manager.transition_to_bob(valid_env_ids)
            start_states = self.episode_manager.initial_states[valid_env_ids]
            origins = self.env.scene.env_origins[valid_env_ids]

            # Reset Target (Local -> World) — 12D Euler layout: [t_pos(3)|t_euler(3)|t2_pos(3)|t2_euler(3)]
            # write_root_pose_to_sim expects 7D (pos3 + quat4), so convert euler→quat first.
            from .observations import _euler_xyz_to_quat

            t_pos_local = start_states[:, 0:3]
            t_quat = _euler_xyz_to_quat(start_states[:, 3:6])
            pos1_global = t_pos_local + origins
            self.env.scene["target_object"].write_root_pose_to_sim(
                torch.cat([pos1_global, t_quat], dim=-1),
                env_ids=valid_env_ids,
            )

            # Check for trivially easy goals: after reset, Bob starts within the
            # success threshold of the goal.  Reject these so Alice must produce
            # goals that actually require movement to reach.
            _goal_pos   = active_goal[successful_goal, 0:3]
            _goal_euler = active_goal[successful_goal, 3:6]
            _start_pos  = start_states[:, 0:3]
            _start_euler = start_states[:, 3:6]
            _pos_dist = (_start_pos - _goal_pos).norm(dim=-1)
            _rot_diff = ((_goal_euler - _start_euler + torch.pi) % (2 * torch.pi)) - torch.pi
            _rot_dist = _rot_diff.abs().max(dim=-1)[0]
            _too_easy = (_pos_dist < self.goal_tolerance) & (_rot_dist < 0.2)
            too_easy_ids = valid_env_ids[_too_easy]
            valid_env_ids = valid_env_ids[~_too_easy]

            if len(too_easy_ids) > 0:
                self.episode_manager.reset_episode(too_easy_ids, reason="Alice Too-Easy Goal")
                self.hide_goal_ghost(too_easy_ids)
                _sp = reset_objects_to_random_safe_pose(self.env, too_easy_ids)
                reset_robot_joints(self.env, too_easy_ids)
                self.env.scene.write_data_to_sim()
                self.episode_manager.initial_states[too_easy_ids] = (
                    self._initial_states_from_spawn(_sp, len(too_easy_ids))
                )
                self._alice_phase_initialized[too_easy_ids] = False
                self.delayed_alice_reward[too_easy_ids] = -3.0
                # Update iter stats: move from valid to invalid
                self._iter_stats["valid_goals"] -= len(too_easy_ids)
                self._iter_stats["invalid_goals"] += len(too_easy_ids)

            # Reset Cube (Local -> World) — only when cube is present (num_objects==2)
            if self.num_objects == 2:
                t2_t2_pos_local = start_states[:, 6:9]
                t2_quat = _euler_xyz_to_quat(start_states[:, 9:12])
                pos2_global = t2_t2_pos_local + origins
                self.env.scene["cube"].write_root_pose_to_sim(
                    torch.cat([pos2_global, t2_quat], dim=-1),
                    env_ids=valid_env_ids,
                )

            # Move marker to Alice's final pose on the table surface (shows goal for Bob)
            if "goal_marker" in self.env.scene.rigid_objects:
                goal_pos_local = goal_state[valid_env_ids, 0:3].clone()
                goal_pos_local[:, 2] = -0.001  # below table surface — no collision with objects
                # Only yaw rotation — keep marker flat on table
                goal_euler = goal_state[valid_env_ids, 3:6].clone()
                goal_euler[:, 0] = 0.0  # zero roll
                goal_euler[:, 1] = 0.0  # zero pitch
                goal_quat = _euler_xyz_to_quat(goal_euler)
                marker_pos_global = goal_pos_local + origins
                self.env.scene["goal_marker"].write_root_pose_to_sim(
                    torch.cat([marker_pos_global, goal_quat], dim=-1),
                    env_ids=valid_env_ids,
                )

            reset_robot_joints(self.env, valid_env_ids)
            
            # Transition table to Blue for Bob phase
            self.set_table_color(valid_env_ids, (0.1, 0.1, 0.8))

        # 6. Handle Invalid
        invalid_env_ids = env_ids[~successful_goal]
        if len(invalid_env_ids) > 0:
            self.episode_manager.reset_episode(
                invalid_env_ids, reason="Alice Invalid Goal"
            )
            # Hide ghost on invalid goal
            self.hide_goal_ghost(invalid_env_ids)
            
            _sp = reset_objects_to_random_safe_pose(self.env, invalid_env_ids)
            reset_robot_joints(self.env, invalid_env_ids)
            self.episode_manager.initial_states[invalid_env_ids] = (
                self._initial_states_from_spawn(_sp, len(invalid_env_ids))
            )
            # Ensure table is Red (reset reason was already Alice Invalid Goal)
            self.set_table_color(invalid_env_ids, (0.8, 0.1, 0.1))

        return valid, invalid_env_ids

    def _log_bob_progress(self, pos_progress, rot_progress, r_progress):
        """Accumulate progress stats and print summary every N completions."""
        for i in range(len(pos_progress)):
            self._bob_progress_log.append((
                r_progress[i].item(),
                pos_progress[i].item(),
                rot_progress[i].item(),
            ))
        if len(self._bob_progress_log) >= self._BOB_PROGRESS_LOG_INTERVAL:
            n = len(self._bob_progress_log)
            r_sum = [x[0] for x in self._bob_progress_log]
            p_sum = [x[1] for x in self._bob_progress_log]
            y_sum = [x[2] for x in self._bob_progress_log]
            print(
                f"  [BobProgress] progress={sum(r_sum)/n:+.3f}  "
                f"pos={sum(p_sum)/n:+.3f}  rot={sum(y_sum)/n:+.3f}  "
                f"pos_n={(sum(1 for v in r_sum if v>0))}  neg_n={(sum(1 for v in r_sum if v<0))}  "
                f"(n={n})",
                flush=True,
            )
            self._bob_progress_log.clear()

    def _handle_bob_completion(
        self, obs_dict: Dict, env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle completion of Bob's phase.
        
        Returns: (success, pos_err, rot_err, progress_reward) where progress_reward
        is a (num_envs,) tensor with values clamped to [−1, +1] for env_ids,
        zero elsewhere.
        """
        # --- 1. Success check ---
        success, pos_err, rot_err, pos_per_obj, rot_per_obj = self._check_bob_success(obs_dict, env_ids)
        self.episode_manager.mark_bob_success(env_ids, success)

        # --- 2. Phase-end progress reward ---
        # r_progress = clamp(w_pos·(init_pos_err − final_pos_err)/init_pos_err
        #                  + w_rot·(init_rot_err − final_rot_err)/init_rot_err, −1, +1)
        w_pos = 0.6
        w_rot = 0.4
        prog_reward = torch.zeros(self.num_envs, device=self.device)
        init_pos = self.bob_init_pos_err[env_ids]
        init_rot = self.bob_init_rot_err[env_ids]
        pos_progress = (init_pos - pos_err) / (init_pos + 1e-6)
        rot_progress = (init_rot - rot_err) / (init_rot + 1e-6)
        r_prog_raw = w_pos * pos_progress + w_rot * rot_progress
        r_prog = r_prog_raw.clamp(-1.0, 1.0)
        prog_reward[env_ids] = r_prog
        # Clear captured flag so next Bob phase re-captures
        self._bob_progress_captured[env_ids] = False

        self._log_bob_progress(pos_progress, rot_progress, r_prog)

        # --- 3. Alice Outcome Reward (+5 if Bob fails, 0 if he succeeds) ---
        outcome_rewards = torch.where(
            success,
            torch.tensor(reward_utils.ALICE_BOB_SUCCESS_REWARD, device=self.device),
            torch.tensor(reward_utils.ALICE_BOB_FAIL_REWARD, device=self.device),
        )
        self.delayed_alice_reward[env_ids] += outcome_rewards

        # --- 4. Aggregate success/failure counts ---
        n_success = int(success.sum().item())
        n_failure = len(env_ids) - n_success
        self._iter_stats["bob_successes"] += n_success
        self._iter_stats["bob_failures"] += n_failure


# --- Robot Control Utilities ---
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
            # Hide ghost when Alice starts her new phase
            self.hide_goal_ghost(continue_ids)
            
            _sp = reset_objects_to_random_safe_pose(self.env, continue_ids)
            reset_robot_joints(self.env, continue_ids)
            self.episode_manager.initial_states[continue_ids] = (
                self._initial_states_from_spawn(_sp, len(continue_ids))
            )
            # Reset potential-shaping state so the new Alice phase captures a fresh
            # start position on its first step. Without this, alice_start_pos_xy
            # holds the stale world position from the *previous* Alice phase, and
            # the first-step dist spike produces large spurious dense rewards even
            # when the object hasn't moved.
            self._alice_phase_initialized[continue_ids] = False
            self.alice_prev_dist[continue_ids] = 0.0
            
            # Transition table back to Red for new Alice phase
            self.set_table_color(continue_ids, (0.8, 0.1, 0.1))

        # Episode End (Bob failed or max goals)
        reset_ids = env_ids[~can_continue]
        if len(reset_ids) > 0:
            succeeded = self.episode_manager.bob_success[reset_ids]
            reason = "Bob Succeeded" if succeeded.any() else "Bob Failed"
            self.episode_manager.reset_episode(reset_ids, reason=reason)
            # Hide ghost on reset
            self.hide_goal_ghost(reset_ids)

            _sp = reset_objects_to_random_safe_pose(self.env, reset_ids)
            reset_robot_joints(self.env, reset_ids)
            self.episode_manager.initial_states[reset_ids] = (
                self._initial_states_from_spawn(_sp, len(reset_ids))
            )
            # Transition table back to Red for Alice phase
            self.set_table_color(reset_ids, (0.8, 0.1, 0.1))

        # Commit all physics writes so the next observation read reflects the reset.
        return success, pos_err, rot_err, pos_per_obj, rot_per_obj, prog_reward

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
            self.hide_goal_ghost(continue_ids)
            _sp = reset_objects_to_random_safe_pose(self.env, continue_ids)
            reset_robot_joints(self.env, continue_ids)
            self.episode_manager.initial_states[continue_ids] = (
                self._initial_states_from_spawn(_sp, len(continue_ids))
            )
            # Reset potential-shaping state (same fix as _handle_bob_completion).
            self._alice_phase_initialized[continue_ids] = False
            self.alice_prev_dist[continue_ids] = 0.0
            
            # Transition table back to Red for new Alice phase
            self.set_table_color(continue_ids, (0.8, 0.1, 0.1))

        # Others reset (reached max goals with success)
        reset_ids = env_ids[~can_continue]
        if len(reset_ids) > 0:
            self.episode_manager.reset_episode(reset_ids, reason="Episode Complete")
            self.hide_goal_ghost(reset_ids)

            _sp = reset_objects_to_random_safe_pose(self.env, reset_ids)
            reset_robot_joints(self.env, reset_ids)
            self.episode_manager.initial_states[reset_ids] = (
                self._initial_states_from_spawn(_sp, len(reset_ids))
            )
            # Transition table back to Red for Alice phase
            self.set_table_color(reset_ids, (0.8, 0.1, 0.1))

        # Commit all physics writes so the next observation read reflects the reset.
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
        success = torch.all(obj_success, dim=-1)

        # Per-object and max-across-objects errors
        max_pos_err, _ = torch.max(pos_dist, dim=-1)
        max_rot_err, _ = torch.max(rot_dist, dim=-1)
        # Per-object errors: (N, num_objects) — used for multi-object logging
        pos_dist_per_obj = pos_dist
        rot_dist_per_obj = rot_dist

        return success, max_pos_err, max_rot_err, pos_dist_per_obj, rot_dist_per_obj

    def _extract_object_states(self, obs_dict: Dict) -> torch.Tensor:
        """Extract object states in the LOCAL frame as Euler poses.

        Returns [pos(3)+euler(3)] per object = 6D per object.
        Total: 6D (num_objects=1) or 12D (num_objects=2).
        Matching the paper's rotation representation (ZYX Euler angles).
        """
        from .observations import _quat_to_euler_xyz

        t_pos_w = self.env.scene["target_object"].data.root_pos_w
        t_quat_w = self.env.scene["target_object"].data.root_quat_w
        env_origins = self.env.scene.env_origins
        t_pos_l = t_pos_w - env_origins
        t_euler = _quat_to_euler_xyz(t_quat_w)  # (N, 3)

        if self.num_objects == 1:
            return torch.cat([t_pos_l, t_euler], dim=-1)  # 6D

        t2_pos_w = self.env.scene["cube"].data.root_pos_w
        t2_quat_w = self.env.scene["cube"].data.root_quat_w
        t2_pos_l = t2_pos_w - env_origins
        t2_euler = _quat_to_euler_xyz(t2_quat_w)  # (N, 3)
        return torch.cat([t_pos_l, t_euler, t2_pos_l, t2_euler], dim=-1)  # 12D

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

        # Alice: NO rewards during her phase receives rewards at goal validation in _handle_alice_completion:
        # Fix 3+11: Alice gets zero per-step reward (paper: only outcome reward at phase end)
        # Removed: base_rewards passthrough (physics penalties) and bounded potential shaping
        if is_alice.any():
            rewards[is_alice] = 0.0

            # Diagnostic per-step shaping (--diag_alice_shaping).
            # Temporarily reward EE→object proximity to confirm Alice can learn contact
            # manipulation.  Remove after the diagnostic confirms ValidGoals trends upward
            # in the --alice_sandbox test.
            if getattr(self, "_diag_alice_shaping", False):
                alice_obs = obs_dict["alice_policy"][is_alice]
                ee_pos = alice_obs[:, :3]
                # Distance to nearest object across all num_objects (14D per object).
                _obj_dim = 14
                obj_pos_list = [
                    alice_obs[:, self.robot_state_dim + k * _obj_dim : self.robot_state_dim + k * _obj_dim + 3]
                    for k in range(self.num_objects)
                ]
                delta = torch.stack([(p - ee_pos).norm(dim=-1) for p in obj_pos_list], dim=1).min(dim=1)[0]
                shaping = 0.005 * torch.clamp(0.3 - delta, 0.0, 0.3)
                rewards[is_alice] += shaping

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

        # Get goal distances per object
        target_dists = goal_distance(
            self.env, object_cfg=SceneEntityCfg("target_object")
        )
        if self.num_objects == 2:
            cube_dists = goal_distance(self.env, object_cfg=SceneEntityCfg("cube"))
            all_dists = torch.cat([target_dists, cube_dists], dim=1)
        else:
            all_dists = target_dists  # (num_envs, 2): [pos_dist, rot_dist]

        batch_size = all_dists.shape[0]
        num_objects = self.num_objects

        # Reshape: (num_envs, 2, 2) -> [obj_idx, (pos, rot)]
        dists = all_dists.view(batch_size, num_objects, 2)

        pos_dists = dists[..., 0]  # (num_envs, num_objects)
        rot_dists = dists[..., 1]

        # Fix 4: removed potential-based shaping (paper: Bob gets sparse {+1/-1/+5} only)
        bob_phase_steps = self.episode_manager.phase_step
        is_bob_phase = self.episode_manager.is_bob_phase()
        is_first_step = (bob_phase_steps == 1) & is_bob_phase

        # Phase-end progress reward: capture initial errors on Bob's first step.
        # These are stored and later used in _handle_bob_completion and the early-
        # success path to compute r_progress = clamp(w_pos·Δpos/init_pos + w_rot·Δrot/init_rot, −1, +1).
        if is_first_step.any():
            f_ids = torch.where(is_first_step)[0]
            _f_pos_dists = pos_dists[f_ids]  # (n, num_objects)
            _f_rot_dists = rot_dists[f_ids]
            self.bob_init_pos_err[f_ids] = _f_pos_dists.max(dim=1)[0]
            self.bob_init_rot_err[f_ids] = _f_rot_dists.max(dim=1)[0]
            self._bob_progress_captured[f_ids] = True

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
        # If an object (like the secondary T-block) is untouched by Alice, its start position is its goal position.
        # It will be evaluated as "True" on step 1, but we don't want to give Bob a free +1 reward for it.
        # Restrict to Bob-phase envs only (Alice-phase envs at step 1 must not interfere).
        # NOTE: bob_phase_steps, is_bob_phase, is_first_step already computed above in the shaping block.
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
        _OBJ_NAMES = ["target", "cube"][: self.num_objects]

        completion_bonus = torch.where(
            should_give_completion,
            torch.ones(batch_size, device=self.device) * 5.0,
            torch.zeros(batch_size, device=self.device),
        )

        # Mark completion as given for environments that just received it
        self.episode_manager.completion_given = (
            self.episode_manager.completion_given | should_give_completion
        )

        rewards = step_rewards + completion_bonus  # Fix 4: sparse only, no potential shaping

        # Skip per-env reward event logging to prevent CPU-GPU stall on HPC

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

    def hide_goal_ghost(self, env_ids: torch.Tensor):
        """Hide the goal marker by moving it under the table."""
        if "goal_marker" in self.env.scene.rigid_objects:
            N = len(env_ids)
            hide_pos = torch.cat([
                self.env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, -1.0], device=self.device),
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(N, -1),
            ], dim=-1)
            self.env.scene["goal_marker"].write_root_pose_to_sim(hide_pos, env_ids=env_ids)

    def close(self):
        """Close wrapped environment"""
        self.env.close()
