"""
Episode manager for asymmetric dual-play.

Tracks episode phases (Alice/Bob), goal states, and success counters.
"""

import torch
from enum import Enum
from typing import Optional


class Phase(Enum):
    """Episode phase"""

    ALICE = 0
    BOB = 1
    RESET = 2


class EpisodeManager:
    """
    Manages episode state and phase transitions for async dual-play.

    Episode structure:
    1. ALICE phase: 250 timesteps to set a goal
    2. Goal validation
    3. BOB phase: Up to 600 timesteps to solve, can attempt up to 5 goals
    4. Reset if invalid or max goals reached
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        alice_timesteps: int = 100,  # Reduced from 250 for better temporal credit assignment
        bob_timesteps: int = 200,
        max_goals_per_episode: int = 5,
        pos_threshold: float = 0.04,
        rot_threshold: float = 0.2,
        min_goal_dist: float = 0.001,  # Minimum distance Alice must move an object (near-zero = always valid)
    ):
        self.num_envs = num_envs
        self.device = device
        self.alice_timesteps = alice_timesteps
        self.bob_timesteps = bob_timesteps
        self.max_goals = max_goals_per_episode
        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold
        self.min_goal_dist = min_goal_dist  # Decoupled from pos_threshold

        # State tracking
        self.current_phase = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )  # 0 = Alice, 1 = Bob
        self.phase_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.goal_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Goal storage
        self.initial_states: Optional[torch.Tensor] = None  # Stored at Alice start
        self.goal_states: Optional[torch.Tensor] = None  # Stored after Alice phase
        self.goal_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Success tracking
        self.bob_success = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )  # Current goal success
        self.bob_ever_failed = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )  # Has Bob failed at any point in episode?
        self.prev_obj_success: Optional[torch.Tensor] = (
            None  # Tracks per-object success for rewards
        )
        self.completion_given = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )  # Tracks if +5 bonus was given
        self.max_contact_force = torch.zeros(
            num_envs, device=device
        )  # Track contact forces for Safe-State HER
        self.bob_success_this_step = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )

        # Per-goal success rate tracking (paper metric: successes / goals_attempted across full episode)
        self.goals_attempted = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.goals_succeeded = torch.zeros(num_envs, dtype=torch.int32, device=device)

        print(f"[EpisodeManager] Initialized for {num_envs} envs", flush=True)
        print(f"  Alice timesteps: {alice_timesteps}", flush=True)
        print(f"  Bob timesteps: {bob_timesteps}", flush=True)
        print(f"  Max goals per episode: {self.max_goals}", flush=True)
        print(
            f"  Success thresholds: pos={pos_threshold:.3f}m, rot={rot_threshold:.3f}rad",
            flush=True,
        )

        # NOTE: Both Alice's movement threshold (wrapper: alice_pos_req=0.05) and
        # Bob's pos_threshold (0.04) use full 3D distance. Gravity drop is excluded
        # by recording initial_states at the settled z=0.023 (not spawn z=0.05).
        # alice_pos_req > pos_threshold prevents instant-win (see wrapper).
        if pos_threshold >= 0.1:  # Sanity check: threshold too large = too easy
            print(
                f"[WARNING] pos_threshold ({pos_threshold}) is very large - goals may be too easy!"
            )
            print(f"  Consider a smaller value (e.g., 0.05m)")

    def get_phase(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get current phase for specified environments"""
        if env_ids is None:
            return self.current_phase
        return self.current_phase[env_ids]

    def is_alice_phase(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check if environments are in Alice phase"""
        phase = self.get_phase(env_ids)
        return phase == Phase.ALICE.value

    def is_bob_phase(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check if environments are in Bob phase"""
        phase = self.get_phase(env_ids)
        return phase == Phase.BOB.value

    def step(self) -> dict:
        """
        Advance episode manager by one timestep.

        Returns:
            Dictionary with transition information
        """
        self.phase_step += 1

        # Check for phase transitions
        alice_done = self.is_alice_phase() & (self.phase_step >= self.alice_timesteps)
        bob_done = self.is_bob_phase() & (self.phase_step >= self.bob_timesteps)

        return {
            "alice_done": alice_done,
            "bob_done": bob_done,
        }

    def transition_to_bob(self, env_ids: torch.Tensor):
        """Transition specified environments from Alice to Bob phase"""
        # Log Alice's completion stats
        for env_id in env_ids:
            steps = self.phase_step[env_id].item()
            goal_num = self.goal_count[env_id].item() + 1
            print(
                f"[Phase] Env {env_id.item()}: Alice→Bob | Goal {goal_num}/5 | Alice completed {steps}/{self.alice_timesteps} steps",
                flush=True,
            )

        self.current_phase[env_ids] = Phase.BOB.value
        self.phase_step[env_ids] = 0
        self.goal_count[env_ids] += 1
        # Reset reward tracking for fresh Bob phase
        self.completion_given[env_ids] = False
        self.max_contact_force[env_ids] = 0.0
        if self.prev_obj_success is not None:
            self.prev_obj_success[env_ids] = False

    def transition_to_alice(self, env_ids: torch.Tensor):
        """Transition specified environments from Bob to Alice phase (for next goal)"""
        # Log Bob's completion stats — capped at 3 per call
        for _log_i, env_id in enumerate(env_ids):
            if _log_i >= 3:
                break
            steps = self.phase_step[env_id].item()
            goal_num = self.goal_count[env_id].item()
            success = self.bob_success[env_id].item()
            status = "SUCCESS" if success else "FAILURE"
            print(
                f"[Phase] Env {env_id.item()}: Bob→Alice | Goal {goal_num}/5 {status} | Bob used {steps}/{self.bob_timesteps} steps",
                flush=True,
            )

        self.current_phase[env_ids] = Phase.ALICE.value
        self.phase_step[env_ids] = 0

    def reset_episode(self, env_ids: torch.Tensor, reason: str = "Unknown"):
        """Reset specified environments to start of episode"""
        # Log detailed reset info — capped at 3 per call to avoid log spam at scale
        for _log_i, env_id in enumerate(env_ids):
            if _log_i >= 3:
                break
            phase_name = (
                "Alice"
                if self.current_phase[env_id].item() == Phase.ALICE.value
                else "Bob"
            )
            steps = self.phase_step[env_id].item()
            goals = self.goal_count[env_id].item()
            max_steps = (
                self.alice_timesteps if phase_name == "Alice" else self.bob_timesteps
            )
            print(
                f"[Reset] Env {env_id.item()}: {phase_name} phase @ step {steps}/{max_steps} | Goals: {goals}/5 | Reason: {reason}",
                flush=True,
            )

        self.current_phase[env_ids] = Phase.ALICE.value
        self.phase_step[env_ids] = 0
        self.goal_count[env_ids] = 0
        self.goal_valid[env_ids] = False
        self.bob_success[env_ids] = False
        self.bob_ever_failed[env_ids] = False  # Reset episode-level failure tracking
        self.bob_success_this_step[env_ids] = False
        self.completion_given[env_ids] = False
        self.max_contact_force[env_ids] = 0.0
        if self.prev_obj_success is not None:
            self.prev_obj_success[env_ids] = False
        self.goals_attempted[env_ids] = 0
        self.goals_succeeded[env_ids] = 0

    def store_initial_state(self, state: torch.Tensor):
        """Store initial state at start of episode"""
        self.initial_states = state.clone()

    def store_goal_state(self, state: torch.Tensor, env_ids: torch.Tensor):
        """Store goal state after Alice's phase.

        Args:
            state: Already-sliced goal states for env_ids, shape (len(env_ids), state_dim).
            env_ids: Global environment indices into the full num_envs pool.
        """
        state_dim = state.shape[-1]
        if self.goal_states is None:
            # Must be sized (num_envs, state_dim), NOT zeros_like(state) which
            # would create a tensor sized to the active subset only.
            self.goal_states = torch.zeros(self.num_envs, state_dim, device=self.device)
        # state is already sliced for env_ids — do not re-index with env_ids.
        self.goal_states[env_ids] = state.clone()

    def mark_goal_valid(self, env_ids: torch.Tensor, valid: torch.Tensor):
        """Mark goals as valid/invalid"""
        self.goal_valid[env_ids] = valid

    def mark_bob_success(self, env_ids: torch.Tensor, success: torch.Tensor):
        """Mark Bob's success/failure"""
        self.bob_success[env_ids] = success
        # Track if Bob ever failed in this episode (for goal skipping logic)
        failed = ~success
        self.bob_ever_failed[env_ids] = self.bob_ever_failed[env_ids] | failed
        # Per-goal counters for success-rate metric (paper: successes / goals_attempted)
        self.goals_attempted[env_ids] += 1
        self.goals_succeeded[env_ids] += success.int()

    def should_reset(self) -> torch.Tensor:
        """Check which environments should reset"""
        # Reset if:
        # - Goal is invalid
        # - Bob failed and reached max goals
        # - Bob succeeded (episode complete)

        invalid_goal = ~self.goal_valid & (self.goal_count > 0)
        max_goals_reached = self.goal_count >= self.max_goals
        bob_succeeded = self.bob_success

        return invalid_goal | (max_goals_reached & ~bob_succeeded) | bob_succeeded

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return all per-env state tensors as a CPU dict for checkpointing.

        Included tensors control phase routing and goal tracking:
          - current_phase  : 0=Alice, 1=Bob per env
          - phase_step     : steps elapsed in the current phase
          - goal_count     : goals attempted this episode
          - goal_valid     : whether the current goal is valid
          - bob_success    : whether Bob succeeded on the current goal
          - bob_ever_failed
          - completion_given
          - goals_attempted / goals_succeeded (episode-level success-rate)
          - initial_states : safe-reset object poses recorded at episode start
          - goal_states    : Alice's target poses stored after Alice phase
        """
        sd = {
            "current_phase":    self.current_phase.cpu(),
            "phase_step":       self.phase_step.cpu(),
            "goal_count":       self.goal_count.cpu(),
            "goal_valid":       self.goal_valid.cpu(),
            "bob_success":      self.bob_success.cpu(),
            "bob_ever_failed":  self.bob_ever_failed.cpu(),
            "completion_given": self.completion_given.cpu(),
            "max_contact_force": self.max_contact_force.cpu(),
            "goals_attempted":  self.goals_attempted.cpu(),
            "goals_succeeded":  self.goals_succeeded.cpu(),
        }
        if self.initial_states is not None:
            sd["initial_states"] = self.initial_states.cpu()
        if self.goal_states is not None:
            sd["goal_states"] = self.goal_states.cpu()
        if self.prev_obj_success is not None:
            sd["prev_obj_success"] = self.prev_obj_success.cpu()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        """Restore per-env state from a dict produced by :meth:`state_dict`.

        Safe to call after :meth:`__init__`; restores all tensors in-place
        to the correct device so training resumes exactly where it left off.
        """
        def _load(key: str, dest: torch.Tensor) -> torch.Tensor:
            """Copy a saved CPU tensor back to *dest* (device-agnostic)."""
            if key in sd:
                dest.copy_(sd[key].to(self.device))
            return dest

        _load("current_phase",    self.current_phase)
        _load("phase_step",       self.phase_step)
        _load("goal_count",       self.goal_count)
        _load("goal_valid",       self.goal_valid)
        _load("bob_success",      self.bob_success)
        _load("bob_ever_failed",  self.bob_ever_failed)
        _load("completion_given", self.completion_given)
        _load("max_contact_force", self.max_contact_force)
        _load("goals_attempted",  self.goals_attempted)
        _load("goals_succeeded",  self.goals_succeeded)

        if "initial_states" in sd:
            self.initial_states = sd["initial_states"].to(self.device)
        if "goal_states" in sd:
            self.goal_states = sd["goal_states"].to(self.device)
        if "prev_obj_success" in sd:
            if self.prev_obj_success is None:
                self.prev_obj_success = sd["prev_obj_success"].to(self.device)
            else:
                self.prev_obj_success.copy_(sd["prev_obj_success"].to(self.device))

        n_bob = int((self.current_phase == 1).sum().item())
        n_alice = int((self.current_phase == 0).sum().item())
        print(
            f"[EpisodeManager] State restored: {n_alice} Alice-phase envs, "
            f"{n_bob} Bob-phase envs",
            flush=True,
        )
