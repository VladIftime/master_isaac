"""
Meta Episode Manager for Phase 2 (Charlie Architecture).

Tracks meta-step horizons, meta-phase transitions, master goal storage,
and Alice's trajectory for Meta-ABC behavioral cloning.
"""

import torch
from enum import IntEnum


class MetaPhase(IntEnum):
    META_ALICE = 0
    META_BOB = 1


class MetaEpisodeManager:
    """
    Manages meta-episode state at the meta-step level.

    Meta-episode structure:
        Phase META_ALICE : TA_meta meta-steps, each = C atomic steps
        Phase META_BOB   : TB_meta meta-steps, each = C atomic steps

    Stores per-env:
        s0_snapshots      : full physics state at episode start
        master_goals      : S* = object state after Meta-Alice's phase
        meta_alice_trajs  : list of (St, g_meta) for Meta-ABC trajectory trimming
        meta_bob_success  : whether Bob succeeded this episode
    """

    def __init__(self, num_envs: int, device: str, TA_meta: int, TB_meta: int):
        self.num_envs = num_envs
        self.device = device
        self.TA_meta = TA_meta
        self.TB_meta = TB_meta

        # Phase & step tracking
        self.current_phase = torch.full(
            (num_envs,), MetaPhase.META_ALICE, dtype=torch.int32, device=device
        )
        self.meta_step = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Master goal storage (object-only state: 14 dims)
        self.master_goals = torch.zeros(num_envs, 14, device=device)

        # Physics snapshots (set by MetaASPWrapper.snapshot())
        self.s0_snapshots = [None] * num_envs

        # Meta-Alice trajectory buffer for Meta-ABC
        # Each env stores list of (global_state, g_meta) tuples
        self.meta_alice_trajs: list[list] = [[] for _ in range(num_envs)]

        # Episode outcome
        self.meta_bob_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset_episode(self, env_ids: torch.Tensor = None):
        """Reset all meta-episode state for the given envs."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.current_phase[env_ids] = MetaPhase.META_ALICE
        self.meta_step[env_ids] = 0
        self.master_goals[env_ids] = 0.0
        self.meta_bob_success[env_ids] = False

        for eid in env_ids.tolist():
            self.s0_snapshots[eid] = None
            self.meta_alice_trajs[eid] = []

    def record_alice_step(
        self,
        env_ids: torch.Tensor,
        global_state: torch.Tensor,
        g_meta: torch.Tensor,
    ):
        """Record one meta-step from Alice's trajectory for Meta-ABC."""
        for i, eid in enumerate(env_ids.tolist()):
            self.meta_alice_trajs[eid].append(
                (global_state[i].clone(), g_meta[i].clone())
            )

    def get_trimmed_alice_traj(self, eid: int) -> list:
        """
        Return the causally-relevant portion of Alice's trajectory.

        Only the latter half (TA_meta // 2) is used for Meta-ABC to avoid
        cloning Alice's early exploratory noise.
        """
        traj = self.meta_alice_trajs[eid]
        trim_start = max(0, len(traj) - (self.TA_meta // 2))
        return traj[trim_start:]

    def transition_to_bob(self, env_ids: torch.Tensor):
        """Transition from Meta-Alice phase to Meta-Bob phase."""
        self.current_phase[env_ids] = MetaPhase.META_BOB
        self.meta_step[env_ids] = 0

    def is_alice_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check if Alice has exhausted her meta-step budget."""
        return self.meta_step[env_ids] >= self.TA_meta

    def is_bob_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check if Bob has exhausted his meta-step budget."""
        return self.meta_step[env_ids] >= self.TB_meta
