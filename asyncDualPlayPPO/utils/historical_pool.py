"""
Historical policy pool for past-opponent sampling (ASP paper, Section 3.3).

The paper trains Bob against a mix of current Alice (80%) and past Alice
policies (20%) to improve stability and prevent Alice from finding exploits
that only work against the current Bob.

Usage in train.py:
    pool = HistoricalPolicyPool(max_size=5)
    # Periodically save:
    pool.add(alice_ppo.actor_critic)
    # During rollout, determine which envs use a historical policy:
    hist_indices, hist_policy = pool.sample_env_subset(all_indices, frac=0.2)
"""

import copy
import random
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class HistoricalPolicyPool:
    """
    Ring buffer of past policy snapshots.

    Snapshots are stored as CPU state_dicts to minimise GPU memory usage.
    When sampling, a temporary clone is created on the target device.
    """

    def __init__(self, max_size: int = 5):
        self.max_size: int = max_size
        self._pool: List[dict] = []  # list of state_dicts (on CPU)

    def add(self, policy: nn.Module):
        """Save a snapshot of the current policy weights."""
        snapshot = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        if len(self._pool) >= self.max_size:
            self._pool.pop(0)  # evict oldest
        self._pool.append(snapshot)

    def sample_policy(
        self, reference_policy: nn.Module, device: str
    ) -> Optional[nn.Module]:
        """
        Return a deep-copy of reference_policy loaded with a random past snapshot.
        Returns None if the pool is empty.
        """
        if not self._pool:
            return None
        snapshot = random.choice(self._pool)
        hist_copy = copy.deepcopy(reference_policy)
        hist_copy.load_state_dict({k: v.to(device) for k, v in snapshot.items()})
        hist_copy.eval()
        return hist_copy

    def sample_env_subset(
        self,
        env_indices: torch.Tensor,
        frac: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split env_indices into (historical_subset, current_subset).

        Returns two tensors:
            hist_ids   — env indices that should use a historical policy (~frac)
            curr_ids   — env indices that should use the current policy (~1-frac)

        If the pool is empty, hist_ids is empty and curr_ids == env_indices.
        """
        if not self._pool or len(env_indices) == 0:
            return (
                torch.tensor([], dtype=env_indices.dtype, device=env_indices.device),
                env_indices,
            )

        n_hist = max(1, int(len(env_indices) * frac))
        perm = torch.randperm(len(env_indices), device=env_indices.device)
        hist_ids = env_indices[perm[:n_hist]]
        curr_ids = env_indices[perm[n_hist:]]
        return hist_ids, curr_ids

    @property
    def size(self) -> int:
        return len(self._pool)
