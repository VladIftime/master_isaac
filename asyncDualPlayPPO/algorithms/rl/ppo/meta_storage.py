"""
Meta-step rollout storage for Phase 2 (Charlie Architecture).

Stores transitions at meta-step frequency (1/C the physics rate).
Each entry corresponds to one Meta-Bob or Meta-Alice decision,
not one physics tick.
"""

import torch


class MetaRolloutStorage:
    """
    Rollout buffer operating at meta-step frequency.

    One entry = one Meta-Bob or Meta-Alice decision.
    Observation: global_state(30) for Meta-Alice,
                 global_state(30) + master_goal(14) for Meta-Bob.
    Action: (g_left, g_right) ∈ R^{2K} — continuous Gaussian.

    Meta-ABC buffer is stored separately (GPUDemonstrationBuffer).
    """

    def __init__(
        self,
        num_envs: int,
        num_meta_steps: int,
        obs_dim: int,
        action_dim: int,
        device: str,
    ):
        self.num_envs = num_envs
        self.num_meta_steps = num_meta_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.observations = torch.zeros(
            num_meta_steps, num_envs, obs_dim, device=device
        )
        self.actions = torch.zeros(
            num_meta_steps, num_envs, action_dim, device=device
        )
        self.rewards = torch.zeros(
            num_meta_steps, num_envs, 1, device=device
        )
        self.values = torch.zeros(
            num_meta_steps, num_envs, 1, device=device
        )
        self.actions_log_prob = torch.zeros(
            num_meta_steps, num_envs, 1, device=device
        )
        self.dones = torch.zeros(
            num_meta_steps, num_envs, 1, device=device
        ).byte()
        self.returns = torch.zeros(
            num_meta_steps, num_envs, 1, device=device
        )
        self.advantages = torch.zeros(
            num_meta_steps, num_envs, 1, device=device
        )
        self.step = 0

    def add_transitions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
    ):
        """Add one meta-step transition to the buffer."""
        if self.step >= self.num_meta_steps:
            raise RuntimeError(
                f"MetaRolloutStorage overflow: step {self.step} >= {self.num_meta_steps}"
            )

        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards)
        self.dones[self.step].copy_(dones.unsqueeze(-1).byte() if dones.dim() == 1 else dones.byte())
        self.values[self.step].copy_(values.unsqueeze(-1) if values.dim() == 1 else values)
        self.actions_log_prob[self.step].copy_(
            actions_log_prob.unsqueeze(-1) if actions_log_prob.dim() == 1 else actions_log_prob
        )
        self.step += 1

    def clear(self):
        """Reset the buffer for the next rollout."""
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        """
        Compute GAE returns.

        last_values: critic values at the final state (num_envs, 1).
        Reward is injected only at the last meta-step (sparse).
        dones[-1] MUST be forced to 1.0 to prevent GAE bleeding across episodes.
        """
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        next_values = last_values

        for step in reversed(range(self.step)):
            not_done = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + gamma * next_values * not_done
                - self.values[step]
            )
            advantage = delta + gamma * lam * not_done * advantage
            self.returns[step] = advantage + self.values[step]
            self.advantages[step] = advantage
            next_values = self.values[step]

    def mini_batch_generator(self, num_mini_batches: int):
        """
        Yield mini-batches for PPO update.

        Flattens (num_meta_steps, num_envs) into a single batch,
        then splits into num_mini_batches equal-sized chunks.
        """
        batch_size = self.step * self.num_envs
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size, device=self.device)

        # Flatten temporal dimension
        obs = self.observations[: self.step].reshape(-1, self.obs_dim)
        acts = self.actions[: self.step].reshape(-1, self.action_dim)
        rets = self.returns[: self.step].reshape(-1, 1)
        advs = self.advantages[: self.step].reshape(-1, 1)
        vals = self.values[: self.step].reshape(-1, 1)
        lps = self.actions_log_prob[: self.step].reshape(-1, 1)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            idx = indices[start:end]
            yield (
                obs[idx],
                acts[idx],
                rets[idx],
                advs[idx],
                vals[idx],
                lps[idx],
            )
