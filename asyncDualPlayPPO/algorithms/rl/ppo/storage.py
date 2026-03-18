import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:
    """
    Fixed-length rollout buffer for on-policy PPO.

    Observations are stored in (time, env) order for efficient mini-batch
    slicing.  A boolean mask distinguishes active agent transitions from
    padding steps inserted during phase transitions in the dual-play setup:
    mask=1 for real steps, mask=0 for padding.
    """

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device="cpu", sampler="sequential"):
        self.device  = device
        self.sampler = sampler

        self.observations    = torch.zeros(num_transitions_per_env, num_envs, *obs_shape,    device=device)
        self.states          = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=device)
        self.rewards         = torch.zeros(num_transitions_per_env, num_envs, 1,             device=device)
        self.actions         = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones           = torch.zeros(num_transitions_per_env, num_envs, 1,             device=device).byte()
        self.masks           = torch.ones( num_transitions_per_env, num_envs, 1,             device=device).byte()
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1,             device=device)
        self.values          = torch.zeros(num_transitions_per_env, num_envs, 1,             device=device)
        self.returns         = torch.zeros(num_transitions_per_env, num_envs, 1,             device=device)
        self.advantages      = torch.zeros(num_transitions_per_env, num_envs, 1,             device=device)
        self.mu              = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma           = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs                = num_envs
        self.step                    = 0

    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma, masks=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        self.masks[self.step].copy_(masks.view(-1, 1) if masks is not None else torch.ones_like(self.masks[self.step]))

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        GAE-λ return computation.

        Uses the mask rather than dones to gate bootstrapping: during the
        Alice→Bob phase transition, mask=0 prevents value estimates from the
        incoming Alice phase from leaking into Bob's return calculation.
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values        = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = self.masks[step].float()
            delta              = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage          = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values

        # Normalize advantages using ONLY valid (mask=1) entries.
        # Alice's buffer is ~95% zero-padding; including those zeros in
        # the normalization makes std tiny → valid advantages explode.
        valid = self.masks.bool().squeeze(-1)  # (T, N)
        if valid.any():
            valid_adv = self.advantages[valid]
            adv_mean = valid_adv.mean()
            adv_std  = valid_adv.std() + 1e-8
        else:
            adv_mean = self.advantages.mean()
            adv_std  = self.advantages.std() + 1e-8
            
        self.advantages = (self.advantages - adv_mean) / adv_std
        
        # CRITICAL FIX: Zero out advantages for invalid/padded entries!
        # Otherwise, the normalization shifts the 0.0 advantages of padded entries 
        # to (0 - mean)/std, which causes massive surrogate loss explosions.
        self.advantages[~valid] = 0.0

    def get_statistics(self):
        done         = self.dones.cpu()
        done[-1]     = 1
        flat_dones   = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        """
        Return a BatchSampler over the full rollout buffer.

        Uses sequential sampling for physics-based RL: parallel envs are
        already independently randomized, so random sub-sampling provides
        no additional diversity but adds CPU overhead.
        """
        batch_size      = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        return BatchSampler(subset, mini_batch_size, drop_last=True)

class GPUDemonstrationBuffer:
    def __init__(self, capacity, obs_shape, states_shape, actions_shape, device="cpu"):
        self.capacity = capacity
        self.device = device
        
        self.observations    = torch.zeros(capacity, *obs_shape,    device=device)
        self.states          = torch.zeros(capacity, *states_shape, device=device)
        self.actions         = torch.zeros(capacity, *actions_shape, device=device)
        self.rewards         = torch.zeros(capacity, 1,             device=device)
        self.dones           = torch.zeros(capacity, 1,             device=device).byte()
        self.values          = torch.zeros(capacity, 1,             device=device)
        self.actions_log_prob = torch.zeros(capacity, 1,             device=device)
        self.mu              = torch.zeros(capacity, *actions_shape, device=device)
        self.sigma           = torch.zeros(capacity, *actions_shape, device=device)
        self.returns         = torch.zeros(capacity, 1,             device=device)
        self.advantages      = torch.zeros(capacity, 1,             device=device)
        
        self.step = 0
        self.full = False

    @property
    def size(self) -> int:
        """Number of valid entries currently in the buffer."""
        return self.capacity if self.full else self.step

    def add_trajectory(self, obs, states, acts, rews, dones, vals, log_probs, mus, sigmas, ret, adv):
        """Add a full trajectory or sequence of steps"""
        n = obs.size(0)
        if n == 0:
            return
            
        if self.step + n > self.capacity:
            end = self.capacity
            size = self.capacity - self.step
            
            self.observations[self.step:end].copy_(obs[:size])
            if states is not None: self.states[self.step:end].copy_(states[:size])
            self.actions[self.step:end].copy_(acts[:size])
            self.rewards[self.step:end].copy_(rews[:size].view(-1,1))
            self.dones[self.step:end].copy_(dones[:size].view(-1,1))
            self.values[self.step:end].copy_(vals[:size].view(-1,1))
            self.actions_log_prob[self.step:end].copy_(log_probs[:size].view(-1,1))
            self.mu[self.step:end].copy_(mus[:size])
            self.sigma[self.step:end].copy_(sigmas[:size])
            self.returns[self.step:end].copy_(ret[:size].view(-1,1))
            self.advantages[self.step:end].copy_(adv[:size].view(-1,1))
            
            self.step = 0
            self.full = True
            
            # Wrap around
            rem = n - size
            if rem > 0:
                self.observations[:rem].copy_(obs[size:])
                if states is not None: self.states[:rem].copy_(states[size:])
                self.actions[:rem].copy_(acts[size:])
                self.rewards[:rem].copy_(rews[size:].view(-1,1))
                self.dones[:rem].copy_(dones[size:].view(-1,1))
                self.values[:rem].copy_(vals[size:].view(-1,1))
                self.actions_log_prob[:rem].copy_(log_probs[size:].view(-1,1))
                self.mu[:rem].copy_(mus[size:])
                self.sigma[:rem].copy_(sigmas[size:])
                self.returns[:rem].copy_(ret[size:].view(-1,1))
                self.advantages[:rem].copy_(adv[size:].view(-1,1))
                self.step = rem
        else:
            self.observations[self.step:self.step+n].copy_(obs)
            if states is not None: self.states[self.step:self.step+n].copy_(states)
            self.actions[self.step:self.step+n].copy_(acts)
            self.rewards[self.step:self.step+n].copy_(rews.view(-1,1))
            self.dones[self.step:self.step+n].copy_(dones.view(-1,1))
            self.values[self.step:self.step+n].copy_(vals.view(-1,1))
            self.actions_log_prob[self.step:self.step+n].copy_(log_probs.view(-1,1))
            self.mu[self.step:self.step+n].copy_(mus)
            self.sigma[self.step:self.step+n].copy_(sigmas)
            self.returns[self.step:self.step+n].copy_(ret.view(-1,1))
            self.advantages[self.step:self.step+n].copy_(adv.view(-1,1))
            self.step += n
            if self.step == self.capacity:
                self.step = 0
                self.full = True

    def sample(self, batch_size):
        if self.step == 0 and not self.full:
            return None
        max_idx = self.capacity if self.full else self.step
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)
        
        states = self.states[indices] if self.states is not None else None
        
        return (
            self.observations[indices],
            states,
            self.actions[indices],
            self.values[indices],
            self.returns[indices],
            self.actions_log_prob[indices],
            self.advantages[indices],
            self.mu[indices],
            self.sigma[indices]
        )
