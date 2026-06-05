import torch


class GPUDemonstrationBuffer:
    def __init__(self, capacity, obs_shape, action_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate flat tensors on GPU
        self.obs = torch.zeros(
            (capacity, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (capacity, *action_shape), dtype=torch.float32, device=device
        )
        # We store 'old_log_prob' here to use as the reference for clipping
        self.old_log_probs = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=device
        )

    def add_batch(self, obs, actions, log_probs):
        """
        Efficiently add a batch of transitions (e.g., from 50 environments that just finished).
        """
        num_samples = obs.shape[0]
        if num_samples == 0:
            return

        # Handle wrap-around
        # If batch size > capacity (unlikely but possible), we just take the last 'capacity' elements
        if num_samples > self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            log_probs = log_probs[-self.capacity :]
            num_samples = self.capacity

        # Indices for the buffer
        indices = (
            torch.arange(self.ptr, self.ptr + num_samples, device=self.device)
            % self.capacity
        )

        self.obs[indices] = obs
        self.actions[indices] = actions
        # Ensure log_probs has correct shape (N, 1) even if passed as (N,)
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(1)
        self.old_log_probs[indices] = log_probs

        self.ptr = (self.ptr + num_samples) % self.capacity
        self.size = min(self.size + num_samples, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            # Return empty tensors that match shape if buffer is empty
            # This prevents crash but gradient will be zero
            return (
                torch.zeros((0, *self.obs.shape[1:]), device=self.device),
                torch.zeros((0, *self.actions.shape[1:]), device=self.device),
                torch.zeros((0, 1), device=self.device),
            )

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[indices], self.actions[indices], self.old_log_probs[indices]
