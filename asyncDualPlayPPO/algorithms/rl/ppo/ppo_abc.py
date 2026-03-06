"""
PPOBCO — PPO with Behavioral Cloning from Observation (BCO).

Replaces the broken NLL-ABC loss with an Inverse Dynamics Model (IDM) approach:

1. The IDM is trained each update step on Bob's own sequential rollout data
   (obs_t → obs_{t+1} → act_t) so it learns Bob's kinematics.
2. In train.py, Alice's successful state trajectories are fed through the IDM to
   produce kinematically correct Bob actions that are stored in the demo buffer.
3. The demo buffer is used here for a standard Behavioral Cloning (Huber) loss,
   which is now safe because the demonstrations are Bob's own actions.

Set abc_coef to 0.0 to disable BCO and run standard PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ppo import PPO
from .module import InverseDynamicsModel


class PPOBCO(PPO):
    def __init__(
        self,
        vec_env,
        cfg_train,
        device="cpu",
        sampler="sequential",
        log_dir="run",
        is_testing=False,
        print_log=True,
        apply_reset=False,
        asymmetric=False,
    ):
        super().__init__(vec_env, cfg_train, device, sampler, log_dir, is_testing, print_log, apply_reset, asymmetric)

        # Recompute mini_batch_size if the parent did not set it, scaling so each
        # mini-batch is ~2048 transitions regardless of num_envs.
        if not hasattr(self, "mini_batch_size"):
            num_envs   = vec_env.unwrapped.num_envs if hasattr(vec_env, "unwrapped") else vec_env.num_envs
            batch_size = num_envs * cfg_train["learn"]["nsteps"]
            self.num_mini_batches = max(4, batch_size // 2048)
            self.mini_batch_size  = batch_size // self.num_mini_batches

        self.abc_coef       = cfg_train["learn"].get("abc_coef",       0.05)
        self.abc_batch_size = cfg_train["learn"].get("abc_batch_size", 2048)

        self.idm = InverseDynamicsModel(
            vec_env.observation_space.shape,
            vec_env.action_space.shape,
        ).to(self.device)
        self.idm_optimizer = optim.Adam(self.idm.parameters(), lr=3e-4)

        self.abc_buffer = None

    def set_abc_buffer(self, abc_buffer):
        """Attach the demonstration buffer populated by train.py."""
        self.abc_buffer = abc_buffer

    def update(self):
        mean_value_loss     = 0
        mean_surrogate_loss = 0
        mean_bc_loss        = 0
        mean_idm_loss       = 0

        # --- IDM training on Bob's sequential rollout data ---
        # Temporal order matters here so we use the raw storage (not mini-batch shuffling).
        # We train for 3 epochs before the PPO epoch loop so that the IDM is up to date
        # when train.py calls bob_ppo.idm() for action relabelling after this update.
        obs_rollout = self.storage.observations   # (nsteps, num_envs, obs_dim)
        act_rollout = self.storage.actions        # (nsteps, num_envs, act_dim)
        obs_t      = obs_rollout[:-1].view(-1, self.observation_space.shape[0])
        obs_tplus1 = obs_rollout[1:].view(-1, self.observation_space.shape[0])
        act_t      = act_rollout[:-1].view(-1, self.action_space.shape[0])

        for _ in range(3):
            predicted_actions = self.idm(obs_t, obs_tplus1)
            idm_loss = F.mse_loss(predicted_actions, act_t)
            self.idm_optimizer.zero_grad()
            idm_loss.backward()
            self.idm_optimizer.step()
        mean_idm_loss = idm_loss.item()

        # --- PPO + BCO update ---
        abc_batch = min(self.abc_batch_size, self.abc_buffer.size if self.abc_buffer else self.abc_batch_size)

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch    = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices] if self.asymmetric else None
                actions_batch              = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch        = self.storage.values.view(-1, 1)[indices]
                returns_batch              = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch           = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch               = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch            = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
                masks_batch                = self.storage.masks.view(-1, 1)[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(
                    obs_batch, states_batch, actions_batch
                )

                if self.desired_kl is not None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch - old_sigma_batch
                        + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                valid_mask = masks_batch.squeeze() > 0
                if valid_mask.sum() > 0:
                    ratio             = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate         = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    surrogate_loss    = torch.max(surrogate, surrogate_clipped)[valid_mask].mean()

                    if self.use_clipped_value_loss:
                        value_clipped        = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses         = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss           = torch.max(value_losses, value_losses_clipped)[valid_mask].mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2)[valid_mask].mean()

                    entropy_mean = entropy_batch[valid_mask].mean()
                else:
                    surrogate_loss = torch.tensor(0.0, device=self.device)
                    value_loss     = torch.tensor(0.0, device=self.device)
                    entropy_mean   = torch.tensor(0.0, device=self.device)

                bc_loss = torch.tensor(0.0, device=self.device)
                if self.abc_buffer is not None and self.abc_buffer.size > 0:
                    bc_obs, bc_act, _ = self.abc_buffer.sample(abc_batch)
                    if bc_obs.shape[0] > 0:
                        predicted = self.actor_critic.actor(bc_obs)
                        # Smooth L1 (Huber) is more robust than MSE for IDM-inferred targets,
                        # which are good-but-not-perfect approximations of Bob's true actions.
                        bc_loss = F.smooth_l1_loss(predicted, bc_act)
                    mean_bc_loss += bc_loss.item()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean + self.abc_coef * bc_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss     += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_bc_loss        /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_idm_loss
