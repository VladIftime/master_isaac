from datetime import datetime
import math
import os
import time

from gymnasium.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .storage import RolloutStorage
from .module import ActorCritic
from .ppo import PPO

class PPOABC(PPO):
    def __init__(self,
                 vec_env,
                 cfg_train,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):
        super().__init__(vec_env, cfg_train, device, sampler, log_dir, is_testing, print_log, apply_reset, asymmetric)
        
        # [CRITICAL FIX] Calculate mini_batch_size if not set by parent
        # PPO should set this, but ABC implementation sometimes misses it
        if not hasattr(self, 'mini_batch_size'):
            # Get values from config and vec_env instead of self (which aren't set yet)
            num_envs = vec_env.unwrapped.num_envs if hasattr(vec_env, 'unwrapped') else vec_env.num_envs
            nsteps = cfg_train["learn"]["nsteps"]
            
            # [SCALING FIX] Dynamically scale nminibatches to keep mini-batch size ~2048
            # This ensures we take more gradient steps per epoch as num_envs increases
            batch_size = num_envs * nsteps
            target_mini_batch_size = 2048
            # Start with at least 4 (default), but scale up if batch is large
            num_mini_batches = max(4, batch_size // target_mini_batch_size)
            
            # Override parent's num_mini_batches
            self.num_mini_batches = num_mini_batches
            self.mini_batch_size = batch_size // num_mini_batches
        
        # ABC Parameters
        self.abc_coef = cfg_train["learn"].get("abc_coef", 0.5)
        self.abc_clip_param = cfg_train["learn"].get("abc_clip_param", 0.2)
        self.abc_batch_size = cfg_train["learn"].get("abc_batch_size", 2048)
        
        # ABC Buffer (GPUDemonstrationBuffer from buffers.py)
        # Initialized in train.py and passed via set_abc_buffer, or init here if dimensions known.
        # Since dimensions depend on env specific wrappers, we'll let train.py set it.
        self.abc_buffer = None

    def set_abc_buffer(self, abc_buffer):
        self.abc_buffer = abc_buffer

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_abc_loss = 0

        # No deepcopy needed — ABC loss uses simple NLL, not clipped surrogate

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        
        # ABC Mini-Batch Size: FIXED size independent of PPO rollout batch
        # PPO batch grows with num_envs for sample efficiency
        # ABC batch should be fixed (sampling from replay buffer)
        # Default 2048 provides good gradient estimates without OOM
        # Ensure we don't request more than available in buffer
        abc_mini_batch_size = min(self.abc_batch_size, self.abc_buffer.size if self.abc_buffer else self.abc_batch_size)
        
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                # --- RL Data ---
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                # Extract mask to identify Bob's valid steps (mask=0 for Alice-phase padding)
                masks_batch = self.storage.masks.view(-1, 1)[indices]
                valid_mask = masks_batch.squeeze() > 0

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # ONLY compute RL losses on valid Bob steps (mask > 0)
                if valid_mask.sum() > 0:
                    # Surrogate loss (Masked)
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                       1.0 + self.clip_param)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped)[valid_mask].mean()

                    # Value function loss (Masked)
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped)[valid_mask].mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2)[valid_mask].mean()

                    entropy_mean = entropy_batch[valid_mask].mean()
                else:
                    # Fallback if batch contains no valid Bob steps
                    surrogate_loss = torch.tensor(0.0, device=self.device)
                    value_loss = torch.tensor(0.0, device=self.device)
                    entropy_mean = torch.tensor(0.0, device=self.device)
                
                # --- ABC Loss (Behavioral Cloning via NLL with clamped sigma) ---
                abc_loss_val = torch.tensor(0.0).to(self.device)
                if self.abc_buffer is not None and self.abc_buffer.size > 0:
                    
                    abc_obs, abc_act, _ = self.abc_buffer.sample(abc_mini_batch_size)
                    
                    if abc_obs.shape[0] > 0:
                        # Get predicted mean from actor
                        abc_mu = self.actor_critic.actor(abc_obs)
                        
                        # Get sigma with safety clamp to prevent variance collapse
                        # Without this clamp, sigma -> 0 causes log_prob -> -inf and NLL -> +inf
                        abc_sigma = torch.clamp(self.actor_critic.log_std.exp(), min=1e-3)
                        
                        # NLL for independent Gaussian per action dimension:
                        # -log p(a|mu,sigma) = 0.5*((a-mu)/sigma)^2 + log(sigma) + 0.5*log(2*pi)
                        nll = 0.5 * ((abc_act - abc_mu) / abc_sigma) ** 2 + torch.log(abc_sigma) + 0.5 * math.log(2 * math.pi)
                        abc_loss_val = nll.sum(dim=-1).mean()  # sum over action dims, mean over batch
                    
                    mean_abc_loss += abc_loss_val.item()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean + self.abc_coef * abc_loss_val

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_abc_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_abc_loss
