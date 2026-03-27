"""
PPOABC — PPO with Alice Behavioral Cloning (ABC).

Implements the Negative Log-Likelihood (NLL) loss for Alice Behavioral Cloning,
as described in the original OpenAI paper:
'Asymmetric Self-Play for Automatic Goal Discovery in Robotic Manipulation'.

Alice's successful trajectories (s_t, a_t) are used as direct demonstrations
for Bob to reach the same goal g = s_{T_A}. Bob's policy is trained to
maximize the likelihood of Alice's actions given the same state and goal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ppo import PPO


class PPOABC(PPO):
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
        super().__init__(
            vec_env,
            cfg_train,
            device,
            sampler,
            log_dir,
            is_testing,
            print_log,
            apply_reset,
            asymmetric,
        )

        # Recompute mini_batch_size if the parent did not set it
        if not hasattr(self, "mini_batch_size"):
            num_envs = (
                vec_env.unwrapped.num_envs
                if hasattr(vec_env, "unwrapped")
                else vec_env.num_envs
            )
            batch_size = num_envs * cfg_train["learn"]["nsteps"]
            self.num_mini_batches = max(4, batch_size // 2048)
            self.mini_batch_size = batch_size // self.num_mini_batches

        self.abc_coef = cfg_train["learn"].get("abc_coef", 0.1)
        self.abc_batch_size = cfg_train["learn"].get("abc_batch_size", 2048)
        self.abc_buffer = None

    def set_abc_buffer(self, abc_buffer):
        """Attach the demonstration buffer populated by train.py."""
        self.abc_buffer = abc_buffer

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_bc_loss = 0

        # --- ABC: sample ONCE and compute fresh θ_old BEFORE mini-epochs ---
        # Paper Section 3.2: θ_old is Bob's behavior policy at the START of
        # this training step (the policy used to collect the current rollout).
        # We compute Bob's current log-probs on the demo batch here, then use
        # them as the denominator for PPO-style ratio clipping during the
        # mini-epoch loop. This prevents the stale-denominator bug where
        # old_log_probs from demo insertion time (potentially hundreds of
        # iterations ago) immediately clip the ratio to 1.2 → zero gradient.
        abc_sample = None
        abc_old_lp = None
        if self.abc_buffer is not None and self.abc_buffer.size > 0:
            abc_batch_size = min(self.abc_batch_size, self.abc_buffer.size)
            abc_sample = self.abc_buffer.sample(abc_batch_size)
            if abc_sample is not None and abc_sample[0].shape[0] > 0:
                with torch.no_grad():
                    abc_old_lp, _, _, _, _ = self.actor_critic.evaluate(
                        abc_sample[0], None, abc_sample[2]
                    )

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(
                    -1, *self.storage.observations.size()[2:]
                )[indices]
                states_batch = (
                    self.storage.states.view(-1, *self.storage.states.size()[2:])[
                        indices
                    ]
                    if self.asymmetric
                    else None
                )
                actions_batch = self.storage.actions.view(
                    -1, self.storage.actions.size(-1)
                )[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[
                    indices
                ]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[
                    indices
                ]
                old_sigma_batch = self.storage.sigma.view(
                    -1, self.storage.actions.size(-1)
                )[indices]
                masks_batch = self.storage.masks.view(-1, 1)[indices]

                (
                    actions_log_prob_batch,
                    entropy_batch,
                    value_batch,
                    mu_batch,
                    sigma_batch,
                ) = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (
                            torch.square(old_sigma_batch.exp())
                            + torch.square(old_mu_batch - mu_batch)
                        )
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
                    ratio = torch.exp(
                        actions_log_prob_batch
                        - torch.squeeze(old_actions_log_prob_batch)
                    )
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    surrogate_loss = torch.max(surrogate, surrogate_clipped)[
                        valid_mask
                    ].mean()

                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (
                            value_batch - target_values_batch
                        ).clamp(-self.clip_param, self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped)[
                            valid_mask
                        ].mean()
                    else:
                        value_loss = (
                            (returns_batch - value_batch).pow(2)[valid_mask].mean()
                        )

                    entropy_mean = entropy_batch[valid_mask].mean()
                else:
                    surrogate_loss = torch.tensor(0.0, device=self.device)
                    value_loss = torch.tensor(0.0, device=self.device)
                    entropy_mean = torch.tensor(0.0, device=self.device)

                # --- ABC (Alice Behavioral Cloning) Loss ---
                # Paper Section 3.2: L_abc = -E[clip(π_B(a|s,g;θ) / π_B(a|s,g;θ_old), 1-ε, 1+ε)]
                # θ_old = Bob's policy at the START of this training step (computed above).
                # The same abc_sample and abc_old_lp are reused across all mini-epochs;
                # only θ (the optimizing params) changes during the loop.
                bc_loss = torch.tensor(0.0, device=self.device)
                if abc_sample is not None and abc_old_lp is not None:
                    bc_obs, bc_act = abc_sample[0], abc_sample[2]
                    abc_log_probs, _, _, _, _ = self.actor_critic.evaluate(
                        bc_obs, None, bc_act, detach_goal_encoder=True
                    )
                    ratio = torch.exp(abc_log_probs - abc_old_lp)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    bc_loss = -torch.min(ratio, clipped).mean()
                    mean_bc_loss += bc_loss.item()

                # --- Auxiliary Distance Prediction Loss ---
                aux_loss_val = torch.tensor(0.0, device=self.device)
                if getattr(self.actor_critic, "use_goal_encoder", False) and hasattr(self.actor_critic.goal_encoder, "aux_loss"):
                    # Extract 14-dim s_t and s_star from 56-dim obs_batch
                    s_t_batch = torch.cat([obs_batch[:, 8:15], obs_batch[:, 32:39]], dim=-1)
                    s_star_batch = torch.cat([obs_batch[:, 23:30], obs_batch[:, 47:54]], dim=-1)
                    
                    aux, pos_loss, rot_loss = self.actor_critic.goal_encoder.aux_loss(s_star_batch, s_t_batch)
                    aux_loss_val = self.cfg_train["learn"].get("aux_coef", 0.1) * aux
                    
                    if hasattr(self, "writer") and self.writer is not None:
                        it = self.current_learning_iteration
                        self.writer.add_scalar("GoalEncoder/aux_pos_loss", pos_loss.item(), it)
                        self.writer.add_scalar("GoalEncoder/aux_rot_loss", rot_loss.item(), it)

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_mean
                    + self.abc_coef * bc_loss
                    + aux_loss_val
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_bc_loss /= num_updates

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_bc_loss,
            0.0,
        )  # 0.0 for IDM loss (removed)
