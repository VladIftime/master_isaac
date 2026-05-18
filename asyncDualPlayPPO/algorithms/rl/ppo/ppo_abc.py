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

        self.abc_coef = cfg_train["learn"].get("abc_coef", 0.5)
        self.abc_batch_size = cfg_train["learn"].get("abc_batch_size", 2048)
        # Fix 1: number of full trajectories for sequential LSTM evaluation
        self.abc_n_trajs = cfg_train["learn"].get("abc_n_trajs", 16)
        # Fix 9: removed abc_warmup_threshold gate — paper uses β=0.5 from iteration 1
        self.abc_warmup_threshold = 0.0  # always active
        self.abc_buffer = None

    def set_abc_buffer(self, abc_buffer):
        """Attach the demonstration buffer populated by train.py."""
        self.abc_buffer = abc_buffer

    def _compute_abc_loss_sequential(self, abc_trajs) -> torch.Tensor:
        """LSTM-threaded ABC loss over a batch of complete trajectories.

        Extracted from the mini-batch loop so it runs once per update() rather
        than num_learning_epochs × num_mini_batches times on the same trajectories.
        """
        n_t = len(abc_trajs)
        max_T = max(t["obs"].shape[0] for t in abc_trajs)
        obs_dim = abc_trajs[0]["obs"].shape[-1]
        act_dim = abc_trajs[0]["acts"].shape[-1]

        padded_obs = torch.zeros(max_T, n_t, obs_dim, device=self.device)
        padded_acts = torch.zeros(max_T, n_t, act_dim, device=self.device)
        padded_old_lp = torch.zeros(max_T, n_t, device=self.device)
        traj_valid = torch.zeros(max_T, n_t, dtype=torch.bool, device=self.device)

        for i, traj in enumerate(abc_trajs):
            T = traj["obs"].shape[0]
            padded_obs[:T, i] = traj["obs"]
            padded_acts[:T, i] = traj["acts"]
            padded_old_lp[:T, i] = traj["old_lp"]
            traj_valid[:T, i] = True

        h = torch.zeros(n_t, self.actor_critic.lstm_hidden_size, device=self.device)
        c = torch.zeros(n_t, self.actor_critic.lstm_hidden_size, device=self.device)

        seq_log_probs = []
        for t in range(max_T):
            # Fix 14: GoalEncoder receives ABC gradients (detach_goal_encoder removed)
            raw, (h, c) = self.actor_critic._actor_forward(
                padded_obs[t], (h, c), detach_goal_encoder=False
            )
            dist = self.actor_critic._make_distribution(raw)
            lp = dist.log_prob(padded_acts[t].long())
            seq_log_probs.append(lp)

        seq_log_probs = torch.stack(seq_log_probs, dim=0)  # (max_T, n_t)
        bc_ratio = torch.exp(seq_log_probs - padded_old_lp)
        bc_clipped = torch.clamp(bc_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        return -torch.min(bc_ratio, bc_clipped)[traj_valid].mean()

    def update(self, alice_mean_rew: float = 0.0):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_bc_loss = 0

        effective_abc_coef = self.abc_coef  # always active per Fix 9

        abc_trajs = None
        if (
            effective_abc_coef > 0.0
            and self.abc_buffer is not None
            and len(self.abc_buffer._traj_store) > 0
        ):
            abc_trajs = self.abc_buffer.sample_trajectories(self.abc_n_trajs)

        for epoch in range(self.num_learning_epochs):
            # Fix 5: compute ABC loss once per epoch so it can be combined with PPO loss
            # in the last mini-batch → single optimizer step with L = L_PPO + β·L_ABC (per paper)
            epoch_bc_loss = None
            if abc_trajs is not None and len(abc_trajs) > 0:
                epoch_bc_loss = self._compute_abc_loss_sequential(abc_trajs)

            # FIX Bug 1: regenerate the BatchSampler each epoch.
            # BatchSampler is a one-shot iterator — reusing the same object
            # across epochs means epochs 2+ silently yield zero mini-batches.
            mini_batches = list(self.storage.mini_batch_generator(self.num_mini_batches))
            for batch_idx, indices in enumerate(mini_batches):
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

                hidden_full = self.storage.get_hidden_states()
                if hidden_full is not None:
                    hidden_batch = (hidden_full[0][indices], hidden_full[1][indices])
                else:
                    hidden_batch = None

                (
                    actions_log_prob_batch,
                    entropy_batch,
                    value_batch,
                    mu_batch,
                    sigma_batch,
                ) = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch, hidden_state=hidden_batch)

                # Fix 16: KL adaptive LR is dead in MC mode (sigma=zeros → KL=0 always).
                # Guard with use_multicategorical to avoid misleading code path.
                if self.desired_kl is not None and self.schedule == "adaptive" and not self.actor_critic.use_multicategorical:
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

                # --- Auxiliary Distance Prediction Loss ---
                aux_loss_val = torch.tensor(0.0, device=self.device)
                if getattr(self.actor_critic, "use_goal_encoder", False) and hasattr(
                    self.actor_critic.goal_encoder, "aux_loss"
                ):
                    # Extract current poses and goal poses dynamically using obs layout constants.
                    # Old code had hardcoded offsets [8:15, 32:39] / [23:30, 47:54] baked for the
                    # 56D quat obs — after the Euler switch these silently truncate and give wrong dims.
                    ge = self.actor_critic
                    r = ge._ge_robot_dim  # robot state dims (7 after Euler switch)
                    od = ge._ge_obj_state_dim  # obj state dims per object (14)
                    gd = ge._ge_goal_dim  # goal dims per object (6 = pos+euler)
                    ro = ge._ge_raw_per_obj  # raw chunk per object (22 = 14+6+2)

                    curr_poses, goal_poses = [], []
                    for _i in range(ge._ge_num_objects):
                        chunk = r + _i * ro
                        # Current pose = first gd dims of the object state (pos+euler)
                        curr_poses.append(obs_batch[:, chunk : chunk + gd])
                        # Goal pose = the gd-dim goal block inside each object chunk
                        goal_poses.append(obs_batch[:, chunk + od : chunk + od + gd])

                    s_t_batch = torch.cat(curr_poses, dim=-1)  # (batch, N*gd)
                    s_star_batch = torch.cat(goal_poses, dim=-1)  # (batch, N*gd)

                    aux, pos_loss, rot_loss = self.actor_critic.goal_encoder.aux_loss(
                        s_star_batch, s_t_batch
                    )
                    aux_loss_val = self.cfg_train["learn"].get("aux_coef", 0.1) * aux

                    if hasattr(self, "writer") and self.writer is not None:
                        it = self.current_learning_iteration
                        self.writer.add_scalar(
                            "GoalEncoder/aux_pos_loss", pos_loss.item(), it
                        )
                        self.writer.add_scalar(
                            "GoalEncoder/aux_rot_loss", rot_loss.item(), it
                        )

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_mean
                    + aux_loss_val
                )

                # Fix 5: on the last mini-batch of each epoch, add ABC loss so gradients
                # co-mingle in a single optimizer step (L = L_PPO + β·L_ABC per paper)
                is_last_batch = (batch_idx == len(mini_batches) - 1)
                if is_last_batch and epoch_bc_loss is not None:
                    loss = loss + effective_abc_coef * epoch_bc_loss
                    mean_bc_loss += epoch_bc_loss.item()

                if not torch.isfinite(loss):
                    print(f"[PPO] WARNING: skipping update — non-finite loss ({loss.item():.4f})", flush=True)
                    continue

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
        if self.num_learning_epochs > 0:
            mean_bc_loss /= self.num_learning_epochs

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_bc_loss,
            0.0,
        )  # 0.0 for IDM loss (removed)
