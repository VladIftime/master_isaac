import isaaclab.app
from isaaclab.app import AppLauncher

import math
import os
import sys
import yaml
import argparse
from datetime import datetime
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SuppressAllOutput:
    """Context manager that silences both C-level (stdout/stderr fd) and Python-level output."""

    def __enter__(self):
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)
        self.devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.devnull, self.stdout_fd)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout, self.stdout_fd)
        os.dup2(self.saved_stderr, self.stderr_fd)
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)
        os.close(self.devnull)
        if exc_type:
            print(f"Error occurred while suppressed: {exc_val}", file=sys.stderr)


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Async Dual Play PPO")
    parser.add_argument("--exp_name", type=str, default="async_dual_play_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=None, help="Override rollout steps per env (to prevent OOM on high num_envs)")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--max_alice_bob_ratio", type=int, default=None,
                        help="Max consecutive Bob-only updates before forcing an Alice update. "
                             "Auto-computed as ceil(bob_timesteps / alice_timesteps) * max(1, 64 // num_envs) "
                             "if not specified.")
    parser.add_argument("--chkpt_alice", type=str, default=None)
    parser.add_argument("--chkpt_bob", type=str, default=None)
    parser.add_argument(
        "--arm_config", type=str, default="default",
        choices=["default", "rotated"],
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer
    from asyncDualPlayPPO.utils.historical_pool import HistoricalPolicyPool
    # Import reward constants so train.py stays in sync with rewards.py.
    # Do not hardcode reward values here — change them in rewards.py instead.
    from asyncDualPlayPPO.tasks.utils.rewards import (
        ALICE_BOB_FAIL_REWARD,
        ALICE_BOB_SUCCESS_REWARD,
        ALICE_VALID_GOAL_BONUS,
        ALICE_INVALID_GOAL_PENALTY,
    )

    # --- Environment ---
    task_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/task/AsyncDualPlay.yaml")
    ppo_cfg_path  = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg = load_cfg(ppo_cfg_path)

    if args.nsteps is not None:
        print(f"[Config] Overriding nsteps: {ppo_cfg['params']['learn']['nsteps']} -> {args.nsteps}")
        ppo_cfg["params"]["learn"]["nsteps"] = args.nsteps

    # --- Multi-categorical action space config ---
    _pol_cfg     = ppo_cfg["params"]["policy"]
    use_mc       = _pol_cfg.get("use_multicategorical", False)
    num_cat_dims = _pol_cfg.get("num_cat_dims", 4)
    num_bins     = _pol_cfg.get("num_bins", 11)
    max_delta_m  = _pol_cfg.get("max_delta_m", 0.05)
    if use_mc:
        print(f"[Config] Multi-categorical action space: {num_cat_dims} dims × {num_bins} bins "
              f"(max delta {max_delta_m*100:.1f} cm, bin size {max_delta_m*100/(num_bins-1)*10:.1f} mm)")

    def bins_to_env_action(bin_indices: "torch.Tensor", gripper_state: "torch.Tensor") -> "torch.Tensor":
        """
        Convert policy bin indices (N, 4) → 7D RMPFlow+gripper env action.

        XYZ: delta = (bin - center) / center * max_delta_m
          env scale=0.05 → 5 cm/step at max bin.

        Gripper (sticky): only the outer bins trigger a state change.
          Dead zone = center 3 bins (4/5/6) → keep previous gripper_state.
          This prevents random-policy spassing at the start of training.
          Threshold: bins 0-3 → close (-1), bins 4-6 → hold, bins 7-10 → open (+1)
        """
        center    = (num_bins - 1) / 2.0          # 5.0 for 11 bins
        threshold = 2.0                            # ±2 bins from center triggers change
        normalized = (bin_indices.float() - center) / center
        xyz        = normalized[:, :3] * max_delta_m

        g_bin      = bin_indices[:, 3].float()
        new_gs     = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0   # bins 0-2  → close
        new_gs[g_bin > center + threshold - 1] =  1.0   # bins 8-10 → open
        # bins 3-7 → keep previous state

        zeros3 = torch.zeros(bin_indices.shape[0], 3, device=bin_indices.device)
        return torch.cat([normalized[:, :3] * max_delta_m, zeros3, new_gs], dim=-1), new_gs

    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    if args.arm_config == "rotated":
        print("[Config] Rotated arm configuration: left shoulder −90°, right shoulder +90°")
        env_cfg.scene.robot.init_state.joint_pos["left_shoulder_pan_joint"]  = -1.57
        env_cfg.scene.robot.init_state.joint_pos["right_shoulder_pan_joint"] =  1.57

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = AsyncDualPlayEnvWrapper(env=base_env, device=base_env.device, arm_config=args.arm_config)

    # --- Agents ---
    alice_ppo = PPO(
        vec_env=env,
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/alice",
        asymmetric=False,
    )
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space

    # Override action space: policy operates on bin indices (4D), not the 7D env action
    if use_mc:
        import gymnasium as gym_mc
        _mc_space = gym_mc.spaces.Box(low=0.0, high=float(num_bins - 1),
                                       shape=(num_cat_dims,), dtype=np.float32)
        alice_ppo.action_space = _mc_space
        alice_ppo.desired_kl   = None   # adaptive KL meaningless for discrete

    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape,
        alice_ppo.action_space.shape,
        alice_ppo.init_noise_std,
        alice_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    max_alice_steps = env.episode_manager.alice_timesteps + 10
    alice_storage_size = alice_ppo.num_transitions_per_env + max_alice_steps
    alice_ppo.storage = alice_ppo.storage.__class__(
        alice_ppo.vec_env.num_envs,
        alice_storage_size,
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape,
        alice_ppo.action_space.shape,
        alice_ppo.device,
        "sequential",
    )
    alice_ppo.optimizer = torch.optim.Adam(
        alice_ppo.actor_critic.parameters(), lr=alice_ppo.learning_rate
    )

    bob_ppo = PPOABC(
        vec_env=env,
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/bob",
        asymmetric=False,
    )
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space

    if use_mc:
        bob_ppo.action_space = _mc_space   # same 4D bin space as Alice
        bob_ppo.desired_kl   = None

    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.init_noise_std,
        bob_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)
    bob_ppo.optimizer = torch.optim.Adam(
        bob_ppo.actor_critic.parameters(), lr=bob_ppo.learning_rate
    )
    
    # Re-initialize Bob's standard PPO storage for 56 dims
    bob_ppo.storage = bob_ppo.storage.__class__(
        bob_ppo.vec_env.num_envs,
        bob_ppo.num_transitions_per_env,
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.device,
        "sequential",
    )
    
    # ABC Buffer for Alice's successful demonstrations
    # actions_shape must match policy action dim (4D bin indices for MC, 7D for Gaussian)
    _abc_act_shape = (num_cat_dims,) if use_mc else env.action_space.shape
    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
        capacity=100000,
        obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape,
        actions_shape=_abc_act_shape,
        device=env.device,
    )

    # --- LSTM hidden state management (Fix 4) ---
    # act_with_hidden() is always used; for non-LSTM models it returns None hidden.
    _use_lstm = alice_ppo.actor_critic.use_lstm
    if _use_lstm:
        _lsz = alice_ppo.actor_critic.lstm_hidden_size
        alice_hidden = [torch.zeros(env.num_envs, _lsz, device=env.device),
                        torch.zeros(env.num_envs, _lsz, device=env.device)]
        bob_hidden   = [torch.zeros(env.num_envs, _lsz, device=env.device),
                        torch.zeros(env.num_envs, _lsz, device=env.device)]
    else:
        alice_hidden = None
        bob_hidden   = None

    # --- Sticky gripper state ---
    # Starts open (+1). Only outer bins (0-2 = close, 8-10 = open) change state;
    # center bins (3-7) hold previous state to prevent random-policy spassing.
    alice_gripper_state = torch.ones(env.num_envs, 1, device=env.device)
    bob_gripper_state   = torch.ones(env.num_envs, 1, device=env.device)

    # --- Historical policy pool (Fix 6) ---
    # Paper: 20% of rollout envs use a past Alice/Bob policy for stability.
    alice_pool = HistoricalPolicyPool(max_size=5)
    bob_pool   = HistoricalPolicyPool(max_size=5)
    HIST_SAVE_INTERVAL = 50  # save snapshot every N bob_updates
    HIST_FRAC          = 0.2  # fraction of envs using historical policy

    # --- Agents ---
    alice_updates = 0
    bob_updates   = 0
    consecutive_alice_skips = 0

    # Resolve max_alice_bob_ratio: if not provided, derive from episode timesteps and num_envs.
    # bob_timesteps / alice_timesteps gives the base ratio (Bob needs proportionally more gradient steps).
    # The num_envs factor scales it down: more envs → more samples per iteration → Bob catches up faster.
    if args.max_alice_bob_ratio is None:
        _bob_ts   = env.episode_manager.bob_timesteps
        _alice_ts = env.episode_manager.alice_timesteps
        max_alice_bob_ratio = max(2, math.ceil(_bob_ts / _alice_ts) * max(1, 64 // args.num_envs))
        print(f"[Config] max_alice_bob_ratio auto-computed: {max_alice_bob_ratio} "
              f"(bob_ts={_bob_ts}, alice_ts={_alice_ts}, num_envs={args.num_envs})")
    else:
        max_alice_bob_ratio = args.max_alice_bob_ratio

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")

    rollout_length = ppo_cfg["params"]["learn"]["nsteps"] * args.num_envs
    alice_rew_buf    = deque(maxlen=rollout_length)
    bob_rew_buf      = deque(maxlen=rollout_length)
    bob_success_buf  = deque(maxlen=rollout_length)
    bob_pos_err_buf  = deque(maxlen=rollout_length)
    bob_rot_err_buf  = deque(maxlen=rollout_length)

    best_bob_success_rate = -1.0

    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print(f"\n{'='*80}\nTRAINING RUN: {args.exp_name}\nLOG DIRECTORY: {run_dir}")
    print(f"  tensorboard --logdir {run_dir}\n{'='*80}\n")

    def perform_alice_update():
        """Run a PPO update for Alice after her rollout is complete."""
        nonlocal alice_updates
        
        # In sequential mode, storage.step MUST be exactly num_transitions_per_env
        if alice_ppo.storage.step < alice_ppo.storage.num_transitions_per_env:
             return

        dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
        alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
        loss_val, loss_surr, _, _ = alice_ppo.update()
        alice_ppo.storage.clear()

        mean_alice_rew = np.mean(alice_rew_buf) if alice_rew_buf else 0.0
        writer.add_scalar("Loss/Alice/Value",     loss_val,       alice_updates)
        writer.add_scalar("Loss/Alice/Surrogate", loss_surr,      alice_updates)
        writer.add_scalar("Reward/Alice",         mean_alice_rew, alice_updates)

        print(f"  [Alice Update {alice_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_alice_rew:.4f}", flush=True)
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_bob_obs):
        """Run a PPO + ABC update for Bob after his rollout is complete."""
        nonlocal bob_updates, best_bob_success_rate
        
        if bob_ppo.storage.step < bob_ppo.storage.num_transitions_per_env:
            return
            
        with torch.no_grad():
            _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(current_bob_obs, None)
        
        # Dynamic ABC weight decay: 0.5 -> 0.01 over max_iterations
        _abc_coef_init = ppo_cfg["params"]["learn"].get("abc_coef", 0.5)
        bob_ppo.abc_coef = max(0.01, _abc_coef_init * (1.0 - (bob_updates / args.max_iterations)))
        
        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        loss_val, loss_surr, loss_abc, _ = bob_ppo.update()
        bob_ppo.storage.clear()

        mean_bob_rew     = np.mean(bob_rew_buf)     if bob_rew_buf     else 0.0
        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        mean_pos_err     = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot_err     = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0

        writer.add_scalar("Loss/Bob/Value",       loss_val,         bob_updates)
        writer.add_scalar("Loss/Bob/Surrogate",   loss_surr,        bob_updates)
        writer.add_scalar("Loss/Bob/ABC",         loss_abc,         bob_updates)
        writer.add_scalar("Reward/Bob",           mean_bob_rew,     bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
        writer.add_scalar("Metrics/Bob/PosError",    mean_pos_err,     bob_updates)
        writer.add_scalar("Metrics/Bob/RotError",    mean_rot_err,     bob_updates)

        print(f"  [Bob Update {bob_updates}] Loss: {loss_surr:.4f} | ABC Loss: {loss_abc:.4f} | SR: {bob_success_rate:.4f}", flush=True)

        if args.save_interval > 0 and (bob_updates + 1) % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir,   f"model_{bob_updates+1}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{bob_updates+1}.pt"))

        if bob_success_rate > best_bob_success_rate:
            best_bob_success_rate = bob_success_rate
            bob_ppo.save(os.path.join(bob_ppo.log_dir,   "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))

        bob_rew_buf.clear()
        bob_success_buf.clear()
        bob_pos_err_buf.clear()
        bob_rot_err_buf.clear()
        
        bob_updates += 1

    print("Initializing environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment initialized. Starting training loop...")

    # Target max timesteps from env configuration
    target_alice_timesteps = env.episode_manager.alice_timesteps

    while bob_updates < args.max_iterations:
        
        # --- 0. SETUP: reset LSTM hidden states and snapshot policies ---
        if alice_hidden is not None:
            alice_hidden[0].zero_()
            alice_hidden[1].zero_()
        if bob_hidden is not None:
            bob_hidden[0].zero_()
            bob_hidden[1].zero_()

        # Periodically snapshot current policies to historical pool (Fix 6)
        if bob_updates > 0 and bob_updates % HIST_SAVE_INTERVAL == 0:
            alice_pool.add(alice_ppo.actor_critic)
            bob_pool.add(bob_ppo.actor_critic)
            print(f"  [HistPool] Saved snapshot at iter {bob_updates} "
                  f"(alice pool={alice_pool.size}, bob pool={bob_pool.size})", flush=True)

        # --- 0. CURRICULUM UPDATE ---
        # Slowly increase Alice's horizon so Bob isn't overwhelmed with max-distance goals at iteration 0
        curriculum_steps = min(target_alice_timesteps, 100 + int(bob_updates * (target_alice_timesteps / 200.0)))
        env.episode_manager.alice_timesteps = curriculum_steps

        # --- 1. ALICE ROLLOUT PHASE ---
        alice_ppo.storage.clear()
        
        # Reset all envs to Alice phase at start of iteration
        env.episode_manager.reset_episode(torch.arange(env.num_envs, device=env.device), reason="Iteration Start")
        
        # Use wrapper's reset/step returned obs or re-compute and slice
        obs_dict = env.env.observation_manager.compute()
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_alice_obs = obs[:, :env.alice_obs_dim]
        
        # Collect S0 for ABC 
        env.episode_manager.store_initial_state(env._extract_object_states(obs_dict))
        
        # Pre-allocate iteration buffers for ABC 
        alice_traj_obs = [] # list of (num_envs, obs_dim)
        alice_traj_act = [] # list of (num_envs, act_dim)

        # Sample a historical Alice policy for ~20% of envs (Fix 6)
        hist_alice = alice_pool.sample_policy(alice_ppo.actor_critic, env.device) if alice_pool.size > 0 else None

        for t in range(env.episode_manager.alice_timesteps):
            # Capture where we are in alice phase
            is_alice = env.episode_manager.is_alice_phase()
            alice_indices = torch.where(is_alice)[0]

            if len(alice_indices) == 0:
                break

            # Split active envs: hist_ids use saved policy, curr_ids use current
            hist_ids, curr_ids = alice_pool.sample_env_subset(alice_indices, frac=HIST_FRAC)

            with torch.no_grad():
                # Current Alice (majority)
                h_in = (alice_hidden[0][curr_ids], alice_hidden[1][curr_ids]) if alice_hidden else None
                a_acts_curr, a_logprob_curr, a_val_curr, a_mu_curr, a_sigma_curr, new_h = \
                    alice_ppo.actor_critic.act_with_hidden(current_alice_obs[curr_ids], None, h_in)
                if alice_hidden and new_h is not None:
                    alice_hidden[0][curr_ids] = new_h[0]
                    alice_hidden[1][curr_ids] = new_h[1]

                # Historical Alice (minority, no grad tracking needed)
                if len(hist_ids) > 0 and hist_alice is not None:
                    a_acts_hist, a_logprob_hist, a_val_hist, a_mu_hist, a_sigma_hist, _ = \
                        hist_alice.act_with_hidden(current_alice_obs[hist_ids], None, None)
                else:
                    # Fallback to current if no history yet
                    hist_ids = torch.tensor([], dtype=torch.long, device=env.device)
                    a_acts_hist = a_logprob_hist = a_val_hist = a_mu_hist = a_sigma_hist = None

            # Policy action dim: 4 bin indices (MC) or 7 continuous (Gaussian)
            _a_pdim = num_cat_dims if use_mc else env.action_space.shape[0]

            # Merge curr + hist actions into full alice_indices tensors
            a_acts_active    = torch.zeros((len(alice_indices), _a_pdim), device=env.device)
            a_logprob_active = torch.zeros(len(alice_indices), device=env.device)
            a_val_active     = torch.zeros(len(alice_indices), 1, device=env.device)
            a_mu_active      = torch.zeros_like(a_acts_active)
            a_sigma_active   = torch.zeros_like(a_acts_active)

            curr_local = torch.searchsorted(alice_indices, curr_ids)
            a_acts_active[curr_local]    = a_acts_curr
            a_logprob_active[curr_local] = a_logprob_curr
            a_val_active[curr_local]     = a_val_curr
            a_mu_active[curr_local]      = a_mu_curr
            a_sigma_active[curr_local]   = a_sigma_curr

            if len(hist_ids) > 0 and a_acts_hist is not None:
                hist_local = torch.searchsorted(alice_indices, hist_ids)
                a_acts_active[hist_local]    = a_acts_hist
                a_logprob_active[hist_local] = a_logprob_hist
                a_val_active[hist_local]     = a_val_hist
                a_mu_active[hist_local]      = a_mu_hist
                a_sigma_active[hist_local]   = a_sigma_hist

            alice_traj_obs.append(current_alice_obs.clone())

            # Policy actions for storage/ABC (bin indices or continuous)
            a_policy = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_policy[alice_indices] = a_acts_active
            alice_traj_act.append(a_policy.clone())   # ABC buffer gets bin indices

            # Full-env tensors for storage (non-active envs get zeros)
            a_lp_full    = torch.zeros(env.num_envs, device=env.device)
            a_val_full   = torch.zeros(env.num_envs, 1, device=env.device)
            a_mu_full    = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_sigma_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_lp_full[alice_indices]    = a_logprob_active
            a_val_full[alice_indices]   = a_val_active
            a_mu_full[alice_indices]    = a_mu_active
            a_sigma_full[alice_indices] = a_sigma_active

            # 7D env action for RMPFlow: convert bins → deltas, zero-pad rotation
            if use_mc:
                a_env_full = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
                a_act_7d, new_ags = bins_to_env_action(a_acts_active, alice_gripper_state[alice_indices])
                a_env_full[alice_indices] = a_act_7d
                alice_gripper_state[alice_indices] = new_ags
            else:
                a_env_full = a_policy   # already 7D continuous

            obs_full, rewards, dones, truncated, extras = env.step(a_env_full)

            # Reset LSTM hidden state and gripper state for envs that terminated
            if alice_hidden is not None:
                done_alice = alice_indices[dones[alice_indices]]
                if len(done_alice) > 0:
                    alice_hidden[0][done_alice] = 0.0
                    alice_hidden[1][done_alice] = 0.0
            alice_gripper_state[alice_indices[dones[alice_indices]]] = 1.0  # reset to open

            # Storage masking
            a_masks = torch.zeros(env.num_envs, 1, device=env.device)
            a_masks[alice_indices[~dones[alice_indices]]] = 1.0

            # Store policy actions (bins) and real log_probs/values for PPO update
            next_alice_obs = obs_full[:, :env.alice_obs_dim]
            alice_ppo.storage.add_transitions(
                current_alice_obs, next_alice_obs, a_policy, rewards, dones,
                a_val_full, a_lp_full, a_mu_full, a_sigma_full, a_masks
            )
            current_alice_obs = next_alice_obs

        # --- 1.5 SETTLE PHYSICS ---
        # Run 20 zero-action steps to let objects stop sliding/bouncing before extracting the goal.
        # Use env.env (ManagerBasedRLEnv) directly to avoid advancing the EpisodeManager state.
        for _ in range(20):
            settle_acts = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
            env.env.step(settle_acts)
            
        # Manually overwrite the goal states to the newly settled states.
        # Apply the same 14D slice as wrapper._handle_alice_completion:
        #   full state is (num_envs, 30) = 2 objects × 15 features
        #   keep only pos+orient (first 7 per object) → (num_envs, 14)
        obs_dict = env.env.observation_manager.compute()
        settled_goal_states = env._extract_object_states(obs_dict)  # (N, 30)
        settled_goal_states_14d = settled_goal_states.view(-1, 2, 15)[:, :, :7].reshape(-1, 14)
        env.episode_manager.store_goal_state(settled_goal_states_14d, torch.arange(env.num_envs, device=env.device))
        
        # Alice Phase Done. Goal states extracted by wrapper during transition.
        goal_states = env.episode_manager.goal_states
        
        # --- 2. BOB ROLLOUT PHASE ---
        bob_ppo.storage.clear()
        
        # Env already reset to S0 by wrapper during transition
        obs_dict = env.env.observation_manager.compute()
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_bob_obs = obs[:, env.alice_obs_dim:]
        
        # Sample a historical Bob policy for ~20% of envs (Fix 6)
        hist_bob = bob_pool.sample_policy(bob_ppo.actor_critic, env.device) if bob_pool.size > 0 else None

        for t in range(env.episode_manager.bob_timesteps):
            is_bob = env.episode_manager.is_bob_phase()
            bob_indices = torch.where(is_bob)[0]
            if len(bob_indices) == 0: break

            # Split active envs: hist_ids use saved policy, curr_ids use current (Fix 6)
            hist_bids, curr_bids = bob_pool.sample_env_subset(bob_indices, frac=HIST_FRAC)

            bob_obs_active = current_bob_obs[bob_indices]

            with torch.no_grad():
                # Current Bob (majority)
                h_in = (bob_hidden[0][curr_bids], bob_hidden[1][curr_bids]) if bob_hidden else None
                b_acts_curr, b_lp_curr, b_val_curr, b_mu_curr, b_sig_curr, new_bh = \
                    bob_ppo.actor_critic.act_with_hidden(current_bob_obs[curr_bids], None, h_in)
                if bob_hidden and new_bh is not None:
                    bob_hidden[0][curr_bids] = new_bh[0]
                    bob_hidden[1][curr_bids] = new_bh[1]

                # Historical Bob (minority)
                if len(hist_bids) > 0 and hist_bob is not None:
                    b_acts_hist, b_lp_hist, b_val_hist, b_mu_hist, b_sig_hist, _ = \
                        hist_bob.act_with_hidden(current_bob_obs[hist_bids], None, None)
                else:
                    hist_bids = torch.tensor([], dtype=torch.long, device=env.device)
                    b_acts_hist = b_lp_hist = b_val_hist = b_mu_hist = b_sig_hist = None

            # Policy action dim: 4 bin indices (MC) or 7 continuous (Gaussian)
            _b_pdim = num_cat_dims if use_mc else env.action_space.shape[0]

            # Merge curr + hist into full bob_indices tensors
            b_acts_active    = torch.zeros((len(bob_indices), _b_pdim), device=env.device)
            b_logprob_active = torch.zeros(len(bob_indices), device=env.device)
            b_val_active     = torch.zeros(len(bob_indices), 1, device=env.device)
            b_mu_active      = torch.zeros_like(b_acts_active)
            b_sigma_active   = torch.zeros_like(b_acts_active)

            curr_bloc = torch.searchsorted(bob_indices, curr_bids)
            b_acts_active[curr_bloc]    = b_acts_curr
            b_logprob_active[curr_bloc] = b_lp_curr
            b_val_active[curr_bloc]     = b_val_curr
            b_mu_active[curr_bloc]      = b_mu_curr
            b_sigma_active[curr_bloc]   = b_sig_curr

            if len(hist_bids) > 0 and b_acts_hist is not None:
                hist_bloc = torch.searchsorted(bob_indices, hist_bids)
                b_acts_active[hist_bloc]    = b_acts_hist
                b_logprob_active[hist_bloc] = b_lp_hist
                b_val_active[hist_bloc]     = b_val_hist
                b_mu_active[hist_bloc]      = b_mu_hist
                b_sigma_active[hist_bloc]   = b_sig_hist

            # 7D env action for RMPFlow
            if use_mc:
                b_env_full = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
                b_act_7d, new_bgs = bins_to_env_action(b_acts_active, bob_gripper_state[bob_indices])
                b_env_full[bob_indices] = b_act_7d
                bob_gripper_state[bob_indices] = new_bgs
            else:
                b_env_full = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
                b_env_full[bob_indices] = b_acts_active

            obs_full, rewards, dones, truncated, extras = env.step(b_env_full)

            bob_done_this_step = extras.get("bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            ended_for_bob = dones | bob_done_this_step
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0

            # Storage-ready tensors — policy actions (bins) stored, not env actions
            _obs     = torch.zeros((env.num_envs, env.bob_obs_dim), device=env.device);  _obs[bob_indices]     = bob_obs_active
            _next_obs= torch.zeros((env.num_envs, env.bob_obs_dim), device=env.device);  _next_obs[bob_indices]= obs_full[bob_indices, env.alice_obs_dim:]
            _acts    = torch.zeros((env.num_envs, _b_pdim),          device=env.device);  _acts[bob_indices]    = b_acts_active
            _rew     = torch.zeros(env.num_envs, device=env.device);                       _rew[bob_indices]     = rewards[bob_indices]
            _val     = torch.zeros(env.num_envs, 1, device=env.device);                   _val[bob_indices]     = b_val_active
            _lp      = torch.zeros(env.num_envs, 1, device=env.device);                   _lp[bob_indices]      = b_logprob_active.unsqueeze(1)
            _mu      = torch.zeros((env.num_envs, _b_pdim),          device=env.device);  _mu[bob_indices]      = b_mu_active
            _sigma   = torch.zeros((env.num_envs, _b_pdim),          device=env.device);  _sigma[bob_indices]   = b_sigma_active

            # Reset LSTM hidden state and gripper state for terminated envs
            done_bob = bob_indices[dones[bob_indices]]
            if bob_hidden is not None and len(done_bob) > 0:
                bob_hidden[0][done_bob] = 0.0
                bob_hidden[1][done_bob] = 0.0
            if len(done_bob) > 0:
                bob_gripper_state[done_bob] = 1.0  # reset to open

            bob_ppo.storage.add_transitions(_obs, _next_obs, _acts, _rew, dones.clone(), _val, _lp, _mu, _sigma, b_masks)
            current_bob_obs = obs_full[:, env.alice_obs_dim:]
            bob_rew_buf.extend(rewards.cpu().numpy().tolist())

        # --- 3. ALICE BEHAVIORAL CLONING (ABC) BUFFER PUSH ---
        goal_valid = env.episode_manager.goal_valid
        bob_success = env.episode_manager.bob_success

        # Paper: ABC demos added ONLY when Bob failed (ξ=False).
        # When Bob succeeds, Alice's trajectory is already learnable — no demo needed.
        # Providing demos on success wastes buffer capacity and dilutes the BC signal.
        if goal_valid.any():
            valid_ids = torch.where(goal_valid & ~bob_success)[0]
            for env_id in valid_ids:
                eid = env_id.item()
                # Construct demo trajectory
                # traj_obs is a list of (num_envs, obs_dim) tensors
                traj_o = torch.stack([alice_traj_obs[step][eid] for step in range(len(alice_traj_obs))])
                traj_a = torch.stack([alice_traj_act[step][eid] for step in range(len(alice_traj_act))])
                
                # Goal for this env
                g = goal_states[eid].unsqueeze(0).expand(len(traj_o), -1)
                
                # Bob-compatible obs: robot (8) + objects (30) + goal (14) + dist (4) = 56
                bc_obs = env.construct_bob_observation(traj_o, g)
                
                # Evaluate Bob's CURRENT policy on Alice's demo to get old_log_probs
                # for PPO-style ratio clipping in ppo_abc.py (prevents stale-demo update explosion)
                with torch.no_grad():
                    old_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(bc_obs, None, traj_a)

                # Add to ABC buffer with old_log_probs for ratio clipping
                bob_ppo.abc_buffer.add_trajectory(
                    bc_obs, bc_obs, traj_a,
                    torch.zeros(len(traj_o), device=env.device), torch.zeros(len(traj_o), device=env.device).byte(),
                    torch.zeros(len(traj_o), device=env.device), old_lp.view(-1, 1),
                    torch.zeros_like(traj_a), torch.zeros_like(traj_a),
                    torch.zeros(len(traj_o), 1, device=env.device), torch.zeros(len(traj_o), 1, device=env.device)
                )

        # --- 4. ALICE REWARD ASSIGNMENT & UPDATE ---
        # Alice Reward: she gets ALICE_BOB_FAIL_REWARD if Bob failed AND the goal was valid
        alice_outcome_rewards = torch.where(
            ~bob_success & goal_valid, 
            torch.tensor(ALICE_BOB_FAIL_REWARD, device=env.device, dtype=torch.float32), 
            torch.tensor(0.0, device=env.device, dtype=torch.float32)
        )
        
        # Inject outcome reward into Alice's last storage step
        if alice_ppo.storage.step > 0:
            last_idx = alice_ppo.storage.step - 1
            alice_ppo.storage.rewards[last_idx].copy_(alice_outcome_rewards.view(-1, 1))
            # FIX: Prevent GAE bleeding by forcing the terminal state to be 'done'
            alice_ppo.storage.dones[last_idx].fill_(1.0)
            
            alice_rew_buf.extend(alice_outcome_rewards.cpu().numpy().tolist())

        # Per-goal success rate: successes / goals_attempted across all envs this iteration
        # (Paper metric: ξ tracks cumulative per-goal outcomes, not per-episode boolean)
        total_attempted = env.episode_manager.goals_attempted.sum().item()
        total_succeeded = env.episode_manager.goals_succeeded.sum().item()
        current_sr = total_succeeded / max(1, total_attempted)
        bob_success_buf.append(current_sr)
        
        # Perform Updates
        # Freeze Alice if Bob is struggling (< 10% SR), but force an Alice update after
        # max_alice_bob_ratio consecutive skips to prevent Alice from going stale.
        force_alice = consecutive_alice_skips >= max_alice_bob_ratio
        if current_sr >= 0.10 or bob_updates < 10 or force_alice:
            if force_alice and current_sr < 0.10:
                print(f"  [Alice Update] Forced after {consecutive_alice_skips} skips "
                      f"(ratio={max_alice_bob_ratio}). Bob SR={current_sr:.2f}", flush=True)
            perform_alice_update()
            consecutive_alice_skips = 0
        else:
            consecutive_alice_skips += 1
            print(f"  [Alice Update] Skipped ({consecutive_alice_skips}/{max_alice_bob_ratio}). "
                  f"Bob SR ({current_sr:.2f}) < 0.10. Letting Bob catch up.", flush=True)
            alice_ppo.storage.clear()
            
        perform_bob_update(current_bob_obs)

        # Logging
        if bob_updates % 1 == 0:
             print(f"Iteration {bob_updates}: SR={current_sr:.2f} | ABC Buffer: {bob_ppo.abc_buffer.step if not bob_ppo.abc_buffer.full else 'FULL'}", flush=True)

    alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_final.pt"))
    bob_ppo.save(os.path.join(bob_ppo.log_dir,     "model_final.pt"))
    print("  ✓ Saved final models")
    writer.close()

    print(f"\n{'='*80}\nTRAINING COMPLETE\n{'='*80}")
    print(f"To resume:\n  python train.py --exp_name {args.exp_name}_resume \\")
    print(f"    --chkpt_alice runs/{args.exp_name}/alice/model_final.pt \\")
    print(f"    --chkpt_bob   runs/{args.exp_name}/bob/model_final.pt\n{'='*80}\n")


if __name__ == "__main__":
    main()
