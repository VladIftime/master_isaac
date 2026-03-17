import isaaclab.app
from isaaclab.app import AppLauncher

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
    parser.add_argument("--max_alice_bob_ratio", type=int, default=5,
                        help="Max Alice PPO updates per Bob update (prevents non-stationarity)")
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
    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
        capacity=100000,
        obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape,
        actions_shape=env.action_space.shape,
        device=env.device,
    )

    # --- Agents ---
    alice_updates = 0
    bob_updates   = 0
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

    while bob_updates < args.max_iterations:
        
        # --- 1. ALICE ROLLOUT PHASE ---
        alice_ppo.storage.clear()
        
        # Reset all envs to Alice phase at start of iteration
        # In a strict 1-to-1 setup, we reset everybody.
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

        for t in range(env.episode_manager.alice_timesteps):
            # Capture where we are in alice phase
            is_alice = env.episode_manager.is_alice_phase()
            alice_indices = torch.where(is_alice)[0]
            
            if len(alice_indices) == 0:
                break

            with torch.no_grad():
                # Slice Alice dims (38)
                a_acts_active, a_logprob_active, a_val_active, a_mu_active, a_sigma_active = alice_ppo.actor_critic.act(current_alice_obs[alice_indices], None)
            
            alice_traj_obs.append(current_alice_obs.clone())
            
            # Action mapping
            a_acts = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
            a_acts[alice_indices] = a_acts_active
            alice_traj_act.append(a_acts.clone())
            
            obs_full, rewards, dones, truncated, extras = env.step(a_acts)
            
            # Storage masking
            a_masks = torch.zeros(env.num_envs, 1, device=env.device)
            a_masks[alice_indices[~dones[alice_indices]]] = 1.0

            # Store transitions for Alice (38 dims)
            next_alice_obs = obs_full[:, :env.alice_obs_dim]
            alice_ppo.storage.add_transitions(
                current_alice_obs, next_alice_obs, a_acts, rewards, dones, 
                torch.zeros(env.num_envs, 1, device=env.device), # dummy values for Alice
                torch.zeros(env.num_envs, 1, device=env.device), # dummy logprobs
                torch.zeros_like(a_acts), torch.zeros_like(a_acts), # mus, sigmas
                a_masks
            )
            current_alice_obs = next_alice_obs

        # Alice Phase Done. Goal states extracted by wrapper during transition.
        goal_states = env.episode_manager.goal_states
        
        # --- 2. BOB ROLLOUT PHASE ---
        bob_ppo.storage.clear()
        
        # Env already reset to S0 by wrapper during transition
        obs_dict = env.env.observation_manager.compute()
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_bob_obs = obs[:, env.alice_obs_dim:]
        
        for t in range(env.episode_manager.bob_timesteps):
            is_bob = env.episode_manager.is_bob_phase()
            bob_indices = torch.where(is_bob)[0]
            if len(bob_indices) == 0: break

            # Slice Bob dims (56)
            bob_obs_active = current_bob_obs[bob_indices]
            with torch.no_grad():
                b_acts_active, b_logprob_active, b_val_active, b_mu_active, b_sigma_active = bob_ppo.actor_critic.act(bob_obs_active, None)
            
            b_acts = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
            b_acts[bob_indices] = b_acts_active
            obs_full, rewards, dones, truncated, extras = env.step(b_acts)
            
            bob_done_this_step = extras.get("bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            ended_for_bob = dones | bob_done_this_step
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0
            
            # Storage-ready tensors
            _obs = torch.zeros((env.num_envs, env.bob_obs_dim), device=env.device); _obs[bob_indices] = bob_obs_active
            _next_obs = torch.zeros((env.num_envs, env.bob_obs_dim), device=env.device); _next_obs[bob_indices] = obs_full[bob_indices, env.alice_obs_dim:]
            _acts = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device); _acts[bob_indices] = b_acts_active
            _rew = torch.zeros(env.num_envs, device=env.device); _rew[bob_indices] = rewards[bob_indices]
            _val = torch.zeros(env.num_envs, 1, device=env.device); _val[bob_indices] = b_val_active
            _lp = torch.zeros(env.num_envs, 1, device=env.device); _lp[bob_indices] = b_logprob_active.unsqueeze(1)
            _mu = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device); _mu[bob_indices] = b_mu_active
            _sigma = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device); _sigma[bob_indices] = b_sigma_active

            bob_ppo.storage.add_transitions(_obs, _next_obs, _acts, _rew, dones.clone(), _val, _lp, _mu, _sigma, b_masks)
            current_bob_obs = obs_full[:, env.alice_obs_dim:]
            bob_rew_buf.extend(rewards.cpu().numpy().tolist())

        # --- 3. ALICE BEHAVIORAL CLONING (ABC) BUFFER PUSH ---
        goal_valid = env.episode_manager.goal_valid
        bob_success = env.episode_manager.bob_success
        
        if goal_valid.any():
            valid_ids = torch.where(goal_valid)[0]
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
                
                # Add to ABC buffer for supervised learning
                bob_ppo.abc_buffer.add_trajectory(
                    bc_obs, bc_obs, traj_a, 
                    torch.zeros(len(traj_o), device=env.device), torch.zeros(len(traj_o), device=env.device).byte(),
                    torch.zeros(len(traj_o), device=env.device), torch.zeros(len(traj_o), 1, device=env.device),
                    torch.zeros_like(traj_a), torch.zeros_like(traj_a),
                    torch.zeros(len(traj_o), 1, device=env.device), torch.zeros(len(traj_o), 1, device=env.device)
                )

        # --- 4. ALICE REWARD ASSIGNMENT & UPDATE ---
        # Alice Reward: she gets 1.0 if Bob failed, 0.0 if he succeeded.
        alice_outcome_rewards = torch.where(bob_success, torch.tensor(0.0, device=env.device), torch.tensor(1.0, device=env.device))
        
        # Inject outcome reward into Alice's last storage step
        if alice_ppo.storage.step > 0:
            last_idx = alice_ppo.storage.step - 1
            alice_ppo.storage.rewards[last_idx].copy_(alice_outcome_rewards.view(-1, 1))
            alice_rew_buf.extend(alice_outcome_rewards.cpu().numpy().tolist())

        # Metrics for the iteration
        current_sr = bob_success.float().mean().item()
        bob_success_buf.append(current_sr)
        
        # Perform Updates
        perform_alice_update()
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
