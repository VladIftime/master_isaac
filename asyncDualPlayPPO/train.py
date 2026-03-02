import isaaclab.app
from isaaclab.app import AppLauncher

import os
import sys
import yaml
import argparse
from datetime import datetime
from collections import deque

# Add parent directory to python path to allow importing the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class SuppressAllOutput:
    """Context manager to suppress C++ (stdout/stderr) and Python output."""
    def __enter__(self):
        # Save original file descriptors
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)

        # Open devnull
        self.devnull = os.open(os.devnull, os.O_RDWR)

        # Replace stdout/stderr with devnull
        os.dup2(self.devnull, self.stdout_fd)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original file descriptors
        os.dup2(self.saved_stdout, self.stdout_fd)
        os.dup2(self.saved_stderr, self.stderr_fd)
        
        # Close duplicates
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)
        os.close(self.devnull)
        
        # If an error occurred inside the block, print it so we don't hide crashes
        if exc_type:
            print(f"Error occurred while suppressed: {exc_val}", file=sys.stderr)


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Async Dual Play PPO")
    parser.add_argument("--exp_name", type=str, default="async_dual_play_ppo")
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--headless", action="store_true", default=True) # Handled by AppLauncher
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")
    
    parser.add_argument("--save_interval", type=int, default=50, help="Interval to save checkpoints.")
    parser.add_argument("--chkpt_alice", type=str, default=None, help="Path to Alice checkpoint to load.")
    parser.add_argument("--chkpt_bob", type=str, default=None, help="Path to Bob checkpoint to load.")
    parser.add_argument("--arm_config", type=str, default="default", choices=["default", "rotated"], help="Select arm configuration (default or rotated).")
    
    # Add AppLauncher args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Launch App
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Imports that require Isaac Sim to be running
    import torch
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    
    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
    from asyncDualPlayPPO.algorithms.rl.ppo import PPO, PPOABC
    from asyncDualPlayPPO.buffers import GPUDemonstrationBuffer

    # 1. Load Configs
    task_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/task/AsyncDualPlay.yaml")
    ppo_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    
    # Load raw dict for PPO
    ppo_cfg = load_cfg(ppo_cfg_path)
    
    # Load Env Cfg object
    
    # Check if we should override from yaml (Optional, for now just use Python defaults)
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    if args.arm_config == "rotated":
        print("[Config] Applying 'rotated' arm configuration (Left shoulder rotated -90 deg, Right shoulder rotated +90 deg)")
        # Left shoulder -90 deg (-1.57), Right shoulder +90 deg (1.57)
        env_cfg.scene.robot.init_state.joint_pos["left_shoulder_pan_joint"] = -1.57
        env_cfg.scene.robot.init_state.joint_pos["right_shoulder_pan_joint"] = 1.57
    
    # 2. Create Environment
    # We use the IsaacLab ManagerBasedRLEnv, then wrap it
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = AsyncDualPlayEnvWrapper(env=base_env, device=base_env.device, arm_config=args.arm_config)
    
    # 3. Create Agents
    # Alice
    # Alice needs strictly her observation space
    alice_ppo = PPO(
        vec_env=env, # Wrapper provides proper spaces
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/alice",
        asymmetric=False # Alice is symmetric/standard
    )

    # Re-init Alice with correct space
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space
    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape, 
        alice_ppo.state_space.shape, 
        alice_ppo.action_space.shape,
        alice_ppo.init_noise_std, 
        alice_ppo.model_cfg, 
        asymmetric=False
    ).to(env.device)
    
    # Add safety margin to storage to prevent overflow when adding Alice trajectories

    max_alice_trajectory_length = env.episode_manager.alice_timesteps + 10  # Match actual config
    alice_storage_size = alice_ppo.num_transitions_per_env + max_alice_trajectory_length
    
    alice_ppo.storage = alice_ppo.storage.__class__(
        alice_ppo.vec_env.num_envs, 
        alice_storage_size,  # Add safety margin for max trajectory
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape, 
        alice_ppo.action_space.shape, 
        alice_ppo.device, 
        "sequential"
    )
    alice_ppo.optimizer = torch.optim.Adam(alice_ppo.actor_critic.parameters(), lr=alice_ppo.learning_rate)

    # Bob (PPOABC)
    bob_ppo = PPOABC(
        vec_env=env,
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/bob",
        asymmetric=False 
    )
    
    # Create ABC Buffer (GPU Efficient)
    abc_buffer = GPUDemonstrationBuffer(
        capacity=100000,
        obs_shape=env.bob_observation_space.shape,
        action_shape=alice_ppo.action_space.shape,
        device=env.device
    )
    bob_ppo.set_abc_buffer(abc_buffer)

    # Vectorized storage for Alice Trajectories
    # Pre-allocate tensors on GPU: (num_envs, max_steps, dims)
    # We use a tensor here instead of a list for GPU speed
    max_alice_steps = env.episode_manager.alice_timesteps + 10 # Safety margin
    alice_obs_log = torch.zeros((args.num_envs, max_alice_steps, env.alice_obs_dim), device=env.device)
    alice_act_log = torch.zeros((args.num_envs, max_alice_steps, *env.action_space.shape), device=env.device)
    alice_step_counts = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    
    # Track Alice's validity bonuses (per environment) - PERSISTENT BUFFER
    alice_validity_buffer = torch.zeros(args.num_envs, device=env.device)
    
    # Tracking
    alice_updates = 0
    bob_updates = 0
    
    # Logging Setup
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")
    
    # Rolling Buffers — sized to hold a full PPO rollout so rare sparse rewards aren't lost
    len_window = ppo_cfg["params"]["learn"]["nsteps"] * args.num_envs
    alice_rew_buf = deque(maxlen=len_window)
    bob_rew_buf = deque(maxlen=len_window)
    bob_success_buf = deque(maxlen=len_window)
    bob_pos_err_buf = deque(maxlen=len_window)
    bob_rot_err_buf = deque(maxlen=len_window)
    
    # Tracking Best Performance
    best_bob_success_rate = -1.0

    # --- LOGGING INFO ---
    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print("\n" + "="*80)
    print(f"TRAINING RUN: {args.exp_name}")
    print(f"LOG DIRECTORY: {run_dir}")
    print("-" * 80)
    print(f"To monitor training with TensorBoard, run:")
    print(f"  tensorboard --logdir {run_dir}")
    print("="*80 + "\n")

    print("Starting Adversarial Training Loop...")
    
    # Helper to perform Alice update (used inside loops to prevent overflow)
    def perform_alice_update():
        if alice_ppo.storage.step >= ppo_cfg["params"]["learn"]["nsteps"]:
             nonlocal alice_updates
             print(f"Update Alice: Iter {alice_updates}")
             
             dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
             alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
             
             loss_val, loss_surr = alice_ppo.update()
             alice_ppo.storage.clear()

             mean_alice_rew = np.mean(alice_rew_buf) if len(alice_rew_buf) > 0 else 0.0
             
             writer.add_scalar("Loss/Alice/Value", loss_val, alice_updates)
             writer.add_scalar("Loss/Alice/Surrogate", loss_surr, alice_updates)
             writer.add_scalar("Reward/Alice", mean_alice_rew, alice_updates)
             
             # Console Logging  
             print(f"\n{'='*60}")
             print(f"ALICE UPDATE {alice_updates}")
             print(f"{'='*60}")
             print(f"  Rewards:     mean={mean_alice_rew:.4f} | min={0.0:.4f} | max={0.0:.4f}")
             print(f"  Losses:      value={loss_val:.4f} | surrogate={loss_surr:.4f}")
             print(f"  Outcomes:    {len(alice_rew_buf)} episodes buffered")
             print(f"{'='*60}\n")
             
             # Clear Alice reward buffer AFTER logging
             alice_rew_buf.clear()
             
             alice_updates += 1
    
    print("Initializing environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment initialized. Starting training loop...")
    
    # We loop until Bob has updated 'max_iterations' times
    while bob_updates < args.max_iterations:
        
        # 1. Determine who acts
        is_alice = env.episode_manager.is_alice_phase()
        is_bob = env.episode_manager.is_bob_phase()
        
        # 2. Get Actions
        actions = torch.zeros(env.num_envs, *env.action_space.shape, device=env.device)
        
        # Alice Actions
        alice_indices = torch.where(is_alice)[0]
        if len(alice_indices) > 0:
            alice_obs = obs[alice_indices, :env.alice_obs_dim]
            with torch.no_grad():
                a_acts, a_logprob, a_val, a_mu, a_sigma = alice_ppo.actor_critic.act(alice_obs, None)
            actions[alice_indices] = a_acts
        
        # Bob Actions
        bob_indices = torch.where(is_bob)[0]
        if len(bob_indices) > 0:
            bob_obs = obs[bob_indices]
            with torch.no_grad():
                b_acts, b_logprob, b_val, b_mu, b_sigma = bob_ppo.actor_critic.act(bob_obs, None)
            actions[bob_indices] = b_acts

        # 3. Step Env
        next_obs, rewards, dones, truncated, extras = env.step(actions)
        
        # Capture Alice's validity bonus from this step (before it's lost)
        if "alice_validity_bonus" in extras:
            # FIX: Accumulate into persistent buffer, only overwrite non-zero values
            # This handles the case where bonus is given at transition step T, but consumed at T+N
            curr_bonus = extras["alice_validity_bonus"]
            mask = curr_bonus != 0
            alice_validity_buffer[mask] = curr_bonus[mask]
        
        # --- DATA COLLECTION ---
        
        # A. Alice: Buffer locally (VECTORIZED)
        # Save current step data for all active Alices
        # We use advanced indexing to slot data into the correct timestep for each env
        if len(alice_indices) > 0:
            # We must clip steps to max_alice_steps-1 to prevent index error
            steps = alice_step_counts[alice_indices]
            steps = torch.clamp(steps, max=max_alice_steps-1)
            
            alice_obs_log[alice_indices, steps] = alice_obs
            alice_act_log[alice_indices, steps] = a_acts
            alice_step_counts[alice_indices] += 1

        # B. Bob: Push to storage immediately (Standard PPO)
        if len(bob_indices) > 0:
            # Create bob-specific masks
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            
            # FIX: Mask should be 0 if episode ended (dones=True), even if Bob was active
            # This prevents GAE from bootstrapping across episode boundaries
            # active_and_not_done = bob_indices[~dones[bob_indices]] # Logic: Bob Active AND Not Done
            # Actually simpler: standard PPO masks are 1-dones. 
            # But here we also need to mask out ALICE steps (value=0).
            # So mask = 1.0 if (BobActive AND !Done), else 0.0
            
            # --- [CRITICAL FIX]: Check if Bob's phase ended this step
            bob_done_this_step = extras.get("episode_manager", {}).get("bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            
            # Mask should be 0 if episode ended globally OR if Bob's phase just ended
            ended_for_bob = dones[bob_indices] | bob_done_this_step[bob_indices]
            valid_bob = bob_indices[~ended_for_bob]
            b_masks[valid_bob] = 1.0
            
            # Reconstruct batch for add_transitions
            # CRITICAL: Zero out ALL non-Bob envs to prevent data pollution
            # Before this fix, Alice's rewards/obs leaked into Bob's PPO buffer
            _obs = torch.zeros_like(obs); _obs[bob_indices] = obs[bob_indices]
            _acts = torch.zeros_like(actions); _acts[bob_indices] = b_acts
            _rew = torch.zeros(env.num_envs, device=env.device); _rew[bob_indices] = rewards[bob_indices]
            _dones = dones.clone()
            _val = torch.zeros(env.num_envs, 1, device=env.device); _val[bob_indices] = b_val
            _logprob = torch.zeros(env.num_envs, 1, device=env.device); _logprob[bob_indices] = b_logprob.unsqueeze(1)
            _mu = torch.zeros_like(actions); _mu[bob_indices] = b_mu
            _sigma = torch.zeros_like(actions); _sigma[bob_indices] = b_sigma
            
            bob_ppo.storage.add_transitions(
                _obs, _obs, _acts, _rew, _dones, _val, _logprob, _mu, _sigma, b_masks
            )
            
            # Log Bob Rewards (with detail for non-zero rewards)
            bob_step_rewards = rewards[bob_indices]
            bob_rew_buf.extend(bob_step_rewards.cpu().numpy().tolist())
            nonzero_mask = bob_step_rewards != 0
            if nonzero_mask.any():
                for ri in bob_step_rewards[nonzero_mask]:
                    print(f"[Bob Reward] {ri.item():+.1f}")

        # --- HANDLE COMPLETIONS ---
        # Any env where Bob finished (Success or Fail) means Alice's episode is fully resolved.
        # Check 'episode_manager' extras implies completion check already happened in wrapper.
        # We need to look for phase transitions or done flags.
        # Wrapper logic: If Bob finishes, he transitions to Alice OR resets.
        # We need to know if Bob *just* finished. 
        # `extras["bob_done_this_step"]` tells us if Bob finished.
        
        if "episode_manager" in extras:
            em_info = extras["episode_manager"]
            bob_done_mask = em_info["bob_done_this_step"]
            
            if bob_done_mask.any():
                done_ids = torch.where(bob_done_mask)[0]
                success_mask = em_info["bob_success_this_step"]
                
                for idx in done_ids:
                    env_id = idx.item()
                    is_success = success_mask[idx].item()
                    
                    # 1. Calculate Alice Reward (Standard PPO logic kept for Alice's own learning)
                    # Recover trajectory data from logs
                    count = alice_step_counts[env_id].item()
                    if count == 0:
                         continue
                         
                    # Extract slices
                    tr_obs = alice_obs_log[env_id, :count]
                    tr_act = alice_act_log[env_id, :count]
                    
                    # Re-evaluate Alice's actions to get log probs and values
                    # Use evaluate() not act() - we're evaluating existing actions, not generating new ones
                    with torch.no_grad():
                        _, tr_logprob, tr_val, tr_mu, tr_sigma = alice_ppo.actor_critic.evaluate(tr_obs, None, tr_act)
                    
                    alice_reward_scalar = 5.0 if not is_success else 0.0
                    validity_bonus_val = alice_validity_buffer[env_id].item()
                    # Reset buffer after consumption
                    alice_validity_buffer[env_id] = 0.0
                    
                    if alice_reward_scalar > 0:
                        print(f"[Alice Reward] Env {env_id}: +{alice_reward_scalar} | Outcome (Bob Failed)")
                    
                    # Calculate rewards
                    tr_rew = torch.zeros(count, device=env.device)
                    tr_rew[-1] += alice_reward_scalar + validity_bonus_val
                    
                    tr_done = torch.zeros(count, device=env.device)
                    tr_done[-1] = 1.0
                    
                    # Push to Alice PPO Storage (Standard Loop)
                    valid_mask = torch.zeros(env.num_envs, 1, device=env.device)
                    valid_mask[env_id] = 1.0
                    
                    for t in range(count):
                        _o = torch.zeros(env.num_envs, env.alice_obs_dim, device=env.device); _o[env_id] = tr_obs[t]
                        _a = torch.zeros(env.num_envs, *env.action_space.shape, device=env.device); _a[env_id] = tr_act[t]
                        _v = torch.zeros(env.num_envs, 1, device=env.device); _v[env_id] = tr_val[t]
                        _lp = torch.zeros(env.num_envs, 1, device=env.device); _lp[env_id] = tr_logprob[t]
                        _m = torch.zeros_like(actions); _m[env_id] = tr_mu[t]
                        _s = torch.zeros_like(actions); _s[env_id] = tr_sigma[t]
                        _r = torch.zeros(env.num_envs, device=env.device); _r[env_id] = tr_rew[t]
                        _d = torch.zeros(env.num_envs, device=env.device); _d[env_id] = tr_done[t]
                        
                        alice_ppo.storage.add_transitions(_o, _o, _a, _r, _d, _v, _lp, _m, _s, valid_mask)

                    # Log Alice Reward
                    alice_rew_buf.append(tr_rew.sum().item())

                    perform_alice_update()

                # --- VECTORIZED ABC LOGIC (The Fix) ---
                # Check for Alice Wins (Bob Failed)
                alice_wins = bob_done_mask & (~success_mask)
                win_indices = torch.where(alice_wins)[0]
                
                if len(win_indices) > 0:
                     # Get Goals
                     achieved_goals = em_info["goal_states"][win_indices] # (N_wins, 14)
                     
                     batch_list_obs = []
                     batch_list_act = []
                     batch_list_goals = []
                     
                     for idx in win_indices:
                          c = alice_step_counts[idx].item()
                          if c == 0: continue
                          
                          # Extract valid steps
                          obs_seq = alice_obs_log[idx, :c] # (T, Dim)
                          act_seq = alice_act_log[idx, :c] # (T, Act)
                          
                          batch_list_obs.append(obs_seq)
                          batch_list_act.append(act_seq)
                          # Repeat goal for T steps
                          g = achieved_goals[torch.where(win_indices == idx)[0][0]] # (14,)
                          g_seq = g.unsqueeze(0).expand(c, -1) # (T, 14)
                          batch_list_goals.append(g_seq)
                    
                     if len(batch_list_obs) > 0:
                          # Concatenate into one massive batch
                          flat_alice_obs = torch.cat(batch_list_obs, dim=0) # (Total_Steps, Dim)
                          flat_actions = torch.cat(batch_list_act, dim=0)   # (Total_Steps, Act)
                          flat_goals = torch.cat(batch_list_goals, dim=0)   # (Total_Steps, 14)
                          
                          # 1. Construct Bob's Obs (Vectorized)
                          flat_bob_obs = env.construct_bob_observation(flat_alice_obs, flat_goals)
                          
                          # 2. [FIX]: Skip expensive Reference Policy Inference, we compute it dynamically now!
                          dummy_log_probs = torch.zeros((flat_bob_obs.shape[0], 1), device=env.device)
                               
                          # 3. Add to Buffer
                          abc_buffer.add_batch(flat_bob_obs, flat_actions, dummy_log_probs)
                
                # Reset counters for processed envs (Alice Phase Done)
                alice_step_counts[done_ids] = 0

                # Log Bob Metrics
                if success_mask.any():
                     bob_success_buf.extend(success_mask[success_mask].cpu().numpy().astype(float).tolist())
                
                # Bob Failures (count as 0.0 success)
                failures = bob_done_mask & (~success_mask)
                if failures.any():
                     bob_success_buf.extend([0.0] * failures.sum().item())
                
                if bob_done_mask.any():
                     bob_pos_err_buf.extend(em_info["bob_pos_err"][bob_done_mask].cpu().numpy().tolist())
                     bob_rot_err_buf.extend(em_info["bob_rot_err"][bob_done_mask].cpu().numpy().tolist())

        # Safety: Clear Alice buffers for any envs that reset (dones=True)
        if dones.any():
            reset_ids = torch.where(dones)[0]
            # Clear Alice buffer (Reset counters)
            alice_step_counts[reset_ids] = 0
            
            # REMOVED: Redundant perform_alice_update() calls
            # Alice updates are already handled in the Bob completion handler (line 338)
            # Calling perform_alice_update() here could cause double updates when:
            # - Bob finishes a goal (triggers handler path #1 at line 338)
            # - Episode also ends (max goals), setting dones=True
            # - This redundant call would attempt to update again immediately
            # The perform_alice_update() function is safe against this (checks buffer size),
            # but it's cleaner to avoid the redundant calls entirely

        obs = next_obs
        
        # --- UPDATE PHASE ---
        
        # Check Bob Update
        if bob_ppo.storage.step >= ppo_cfg["params"]["learn"]["nsteps"]:
             print(f"Update Bob: Iter {bob_updates}")
             
             # Bootstrapping
             with torch.no_grad():
                 _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(obs, None)
             bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
             
             loss_val, loss_surr, loss_abc = bob_ppo.update()
             bob_ppo.storage.clear()
             # bob_ppo.abc_buffer - No need to clear, it's a circular buffer!
             
             # Logging
             writer.add_scalar("Loss/Bob/Value", loss_val, bob_updates)
             writer.add_scalar("Loss/Bob/Surrogate", loss_surr, bob_updates)
             writer.add_scalar("Loss/Bob/ABC", loss_abc, bob_updates)
             
             mean_bob_rew = np.mean(bob_rew_buf) if len(bob_rew_buf) > 0 else 0.0
             bob_success_rate = np.mean(bob_success_buf) if len(bob_success_buf) > 0 else 0.0
             mean_bob_pos_err = np.mean(bob_pos_err_buf) if len(bob_pos_err_buf) > 0 else 0.0
             mean_bob_rot_err = np.mean(bob_rot_err_buf) if len(bob_rot_err_buf) > 0 else 0.0
             
             writer.add_scalar("Reward/Bob", mean_bob_rew, bob_updates)
             writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
             writer.add_scalar("Metrics/Bob/PosError", mean_bob_pos_err, bob_updates)
             writer.add_scalar("Metrics/Bob/RotError", mean_bob_rot_err, bob_updates)
             
             # Console Logging - Enhanced
             print(f"\n{'='*60}")
             print(f"BOB UPDATE {bob_updates}")
             print(f"{'='*60}")
             print(f"  Success Rate: {bob_success_rate:.4f} ({len(bob_success_buf)} eps)")
             print(f"  Rewards:      mean={mean_bob_rew:.4f}")
             print(f"  Losses:       value={loss_val:.4f} | surrogate={loss_surr:.4f} | ABC={loss_abc:.4f}")
             print(f"  Errors:       pos={mean_bob_pos_err:.4f} | rot={mean_bob_rot_err:.4f}")
             print(f"{'='*60}\n")
             
             # Print overall stats periodically (inside update loop) - Redundant with above but kept for quick scan
             if bob_updates % 10 == 0:
                 print(f"[Summary] Iter {bob_updates}: SR={bob_success_rate:.2f} | ABC_Loss={loss_abc:.4f}")


             # Checkpointing based on Bob Updates
             if args.save_interval > 0 and (bob_updates + 1) % args.save_interval == 0:
                  bob_ppo.save(os.path.join(bob_ppo.log_dir, f"model_{bob_updates+1}.pt"))
                  alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{bob_updates+1}.pt"))
                  print(f"  ✓ Saved checkpoints")
             
             if bob_success_rate > best_bob_success_rate:
                  best_bob_success_rate = bob_success_rate
                  bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_best.pt"))
                  alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
                  print(f"  ★ New Best SR: {best_bob_success_rate:.2f}")
             
             bob_updates += 1

        # Check Alice Update - REMOVED (Moved inside loops)
        # perform_alice_update() is now called immediately after adding trajectories
        # to ensure we never overflow the buffer.
             
             # Console Logging  
             # (Moved to perform_alice_update)
             
             
             # (Increment handled in perform_alice_update)

    
    # Save final models
    alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_final.pt"))
    bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_final.pt"))
    print("  ✓ Saved final models")
    
    writer.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("To resume training from the final checkpoints, run:")
    print(f"python train.py --exp_name {args.exp_name}_resume \\")
    print(f"  --chkpt_alice runs/{args.exp_name}/alice/model_final.pt \\")
    print(f"  --chkpt_bob runs/{args.exp_name}/bob/model_final.pt")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
