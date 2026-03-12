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
    from asyncDualPlayPPO.algorithms.rl.ppo import PPO
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

    bob_ppo = PPO(
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
    bob_ppo.demo_buffer = GPUDemonstrationBuffer(
        capacity=100000,
        obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape,
        actions_shape=env.action_space.shape,
        device=env.device,
    )

    # --- Pre-allocated trajectory buffers for Alice ---
    alice_obs_log    = torch.zeros((args.num_envs, max_alice_steps, env.alice_obs_dim), device=env.device)
    alice_act_log    = torch.zeros((args.num_envs, max_alice_steps, *env.action_space.shape), device=env.device)
    alice_step_counts = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    alice_validity_buffer = torch.zeros(args.num_envs, device=env.device)

    alice_updates = 0
    bob_updates   = 0
    max_alice_bob_ratio = args.max_alice_bob_ratio

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")

    rollout_length = ppo_cfg["params"]["learn"]["nsteps"] * args.num_envs
    alice_rew_buf  = deque(maxlen=rollout_length)
    bob_rew_buf    = deque(maxlen=rollout_length)
    bob_success_buf = deque(maxlen=rollout_length)
    bob_pos_err_buf = deque(maxlen=rollout_length)
    bob_rot_err_buf = deque(maxlen=rollout_length)

    best_bob_success_rate = -1.0

    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print(f"\n{'='*80}\nTRAINING RUN: {args.exp_name}\nLOG DIRECTORY: {run_dir}")
    print(f"  tensorboard --logdir {run_dir}\n{'='*80}\n")

    def perform_alice_update():
        """Run a PPO update for Alice when her rollout buffer is full."""
        if alice_ppo.storage.step < ppo_cfg["params"]["learn"]["nsteps"]:
            return

        nonlocal alice_updates

        # Gate: don't let Alice outpace Bob too much
        if alice_updates >= (bob_updates + 1) * max_alice_bob_ratio:
            alice_ppo.storage.clear()
            alice_rew_buf.clear()
            return  # Alice waits for Bob to catch up

        dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
        alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
        loss_val, loss_surr = alice_ppo.update()
        alice_ppo.storage.clear()

        mean_alice_rew = np.mean(alice_rew_buf) if alice_rew_buf else 0.0
        writer.add_scalar("Loss/Alice/Value",     loss_val,       alice_updates)
        writer.add_scalar("Loss/Alice/Surrogate", loss_surr,      alice_updates)
        writer.add_scalar("Reward/Alice",         mean_alice_rew, alice_updates)

        print(f"\n{'='*60}\nALICE UPDATE {alice_updates}\n{'='*60}")
        print(f"  Rewards:     mean={mean_alice_rew:.4f} | min={0.0:.4f} | max={0.0:.4f}")
        print(f"  Losses:      value={loss_val:.4f} | surrogate={loss_surr:.4f}")
        print(f"  Outcomes:    {len(alice_rew_buf)} episodes buffered\n{'='*60}\n")

        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_obs):
        """Run a PPO update for Bob when his rollout buffer is full."""
        if bob_ppo.storage.step < ppo_cfg["params"]["learn"]["nsteps"]:
            return current_obs
            
        nonlocal bob_updates, best_bob_success_rate
        with torch.no_grad():
            _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(current_obs, None)
        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        loss_val, loss_surr = bob_ppo.update()
        bob_ppo.storage.clear()

        mean_bob_rew     = np.mean(bob_rew_buf)     if bob_rew_buf     else 0.0
        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        mean_pos_err     = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot_err     = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0

        writer.add_scalar("Loss/Bob/Value",       loss_val,         bob_updates)
        writer.add_scalar("Loss/Bob/Surrogate",   loss_surr,        bob_updates)
        writer.add_scalar("Reward/Bob",           mean_bob_rew,     bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
        writer.add_scalar("Metrics/Bob/PosError",    mean_pos_err,     bob_updates)
        writer.add_scalar("Metrics/Bob/RotError",    mean_rot_err,     bob_updates)

        print(f"\n{'='*60}\nBOB UPDATE {bob_updates}\n{'='*60}")
        print(f"  Success Rate: {bob_success_rate:.4f} ({len(bob_success_buf)} eps)")
        print(f"  Rewards:      mean={mean_bob_rew:.4f}")
        print(f"  Losses:       value={loss_val:.4f} | surrogate={loss_surr:.4f}")
        print(f"  Errors:       pos={mean_pos_err:.4f} | rot={mean_rot_err:.4f}")
        print(f"  Alice/Bob:    {alice_updates}/{bob_updates} updates (ratio cap={max_alice_bob_ratio})\n{'='*60}\n")

        if bob_updates % 10 == 0:
            print(f"[Summary] Iter {bob_updates}: SR={bob_success_rate:.2f}")

        if args.save_interval > 0 and (bob_updates + 1) % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir,   f"model_{bob_updates+1}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{bob_updates+1}.pt"))
            print("  ✓ Saved checkpoints")

        if bob_success_rate > best_bob_success_rate:
            best_bob_success_rate = bob_success_rate
            bob_ppo.save(os.path.join(bob_ppo.log_dir,   "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
            print(f"  ★ New Best SR: {best_bob_success_rate:.2f}")

        bob_rew_buf.clear()
        bob_success_buf.clear()
        bob_pos_err_buf.clear()
        bob_rot_err_buf.clear()
        
        bob_updates += 1
        return current_obs

    print("Initializing environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment initialized. Starting training loop...")

    while bob_updates < args.max_iterations:

        # --- PHASE 2: ALPHA ANNEALING ---
        # Decay alpha linearly from 1.0 to 0.0 over the course of training
        alpha = max(0.0, 1.0 - (bob_updates / args.max_iterations))
        env.bob_dense_reward_alpha = alpha
        # --------------------------------

        is_alice = env.episode_manager.is_alice_phase()
        is_bob   = env.episode_manager.is_bob_phase()

        actions = torch.zeros(env.num_envs, *env.action_space.shape, device=env.device)

        alice_indices = torch.where(is_alice)[0]
        if len(alice_indices) > 0:
            alice_obs = obs[alice_indices, :env.alice_obs_dim]
            with torch.no_grad():
                a_acts, a_logprob, a_val, a_mu, a_sigma = alice_ppo.actor_critic.act(alice_obs, None)
            actions[alice_indices] = a_acts

        bob_indices = torch.where(is_bob)[0]
        if len(bob_indices) > 0:
            bob_obs = obs[bob_indices]
            with torch.no_grad():
                b_acts, b_logprob, b_val, b_mu, b_sigma = bob_ppo.actor_critic.act(bob_obs, None)
            actions[bob_indices] = b_acts

        next_obs, rewards, dones, truncated, extras = env.step(actions)

        if "alice_validity_bonus" in extras:
            curr_bonus = extras["alice_validity_bonus"]
            mask = curr_bonus != 0
            alice_validity_buffer[mask] = curr_bonus[mask]
            
            # --- HINDSIGHT GOAL INJECTION ---
            # Inject successful Alice trajectories into Bob's PPO buffer as demonstrations
            alice_success_mask = curr_bonus == 1.0 # ALICE_VALID_GOAL_BONUS
            if alice_success_mask.any():
                success_ids = torch.where(alice_success_mask)[0]
                goal_states = env.episode_manager.goal_states
                
                hgi_count = 0
                for idx in success_ids:
                    env_id = idx.item()
                    s_count = min(alice_step_counts[env_id].item(), max_alice_steps)
                    if s_count == 0: continue
                    
                    a_obs = alice_obs_log[env_id, :s_count]
                    a_acts = alice_act_log[env_id, :s_count]
                    
                    _o = torch.zeros((s_count, env.bob_obs_dim), device=env.device)
                    _r = torch.zeros((s_count,), device=env.device)
                    _d = torch.zeros((s_count,), device=env.device)
                    
                    # Construct Bob's obs (robot arm/objects states + HGI goals)
                    goal_state = goal_states[env_id].unsqueeze(0).expand(s_count, -1)
                    b_obs = env.construct_bob_observation(a_obs, goal_state)
                    _o[:] = b_obs
                    
                    # Reward Injection: Give Bob's completion reward (+5) on the very last step.
                    _r[-1] = 5.0
                    _d[-1] = 1.0
                    
                    # Evaluate under Bob's current policy
                    with torch.no_grad():
                        _lp, _, _v, _m, _s = bob_ppo.actor_critic.evaluate(_o, None, a_acts)
                        
                    # Compute offline GAE returns and advantages for the trajectory
                    _ret = torch.zeros((s_count,), device=env.device)
                    _adv = torch.zeros((s_count,), device=env.device)
                    
                    adv = 0.0
                    gamma = bob_ppo.gamma
                    lam = bob_ppo.lam
                    
                    for step in reversed(range(s_count)):
                        next_val = 0.0 if step == s_count - 1 else _v[step + 1].item()
                        next_not_done = 0.0 if step == s_count - 1 else 1.0
                        delta = _r[step] + gamma * next_val * next_not_done - _v[step].item()
                        adv = delta + gamma * lam * next_not_done * adv
                        _adv[step] = adv
                        _ret[step] = adv + _v[step].item()
                        
                    # Add to offline demo buffer (does not crash or prematurely sync RolloutStorage)
                    none_states = torch.zeros((s_count, *env.bob_observation_space.shape), device=env.device)
                    bob_ppo.demo_buffer.add_trajectory(_o, none_states, a_acts, _r, _d, _v, _lp, _m, _s, _ret, _adv)
                    hgi_count += s_count
                    
                if hgi_count > 0:
                    print(f"  [HGI] Added {hgi_count} steps into Bob's Demonstration Buffer!")

        if len(alice_indices) > 0:
            steps = torch.clamp(alice_step_counts[alice_indices], max=max_alice_steps - 1)
            alice_obs_log[alice_indices, steps]  = alice_obs
            alice_act_log[alice_indices, steps]  = a_acts
            alice_step_counts[alice_indices] += 1

        if len(bob_indices) > 0:
            bob_done_this_step = extras.get("episode_manager", {}).get(
                "bob_done_this_step",
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            )
            ended_for_bob = dones[bob_indices] | bob_done_this_step[bob_indices]
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob]] = 1.0

            _obs   = torch.zeros_like(obs);     _obs[bob_indices]    = obs[bob_indices]
            _acts  = torch.zeros_like(actions); _acts[bob_indices]   = b_acts
            _rew   = torch.zeros(env.num_envs, device=env.device)
            _rew[bob_indices] = rewards[bob_indices]
            _val   = torch.zeros(env.num_envs, 1, device=env.device); _val[bob_indices]    = b_val
            _lp    = torch.zeros(env.num_envs, 1, device=env.device); _lp[bob_indices]     = b_logprob.unsqueeze(1)
            _mu    = torch.zeros_like(actions); _mu[bob_indices]     = b_mu
            _sigma = torch.zeros_like(actions); _sigma[bob_indices]  = b_sigma

            bob_ppo.storage.add_transitions(_obs, _obs, _acts, _rew, dones.clone(), _val, _lp, _mu, _sigma, b_masks)

            # --- PHASE 4: Safe-State Filtered HER (Retroactive Relabeling) ---
            if "episode_manager" in extras:
                em_info = extras["episode_manager"]
                bob_done_mask = em_info["bob_done_this_step"]
                
                if bob_done_mask.any():
                    bob_success_mask = em_info["bob_success_this_step"]
                    max_forces = em_info.get("max_contact_force", torch.zeros(env.num_envs, device=env.device))
                    
                    # Safe failure: Bob failed (timed out or reached max goals) but didn't crash
                    THRESHOLD = 50.0  # Safe physical threshold
                    is_safe_failure = bob_done_mask & (~bob_success_mask) & (max_forces < THRESHOLD)
                    
                    safe_failure_ids = torch.where(is_safe_failure)[0]
                    if len(safe_failure_ids) > 0 and "bob_achieved_states" in em_info:
                        achieved_states = em_info["bob_achieved_states"]
                        relabel_count = 0
                        # Relabel all transitions currently sitting in Bob's storage buffer for this environment
                        for eid in safe_failure_ids:
                            eid_int = eid.item()
                            achieved = achieved_states[eid_int].clone()
                            
                            # Get the active transitions for this environment in the current buffer
                            buffer_masks = bob_ppo.storage.masks[:, eid_int, 0]
                            valid_steps = torch.where(buffer_masks > 0)[0]
                            
                            if len(valid_steps) > 0:
                                for t in valid_steps:
                                    # Copy Alice's observations (robot arm/objects states)
                                    a_obs = bob_ppo.storage.observations[t, eid_int, :env.unwrapped.alice_obs_dim].unsqueeze(0)
                                    # Overwrite the goal with Bob's achieved state and implicitly recompute distance features
                                    b_obs = env.unwrapped.construct_bob_observation(a_obs, achieved.unsqueeze(0))
                                    bob_ppo.storage.observations[t, eid_int] = b_obs[0]
                                
                                # Give +1.0 success reward to the very last valid step in Bob's buffer
                                last_step = valid_steps[-1]
                                bob_ppo.storage.rewards[last_step, eid_int] = 1.0
                                relabel_count += 1
                        
                        if relabel_count > 0:
                            print(f"  [HER] Relabeled {relabel_count} safe-failure Bob trajectories!")
            # ----------------------------------------------------------------

            bob_step_rewards = rewards[bob_indices]
            bob_rew_buf.extend(bob_step_rewards.cpu().numpy().tolist())
            for ri in bob_step_rewards[bob_step_rewards != 0]:
                print(f"[Bob Reward] {ri.item():+.1f}")

        if "episode_manager" in extras:
            em_info = extras["episode_manager"]
            
            # Identify which environments need Alice's PPO buffer updated
            bob_done_mask = em_info["bob_done_this_step"]
            alice_failed_mask = extras.get("alice_failed_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            
            alice_eval_mask = alice_failed_mask | bob_done_mask
            if alice_eval_mask.any():
                eval_ids = torch.where(alice_eval_mask)[0]
                bob_success_mask = em_info["bob_success_this_step"]
                
                # First pass: precalculate lengths and outcome rewards
                max_count = 0
                env_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                env_rewards = torch.zeros(env.num_envs, device=env.device)
                
                for idx in eval_ids:
                    env_id = idx.item()
                    count = min(alice_step_counts[env_id].item(), max_alice_steps)
                    if count == 0:
                        continue
                        
                    env_counts[env_id] = count
                    max_count = max(max_count, count)
                    
                    # Determine exact outcome reward based on how the phase ended
                    if alice_failed_mask[env_id]:
                        alice_reward = ALICE_INVALID_GOAL_PENALTY
                        reason = "Invalid Goal"
                    else:
                        is_success = bob_success_mask[idx].item()
                        alice_reward = ALICE_BOB_SUCCESS_REWARD if is_success else ALICE_BOB_FAIL_REWARD
                        reason = "Bob Succeeded" if is_success else "Bob Failed"
                        
                    validity_bonus = alice_validity_buffer[env_id].item()
                    alice_validity_buffer[env_id] = 0.0
                    
                    env_rewards[env_id] = alice_reward + validity_bonus
                    alice_rew_buf.append(env_rewards[env_id].item())
                
                # Second pass: densely insert transitions in parallel across environments up to max_count
                if max_count > 0:
                    for t in range(max_count):
                        # Mask for environments that are still active at time t
                        active_mask = (env_counts > t).float().unsqueeze(1)  # [num_envs, 1]
                        
                        _o = torch.zeros((env.num_envs, env.alice_obs_dim), device=env.device)
                        _a = torch.zeros((env.num_envs, *env.action_space.shape), device=env.device)
                        _r = torch.zeros((env.num_envs,), device=env.device)
                        _d = torch.zeros((env.num_envs,), device=env.device)
                        
                        active_ids = torch.where(env_counts > t)[0]
                        if len(active_ids) > 0:
                            _o[active_ids] = alice_obs_log[active_ids, t]
                            _a[active_ids] = alice_act_log[active_ids, t]
                            
                            # Reward and done flag only given on the exact target step
                            is_last_step = (env_counts == (t + 1))
                            last_step_ids = torch.where(is_last_step)[0]
                            if len(last_step_ids) > 0:
                                _r[last_step_ids] = env_rewards[last_step_ids]
                                _d[last_step_ids] = 1.0
                                
                            with torch.no_grad():
                                _lp, _, _v, _m, _s = alice_ppo.actor_critic.evaluate(_o, None, _a)
                            
                            # Zero out unused values/logprobs for cleanliness
                            _v = _v.view(-1, 1) * active_mask
                            _lp = _lp.view(-1, 1) * active_mask
                            
                            alice_ppo.storage.add_transitions(_o, _o, _a, _r, _d, _v, _lp, _m, _s, active_mask)
                            
                    perform_alice_update()
                    
                alice_step_counts[eval_ids] = 0

            # Bob metric logging
            if bob_done_mask.any():
                bob_success_mask = em_info["bob_success_this_step"]
                bob_success_buf.extend(bob_success_mask[bob_success_mask].cpu().numpy().astype(float).tolist())
                bob_success_buf.extend([0.0] * (bob_done_mask & ~bob_success_mask).sum().item())
                bob_pos_err_buf.extend(em_info["bob_pos_err"][bob_done_mask].cpu().numpy().tolist())
                bob_rot_err_buf.extend(em_info["bob_rot_err"][bob_done_mask].cpu().numpy().tolist())

        if dones.any():
            alice_step_counts[torch.where(dones)[0]] = 0

        obs = next_obs
        obs = perform_bob_update(obs)

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
