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
    # Import reward constants so train.py stays in sync with rewards.py.
    # Do not hardcode reward values here — change them in rewards.py instead.
    from asyncDualPlayPPO.tasks.utils.rewards import (
        ALICE_BOB_FAIL_REWARD,
        ALICE_BOB_SUCCESS_REWARD,
        ALICE_VALID_GOAL_BONUS,
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

    print("Initializing environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment initialized. Starting training loop...")

    while bob_updates < args.max_iterations:

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

            bob_step_rewards = rewards[bob_indices]
            bob_rew_buf.extend(bob_step_rewards.cpu().numpy().tolist())
            for ri in bob_step_rewards[bob_step_rewards != 0]:
                print(f"[Bob Reward] {ri.item():+.1f}")

        if "episode_manager" in extras:
            em_info = extras["episode_manager"]
            bob_done_mask = em_info["bob_done_this_step"]

            if bob_done_mask.any():
                done_ids     = torch.where(bob_done_mask)[0]
                success_mask = em_info["bob_success_this_step"]

                for idx in done_ids:
                    env_id     = idx.item()
                    is_success = success_mask[idx].item()
                    count      = alice_step_counts[env_id].item()
                    if count == 0:
                        continue

                    tr_obs = alice_obs_log[env_id, :count]
                    tr_act = alice_act_log[env_id, :count]
                    with torch.no_grad():
                        _, tr_logprob, tr_val, tr_mu, tr_sigma = alice_ppo.actor_critic.evaluate(
                            tr_obs, None, tr_act
                        )

                    # Outcome reward: defined in rewards.py — do not hardcode here.
                    alice_reward = ALICE_BOB_SUCCESS_REWARD if is_success else ALICE_BOB_FAIL_REWARD
                    validity_bonus = alice_validity_buffer[env_id].item()
                    alice_validity_buffer[env_id] = 0.0

                    if alice_reward > 0:
                        print(f"[Alice Reward] Env {env_id}: +{alice_reward} | Bob Failed")
                    if validity_bonus != 0:
                        print(f"[Alice Reward] Env {env_id}: +{validity_bonus} | Valid Goal")

                    tr_rew       = torch.zeros(count, device=env.device)
                    tr_rew[-1]  += alice_reward + validity_bonus
                    tr_done      = torch.zeros(count, device=env.device)
                    tr_done[-1]  = 1.0

                    valid_mask = torch.zeros(env.num_envs, 1, device=env.device)
                    valid_mask[env_id] = 1.0

                    for t in range(count):
                        _o  = torch.zeros(env.num_envs, env.alice_obs_dim, device=env.device); _o[env_id]  = tr_obs[t]
                        _a  = torch.zeros(env.num_envs, *env.action_space.shape, device=env.device); _a[env_id]  = tr_act[t]
                        _v  = torch.zeros(env.num_envs, 1, device=env.device); _v[env_id]  = tr_val[t]
                        _lp = torch.zeros(env.num_envs, 1, device=env.device); _lp[env_id] = tr_logprob[t]
                        _m  = torch.zeros_like(actions); _m[env_id]  = tr_mu[t]
                        _s  = torch.zeros_like(actions); _s[env_id]  = tr_sigma[t]
                        _r  = torch.zeros(env.num_envs, device=env.device); _r[env_id]  = tr_rew[t]
                        _d  = torch.zeros(env.num_envs, device=env.device); _d[env_id]  = tr_done[t]
                        alice_ppo.storage.add_transitions(_o, _o, _a, _r, _d, _v, _lp, _m, _s, valid_mask)

                    alice_rew_buf.append(tr_rew.sum().item())
                    perform_alice_update()

                alice_step_counts[done_ids] = 0

                bob_success_buf.extend(success_mask[success_mask].cpu().numpy().astype(float).tolist())
                bob_success_buf.extend([0.0] * (bob_done_mask & ~success_mask).sum().item())
                bob_pos_err_buf.extend(em_info["bob_pos_err"][bob_done_mask].cpu().numpy().tolist())
                bob_rot_err_buf.extend(em_info["bob_rot_err"][bob_done_mask].cpu().numpy().tolist())

        if dones.any():
            alice_step_counts[torch.where(dones)[0]] = 0

        obs = next_obs

        if bob_ppo.storage.step >= ppo_cfg["params"]["learn"]["nsteps"]:
            with torch.no_grad():
                _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(obs, None)
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

            bob_updates += 1

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
