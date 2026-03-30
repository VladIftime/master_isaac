"""
Test ABC (Alice Behavioral Cloning) in a real IsaacLab dual-arm robot simulation.

Alice is a FIXED deterministic policy (always outputs constant small deltas).
Bob starts random and must learn to imitate Alice purely via the NLL behavioral
cloning loss from ppo_abc.py — NO environment reward is used.

This verifies the full ABC pipeline end-to-end:
  1. Environment creation and observation shapes
  2. Alice trajectory recording in the real physics sim
  3. construct_bob_observation building correct 56-dim inputs
  4. GPUDemonstrationBuffer storing and sampling demos
  5. NLL loss computation and gradient flow in PPOABC.update
  6. Bob's action distribution converging toward Alice's fixed policy

Usage:
    source /home/vlad/env_isaaclab/bin/activate && \\
    python tests/test_abc_sim.py --num_envs 4 --num_iterations 100
"""

import isaaclab.app
from isaaclab.app import AppLauncher

import argparse
import os
import sys
import copy
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    parser = argparse.ArgumentParser(description="Test ABC in robot simulation")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np

    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import (
        RolloutStorage,
        GPUDemonstrationBuffer,
    )

    import gymnasium as gym

    torch.manual_seed(args.seed)

    # ─── Config ────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "..", "cfg", "ppo", "ppo_continuous.yaml")
    with open(cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    # ─── Environment ───────────────────────────────────────────
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = AsyncDualPlayEnvWrapper(base_env)

    print(f"\n{'='*60}")
    print(f"  ABC Simulation Test")
    print(f"  Envs: {args.num_envs}")
    print(f"  Alice obs dim: {env.alice_obs_dim}")
    print(f"  Bob obs shape: {env.bob_observation_space.shape}")
    print(f"  Action shape:  {env.action_space.shape}")
    print(f"{'='*60}\n")

    # ─── Fixed Alice action ────────────────────────────────────
    # Alice always outputs the same small action.
    # This is trivially simple: Bob should converge quickly if ABC works.
    action_dim = env.action_space.shape[0]
    alice_fixed_action = torch.zeros(args.num_envs, action_dim, device=env.device)
    alice_fixed_action[:, 0] = 0.05    # small +X delta
    alice_fixed_action[:, 1] = -0.03   # small -Y delta
    alice_fixed_action[:, 2] = 0.01    # small +Z delta
    # remaining dims (rotation, gripper) stay 0

    print(f"  Alice fixed action: {alice_fixed_action[0].tolist()}")

    # ─── Bob PPO Agent ─────────────────────────────────────────
    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_goal_encoder"] = True
    bob_cfg["learn"]["abc_coef"] = 1.0     # Pure BC — no PPO reward signal
    bob_cfg["learn"]["ent_coef"] = 0.0     # No entropy regularization (pure imitation)
    bob_cfg["learn"]["nsteps"] = 32        # Short rollout
    bob_cfg["learn"]["noptepochs"] = 5
    bob_cfg["learn"]["nminibatches"] = 2

    bob_ppo = PPOABC(
        vec_env=env,
        cfg_train=bob_cfg,
        device=env.device,
        sampler="sequential",
        log_dir="runs/test_abc_sim/bob",
        asymmetric=False,
    )
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space

    # Rebuild actor_critic for correct Bob obs shape (56-dim, not concat)
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

    # Rebuild storage for correct shapes
    nsteps = bob_cfg["learn"]["nsteps"]
    bob_ppo.storage = RolloutStorage(
        bob_ppo.vec_env.num_envs,
        nsteps,
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.device,
        "sequential",
    )

    # ABC demo buffer
    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
        capacity=50000,
        obs_shape=bob_ppo.observation_space.shape,
        states_shape=bob_ppo.observation_space.shape,
        actions_shape=bob_ppo.action_space.shape,
        device=env.device,
    )
    bob_ppo.set_abc_buffer(bob_ppo.abc_buffer)

    # ─── Phase 1: Collect Alice Demonstrations ─────────────────
    print("\n  Phase 1: Collecting Alice demonstrations in simulation...")
    NUM_DEMO_STEPS = 30
    NUM_DEMO_EPISODES = 10

    for ep in range(NUM_DEMO_EPISODES):
        obs_concat, info = env.reset()
        alice_obs = obs_concat[:, :env.alice_obs_dim]

        ep_obs = []
        ep_acts = []

        for step in range(NUM_DEMO_STEPS):
            ep_obs.append(alice_obs.clone())
            ep_acts.append(alice_fixed_action.clone())

            # Step with Alice's constant action
            obs_concat, rewards, terminated, truncated, extras = env.step(alice_fixed_action)
            alice_obs = obs_concat[:, :env.alice_obs_dim]

        # After Alice's phase, use current object state as the "goal"
        # (what Bob would see as the target to reproduce)
        goal_states = env.episode_manager.goal_states
        if goal_states is None:
            # If no goal was captured by the episode manager, use object state directly
            obs_dict = env.env.observation_manager.compute()
            bob_obs_raw = obs_dict["bob_policy"]
            # Fallback: extract object poses from Alice obs
            goal_states = alice_obs[:, env.robot_state_dim:env.robot_state_dim + 14]
            print(f"    [ep {ep}] Using fallback goal states (shape={goal_states.shape})")

        # Construct Bob-compatible obs and push to ABC buffer
        for eid in range(args.num_envs):
            traj_o = torch.stack([ep_obs[s][eid] for s in range(NUM_DEMO_STEPS)])
            traj_a = torch.stack([ep_acts[s][eid] for s in range(NUM_DEMO_STEPS)])
            g = goal_states[eid].unsqueeze(0).expand(NUM_DEMO_STEPS, -1)

            bc_obs = env.construct_bob_observation(traj_o, g)

            bob_ppo.abc_buffer.add_trajectory(
                bc_obs,                                                   # observations
                bc_obs,                                                   # states
                traj_a,                                                   # actions
                torch.zeros(NUM_DEMO_STEPS, device=env.device),           # rewards
                torch.zeros(NUM_DEMO_STEPS, device=env.device).byte(),    # dones
                torch.zeros(NUM_DEMO_STEPS, device=env.device),           # values
                torch.zeros(NUM_DEMO_STEPS, 1, device=env.device),        # log_probs
                torch.zeros_like(traj_a),                                 # mu
                torch.zeros_like(traj_a),                                 # sigma
                torch.zeros(NUM_DEMO_STEPS, 1, device=env.device),        # returns
                torch.zeros(NUM_DEMO_STEPS, 1, device=env.device),        # advantages
            )

        if (ep + 1) % 5 == 0:
            print(f"    Collected {ep+1}/{NUM_DEMO_EPISODES} episodes, "
                  f"buffer size: {bob_ppo.abc_buffer.size}")

    print(f"  ABC buffer: {bob_ppo.abc_buffer.size} transitions\n")

    # ─── Phase 2: Train Bob with Pure ABC ──────────────────────
    print(f"  Phase 2: Training Bob with ABC for {args.num_iterations} iters...")

    bc_losses = []
    action_mses = []

    for it in range(1, args.num_iterations + 1):
        # Fill Bob's rollout storage with his exploration (zero reward)
        bob_ppo.storage.clear()
        obs_concat, info = env.reset()
        bob_obs = obs_concat[:, env.alice_obs_dim:]

        for step in range(nsteps):
            with torch.no_grad():
                acts, lp, val, mu, sigma = bob_ppo.actor_critic.act(bob_obs, bob_obs)

            obs_concat, rewards, terminated, truncated, extras = env.step(acts)
            bob_obs_new = obs_concat[:, env.alice_obs_dim:]

            # Zero rewards — force Bob to rely purely on ABC
            zero_rew = torch.zeros(args.num_envs, 1, device=env.device)
            done_mask = (terminated | truncated).float().unsqueeze(-1)

            bob_ppo.storage.add_transitions(
                bob_obs, bob_obs, acts,
                zero_rew, done_mask,
                val,
                lp.unsqueeze(-1) if lp.dim() == 1 else lp,
                mu, sigma,
            )
            bob_obs = bob_obs_new

        # Compute returns (all ~zero)
        with torch.no_grad():
            _, _, last_val, _, _ = bob_ppo.actor_critic.act(bob_obs, bob_obs)
        bob_ppo.storage.compute_returns(last_val, gamma=0.99, lam=0.95)

        # Update — PPO + ABC
        val_loss, surr_loss, bc_loss, _ = bob_ppo.update()
        bc_losses.append(bc_loss)

        # Measure: does Bob output Alice's actions?
        test_sample = bob_ppo.abc_buffer.sample(min(256, bob_ppo.abc_buffer.size))
        if test_sample is not None:
            with torch.no_grad():
                bob_acts = bob_ppo.actor_critic.act_inference(test_sample[0])
            mse = (bob_acts - test_sample[2]).pow(2).mean().item()
            action_mses.append(mse)
        else:
            action_mses.append(float('nan'))
            mse = float('nan')

        if it % 10 == 0 or it == 1:
            noise_std = bob_ppo.actor_critic.log_std.exp().mean().item()
            print(
                f"    Iter {it:4d} | BC Loss: {bc_loss:+.4f} | "
                f"Action MSE: {mse:.6f} | noise_std: {noise_std:.4f}"
            )

    # ─── Results ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  BC Loss:   {bc_losses[0]:.4f} → {bc_losses[-1]:.4f}")
    print(f"  Action MSE: {action_mses[0]:.6f} → {action_mses[-1]:.6f}")

    converged = len(action_mses) > 1 and action_mses[-1] < action_mses[0] * 0.5
    bc_decreased = len(bc_losses) > 1 and bc_losses[-1] < bc_losses[0]

    if converged and bc_decreased:
        print(f"\n  ✓ ABC WORKS — Bob converged toward Alice's fixed actions")
    elif bc_decreased:
        print(f"\n  ~ PARTIAL — BC loss decreased but action MSE didn't halve")
        print(f"    (may need more iterations or lower init_noise_std)")
    else:
        print(f"\n  ✗ FAILED — BC loss did not decrease")

    print()
    simulation_app.close()


if __name__ == "__main__":
    main()
