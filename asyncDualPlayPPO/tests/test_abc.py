"""
3-Step Diagnostic Test Plan for PPO + ABC Pipeline Verification
================================================================

Each step isolates a different learning mechanism to pinpoint failures.

Step 1: Pure PPO Overfit (Isolate Sparse Reward)
  - DummyAliceWrapper active (goal locked at [0.15, 0.5, 0.05])
  - abc_coef=0.0, ent_coef=0.0
  - Proves: PPO math, RMPflow action mapping, environment resets
  - Pass: SR -> 1.0 within 100-200 updates

Step 2: Pure BC Overfit (Isolate ABC Loss)
  - "God Mode" trajectory injected (constant bin=8 for X, bin=5 for rest)
  - abc_coef=1.0, value_loss_coef=0.0
  - Proves: BC loss formula, gradient flow, action convergence
  - Pass: BC loss -> large negative, Bob's greedy actions -> hardcoded bins

Step 3: Easy Mode Integration (Full ASP)
  - Real Alice (no DummyAliceWrapper), abc_coef=0.5, ent_coef=0.01
  - pos_threshold=0.20 (massive 20cm success radius)
  - Proves: Full asymmetric self-play loop
  - Pass: Both Alice and Bob SR climb synchronously

Usage:
    python tests/test_abc.py --step 1 --num_envs 64 --num_iterations 200
    python tests/test_abc.py --step 2 --num_envs 4  --num_iterations 100
    python tests/test_abc.py --step 3 --num_envs 64 --num_iterations 200
"""

import isaaclab.app
from isaaclab.app import AppLauncher

import argparse
import os
import sys
import copy
import yaml
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def install_noise_filter():
    """Filter C-level stderr noise (URDF material warnings, Lula, carb) via a pipe thread."""
    _DROP = (
        b"[Lula] Joint",
        b"Warning: link",
        b"urdf_parser",
        b"flat_black",
        b"IMemoryBudgetManager",
        b"robotiq_coupler",
    )
    orig_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _worker():
        orig = os.fdopen(orig_fd, "wb", buffering=0)
        buf = b""
        with os.fdopen(read_fd, "rb", buffering=0) as src:
            while True:
                chunk = src.read(256)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_nl = line + b"\n"
                    if not any(p in line_nl for p in _DROP):
                        orig.write(line_nl)
                        orig.flush()
        if buf and not any(p in buf for p in _DROP):
            orig.write(buf)
            orig.flush()

    threading.Thread(target=_worker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="3-Step PPO+ABC Diagnostic Test")
    parser.add_argument(
        "--step", choices=["1", "2", "3", "4", "5", "6"], required=True,
        help="Which diagnostic step to run (1=Pure PPO, 2=Pure BC, 3=Integration, "
             "4=GoalDist check, 5=Reward pipeline, 6=Movement detection)",
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_threshold", type=float, default=None,
                        help="Override Bob pos_threshold in metres (default: env default 0.04 m)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    install_noise_filter()  # suppress URDF material warnings from C++ stderr

    import torch
    import torch.nn as nn
    import numpy as np
    from collections import deque

    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
    from asyncDualPlayPPO.tasks.utils.dummy_alice_wrapper import (
        DummyAliceWrapper,
        DummyBobWrapper,
        DummyGoalDistanceWrapper,
        DummyMovementWrapper,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import (
        RolloutStorage,
        GPUDemonstrationBuffer,
    )

    torch.manual_seed(args.seed)

    # ── Load base config ────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "..", "cfg", "ppo", "ppo_continuous.yaml")
    with open(cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    pol_cfg = ppo_cfg["params"]["policy"]
    use_mc = pol_cfg.get("use_multicategorical", False)
    num_cat_dims = pol_cfg.get("num_cat_dims", 6)
    num_bins = pol_cfg.get("num_bins", 11)

    # ── Shared helpers ──────────────────────────────────────────

    def bins_to_env_action(bin_indices, gripper_state):
        """Convert 6D multi-categorical bin indices -> 7D RMPFlow env action."""
        center = (num_bins - 1) / 2.0
        threshold = 2.0
        normalized = (bin_indices.float() - center) / center
        xyz = normalized[:, :3]
        max_delta_rot = 0.5
        rot_xy = normalized[:, 3:5] * max_delta_rot

        g_bin = bin_indices[:, 5].float()
        new_gs = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0
        new_gs[g_bin > center + threshold - 1] = 1.0

        zeros1 = torch.zeros(bin_indices.shape[0], 1, device=bin_indices.device)
        env_act = torch.cat([xyz, rot_xy, zeros1, new_gs], dim=-1)
        return env_act, new_gs

    def create_env(use_dummy_alice=True, pos_threshold=None):
        """Create the IsaacLab environment with wrapper."""
        env_cfg = AsyncDualPlayEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        base_env = ManagerBasedRLEnv(cfg=env_cfg)

        wrapper_kwargs = dict(
            alice_timesteps=100,
            bob_timesteps=200,
            max_goals_per_episode=5,
            device=base_env.device,
        )

        if use_dummy_alice:
            env = DummyAliceWrapper(env=base_env, **wrapper_kwargs)
        else:
            env = AsyncDualPlayEnvWrapper(base_env, **wrapper_kwargs)

        if pos_threshold is not None:
            env.episode_manager.pos_threshold = pos_threshold
            print(f"  [Override] pos_threshold = {pos_threshold}")

        return env

    def create_bob_agent(env, learn_overrides=None, policy_overrides=None):
        """Create Bob's PPOABC agent with config overrides."""
        bob_cfg = copy.deepcopy(ppo_cfg["params"])
        bob_cfg["policy"]["use_goal_encoder"] = True

        # Shorten rollout for test speed
        bob_cfg["learn"]["nsteps"] = 64
        bob_cfg["learn"]["noptepochs"] = 3
        bob_cfg["learn"]["nminibatches"] = 2

        if learn_overrides:
            bob_cfg["learn"].update(learn_overrides)
        if policy_overrides:
            bob_cfg["policy"].update(policy_overrides)

        # Multi-categorical action space
        _mc_act_dim = num_cat_dims if use_mc else env.action_space.shape[0]
        _mc_space = gym.spaces.Box(
            low=0, high=num_bins - 1, shape=(_mc_act_dim,), dtype=np.float32
        )

        bob_ppo = PPOABC(
            vec_env=env,
            cfg_train=bob_cfg,
            device=env.device,
            sampler="sequential",
            log_dir=f"runs/test_abc_step{args.step}/bob",
            asymmetric=False,
        )
        bob_ppo.observation_space = env.bob_observation_space
        bob_ppo.state_space = bob_ppo.observation_space
        if use_mc:
            bob_ppo.action_space = _mc_space
            bob_ppo.desired_kl = None

        # Rebuild actor_critic with the correct Bob obs shape (51-dim after Euler switch)
        bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
            bob_ppo.observation_space.shape,
            bob_ppo.state_space.shape,
            bob_ppo.action_space.shape if use_mc else env.action_space.shape,
            bob_ppo.init_noise_std,
            bob_ppo.model_cfg,
            asymmetric=False,
        ).to(env.device)
        bob_ppo.optimizer = torch.optim.Adam(
            bob_ppo.actor_critic.parameters(), lr=bob_ppo.learning_rate
        )

        # Storage for Bob rollouts
        max_bob_steps = env.episode_manager.bob_timesteps + 10
        bob_ppo.storage = RolloutStorage(
            env.num_envs,
            max_bob_steps,
            bob_ppo.observation_space.shape,
            bob_ppo.state_space.shape,
            bob_ppo.action_space.shape if use_mc else env.action_space.shape,
            bob_ppo.device,
            "sequential",
        )

        # ABC demo buffer
        _abc_act_shape = (num_cat_dims,) if use_mc else env.action_space.shape
        bob_ppo.abc_buffer = GPUDemonstrationBuffer(
            capacity=100000,
            obs_shape=bob_ppo.observation_space.shape,
            states_shape=bob_ppo.observation_space.shape,
            actions_shape=_abc_act_shape,
            device=env.device,
        )
        bob_ppo.set_abc_buffer(bob_ppo.abc_buffer)

        return bob_ppo

    def create_alice_agent(env, learn_overrides=None):
        """Create Alice's PPO agent."""
        alice_cfg = copy.deepcopy(ppo_cfg["params"])
        alice_cfg["learn"]["nsteps"] = 64
        alice_cfg["learn"]["noptepochs"] = 3
        alice_cfg["learn"]["nminibatches"] = 2
        if learn_overrides:
            alice_cfg["learn"].update(learn_overrides)

        _mc_act_dim = num_cat_dims if use_mc else env.action_space.shape[0]
        _mc_space = gym.spaces.Box(
            low=0, high=num_bins - 1, shape=(_mc_act_dim,), dtype=np.float32
        )

        alice_ppo = PPO(
            vec_env=env,
            cfg_train=alice_cfg,
            device=env.device,
            sampler="sequential",
            log_dir=f"runs/test_abc_step{args.step}/alice",
            asymmetric=False,
        )
        alice_ppo.observation_space = env.alice_observation_space
        alice_ppo.state_space = alice_ppo.observation_space
        if use_mc:
            alice_ppo.action_space = _mc_space
            alice_ppo.desired_kl = None

        alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
            alice_ppo.observation_space.shape,
            alice_ppo.state_space.shape,
            alice_ppo.action_space.shape if use_mc else env.action_space.shape,
            alice_ppo.init_noise_std,
            alice_ppo.model_cfg,
            asymmetric=False,
        ).to(env.device)
        alice_ppo.optimizer = torch.optim.Adam(
            alice_ppo.actor_critic.parameters(), lr=alice_ppo.learning_rate
        )

        max_alice_steps = env.episode_manager.alice_timesteps + 10
        alice_ppo.storage = RolloutStorage(
            env.num_envs,
            max_alice_steps,
            alice_ppo.observation_space.shape,
            alice_ppo.state_space.shape,
            alice_ppo.action_space.shape if use_mc else env.action_space.shape,
            alice_ppo.device,
            "sequential",
        )

        return alice_ppo

    # ════════════════════════════════════════════════════════════
    # STEP 1: Pure PPO Overfit
    # ════════════════════════════════════════════════════════════

    def run_step1():
        """
        Prove Bob can learn a basic Cartesian movement from sparse rewards ONLY.

        DummyAliceWrapper teleports the block to [0.15, 0.5, 0.05].
        abc_coef=0.0 (no behavioral cloning), ent_coef=0.0 (no exploration noise).

        If this works: PPO, RMPFlow, and env resets are mathematically sound.
        """
        print("\n" + "=" * 70)
        print("  STEP 1: Pure PPO Overfit (Isolate Sparse Reward)")
        print("  DummyAlice=ON, abc_coef=0.0, ent_coef=0.0")
        print("=" * 70)

        env = create_env(use_dummy_alice=True, pos_threshold=args.pos_threshold)
        bob_ppo = create_bob_agent(env, learn_overrides={
            "abc_coef": 0.0,
            "ent_coef": 0.0,
        })
        alice_ppo = create_alice_agent(env)

        gripper_state = torch.ones(env.num_envs, 1, device=env.device)
        sr_buf = deque(maxlen=50)
        max_rew_ever = 0.0  # highest reward seen across all iterations

        thr = env.episode_manager.pos_threshold
        print(f"  pos_threshold = {thr:.3f} m")
        print(f"\n  {'Iter':>6} | {'SR':>8} | {'Val Loss':>10} | {'Surr Loss':>10} | {'MaxRew':>8}")
        print("  " + "-" * 60)

        for it in range(1, args.num_iterations + 1):
            alice_ppo.storage.clear()
            bob_ppo.storage.clear()

            obs, info = env.reset()
            alice_obs = obs[:, :env.alice_obs_dim]
            gripper_state.fill_(1.0)

            # -- Alice phase: DummyAliceWrapper teleports the block --
            alice_timesteps = env.episode_manager.alice_timesteps
            for step in range(alice_timesteps):
                with torch.no_grad():
                    if use_mc:
                        a_acts, a_lp, a_val, a_mu, a_sigma = alice_ppo.actor_critic.act(alice_obs, None)
                        a_env, gripper_state = bins_to_env_action(a_acts, gripper_state)
                    else:
                        a_acts, a_lp, a_val, a_mu, a_sigma = alice_ppo.actor_critic.act(alice_obs, None)
                        a_env = a_acts

                obs, rew, term, trunc, extras = env.step(a_env)
                alice_obs_new = obs[:, :env.alice_obs_dim]

                a_masks = torch.ones(env.num_envs, 1, device=env.device)
                alice_ppo.storage.add_transitions(
                    alice_obs, alice_obs_new, a_acts, rew, term | trunc,
                    a_val, a_lp.unsqueeze(-1) if a_lp.dim() == 1 else a_lp,
                    a_mu, a_sigma, a_masks,
                )
                alice_obs = alice_obs_new

            # -- Bob phase: learn from sparse reward --
            bob_timesteps = env.episode_manager.bob_timesteps
            bob_obs = obs[:, env.alice_obs_dim:]
            iter_successes = 0
            iter_attempts = 0

            gripper_state.fill_(1.0)
            iter_rew_max = 0.0
            for step in range(bob_timesteps):
                with torch.no_grad():
                    if use_mc:
                        b_acts, b_lp, b_val, b_mu, b_sigma = bob_ppo.actor_critic.act(bob_obs, None)
                        b_env, gripper_state = bins_to_env_action(b_acts, gripper_state)
                    else:
                        b_acts, b_lp, b_val, b_mu, b_sigma = bob_ppo.actor_critic.act(bob_obs, None)
                        b_env = b_acts

                obs, rew, term, trunc, extras = env.step(b_env)
                iter_rew_max = max(iter_rew_max, rew.max().item())
                bob_obs_new = obs[:, env.alice_obs_dim:]

                b_masks = torch.ones(env.num_envs, 1, device=env.device)
                bob_ppo.storage.add_transitions(
                    bob_obs, bob_obs_new, b_acts, rew, term | trunc,
                    b_val, b_lp.unsqueeze(-1) if b_lp.dim() == 1 else b_lp,
                    b_mu, b_sigma, b_masks,
                )
                bob_obs = bob_obs_new

                # Track Bob success from extras
                ep_info = extras.get("episode_manager", {})
                if ep_info and "bob_done_this_step" in ep_info:
                    finished = torch.where(ep_info["bob_done_this_step"])[0]
                    if len(finished) > 0:
                        iter_attempts += len(finished)
                        iter_successes += int(
                            ep_info["bob_success_this_step"][finished].sum().item()
                        )

            # -- Updates --
            dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
            alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
            alice_ppo.update()
            alice_ppo.storage.clear()

            with torch.no_grad():
                _, _, last_val, _, _ = bob_ppo.actor_critic.act(bob_obs, None)
            bob_ppo.storage.compute_returns(last_val, bob_ppo.gamma, bob_ppo.lam)
            val_loss, surr_loss, bc_loss, _ = bob_ppo.update()
            bob_ppo.storage.clear()

            sr = iter_successes / max(1, iter_attempts)
            sr_buf.append(sr)
            max_rew_ever = max(max_rew_ever, iter_rew_max)

            if it % 10 == 0 or it == 1:
                avg_sr = np.mean(sr_buf) if sr_buf else 0.0
                print(f"  {it:>6} | {avg_sr:>7.1%} | {val_loss:>10.4f} | {surr_loss:>10.4f} | {max_rew_ever:>8.4f}")

        # -- Verdict --
        final_sr = np.mean(sr_buf) if sr_buf else 0.0
        print(f"\n  Final rolling SR:  {final_sr:.1%}")
        print(f"  Max reward seen:   {max_rew_ever:.4f}  (0.0 = reward never fired → check reward pipeline)")
        passed = final_sr > 0.5
        print(f"  STEP 1 {'PASSED' if passed else 'FAILED'}: "
              f"SR {final_sr:.1%} {'>' if passed else '<='} 50% threshold")
        print(f"  (Paper expects SR -> 1.0 with enough iterations)")

        simulation_app.close()
        return passed

    # ════════════════════════════════════════════════════════════
    # STEP 2: Pure BC Overfit
    # ════════════════════════════════════════════════════════════

    def run_step2():
        """
        Prove the ABC loss formula works by injecting a perfect trajectory.

        value_loss_coef=0.0 (ignore PPO), abc_coef=1.0 (pure BC).
        Perfect trajectory: constant bin=8 for X (positive movement),
        bin=5 (center/no movement) for everything else.

        If this works: BC loss, gradient flow, and action convergence are correct.
        Bob's arm should blindly drift in +X regardless of the actual goal.
        """
        print("\n" + "=" * 70)
        print("  STEP 2: Pure BC Overfit (Isolate ABC Loss)")
        print("  God Mode trajectory injected, abc_coef=1.0, value_loss_coef=0.0")
        print("=" * 70)

        env = create_env(use_dummy_alice=True)  # env needed for obs shape
        bob_ppo = create_bob_agent(env, learn_overrides={
            "abc_coef": 1.0,
            "value_loss_coef": 0.0,
            "ent_coef": 0.0,
        })

        # ── Inject God Mode trajectory ─────────────────────────
        # Constant action: bin 8 for X (strong positive), bin 5 for rest.
        # _act_dim matches what the abc_buffer stores (num_cat_dims if MC, else env action dim).
        _act_dim = num_cat_dims if use_mc else env.action_space.shape[0]
        GOD_ACTION = torch.full((_act_dim,), 5.0, device=env.device)  # center = no movement
        GOD_ACTION[0] = 8.0   # X: bin 8 → positive movement

        print(f"\n  God Mode action (dim={_act_dim}): {GOD_ACTION.tolist()}")
        center = (num_bins - 1) / 2.0
        delta_x = (GOD_ACTION[0].item() - center) / center * 0.05
        print(f"  → X delta per step: {delta_x:+.4f} m (should be +0.030)")

        # Generate fake observations from the real env and pair with God actions
        NUM_FAKE_STEPS = 200
        print(f"  Injecting {NUM_FAKE_STEPS} God Mode demo steps into ABC buffer...")

        obs, info = env.reset()
        alice_obs = obs[:, :env.alice_obs_dim]
        gripper_state = torch.ones(env.num_envs, 1, device=env.device)

        # Step a few times to collect varied obs. goal_states is None until Alice completes,
        # so we also keep raw alice obs as a fallback. We tile to NUM_FAKE_STEPS afterwards.
        fake_obs_list = []
        alice_obs_list = []
        for step in range(NUM_FAKE_STEPS // env.num_envs + 2):
            zero_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
            obs, _, _, _, _ = env.step(zero_act)
            alice_obs = obs[:, :env.alice_obs_dim]
            alice_obs_list.append(alice_obs.clone())
            goal_states = env.episode_manager.goal_states
            if goal_states is not None:
                bc_obs = env.construct_bob_observation(alice_obs, goal_states)
                fake_obs_list.append(bc_obs.clone())

        if not fake_obs_list:
            print("  [Step2] goal_states never set; using dummy zero-goal obs")
            all_alice = torch.cat(alice_obs_list, dim=0)
            fake_goal = torch.zeros(all_alice.shape[0], 12, device=env.device)  # 6D per object × 2 (Euler format)
            bc_obs = env.construct_bob_observation(all_alice, fake_goal)
            fake_obs_list.append(bc_obs)

        # Tile to reach exactly NUM_FAKE_STEPS rows
        raw_obs = torch.cat(fake_obs_list, dim=0)
        if raw_obs.shape[0] < NUM_FAKE_STEPS:
            reps = (NUM_FAKE_STEPS + raw_obs.shape[0] - 1) // raw_obs.shape[0]
            raw_obs = raw_obs.repeat(reps, 1)
        fake_obs = raw_obs[:NUM_FAKE_STEPS]
        print(f"  fake_obs shape: {fake_obs.shape}")  # should be [200, bob_obs_dim]

        fake_acts = GOD_ACTION.unsqueeze(0).expand(NUM_FAKE_STEPS, -1).clone()

        # Compute old_log_probs for clipped ratio ABC
        with torch.no_grad():
            old_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(fake_obs, None, fake_acts)

        bob_ppo.abc_buffer.add_trajectory(
            fake_obs,                                                     # observations
            fake_obs,                                                     # states
            fake_acts,                                                    # actions
            torch.zeros(NUM_FAKE_STEPS, device=env.device),               # rewards
            torch.zeros(NUM_FAKE_STEPS, device=env.device).byte(),        # dones
            torch.zeros(NUM_FAKE_STEPS, device=env.device),               # values
            old_lp.view(-1, 1),                                           # log_probs
            torch.zeros_like(fake_acts),                                  # mu
            torch.zeros_like(fake_acts),                                  # sigma
            torch.zeros(NUM_FAKE_STEPS, 1, device=env.device),            # returns
            torch.zeros(NUM_FAKE_STEPS, 1, device=env.device),            # advantages
        )
        print(f"  ABC buffer size: {bob_ppo.abc_buffer.size}")

        # ── Training loop ──────────────────────────────────────
        print(f"\n  Training Bob with pure ABC for {args.num_iterations} iters...")
        print(f"  {'Iter':>6} | {'BC Loss':>10} | {'Dim0 Mode':>10} | {'All Match':>10}")
        print("  " + "-" * 50)

        bc_losses = []
        dim0_modes = []

        for it in range(1, args.num_iterations + 1):
            # Fill Bob's rollout storage (needed for update() to run)
            bob_ppo.storage.clear()
            obs, info = env.reset()
            bob_obs = obs[:, env.alice_obs_dim:]
            nsteps = min(32, bob_ppo.storage.num_transitions_per_env)

            for step in range(nsteps):
                with torch.no_grad():
                    b_acts, b_lp, b_val, b_mu, b_sigma = bob_ppo.actor_critic.act(bob_obs, None)
                    if use_mc:
                        b_env, gripper_state = bins_to_env_action(b_acts, gripper_state)
                    else:
                        b_env = b_acts

                obs, rew, term, trunc, extras = env.step(b_env)
                bob_obs_new = obs[:, env.alice_obs_dim:]

                bob_ppo.storage.add_transitions(
                    bob_obs, bob_obs_new, b_acts,
                    torch.zeros(env.num_envs, 1, device=env.device),  # zero reward
                    term | trunc, b_val,
                    b_lp.unsqueeze(-1) if b_lp.dim() == 1 else b_lp,
                    b_mu, b_sigma,
                )
                bob_obs = bob_obs_new

            with torch.no_grad():
                _, _, last_val, _, _ = bob_ppo.actor_critic.act(bob_obs, None)
            bob_ppo.storage.compute_returns(last_val, bob_ppo.gamma, bob_ppo.lam)

            # Refresh old_log_probs EVERY iteration so ratio never saturates the clip.
            # When old_lp is current, ratio≈1 and gradients flow continuously.
            # This is equivalent to NLL while keeping the clip for numerical stability.
            with torch.no_grad():
                new_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(
                    fake_obs, None, fake_acts
                )
            bob_ppo.abc_buffer.step = 0
            bob_ppo.abc_buffer.full = False
            bob_ppo.abc_buffer.add_trajectory(
                fake_obs, fake_obs, fake_acts,
                torch.zeros(NUM_FAKE_STEPS, device=env.device),
                torch.zeros(NUM_FAKE_STEPS, device=env.device).byte(),
                torch.zeros(NUM_FAKE_STEPS, device=env.device),
                new_lp.view(-1, 1),           # ← fresh old_lp: ratio starts at 1
                torch.zeros_like(fake_acts),
                torch.zeros_like(fake_acts),
                torch.zeros(NUM_FAKE_STEPS, 1, device=env.device),
                torch.zeros(NUM_FAKE_STEPS, 1, device=env.device),
            )

            val_loss, surr_loss, bc_loss, _ = bob_ppo.update()
            bob_ppo.storage.clear()

            # Also track raw NLL separately (always meaningful, not clipped)
            raw_nll = -new_lp.mean().item()
            bc_losses.append(raw_nll)

            # Check: does Bob's greedy action match the God Mode action?
            with torch.no_grad():
                test_acts = bob_ppo.actor_critic.act_inference(fake_obs[:64])
                dim0_mode = test_acts[:, 0].mode().values.item()
                all_match = (test_acts == GOD_ACTION.unsqueeze(0)).all(dim=-1).float().mean().item()
                dim0_modes.append(dim0_mode)

            if it % 10 == 0 or it == 1:
                print(f"  {it:>6} | NLL:{raw_nll:>+8.3f} | BC:{bc_loss:>+8.3f} | Dim0:{dim0_mode:>4.0f} | Match:{all_match:>5.1%}")

        # -- Verdict --
        print(f"\n  BC Loss:    {bc_losses[0]:+.4f} -> {bc_losses[-1]:+.4f}")
        final_mode = dim0_modes[-1] if dim0_modes else -1
        bc_decreased = bc_losses[-1] < bc_losses[0]
        dim0_correct = abs(final_mode - GOD_ACTION[0].item()) < 1.0

        passed = bc_decreased and dim0_correct
        print(f"  Dim 0 mode: {final_mode:.0f} (target: {GOD_ACTION[0].item():.0f})")
        print(f"  STEP 2 {'PASSED' if passed else 'FAILED'}: "
              f"BC loss {'decreased' if bc_decreased else 'DID NOT decrease'}, "
              f"Dim 0 {'converged' if dim0_correct else 'DID NOT converge'}")

        simulation_app.close()
        return passed

    # ════════════════════════════════════════════════════════════
    # STEP 3: Easy Mode Integration
    # ════════════════════════════════════════════════════════════

    def run_step3():
        """
        Full ASP integration with relaxed success threshold.

        Real Alice (no DummyAliceWrapper), abc_coef=0.5, ent_coef=0.01.
        pos_threshold=0.20 (20cm success radius) so Alice's random flailing
        easily creates achievable goals and Bob easily stumbles into success.

        If this works: the entire codebase is correct. Scale up and tighten.
        """
        print("\n" + "=" * 70)
        print("  STEP 3: Easy Mode Integration (Full ASP)")
        print("  Real Alice, abc_coef=0.5, ent_coef=0.01, pos_threshold=0.20")
        print("=" * 70)

        env = create_env(use_dummy_alice=False, pos_threshold=0.20)
        alice_ppo = create_alice_agent(env)
        bob_ppo = create_bob_agent(env, learn_overrides={
            "abc_coef": 0.5,
            "ent_coef": 0.01,
        })

        alice_gripper = torch.ones(env.num_envs, 1, device=env.device)
        bob_gripper = torch.ones(env.num_envs, 1, device=env.device)

        alice_sr_buf = deque(maxlen=50)
        bob_sr_buf = deque(maxlen=50)

        print(f"\n  {'Iter':>6} | {'Alice Rew':>10} | {'Bob SR':>8} | {'ABC Buf':>8} | {'BC Loss':>10}")
        print("  " + "-" * 60)

        for it in range(1, args.num_iterations + 1):
            alice_ppo.storage.clear()
            bob_ppo.storage.clear()

            obs, info = env.reset()
            alice_obs = obs[:, :env.alice_obs_dim]
            alice_gripper.fill_(1.0)
            bob_gripper.fill_(1.0)

            # -- Alice phase --
            alice_traj_obs = []
            alice_traj_act = []
            alice_timesteps = env.episode_manager.alice_timesteps

            for step in range(alice_timesteps):
                with torch.no_grad():
                    a_acts, a_lp, a_val, a_mu, a_sigma = alice_ppo.actor_critic.act(alice_obs, None)
                    if use_mc:
                        a_env, alice_gripper = bins_to_env_action(a_acts, alice_gripper)
                    else:
                        a_env = a_acts

                alice_traj_obs.append(alice_obs.clone())
                alice_traj_act.append(a_acts.clone())

                obs, rew, term, trunc, extras = env.step(a_env)
                alice_obs_new = obs[:, :env.alice_obs_dim]

                a_masks = torch.ones(env.num_envs, 1, device=env.device)
                alice_ppo.storage.add_transitions(
                    alice_obs, alice_obs_new, a_acts, rew, term | trunc,
                    a_val, a_lp.unsqueeze(-1) if a_lp.dim() == 1 else a_lp,
                    a_mu, a_sigma, a_masks,
                )
                alice_obs = alice_obs_new

            goal_states = env.episode_manager.goal_states
            goal_valid = env.episode_manager.goal_valid

            # Track Alice valid goals
            n_valid = goal_valid.sum().item() if goal_valid is not None else 0
            alice_sr_buf.append(n_valid / env.num_envs)

            # -- Bob phase --
            bob_obs = obs[:, env.alice_obs_dim:]
            bob_timesteps = env.episode_manager.bob_timesteps
            iter_successes = 0
            iter_attempts = 0

            for step in range(bob_timesteps):
                with torch.no_grad():
                    b_acts, b_lp, b_val, b_mu, b_sigma = bob_ppo.actor_critic.act(bob_obs, None)
                    if use_mc:
                        b_env, bob_gripper = bins_to_env_action(b_acts, bob_gripper)
                    else:
                        b_env = b_acts

                obs, rew, term, trunc, extras = env.step(b_env)
                bob_obs_new = obs[:, env.alice_obs_dim:]

                b_masks = torch.ones(env.num_envs, 1, device=env.device)
                bob_ppo.storage.add_transitions(
                    bob_obs, bob_obs_new, b_acts, rew, term | trunc,
                    b_val, b_lp.unsqueeze(-1) if b_lp.dim() == 1 else b_lp,
                    b_mu, b_sigma, b_masks,
                )
                bob_obs = bob_obs_new

                ep_info = extras.get("episode_manager", {})
                if ep_info and "bob_done_this_step" in ep_info:
                    finished = torch.where(ep_info["bob_done_this_step"])[0]
                    if len(finished) > 0:
                        iter_attempts += len(finished)
                        iter_successes += int(
                            ep_info["bob_success_this_step"][finished].sum().item()
                        )

            bob_sr = iter_successes / max(1, iter_attempts)
            bob_sr_buf.append(bob_sr)

            # -- ABC buffer push (paper: only when Bob failed) --
            if goal_valid is not None and goal_valid.any():
                bob_success = env.episode_manager.bob_success
                valid_failed = torch.where(goal_valid & ~bob_success)[0]
                for eid in valid_failed:
                    eid = eid.item()
                    traj_o = torch.stack(
                        [alice_traj_obs[s][eid] for s in range(len(alice_traj_obs))]
                    )
                    traj_a = torch.stack(
                        [alice_traj_act[s][eid] for s in range(len(alice_traj_act))]
                    )
                    g = goal_states[eid].unsqueeze(0).expand(len(traj_o), -1)
                    bc_obs = env.construct_bob_observation(traj_o, g)

                    with torch.no_grad():
                        old_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(
                            bc_obs, None, traj_a
                        )

                    bob_ppo.abc_buffer.add_trajectory(
                        bc_obs, bc_obs, traj_a,
                        torch.zeros(len(traj_o), device=env.device),
                        torch.zeros(len(traj_o), device=env.device).byte(),
                        torch.zeros(len(traj_o), device=env.device),
                        old_lp.view(-1, 1),
                        torch.zeros_like(traj_a),
                        torch.zeros_like(traj_a),
                        torch.zeros(len(traj_o), 1, device=env.device),
                        torch.zeros(len(traj_o), 1, device=env.device),
                    )

            # -- Alice reward & update --
            alice_outcome_rewards = torch.where(
                ~env.episode_manager.bob_success & (goal_valid if goal_valid is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
                torch.tensor(5.0, device=env.device),
                torch.tensor(0.0, device=env.device),
            )
            if alice_ppo.storage.step > 0:
                last_idx = alice_ppo.storage.step - 1
                alice_ppo.storage.rewards[last_idx].copy_(alice_outcome_rewards.view(-1, 1))
                alice_ppo.storage.dones[last_idx].fill_(1.0)

            dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
            alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
            alice_ppo.update()
            alice_ppo.storage.clear()

            # -- Bob update --
            with torch.no_grad():
                _, _, last_val, _, _ = bob_ppo.actor_critic.act(bob_obs, None)
            bob_ppo.storage.compute_returns(last_val, bob_ppo.gamma, bob_ppo.lam)
            bob_ppo.current_learning_iteration = it
            val_loss, surr_loss, bc_loss, _ = bob_ppo.update()
            bob_ppo.storage.clear()

            if it % 10 == 0 or it == 1:
                avg_alice = np.mean(alice_sr_buf) if alice_sr_buf else 0.0
                avg_bob = np.mean(bob_sr_buf) if bob_sr_buf else 0.0
                buf_size = bob_ppo.abc_buffer.size
                print(
                    f"  {it:>6} | {avg_alice:>9.1%} | {avg_bob:>7.1%} | {buf_size:>8} | {bc_loss:>+10.4f}"
                )

        # -- Verdict --
        avg_alice = np.mean(alice_sr_buf) if alice_sr_buf else 0.0
        avg_bob = np.mean(bob_sr_buf) if bob_sr_buf else 0.0
        print(f"\n  Final Alice valid-goal rate: {avg_alice:.1%}")
        print(f"  Final Bob success rate:      {avg_bob:.1%}")
        print(f"  ABC buffer size:             {bob_ppo.abc_buffer.size}")

        alice_ok = avg_alice > 0.1  # Alice should set some valid goals with 20cm threshold
        bob_ok = avg_bob > 0.05    # Bob should stumble into at least a few successes
        passed = alice_ok and bob_ok
        print(f"\n  STEP 3 {'PASSED' if passed else 'FAILED'}:")
        print(f"    Alice valid goals: {avg_alice:.1%} {'> 10%' if alice_ok else '<= 10%'}")
        print(f"    Bob SR:            {avg_bob:.1%} {'> 5%' if bob_ok else '<= 5%'}")
        if passed:
            print(f"\n  Full ASP pipeline verified. Scale up to 1024 envs and")
            print(f"  shrink pos_threshold from 0.20 -> 0.04 as they master the physics.")

        simulation_app.close()
        return passed

    # ════════════════════════════════════════════════════════════
    # STEP 4: Goal Distance Sanity Check
    # ════════════════════════════════════════════════════════════

    def run_step4():
        """
        Prove goal_distance() returns ~0 when objects ARE at their stored goals.

        DummyGoalDistanceWrapper teleports BOTH objects to their stored goal at
        Bob step 30, then immediately measures goal_distance().

        PASS: [DistCheck] lines show pos < 0.01 and rot < 0.01 (marked ✓)
        FAIL: pos ≈ 2 m  → double env_origins subtraction in goal_distance()
              pos ≈ large → world/local frame mismatch in stored goal_states
        """
        print("\n" + "=" * 70)
        print("  STEP 4: Goal Distance Sanity Check (DummyGoalDistanceWrapper)")
        print("  Teleports objects to stored goals; goal_distance() must return ~0")
        print("=" * 70)

        env_cfg = AsyncDualPlayEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = DummyGoalDistanceWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=100,
            bob_timesteps=200,
            teleport_step=30,
        )

        bugs_found = 0
        checks_done = 0

        for it in range(1, args.num_iterations + 1):
            obs, _ = env.reset()
            zero_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)

            # Step through Alice phase
            for _ in range(env.episode_manager.alice_timesteps):
                obs, _, _, _, _ = env.step(zero_act)

            # Step through enough of Bob phase to trigger teleport (step 30) + a few more
            for bob_step in range(env.teleport_step + 5):
                obs, _, _, _, extras = env.step(zero_act)

            # Count ✗ BUG markers from wrapper prints (we track via extras if available)
            checks_done += env.num_envs

        print(f"\n  STEP 4 SUMMARY: reviewed [DistCheck] lines above ({checks_done} checks)")
        print(f"  PASS: all lines marked ✓  (pos < 0.01, rot < 0.01)")
        print(f"  FAIL: any line marked ✗ BUG  (frame error in goal_distance)")

        simulation_app.close()

    # ════════════════════════════════════════════════════════════
    # STEP 5: Reward Pipeline Sanity Check
    # ════════════════════════════════════════════════════════════

    def run_step5():
        """
        Prove sparse reward fires when the object IS physically at the goal.

        DummyBobWrapper teleports the target TO its goal at Bob step 50.
        The very next env.step() should return rew > 0 and SR should → 1.0.

        PASS: SR > 0.9  (reward fires every episode after teleport)
        FAIL: SR = 0.0  → success threshold is broken or reward function never fires
        """
        print("\n" + "=" * 70)
        print("  STEP 5: Reward Pipeline Sanity Check (DummyBobWrapper)")
        print("  Teleports target to goal at Bob step 50; reward must fire")
        print("=" * 70)

        env_cfg = AsyncDualPlayEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = DummyBobWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=100,
            bob_timesteps=200,
            teleport_step=50,
        )

        sr_buf = deque(maxlen=50)
        max_rew_ever = 0.0

        print(f"\n  {'Iter':>6} | {'SR':>8} | {'MaxRew':>8}")
        print("  " + "-" * 30)

        for it in range(1, args.num_iterations + 1):
            obs, _ = env.reset()
            zero_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)

            # Alice phase — zero actions; DummyAliceWrapper (base of DummyBobWrapper)
            # teleports target to fixed goal so goal_states is set
            for _ in range(env.episode_manager.alice_timesteps):
                obs, _, _, _, _ = env.step(zero_act)

            # Bob phase — zero actions; DummyBobWrapper teleports target onto the goal
            # at step 50; reward should fire at step 51
            iter_successes = 0
            iter_attempts = 0
            iter_rew_max = 0.0

            for _ in range(env.episode_manager.bob_timesteps):
                obs, rew, term, trunc, extras = env.step(zero_act)
                iter_rew_max = max(iter_rew_max, rew.max().item())

                ep_info = extras.get("episode_manager", {})
                if ep_info and "bob_done_this_step" in ep_info:
                    finished = torch.where(ep_info["bob_done_this_step"])[0]
                    if len(finished) > 0:
                        iter_attempts += len(finished)
                        iter_successes += int(
                            ep_info["bob_success_this_step"][finished].sum().item()
                        )

            sr = iter_successes / max(1, iter_attempts)
            sr_buf.append(sr)
            max_rew_ever = max(max_rew_ever, iter_rew_max)

            if it % 5 == 0 or it == 1:
                avg_sr = np.mean(sr_buf) if sr_buf else 0.0
                print(f"  {it:>6} | {avg_sr:>7.1%} | {max_rew_ever:>8.4f}")

        final_sr = np.mean(sr_buf) if sr_buf else 0.0
        passed = final_sr > 0.9
        print(f"\n  Final SR:        {final_sr:.1%}  (target > 90%)")
        print(f"  Max reward seen: {max_rew_ever:.4f}")
        print(f"  STEP 5 {'PASSED' if passed else 'FAILED'}: "
              f"{'reward fires correctly' if passed else 'reward never fired — check _check_bob_success or pos_threshold'}")

        simulation_app.close()
        return passed

    # ════════════════════════════════════════════════════════════
    # STEP 6: Alice Movement Detection Check
    # ════════════════════════════════════════════════════════════

    def run_step6():
        """
        Prove validate_goal() correctly detects that Alice moved the object.

        DummyMovementWrapper teleports the target 30 cm during Alice's phase.
        At Alice phase end, validate_goal() should measure ~0.30 m and emit
        "Valid Goal" lines.

        PASS: [MoveCheck] shows measured_dist ≈ 0.30 m (marked ✓)
        FAIL: measured_dist ≈ 0.00 m → initial_states stale or wrong frame
              measured_dist ≈ 2 m   → env_origins subtraction bug
        """
        print("\n" + "=" * 70)
        print("  STEP 6: Alice Movement Detection (DummyMovementWrapper)")
        print("  Teleports target 30 cm during Alice phase; validate_goal must detect it")
        print("=" * 70)

        env_cfg = AsyncDualPlayEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = DummyMovementWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=100,
            bob_timesteps=200,
        )

        zero_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)

        for it in range(1, args.num_iterations + 1):
            obs, _ = env.reset()

            # Step through full Alice phase — wrapper emits [MoveCheck] at end
            for _ in range(env.episode_manager.alice_timesteps + 2):
                obs, _, _, _, _ = env.step(zero_act)

        simulation_app.close()

    # ── Dispatch ────────────────────────────────────────────────
    if args.step == "1":
        run_step1()
    elif args.step == "2":
        run_step2()
    elif args.step == "3":
        run_step3()
    elif args.step == "4":
        run_step4()
    elif args.step == "5":
        run_step5()
    elif args.step == "6":
        run_step6()


if __name__ == "__main__":
    main()
