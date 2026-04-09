import isaaclab.app
from isaaclab.app import AppLauncher

import gc
import math
import os
import sys
import threading
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


def install_noise_filter():
    """
    Permanently redirect C-level stderr (fd 2) through a pipe and drop lines
    that match known Lula/URDF/carb noise patterns.  Everything else is passed
    through to the original stderr unchanged.  Call once at process start.
    """
    _DROP = (
        b"[Lula] Joint",
        b"Warning: link",
        b"urdf_parser",
        b"flat_black",
        b"IMemoryBudgetManager",
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


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Async Dual Play PPO")
    parser.add_argument("--exp_name", type=str, default="async_dual_play_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Override rollout steps per env (to prevent OOM on high num_envs)",
    )
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--max_alice_bob_ratio",
        type=int,
        default=None,
        help="Max consecutive Bob-only updates before forcing an Alice update. "
        "Auto-computed as ceil(bob_timesteps / alice_timesteps) * max(1, 64 // num_envs) "
        "if not specified.",
    )
    parser.add_argument(
        "--chkpt_alice",
        type=str,
        default=None,
        help="Path to Alice checkpoint (.pt) for resuming training",
    )
    parser.add_argument(
        "--chkpt_bob",
        type=str,
        default=None,
        help="Path to Bob checkpoint (.pt) for resuming training",
    )
    parser.add_argument(
        "--resume_iteration",
        type=int,
        default=0,
        help="Starting iteration count when resuming from checkpoint",
    )
    parser.add_argument(
        "--arm_config",
        type=str,
        default="default",
        choices=["default", "rotated"],
    )
    parser.add_argument(
        "--dummy_alice", action="store_true", help="Use dummy Alice wrapper"
    )
    parser.add_argument(
        "--test_bob_reward",
        action="store_true",
        help="Test: use DummyBobWrapper (teleports target→goal at step 50). "
        "Expected: Bob Rew > 0, SR > 0 from iter 1. "
        "Verifies sparse reward threshold and +5 completion bonus.",
    )
    parser.add_argument(
        "--test_abc_verbose",
        action="store_true",
        help="Test: print ABC demo content (goal shape, obs range) each time "
        "a trajectory is added. Verifies goal is appended to obs and "
        "filtering only stores Bob-failure episodes.",
    )
    parser.add_argument(
        "--test_hparams",
        action="store_true",
        help="Test: print hyperparameter audit comparing loaded config "
        "against paper Table 2 values and exit.",
    )
    parser.add_argument(
        "--dummy_goal_distance",
        action="store_true",
        help="Test: use DummyGoalDistanceWrapper (snaps both objects to stored "
        "goals at Bob step 30, then measures goal_distance()). "
        "Expected: pos_dist < 0.01 and rot_dist < 0.01 (printed as ✓). "
        "A distance ≈ env_origin magnitude (~2m) means double-subtraction bug.",
    )
    parser.add_argument(
        "--test_movement",
        action="store_true",
        help="Test: use DummyMovementWrapper (teleports target to [0.15,0.5,0.05] "
        "local during Alice's phase). "
        "Expected: measured_dist ≈ 0.362 and validate_goal says 'Valid Goal'. "
        "measured_dist ≈ 0.000 means stale initial_states.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    install_noise_filter()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

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
    task_cfg_path = os.path.join(
        os.path.dirname(__file__), "cfg/task/AsyncDualPlay.yaml"
    )
    ppo_cfg_path = os.path.join(
        os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml"
    )
    ppo_cfg = load_cfg(ppo_cfg_path)
    task_cfg = load_cfg(task_cfg_path)

    alice_timesteps = task_cfg.get("alice_timesteps", 150)
    bob_timesteps = task_cfg.get("bob_timesteps", 200)
    max_goals_per_episode = task_cfg.get("max_goals_per_episode", 5)
    print(
        f"[Config] Episode structure: alice_timesteps={alice_timesteps}, "
        f"bob_timesteps={bob_timesteps}, max_goals={max_goals_per_episode} "
        f"(from AsyncDualPlay.yaml)"
    )

    if args.test_hparams:
        # Paper Table 2 reference values
        PAPER = {
            "gamma": 0.998,
            "lam": 0.95,
            "e_clip": 0.2,
            "entropy_coef": 0.01,
            "learning_rate": 3e-4,
            "mini_epochs": 3,  # "sample reuse (experience replay)"
            "critic_coef": 1.0,  # "value loss weight"
            "abc_coef": 0.5,  # "ABC loss weight β"
        }
        learn = ppo_cfg["params"].get("learn", {})
        actual = {
            "gamma": learn.get("gamma", "?"),
            "lam": learn.get("lam", "?"),
            "e_clip": learn.get("cliprange", "?"),
            "entropy_coef": learn.get("ent_coef", "?"),
            "learning_rate": learn.get("optim_stepsize", "?"),
            "mini_epochs": learn.get("noptepochs", "?"),
            "critic_coef": learn.get("value_loss_coef", "?"),
            "abc_coef": learn.get("abc_coef", "?"),
        }
        print("\n" + "=" * 60)
        print("HYPERPARAMETER AUDIT  (paper Table 2 vs loaded config)")
        print(f"{'Key':<18} {'Paper':>10} {'Loaded':>10}  {'OK?':>6}")
        print("-" * 60)
        all_ok = True
        for k, pv in PAPER.items():
            av = actual[k]
            ok = "✓" if av == pv else "✗ MISMATCH"
            if av != pv:
                all_ok = False
            print(f"  {k:<16} {str(pv):>10} {str(av):>10}  {ok}")
        print("=" * 60)
        print("All OK!" if all_ok else "Fix mismatches before large-scale training.")
        print("=" * 60 + "\n")
        import sys as _sys

        _sys.exit(0)

    if args.nsteps is not None:
        print(
            f"[Config] Overriding nsteps: {ppo_cfg['params']['learn']['nsteps']} -> {args.nsteps}"
        )
        ppo_cfg["params"]["learn"]["nsteps"] = args.nsteps

    # --- Multi-categorical action space config ---
    _pol_cfg = ppo_cfg["params"]["policy"]
    use_mc = _pol_cfg.get("use_multicategorical", False)
    num_cat_dims = _pol_cfg.get("num_cat_dims", 6)
    num_bins = _pol_cfg.get("num_bins", 11)
    if use_mc:
        print(
            f"[Config] Multi-categorical action space: {num_cat_dims} dims × {num_bins} bins "
            f"(physical scale set by env RMPFlow scale factor)"
        )

    def bins_to_env_action(
        bin_indices: "torch.Tensor", gripper_state: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Convert policy bin indices (N, 6) → 7D RMPFlow+gripper env action.

        XYZ: normalized = (bin - center) / center → [-1, 1]; env scale=0.05 → ±5cm/step.
        Rx, Ry: delta = (bin - center) / center * max_delta_rot (0.5 rad)

        Gripper (sticky): only the outer bins trigger a state change.
          Dead zone = center 3 bins (4/5/6) → keep previous gripper_state.
          This prevents random-policy spassing at the start of training.
          Threshold: bins 0-3 → close (-1), bins 4-6 → hold, bins 7-10 → open (+1)
        """
        center = (num_bins - 1) / 2.0  # 5.0 for 11 bins
        threshold = 2.0  # ±2 bins from center triggers change
        normalized = (bin_indices.float() - center) / center
        xyz = normalized[:, :3]  # [-1, 1]; RMPFlow scale=0.05 gives ±5cm/step

        # 2D wrist rotation (Rx, Ry) - scale 0.5 rad (~28 deg) at max bin
        max_delta_rot = 0.5
        rot_xy = normalized[:, 3:5] * max_delta_rot

        g_bin = bin_indices[:, 5].float()
        new_gs = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0  # bins 0-2  → close
        new_gs[g_bin > center + threshold - 1] = 1.0  # bins 8-10 → open
        # bins 3-7 → keep previous state

        # RMPFlow expects 6D (xyz, rpy) + 1D gripper. We zero out Rz.
        zeros1 = torch.zeros(bin_indices.shape[0], 1, device=bin_indices.device)
        return (
            torch.cat([xyz, rot_xy, zeros1, new_gs], dim=-1),
            new_gs,
        )

    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    if args.arm_config == "rotated":
        print(
            "[Config] Rotated arm configuration: shoulder −90°"
        )
        env_cfg.scene.robot.init_state.joint_pos["shoulder_pan_joint"] = 1.57
        env_cfg.scene.robot.init_state.joint_pos["elbow_joint"] = 2.356
        env_cfg.scene.robot.init_state.joint_pos["wrist_1_joint"] = -0.785
        env_cfg.scene.robot.init_state.joint_pos["wrist_2_joint"] = 1.57
        env_cfg.scene.robot.init_state.joint_pos["wrist_3_joint"] = 0
        

    print("Creating environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    if args.test_bob_reward:
        print(
            "[Test] --test_bob_reward: using DummyBobWrapper (teleports target→goal at Bob step 50)."
        )
        print("[Test] Expected: Bob Rew > 0 and SR > 0 from iteration 1.")
        env = DummyBobWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            teleport_step=50,
        )
    elif args.dummy_goal_distance:
        print(
            "[Test] --dummy_goal_distance: using DummyGoalDistanceWrapper "
            "(snaps both objects to stored goals at Bob step 30)."
        )
        print("[Test] Expected: [DistCheck] pos < 0.01 and rot < 0.01 marked ✓.")
        env = DummyGoalDistanceWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            teleport_step=30,
        )
    elif args.test_movement:
        print(
            "[Test] --test_movement: using DummyMovementWrapper "
            "(teleports target to [0.15,0.5,0.05] local during Alice's phase)."
        )
        print("[Test] Expected: [MoveCheck] measured_dist≈0.362 marked ✓.")
        env = DummyMovementWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
        )
    elif args.dummy_alice:
        env = DummyAliceWrapper(
            env=base_env, device=base_env.device,
            alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
        )
    else:
        env = AsyncDualPlayEnvWrapper(
            env=base_env, device=base_env.device, arm_config=args.arm_config,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            max_goals_per_episode=max_goals_per_episode,
        )

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

        _mc_space = gym_mc.spaces.Box(
            low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32
        )
        alice_ppo.action_space = _mc_space
        alice_ppo.desired_kl = None  # adaptive KL meaningless for discrete

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

    import copy

    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_goal_encoder"] = True

    bob_ppo = PPOABC(
        vec_env=env,
        cfg_train=bob_cfg,
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/bob",
        asymmetric=False,
    )
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space

    if use_mc:
        bob_ppo.action_space = _mc_space  # same 4D bin space as Alice
        bob_ppo.desired_kl = None

    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.init_noise_std,
        bob_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    # Scale goal_proj weights down at init: a large W_g saturates ReLUs before training starts.
    if hasattr(bob_ppo.actor_critic, '_goal_proj') and bob_ppo.actor_critic._goal_proj is not None:
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.01 / 0.5)
        print(f"  [Init] goal_proj scale reduced: ||W_g|| = {bob_ppo.actor_critic._goal_proj.weight.norm():.4f}")

    bob_ppo.optimizer = torch.optim.Adam(
        bob_ppo.actor_critic.parameters(), lr=bob_ppo.learning_rate
    )

    # Re-initialize Bob's standard PPO storage for 56 dims.
    # Must hold the full bob_timesteps rollout (same pattern as Alice's oversized storage).
    max_bob_steps = env.episode_manager.bob_timesteps + 10
    bob_storage_size = bob_ppo.num_transitions_per_env + max_bob_steps
    bob_ppo.storage = bob_ppo.storage.__class__(
        bob_ppo.vec_env.num_envs,
        bob_storage_size,
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
        capacity=50000,
        obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape,
        actions_shape=_abc_act_shape,
        device=env.device,
    )

    # LSTM hidden states (None for non-LSTM models — act_with_hidden handles both).
    _use_lstm = alice_ppo.actor_critic.use_lstm
    if _use_lstm:
        _lsz = alice_ppo.actor_critic.lstm_hidden_size
        alice_hidden = [
            torch.zeros(env.num_envs, _lsz, device=env.device),
            torch.zeros(env.num_envs, _lsz, device=env.device),
        ]
        bob_hidden = [
            torch.zeros(env.num_envs, _lsz, device=env.device),
            torch.zeros(env.num_envs, _lsz, device=env.device),
        ]
    else:
        alice_hidden = None
        bob_hidden = None

    # Sticky gripper state: starts open (+1), outer bins change it, center bins hold.
    alice_gripper_state = torch.ones(env.num_envs, 1, device=env.device)
    bob_gripper_state = torch.ones(env.num_envs, 1, device=env.device)

    # Historical policy pool: 20% of rollout envs use a past policy for stability.
    alice_pool = HistoricalPolicyPool(max_size=5)
    bob_pool = HistoricalPolicyPool(max_size=5)
    HIST_SAVE_INTERVAL = 50  # save snapshot every N bob_updates
    HIST_FRAC = 0.2  # fraction of envs using historical policy

    # --- Resume from checkpoint ---
    if args.chkpt_alice and os.path.isfile(args.chkpt_alice):
        alice_ppo.load(args.chkpt_alice)
        print(f"[Resume] Loaded Alice from {args.chkpt_alice}")
    if args.chkpt_bob and os.path.isfile(args.chkpt_bob):
        bob_ppo.load(args.chkpt_bob)
        print(f"[Resume] Loaded Bob from {args.chkpt_bob}")
        _abc_buf_path = os.path.join(os.path.dirname(args.chkpt_bob), "abc_buffer.pt")
        if os.path.isfile(_abc_buf_path):
            bob_ppo.abc_buffer.load(_abc_buf_path)
            print(f"[Resume] Loaded ABC buffer ({bob_ppo.abc_buffer.size} entries) from {_abc_buf_path}")
            
        _ep_mgr_path = args.chkpt_bob.replace("model_", "episode_manager_")
        if os.path.isfile(_ep_mgr_path):
            ep_sd = torch.load(_ep_mgr_path, map_location=env.device)
            env.episode_manager.load_state_dict(ep_sd)
            print(f"[Resume] Loaded EpisodeManager state from {_ep_mgr_path}")
        else:
            print(f"[Resume] WARNING: EpisodeManager checkpoint not found at {_ep_mgr_path}. Envs will start fresh.")

    # --- Agents ---
    alice_updates = 0
    bob_updates = args.resume_iteration

    if args.num_envs < 32:
        print(
            f"[WARNING] num_envs={args.num_envs} is very low. "
            f"Paper uses 1856 envs (batch_size=4096). Recommend at least 32 for sparse rewards."
        )

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")

    rollout_length = ppo_cfg["params"]["learn"]["nsteps"] * args.num_envs
    alice_rew_buf = deque(maxlen=rollout_length)
    bob_rew_buf = deque(maxlen=rollout_length)
    bob_success_buf = deque(maxlen=rollout_length)
    bob_pos_err_buf = deque(maxlen=rollout_length)
    bob_rot_err_buf = deque(maxlen=rollout_length)

    best_bob_success_rate = -1.0
    last_alice_mean_rew = 0.0  # gating value for Bob's ABC loss warmup

    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print(f"\n{'='*80}\nTRAINING RUN: {args.exp_name}\nLOG DIRECTORY: {run_dir}")
    print(f"  tensorboard --logdir {run_dir}\n{'='*80}\n")

    def perform_alice_update():
        """Run a PPO update for Alice after her rollout is complete."""
        nonlocal alice_updates, last_alice_mean_rew

        # Storage is oversized to alice_storage_size; do NOT gate on num_transitions_per_env.
        if alice_ppo.storage.step == 0:
            return

        dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
        alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)

        loss_val, loss_surr = alice_ppo.update()
        alice_ppo.storage.clear()

        mean_alice_rew = np.mean(alice_rew_buf) if alice_rew_buf else 0.0
        last_alice_mean_rew = mean_alice_rew
        writer.add_scalar("Loss/Alice/Value", loss_val, alice_updates)
        writer.add_scalar("Loss/Alice/Surrogate", loss_surr, alice_updates)
        writer.add_scalar("Reward/Alice", mean_alice_rew, alice_updates)

        print(
            f"  [Alice Update {alice_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_alice_rew:.4f}",
            flush=True,
        )
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_bob_obs):
        """Run a PPO + ABC update for Bob after his rollout is complete."""
        nonlocal bob_updates, best_bob_success_rate

        if bob_ppo.storage.step == 0:
            # No Bob transitions collected (e.g. Alice produced no valid goals).
            # Still increment bob_updates to prevent deadlock — paper Algorithm 1
            # always advances both agents per training step.
            print(
                f"  [Bob Update {bob_updates}] SKIPPED (no Bob transitions)",
                flush=True,
            )
            bob_rew_buf.clear()
            bob_success_buf.clear()
            bob_pos_err_buf.clear()
            bob_rot_err_buf.clear()
            bob_updates += 1
            return

        with torch.no_grad():
            _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(current_bob_obs, None)

        # --- Goal Encoder Monitoring ---
        if bob_ppo.actor_critic.use_goal_encoder:
            with torch.no_grad():
                sample_obs = current_bob_obs[:8]
                # Bob obs layout: robot(7) | obj1_state(14) | obj2_state(14) |
                #                 obj1_goal(6) | obj2_goal(6) | obj1_dist(2) | obj2_dist(2)
                # Current poses: first 6D (pos+euler) of each object_state
                s_t_batch = torch.cat(
                    [sample_obs[:, 7:13], sample_obs[:, 21:27]], dim=-1
                )
                # Goal poses: obj1_goal + obj2_goal
                s_star_batch = sample_obs[:, 35:47]
                g_sample = bob_ppo.actor_critic.goal_encoder(s_star_batch, s_t_batch)
                writer.add_scalar(
                    "GoalEncoder/embedding_norm",
                    g_sample.norm(dim=-1).mean().item(),
                    bob_updates,
                )
                writer.add_scalar(
                    "GoalEncoder/embedding_std",
                    g_sample.std(dim=-1).mean().item(),
                    bob_updates,
                )

        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        bob_ppo.current_learning_iteration = bob_updates
        loss_val, loss_surr, loss_abc, _ = bob_ppo.update(
            alice_mean_rew=last_alice_mean_rew
        )
        bob_ppo.storage.clear()

        mean_bob_rew = np.mean(bob_rew_buf) if bob_rew_buf else 0.0
        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        mean_pos_err = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot_err = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0

        writer.add_scalar("Loss/Bob/Value", loss_val, bob_updates)
        writer.add_scalar("Loss/Bob/Surrogate", loss_surr, bob_updates)
        writer.add_scalar("Loss/Bob/ABC", loss_abc, bob_updates)
        writer.add_scalar("Reward/Bob", mean_bob_rew, bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
        writer.add_scalar("Metrics/Bob/PosError", mean_pos_err, bob_updates)
        writer.add_scalar("Metrics/Bob/RotError", mean_rot_err, bob_updates)

        print(
            f"  [Bob Update {bob_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_bob_rew:.4f} | ABC: {loss_abc:.4f} | SR: {bob_success_rate:.4f}",
            flush=True,
        )

        if args.save_interval > 0 and (bob_updates + 1) % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir, f"model_{bob_updates+1}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{bob_updates+1}.pt"))
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(env.episode_manager.state_dict(), os.path.join(bob_ppo.log_dir, f"episode_manager_{bob_updates+1}.pt"))

        if bob_success_rate > best_bob_success_rate:
            best_bob_success_rate = bob_success_rate
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
            torch.save(env.episode_manager.state_dict(), os.path.join(bob_ppo.log_dir, "episode_manager_best.pt"))

        bob_rew_buf.clear()
        bob_success_buf.clear()
        bob_pos_err_buf.clear()
        bob_rot_err_buf.clear()

        bob_updates += 1

        # Force Python GC every 10 iterations to reclaim Python-side memory.
        # IsaacSim's PhysX heap is not affected, but this helps with Python objects.
        if bob_updates % 10 == 0:
            gc.collect()

    print("Initializing environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment initialized. Starting training loop...")

    target_alice_timesteps = env.episode_manager.alice_timesteps

    while bob_updates < args.max_iterations:

        # --- 0. SETUP: reset LSTM hidden states and snapshot policies ---
        if alice_hidden is not None:
            alice_hidden[0].zero_()
            alice_hidden[1].zero_()
        if bob_hidden is not None:
            bob_hidden[0].zero_()
            bob_hidden[1].zero_()

        if bob_updates > 0 and bob_updates % HIST_SAVE_INTERVAL == 0:
            alice_pool.add(alice_ppo.actor_critic)
            bob_pool.add(bob_ppo.actor_critic)
            print(
                f"  [HistPool] Saved snapshot at iter {bob_updates} "
                f"(alice pool={alice_pool.size}, bob pool={bob_pool.size})",
                flush=True,
            )

        # Alice entropy annealing: 1.0 → 0.01 over first 100 iterations.
        _ALICE_ENT_START = 1.0
        _ALICE_ENT_END = 0.01
        _ALICE_ENT_ANNEAL_ITERS = 100
        frac = min(1.0, bob_updates / _ALICE_ENT_ANNEAL_ITERS)
        alice_ppo.entropy_coef = _ALICE_ENT_START + frac * (
            _ALICE_ENT_END - _ALICE_ENT_START
        )
        writer.add_scalar("Alice/EntropyCoef", alice_ppo.entropy_coef, bob_updates)

        # --- 1. ALICE ROLLOUT PHASE ---
        alice_ppo.storage.clear()

        iter_sr_counts = [
            0,
            0,
        ]  # [attempted, succeeded] — moved here so Alice-loop Bob completions are counted

        obs_dict = env.env.observation_manager.compute()
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_alice_obs = obs[:, : env.alice_obs_dim]

        # Pre-allocate iteration buffers for ABC
        alice_traj_obs = []  # list of (num_envs, obs_dim)
        alice_traj_act = []  # list of (num_envs, act_dim)

        hist_alice = (
            alice_pool.sample_policy(alice_ppo.actor_critic, env.device)
            if alice_pool.size > 0
            else None
        )

        for t in range(env.episode_manager.alice_timesteps):
            # Capture where we are in alice phase
            is_alice = env.episode_manager.is_alice_phase()
            alice_indices = torch.where(is_alice)[0]

            if len(alice_indices) == 0:
                break

            # Split active envs: hist_ids use saved policy, curr_ids use current
            hist_ids, curr_ids = alice_pool.sample_env_subset(
                alice_indices, frac=HIST_FRAC
            )

            with torch.no_grad():
                # Current Alice (majority)
                h_in = (
                    (alice_hidden[0][curr_ids], alice_hidden[1][curr_ids])
                    if alice_hidden
                    else None
                )
                (
                    a_acts_curr,
                    a_logprob_curr,
                    a_val_curr,
                    a_mu_curr,
                    a_sigma_curr,
                    new_h,
                ) = alice_ppo.actor_critic.act_with_hidden(
                    current_alice_obs[curr_ids], None, h_in
                )
                if alice_hidden and new_h is not None:
                    alice_hidden[0][curr_ids] = new_h[0]
                    alice_hidden[1][curr_ids] = new_h[1]

                # Historical Alice (minority, no grad tracking needed)
                if len(hist_ids) > 0 and hist_alice is not None:
                    (
                        a_acts_hist,
                        a_logprob_hist,
                        a_val_hist,
                        a_mu_hist,
                        a_sigma_hist,
                        _,
                    ) = hist_alice.act_with_hidden(
                        current_alice_obs[hist_ids], None, None
                    )
                else:
                    # Fallback to current if no history yet
                    hist_ids = torch.tensor([], dtype=torch.long, device=env.device)
                    a_acts_hist = a_logprob_hist = a_val_hist = a_mu_hist = (
                        a_sigma_hist
                    ) = None

            # Policy action dim: 4 bin indices (MC) or 7 continuous (Gaussian)
            _a_pdim = num_cat_dims if use_mc else env.action_space.shape[0]

            # Merge curr + hist actions into full alice_indices tensors
            a_acts_active = torch.zeros(
                (len(alice_indices), _a_pdim), device=env.device
            )
            a_logprob_active = torch.zeros(len(alice_indices), device=env.device)
            a_val_active = torch.zeros(len(alice_indices), 1, device=env.device)
            a_mu_active = torch.zeros_like(a_acts_active)
            a_sigma_active = torch.zeros_like(a_acts_active)

            curr_local = torch.searchsorted(alice_indices, curr_ids)
            a_acts_active[curr_local] = a_acts_curr
            a_logprob_active[curr_local] = a_logprob_curr
            a_val_active[curr_local] = a_val_curr
            a_mu_active[curr_local] = a_mu_curr
            a_sigma_active[curr_local] = a_sigma_curr

            if len(hist_ids) > 0 and a_acts_hist is not None:
                hist_local = torch.searchsorted(alice_indices, hist_ids)
                a_acts_active[hist_local] = a_acts_hist
                a_logprob_active[hist_local] = a_logprob_hist
                a_val_active[hist_local] = a_val_hist
                a_mu_active[hist_local] = a_mu_hist
                a_sigma_active[hist_local] = a_sigma_hist

            alice_traj_obs.append(current_alice_obs.clone())

            # Policy actions for storage/ABC (bin indices or continuous)
            a_policy = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_policy[alice_indices] = a_acts_active
            alice_traj_act.append(a_policy.clone())  # ABC buffer gets bin indices

            # Full-env tensors for storage (non-active envs get zeros)
            a_lp_full = torch.zeros(env.num_envs, device=env.device)
            a_val_full = torch.zeros(env.num_envs, 1, device=env.device)
            a_mu_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_sigma_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_lp_full[alice_indices] = a_logprob_active
            a_val_full[alice_indices] = a_val_active
            a_mu_full[alice_indices] = a_mu_active
            a_sigma_full[alice_indices] = a_sigma_active

            # 7D env action for RMPFlow: convert bins → deltas, zero-pad rotation
            if use_mc:
                a_env_full = torch.zeros(
                    (env.num_envs, env.action_space.shape[0]), device=env.device
                )
                a_act_7d, new_ags = bins_to_env_action(
                    a_acts_active, alice_gripper_state[alice_indices]
                )
                a_env_full[alice_indices] = a_act_7d
                alice_gripper_state[alice_indices] = new_ags
            else:
                a_env_full = a_policy  # already 7D continuous

            obs_full, rewards, dones, truncated, extras = env.step(a_env_full)

            # Count Bob completions from envs still in Bob phase during this step.
            ep_info_a = extras.get("episode_manager", {})
            if ep_info_a:
                finished_bob_a = torch.where(ep_info_a["bob_done_this_step"])[0]
                if len(finished_bob_a) > 0:
                    iter_sr_counts[0] += len(finished_bob_a)
                    iter_sr_counts[1] += int(
                        ep_info_a["bob_success_this_step"][finished_bob_a].sum().item()
                    )

            # Reset LSTM hidden state and gripper state for envs that terminated
            if alice_hidden is not None:
                done_alice = alice_indices[dones[alice_indices]]
                if len(done_alice) > 0:
                    alice_hidden[0][done_alice] = 0.0
                    alice_hidden[1][done_alice] = 0.0
            alice_gripper_state[alice_indices[dones[alice_indices]]] = (
                1.0  # reset to open
            )

            # Storage masking
            a_masks = torch.zeros(env.num_envs, 1, device=env.device)
            a_masks[alice_indices[~dones[alice_indices]]] = 1.0

            # Store policy actions (bins) and real log_probs/values for PPO update
            next_alice_obs = obs_full[:, : env.alice_obs_dim]
            alice_ppo.storage.add_transitions(
                current_alice_obs,
                next_alice_obs,
                a_policy,
                rewards,
                dones,
                a_val_full,
                a_lp_full,
                a_mu_full,
                a_sigma_full,
                a_masks,
            )
            current_alice_obs = next_alice_obs

        # Goal states already extracted by wrapper during Alice→Bob transition
        # (in _handle_alice_completion, before objects were reset to S0).
        # DO NOT re-extract here: the objects have already been reset to S0 by the wrapper,
        # so any extraction now would overwrite the goal with the reset position.
        goal_states = env.episode_manager.goal_states

        # --- 2. BOB ROLLOUT PHASE ---
        bob_ppo.storage.clear()

        # Env already reset to S0 by wrapper during transition
        obs_dict = env.env.observation_manager.compute()
        obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_bob_obs = obs[:, env.alice_obs_dim :]

        hist_bob = (
            bob_pool.sample_policy(bob_ppo.actor_critic, env.device)
            if bob_pool.size > 0
            else None
        )

        for t in range(env.episode_manager.bob_timesteps):
            is_bob = env.episode_manager.is_bob_phase()
            bob_indices = torch.where(is_bob)[0]
            if len(bob_indices) == 0:
                break

            hist_bids, curr_bids = bob_pool.sample_env_subset(
                bob_indices, frac=HIST_FRAC
            )

            bob_obs_active = current_bob_obs[bob_indices]

            with torch.no_grad():
                # Current Bob (majority)
                h_in = (
                    (bob_hidden[0][curr_bids], bob_hidden[1][curr_bids])
                    if bob_hidden
                    else None
                )
                b_acts_curr, b_lp_curr, b_val_curr, b_mu_curr, b_sig_curr, new_bh = (
                    bob_ppo.actor_critic.act_with_hidden(
                        current_bob_obs[curr_bids], None, h_in
                    )
                )
                if bob_hidden and new_bh is not None:
                    bob_hidden[0][curr_bids] = new_bh[0]
                    bob_hidden[1][curr_bids] = new_bh[1]

                # Historical Bob (minority)
                if len(hist_bids) > 0 and hist_bob is not None:
                    b_acts_hist, b_lp_hist, b_val_hist, b_mu_hist, b_sig_hist, _ = (
                        hist_bob.act_with_hidden(current_bob_obs[hist_bids], None, None)
                    )
                else:
                    hist_bids = torch.tensor([], dtype=torch.long, device=env.device)
                    b_acts_hist = b_lp_hist = b_val_hist = b_mu_hist = b_sig_hist = None

            # Policy action dim: 4 bin indices (MC) or 7 continuous (Gaussian)
            _b_pdim = num_cat_dims if use_mc else env.action_space.shape[0]

            # Merge curr + hist into full bob_indices tensors
            b_acts_active = torch.zeros((len(bob_indices), _b_pdim), device=env.device)
            b_logprob_active = torch.zeros(len(bob_indices), device=env.device)
            b_val_active = torch.zeros(len(bob_indices), 1, device=env.device)
            b_mu_active = torch.zeros_like(b_acts_active)
            b_sigma_active = torch.zeros_like(b_acts_active)

            curr_bloc = torch.searchsorted(bob_indices, curr_bids)
            b_acts_active[curr_bloc] = b_acts_curr
            b_logprob_active[curr_bloc] = b_lp_curr
            b_val_active[curr_bloc] = b_val_curr
            b_mu_active[curr_bloc] = b_mu_curr
            b_sigma_active[curr_bloc] = b_sig_curr

            if len(hist_bids) > 0 and b_acts_hist is not None:
                hist_bloc = torch.searchsorted(bob_indices, hist_bids)
                b_acts_active[hist_bloc] = b_acts_hist
                b_logprob_active[hist_bloc] = b_lp_hist
                b_val_active[hist_bloc] = b_val_hist
                b_mu_active[hist_bloc] = b_mu_hist
                b_sigma_active[hist_bloc] = b_sig_hist

            # 7D env action for RMPFlow
            if use_mc:
                b_env_full = torch.zeros(
                    (env.num_envs, env.action_space.shape[0]), device=env.device
                )
                b_act_7d, new_bgs = bins_to_env_action(
                    b_acts_active, bob_gripper_state[bob_indices]
                )
                b_env_full[bob_indices] = b_act_7d
                bob_gripper_state[bob_indices] = new_bgs
            else:
                b_env_full = torch.zeros(
                    (env.num_envs, env.action_space.shape[0]), device=env.device
                )
                b_env_full[bob_indices] = b_acts_active

            obs_full, rewards, dones, truncated, extras = env.step(b_env_full)

            bob_done_this_step = extras.get(
                "bob_done_this_step",
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            )
            ended_for_bob = dones | bob_done_this_step
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0

            # Storage-ready tensors — policy actions (bins) stored, not env actions
            _obs = torch.zeros((env.num_envs, env.bob_obs_dim), device=env.device)
            _obs[bob_indices] = bob_obs_active
            _next_obs = torch.zeros((env.num_envs, env.bob_obs_dim), device=env.device)
            _next_obs[bob_indices] = obs_full[bob_indices, env.alice_obs_dim :]
            _acts = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            _acts[bob_indices] = b_acts_active
            _rew = torch.zeros(env.num_envs, device=env.device)
            _rew[bob_indices] = rewards[bob_indices]
            _val = torch.zeros(env.num_envs, 1, device=env.device)
            _val[bob_indices] = b_val_active
            _lp = torch.zeros(env.num_envs, 1, device=env.device)
            _lp[bob_indices] = b_logprob_active.unsqueeze(1)
            _mu = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            _mu[bob_indices] = b_mu_active
            _sigma = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            _sigma[bob_indices] = b_sigma_active

            # Reset LSTM hidden state and gripper state for terminated envs
            done_bob = bob_indices[dones[bob_indices]]
            if bob_hidden is not None and len(done_bob) > 0:
                bob_hidden[0][done_bob] = 0.0
                bob_hidden[1][done_bob] = 0.0
            if len(done_bob) > 0:
                bob_gripper_state[done_bob] = 1.0  # reset to open

            bob_ppo.storage.add_transitions(
                _obs,
                _next_obs,
                _acts,
                _rew,
                dones.clone(),
                _val,
                _lp,
                _mu,
                _sigma,
                b_masks,
            )
            current_bob_obs = obs_full[:, env.alice_obs_dim :]
            bob_rew_buf.extend(rewards.cpu().numpy().tolist())

            # Populate error buffers for TensorBoard (only for envs where Bob just finished)
            ep_info = extras.get("episode_manager", {})
            if ep_info:
                pos_err = ep_info["bob_pos_err"]
                rot_err = ep_info["bob_rot_err"]
                finished_bob = torch.where(ep_info["bob_done_this_step"])[0]
                if len(finished_bob) > 0:
                    bob_pos_err_buf.extend(pos_err[finished_bob].cpu().numpy().tolist())
                    bob_rot_err_buf.extend(rot_err[finished_bob].cpu().numpy().tolist())
                    iter_sr_counts[0] = iter_sr_counts[0] + len(finished_bob)
                    iter_sr_counts[1] = iter_sr_counts[1] + int(
                        ep_info["bob_success_this_step"][finished_bob].sum().item()
                    )

        # --- 3. ALICE BEHAVIORAL CLONING (ABC) BUFFER PUSH ---
        goal_valid = env.episode_manager.goal_valid
        bob_success = env.episode_manager.bob_success

        # Add demos only when Bob failed: success means the goal is already reachable.
        # Skip trajectories shorter than 50% of alice_timesteps — Alice crashed early.
        min_demo_steps = max(10, target_alice_timesteps // 2)
        if goal_valid.any():
            valid_ids = torch.where(goal_valid & ~bob_success)[0]
            skipped_short = 0
            for env_id in valid_ids:
                eid = env_id.item()
                traj_o = torch.stack(
                    [alice_traj_obs[step][eid] for step in range(len(alice_traj_obs))]
                )
                traj_a = torch.stack(
                    [alice_traj_act[step][eid] for step in range(len(alice_traj_act))]
                )

                if len(traj_o) < min_demo_steps:
                    skipped_short += 1
                    continue

                g = goal_states[eid].unsqueeze(0).expand(len(traj_o), -1)

                bc_obs = env.construct_bob_observation(traj_o, g)

                # Evaluate current Bob policy on Alice's demo for PPO ratio clipping.
                with torch.no_grad():
                    old_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(
                        bc_obs, None, traj_a
                    )

                if args.test_abc_verbose:
                    g_slice = goal_states[eid]
                    obs_dim = bc_obs.shape[-1]
                    # Bob obs layout (interleaved): Robot(7) + [Obj(14)+Goal(6)+Dist(2)] × 2 = 51D
                    # Object 1 goal starts at index 7+14=21, Object 2 goal at 7+36=43
                    goal_start = 7 + 14  # goal for first object
                    print(
                        f"[ABC Verbose] env={eid} | traj_len={len(traj_o)} | "
                        f"obs_shape={bc_obs.shape} | goal_shape={g_slice.shape} | "
                        f"goal_pos={g_slice[0:3].tolist()} | "
                        f"obs[0,goal_start:goal_start+3]={bc_obs[0, goal_start:goal_start+3].tolist()} "
                        f"(should match goal_pos)",
                        flush=True,
                    )

                bob_ppo.abc_buffer.add_trajectory(
                    bc_obs,
                    bc_obs,
                    traj_a,
                    torch.zeros(len(traj_o), device=env.device),
                    torch.zeros(len(traj_o), device=env.device).byte(),
                    torch.zeros(len(traj_o), device=env.device),
                    old_lp.view(-1, 1),
                    torch.zeros_like(traj_a),
                    torch.zeros_like(traj_a),
                    torch.zeros(len(traj_o), 1, device=env.device),
                    torch.zeros(len(traj_o), 1, device=env.device),
                )

        # --- 4. ALICE REWARD ASSIGNMENT & UPDATE ---
        alice_outcome_rewards = torch.where(
            ~bob_success & goal_valid,
            torch.tensor(ALICE_BOB_FAIL_REWARD, device=env.device, dtype=torch.float32),
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
        )

        if alice_ppo.storage.step > 0:
            last_idx = alice_ppo.storage.step - 1
            alice_ppo.storage.rewards[last_idx].copy_(alice_outcome_rewards.view(-1, 1))
            alice_ppo.storage.dones[last_idx].fill_(1.0)  # prevent GAE bleeding

            alice_rew_buf.extend(alice_outcome_rewards.cpu().numpy().tolist())

        current_sr = iter_sr_counts[1] / max(1, iter_sr_counts[0])
        bob_success_buf.append(current_sr)

        perform_alice_update()
        perform_bob_update(current_bob_obs)

        print(
            f"Iteration {bob_updates}: SR={current_sr:.2f} | ABC Buffer: {bob_ppo.abc_buffer.step if not bob_ppo.abc_buffer.full else 'FULL'}",
            flush=True,
        )

    alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_final.pt"))
    bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_final.pt"))
    torch.save(env.episode_manager.state_dict(), os.path.join(bob_ppo.log_dir, "episode_manager_final.pt"))
    print("  ✓ Saved final models")
    writer.close()

    print(f"\n{'='*80}\nTRAINING COMPLETE ({bob_updates} iterations)\n{'='*80}")
    print(f"To resume:\n  python train.py --exp_name {args.exp_name} \\")
    print(f"    --chkpt_alice runs/{args.exp_name}/alice/model_final.pt \\")
    print(f"    --chkpt_bob   runs/{args.exp_name}/bob/model_final.pt \\")
    print(f"    --resume_iteration {bob_updates}\n{'='*80}\n")


if __name__ == "__main__":
    main()
