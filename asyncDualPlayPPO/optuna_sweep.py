"""
Optuna Hyperparameter Sweep for Asymmetric Dual-Play PPO.

Architecture:
  - SimulationApp and Environment are created ONCE globally.
  - Each Optuna trial re-instantiates only the PPO agents (cheap).
  - EpisodeManager thresholds are patched dynamically per trial.
  - Pruning kills bad trials early via trial.report().
  - Max-steps bailout prevents infinite loops on dead trials.

Usage:
  python optuna_sweep.py --num_envs 64 --n_trials 50 --trial_iters 50 --headless
"""

import isaaclab.app
from isaaclab.app import AppLauncher

import os
import sys
import yaml
import argparse
import copy
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


def auto_nsteps(num_envs: int) -> int:
    """Auto-calculate nsteps so the rollout buffer fills in a reasonable time.
    
    Target: ~32K transitions per rollout (num_envs * nsteps).
    With 64 envs: nsteps=512, with 512 envs: nsteps=64, with 16 envs: nsteps=2048.
    Clamped to [32, 2048].
    """
    target_transitions = 32768  # 32K transitions per rollout
    nsteps = max(32, min(2048, target_transitions // num_envs))
    return nsteps


# ─── Global argument parsing & launcher ───────────────────────────────────────
parser = argparse.ArgumentParser(description="Optuna HPO Sweep for Async Dual Play PPO")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
parser.add_argument("--trial_iters", type=int, default=50, help="Bob updates per trial")
parser.add_argument("--max_steps_per_trial", type=int, default=500000,
                    help="Max env steps before force-terminating a trial (escape hatch)")
parser.add_argument("--study_name", type=str, default="asp_sweep")
parser.add_argument("--db_path", type=str, default=None, help="SQLite path for persistent study (e.g. sqlite:///sweep.db)")
parser.add_argument("--arm_config", type=str, default="rotated", choices=["default", "rotated"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ─── Imports that require SimulationApp ───────────────────────────────────────
import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner

from isaaclab.envs import ManagerBasedRLEnv
from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
from asyncDualPlayPPO.algorithms.rl.ppo import PPO
from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer
from asyncDualPlayPPO.tasks.utils.rewards import (
    ALICE_BOB_FAIL_REWARD,
    ALICE_BOB_SUCCESS_REWARD,
    ALICE_VALID_GOAL_BONUS,
    ALICE_INVALID_GOAL_PENALTY,
)

# ─── Auto-calculate nsteps ───────────────────────────────────────────────────
computed_nsteps = auto_nsteps(args.num_envs)
print(f"[Optuna] Auto nsteps: {computed_nsteps} (target ~32K transitions with {args.num_envs} envs)")

# ─── Create environment ONCE ─────────────────────────────────────────────────
ppo_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
base_ppo_cfg = load_cfg(ppo_cfg_path)
# Override nsteps with auto-calculated value
base_ppo_cfg["params"]["learn"]["nsteps"] = computed_nsteps

env_cfg = AsyncDualPlayEnvCfg()
env_cfg.scene.num_envs = args.num_envs

if args.arm_config == "rotated":
    env_cfg.scene.robot.init_state.joint_pos["left_shoulder_pan_joint"] = -1.57
    env_cfg.scene.robot.init_state.joint_pos["right_shoulder_pan_joint"] = 1.57

print("[Optuna] Creating environment (once)...")
base_env = ManagerBasedRLEnv(cfg=env_cfg)
env = AsyncDualPlayEnvWrapper(env=base_env, device=base_env.device, arm_config=args.arm_config)
print(f"[Optuna] Environment ready: {args.num_envs} envs, nsteps={computed_nsteps}")


# ─── Optuna Objective ────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """Run a short training budget and return Bob's success rate."""

    # ── 1. Suggest Hyperparameters ──────────────────────────────────────────
    # Category 1: Curriculum Gatekeepers (DECOUPLED)
    # min_goal_dist: how far Alice must move an object (low = easy goals for early training)
    min_goal_dist = trial.suggest_float("min_goal_dist", 0.001, 0.05)
    rot_threshold = trial.suggest_float("rot_threshold", 0.05, 0.50)

    # Category 2: Reward Shaping
    alpha_decay_steps = trial.suggest_int("alpha_decay_steps", 100, 1000)
    # bob_completion_radius: how close Bob must get to Alice's goal to succeed
    bob_completion_radius = trial.suggest_float("bob_completion_radius", 0.01, 0.05)

    # Category 3: PPO Exploration
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 5e-2, log=True)
    clip_param = trial.suggest_float("clip_param", 0.1, 0.3)

    # Category 4: HGI Dynamics
    demo_batch_ratio = trial.suggest_float("demo_batch_ratio", 0.1, 0.5)

    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*60}")
    print(f"  min_goal_dist:       {min_goal_dist:.4f} m (Alice validation)")
    print(f"  rot_threshold:       {rot_threshold:.4f} rad")
    print(f"  alpha_decay_steps:   {alpha_decay_steps}")
    print(f"  bob_completion:      {bob_completion_radius:.4f} m (Bob success)")
    print(f"  learning_rate:       {lr:.6f}")
    print(f"  entropy_coef:        {entropy_coef:.6f}")
    print(f"  clip_param:          {clip_param:.3f}")
    print(f"  demo_batch_ratio:    {demo_batch_ratio:.3f}")
    print(f"  nsteps(auto):        {computed_nsteps}")
    print(f"  max_steps_bailout:   {args.max_steps_per_trial}")
    print(f"{'='*60}\n")

    # ── 2. Dynamically Inject into Living Environment ───────────────────────
    # SINGLE SOURCE: EpisodeManager owns all thresholds
    env.episode_manager.pos_threshold = bob_completion_radius  # Bob's success radius
    env.episode_manager.rot_threshold = rot_threshold
    env.episode_manager.min_goal_dist = min_goal_dist  # Alice's validation threshold (decoupled!)

    # ── 3. Build PPO config for this trial ──────────────────────────────────
    ppo_cfg = copy.deepcopy(base_ppo_cfg)
    ppo_cfg["params"]["learn"]["optim_stepsize"] = lr
    ppo_cfg["params"]["learn"]["ent_coef"] = entropy_coef
    ppo_cfg["params"]["learn"]["cliprange"] = clip_param

    # ── 4. Re-instantiate PPO Agents (cheap) ────────────────────────────────
    alice_ppo = PPO(
        vec_env=env,
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/optuna/{args.study_name}/trial_{trial.number}/alice",
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
        log_dir=f"runs/optuna/{args.study_name}/trial_{trial.number}/bob",
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
    bob_ppo.demo_batch_ratio = demo_batch_ratio

    # ── 5. Pre-allocate Trajectory Buffers ──────────────────────────────────
    nsteps = computed_nsteps
    alice_obs_log = torch.zeros((args.num_envs, max_alice_steps, env.alice_obs_dim), device=env.device)
    alice_act_log = torch.zeros((args.num_envs, max_alice_steps, *env.action_space.shape), device=env.device)
    alice_step_counts = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    alice_validity_buffer = torch.zeros(args.num_envs, device=env.device)

    alice_updates = 0
    bob_updates = 0
    total_env_steps = 0  # Escape hatch counter
    max_alice_bob_ratio = 5

    rollout_length = nsteps * args.num_envs
    alice_rew_buf = deque(maxlen=rollout_length)
    bob_rew_buf = deque(maxlen=rollout_length)
    bob_success_buf = deque(maxlen=rollout_length)
    bob_pos_err_buf = deque(maxlen=rollout_length)
    bob_rot_err_buf = deque(maxlen=rollout_length)

    # ── 6. Reset Environment ────────────────────────────────────────────────
    with SuppressAllOutput():
        obs = env.reset()[0]

    # ── 7. Training Loop (Short Budget with Escape Hatch) ───────────────────
    def perform_alice_update():
        nonlocal alice_updates
        if alice_ppo.storage.step < nsteps:
            return
        if alice_updates >= (bob_updates + 1) * max_alice_bob_ratio:
            alice_ppo.storage.clear()
            alice_rew_buf.clear()
            return
        dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
        alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
        alice_ppo.update()
        alice_ppo.storage.clear()
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_obs):
        nonlocal bob_updates
        if bob_ppo.storage.step < nsteps:
            return current_obs
        with torch.no_grad():
            _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(current_obs, None)
        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        bob_ppo.update()
        bob_ppo.storage.clear()

        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        print(f"  [Trial {trial.number}] Bob Update {bob_updates}: SR={bob_success_rate:.4f} (steps={total_env_steps})")

        bob_rew_buf.clear()
        bob_success_buf.clear()
        bob_pos_err_buf.clear()
        bob_rot_err_buf.clear()

        bob_updates += 1

        # Report to Optuna for pruning
        trial.report(bob_success_rate, bob_updates)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return current_obs

    while bob_updates < args.trial_iters:
        # ── ESCAPE HATCH: bail out if trial is stuck ────────────────────
        total_env_steps += 1
        if total_env_steps > args.max_steps_per_trial:
            print(f"  [Trial {trial.number}] BAILOUT: {total_env_steps} steps, only {bob_updates} Bob updates. Pruning.")
            raise optuna.exceptions.TrialPruned()

        # Alpha annealing (uses trial-specific alpha_decay_steps)
        alpha = max(0.0, 1.0 - (bob_updates / alpha_decay_steps))
        env.bob_dense_reward_alpha = alpha

        is_alice = env.episode_manager.is_alice_phase()
        is_bob = env.episode_manager.is_bob_phase()

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

        # ── Hindsight Goal Injection ────────────────────────────────────
        if "alice_validity_bonus" in extras:
            curr_bonus = extras["alice_validity_bonus"]
            mask = curr_bonus != 0
            alice_validity_buffer[mask] = curr_bonus[mask]

            alice_success_mask = curr_bonus == 1.0
            if alice_success_mask.any():
                success_ids = torch.where(alice_success_mask)[0]
                goal_states = env.episode_manager.goal_states

                hgi_count = 0
                for idx in success_ids:
                    env_id = idx.item()
                    s_count = min(alice_step_counts[env_id].item(), max_alice_steps)
                    if s_count == 0:
                        continue

                    demo_obs = alice_obs_log[env_id, :s_count]
                    demo_acts = alice_act_log[env_id, :s_count]

                    _o = torch.zeros((s_count, env.bob_obs_dim), device=env.device)
                    _r = torch.zeros((s_count,), device=env.device)
                    _d = torch.zeros((s_count,), device=env.device)

                    goal_state = goal_states[env_id].unsqueeze(0).expand(s_count, -1)
                    b_obs = env.construct_bob_observation(demo_obs, goal_state)
                    _o[:] = b_obs

                    _r[-1] = 5.0
                    _d[-1] = 1.0

                    with torch.no_grad():
                        _lp, _, _v, _m, _s = bob_ppo.actor_critic.evaluate(_o, None, demo_acts)

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

                    none_states = torch.zeros((s_count, *env.bob_observation_space.shape), device=env.device)
                    bob_ppo.demo_buffer.add_trajectory(_o, none_states, demo_acts, _r, _d, _v, _lp, _m, _s, _ret, _adv)
                    hgi_count += s_count

        # ── Store Alice observations ────────────────────────────────────
        if len(alice_indices) > 0:
            steps = torch.clamp(alice_step_counts[alice_indices], max=max_alice_steps - 1)
            alice_obs_log[alice_indices, steps] = alice_obs
            alice_act_log[alice_indices, steps] = a_acts.clone()
            alice_step_counts[alice_indices] += 1

        # ── Store Bob transitions ───────────────────────────────────────
        if len(bob_indices) > 0:
            bob_done_this_step = extras.get("episode_manager", {}).get(
                "bob_done_this_step",
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            )
            ended_for_bob = dones[bob_indices] | bob_done_this_step[bob_indices]
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob]] = 1.0

            _obs = torch.zeros_like(obs);     _obs[bob_indices] = obs[bob_indices]
            _acts = torch.zeros_like(actions); _acts[bob_indices] = b_acts
            _rew = torch.zeros(env.num_envs, device=env.device)
            _rew[bob_indices] = rewards[bob_indices]
            _val = torch.zeros(env.num_envs, 1, device=env.device); _val[bob_indices] = b_val
            _lp = torch.zeros(env.num_envs, 1, device=env.device); _lp[bob_indices] = b_logprob.unsqueeze(1)
            _mu = torch.zeros_like(actions); _mu[bob_indices] = b_mu
            _sigma = torch.zeros_like(actions); _sigma[bob_indices] = b_sigma

            bob_ppo.storage.add_transitions(_obs, _obs, _acts, _rew, dones.clone(), _val, _lp, _mu, _sigma, b_masks)

            bob_step_rewards = rewards[bob_indices]
            bob_rew_buf.extend(bob_step_rewards.cpu().numpy().tolist())

        # ── Alice evaluation & PPO update ───────────────────────────────
        if "episode_manager" in extras:
            em_info = extras["episode_manager"]

            bob_done_mask = em_info["bob_done_this_step"]
            alice_failed_mask = extras.get("alice_failed_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

            alice_eval_mask = alice_failed_mask | bob_done_mask
            if alice_eval_mask.any():
                eval_ids = torch.where(alice_eval_mask)[0]
                bob_success_mask = em_info["bob_success_this_step"]

                max_count = 0
                env_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                env_rewards_buf = torch.zeros(env.num_envs, device=env.device)

                for idx in eval_ids:
                    env_id = idx.item()
                    count = min(alice_step_counts[env_id].item(), max_alice_steps)
                    if count == 0:
                        continue
                    env_counts[env_id] = count
                    max_count = max(max_count, count)

                    if alice_failed_mask[env_id]:
                        alice_reward = ALICE_INVALID_GOAL_PENALTY
                    else:
                        is_success = bob_success_mask[idx].item()
                        alice_reward = ALICE_BOB_SUCCESS_REWARD if is_success else ALICE_BOB_FAIL_REWARD

                    validity_bonus = alice_validity_buffer[env_id].item()
                    alice_validity_buffer[env_id] = 0.0
                    env_rewards_buf[env_id] = alice_reward + validity_bonus
                    alice_rew_buf.append(env_rewards_buf[env_id].item())

                if max_count > 0:
                    for t in range(max_count):
                        active_mask = (env_counts > t).float().unsqueeze(1)
                        _o = torch.zeros((env.num_envs, env.alice_obs_dim), device=env.device)
                        _a = torch.zeros((env.num_envs, *env.action_space.shape), device=env.device)
                        _r = torch.zeros((env.num_envs,), device=env.device)
                        _d = torch.zeros((env.num_envs,), device=env.device)

                        active_ids = torch.where(env_counts > t)[0]
                        if len(active_ids) > 0:
                            _o[active_ids] = alice_obs_log[active_ids, t]
                            _a[active_ids] = alice_act_log[active_ids, t]

                            is_last_step = (env_counts == (t + 1))
                            last_step_ids = torch.where(is_last_step)[0]
                            if len(last_step_ids) > 0:
                                _r[last_step_ids] = env_rewards_buf[last_step_ids]
                                _d[last_step_ids] = 1.0

                            with torch.no_grad():
                                _lp, _, _v, _m, _s = alice_ppo.actor_critic.evaluate(_o, None, _a)

                            _v = _v.view(-1, 1) * active_mask
                            _lp = _lp.view(-1, 1) * active_mask

                            alice_ppo.storage.add_transitions(_o, _o, _a, _r, _d, _v, _lp, _m, _s, active_mask)

                    perform_alice_update()

                alice_step_counts[eval_ids] = 0

            # Bob success metric logging
            if bob_done_mask.any():
                bob_success_mask_log = em_info["bob_success_this_step"]
                bob_success_buf.extend(bob_success_mask_log[bob_success_mask_log].cpu().numpy().astype(float).tolist())
                bob_success_buf.extend([0.0] * (bob_done_mask & ~bob_success_mask_log).sum().item())
                bob_pos_err_buf.extend(em_info["bob_pos_err"][bob_done_mask].cpu().numpy().tolist())
                bob_rot_err_buf.extend(em_info["bob_rot_err"][bob_done_mask].cpu().numpy().tolist())

        if dones.any():
            alice_step_counts[torch.where(dones)[0]] = 0

        obs = next_obs
        obs = perform_bob_update(obs)

    # ── 8. Return Final Metric ──────────────────────────────────────────────
    final_sr = np.mean(bob_success_buf) if bob_success_buf else 0.0
    print(f"\n[Trial {trial.number}] FINAL Bob Success Rate: {final_sr:.4f} ({total_env_steps} env steps)\n")
    return final_sr


# ─── Main: Run the Sweep ─────────────────────────────────────────────────────
def main():
    storage = args.db_path if args.db_path else None
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    print(f"\n{'='*60}")
    print(f"OPTUNA SWEEP: {args.study_name}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Budget per trial: {args.trial_iters} Bob updates")
    print(f"  Envs: {args.num_envs}")
    print(f"  nsteps (auto): {computed_nsteps}")
    print(f"  Max steps/trial: {args.max_steps_per_trial}")
    print(f"{'='*60}\n")

    study.optimize(objective, n_trials=args.n_trials)

    # Print results
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best SR:    {study.best_trial.value:.4f}")
    print(f"Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
