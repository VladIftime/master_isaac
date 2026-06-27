"""
gym-pusht PBRS Model C — Asymmetric Self-Play (Alice + Bob).

Native gym-pusht counterpart of train_c_pbrs_asp.py.  Reuses the project's
custom PPO (Alice) + PPOABC/GoalEncoder (Bob) + EpisodeManager + validate_goal
+ PBRS unchanged; only the environment differs (GymPushASPEnv, synchronous).
ASP orchestration (phases, delayed Alice reward, ABC, historical pool) is
identical to train_c.  Runs in .master_venv (no Isaac / cuRobo).

Run:
  .master_venv/bin/python3 -m asyncDualPlayPPO.train_c_gym_pbrs_asp \
      --num_envs 8 --max_iterations 3 --device cpu --no_abc --no_hist_pool \
      --exp_name gym_pbrs_c_smoke
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
import copy
import math
import signal
import argparse
from collections import deque

import numpy as np
import torch
torch.set_num_threads(1)
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asyncDualPlayPPO.tasks.utils.gym_push_asp_env import GymPushASPEnv, _OBS_ROBOT_DIM
from asyncDualPlayPPO.tasks.utils.action_push_relative import decode_push_action_relative
from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer
from asyncDualPlayPPO.utils.historical_pool import HistoricalPolicyPool
from asyncDualPlayPPO.tasks.utils.reward_pbrs import (
    potential_pos, potential_rot, compute_pbrs_reward,
    PBRS_W_POS, PBRS_W_ROT,
)

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="gym-pusht PBRS Model C (ASP)")
    p.add_argument("--exp_name", type=str, default="gym_pbrs_c")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--max_iterations", type=int, default=3000)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alice_pushes", type=int, default=5)
    p.add_argument("--bob_pushes", type=int, default=10)
    p.add_argument("--max_goals_per_episode", type=int, default=2)
    p.add_argument("--no_abc", action="store_true")
    p.add_argument("--no_hist_pool", action="store_true")
    p.add_argument("--rel_obs", action="store_true")
    p.add_argument("--debug_rewards", action="store_true")
    p.add_argument("--chkpt_alice", type=str, default=None)
    p.add_argument("--chkpt_bob", type=str, default=None)
    p.add_argument("--resume_iteration", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg = load_cfg(cfg_path)

    alice_pushes, bob_pushes = args.alice_pushes, args.bob_pushes
    num_cat_dims, num_bins = 4, 21
    rollout_len = alice_pushes + bob_pushes

    env = GymPushASPEnv(num_envs=args.num_envs, alice_pushes=alice_pushes,
                        bob_pushes=bob_pushes, max_goals_per_episode=args.max_goals_per_episode,
                        num_objects=1, rel_obs=args.rel_obs, device=device, seed=args.seed)
    print(f"[gym-C] num_envs={env.num_envs} device={device} "
          f"alice={alice_pushes} bob={bob_pushes} ABC={'OFF' if args.no_abc else 'ON'} "
          f"hist={'OFF' if args.no_hist_pool else 'ON'}")

    import gymnasium as gym_mc
    _mc = gym_mc.spaces.Box(0.0, float(num_bins - 1), (num_cat_dims,), np.float32)

    # ── Alice (PPO, no GoalEncoder) ───────────────────────────────────────────
    alice_cfg = copy.deepcopy(ppo_cfg["params"])
    alice_cfg["policy"].update(dict(use_pi_encoder=True, use_multicategorical=True,
                                    use_lstm=True, use_goal_encoder=False,
                                    num_cat_dims=num_cat_dims, num_bins=num_bins,
                                    robot_state_dim=6))
    alice_ppo = PPO(vec_env=env, cfg_train=alice_cfg, device=device, sampler="sequential",
                    log_dir=f"runs/{args.exp_name}/alice", asymmetric=False)
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space
    alice_ppo.action_space = _mc
    alice_ppo.desired_kl = None
    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape, alice_ppo.state_space.shape, _mc.shape,
        alice_ppo.init_noise_std, alice_ppo.model_cfg, asymmetric=False).to(device)
    a_size = max(alice_ppo.num_transitions_per_env + alice_pushes + 5, rollout_len + 5)
    alice_ppo.storage = alice_ppo.storage.__class__(
        env.num_envs, a_size, alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape, _mc.shape, device, "sequential")
    alice_ppo.optimizer = torch.optim.Adam(alice_ppo.actor_critic.parameters(),
                                           lr=alice_ppo.learning_rate)

    # ── Bob (PPOABC + GoalEncoder) ────────────────────────────────────────────
    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"].update(dict(use_pi_encoder=True, use_multicategorical=True,
                                  use_lstm=True, use_goal_encoder=True,
                                  num_cat_dims=num_cat_dims, num_bins=num_bins,
                                  num_objects=1, robot_state_dim=6, goal_embed_dim=8))
    bob_ppo = PPOABC(vec_env=env, cfg_train=bob_cfg, device=device, sampler="sequential",
                     log_dir=f"runs/{args.exp_name}/bob", asymmetric=False)
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space
    bob_ppo.action_space = _mc
    bob_ppo.desired_kl = None
    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape, bob_ppo.state_space.shape, _mc.shape,
        bob_ppo.init_noise_std, bob_ppo.model_cfg, asymmetric=False).to(device)
    if hasattr(bob_ppo.actor_critic, "_goal_proj") and bob_ppo.actor_critic._goal_proj is not None:
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.1)
    b_size = max(bob_ppo.num_transitions_per_env + bob_pushes + 5, rollout_len + 5)
    bob_ppo.storage = bob_ppo.storage.__class__(
        env.num_envs, b_size, bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape, _mc.shape, device, "sequential")
    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
        capacity=50000, obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape, actions_shape=(num_cat_dims,),
        device=device, traj_maxlen=ppo_cfg["params"]["learn"].get("abc_traj_maxlen", 500))

    if args.chkpt_alice and os.path.isfile(args.chkpt_alice):
        alice_ppo.load(args.chkpt_alice)
    if args.chkpt_bob and os.path.isfile(args.chkpt_bob):
        bob_ppo.load(args.chkpt_bob)

    lsz = alice_ppo.actor_critic.lstm_hidden_size
    alice_hidden = [torch.zeros(env.num_envs, lsz, device=device),
                    torch.zeros(env.num_envs, lsz, device=device)]
    bob_hidden = [torch.zeros(env.num_envs, lsz, device=device),
                  torch.zeros(env.num_envs, lsz, device=device)]

    alice_pool = HistoricalPolicyPool(max_size=5)
    bob_pool = HistoricalPolicyPool(max_size=5)
    HIST_SAVE_INTERVAL, HIST_FRAC = 100, 0.2

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")
    os.makedirs(alice_ppo.log_dir, exist_ok=True)
    os.makedirs(bob_ppo.log_dir, exist_ok=True)
    alice_rew_buf = deque(maxlen=args.num_envs * alice_pushes)
    bob_rew_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_pos_err_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_rot_err_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_success_buf = deque(maxlen=200)
    alice_updates = bob_updates = args.resume_iteration
    best_bob_sr = -1.0
    last_alice_mean_rew = 0.0

    _a_pdim = _b_pdim = num_cat_dims
    _bob_gave_rot_bonus = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    def perform_alice_update():
        nonlocal alice_updates, last_alice_mean_rew
        if alice_ppo.storage.step == 0:
            return
        dummy = torch.zeros(env.num_envs, 1, device=device)
        alice_ppo.storage.compute_returns(dummy, alice_ppo.gamma, alice_ppo.lam)
        lv, ls = alice_ppo.update()
        alice_ppo.storage.clear()
        last_alice_mean_rew = np.mean(alice_rew_buf) if alice_rew_buf else 0.0
        writer.add_scalar("Reward/Alice", last_alice_mean_rew, alice_updates)
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(cur_bob_obs):
        nonlocal bob_updates, best_bob_sr
        if bob_ppo.storage.step * env.num_envs < bob_ppo.num_mini_batches:
            bob_updates += 1
            return
        with torch.no_grad():
            _, _, last_v, _, _ = bob_ppo.actor_critic.act(cur_bob_obs, None)
        bob_ppo.storage.compute_returns(last_v, bob_ppo.gamma, bob_ppo.lam)
        bob_ppo.current_learning_iteration = bob_updates
        lv, ls, labc, _ = bob_ppo.update(alice_mean_rew=last_alice_mean_rew)
        bob_ppo.storage.clear()
        mr = np.mean(bob_rew_buf) if bob_rew_buf else 0.0
        sr = np.mean(bob_success_buf) if bob_success_buf else 0.0
        writer.add_scalar("Reward/Bob", mr, bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", sr, bob_updates)
        if sr > best_bob_sr:
            best_bob_sr = sr
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
        bob_rew_buf.clear()
        bob_updates += 1

    _stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: _stop.update(flag=True))

    obs = env.reset()
    _stagger = torch.randint(0, max(1, alice_pushes), (env.num_envs,), device=device, dtype=torch.int32)
    env.episode_manager.phase_step.copy_(_stagger)

    while bob_updates < args.max_iterations:
        alice_hidden[0].zero_(); alice_hidden[1].zero_()
        bob_hidden[0].zero_(); bob_hidden[1].zero_()

        if bob_updates > 0 and bob_updates % HIST_SAVE_INTERVAL == 0 and not args.no_hist_pool:
            alice_pool.add(alice_ppo.actor_critic)
            bob_pool.add(bob_ppo.actor_critic)

        alice_ppo.storage.clear(); bob_ppo.storage.clear()
        env.reset_iter_stats()
        iter_sr = [0, 0]

        full_push_obs = env._get_push_obs()
        current_alice_obs = env._get_alice_obs(full_push_obs)
        current_bob_obs = env._get_bob_obs(full_push_obs)
        env.capture_pre_push(full_push_obs)
        prev_phi_pos = potential_pos(full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3],
                                     full_push_obs[:, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 17])
        prev_phi_rot = potential_rot(full_push_obs[:, _OBS_ROBOT_DIM + 5],
                                     full_push_obs[:, _OBS_ROBOT_DIM + 14 + 5])

        alice_traj_obs = torch.zeros((env.num_envs, alice_pushes, env.alice_obs_dim), device=device)
        alice_traj_act = torch.zeros((env.num_envs, alice_pushes, _a_pdim), device=device)
        alice_traj_len = torch.zeros(env.num_envs, dtype=torch.long, device=device)

        hist_alice = (alice_pool.sample_policy(alice_ppo.actor_critic, device)
                      if alice_pool.size > 0 and not args.no_hist_pool else None)
        hist_bob = (bob_pool.sample_policy(bob_ppo.actor_critic, device)
                    if bob_pool.size > 0 and not args.no_hist_pool else None)

        for t in range(rollout_len):
            is_alice = env.episode_manager.is_alice_phase()
            is_bob = env.episode_manager.is_bob_phase()
            alice_indices = torch.where(is_alice)[0]
            bob_indices = torch.where(is_bob)[0]

            alice_loc = torch.empty(env.num_envs, dtype=torch.long, device=device)
            if len(alice_indices) > 0:
                alice_loc[alice_indices] = torch.arange(len(alice_indices), device=device)
            bob_loc = torch.empty(env.num_envs, dtype=torch.long, device=device)
            if len(bob_indices) > 0:
                bob_loc[bob_indices] = torch.arange(len(bob_indices), device=device)

            _alice_hidden_pre = (alice_hidden[0].clone(), alice_hidden[1].clone())
            a_acts = torch.zeros((len(alice_indices), _a_pdim), device=device)
            a_lp = torch.zeros(len(alice_indices), device=device)
            a_val = torch.zeros(len(alice_indices), 1, device=device)
            a_mu = torch.zeros_like(a_acts); a_sig = torch.zeros_like(a_acts)
            if len(alice_indices) > 0:
                hist_ids, curr_ids = (alice_pool.sample_env_subset(alice_indices, frac=HIST_FRAC)
                                      if not args.no_hist_pool and hist_alice is not None
                                      else (torch.tensor([], dtype=torch.long, device=device), alice_indices))
                with torch.no_grad():
                    h_in = (alice_hidden[0][curr_ids], alice_hidden[1][curr_ids])
                    (ac, lp, vl, mu, sg, nh) = alice_ppo.actor_critic.act_with_hidden(
                        current_alice_obs[curr_ids], None, h_in)
                    if nh is not None:
                        alice_hidden[0][curr_ids] = nh[0]; alice_hidden[1][curr_ids] = nh[1]
                    if len(hist_ids) > 0 and hist_alice is not None:
                        (ach, lph, vlh, muh, sgh, _) = hist_alice.act_with_hidden(
                            current_alice_obs[hist_ids], None, None)
                    else:
                        ach = None
                cl = alice_loc[curr_ids]
                a_acts[cl] = ac; a_lp[cl] = lp; a_val[cl] = vl; a_mu[cl] = mu; a_sig[cl] = sg
                if ach is not None:
                    hl = alice_loc[hist_ids]
                    a_acts[hl] = ach; a_lp[hl] = lph; a_val[hl] = vlh; a_mu[hl] = muh; a_sig[hl] = sgh
                    _alice_hidden_pre[0][hist_ids] = 0.0; _alice_hidden_pre[1][hist_ids] = 0.0
            _alice_hidden_pre[0][~is_alice] = 0.0; _alice_hidden_pre[1][~is_alice] = 0.0

            _bob_hidden_pre = (bob_hidden[0].clone(), bob_hidden[1].clone())
            b_acts = torch.zeros((len(bob_indices), _b_pdim), device=device)
            b_lp = torch.zeros(len(bob_indices), device=device)
            b_val = torch.zeros(len(bob_indices), 1, device=device)
            b_mu = torch.zeros_like(b_acts); b_sig = torch.zeros_like(b_acts)
            if len(bob_indices) > 0:
                hist_bids, curr_bids = (bob_pool.sample_env_subset(bob_indices, frac=HIST_FRAC)
                                        if not args.no_hist_pool and hist_bob is not None
                                        else (torch.tensor([], dtype=torch.long, device=device), bob_indices))
                with torch.no_grad():
                    h_in = (bob_hidden[0][curr_bids], bob_hidden[1][curr_bids])
                    (bc, blp, bvl, bmu, bsg, nbh) = bob_ppo.actor_critic.act_with_hidden(
                        current_bob_obs[curr_bids], None, h_in)
                    if nbh is not None:
                        bob_hidden[0][curr_bids] = nbh[0]; bob_hidden[1][curr_bids] = nbh[1]
                    if len(hist_bids) > 0 and hist_bob is not None:
                        (bch, blph, bvlh, bmuh, bsgh, _) = hist_bob.act_with_hidden(
                            current_bob_obs[hist_bids], None, None)
                    else:
                        bch = None
                cl = bob_loc[curr_bids]
                b_acts[cl] = bc; b_lp[cl] = blp; b_val[cl] = bvl; b_mu[cl] = bmu; b_sig[cl] = bsg
                if bch is not None:
                    hl = bob_loc[hist_bids]
                    b_acts[hl] = bch; b_lp[hl] = blph; b_val[hl] = bvlh; b_mu[hl] = bmuh; b_sig[hl] = bsgh
                    _bob_hidden_pre[0][hist_bids] = 0.0; _bob_hidden_pre[1][hist_bids] = 0.0
            _bob_hidden_pre[0][~is_bob] = 0.0; _bob_hidden_pre[1][~is_bob] = 0.0

            a_policy = torch.zeros((env.num_envs, _a_pdim), device=device)
            a_policy[alice_indices] = a_acts
            b_policy = torch.zeros((env.num_envs, _b_pdim), device=device)
            b_policy[bob_indices] = b_acts

            # Store Alice traj for ABC
            alice_step_raw = env.episode_manager.phase_step[alice_indices] - 1
            valid_t = alice_step_raw < alice_pushes
            active_alice = alice_indices[valid_t]; active_steps = alice_step_raw[valid_t]
            phase_start = alice_step_raw == 0
            if phase_start.any():
                alice_traj_len[alice_indices[phase_start]] = 0
            if len(active_alice) > 0:
                alice_traj_obs[active_alice, active_steps] = current_alice_obs[active_alice].clone()
                alice_traj_act[active_alice, active_steps] = a_acts[valid_t].clone()
                alice_traj_len[active_alice] = torch.max(
                    alice_traj_len[active_alice], (active_steps + 1).to(alice_traj_len.dtype))

            # Decode pushes (object-relative)
            obj_xy = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 2]
            obj_yaw = full_push_obs[:, _OBS_ROBOT_DIM + 5]
            Xs = torch.zeros(env.num_envs, device=device); Ys = torch.zeros(env.num_envs, device=device)
            length = torch.zeros(env.num_envs, device=device); theta = torch.zeros(env.num_envs, device=device)
            for idxs, acts in ((alice_indices, a_acts), (bob_indices, b_acts)):
                if len(idxs) > 0:
                    xs, ys, ln, th = decode_push_action_relative(
                        acts, obj_xy[idxs], obj_yaw[idxs], num_bins=num_bins,
                        min_r=0.04, max_r=0.08, max_len=0.20)
                    Xs[idxs] = xs; Ys[idxs] = ys; length[idxs] = ln; theta[idxs] = th
            _m = 0.02
            Xs.clamp_(_WS_X[0] + _m, _WS_X[1] - _m); Ys.clamp_(_WS_Y[0] + _m, _WS_Y[1] - _m)
            _Xf = (Xs + length * torch.cos(theta)).clamp(_WS_X[0] + _m, _WS_X[1] - _m)
            _Yf = (Ys + length * torch.sin(theta)).clamp(_WS_Y[0] + _m, _WS_Y[1] - _m)
            length = torch.sqrt((_Xf - Xs) ** 2 + (_Yf - Ys) ** 2)
            theta = torch.atan2(_Yf - Ys, _Xf - Xs)

            env.execute_push(Xs, Ys, length, theta)

            full_push_obs = env._get_push_obs()
            current_alice_obs = env._get_alice_obs(full_push_obs)
            current_bob_obs = env._get_bob_obs(full_push_obs)

            obj_pos_all = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            _oob_ws = ((obj_pos_all[:, 0] < _WS_X[0]) | (obj_pos_all[:, 0] > _WS_X[1]) |
                       (obj_pos_all[:, 1] < _WS_Y[0]) | (obj_pos_all[:, 1] > _WS_Y[1]))

            # Bob PBRS reward
            bob_rewards = torch.zeros(env.num_envs, device=device)
            pbrs_result = None
            _catastrophe = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            if len(bob_indices) > 0:
                op = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
                oe = full_push_obs[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
                gp = full_push_obs[:, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 17]
                ge = full_push_obs[:, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]
                pbrs_result = compute_pbrs_reward(op, oe, gp, ge, prev_phi_pos, prev_phi_rot,
                                                  env._bob_gave_completion, _bob_gave_rot_bonus)
                bob_rewards = pbrs_result["reward"]
                env._bob_gave_completion = pbrs_result["gave_completion"]
                _bob_gave_rot_bonus = pbrs_result["gave_rot_bonus"]
                _catastrophe = (pbrs_result["tipped"] | (op[:, 2] > 0.10) | _oob_ws) & is_bob
                bob_rewards[_catastrophe] = -10.0

            bob_achieved = (pbrs_result["at_goal"] if pbrs_result is not None
                            else torch.zeros(env.num_envs, dtype=torch.bool, device=device))

            alice_rewards_now = torch.zeros(env.num_envs, device=device)
            alice_done_now = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            bob_done_now = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            bob_success_now = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            bob_pos_err_now = torch.zeros(env.num_envs, device=device)
            bob_rot_err_now = torch.zeros(env.num_envs, device=device)
            bob_progress_rew = torch.zeros(env.num_envs, device=device)

            phase_info = env.episode_manager.step()
            alice_done_mask = phase_info["alice_done"]
            bob_done_mask = phase_info["bob_done"]

            alice_done_ids = torch.where(alice_done_mask)[0]
            if len(alice_done_ids) > 0:
                valid_ids, invalid_ids = env.handle_alice_phase_end(alice_done_ids, full_push_obs)
                alice_rewards_now[alice_done_ids] = env.delayed_alice_reward[alice_done_ids]
                env.delayed_alice_reward[alice_done_ids] = 0.0
                alice_done_now[invalid_ids] = True

            _goal_valid_pre = env.episode_manager.goal_valid.clone()

            bob_done_ids = torch.where(bob_done_mask)[0]
            if len(bob_done_ids) > 0:
                bs, bp, br, bd, _ = env.handle_bob_phase_end(bob_done_ids, full_push_obs)
                bob_success_now[bob_done_ids] = bs[bob_done_ids]
                bob_pos_err_now[bob_done_ids] = bp[bob_done_ids]
                bob_rot_err_now[bob_done_ids] = br[bob_done_ids]
                bob_done_now[bob_done_ids] = bd[bob_done_ids]

            if bob_achieved.any():
                cids = torch.where(bob_achieved)[0]
                bob_progress_rew += env.handle_bob_early_success(cids, full_push_obs)
                bob_done_now[cids] = True
                bob_success_now[cids] = True

            _cata_early = _catastrophe & is_bob & ~bob_done_mask & ~bob_achieved
            if _cata_early.any():
                env.handle_bob_phase_end(torch.where(_cata_early)[0], full_push_obs)
                bob_done_now[_cata_early] = True

            _alice_oob = _oob_ws & is_alice & ~alice_done_mask
            if _alice_oob.any():
                aoob = torch.where(_alice_oob)[0]
                env.episode_manager.reset_episode(aoob, reason="Object OOB")
                sp = env._rand_reset_objs(aoob)
                env.episode_manager.initial_states[aoob] = env._initial_states_from_spawn(sp, len(aoob))
                alice_rewards_now[aoob] = -3.0
                alice_done_now[aoob] = True

            full_push_obs = env._get_push_obs()
            next_alice_obs = env._get_alice_obs(full_push_obs)
            next_bob_obs = env._get_bob_obs(full_push_obs)

            rewards = torch.zeros(env.num_envs, device=device)
            if len(bob_indices) > 0:
                rewards[bob_indices] = bob_rewards[bob_indices]
            rewards += bob_progress_rew
            if len(bob_indices) > 0:
                bob_rew_buf.extend(rewards[bob_indices].cpu().tolist())

            dones_all = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

            if len(alice_indices) > 0:
                da = alice_indices[alice_done_mask[alice_indices]]
                if len(da) > 0:
                    alice_hidden[0][da] = 0.0; alice_hidden[1][da] = 0.0
            if len(bob_indices) > 0:
                db = bob_indices[bob_done_mask[bob_indices]]
                if len(db) > 0:
                    bob_hidden[0][db] = 0.0; bob_hidden[1][db] = 0.0
                    _bob_gave_rot_bonus[db] = False
            early = bob_done_now & ~bob_done_mask
            if early.any():
                eids = torch.where(early)[0]
                bob_hidden[0][eids] = 0.0; bob_hidden[1][eids] = 0.0
                _bob_gave_rot_bonus[eids] = False

            env.push_count += 1

            # Alice storage
            a_lp_full = torch.zeros(env.num_envs, device=device)
            a_val_full = torch.zeros(env.num_envs, 1, device=device)
            a_mu_full = torch.zeros((env.num_envs, _a_pdim), device=device)
            a_sig_full = torch.zeros((env.num_envs, _a_pdim), device=device)
            a_lp_full[alice_indices] = a_lp; a_val_full[alice_indices] = a_val
            a_mu_full[alice_indices] = a_mu; a_sig_full[alice_indices] = a_sig
            a_masks = torch.zeros(env.num_envs, 1, device=device)
            a_masks[alice_indices] = 1.0
            alice_ppo.storage.add_transitions(
                current_alice_obs, next_alice_obs, a_policy, rewards, dones_all,
                a_val_full, a_lp_full, a_mu_full, a_sig_full, a_masks,
                hidden_state=_alice_hidden_pre)
            current_alice_obs = next_alice_obs

            # Bob storage
            b_lp_full = torch.zeros(env.num_envs, 1, device=device)
            b_val_full = torch.zeros(env.num_envs, 1, device=device)
            b_mu_full = torch.zeros((env.num_envs, _b_pdim), device=device)
            b_sig_full = torch.zeros((env.num_envs, _b_pdim), device=device)
            b_lp_full[bob_indices] = b_lp.unsqueeze(1); b_val_full[bob_indices] = b_val
            b_mu_full[bob_indices] = b_mu; b_sig_full[bob_indices] = b_sig
            ended_for_bob = dones_all | bob_done_now
            b_masks = torch.zeros(env.num_envs, 1, device=device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0
            bob_ppo.storage.add_transitions(
                current_bob_obs, next_bob_obs, b_policy, rewards, dones_all,
                b_val_full, b_lp_full, b_mu_full, b_sig_full, b_masks,
                hidden_state=_bob_hidden_pre)
            current_bob_obs = next_bob_obs

            bde = torch.where(bob_done_now)[0]
            if len(bde) > 0:
                bob_pos_err_buf.extend(bob_pos_err_now[bde].cpu().tolist())
                bob_rot_err_buf.extend(bob_rot_err_now[bde].cpu().tolist())
                iter_sr[0] += len(bde); iter_sr[1] += int(bob_success_now[bde].sum().item())

            # Alice delayed reward -> last valid transition
            a_rew_envs = torch.where((alice_rewards_now != 0) | alice_done_now)[0]
            if len(a_rew_envs) > 0:
                filled = alice_ppo.storage.step
                if filled > 0:
                    masks = alice_ppo.storage.masks[:filled, :, 0]
                    row_idx = torch.arange(filled, device=device).unsqueeze(1).expand_as(masks)
                    last_valid = torch.where(masks.bool(), row_idx,
                                             torch.tensor(-1, device=device)).max(dim=0).values
                    has_valid = last_valid >= 0
                    rewarded = a_rew_envs[has_valid[a_rew_envs]]
                    if len(rewarded) > 0:
                        rows = last_valid[rewarded]
                        alice_ppo.storage.rewards[rows, rewarded, 0] += alice_rewards_now[rewarded]
                        alice_rew_buf.extend(alice_rewards_now[rewarded].cpu().tolist())

            # ABC population
            if not args.no_abc:
                try:
                    jf = bob_done_now & (~bob_success_now) & _goal_valid_pre
                    min_demo = max(2, alice_pushes // 2)
                    trajs = []
                    for env_id in torch.where(jf)[0]:
                        eid = env_id.item(); tl = alice_traj_len[eid].item()
                        if tl < min_demo:
                            continue
                        g = env.episode_manager.goal_states
                        g = (g[eid].unsqueeze(0).expand(tl, -1) if g is not None
                             else torch.zeros(tl, 6, device=device))
                        bc_obs = env.construct_bob_observation(alice_traj_obs[eid, :tl], g)
                        trajs.append((bc_obs, alice_traj_act[eid, :tl].long()))
                    if trajs:
                        allo = torch.cat([x[0] for x in trajs], 0)
                        alla = torch.cat([x[1] for x in trajs], 0)
                        with torch.no_grad():
                            old_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(allo, None, alla)
                        off = 0
                        for bo, ba in trajs:
                            n = bo.shape[0]
                            bob_ppo.abc_buffer.add_trajectory(bo, ba, old_lp[off:off + n])
                            off += n
                except Exception as e:
                    if args.debug_rewards:
                        print(f"  [ABC] {e}", flush=True)

            env.capture_pre_push(full_push_obs)
            prev_phi_pos = potential_pos(full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3],
                                         full_push_obs[:, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 17])
            prev_phi_rot = potential_rot(full_push_obs[:, _OBS_ROBOT_DIM + 5],
                                         full_push_obs[:, _OBS_ROBOT_DIM + 14 + 5])

        perform_alice_update()
        perform_bob_update(current_bob_obs)

        current_sr = iter_sr[1] / max(1, iter_sr[0])
        bob_success_buf.append(current_sr)
        st = env.get_iter_stats()
        mean_pos = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0
        writer.add_scalar("Metrics/SR", current_sr, bob_updates)
        writer.add_scalar("Metrics/Alice/ValidGoals", st["valid_goals"], bob_updates)
        print(f"[Iter {bob_updates}] SR={current_sr:.3f} PosErr={mean_pos:.4f} RotErr={mean_rot:.4f} "
              f"goals(v/i)={st['valid_goals']}/{st['invalid_goals']} "
              f"bob(s/f)={st['bob_successes']}/{st['bob_failures']} "
              f"AliceRew={last_alice_mean_rew:+.3f}", flush=True)

        if args.save_interval > 0 and bob_updates % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "latest_checkpoint.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "latest_checkpoint.pt"))
            with open(os.path.join(f"runs/{args.exp_name}", "latest_iter.txt"), "w") as f:
                f.write(str(bob_updates))
        if _stop["flag"]:
            break

    bob_ppo.save(os.path.join(bob_ppo.log_dir, "latest_checkpoint.pt"))
    alice_ppo.save(os.path.join(alice_ppo.log_dir, "latest_checkpoint.pt"))
    print(f"Done. Best Bob SR={best_bob_sr:.4f}")
    env.close()


if __name__ == "__main__":
    main()
