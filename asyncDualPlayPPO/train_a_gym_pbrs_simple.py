"""
gym-pusht PBRS Model A — single-agent PPO, no curriculum, no ASP.

Smart-env architecture: GymPushPrimitiveEnv (push macro + PBRS + done inside)
vectorized via AsyncVectorEnv (TorchVecAdapter).  Reuses the project's custom
PPO + ActorCriticPush unchanged — only the environment differs.  Runs in
.master_venv (no Isaac / cuRobo).

Run:
  .master_venv/bin/python3 -m asyncDualPlayPPO.train_a_gym_pbrs_simple \
      --num_envs 16 --max_iterations 50 --exp_name gym_pbrs_a_smoke
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
import copy
import signal
import argparse

import numpy as np
import torch
torch.set_num_threads(1)
import yaml
from gymnasium import spaces as gym_spaces
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asyncDualPlayPPO.tasks.utils.gym_push_primitive_env import TorchVecAdapter
from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
from asyncDualPlayPPO.algorithms.rl.ppo.storage import RolloutStorage


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="gym-pusht PBRS Model A")
    p.add_argument("--exp_name", type=str, default="gym_pbrs_a")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--push_nsteps", type=int, default=45,
                   help="PPO rollout length (pushes/env/update). Larger = bigger batch; "
                        "important at low num_envs to keep the PPO batch usable.")
    p.add_argument("--max_iterations", type=int, default=3000)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chkpt", type=str, default=None)
    p.add_argument("--resume_iteration", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg = load_cfg(cfg_path)

    push_nsteps = args.push_nsteps
    num_cat_dims, num_bins = 4, 21
    nminibatches = max(1, args.num_envs // 16)
    while nminibatches > 1 and args.num_envs % nminibatches != 0:
        nminibatches -= 1

    env = TorchVecAdapter(num_envs=args.num_envs, device=device,
                          max_pushes_per_episode=5, seed=args.seed)
    print(f"[gym-A] num_envs={env.num_envs} device={device} obs_dim={env.obs_dim}")

    _mc = gym_spaces.Box(0.0, float(num_bins - 1), (num_cat_dims,), np.float32)

    agent_cfg = copy.deepcopy(ppo_cfg["params"])
    agent_cfg["learn"].update(dict(nsteps=push_nsteps, noptepochs=3, nminibatches=nminibatches,
                                   cliprange=0.2, ent_coef=0.002, gamma=0.95, lam=0.95,
                                   optim_stepsize=3e-4))
    agent_cfg["policy"]["num_bins"] = num_bins
    agent_cfg["policy"]["num_cat_dims"] = num_cat_dims

    agent = PPO(vec_env=env, cfg_train=agent_cfg, device=device,
                sampler="sequential", log_dir=f"runs/{args.exp_name}/agent", asymmetric=False)
    agent.observation_space = env.observation_space
    agent.state_space = env.state_space
    agent.action_space = _mc
    agent.desired_kl = None
    agent.actor_critic = ActorCriticPush(env.observation_space.shape, env.state_space.shape,
                                         _mc.shape, agent.init_noise_std, agent.model_cfg,
                                         asymmetric=False).to(device)
    agent.storage = RolloutStorage(env.num_envs, push_nsteps, env.observation_space.shape,
                                   env.state_space.shape, _mc.shape, device, "sequential")
    agent.optimizer = torch.optim.Adam(agent.actor_critic.parameters(), lr=agent.learning_rate)

    if args.chkpt and os.path.isfile(args.chkpt):
        agent.load(args.chkpt)
        print(f"[Resume] {args.chkpt}")
        # Embedded checkpoint iteration is authoritative (latest_iter.txt advisory).
        if agent.loaded_iteration is not None:
            args.resume_iteration = int(agent.loaded_iteration)
            print(f"[Resume] iter from checkpoint: {args.resume_iteration}")

    use_lstm = agent.actor_critic.use_lstm
    lsz = agent.actor_critic.lstm_hidden_size if use_lstm else 0
    hidden = [torch.zeros(env.num_envs, lsz, device=device),
              torch.zeros(env.num_envs, lsz, device=device)] if use_lstm else None

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")
    os.makedirs(agent.log_dir, exist_ok=True)
    best_sr = -1.0
    _best_path = os.path.join(agent.log_dir, "best_sr.txt")
    if args.chkpt and os.path.isfile(_best_path):
        try:
            best_sr = float(open(_best_path).read().strip())
            print(f"[Resume] Restored best SR: {best_sr:.4f}")
        except Exception:
            pass
    iteration = args.resume_iteration if args.chkpt else 0
    ema_rew = 0.0

    _stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: _stop.update(flag=True))

    obs = env.reset()

    while iteration < args.max_iterations:
        agent.storage.clear()
        rew_buf, sr_buf, pos_buf, rot_buf = [], [], [], []
        ep_pushes, ep_succ = [], []

        for _ in range(push_nsteps):
            with torch.no_grad():
                h_in = (hidden[0], hidden[1]) if hidden else None
                actions, log_prob, value, mu, sigma, stored_h, new_h = \
                    agent.actor_critic.act_with_hidden(obs, None, h_in)
                if hidden is not None and new_h is not None:
                    hidden[0], hidden[1] = new_h[0], new_h[1]

            next_obs, reward, done, info = env.step(actions)

            agent.storage.add_transitions(obs, obs, actions, reward, done, value, log_prob,
                                          mu, sigma, masks=(~done).float(), hidden_state=stored_h)

            rew_buf.extend(reward.detach().cpu().tolist())
            sr_buf.extend(np.asarray(info["at_goal"], dtype=np.float32).tolist())
            pos_buf.extend(np.asarray(info["pos_err"], dtype=np.float32).tolist())
            rot_buf.extend(np.asarray(info["cos_rot_err"], dtype=np.float32).tolist())
            dmask = np.asarray(info["done"], dtype=bool)
            if dmask.any():
                ep_pushes.extend(np.asarray(info["ep_pushes"])[dmask].tolist())
                ep_succ.extend(np.asarray(info["success"], dtype=bool)[dmask].tolist())
                if hidden is not None:
                    dt = torch.as_tensor(dmask, device=device)
                    hidden[0][dt] = 0.0
                    hidden[1][dt] = 0.0
            obs = next_obs

        with torch.no_grad():
            last_val = agent.actor_critic.critic(obs)
        agent.storage.compute_returns(last_val, agent.gamma, agent.lam)
        loss_val, loss_surr = agent.update()
        agent.storage.clear()

        mean_rew = float(np.mean(rew_buf)) if rew_buf else 0.0
        sr = float(np.mean(sr_buf)) if sr_buf else 0.0
        mean_pos = float(np.mean(pos_buf)) if pos_buf else 0.0
        mean_rot = float(np.mean(rot_buf)) if rot_buf else 0.0
        ep_sr = float(np.mean(ep_succ)) if ep_succ else 0.0
        ema_rew = 0.9 * ema_rew + 0.1 * mean_rew

        writer.add_scalar("Loss/Value", loss_val, iteration)
        writer.add_scalar("Loss/Surrogate", loss_surr, iteration)
        writer.add_scalar("Reward/Mean", mean_rew, iteration)
        writer.add_scalar("Metrics/SuccessRate", sr, iteration)
        writer.add_scalar("Metrics/PosError", mean_pos, iteration)
        writer.add_scalar("Metrics/CosRotErr", mean_rot, iteration)
        writer.add_scalar("Metrics/EpisodicSR", ep_sr, iteration)

        print(f"[Iter {iteration:5d}] Loss={loss_surr:.4f} Val={loss_val:.4f} "
              f"Rew={mean_rew:+.4f}(EMA {ema_rew:+.4f}) PosErr={mean_pos:.4f} "
              f"CosRot={mean_rot:.4f} SR={sr:.4f} EpiSR={ep_sr:.3f} Ep={len(ep_succ)} "
              f"Best={best_sr:.4f}", flush=True)

        if sr > best_sr:
            best_sr = sr
            agent.save(os.path.join(agent.log_dir, "model_best.pt"))
            try:
                with open(_best_path + ".tmp", "w") as _bf:
                    _bf.write(repr(float(best_sr)))
                os.replace(_best_path + ".tmp", _best_path)
            except Exception:
                pass
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(agent.log_dir, "latest_checkpoint.pt"), iteration=iteration)
            with open(os.path.join(agent.log_dir, "latest_iter.txt"), "w") as f:
                f.write(str(iteration))

        iteration += 1
        if _stop["flag"]:
            agent.save(os.path.join(agent.log_dir, "latest_checkpoint.pt"), iteration=iteration)
            break

    agent.save(os.path.join(agent.log_dir, "latest_checkpoint.pt"), iteration=iteration)
    with open(os.path.join(agent.log_dir, "latest_iter.txt"), "w") as f:
        f.write(str(iteration))
    print(f"Done. Best SR={best_sr:.4f}")
    env.close()


if __name__ == "__main__":
    main()
