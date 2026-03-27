"""
Phase 2 Training Entry Point: Meta-Asymmetric Self-Play (Charlie Architecture).

Two-tier hierarchy:
    Low-level:  Frozen Phase 1.5 workers execute sub-goals at 60Hz
    Meta-level: Meta-Alice & Meta-Bob operate at 1/C Hz, issuing K-dim latents

Episode structure:
    1. Reset to S0, snapshot physics
    2. Meta-Alice: TA_meta meta-steps → produce S* (master goal)
    3. Restore to S0
    4. Meta-Bob: TB_meta meta-steps → attempt to reach S*
    5. Evaluate D(S_final, S*)
    6. Reward injection + Meta-ABC trajectory trimming
"""

import isaaclab.app
from isaaclab.app import AppLauncher

import os
import sys
import yaml
import copy
import argparse
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 Meta-ASP")
    parser.add_argument("--exp_name", type=str, default="meta_asp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_meta_iterations", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=50)

    # Phase 1.5 worker checkpoints (required)
    parser.add_argument("--chkpt_worker_left", type=str, required=True,
                        help="Path to Phase 1.5 left arm worker checkpoint")
    parser.add_argument("--chkpt_worker_right", type=str, required=True,
                        help="Path to Phase 1.5 right arm worker checkpoint")

    # Meta-level overrides
    parser.add_argument("--C", type=int, default=None,
                        help="Override atomic steps per meta-step")
    parser.add_argument("--TA_meta", type=int, default=None,
                        help="Override Meta-Alice meta-step budget")
    parser.add_argument("--TB_meta", type=int, default=None,
                        help="Override Meta-Bob meta-step budget")

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
    from asyncDualPlayPPO.tasks.meta_wrapper import MetaASPWrapper
    from asyncDualPlayPPO.utils.meta_episode_manager import MetaEpisodeManager
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.meta_storage import MetaRolloutStorage
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer

    import gymnasium as gym

    # ─── Configuration ─────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_cfg = load_cfg(os.path.join(script_dir, "cfg", "ppo", "ppo_meta.yaml"))
    worker_cfg = load_cfg(os.path.join(script_dir, "cfg", "ppo", "ppo_continuous.yaml"))

    meta_params = meta_cfg["params"]["meta"]
    C = args.C or meta_params["C"]
    TA_meta = args.TA_meta or meta_params["TA_meta"]
    TB_meta = args.TB_meta or meta_params["TB_meta"]
    success_threshold = meta_params["success_threshold"]
    K = worker_cfg["params"]["policy"].get("goal_embed_dim", 8)

    # ─── Environment ───────────────────────────────────────────
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = AsyncDualPlayEnvWrapper(base_env)
    device = env.device

    # ─── Load Frozen Workers ───────────────────────────────────
    def load_worker(chkpt_path, cfg_params, device):
        """Load a Phase 1.5 ActorCritic checkpoint and freeze weights."""
        bob_cfg = copy.deepcopy(cfg_params)
        bob_cfg["policy"]["use_goal_encoder"] = True

        obs_shape = env.bob_observation_space.shape
        state_shape = obs_shape
        action_shape = env.action_space.shape

        model = ActorCritic(obs_shape, state_shape, action_shape, bob_cfg["policy"])
        model.to(device)

        checkpoint = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(checkpoint["actor_critic"], strict=False)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        print(f"  [Worker] Loaded from {chkpt_path} — {sum(p.numel() for p in model.parameters())} params (frozen)")
        return model

    print("\n═══ Loading frozen Phase 1.5 workers ═══")
    worker_left = load_worker(args.chkpt_worker_left, worker_cfg["params"], device)
    worker_right = load_worker(args.chkpt_worker_right, worker_cfg["params"], device)

    # ─── Meta Wrapper ──────────────────────────────────────────
    meta_env = MetaASPWrapper(
        env=env,
        worker_left=worker_left,
        worker_right=worker_right,
        K=K,
        C=C,
        TA_meta=TA_meta,
        TB_meta=TB_meta,
        success_threshold=success_threshold,
        device=device,
    )

    # ─── Meta Episode Manager ─────────────────────────────────
    episode_mgr = MetaEpisodeManager(
        num_envs=args.num_envs,
        device=device,
        TA_meta=TA_meta,
        TB_meta=TB_meta,
    )

    # ─── Meta-Alice (standard PPO, obs=30, action=2K) ─────────
    meta_alice_obs_dim = 30
    meta_action_dim = 2 * K

    meta_alice_storage = MetaRolloutStorage(
        num_envs=args.num_envs,
        num_meta_steps=TA_meta,
        obs_dim=meta_alice_obs_dim,
        action_dim=meta_action_dim,
        device=device,
    )

    meta_alice_cfg = copy.deepcopy(meta_cfg["params"])
    alice_obs_shape = (meta_alice_obs_dim,)
    alice_action_shape = (meta_action_dim,)
    meta_alice_model = ActorCritic(
        alice_obs_shape, alice_obs_shape, alice_action_shape,
        meta_alice_cfg["policy"],
    ).to(device)

    # ─── Meta-Bob / Charlie (PPOABC, obs=44, action=2K) ───────
    meta_bob_obs_dim = 44  # 30 (global state) + 14 (master goal S*)
    meta_bob_storage = MetaRolloutStorage(
        num_envs=args.num_envs,
        num_meta_steps=TB_meta,
        obs_dim=meta_bob_obs_dim,
        action_dim=meta_action_dim,
        device=device,
    )

    meta_bob_cfg = copy.deepcopy(meta_cfg["params"])
    bob_obs_shape = (meta_bob_obs_dim,)
    meta_bob_model = ActorCritic(
        bob_obs_shape, bob_obs_shape, alice_action_shape,
        meta_bob_cfg["policy"],
    ).to(device)

    # Meta-ABC demonstration buffer
    meta_abc_buffer = GPUDemonstrationBuffer(
        capacity=meta_params["abc_buffer_capacity"],
        obs_shape=bob_obs_shape,
        states_shape=bob_obs_shape,
        actions_shape=alice_action_shape,
        device=device,
    )

    # ─── Optimizers ────────────────────────────────────────────
    lr = meta_cfg["params"]["learn"]["optim_stepsize"]
    alice_optimizer = torch.optim.Adam(meta_alice_model.parameters(), lr=lr)
    bob_optimizer = torch.optim.Adam(meta_bob_model.parameters(), lr=lr)

    # ─── TensorBoard ───────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{args.exp_name}_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"\n═══ Phase 2 Meta-ASP Training ═══")
    print(f"  K={K}, C={C}, TA_meta={TA_meta}, TB_meta={TB_meta}")
    print(f"  Meta-Alice obs={meta_alice_obs_dim}, Meta-Bob obs={meta_bob_obs_dim}")
    print(f"  Meta action dim={meta_action_dim}")
    print(f"  Logging to {log_dir}\n")

    # ─── Training Loop ─────────────────────────────────────────
    gamma = meta_cfg["params"]["learn"]["gamma"]
    lam = meta_cfg["params"]["learn"]["lam"]
    clip_param = meta_cfg["params"]["learn"]["cliprange"]
    abc_coef = meta_cfg["params"]["learn"]["abc_coef"]
    num_epochs = meta_cfg["params"]["learn"]["noptepochs"]
    num_mini_batches = meta_cfg["params"]["learn"]["nminibatches"]
    ent_coef = meta_cfg["params"]["learn"]["ent_coef"]
    value_loss_coef = meta_cfg["params"]["learn"]["value_loss_coef"]
    max_grad_norm = meta_cfg["params"]["learn"]["max_grad_norm"]

    all_env_ids = torch.arange(args.num_envs, device=device)
    running_sr = 0.0
    sr_alpha = 0.05

    for meta_iter in range(1, args.max_meta_iterations + 1):
        meta_alice_storage.clear()
        meta_bob_storage.clear()
        episode_mgr.reset_episode(all_env_ids)
        meta_env.reset_worker_hidden()

        # ─── 1. RESET & SNAPSHOT S0 ───────────────────────────
        env.reset()
        s0_snap = meta_env.snapshot()

        # ─── 2. META-ALICE PHASE ──────────────────────────────
        meta_alice_traj = []
        for t in range(TA_meta):
            St = meta_env.get_global_state()  # (N, 30)

            with torch.no_grad():
                g_meta, lp, val, mu, sigma = meta_alice_model.act(St, St)

            g_left = g_meta[:, :K]   # (N, K)
            g_right = g_meta[:, K:]  # (N, K)

            # Record for Meta-ABC trajectory trimming
            meta_alice_traj.append((St.clone(), g_meta.clone()))
            episode_mgr.record_alice_step(all_env_ids, St, g_meta)

            # Execute workers for C atomic steps
            meta_env.execute_workers(g_left, g_right)

            # Store in Alice's rollout buffer (reward injected at end)
            meta_alice_storage.add_transitions(
                observations=St,
                actions=g_meta,
                rewards=torch.zeros(args.num_envs, device=device),
                dones=torch.zeros(args.num_envs, device=device),
                values=val.squeeze(-1) if val.dim() > 1 else val,
                actions_log_prob=lp,
            )

        # ─── 3. RECORD S* & RESTORE S0 ───────────────────────
        S_star = meta_env.get_object_state()  # (N, 14)
        episode_mgr.master_goals.copy_(S_star)
        meta_env.restore(s0_snap)
        meta_env.reset_worker_hidden()

        # ─── 4. META-BOB PHASE ────────────────────────────────
        for t in range(TB_meta):
            St = meta_env.get_global_state()            # (N, 30)
            St_bob = torch.cat([St, S_star], dim=-1)    # (N, 44)

            with torch.no_grad():
                g_meta, lp, val, mu, sigma = meta_bob_model.act(St_bob, St_bob)

            g_left = g_meta[:, :K]
            g_right = g_meta[:, K:]

            meta_env.execute_workers(g_left, g_right)

            meta_bob_storage.add_transitions(
                observations=St_bob,
                actions=g_meta,
                rewards=torch.zeros(args.num_envs, device=device),
                dones=torch.zeros(args.num_envs, device=device),
                values=val.squeeze(-1) if val.dim() > 1 else val,
                actions_log_prob=lp,
            )

        # ─── 5. EVALUATE ──────────────────────────────────────
        S_final = meta_env.get_object_state()  # (N, 14)
        bob_success = meta_env.compute_success(S_final, S_star)  # (N,) bool

        sr = bob_success.float().mean().item()
        running_sr = sr_alpha * sr + (1 - sr_alpha) * running_sr

        # ─── 6. REWARD INJECTION ──────────────────────────────
        # Meta-Bob: +1 success, 0 failure (sparse, at last storage index)
        # Meta-Alice: +1 if Bob failed, 0 if Bob succeeded
        # Force dones[-1] = 1.0 to prevent GAE bleeding
        bob_reward = bob_success.float()
        alice_reward = (~bob_success).float()

        meta_bob_storage.rewards[TB_meta - 1, :, 0] = bob_reward
        meta_bob_storage.dones[TB_meta - 1, :, 0] = 1

        meta_alice_storage.rewards[TA_meta - 1, :, 0] = alice_reward
        meta_alice_storage.dones[TA_meta - 1, :, 0] = 1

        # ─── 7. META-ABC BUFFER PUSH (Bob failures only) ─────
        failed_envs = torch.where(~bob_success)[0]
        for eid in failed_envs.tolist():
            trimmed = episode_mgr.get_trimmed_alice_traj(eid)
            for St_t, g_t in trimmed:
                obs_t = torch.cat([St_t, S_star[eid]])  # (44,)
                meta_abc_buffer.add_trajectory(
                    obs_t.unsqueeze(0),      # (1, 44)
                    obs_t.unsqueeze(0),      # (1, 44) — states = obs
                    g_t.unsqueeze(0),        # (1, 2K)
                )

        # ─── 8. COMPUTE RETURNS & PPO UPDATES ─────────────────
        with torch.no_grad():
            St_final = meta_env.get_global_state()
            alice_last_val = meta_alice_model.critic(St_final)
            bob_last_val = meta_bob_model.critic(
                torch.cat([St_final, S_star], dim=-1)
            )

        meta_alice_storage.compute_returns(alice_last_val, gamma, lam)
        meta_bob_storage.compute_returns(bob_last_val, gamma, lam)

        # PPO update for Meta-Alice
        alice_loss_sum = 0.0
        for _ in range(num_epochs):
            for batch in meta_alice_storage.mini_batch_generator(num_mini_batches):
                obs_b, acts_b, rets_b, advs_b, vals_b, old_lp_b = batch
                advs_b = (advs_b - advs_b.mean()) / (advs_b.std() + 1e-8)

                raw, _ = meta_alice_model._actor_forward(obs_b)
                dist = meta_alice_model._make_distribution(raw)
                new_lp = dist.log_prob(acts_b).unsqueeze(-1)
                entropy = dist.entropy().mean()
                new_val = meta_alice_model.critic(obs_b)

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * advs_b
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (new_val - rets_b).pow(2).mean()

                loss = policy_loss + value_loss_coef * value_loss - ent_coef * entropy
                alice_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_alice_model.parameters(), max_grad_norm)
                alice_optimizer.step()
                alice_loss_sum += loss.item()

        # PPO update for Meta-Bob (with Meta-ABC)
        bob_loss_sum = 0.0
        abc_loss_sum = 0.0

        # Sample Meta-ABC batch once per update
        abc_sample = None
        abc_batch_size = meta_cfg["params"]["learn"]["abc_batch_size"]
        if len(meta_abc_buffer) >= abc_batch_size:
            abc_sample = meta_abc_buffer.sample(abc_batch_size)
            with torch.no_grad():
                raw_old, _ = meta_bob_model._actor_forward(abc_sample[0])
                dist_old = meta_bob_model._make_distribution(raw_old)
                abc_old_lp = dist_old.log_prob(abc_sample[2]).unsqueeze(-1)

        for _ in range(num_epochs):
            for batch in meta_bob_storage.mini_batch_generator(num_mini_batches):
                obs_b, acts_b, rets_b, advs_b, vals_b, old_lp_b = batch
                advs_b = (advs_b - advs_b.mean()) / (advs_b.std() + 1e-8)

                raw, _ = meta_bob_model._actor_forward(obs_b)
                dist = meta_bob_model._make_distribution(raw)
                new_lp = dist.log_prob(acts_b).unsqueeze(-1)
                entropy = dist.entropy().mean()
                new_val = meta_bob_model.critic(obs_b)

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * advs_b
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (new_val - rets_b).pow(2).mean()

                # Meta-ABC loss
                bc_loss = torch.tensor(0.0, device=device)
                if abc_sample is not None:
                    raw_abc, _ = meta_bob_model._actor_forward(abc_sample[0])
                    dist_abc = meta_bob_model._make_distribution(raw_abc)
                    abc_new_lp = dist_abc.log_prob(abc_sample[2]).unsqueeze(-1)
                    abc_ratio = torch.exp(abc_new_lp - abc_old_lp)
                    abc_clipped = torch.clamp(abc_ratio, 1.0 - clip_param, 1.0 + clip_param)
                    bc_loss = -torch.min(abc_ratio, abc_clipped).mean()
                    abc_loss_sum += bc_loss.item()

                loss = (policy_loss + value_loss_coef * value_loss
                        - ent_coef * entropy + abc_coef * bc_loss)
                bob_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_bob_model.parameters(), max_grad_norm)
                bob_optimizer.step()
                bob_loss_sum += loss.item()

        # ─── 9. LOGGING ───────────────────────────────────────
        n_updates = num_epochs * num_mini_batches

        writer.add_scalar("Meta/Alice/Loss", alice_loss_sum / max(n_updates, 1), meta_iter)
        writer.add_scalar("Meta/Alice/Reward", alice_reward.mean().item(), meta_iter)
        writer.add_scalar("Meta/Bob/Loss", bob_loss_sum / max(n_updates, 1), meta_iter)
        writer.add_scalar("Meta/Bob/Reward", bob_reward.mean().item(), meta_iter)
        writer.add_scalar("Meta/Bob/SuccessRate", sr, meta_iter)
        writer.add_scalar("Meta/Bob/SuccessRate_EMA", running_sr, meta_iter)
        writer.add_scalar("Meta/ABC/Loss", abc_loss_sum / max(n_updates, 1), meta_iter)
        writer.add_scalar("Meta/ABC/BufferSize", len(meta_abc_buffer), meta_iter)

        # Goal override norm monitoring (OOD detection)
        if len(meta_alice_traj) > 0:
            last_g = meta_alice_traj[-1][1]  # (N, 2K)
            writer.add_scalar("GoalOverride/NormLeft", last_g[:, :K].norm(dim=-1).mean().item(), meta_iter)
            writer.add_scalar("GoalOverride/NormRight", last_g[:, K:].norm(dim=-1).mean().item(), meta_iter)

        # Positional error
        target_dist = torch.norm(S_final[:, :3] - S_star[:, :3], dim=-1).mean().item()
        cube_dist = torch.norm(S_final[:, 7:10] - S_star[:, 7:10], dim=-1).mean().item()
        writer.add_scalar("Meta/Bob/TargetPosError", target_dist, meta_iter)
        writer.add_scalar("Meta/Bob/CubePosError", cube_dist, meta_iter)

        if meta_iter % 10 == 0:
            print(
                f"Meta-Iter {meta_iter:5d} | SR={sr:.3f} (EMA={running_sr:.3f}) | "
                f"Alice Rew={alice_reward.mean():.3f} | Bob Rew={bob_reward.mean():.3f} | "
                f"TargErr={target_dist:.4f} | CubeErr={cube_dist:.4f} | "
                f"ABC Buf={len(meta_abc_buffer)}"
            )

        # ─── 10. CHECKPOINTING ────────────────────────────────
        if meta_iter % args.save_interval == 0:
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            torch.save({
                "meta_alice": meta_alice_model.state_dict(),
                "meta_alice_optimizer": alice_optimizer.state_dict(),
                "meta_bob": meta_bob_model.state_dict(),
                "meta_bob_optimizer": bob_optimizer.state_dict(),
                "meta_iter": meta_iter,
            }, os.path.join(ckpt_dir, f"meta_iter_{meta_iter}.pt"))

    writer.close()
    simulation_app.close()
    print("\n═══ Phase 2 Meta-ASP Training Complete ═══")


if __name__ == "__main__":
    main()
