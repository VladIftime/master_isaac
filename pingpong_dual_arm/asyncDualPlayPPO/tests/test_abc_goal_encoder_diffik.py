"""
Test 6 (DiffIK): ABC + GoalEncoder Integration (Visual, Isaac Sim)
===================================================================

DifferentialIK variant of test_abc_goal_encoder.py.  Identical logic
except the environment uses DLS Jacobian IK (relative-pose mode) instead
of RMPflow.  Run both tests and compare convergence speed and final
success rates to evaluate the two IK solvers under the full production
architecture.

Full architecture:
  - PermInvEncoder (PI encoder)
  - GoalEncoder (difference variant)
  - LSTM trunk
  - DifferentialIK action layer

Videos are saved with a _diffik suffix so they don't overwrite RMPflow
recordings in tests/videos/.

Usage:
    cd asyncDualPlayPPO
    python tests/test_abc_goal_encoder_diffik.py --num_iterations 200

Compare against RMPflow:
    python tests/test_abc_goal_encoder.py --num_iterations 200
"""

import isaaclab.app
from isaaclab.app import AppLauncher

import argparse
import os
import sys
import copy
import yaml
import threading
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _write_mp4(frames: list, path: str, fps: int) -> None:
    import cv2
    if not frames:
        return
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


class _VideoRecorder:
    def __init__(self, cam_pos, look_at, fps=24, resolution=(1920, 1080)):
        import omni.replicator.core as rep
        self._rep   = rep
        self._fps   = fps
        self._frames: list = []
        self.active = False
        cam  = rep.create.camera(position=cam_pos, look_at=look_at,
                                  focal_length=18.0, clipping_range=(0.01, 10000.0))
        rp   = rep.create.render_product(cam, resolution)
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annot.attach([rp])

    def start(self):
        self._frames = []
        self.active  = True

    def capture(self):
        if not self.active:
            return
        data = self._annot.get_data()
        if data is None:
            return
        arr = data["data"] if isinstance(data, dict) else data
        if arr is None or arr.size == 0:
            return
        self._frames.append(arr[:, :, :3].copy())

    def stop_and_save(self, output_path):
        self.active = False
        if not self._frames:
            print(f"  [VideoRecorder] No frames captured — skipping {output_path}")
            return
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _write_mp4(self._frames, output_path, self._fps)
        print(f"  [VideoRecorder] {len(self._frames)} frames → {output_path}")


def install_noise_filter():
    _DROP = (b"[Lula] Joint", b"Warning: link", b"urdf_parser",
             b"flat_black", b"IMemoryBudgetManager", b"robotiq_coupler")
    orig_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _worker():
        orig = os.fdopen(orig_fd, "wb", buffering=0)
        buf  = b""
        with os.fdopen(read_fd, "rb", buffering=0) as src:
            while True:
                chunk = src.read(256)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_nl   = line + b"\n"
                    if not any(p in line_nl for p in _DROP):
                        orig.write(line_nl)
                        orig.flush()
        if buf and not any(p in buf for p in _DROP):
            orig.write(buf)
            orig.flush()

    threading.Thread(target=_worker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description="Test 6 DiffIK: ABC + GoalEncoder Integration"
    )
    parser.add_argument("--max_iterations",   type=int,   default=2000)
    parser.add_argument("--episode_steps",    type=int,   default=100)
    parser.add_argument("--abc_epochs",       type=int,   default=1)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--success_window",   type=int,   default=10)
    parser.add_argument("--success_threshold",type=float, default=0.8)
    parser.add_argument("--eval_episodes",    type=int,   default=50)
    parser.add_argument(
        "--record_video", action="store_true",
        help="Record start (random Bob) and end (converged) episodes to tests/videos/",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    install_noise_filter()
    app_launcher   = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import torch.nn as nn
    import numpy as np

    from isaaclab.envs import ManagerBasedRLEnv
    # ── Only line that differs from test_abc_goal_encoder.py ─────────────────
    from asyncDualPlayPPO.tasks.async_dual_play_diffik import AsyncDualPlayDiffIKEnvCfg as AsyncDualPlayEnvCfg
    # ─────────────────────────────────────────────────────────────────────────
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic

    torch.manual_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path   = os.path.join(script_dir, "..", "cfg", "ppo", "ppo_continuous.yaml")
    with open(cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    pol_cfg      = ppo_cfg["params"]["policy"]
    num_cat_dims = pol_cfg.get("num_cat_dims", 6)
    num_bins     = pol_cfg.get("num_bins",     11)
    aux_coef     = ppo_cfg["params"]["learn"].get("aux_coef", 0.1)

    print("\n" + "=" * 70)
    print("  TEST 6 (DiffIK): ABC + GoalEncoder Integration")
    print("  Controller: DLS Jacobian IK (relative-pose mode, scale=0.5)")
    print("  Full architecture: PI encoder + Goal encoder + LSTM")
    print("  Env 0 (left):  Alice — hard-coded trajectory (reference)")
    print("  Env 1 (right): Bob   — full pipeline trained via ABC")
    print("=" * 70)

    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs  = 2
    env_cfg.scene.env_spacing = 3.0

    print("\nCreating environment (2 envs, DiffIK)...")
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device   = base_env.device
    print(f"  Device: {device}")

    alice_dim_info = base_env.unwrapped.observation_manager.group_obs_dim["alice_policy"]
    bob_dim_info   = base_env.unwrapped.observation_manager.group_obs_dim["bob_policy"]
    alice_obs_dim  = alice_dim_info[0] if isinstance(alice_dim_info, (tuple, list)) else alice_dim_info
    bob_obs_dim    = bob_dim_info[0]   if isinstance(bob_dim_info,   (tuple, list)) else bob_dim_info
    print(f"  Alice obs dim: {alice_obs_dim}, Bob obs dim: {bob_obs_dim}")

    if len(base_env.action_space.shape) > 1:
        env_action_dim = base_env.action_space.shape[1]
    else:
        env_action_dim = base_env.action_space.shape[0]
    print(f"  Env action dim: {env_action_dim}")

    model_cfg = copy.deepcopy(pol_cfg)
    model_cfg["use_goal_encoder"]     = True
    model_cfg["use_pi_encoder"]       = True
    model_cfg["use_lstm"]             = True
    model_cfg["use_multicategorical"] = True

    ac = ActorCritic(
        obs_shape=(bob_obs_dim,),
        states_shape=(bob_obs_dim,),
        actions_shape=(num_cat_dims,),
        initial_std=ppo_cfg["params"]["learn"].get("init_noise_std", 1.0),
        model_cfg=model_cfg,
        asymmetric=False,
    ).to(device)

    if ac._goal_proj is not None:
        with torch.no_grad():
            ac._goal_proj.weight.mul_(0.01 / 0.5)
        print(f"\n  [Init] goal_proj scale reduced: ||W_g|| = {ac._goal_proj.weight.norm():.4f}")

    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    def bins_to_env_action(bin_indices, gripper_state):
        """Convert 6D bin indices → 7D DiffIK env action."""
        center     = (num_bins - 1) / 2.0
        threshold  = 2.0
        normalized = (bin_indices.float() - center) / center
        xyz        = normalized[:, :3]
        rot_xy     = normalized[:, 3:5] * 0.5

        g_bin  = bin_indices[:, 5].float()
        new_gs = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0
        new_gs[g_bin > center + threshold - 1] =  1.0

        zeros1  = torch.zeros(bin_indices.shape[0], 1, device=bin_indices.device)
        env_act = torch.cat([xyz, rot_xy, zeros1, new_gs], dim=-1)
        return env_act, new_gs

    N = args.episode_steps
    alice_bins = torch.zeros(N, num_cat_dims, device=device)
    alice_bins[:, 0] = 5.0

    PUSH_STEP = int(N * (1 / 2))
    alice_bins[:PUSH_STEP, 1] = torch.linspace(5, 6, PUSH_STEP, device=device).round()
    alice_bins[PUSH_STEP:, 1] = 7.0

    DROP_STEP = int(N * (1 / 2))
    alice_bins[:DROP_STEP, 2] = torch.linspace(3, 5, DROP_STEP, device=device).round()
    alice_bins[DROP_STEP:, 2] = 5.0

    alice_bins[:, 3] = 5.0
    alice_bins[:, 4] = 5.0
    alice_bins[:, 5] = 5.0

    alice_gripper_traj = torch.ones(1, 1, device=device)
    alice_env_actions  = []
    for t in range(N):
        act_t, alice_gripper_traj = bins_to_env_action(alice_bins[t:t+1], alice_gripper_traj)
        alice_env_actions.append(act_t.squeeze(0))
    alice_env_actions = torch.stack(alice_env_actions)

    class _GoalEpisodeManager:
        def __init__(self, num_envs, device):
            self.goal_states  = torch.zeros(num_envs, 12, device=device)
            self.goal_valid   = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.pos_threshold = 0.04
            self.rot_threshold = 0.5

    base_env.episode_manager = _GoalEpisodeManager(2, device)

    def _read_pose_local(env, object_name, env_idx):
        obj    = env.scene[object_name]
        pos_w  = obj.data.root_pos_w
        quat_w = obj.data.root_quat_w
        if pos_w.dim() == 3:
            pos_w, quat_w = pos_w[:, 0], quat_w[:, 0]
        p = pos_w[env_idx] - env.scene.env_origins[env_idx]
        q = quat_w[env_idx]
        w_q, x_q, y_q, z_q = q[0], q[1], q[2], q[3]
        roll  = torch.atan2(2 * (w_q*x_q + y_q*z_q), 1 - 2*(x_q*x_q + y_q*y_q))
        pitch = torch.asin((2 * (w_q*y_q - z_q*x_q)).clamp(-1.0, 1.0))
        yaw   = torch.atan2(2 * (w_q*z_q + x_q*y_q), 1 - 2*(y_q*y_q + z_q*z_q))
        return torch.cat([p, torch.stack([roll, pitch, yaw])])

    print("\n  [Warmup] Running Alice's trajectory to discover real goal state...")
    warmup_obs, _ = base_env.reset()
    for t in range(N):
        alice_act_w = alice_env_actions[t].unsqueeze(0)
        bob_act_w   = torch.zeros(1, env_action_dim, device=device)
        combined_w  = torch.cat([alice_act_w, bob_act_w], dim=0)
        warmup_obs, _, term_w, trunc_w, _ = base_env.step(combined_w)
        if (term_w | trunc_w).any():
            print("  [Warmup] Episode terminated early at step", t)
            break

    tgt_pose  = _read_pose_local(base_env, "target_object", 0)
    cube_pose = _read_pose_local(base_env, "cube",          0)
    goal_12d  = torch.cat([tgt_pose, cube_pose])
    base_env.episode_manager.goal_states[:] = goal_12d.unsqueeze(0)
    base_env.episode_manager.goal_valid[:]  = True
    print(f"  [Warmup] Goal: target={tgt_pose[:3].cpu().tolist()}  cube={cube_pose[:3].cpu().tolist()}")

    def _bob_achieved_goal():
        import math as _math
        thr    = base_env.episode_manager.pos_threshold
        tgt_c  = _read_pose_local(base_env, "target_object", 1)
        cube_c = _read_pose_local(base_env, "cube",          1)
        tgt_g  = base_env.episode_manager.goal_states[1, 0:6]
        cube_g = base_env.episode_manager.goal_states[1, 6:12]
        tgt_dist  = (tgt_c[:3]  - tgt_g[:3]).norm().item()
        cube_dist = (cube_c[:3] - cube_g[:3]).norm().item()
        return tgt_dist < thr and cube_dist < thr, tgt_dist, cube_dist

    def _verify_encoder_semantics(n_episodes=3):
        import numpy as np
        CORR_THRESHOLD      = 0.30
        COLLAPSE_THRESHOLD  = 0.01
        _r  = ac._ge_robot_dim
        _s  = ac._ge_obj_state_dim
        _g  = ac._ge_goal_dim
        _ch = ac._ge_raw_per_obj
        d1_idx = _r + _s + _g
        print(f"\n  [EncoderCheck] Verifying GoalEncoder semantics ({n_episodes} eps, DiffIK)...")
        g_norms_all = []
        pos_dists_all = []
        for _ in range(n_episodes):
            obs_dict_c, _ = base_env.reset()
            bob_obs_c = obs_dict_c["bob_policy"]
            bob_gripper_c = torch.ones(1, 1, device=device)
            h_c = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_c = torch.zeros(1, ac.lstm_hidden_size, device=device)
            for t in range(N):
                obs_t = bob_obs_c[1:2]
                with torch.no_grad():
                    obj_section = obs_t[:, _r:]
                    obj_chunks  = obj_section.view(1, ac._ge_num_objects, _ch)
                    goal_poses_t    = obj_chunks[:, :, _s:_s+_g]
                    current_poses_t = obj_chunks[:, :, :_g]
                    g = ac.goal_encoder(
                        goal_poses_t.reshape(1, -1),
                        current_poses_t.reshape(1, -1),
                    )
                    g_norms_all.append(g.norm(dim=-1).item())
                    pos_dists_all.append(obs_t[0, d1_idx].item())
                    bob_bins_c, _, _, _, _, (h_c, c_c) = ac.act_with_hidden(obs_t, None, (h_c, c_c))
                    bob_act_c, bob_gripper_c = bins_to_env_action(bob_bins_c, bob_gripper_c)
                alice_act_c = alice_env_actions[t].unsqueeze(0)
                combined_c  = torch.cat([alice_act_c, bob_act_c], dim=0)
                obs_dict_c, _, term_c, trunc_c, _ = base_env.step(combined_c)
                bob_obs_c = obs_dict_c["bob_policy"]
                if (term_c | trunc_c).any():
                    obs_dict_c, _ = base_env.reset()
                    bob_obs_c = obs_dict_c["bob_policy"]
                    break
        if len(g_norms_all) < 10:
            print("  [EncoderCheck] Too few steps — SKIP")
            return False
        g_arr = np.array(g_norms_all)
        d_arr = np.array(pos_dists_all)
        corr  = float(np.corrcoef(g_arr, d_arr)[0, 1]) if g_arr.std() > 1e-6 else 0.0
        g_std = g_arr.std()
        corr_ok       = corr  > CORR_THRESHOLD
        not_collapsed = g_std > COLLAPSE_THRESHOLD
        print(f"  [EncoderCheck] Pearson corr(||g||, pos_dist) = {corr:+.4f}  → {'PASS ✓' if corr_ok else 'FAIL ✗'}")
        print(f"  [EncoderCheck] g_norm non-collapsed (std > {COLLAPSE_THRESHOLD})  → {'PASS ✓' if not_collapsed else 'FAIL ✗'}")
        return corr_ok and not_collapsed

    video_dir = os.path.join(script_dir, "videos")
    e0 = base_env.scene.env_origins[0].cpu().tolist()
    e1 = base_env.scene.env_origins[1].cpu().tolist()
    mid_x = (e0[0] + e1[0]) / 2
    mid_y = (e0[1] + e1[1]) / 2
    cam_pos = (mid_x, mid_y + 5.5, 1.5)
    look_at = (mid_x, mid_y + 0.5, 0.5)
    recorder = _VideoRecorder(cam_pos=cam_pos, look_at=look_at)
    print(f"\n  Recorder ready → {video_dir}/  [_diffik suffix]")

    if args.record_video:
        print("\n  Capturing START episode (untrained Bob, DiffIK)...")
        recorder.start()
        obs_dict_rec, _ = base_env.reset()
        bob_obs_rec = obs_dict_rec["bob_policy"]
        bob_gripper_rec = torch.ones(1, 1, device=device)
        for t in range(N):
            alice_act_rec = alice_env_actions[t].unsqueeze(0)
            with torch.no_grad():
                bob_bins_rec, _, _, _, _ = ac.act(bob_obs_rec[1:2], None)
                bob_act_rec, bob_gripper_rec = bins_to_env_action(bob_bins_rec, bob_gripper_rec)
            combined_rec = torch.cat([alice_act_rec, bob_act_rec], dim=0)
            obs_dict_rec, _, term_rec, trunc_rec, _ = base_env.step(combined_rec)
            recorder.capture()
            bob_obs_rec = obs_dict_rec["bob_policy"]
            if (term_rec | trunc_rec).any():
                obs_dict_rec, _ = base_env.reset()
                bob_obs_rec = obs_dict_rec["bob_policy"]
        recorder.stop_and_save(os.path.join(video_dir, "start_random_abc_encoder_diffik.mp4"))

    print(
        f"\n  Training until rolling SR >= {args.success_threshold:.0%} "
        f"over last {args.success_window} eps | aux_coef={aux_coef}"
    )
    print(
        f"\n  {'Iter':>6} | {'NLL':>8} | {'Aux':>8} | {'GEGrad':>7} | {'GECorr':>7} | "
        f"{'Match%':>8} | {'TgtD':>6} | {'CubD':>6} | {'WinRate':>8} | {'Status':>8}"
    )
    print("  " + "-" * 95)

    nll_history    = []
    aux_history    = []
    it             = 0
    converged      = False
    convergence_iter = None
    success_window = deque(maxlen=args.success_window)

    while it < args.max_iterations and not converged:
        it += 1

        obs_dict, _ = base_env.reset()
        bob_obs_all = obs_dict["bob_policy"]

        bob_gripper      = torch.ones(1, 1, device=device)
        demo_obs_list    = []
        bob_obs_list     = []
        demo_act_list    = []
        episode_corrupted = False
        bob_terminated_early = False

        h_bob = torch.zeros(1, ac.lstm_hidden_size, device=device)
        c_bob = torch.zeros(1, ac.lstm_hidden_size, device=device)

        for t in range(N):
            alice_act = alice_env_actions[t].unsqueeze(0)
            bob_obs_t = bob_obs_all[1:2]
            with torch.no_grad():
                bob_bins, _, _, _, _, (h_bob, c_bob) = ac.act_with_hidden(bob_obs_t, None, (h_bob, c_bob))
                bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

            combined_action = torch.cat([alice_act, bob_act], dim=0)

            if not episode_corrupted:
                demo_obs_list.append(bob_obs_all[0:1].clone())
                bob_obs_list.append(bob_obs_all[1:2].clone())
                demo_act_list.append(alice_bins[t:t+1].clone())

            obs_dict, _, terminated, truncated, _ = base_env.step(combined_action)
            bob_obs_all = obs_dict["bob_policy"]

            dones = terminated | truncated
            if dones.any():
                if dones[0].item():
                    episode_corrupted = True
                if dones[1].item():
                    bob_terminated_early = True
                obs_dict, _ = base_env.reset()
                bob_obs_all = obs_dict["bob_policy"]

        if episode_corrupted or len(demo_obs_list) < 10:
            nll_history.append(nll_history[-1] if nll_history else 20.0)
            aux_history.append(aux_history[-1] if aux_history else 1.0)
            continue

        demo_obs  = torch.cat(demo_obs_list, dim=0)
        bob_obs   = torch.cat(bob_obs_list,  dim=0)
        demo_acts = torch.cat(demo_act_list, dim=0)
        T = demo_obs.shape[0]

        aux_loss_val = torch.tensor(0.0, device=device)
        ge_grad_norm = 0.0

        for _ in range(args.abc_epochs):
            h = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c = torch.zeros(1, ac.lstm_hidden_size, device=device)
            seq_lps = []
            for step in range(T):
                obs_t = demo_obs[step:step+1]
                raw, (h, c) = ac._actor_forward(obs_t, (h, c))
                dist = ac._make_distribution(raw)
                lp   = dist.log_prob(demo_acts[step:step+1].long())
                seq_lps.append(lp)
            bc_loss = -torch.stack(seq_lps).mean()

            aux_loss_val = torch.tensor(0.0, device=device)
            if ac.goal_encoder is not None and ac.goal_encoder.use_aux_loss:
                robot_dim   = ac._ge_robot_dim
                obj_section = demo_obs[:, robot_dim:]
                obj_chunks  = obj_section.view(T, ac._ge_num_objects, ac._ge_raw_per_obj)
                goal_poses    = obj_chunks[:, :, ac._ge_obj_state_dim:ac._ge_obj_state_dim+ac._ge_goal_dim]
                current_poses = obj_chunks[:, :, :6]
                aux_total, _, _ = ac.goal_encoder.aux_loss(
                    goal_poses.reshape(T, -1), current_poses.reshape(T, -1)
                )
                aux_loss_val = aux_total

            total_loss = bc_loss + aux_coef * aux_loss_val
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)

            ge_grad_sq = 0.0
            if ac.goal_encoder is not None:
                for p in ac.goal_encoder.parameters():
                    if p.grad is not None:
                        ge_grad_sq += p.grad.norm().item() ** 2
            ge_grad_norm = ge_grad_sq ** 0.5
            optimizer.step()

        with torch.no_grad():
            h_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            eval_lps = []
            for step in range(T):
                obs_t = demo_obs[step:step+1]
                raw, (h_eval, c_eval) = ac._actor_forward(obs_t, (h_eval, c_eval))
                dist = ac._make_distribution(raw)
                eval_lps.append(dist.log_prob(demo_acts[step:step+1].long()))
            raw_nll = -torch.stack(eval_lps).mean().item()

            h_bob_e = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_bob_e = torch.zeros(1, ac.lstm_hidden_size, device=device)
            eval_greedy = []
            for step in range(T):
                obs_t = bob_obs[step:step+1]
                raw, (h_bob_e, c_bob_e) = ac._actor_forward(obs_t, (h_bob_e, c_bob_e))
                logits = raw.view(1, ac.num_cat_dims, ac.num_bins)
                eval_greedy.append(logits.argmax(dim=-1).squeeze(0))
            greedy    = torch.stack(eval_greedy)
            all_match = (greedy == demo_acts).all(dim=-1).float().mean().item()

            ge_corr = float("nan")
            if ac.goal_encoder is not None:
                _r  = ac._ge_robot_dim
                _ch = ac._ge_raw_per_obj
                _s  = ac._ge_obj_state_dim
                _gd = ac._ge_goal_dim
                d1  = _r + _s + _gd
                o_sec  = demo_obs[:, _r:]
                o_chk  = o_sec.view(T, ac._ge_num_objects, _ch)
                g_flat = o_chk[:, :, _s:_s+_gd].reshape(T, -1)
                c_flat = o_chk[:, :, :_gd].reshape(T, -1)
                g_emb  = ac.goal_encoder(g_flat, c_flat)
                g_nrm  = g_emb.norm(dim=-1)
                pos_d  = demo_obs[:, d1]
                if g_nrm.std() > 1e-6 and pos_d.std() > 1e-6:
                    ge_corr = torch.corrcoef(torch.stack([g_nrm, pos_d]))[0, 1].item()

        nll_history.append(raw_nll)
        aux_val = aux_loss_val.item() if isinstance(aux_loss_val, torch.Tensor) else aux_loss_val
        aux_history.append(aux_val)

        tgt_dist = cube_dist = float("inf")
        episode_success = False
        if not bob_terminated_early:
            episode_success, tgt_dist, cube_dist = _bob_achieved_goal()

        success_window.append(episode_success)
        win_rate = sum(success_window) / len(success_window) if success_window else 0.0

        if len(success_window) == args.success_window and win_rate >= args.success_threshold:
            converged        = True
            convergence_iter = it

        status = "CONVERGED" if converged else ("GOAL" if episode_success else "      ")

        if it % 5 == 0 or it == 1 or episode_success or converged:
            ge_corr_s = f"{ge_corr:>+7.3f}" if ge_corr == ge_corr else "    nan"
            print(
                f"  {it:>6} | {raw_nll:>+8.3f} | {aux_val:>8.4f} | "
                f"{ge_grad_norm:>7.4f} | {ge_corr_s} | "
                f"{all_match:>8.1%} | {tgt_dist:>6.3f} | {cube_dist:>6.3f} | "
                f"{win_rate:>8.1%} | {status}"
            )

    final_nll     = nll_history[-1] if nll_history else float("inf")
    nll_decreased = len(nll_history) >= 2 and nll_history[-1] < nll_history[0]

    print(f"\n{'=' * 70}")
    if converged:
        print(f"  CONVERGED at iteration {convergence_iter}")
    else:
        final_wr = sum(success_window) / len(success_window) if success_window else 0.0
        print(f"  Max iterations reached — NOT converged.  Final win rate: {final_wr:.1%}")
    print(f"  NLL: {nll_history[0]:+.3f} → {final_nll:+.3f}  ({'decreased' if nll_decreased else 'DID NOT decrease'})")
    if aux_history:
        print(f"  Aux loss: {aux_history[0]:.4f} → {aux_history[-1]:.4f}")
    print(f"{'=' * 70}")

    print(f"\n  Running frozen eval: {args.eval_episodes} episodes...")
    eval_successes = 0
    eval_tgt_dists = []
    eval_cube_dists = []

    for ep in range(args.eval_episodes):
        obs_dict_e, _ = base_env.reset()
        bob_obs_e = obs_dict_e["bob_policy"]
        bob_gripper_e = torch.ones(1, 1, device=device)
        bob_term_early_e = False
        h_bob_e = torch.zeros(1, ac.lstm_hidden_size, device=device)
        c_bob_e = torch.zeros(1, ac.lstm_hidden_size, device=device)

        for t in range(N):
            alice_act_e = alice_env_actions[t].unsqueeze(0)
            with torch.no_grad():
                bob_bins_e, _, _, _, _, (h_bob_e, c_bob_e) = ac.act_with_hidden(bob_obs_e[1:2], None, (h_bob_e, c_bob_e))
                bob_act_e, bob_gripper_e = bins_to_env_action(bob_bins_e, bob_gripper_e)
            combined_e = torch.cat([alice_act_e, bob_act_e], dim=0)
            obs_dict_e, _, term_e, trunc_e, _ = base_env.step(combined_e)
            bob_obs_e = obs_dict_e["bob_policy"]
            dones_e   = term_e | trunc_e
            if dones_e.any():
                if dones_e[1].item():
                    bob_term_early_e = True
                obs_dict_e, _ = base_env.reset()
                bob_obs_e = obs_dict_e["bob_policy"]

        ep_success = False
        ep_tgt = ep_cube = float("inf")
        if not bob_term_early_e:
            ep_success, ep_tgt, ep_cube = _bob_achieved_goal()
        if ep_success:
            eval_successes += 1
        eval_tgt_dists.append(ep_tgt)
        eval_cube_dists.append(ep_cube)

        if (ep + 1) % 10 == 0 or (ep + 1) == args.eval_episodes:
            print(
                f"    Ep {ep+1:>3}/{args.eval_episodes}  "
                f"success={eval_successes}/{ep+1} ({eval_successes/(ep+1):.1%})  "
                f"avg_tgt={sum(eval_tgt_dists)/(ep+1):.3f}  avg_cube={sum(eval_cube_dists)/(ep+1):.3f}"
            )

    final_sr = eval_successes / args.eval_episodes
    finite_tgt  = [d for d in eval_tgt_dists  if d < float("inf")]
    finite_cube = [d for d in eval_cube_dists if d < float("inf")]
    avg_tgt  = sum(finite_tgt)  / len(finite_tgt)  if finite_tgt  else float("inf")
    avg_cube = sum(finite_cube) / len(finite_cube) if finite_cube else float("inf")

    print(f"\n{'=' * 70}")
    print(f"  EVAL RESULT over {args.eval_episodes} episodes (DiffIK):")
    print(f"    Success rate : {final_sr:.1%}  ({eval_successes}/{args.eval_episodes})")
    print(f"    Avg tgt dist : {avg_tgt:.4f} m")
    print(f"    Avg cube dist: {avg_cube:.4f} m")
    print(f"{'=' * 70}")

    encoder_ok = _verify_encoder_semantics(n_episodes=3)
    if not encoder_ok:
        print("\n  [EncoderCheck] WARNING: GoalEncoder semantics FAILED under DiffIK.")

    print(f"\n  Recording {'converged' if converged else 'final'} episode (DiffIK)...")
    recorder.start()
    obs_dict, _ = base_env.reset()
    bob_obs_all  = obs_dict["bob_policy"]
    bob_gripper  = torch.ones(1, 1, device=device)
    action_diffs = []
    bob_bins_traj = []
    h_bob_rec = torch.zeros(1, ac.lstm_hidden_size, device=device)
    c_bob_rec = torch.zeros(1, ac.lstm_hidden_size, device=device)

    for t in range(N):
        alice_act = alice_env_actions[t].unsqueeze(0)
        bob_obs_t = bob_obs_all[1:2]
        with torch.no_grad():
            bob_bins, _, _, _, _, (h_bob_rec, c_bob_rec) = ac.act_with_hidden(bob_obs_t, None, (h_bob_rec, c_bob_rec))
            bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)
        bob_bins_traj.append(bob_bins.squeeze(0).long().cpu().tolist())
        action_diffs.append((alice_act - bob_act).abs().mean().item())
        combined = torch.cat([alice_act, bob_act], dim=0)
        obs_dict, _, terminated, truncated, _ = base_env.step(combined)
        recorder.capture()
        bob_obs_all = obs_dict["bob_policy"]
        if (terminated | truncated).any():
            obs_dict, _ = base_env.reset()
            bob_obs_all = obs_dict["bob_policy"]

    video_name = (
        f"converged_iter{convergence_iter}_diffik.mp4"
        if converged
        else f"final_iter{it}_diffik.mp4"
    )
    recorder.stop_and_save(os.path.join(video_dir, video_name))

    alice_bins_list = alice_bins.long().cpu().tolist()
    dim_names = ["X ", "Y ", "Z ", "Rx", "Ry", "Gr"]
    print(f"\n  {'Step':>4}  {'':4}  " + "  ".join(f"{d:>4}" for d in dim_names))
    print("  " + "-" * (6 + 4 + len(dim_names) * 6))
    for t in range(N):
        a = alice_bins_list[t]
        b = bob_bins_traj[t]
        tag = "  " if all(a[d] == b[d] for d in range(num_cat_dims)) else "!!"
        print(
            f"  {t:>4}  {tag}  "
            + "  ".join(
                (f"\033[92m{b[d]:>4}\033[0m" if a[d] == b[d] else f"\033[91m{b[d]:>4}\033[0m")
                for d in range(num_cat_dims)
            )
            + "   Alice: " + " ".join(f"{a[d]:>2}" for d in range(num_cat_dims))
        )

    import numpy as _np
    print(f"\n  Mean action diff: {_np.mean(action_diffs):.4f}")

    print("\n  Replaying continuously (Ctrl+C to exit)...")
    try:
        while simulation_app.is_running():
            obs_dict, _ = base_env.reset()
            bob_obs_all  = obs_dict["bob_policy"]
            bob_gripper  = torch.ones(1, 1, device=device)
            for t in range(N):
                alice_act = alice_env_actions[t].unsqueeze(0)
                with torch.no_grad():
                    bob_bins = ac.act_inference(bob_obs_all[1:2])
                    bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)
                combined = torch.cat([alice_act, bob_act], dim=0)
                obs_dict, _, terminated, truncated, _ = base_env.step(combined)
                bob_obs_all = obs_dict["bob_policy"]
                if (terminated | truncated).any():
                    obs_dict, _ = base_env.reset()
                    bob_obs_all = obs_dict["bob_policy"]
                    break
    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
