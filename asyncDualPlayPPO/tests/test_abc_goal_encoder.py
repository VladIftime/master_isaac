"""
Test 6: ABC + GoalEncoder Integration (Visual, Isaac Sim)
==========================================================

Re-enables the FULL production architecture:
  - PI encoder (PermInvEncoder)
  - Goal encoder (GoalEncoder, difference variant)
  - LSTM trunk

Verifies that the GoalEncoder doesn't destabilize ABC convergence
when goal_states are properly populated (not zeros).

Critical: goal_proj.weight is initialized with small scale (0.01)
to prevent the "Initialization Jump" — large random goal embeddings
would saturate ReLUs in the policy trunk.

The aux loss (GoalEncoder auxiliary head) is included as a secondary
objective to keep the encoder grounded in physical reality.

Usage:
    cd asyncDualPlayPPO
    python tests/test_abc_goal_encoder.py --num_iterations 200

Expected:
    - NLL should steadily decrease
    - Match% should reach >50% within 200 iterations
    - Both arms should converge to the same trajectory visually
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
    """Write a list of RGB uint8 (H×W×3) numpy arrays to an MP4 using OpenCV."""
    import cv2

    if not frames:
        return
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


class _VideoRecorder:
    """Captures rendered frames via omni.replicator and encodes to MP4.

    Usage:
        rec = _VideoRecorder(cam_pos=(x, y, z), look_at=(x, y, z))
        rec.start()
        for each sim step:
            base_env.step(...)
            rec.capture()
        rec.stop_and_save("out.mp4")
    """

    def __init__(
        self,
        cam_pos: tuple,
        look_at: tuple,
        fps: int = 24,
        resolution: tuple = (1280, 720),
    ):
        import omni.replicator.core as rep

        self._rep = rep
        self._fps = fps
        self._frames: list = []
        self.active = False

        cam = rep.create.camera(
            position=cam_pos,
            look_at=look_at,
            focal_length=18.0,
            clipping_range=(0.01, 10000.0),
        )
        rp = rep.create.render_product(cam, resolution)
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annot.attach([rp])

    def start(self) -> None:
        self._frames = []
        self.active = True

    def capture(self) -> None:
        """Grab one frame from the render annotator (call after each sim step)."""
        if not self.active:
            return
        data = self._annot.get_data()
        if data is None:
            return
        arr = data["data"] if isinstance(data, dict) else data
        if arr is None or arr.size == 0:
            return
        self._frames.append(arr[:, :, :3].copy())  # RGBA → RGB

    def stop_and_save(self, output_path: str) -> None:
        self.active = False
        if not self._frames:
            print(f"  [VideoRecorder] No frames captured — skipping {output_path}")
            return
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _write_mp4(self._frames, output_path, self._fps)
        print(f"  [VideoRecorder] {len(self._frames)} frames → {output_path}")


def install_noise_filter():
    """Filter C-level stderr noise."""
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
    parser = argparse.ArgumentParser(
        description="Test 6: ABC + GoalEncoder Integration"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="Safety cap on training iterations (default: 2000)",
    )
    parser.add_argument(
        "--episode_steps",
        type=int,
        default=100,
        help="Steps per episode (default: 100)",
    )
    parser.add_argument(
        "--abc_epochs",
        type=int,
        default=1,
        help="ABC gradient steps per iteration (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--success_window",
        type=int,
        default=10,
        help="Rolling window size for success rate (default: 10)",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.8,
        help="Rolling success rate to consider policy converged (default: 0.8)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=50,
        help="Frozen eval episodes after training converges (default: 50)",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record start (random Bob) and end (converged) episodes to tests/videos/",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    install_noise_filter()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import torch.nn as nn
    import numpy as np

    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic

    torch.manual_seed(args.seed)

    # ── Load PPO config ─────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "..", "cfg", "ppo", "ppo_continuous.yaml")
    with open(cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    pol_cfg = ppo_cfg["params"]["policy"]
    num_cat_dims = pol_cfg.get("num_cat_dims", 6)
    num_bins = pol_cfg.get("num_bins", 11)
    aux_coef = ppo_cfg["params"]["learn"].get("aux_coef", 0.1)

    # ── Create Environment (2 envs) ────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST 6: ABC + GoalEncoder Integration")
    print("  Full architecture: PI encoder + Goal encoder + LSTM")
    print("  Env 0 (left):  Alice — hard-coded trajectory (reference)")
    print("  Env 1 (right): Bob   — full pipeline trained via ABC")
    print("=" * 70)

    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.scene.env_spacing = 3.0

    print("\nCreating environment (2 envs)...")
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device
    print(f"  Device: {device}")

    # ── Observation Dimensions ──────────────────────────────────
    alice_dim_info = base_env.unwrapped.observation_manager.group_obs_dim[
        "alice_policy"
    ]
    bob_dim_info = base_env.unwrapped.observation_manager.group_obs_dim["bob_policy"]
    alice_obs_dim = (
        alice_dim_info[0]
        if isinstance(alice_dim_info, (tuple, list))
        else alice_dim_info
    )
    bob_obs_dim = (
        bob_dim_info[0] if isinstance(bob_dim_info, (tuple, list)) else bob_dim_info
    )
    print(f"  Alice obs dim: {alice_obs_dim}, Bob obs dim: {bob_obs_dim}")

    # ── Action space ────────────────────────────────────────────
    if len(base_env.action_space.shape) > 1:
        env_action_dim = base_env.action_space.shape[1]
    else:
        env_action_dim = base_env.action_space.shape[0]
    print(f"  Env action dim: {env_action_dim}")

    # ── Build Bob's ActorCritic (FULL PRODUCTION ARCHITECTURE) ──
    # This is the key difference from test_abc.py:
    #   use_goal_encoder=True, use_pi_encoder=True, use_lstm=True
    model_cfg = copy.deepcopy(pol_cfg)
    model_cfg["use_goal_encoder"] = True
    model_cfg["use_pi_encoder"] = True
    model_cfg["use_lstm"] = True
    model_cfg["use_multicategorical"] = True

    ac = ActorCritic(
        obs_shape=(bob_obs_dim,),
        states_shape=(bob_obs_dim,),
        actions_shape=(num_cat_dims,),
        initial_std=ppo_cfg["params"]["learn"].get("init_noise_std", 1.0),
        model_cfg=model_cfg,
        asymmetric=False,
    ).to(device)

    # ────────────────────────────────────────────────────────────
    # CRITICAL: "Initialization Jump" prevention
    # ────────────────────────────────────────────────────────────
    # The goal embedding is additively injected into the actor trunk:
    #   h = act(LN(W1 @ enc + W_g @ g_pooled))
    #
    # At init, if W_g is large, the random goal embedding dominates the
    # hidden state, saturates ReLUs, and kills gradients.
    #
    # Fix: initialize goal_proj with very small scale (0.01) so the goal
    # starts as a tiny hint rather than a dominant signal.
    # ────────────────────────────────────────────────────────────
    if ac._goal_proj is not None:
        with torch.no_grad():
            ac._goal_proj.weight.mul_(0.01 / 0.5)  # original gain=0.5, target=0.01
        print(
            f"\n  [Init] goal_proj scale reduced: "
            f"||W_g|| = {ac._goal_proj.weight.norm():.4f}"
        )

    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    # ── bin → env action conversion ─────────────────────────────
    def bins_to_env_action(bin_indices, gripper_state):
        """Convert 6D bin indices → 7D RMPFlow env action."""
        center = (num_bins - 1) / 2.0
        threshold = 2.0
        normalized = (bin_indices.float() - center) / center
        xyz = normalized[:, :3]
        rot_xy = normalized[:, 3:5] * 0.5

        g_bin = bin_indices[:, 5].float()
        new_gs = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0
        new_gs[g_bin > center + threshold - 1] = 1.0

        zeros1 = torch.zeros(bin_indices.shape[0], 1, device=bin_indices.device)
        env_act = torch.cat([xyz, rot_xy, zeros1, new_gs], dim=-1)
        return env_act, new_gs

    # ── Alice's hard-coded trajectory (gentle sweep) ────────────
    N = args.episode_steps
    alice_bins = torch.zeros(N, num_cat_dims, device=device)
    alice_bins[:, 0] = 5.0  # X: neutral

    PUSH_STEP = int(N * (1 / 2))
    alice_bins[:PUSH_STEP, 1] = torch.linspace(5, 6, PUSH_STEP, device=device).round()
    alice_bins[PUSH_STEP:, 1] = 7.0

    DROP_STEP = int(N * (1 / 2))
    alice_bins[:DROP_STEP, 2] = torch.linspace(
        3, 5, DROP_STEP, device=device
    ).round()  # Z: gentle up to down
    alice_bins[DROP_STEP:, 2] = 5.0

    alice_bins[:, 3] = 5.0  # Rx: neutral
    alice_bins[:, 4] = 5.0  # Ry: neutral
    alice_bins[:, 5] = 5.0  # Gripper: neutral

    print(f"\n  Alice trajectory ({N} steps):")
    print(f"    X bins (first 20): {alice_bins[:20, 0].long().tolist()}")
    print(f"    Z bin (constant):  {alice_bins[0, 2].long().item()}")
    print(f"    Gripper bin:       {alice_bins[0, 5].long().item()} (neutral)")

    # Pre-convert Alice's trajectory to env actions
    alice_gripper_traj = torch.ones(1, 1, device=device)
    alice_env_actions = []
    for t in range(N):
        act_t, alice_gripper_traj = bins_to_env_action(
            alice_bins[t : t + 1], alice_gripper_traj
        )
        alice_env_actions.append(act_t.squeeze(0))
    alice_env_actions = torch.stack(alice_env_actions)

    # ── Discover actual goal by running Alice's trajectory once ────
    # Instead of hardcoding target positions, run a warmup episode where
    # Alice executes her full trajectory and Bob does nothing.  At the end
    # we read where each object actually landed (in environment-local frame)
    # and store that as the goal.  This automatically tracks any future
    # changes to Alice's trajectory without manual updates.

    class _GoalEpisodeManager:
        def __init__(self, num_envs, device):
            # Goal layout: [target_object(0:6), cube(6:12)] = [pos(3)+euler(3)] each
            self.goal_states = torch.zeros(num_envs, 12, device=device)
            self.goal_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.pos_threshold = 0.04
            self.rot_threshold = 0.5

    base_env.episode_manager = _GoalEpisodeManager(2, device)

    def _read_pose_local(env, object_name, env_idx):
        """Return [pos(3), euler(3)] in environment-local frame for env_idx.

        Matches the convention used by object_states() and goal_states() in
        observations.py: positions are world-space minus env_origins, rotation
        is ZYX Euler (roll, pitch, yaw) converted from the world quaternion.
        """
        obj = env.scene[object_name]
        pos_w = obj.data.root_pos_w
        quat_w = obj.data.root_quat_w
        # Handle optional instance dimension
        if pos_w.dim() == 3:
            pos_w, quat_w = pos_w[:, 0], quat_w[:, 0]
        p = pos_w[env_idx] - env.scene.env_origins[env_idx]
        # Inline quat(w,x,y,z) → ZYX Euler so we don't import private helpers
        q = quat_w[env_idx]
        w_q, x_q, y_q, z_q = q[0], q[1], q[2], q[3]
        roll = torch.atan2(2 * (w_q * x_q + y_q * z_q), 1 - 2 * (x_q * x_q + y_q * y_q))
        pitch = torch.asin((2 * (w_q * y_q - z_q * x_q)).clamp(-1.0, 1.0))
        yaw = torch.atan2(2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q))
        euler = torch.stack([roll, pitch, yaw])
        return torch.cat([p, euler])  # (6,)

    print("\n  [Warmup] Running Alice's trajectory to discover real goal state...")
    warmup_obs, _ = base_env.reset()
    warmup_terminated_early = False
    for t in range(N):
        alice_act_w = alice_env_actions[t].unsqueeze(0)
        bob_act_w = torch.zeros(1, env_action_dim, device=device)
        combined_w = torch.cat([alice_act_w, bob_act_w], dim=0)
        warmup_obs, _, term_w, trunc_w, _ = base_env.step(combined_w)
        if (term_w | trunc_w).any():
            print("  [Warmup] Episode terminated early — goal captured at step", t)
            warmup_terminated_early = True
            break

    # Read env-0 final object poses in local frame.
    # Both envs share the same layout (just different origins), so one
    # measurement drives goal_states for all envs.
    tgt_pose = _read_pose_local(base_env, "target_object", 0)
    cube_pose = _read_pose_local(base_env, "cube", 0)
    goal_12d = torch.cat([tgt_pose, cube_pose])  # (12,)
    base_env.episode_manager.goal_states[:] = goal_12d.unsqueeze(0)
    base_env.episode_manager.goal_valid[:] = True

    print(f"  [Warmup] Goal discovered (local frame):")
    print(
        f"    target_object: pos={tgt_pose[:3].cpu().tolist()}  "
        f"euler={tgt_pose[3:].cpu().tolist()}"
    )
    print(
        f"    cube:          pos={cube_pose[:3].cpu().tolist()}  "
        f"euler={cube_pose[3:].cpu().tolist()}"
    )

    def _bob_achieved_goal():
        """Check whether Bob's objects (env 1) are within threshold of the goal.

        Reads live physics state — call after the last env.step() of an episode,
        before the next reset.  Returns (achieved, tgt_dist, cube_dist).
        """
        import math as _math

        thr = base_env.episode_manager.pos_threshold
        tgt_c = _read_pose_local(base_env, "target_object", 1)
        cube_c = _read_pose_local(base_env, "cube", 1)
        tgt_g = base_env.episode_manager.goal_states[1, 0:6]
        cube_g = base_env.episode_manager.goal_states[1, 6:12]
        tgt_dist = (tgt_c[:3] - tgt_g[:3]).norm().item()
        cube_dist = (cube_c[:3] - cube_g[:3]).norm().item()
        achieved = tgt_dist < thr and cube_dist < thr
        return achieved, tgt_dist, cube_dist

    # ── Video recorder (always created — used for goal-achievement recording) ──
    video_dir = os.path.join(script_dir, "videos")
    e0 = base_env.scene.env_origins[0].cpu().tolist()
    e1 = base_env.scene.env_origins[1].cpu().tolist()
    mid_x = (e0[0] + e1[0]) / 2
    mid_y = (e0[1] + e1[1]) / 2
    cam_pos = (mid_x, mid_y + 5.5, 1.5)
    look_at = (mid_x, mid_y + 0.5, 0.5)
    recorder = _VideoRecorder(cam_pos=cam_pos, look_at=look_at)
    print(f"\n  Recorder ready → {video_dir}/")
    print(f"    Camera: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")

    # ── Capture pre-training episode (random Bob) ────────────────
    if args.record_video:
        print("\n  Capturing START episode (untrained / random Bob)...")
        recorder.start()
        obs_dict_rec, _ = base_env.reset()
        bob_obs_rec = obs_dict_rec["bob_policy"]
        bob_gripper_rec = torch.ones(1, 1, device=device)
        for t in range(N):
            alice_act_rec = alice_env_actions[t].unsqueeze(0)
            with torch.no_grad():
                bob_bins_rec, _, _, _, _ = ac.act(bob_obs_rec[1:2], None)
                bob_act_rec, bob_gripper_rec = bins_to_env_action(
                    bob_bins_rec, bob_gripper_rec
                )
            combined_rec = torch.cat([alice_act_rec, bob_act_rec], dim=0)
            obs_dict_rec, _, term_rec, trunc_rec, _ = base_env.step(combined_rec)
            recorder.capture()
            bob_obs_rec = obs_dict_rec["bob_policy"]
            if (term_rec | trunc_rec).any():
                obs_dict_rec, _ = base_env.reset()
                bob_obs_rec = obs_dict_rec["bob_policy"]
        recorder.stop_and_save(os.path.join(video_dir, "start_random_abc_encoder.mp4"))

    # ══════════════════════════════════════════════════════════════
    # TRAINING LOOP  (runs until Bob achieves both goals)
    # ══════════════════════════════════════════════════════════════
    thr = base_env.episode_manager.pos_threshold
    print(
        f"\n  Training until rolling success rate >= {args.success_threshold:.0%} "
        f"over last {args.success_window} episodes"
    )
    print(
        f"  pos_threshold={thr:.3f}m | max_iterations={args.max_iterations} | "
        f"aux_coef={aux_coef}"
    )
    print(
        f"\n  {'Iter':>6} | {'NLL':>8} | {'Aux':>8} | {'Match%':>8} | "
        f"{'TgtD':>6} | {'CubD':>6} | {'WinRate':>8} | {'Status':>8}"
    )
    print("  " + "-" * 80)

    nll_history = []
    aux_history = []
    it = 0
    converged = False
    convergence_iter = None
    success_window = deque(maxlen=args.success_window)  # True/False per episode

    while it < args.max_iterations and not converged:
        it += 1
        # ── Phase A: Replay episode ──
        obs_dict, info = base_env.reset()
        bob_obs_all = obs_dict["bob_policy"]

        bob_gripper = torch.ones(1, 1, device=device)
        demo_obs_list = []
        bob_obs_list = []  # Bob's actual env-1 observations (for match% eval)
        demo_act_list = []
        episode_corrupted = False
        bob_terminated_early = False  # track so we don't check goal on a reset state

        h_bob = torch.zeros(1, ac.lstm_hidden_size, device=device)
        c_bob = torch.zeros(1, ac.lstm_hidden_size, device=device)

        for t in range(N):
            # Env 0: Alice's hard-coded action
            alice_act = alice_env_actions[t].unsqueeze(0)

            # Env 1: Bob's policy (with LSTM hidden state management)
            bob_obs_t = bob_obs_all[1:2]
            with torch.no_grad():
                bob_bins, _, _, _, _, (h_bob, c_bob) = ac.act_with_hidden(
                    bob_obs_t, None, (h_bob, c_bob)
                )
                bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

            combined_action = torch.cat([alice_act, bob_act], dim=0)

            # Collect demo (env 0's obs + Alice's actions) and Bob's actual obs
            if not episode_corrupted:
                demo_obs_list.append(bob_obs_all[0:1].clone())  # Alice's obs (training)
                bob_obs_list.append(
                    bob_obs_all[1:2].clone()
                )  # Bob's obs   (eval match%)
                demo_act_list.append(alice_bins[t : t + 1].clone())

            # Step
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

        # ── Phase B: Train via ABC + Aux loss ──
        if episode_corrupted or len(demo_obs_list) < 10:
            nll_history.append(nll_history[-1] if nll_history else 20.0)
            aux_history.append(aux_history[-1] if aux_history else 1.0)
            if episode_corrupted:
                print(f"  [iter {it}] Env 0 terminated — skipping training")
            continue

        demo_obs = torch.cat(
            demo_obs_list, dim=0
        )  # (T, obs_dim) Alice's obs — training
        bob_obs = torch.cat(
            bob_obs_list, dim=0
        )  # (T, obs_dim) Bob's obs  — match% eval
        demo_acts = torch.cat(demo_act_list, dim=0)  # (T, num_cat_dims)
        T = demo_obs.shape[0]

        for _ in range(args.abc_epochs):
            # ── Sequential LSTM ABC loss ──
            # Process trajectory step-by-step, carrying (h,c) forward.
            # This is critical: batch evaluate() resets h,c=0 for every obs,
            # so the LSTM can't distinguish step 0 from step 79.
            # Sequential processing gives the LSTM temporal context.
            h = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c = torch.zeros(1, ac.lstm_hidden_size, device=device)

            seq_lps = []
            for step in range(T):
                obs_t = demo_obs[step : step + 1]  # (1, obs_dim)
                # _actor_forward returns (raw, (h_detach, c_detach))
                # We use h_detach, c_detach for the NEXT step (TBPTT-1),
                # but gradients still flow through the current step's LSTM.
                raw, (h, c) = ac._actor_forward(obs_t, (h, c))
                dist = ac._make_distribution(raw)
                lp = dist.log_prob(demo_acts[step : step + 1].long())
                seq_lps.append(lp)

            bc_loss = -torch.stack(seq_lps).mean()  # raw NLL

            # ── Aux loss: GoalEncoder distance prediction ──
            aux_loss_val = torch.tensor(0.0, device=device)
            if ac.goal_encoder is not None and ac.goal_encoder.use_aux_loss:
                robot_dim = ac._ge_robot_dim
                obj_section = demo_obs[:, robot_dim:]
                obj_chunks = obj_section.view(T, ac._ge_num_objects, ac._ge_raw_per_obj)
                goal_poses = obj_chunks[
                    :, :, ac._ge_obj_state_dim : ac._ge_obj_state_dim + ac._ge_goal_dim
                ]
                current_poses = obj_chunks[:, :, :6]

                goal_flat = goal_poses.reshape(T, -1)
                current_flat = current_poses.reshape(T, -1)

                aux_total, _, _ = ac.goal_encoder.aux_loss(goal_flat, current_flat)
                aux_loss_val = aux_total

            total_loss = bc_loss + aux_coef * aux_loss_val

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()

        # ── Evaluate ──
        with torch.no_grad():
            # NLL: measured on Alice's obs (mirrors the training objective)
            h_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            eval_lps = []
            for step in range(T):
                obs_t = demo_obs[step : step + 1]
                raw, (h_eval, c_eval) = ac._actor_forward(obs_t, (h_eval, c_eval))
                dist = ac._make_distribution(raw)
                eval_lps.append(dist.log_prob(demo_acts[step : step + 1].long()))
            raw_nll = -torch.stack(eval_lps).mean().item()

            # Match%: measured on Bob's ACTUAL env-1 observations.
            # This is the real generalization metric — does Bob, from his own
            # (potentially diverged) state, still choose Alice's actions?
            h_bob = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_bob = torch.zeros(1, ac.lstm_hidden_size, device=device)
            eval_greedy = []
            for step in range(T):
                obs_t = bob_obs[step : step + 1]
                raw, (h_bob, c_bob) = ac._actor_forward(obs_t, (h_bob, c_bob))
                logits = raw.view(1, ac.num_cat_dims, ac.num_bins)
                eval_greedy.append(logits.argmax(dim=-1).squeeze(0))  # (num_cat_dims,)

            greedy = torch.stack(eval_greedy)  # (T, num_cat_dims)
            x_mode = greedy[:, 0].mode().values.item()
            z_mode = greedy[:, 2].mode().values.item()
            gr_mode = greedy[:, 5].mode().values.item()
            all_match = (greedy == demo_acts).all(dim=-1).float().mean().item()

        nll_history.append(raw_nll)
        aux_val = (
            aux_loss_val.item()
            if isinstance(aux_loss_val, torch.Tensor)
            else aux_loss_val
        )
        aux_history.append(aux_val)

        # ── Goal check: read live physics state for Bob's env ──
        # Only valid if Bob's env ran to the end of the episode (no early reset).
        tgt_dist = cube_dist = float("inf")
        episode_success = False
        if not bob_terminated_early:
            episode_success, tgt_dist, cube_dist = _bob_achieved_goal()

        success_window.append(episode_success)
        win_rate = sum(success_window) / len(success_window) if success_window else 0.0

        # Check convergence: window must be full AND rate >= threshold
        if (
            len(success_window) == args.success_window
            and win_rate >= args.success_threshold
        ):
            converged = True
            convergence_iter = it

        status = "CONVERGED" if converged else ("GOAL" if episode_success else "      ")

        if it % 5 == 0 or it == 1 or episode_success or converged:
            print(
                f"  {it:>6} | {raw_nll:>+8.3f} | {aux_val:>8.4f} | "
                f"{all_match:>8.1%} | {tgt_dist:>6.3f} | {cube_dist:>6.3f} | "
                f"{win_rate:>8.1%} | {status}"
            )

    # ══════════════════════════════════════════════════════════════
    # RESULT
    # ══════════════════════════════════════════════════════════════
    final_nll = nll_history[-1] if nll_history else float("inf")
    nll_decreased = len(nll_history) >= 2 and nll_history[-1] < nll_history[0]

    print(f"\n{'=' * 70}")
    if converged:
        print(
            f"  CONVERGED at iteration {convergence_iter}  "
            f"(win rate {args.success_threshold:.0%} over last {args.success_window} eps)"
        )
    else:
        print(
            f"  Max iterations ({args.max_iterations}) reached — policy did NOT converge."
        )
        final_win_rate = (
            sum(success_window) / len(success_window) if success_window else 0.0
        )
        print(
            f"  Final win rate: {final_win_rate:.1%} over last {len(success_window)} episodes"
        )
    print(
        f"  NLL: {nll_history[0]:+.3f} → {final_nll:+.3f}  "
        f"({'decreased' if nll_decreased else 'DID NOT decrease'})"
    )
    if aux_history:
        print(f"  Aux loss: {aux_history[0]:.4f} → {aux_history[-1]:.4f}")
    print(f"{'=' * 70}")

    # ══════════════════════════════════════════════════════════════
    # FROZEN EVALUATION PHASE
    # ══════════════════════════════════════════════════════════════
    print(
        f"\n  Running frozen eval: {args.eval_episodes} episodes (no gradient updates)..."
    )
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
                bob_bins_e, _, _, _, _, (h_bob_e, c_bob_e) = ac.act_with_hidden(
                    bob_obs_e[1:2], None, (h_bob_e, c_bob_e)
                )
                bob_act_e, bob_gripper_e = bins_to_env_action(bob_bins_e, bob_gripper_e)

            combined_e = torch.cat([alice_act_e, bob_act_e], dim=0)
            obs_dict_e, _, term_e, trunc_e, _ = base_env.step(combined_e)
            bob_obs_e = obs_dict_e["bob_policy"]

            dones_e = term_e | trunc_e
            if dones_e.any():
                if dones_e[1].item():
                    bob_term_early_e = True
                obs_dict_e, _ = base_env.reset()
                bob_obs_e = obs_dict_e["bob_policy"]

        ep_success = False
        ep_tgt_dist = ep_cube_dist = float("inf")
        if not bob_term_early_e:
            ep_success, ep_tgt_dist, ep_cube_dist = _bob_achieved_goal()

        if ep_success:
            eval_successes += 1
        eval_tgt_dists.append(ep_tgt_dist)
        eval_cube_dists.append(ep_cube_dist)

        if (ep + 1) % 10 == 0 or (ep + 1) == args.eval_episodes:
            print(
                f"    Ep {ep+1:>3}/{args.eval_episodes}  "
                f"success={eval_successes}/{ep+1}  "
                f"({eval_successes/(ep+1):.1%})  "
                f"avg_tgt={sum(eval_tgt_dists)/(ep+1):.3f}  "
                f"avg_cube={sum(eval_cube_dists)/(ep+1):.3f}"
            )

    final_success_rate = eval_successes / args.eval_episodes
    finite_tgt = [d for d in eval_tgt_dists if d < float("inf")]
    finite_cube = [d for d in eval_cube_dists if d < float("inf")]
    avg_tgt = sum(finite_tgt) / len(finite_tgt) if finite_tgt else float("inf")
    avg_cube = sum(finite_cube) / len(finite_cube) if finite_cube else float("inf")

    print(f"\n{'=' * 70}")
    print(f"  EVAL RESULT over {args.eval_episodes} episodes:")
    print(
        f"    Success rate : {final_success_rate:.1%}  ({eval_successes}/{args.eval_episodes})"
    )
    print(f"    Avg tgt dist : {avg_tgt:.4f} m")
    print(f"    Avg cube dist: {avg_cube:.4f} m")
    print(f"{'=' * 70}")

    # ── Record the post-eval episode ──────────────────────────────
    print(f"\n  Recording {'converged' if converged else 'final'} episode...")
    recorder.start()
    obs_dict, _ = base_env.reset()
    bob_obs_all = obs_dict["bob_policy"]
    bob_gripper = torch.ones(1, 1, device=device)

    action_diffs = []
    bob_bins_traj = []

    h_bob_rec = torch.zeros(1, ac.lstm_hidden_size, device=device)
    c_bob_rec = torch.zeros(1, ac.lstm_hidden_size, device=device)

    for t in range(N):
        alice_act = alice_env_actions[t].unsqueeze(0)
        bob_obs_t = bob_obs_all[1:2]
        with torch.no_grad():
            bob_bins, _, _, _, _, (h_bob_rec, c_bob_rec) = ac.act_with_hidden(
                bob_obs_t, None, (h_bob_rec, c_bob_rec)
            )
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
        f"converged_iter{convergence_iter}.mp4" if converged else f"final_iter{it}.mp4"
    )
    recorder.stop_and_save(os.path.join(video_dir, video_name))

    # ── Print full trajectory comparison ─────────────────────────
    alice_bins_list = alice_bins.long().cpu().tolist()
    dim_names = ["X ", "Y ", "Z ", "Rx", "Ry", "Gr"]
    print(f"\n  {'Step':>4}  {'':4}  " + "  ".join(f"{d:>4}" for d in dim_names))
    print("  " + "-" * (6 + 4 + len(dim_names) * 6))
    for t in range(N):
        a = alice_bins_list[t]
        b = bob_bins_traj[t]
        match_step = all(a[d] == b[d] for d in range(num_cat_dims))
        tag = "  " if match_step else "!!"
        print(
            f"  {t:>4}  {tag}  "
            + "  ".join(
                (
                    f"\033[92m{b[d]:>4}\033[0m"
                    if a[d] == b[d]
                    else f"\033[91m{b[d]:>4}\033[0m"
                )
                for d in range(num_cat_dims)
            )
            + "   Alice: "
            + " ".join(f"{a[d]:>2}" for d in range(num_cat_dims))
        )

    print(f"\n  Mean action diff: {np.mean(action_diffs):.4f}")

    # ── Continuous replay for visual inspection ──
    print("\n  Replaying continuously for visual inspection...")
    print("  (Close window or Ctrl+C to exit)\n")

    try:
        while simulation_app.is_running():
            obs_dict, _ = base_env.reset()
            bob_obs_all = obs_dict["bob_policy"]
            bob_gripper = torch.ones(1, 1, device=device)

            for t in range(N):
                alice_act = alice_env_actions[t].unsqueeze(0)

                bob_obs_t = bob_obs_all[1:2]
                with torch.no_grad():
                    bob_bins = ac.act_inference(bob_obs_t)
                    bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

                combined = torch.cat([alice_act, bob_act], dim=0)
                obs_dict, _, terminated, truncated, _ = base_env.step(combined)
                bob_obs_all = obs_dict["bob_policy"]

                dones = terminated | truncated
                if dones.any():
                    obs_dict, _ = base_env.reset()
                    bob_obs_all = obs_dict["bob_policy"]
                    break

    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
