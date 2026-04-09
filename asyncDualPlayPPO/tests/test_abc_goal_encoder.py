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
        "--num_iterations", type=int, default=200,
        help="Training iterations (default: 200)",
    )
    parser.add_argument(
        "--episode_steps", type=int, default=80,
        help="Steps per episode (default: 80)",
    )
    parser.add_argument(
        "--abc_epochs", type=int, default=1,
        help="ABC gradient steps per iteration (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    alice_dim_info = base_env.unwrapped.observation_manager.group_obs_dim["alice_policy"]
    bob_dim_info = base_env.unwrapped.observation_manager.group_obs_dim["bob_policy"]
    alice_obs_dim = alice_dim_info[0] if isinstance(alice_dim_info, (tuple, list)) else alice_dim_info
    bob_obs_dim = bob_dim_info[0] if isinstance(bob_dim_info, (tuple, list)) else bob_dim_info
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
        print(f"\n  [Init] goal_proj scale reduced: "
              f"||W_g|| = {ac._goal_proj.weight.norm():.4f}")

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
    alice_bins[:, 0] = torch.linspace(4, 7, N, device=device).round()  # X: gentle L→R
    alice_bins[:, 1] = 5.0  # Y: neutral
    alice_bins[:, 2] = 5.0  # Z: same height
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
            alice_bins[t:t+1], alice_gripper_traj
        )
        alice_env_actions.append(act_t.squeeze(0))
    alice_env_actions = torch.stack(alice_env_actions)

    # ── Fake goal_states (Alice's "final pose" as the goal) ─────
    # Instead of zeros, populate with a meaningful goal:
    # the object positions at the END of Alice's trajectory.
    # This gives the GoalEncoder a non-trivial signal to encode.
    #
    # For this test we use a fixed known goal pose:
    #   obj1 goal: [0.15, 0.5, 0.05, 0, 0, 0]  (target object moved right)
    #   obj2 goal: [-0.10, 0.5, 0.05, 0, 0, 0]  (cube stays near center)
    class _GoalEpisodeManager:
        def __init__(self, num_envs, device):
            # Goal layout: [obj1_pos(3), obj1_euler(3), obj2_pos(3), obj2_euler(3)] = 12D
            self.goal_states = torch.zeros(num_envs, 12, device=device)
            # Object 1 goal (target_object moved to the right)
            self.goal_states[:, 0] = 0.15   # x
            self.goal_states[:, 1] = 0.5    # y
            self.goal_states[:, 2] = 0.05   # z
            # Object 2 goal (cube near center)
            self.goal_states[:, 6] = -0.10  # x
            self.goal_states[:, 7] = 0.5    # y
            self.goal_states[:, 8] = 0.05   # z
            # Euler angles all zero (no rotation goal)
            self.goal_valid = torch.ones(num_envs, dtype=torch.bool, device=device)
            self.pos_threshold = 0.04
            self.rot_threshold = 0.5

    base_env.episode_manager = _GoalEpisodeManager(2, device)

    # ── Optional video recorder ──────────────────────────────────
    recorder = None
    video_dir = os.path.join(script_dir, "videos")
    if args.record_video:
        e0 = base_env.scene.env_origins[0].cpu().tolist()
        e1 = base_env.scene.env_origins[1].cpu().tolist()
        mid_x = (e0[0] + e1[0]) / 2
        mid_y = (e0[1] + e1[1]) / 2
        cam_pos = (mid_x, mid_y + 2.0, 1.5)
        look_at = (mid_x, mid_y - 0.5, 0.5)
        recorder = _VideoRecorder(cam_pos=cam_pos, look_at=look_at)
        print(f"\n  Video recording enabled → {video_dir}/")
        print(f"    Camera: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
        print(f"    Look-at: ({look_at[0]:.2f}, {look_at[1]:.2f}, {look_at[2]:.2f})")

    # ── Capture pre-training episode (random Bob) ────────────────
    if recorder is not None:
        print("\n  Capturing START episode (untrained / random Bob)...")
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
        recorder.stop_and_save(os.path.join(video_dir, "start_random.mp4"))

    # ══════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Training for {args.num_iterations} iterations "
          f"(raw NLL + aux_coef={aux_coef})...")
    print(f"\n  {'Iter':>6} | {'NLL':>8} | {'Aux':>8} | {'X md':>6} | "
          f"{'Z md':>6} | {'Gr md':>6} | {'Match%':>8}")
    print("  " + "-" * 65)

    nll_history = []
    aux_history = []

    for it in range(1, args.num_iterations + 1):
        # ── Phase A: Replay episode ──
        obs_dict, info = base_env.reset()
        bob_obs_all = obs_dict["bob_policy"]

        bob_gripper = torch.ones(1, 1, device=device)
        demo_obs_list = []
        demo_act_list = []
        episode_corrupted = False

        for t in range(N):
            # Env 0: Alice's hard-coded action
            alice_act = alice_env_actions[t].unsqueeze(0)

            # Env 1: Bob's policy (with LSTM hidden state management)
            bob_obs_t = bob_obs_all[1:2]
            with torch.no_grad():
                bob_bins, _, _, _, _ = ac.act(bob_obs_t, None)
                bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

            combined_action = torch.cat([alice_act, bob_act], dim=0)

            # Collect demo (env 0's obs + Alice's actions)
            if not episode_corrupted:
                demo_obs_list.append(bob_obs_all[0:1].clone())
                demo_act_list.append(alice_bins[t:t+1].clone())

            # Step
            obs_dict, _, terminated, truncated, _ = base_env.step(combined_action)
            bob_obs_all = obs_dict["bob_policy"]

            dones = terminated | truncated
            if dones.any():
                if dones[0].item():
                    episode_corrupted = True
                obs_dict, _ = base_env.reset()
                bob_obs_all = obs_dict["bob_policy"]

        # ── Phase B: Train via ABC + Aux loss ──
        if episode_corrupted or len(demo_obs_list) < 10:
            nll_history.append(nll_history[-1] if nll_history else 20.0)
            aux_history.append(aux_history[-1] if aux_history else 1.0)
            if episode_corrupted:
                print(f"  [iter {it}] Env 0 terminated — skipping training")
            continue

        demo_obs = torch.cat(demo_obs_list, dim=0)   # (T, obs_dim)
        demo_acts = torch.cat(demo_act_list, dim=0)   # (T, num_cat_dims)
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
                obs_t = demo_obs[step:step+1]  # (1, obs_dim)
                # _actor_forward returns (raw, (h_detach, c_detach))
                # We use h_detach, c_detach for the NEXT step (TBPTT-1),
                # but gradients still flow through the current step's LSTM.
                raw, (h, c) = ac._actor_forward(obs_t, (h, c))
                dist = ac._make_distribution(raw)
                lp = dist.log_prob(demo_acts[step:step+1].long())
                seq_lps.append(lp)

            bc_loss = -torch.stack(seq_lps).mean()  # raw NLL

            # ── Aux loss: GoalEncoder distance prediction ──
            aux_loss_val = torch.tensor(0.0, device=device)
            if ac.goal_encoder is not None and ac.goal_encoder.use_aux_loss:
                robot_dim = ac._ge_robot_dim
                obj_section = demo_obs[:, robot_dim:]
                obj_chunks = obj_section.view(
                    T, ac._ge_num_objects, ac._ge_raw_per_obj
                )
                goal_poses = obj_chunks[
                    :, :,
                    ac._ge_obj_state_dim : ac._ge_obj_state_dim + ac._ge_goal_dim
                ]
                current_poses = obj_chunks[:, :, :6]

                goal_flat = goal_poses.reshape(T, -1)
                current_flat = current_poses.reshape(T, -1)

                aux_total, _, _ = ac.goal_encoder.aux_loss(
                    goal_flat, current_flat
                )
                aux_loss_val = aux_total

            total_loss = bc_loss + aux_coef * aux_loss_val

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()

        # ── Evaluate (also sequential for fair NLL measurement) ──
        with torch.no_grad():
            h_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            eval_lps = []
            eval_greedy = []
            for step in range(T):
                obs_t = demo_obs[step:step+1]
                raw, (h_eval, c_eval) = ac._actor_forward(obs_t, (h_eval, c_eval))
                dist = ac._make_distribution(raw)
                eval_lps.append(dist.log_prob(demo_acts[step:step+1].long()))
                # Greedy: argmax per dim
                logits = raw.view(1, ac.num_cat_dims, ac.num_bins)
                eval_greedy.append(logits.argmax(dim=-1).squeeze(0))  # (num_cat_dims,)

            raw_nll = -torch.stack(eval_lps).mean().item()
            greedy = torch.stack(eval_greedy)  # (T, num_cat_dims)
            x_mode = greedy[:, 0].mode().values.item()
            z_mode = greedy[:, 2].mode().values.item()
            gr_mode = greedy[:, 5].mode().values.item()
            all_match = (greedy == demo_acts).all(dim=-1).float().mean().item()

        nll_history.append(raw_nll)
        aux_val = aux_loss_val.item() if isinstance(aux_loss_val, torch.Tensor) else aux_loss_val
        aux_history.append(aux_val)

        if it % 5 == 0 or it == 1:
            print(
                f"  {it:>6} | {raw_nll:>+8.3f} | {aux_val:>8.4f} | "
                f"{x_mode:>6.0f} | {z_mode:>6.0f} | {gr_mode:>6.0f} | "
                f"{all_match:>7.1%}"
            )

    # ══════════════════════════════════════════════════════════════
    # FINAL VERIFICATION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  FINAL EPISODE — Both envs should now move IDENTICALLY")
    print(f"{'=' * 70}")

    obs_dict, _ = base_env.reset()
    bob_obs_all = obs_dict["bob_policy"]
    bob_gripper = torch.ones(1, 1, device=device)

    if recorder is not None:
        print("\n  Capturing END episode (converged Bob)...")
        recorder.start()

    action_diffs = []
    for t in range(N):
        alice_act = alice_env_actions[t].unsqueeze(0)

        bob_obs_t = bob_obs_all[1:2]
        with torch.no_grad():
            bob_bins = ac.act_inference(bob_obs_t)
            bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

        diff = (alice_act - bob_act).abs().mean().item()
        action_diffs.append(diff)

        combined = torch.cat([alice_act, bob_act], dim=0)
        obs_dict, _, terminated, truncated, _ = base_env.step(combined)
        if recorder is not None:
            recorder.capture()
        bob_obs_all = obs_dict["bob_policy"]

        dones = terminated | truncated
        if dones.any():
            obs_dict, _ = base_env.reset()
            bob_obs_all = obs_dict["bob_policy"]

    if recorder is not None:
        recorder.stop_and_save(os.path.join(video_dir, "end_converged.mp4"))

    # ── Results ──
    mean_diff = np.mean(action_diffs)
    final_nll = nll_history[-1] if nll_history else float("inf")
    nll_decreased = len(nll_history) >= 2 and nll_history[-1] < nll_history[0]

    print(f"\n  NLL:         {nll_history[0]:+.3f} → {final_nll:+.3f}  "
          f"({'✓ decreased' if nll_decreased else '✗ DID NOT decrease'})")
    print(f"  Action diff: {mean_diff:.4f}  "
          f"({'✓ small' if mean_diff < 0.3 else '✗ large'})")

    with torch.no_grad():
        h_f = torch.zeros(1, ac.lstm_hidden_size, device=device)
        c_f = torch.zeros(1, ac.lstm_hidden_size, device=device)
        final_greedy_list = []
        for step in range(T):
            obs_t = demo_obs[step:step+1]
            raw, (h_f, c_f) = ac._actor_forward(obs_t, (h_f, c_f))
            logits = raw.view(1, ac.num_cat_dims, ac.num_bins)
            final_greedy_list.append(logits.argmax(dim=-1).squeeze(0))
        final_greedy = torch.stack(final_greedy_list)
        z_mode = final_greedy[:, 2].mode().values.item()
        gr_mode = final_greedy[:, 5].mode().values.item()
        final_match = (final_greedy == demo_acts).all(dim=-1).float().mean().item()

    print(f"  Z mode:      {z_mode:.0f} (target: 5)  "
          f"{'✓' if abs(z_mode - 5) < 1 else '✗'}")
    print(f"  Gripper mode:{gr_mode:.0f} (target: 5)  "
          f"{'✓' if abs(gr_mode - 5) < 1 else '✗'}")
    print(f"  Final Match: {final_match:.1%}  "
          f"({'✓ >50%' if final_match > 0.5 else '✗ <50% — GoalEncoder may be destabilizing'})")

    if aux_history:
        print(f"  Aux loss:    {aux_history[0]:.4f} → {aux_history[-1]:.4f}")

    passed = nll_decreased and final_match > 0.5
    print(f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}: "
          f"{'GoalEncoder + LSTM + PI encoder works with ABC!' if passed else 'GoalEncoder destabilized training'}")

    if not passed and final_match < 0.3:
        print("  → TIP: Try reducing aux_coef or further shrinking goal_proj init scale")

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
