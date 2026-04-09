"""
Visual ABC (Alice Behavioral Cloning) Loss Test
================================================

Spawns TWO Isaac Lab environments side-by-side (arm, table, objects):

  Env 0 (LEFT)  — "Alice": Replays a hard-coded trajectory every episode.
                   The arm sweeps LEFT→RIGHT (increasing X) while staying at
                   constant height, gripper open. This is the REFERENCE.

  Env 1 (RIGHT) — "Bob":   Runs Bob's policy, which starts RANDOM and is
                   trained via pure ABC loss on Alice's trajectory between
                   episodes. Over 30–50 iterations, Bob's movement should
                   converge to match Alice's exactly.

At the end of training, both environments replay one final episode
simultaneously.  If ABC works correctly, both arms should move IDENTICALLY.

Usage:
    # From the project root (NOT headless — the whole point is visual!)
    python asyncDualPlayPPO/tests/test_abc.py --num_iterations 50

    # Quick smoke test (fewer iterations, won't fully converge)
    python asyncDualPlayPPO/tests/test_abc.py --num_iterations 10
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
    """Filter C-level stderr noise (URDF warnings, Lula, carb) via a pipe thread."""
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
        description="Visual ABC Test — two envs side by side"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of train-then-replay iterations (default: 50)",
    )
    parser.add_argument(
        "--episode_steps",
        type=int,
        default=80,
        help="Steps per replay episode (default: 80)",
    )
    parser.add_argument(
        "--abc_epochs",
        type=int,
        default=1,
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
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer

    torch.manual_seed(args.seed)

    # ── Load PPO config ─────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "..", "cfg", "ppo", "ppo_continuous.yaml")
    with open(cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    pol_cfg = ppo_cfg["params"]["policy"]
    num_cat_dims = pol_cfg.get("num_cat_dims", 6)
    num_bins = pol_cfg.get("num_bins", 11)

    # ── Create Environment (2 envs, side by side) ───────────────
    print("\n" + "=" * 70)
    print("  VISUAL ABC TEST")
    print("  Env 0 (left):  Alice — hard-coded trajectory (reference)")
    print("  Env 1 (right): Bob   — policy trained via ABC (should converge)")
    print("=" * 70)

    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.scene.env_spacing = 3.0  # wider spacing so both arms visible

    print("\nCreating environment (2 envs)...")
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device
    print(f"  Device: {device}")

    # ── Observation Dimensions ──────────────────────────────────
    # Alice obs: ee_pose(6) + gripper(1) + obj1_state(14) + obj2_state(14) = 35
    # Bob obs:   ee_pose(6) + gripper(1) + [obj_state(14)+goal(6)+dist(2)]×2 = 51
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

    # ── Build Bob's ActorCritic (SIMPLE MLP for test isolation) ──
    # Disable PI encoder, goal encoder, and LSTM to eliminate
    # architectural noise. Pure MLP → clean gradient flow → verifies
    # that the ABC loss math itself is correct.
    model_cfg = copy.deepcopy(pol_cfg)
    model_cfg["use_goal_encoder"] = False
    model_cfg["use_pi_encoder"] = False
    model_cfg["use_lstm"] = False
    model_cfg["use_multicategorical"] = True  # keep MC (paper's action space)
    # Smaller MLP for faster convergence on 80-step trajectory
    model_cfg["pi_hid_sizes"] = [256, 128]
    model_cfg["vf_hid_sizes"] = [256, 128]

    ac = ActorCritic(
        obs_shape=(bob_obs_dim,),
        states_shape=(bob_obs_dim,),
        actions_shape=(num_cat_dims,),
        initial_std=ppo_cfg["params"]["learn"].get("init_noise_std", 1.0),
        model_cfg=model_cfg,
        asymmetric=False,
    ).to(device)

    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    # ── ABC Demo Buffer ─────────────────────────────────────────
    abc_buffer = GPUDemonstrationBuffer(
        capacity=100_000,
        obs_shape=(bob_obs_dim,),
        states_shape=(bob_obs_dim,),
        actions_shape=(num_cat_dims,),
        device=device,
    )

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

    # ── Alice's hard-coded trajectory ───────────────────────────
    # Design: GENTLE sweep so the arm stays in bounds (no terminations)
    #
    # Bin layout (11 bins): 0=full-negative, 5=neutral, 10=full-positive
    #   X (dim 0): linearly ramp 4→7 over episode_steps  (gentle L→R)
    #   Y (dim 1): hold at 5                               (no lateral)
    #   Z (dim 2): hold at 5                               (same height)
    #   Rx (dim 3): hold at 5                              (no rotation)
    #   Ry (dim 4): hold at 5                              (no rotation)
    #   Gripper (dim 5): hold at 5                         (neutral/hold)
    N = args.episode_steps
    alice_bins = torch.zeros(N, num_cat_dims, device=device)
    alice_bins[:, 0] = torch.linspace(4, 7, N, device=device).round()  # X: gentle L→R
    alice_bins[:, 1] = 5.0  # Y: neutral
    alice_bins[:, 2] = 5.0  # Z: same height
    alice_bins[:, 3] = 5.0  # Rx: neutral
    alice_bins[:, 4] = 5.0  # Ry: neutral
    alice_bins[:, 5] = 5.0  # Gripper: neutral (hold)

    print(f"\n  Alice trajectory ({N} steps):")
    print(f"    X bins (first 10): {alice_bins[:20, 0].long().tolist()}")
    print(f"    Z bin (constant):  {alice_bins[0, 2].long().item()}")
    print(f"    Gripper bin:       {alice_bins[0, 5].long().item()} (open)")

    # Pre-convert Alice's full trajectory to env actions (for env 0)
    alice_gripper_traj = torch.ones(1, 1, device=device)  # starts open
    alice_env_actions = []
    for t in range(N):
        act_t, alice_gripper_traj = bins_to_env_action(
            alice_bins[t:t+1], alice_gripper_traj
        )
        alice_env_actions.append(act_t.squeeze(0))  # (7,)
    alice_env_actions = torch.stack(alice_env_actions)  # (N, 7)

    # ── Attach dummy episode_manager for goal_states obs term ───
    # The observation manager's goal_states term reads env.episode_manager.
    # We create a minimal stub that returns zeros (no goal during this test).
    class _DummyEpisodeManager:
        def __init__(self, num_envs, device):
            self.goal_states = torch.zeros(num_envs, 12, device=device)
            self.goal_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.pos_threshold = 0.04
            self.rot_threshold = 0.5
    base_env.episode_manager = _DummyEpisodeManager(2, device)

    # ── Optional video recorder ──────────────────────────────────
    recorder = None
    video_dir = os.path.join(script_dir, "videos")
    if args.record_video:
        e0 = base_env.scene.env_origins[0].cpu().tolist()
        e1 = base_env.scene.env_origins[1].cpu().tolist()
        mid_x = (e0[0] + e1[0]) / 2
        mid_y = (e0[1] + e1[1]) / 2
        cam_pos = (mid_x, mid_y + 5.5, 1.5)
        look_at = (mid_x, mid_y + 0.5, 0.5)
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
        recorder.stop_and_save(os.path.join(video_dir, "start_random_abc.mp4"))

    # ══════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Training for {args.num_iterations} iterations "
          f"({args.abc_epochs} ABC gradient step(s) each, raw NLL loss)...")
    print(f"\n  {'Iter':>6} | {'NLL':>10} | {'X mode':>8} | {'Z mode':>8} | {'Gr mode':>8} | {'Match%':>8}")
    print("  " + "-" * 65)

    nll_history = []

    for it in range(1, args.num_iterations + 1):
        # ── Phase A: Replay episode (Alice on env 0, Bob on env 1) ──
        obs_dict, info = base_env.reset()
        alice_obs_all = obs_dict["alice_policy"]  # (2, alice_obs_dim)
        bob_obs_all = obs_dict["bob_policy"]      # (2, bob_obs_dim)

        bob_gripper = torch.ones(1, 1, device=device)

        # Collect Alice's (bob-format obs, action) pairs for ABC
        demo_obs_list = []
        demo_act_list = []
        episode_corrupted = False  # flag if env 0 terminates

        for t in range(N):
            # ── Construct per-env actions ──
            # Env 0: Alice's hard-coded action
            alice_act = alice_env_actions[t].unsqueeze(0)  # (1, 7)

            # Env 1: Bob's current policy
            bob_obs_t = bob_obs_all[1:2]  # (1, bob_obs_dim)
            with torch.no_grad():
                bob_bins, _, _, _, _ = ac.act(bob_obs_t, None)
                bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

            # Combine: (2, 7) actions
            combined_action = torch.cat([alice_act, bob_act], dim=0)

            # ── Collect demo data (env 0's bob-obs + Alice's bin actions) ──
            # Only collect if env 0 hasn't been reset mid-episode
            if not episode_corrupted:
                demo_obs_list.append(bob_obs_all[0:1].clone())  # env 0's bob obs
                demo_act_list.append(alice_bins[t:t+1].clone())   # Alice's bin indices

            # ── Step both environments ──
            obs_dict, rewards_t, terminated, truncated, extras = base_env.step(
                combined_action
            )
            alice_obs_all = obs_dict["alice_policy"]
            bob_obs_all = obs_dict["bob_policy"]

            # Handle physics terminations — skip demo collection after reset
            dones = terminated | truncated
            if dones.any():
                if dones[0].item():
                    episode_corrupted = True
                    print(f"  [iter {it}] Env 0 terminated at step {t} — skipping demo")
                obs_dict, _ = base_env.reset()
                alice_obs_all = obs_dict["alice_policy"]
                bob_obs_all = obs_dict["bob_policy"]

        # ── Phase B: Train Bob via ABC (raw NLL) ──
        if episode_corrupted or len(demo_obs_list) < 10:
            # Skip training if env 0 was reset (corrupted obs sequence)
            nll_history.append(nll_history[-1] if nll_history else 20.0)
            continue

        demo_obs = torch.cat(demo_obs_list, dim=0)    # (<=N, bob_obs_dim)
        demo_acts = torch.cat(demo_act_list, dim=0)    # (<=N, num_cat_dims)
        T = demo_obs.shape[0]

        # Raw NLL loss: direct -log_prob(alice_actions | obs)
        # Much more stable than clipped ratio for pure BC
        for _ in range(args.abc_epochs):
            log_probs, _, _, _, _ = ac.evaluate(demo_obs, None, demo_acts)
            bc_loss = -log_probs.mean()  # raw NLL

            optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()

        # ── Evaluate convergence ──
        with torch.no_grad():
            eval_lp, _, _, _, _ = ac.evaluate(demo_obs, None, demo_acts)
            raw_nll = -eval_lp.mean().item()
            greedy = ac.act_inference(demo_obs)
            x_mode = greedy[:, 0].mode().values.item()
            z_mode = greedy[:, 2].mode().values.item()
            gr_mode = greedy[:, 5].mode().values.item()
            all_match = (
                (greedy == demo_acts).all(dim=-1).float().mean().item()
            )
        nll_history.append(raw_nll)

        if it % 5 == 0 or it == 1:
            print(
                f"  {it:>6} | {raw_nll:>+10.3f} | {x_mode:>8.0f} | "
                f"{z_mode:>8.0f} | {gr_mode:>8.0f} | {all_match:>7.1%}"
            )

    # ══════════════════════════════════════════════════════════════
    # FINAL VERIFICATION EPISODE
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
        # Env 0: Alice
        alice_act = alice_env_actions[t].unsqueeze(0)

        # Env 1: Bob (greedy / deterministic)
        bob_obs_t = bob_obs_all[1:2]
        with torch.no_grad():
            bob_bins = ac.act_inference(bob_obs_t)
            bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

        # Measure difference between Alice and Bob env actions
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
        recorder.stop_and_save(os.path.join(video_dir, "end_converged_abc.mp4"))

    # ── Results ──
    mean_diff = np.mean(action_diffs)
    final_nll = nll_history[-1] if nll_history else float("inf")
    nll_decreased = len(nll_history) >= 2 and nll_history[-1] < nll_history[0]

    print(f"\n  NLL:         {nll_history[0]:+.3f} → {final_nll:+.3f}  "
          f"({'✓ decreased' if nll_decreased else '✗ DID NOT decrease'})")
    print(f"  Action diff: {mean_diff:.4f}  "
          f"({'✓ small' if mean_diff < 0.3 else '✗ large — ABC did not converge'})")

    with torch.no_grad():
        final_greedy = ac.act_inference(demo_obs)
        z_mode = final_greedy[:, 2].mode().values.item()
        gr_mode = final_greedy[:, 5].mode().values.item()

    print(f"  Z mode:      {z_mode:.0f} (target: 5)  "
          f"{'✓' if abs(z_mode - 5) < 1 else '✗'}")
    print(f"  Gripper mode:{gr_mode:.0f} (target: 5)  "
          f"{'✓' if abs(gr_mode - 5) < 1 else '✗'}")

    passed = nll_decreased and mean_diff < 0.5
    print(f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}: "
          f"{'Bob successfully replicated Alice trajectory' if passed else 'ABC did not converge — check gradient flow'}")

    if passed:
        print("  → Watch the viewport: both arms should be moving the same way!")

    # Keep the simulation running for visual inspection
    print("\n  Replaying final episode continuously for visual inspection...")
    print("  (Close the window or Ctrl+C to exit)\n")

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
