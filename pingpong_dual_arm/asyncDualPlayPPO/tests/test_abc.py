"""
Visual ABC (Alice Behavioral Cloning) Loss Test
================================================

Spawns TWO Isaac Lab environments side-by-side (arm, table, objects):

  Env 0 (LEFT)  — "Alice": Replays a hard-coded trajectory every episode.
                   Arm sweeps gently forward / upward then holds, gripper neutral.
                   This is the REFERENCE — Bob must clone it exactly.

  Env 1 (RIGHT) — "Bob":   Runs Bob's policy, which starts RANDOM and is
                   trained via pure ABC (NLL) loss on Alice's trajectory.
                   Over iterations, Bob's movement should converge to match
                   Alice's exactly — demonstrating that ABC alone is enough
                   to copy a trajectory.

Convergence criterion: rolling Match% over the last `--success_window`
episodes must reach `--success_threshold`.  When converged, a video is
recorded automatically (no need for --record_video).

Usage (from project root — NOT headless):
    python asyncDualPlayPPO/tests/test_abc.py --max_iterations 500

Expected:
    - NLL should steadily decrease
    - Match% should reach >80% within a few hundred iterations
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
    """Captures rendered frames via omni.replicator and encodes to MP4 (1080p).

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
        resolution: tuple = (1920, 1080),  # HD
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
        description="Visual ABC Test — two envs side by side, pure NLL behavioural cloning"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="Safety cap on training iterations (default: 500)",
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
        help="Rolling window size for Match%% success rate (default: 10)",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.8,
        help="Rolling Match%% to consider policy converged (default: 0.8)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=20,
        help="Frozen eval episodes after training converges (default: 20)",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Also record the START (random Bob) episode; convergence episode is always recorded",
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
    print("  VISUAL ABC TEST  (pure NLL behavioural cloning)")
    print("  Env 0 (left):  Alice — hard-coded trajectory (reference)")
    print("  Env 1 (right): Bob   — simple MLP trained via ABC (should converge)")
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

    # ── Build Bob's ActorCritic (SIMPLE MLP — test isolation) ───
    # Disable PI encoder, goal encoder, and LSTM so we isolate the ABC
    # loss itself.  A simple MLP has a clear gradient path and should
    # converge quickly, proving that pure NLL BC can copy a trajectory.
    model_cfg = copy.deepcopy(pol_cfg)
    model_cfg["use_goal_encoder"] = False
    model_cfg["use_pi_encoder"] = False
    model_cfg["use_lstm"] = False
    model_cfg["use_multicategorical"] = True
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

    # ── ABC Demo Buffer (kept for compatibility; training uses fresh lists) ─
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
    # Identical to test_abc_goal_encoder.py so both tests exercise the
    # same motion and results are directly comparable.
    #
    # Bin layout (11 bins): 0=full-negative, 5=neutral, 10=full-positive
    #   X (dim 0): hold at 5                               (no lateral)
    #   Y (dim 1): ramp 5→6 for first half, hold 7 after  (gentle forward)
    #   Z (dim 2): ramp 3→5 for first half, hold 5 after  (gentle lift)
    #   Rx (dim 3): hold at 5                              (no rotation)
    #   Ry (dim 4): hold at 5                              (no rotation)
    #   Gripper (dim 5): hold at 5                         (neutral / hold)
    N = args.episode_steps
    alice_bins = torch.zeros(N, num_cat_dims, device=device)
    alice_bins[:, 0] = 5.0  # X: neutral

    PUSH_STEP = int(N * (1 / 2))
    alice_bins[:PUSH_STEP, 1] = torch.linspace(5, 6, PUSH_STEP, device=device).round()
    alice_bins[PUSH_STEP:, 1] = 7.0

    DROP_STEP = int(N * (1 / 2))
    alice_bins[:DROP_STEP, 2] = torch.linspace(
        3, 5, DROP_STEP, device=device
    ).round()  # Z: gentle lift
    alice_bins[DROP_STEP:, 2] = 5.0

    alice_bins[:, 3] = 5.0  # Rx: neutral
    alice_bins[:, 4] = 5.0  # Ry: neutral
    alice_bins[:, 5] = 5.0  # Gripper: neutral

    print(f"\n  Alice trajectory ({N} steps):")
    print(f"    X bins (first 20): {alice_bins[:20, 0].long().tolist()}")
    print(f"    Y bins (first 20): {alice_bins[:20, 1].long().tolist()}")
    print(f"    Z bins (first 20): {alice_bins[:20, 2].long().tolist()}")
    print(f"    Gripper bin:       {alice_bins[0, 5].long().item()} (neutral)")

    # Pre-convert Alice's full trajectory to env actions (for env 0)
    alice_gripper_traj = torch.ones(1, 1, device=device)
    alice_env_actions = []
    for t in range(N):
        act_t, alice_gripper_traj = bins_to_env_action(
            alice_bins[t : t + 1], alice_gripper_traj
        )
        alice_env_actions.append(act_t.squeeze(0))  # (7,)
    alice_env_actions = torch.stack(alice_env_actions)  # (N, 7)

    # ── Attach dummy episode_manager for goal_states obs term ───
    class _DummyEpisodeManager:
        def __init__(self, num_envs, device):
            self.goal_states = torch.zeros(num_envs, 12, device=device)
            self.goal_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.pos_threshold = 0.04
            self.rot_threshold = 0.5

    base_env.episode_manager = _DummyEpisodeManager(2, device)

    # ── Video recorder (always created, 1080p HD) ────────────────
    video_dir = os.path.join(script_dir, "videos")
    e0 = base_env.scene.env_origins[0].cpu().tolist()
    e1 = base_env.scene.env_origins[1].cpu().tolist()
    mid_x = (e0[0] + e1[0]) / 2
    mid_y = (e0[1] + e1[1]) / 2
    cam_pos = (mid_x, mid_y + 5.5, 1.5)
    look_at = (mid_x, mid_y + 0.5, 0.5)
    recorder = _VideoRecorder(cam_pos=cam_pos, look_at=look_at)
    print(f"\n  Recorder ready (1080p) → {video_dir}/")
    print(f"    Camera: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
    print(f"    Look-at: ({look_at[0]:.2f}, {look_at[1]:.2f}, {look_at[2]:.2f})")

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
        recorder.stop_and_save(os.path.join(video_dir, "start_random_abc.mp4"))

    # ══════════════════════════════════════════════════════════════
    # TRAINING LOOP  (runs until Bob's Match% converges)
    # ══════════════════════════════════════════════════════════════
    print(
        f"\n  Training until rolling Match%% >= {args.success_threshold:.0%} "
        f"over last {args.success_window} episodes"
    )
    print(f"  max_iterations={args.max_iterations} | abc_epochs={args.abc_epochs}")
    print(
        f"\n  {'Iter':>6} | {'NLL':>10} | {'X mode':>8} | {'Y mode':>8} | "
        f"{'Z mode':>8} | {'Gr mode':>8} | {'Match%':>8} | {'WinRate':>8} | {'Status':>10}"
    )
    print("  " + "-" * 95)

    nll_history = []
    it = 0
    converged = False
    convergence_iter = None
    success_window = deque(maxlen=args.success_window)  # True/False per episode

    while it < args.max_iterations and not converged:
        it += 1

        # ── Phase A: Replay episode (Alice on env 0, Bob on env 1) ──
        obs_dict, info = base_env.reset()
        alice_obs_all = obs_dict["alice_policy"]
        bob_obs_all = obs_dict["bob_policy"]

        bob_gripper = torch.ones(1, 1, device=device)

        demo_obs_list = []   # env-0 bob obs (Alice's viewpoint) — training target
        bob_obs_list = []    # env-1 bob obs (Bob's actual state) — match% eval
        demo_act_list = []
        episode_corrupted = False
        bob_terminated_early = False

        for t in range(N):
            alice_act = alice_env_actions[t].unsqueeze(0)  # (1, 7)

            bob_obs_t = bob_obs_all[1:2]  # (1, bob_obs_dim)
            with torch.no_grad():
                bob_bins, _, _, _, _ = ac.act(bob_obs_t, None)
                bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

            combined_action = torch.cat([alice_act, bob_act], dim=0)

            if not episode_corrupted:
                demo_obs_list.append(bob_obs_all[0:1].clone())   # Alice's obs
                bob_obs_list.append(bob_obs_all[1:2].clone())    # Bob's obs
                demo_act_list.append(alice_bins[t : t + 1].clone())

            obs_dict, _, terminated, truncated, _ = base_env.step(combined_action)
            alice_obs_all = obs_dict["alice_policy"]
            bob_obs_all = obs_dict["bob_policy"]

            dones = terminated | truncated
            if dones.any():
                if dones[0].item():
                    episode_corrupted = True
                    print(f"  [iter {it}] Env 0 terminated at step {t} — skipping demo")
                if dones[1].item():
                    bob_terminated_early = True
                obs_dict, _ = base_env.reset()
                alice_obs_all = obs_dict["alice_policy"]
                bob_obs_all = obs_dict["bob_policy"]

        # ── Phase B: Train Bob via ABC (raw NLL) ──
        if episode_corrupted or len(demo_obs_list) < 10:
            nll_history.append(nll_history[-1] if nll_history else 20.0)
            continue

        demo_obs = torch.cat(demo_obs_list, dim=0)   # (T, bob_obs_dim)
        bob_obs_ep = torch.cat(bob_obs_list, dim=0)  # (T, bob_obs_dim)
        demo_acts = torch.cat(demo_act_list, dim=0)  # (T, num_cat_dims)
        T = demo_obs.shape[0]

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

            # Greedy actions from Bob's ACTUAL env-1 observations
            # (generalization metric — does Bob match Alice from his own state?)
            greedy = ac.act_inference(bob_obs_ep)  # (T, num_cat_dims)
            x_mode = greedy[:, 0].mode().values.item()
            y_mode = greedy[:, 1].mode().values.item()
            z_mode = greedy[:, 2].mode().values.item()
            gr_mode = greedy[:, 5].mode().values.item()
            all_match = (greedy == demo_acts).all(dim=-1).float().mean().item()

        nll_history.append(raw_nll)

        # ── Convergence check (rolling Match% window) ──
        # A "success" means the greedy Match% for this episode >= 0.5
        # (per-step all-dim match is strict; 50% is a reasonable episode threshold)
        episode_success = all_match >= 0.5
        success_window.append(episode_success)
        win_rate = sum(success_window) / len(success_window) if success_window else 0.0

        if (
            len(success_window) == args.success_window
            and win_rate >= args.success_threshold
        ):
            converged = True
            convergence_iter = it

        status = "CONVERGED" if converged else ("MATCH" if episode_success else "      ")

        if it % 5 == 0 or it == 1 or episode_success or converged:
            print(
                f"  {it:>6} | {raw_nll:>+10.3f} | {x_mode:>8.0f} | {y_mode:>8.0f} | "
                f"{z_mode:>8.0f} | {gr_mode:>8.0f} | {all_match:>8.1%} | "
                f"{win_rate:>8.1%} | {status}"
            )

    # ══════════════════════════════════════════════════════════════
    # RESULT SUMMARY
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
    print(f"{'=' * 70}")

    # ══════════════════════════════════════════════════════════════
    # FROZEN EVALUATION PHASE
    # ══════════════════════════════════════════════════════════════
    print(
        f"\n  Running frozen eval: {args.eval_episodes} episodes (no gradient updates)..."
    )
    eval_successes = 0
    eval_match_rates = []

    for ep in range(args.eval_episodes):
        obs_dict_e, _ = base_env.reset()
        bob_obs_e = obs_dict_e["bob_policy"]
        bob_gripper_e = torch.ones(1, 1, device=device)

        ep_obs_list = []
        ep_act_list = []

        for t in range(N):
            alice_act_e = alice_env_actions[t].unsqueeze(0)
            with torch.no_grad():
                bob_bins_e, _, _, _, _ = ac.act(bob_obs_e[1:2], None)
                bob_act_e, bob_gripper_e = bins_to_env_action(
                    bob_bins_e, bob_gripper_e
                )

            ep_obs_list.append(bob_obs_e[1:2].clone())
            ep_act_list.append(alice_bins[t : t + 1].clone())

            combined_e = torch.cat([alice_act_e, bob_act_e], dim=0)
            obs_dict_e, _, term_e, trunc_e, _ = base_env.step(combined_e)
            bob_obs_e = obs_dict_e["bob_policy"]

            if (term_e | trunc_e).any():
                obs_dict_e, _ = base_env.reset()
                bob_obs_e = obs_dict_e["bob_policy"]

        if ep_obs_list:
            with torch.no_grad():
                ep_obs_t = torch.cat(ep_obs_list, dim=0)
                ep_act_t = torch.cat(ep_act_list, dim=0)
                ep_greedy = ac.act_inference(ep_obs_t)
                ep_match = (ep_greedy == ep_act_t).all(dim=-1).float().mean().item()
            eval_match_rates.append(ep_match)
            if ep_match >= 0.5:
                eval_successes += 1

        if (ep + 1) % 5 == 0 or (ep + 1) == args.eval_episodes:
            avg_match = (
                sum(eval_match_rates) / len(eval_match_rates)
                if eval_match_rates else 0.0
            )
            print(
                f"    Ep {ep+1:>3}/{args.eval_episodes}  "
                f"success={eval_successes}/{ep+1}  "
                f"({eval_successes/(ep+1):.1%})  "
                f"avg_match={avg_match:.1%}"
            )

    final_success_rate = eval_successes / args.eval_episodes if args.eval_episodes else 0.0
    avg_match_overall = (
        sum(eval_match_rates) / len(eval_match_rates) if eval_match_rates else 0.0
    )

    print(f"\n{'=' * 70}")
    print(f"  EVAL RESULT over {args.eval_episodes} episodes:")
    print(
        f"    Success rate (Match%>=50%): {final_success_rate:.1%}  "
        f"({eval_successes}/{args.eval_episodes})"
    )
    print(f"    Avg Match%%: {avg_match_overall:.1%}")
    print(f"{'=' * 70}")

    # ══════════════════════════════════════════════════════════════
    # RECORD CONVERGED (or final) EPISODE  — always, 1080p HD
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Recording {'converged' if converged else 'final'} episode (1080p)...")
    recorder.start()
    obs_dict, _ = base_env.reset()
    bob_obs_all = obs_dict["bob_policy"]
    bob_gripper = torch.ones(1, 1, device=device)

    action_diffs = []
    bob_bins_traj = []

    for t in range(N):
        alice_act = alice_env_actions[t].unsqueeze(0)
        bob_obs_t = bob_obs_all[1:2]
        with torch.no_grad():
            bob_bins = ac.act_inference(bob_obs_t)
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
        f"converged_iter{convergence_iter}_abc.mp4"
        if converged
        else f"final_iter{it}_abc.mp4"
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

    mean_diff = np.mean(action_diffs) if action_diffs else float("inf")
    print(f"\n  Mean action diff: {mean_diff:.4f}")
    passed = nll_decreased and mean_diff < 0.5
    print(
        f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}: "
        f"{'Bob successfully replicated Alice trajectory' if passed else 'ABC did not converge — check gradient flow'}"
    )
    if passed:
        print("  → Watch the viewport: both arms should be moving the same way!")

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
