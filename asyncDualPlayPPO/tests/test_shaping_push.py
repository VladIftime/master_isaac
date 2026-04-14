"""
Test 7: Potential-Based Shaping + ABC with Push Trajectory (Visual, Isaac Sim)
==============================================================================

Tests that Bob can learn to replicate a pre-defined push trajectory
(cube pushed to a new valid goal position) via sequential LSTM ABC.

Layout:
  - Env 0 (LEFT):  Bob — trained via ABC behavioral cloning
  - Env 1 (RIGHT): Demo — hard-coded push trajectory that physically sweeps the cube

The push trajectory (RIGHT, env 1) is designed in three phases:
  Phase 1 (0-59 steps):   Approach — EE descends toward the cube
  Phase 2 (60-119 steps): Push     — EE drives in +X, sweeping cube ≥7 cm (valid goal)
  Phase 3 (120-149 steps): Retreat — EE rises and returns to neutral height

Goal states are set to reflect the cube's pushed position (~0.20m X displacement
from default), which satisfies the valid-goal pos_threshold of 0.07m.

The potential-based shaping reward in wrapper.py provides per-step guidance during
full training; this test isolates the ABC convergence signal and verifies that
Bob can learn to imitate the push.

Usage:
    cd asyncDualPlayPPO
    python tests/test_shaping_push.py --num_iterations 200

Expected:
    - NLL should steadily decrease
    - Match% should reach >50% within 200 iterations
    - RIGHT arm should visually show cube being swept
    - LEFT arm (Bob) should converge to replicate the push motion
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
        description="Test 7: Potential-Based Shaping + ABC — Push Trajectory"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=200,
        help="Training iterations (default: 200)",
    )
    parser.add_argument(
        "--episode_steps",
        type=int,
        default=150,
        help="Steps per episode (default: 150, matches push trajectory length)",
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
    print("  TEST 7: Potential-Based Shaping + ABC — Push Trajectory")
    print("  Full architecture: PI encoder + Goal encoder + LSTM")
    print("  Env 0 (LEFT):  Bob  — trained via ABC behavioral cloning")
    print("  Env 1 (RIGHT): Demo — hard-coded push that sweeps the cube")
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

    # ── Build Bob's ActorCritic (full production architecture) ──
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

    # "Initialization Jump" prevention — same as in test_abc_goal_encoder.py
    if ac._goal_proj is not None:
        with torch.no_grad():
            ac._goal_proj.weight.mul_(0.01 / 0.5)
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

    # ── Push trajectory (env 1 / RIGHT) ─────────────────────────
    # Bin layout (0-10): center=5 is neutral, >5 = positive, <5 = negative.
    #
    # Strategy: first move the arm LEFT (X-) and descend to be left of the
    # cube, then sweep RIGHT (X+) so the arm sweeps THROUGH the cube's
    # position and physically pushes it.  This avoids the failure mode where
    # the arm starts to the right of the cube and sweeping right misses it.
    #
    # Phase 1 (steps 0-59):   Position left + descend
    #   - X: negative (3)  — move EE left to be left of cube
    #   - Y: neutral (5)   — stay at current Y
    #   - Z: descent (3)   — lower to cube contact height
    #   - Rx,Ry: neutral (5)
    #   - Gripper: open (5)
    #
    # Phase 2 (steps 60-119): Sweep right — push cube in +X
    #   - X: positive (8)  — sweep EE rightward through cube position
    #   - Y: neutral (5)
    #   - Z: hold low (4)  — maintain contact height
    #   - Rx,Ry: neutral (5)
    #   - Gripper: open (5) — flat surface push, no grip needed
    #
    # Phase 3 (steps 120-149): Retreat upward
    #   - X: neutral (5)
    #   - Y: neutral (5)
    #   - Z: rise (7)      — lift EE clear of cube
    #   - Rx,Ry: neutral (5)
    #   - Gripper: open (5)
    N = args.episode_steps  # 150

    push_bins = torch.zeros(N, num_cat_dims, device=device)

    # Phase 1: position left + descend (steps 0-59)
    push_bins[:60, 0] = 3  # X: negative — move arm to left of cube
    push_bins[:60, 1] = 5  # Y: neutral
    push_bins[:60, 2] = 3  # Z: descend aggressively to contact height
    push_bins[:60, 3] = 5  # Rx: neutral
    push_bins[:60, 4] = 5  # Ry: neutral
    push_bins[:60, 5] = 5  # Gripper: open

    # Phase 2: sweep right through cube (steps 60-119)
    push_bins[60:120, 0] = 8  # X: positive — sweep through cube, pushing it right
    push_bins[60:120, 1] = 5  # Y: neutral
    push_bins[60:120, 2] = 4  # Z: hold at contact height
    push_bins[60:120, 3] = 5  # Rx: neutral
    push_bins[60:120, 4] = 5  # Ry: neutral
    push_bins[60:120, 5] = 5  # Gripper: open (EE face pushes cube)

    # Phase 3: retreat up (steps 120-149)
    push_bins[120:, 0] = 5  # X: neutral
    push_bins[120:, 1] = 5  # Y: neutral
    push_bins[120:, 2] = 7  # Z: rise clear
    push_bins[120:, 3] = 5  # Rx: neutral
    push_bins[120:, 4] = 5  # Ry: neutral
    push_bins[120:, 5] = 5  # Gripper: open

    print(f"\n  Push trajectory ({N} steps, 3 phases):")
    print(
        f"    Phase 1 (0-59):   Position left + descend — X={push_bins[0,0].int()}, Z={push_bins[0,2].int()}"
    )
    print(f"    Phase 2 (60-119): Sweep right through cube — X={push_bins[60,0].int()}")
    print(
        f"    Phase 3 (120-149): Retreat up              — Z={push_bins[120,2].int()}"
    )

    # Pre-convert push trajectory to env actions
    demo_gripper_traj = torch.ones(1, 1, device=device)
    demo_env_actions = []
    for t in range(N):
        act_t, demo_gripper_traj = bins_to_env_action(
            push_bins[t : t + 1], demo_gripper_traj
        )
        demo_env_actions.append(act_t.squeeze(0))
    demo_env_actions = torch.stack(demo_env_actions)  # (N, env_action_dim)

    # ── Goal states — pushed cube position ──────────────────────
    # Goal layout: [obj1_pos(3), obj1_euler(3), obj2_pos(3), obj2_euler(3)] = 12D
    #
    # Cube (obj2) is pushed ~0.20m in +X from its default position.
    # Default cube X ≈ -0.25 → pushed cube X ≈ -0.05  (Δ = 0.20m > 0.07m threshold)
    # Target object (obj1) stays near its default position (not being pushed).
    #
    # These values match the LOCAL-frame Euler convention used by
    # wrapper.py / episode_manager.goal_states (12D, pos+euler per object).
    class _GoalEpisodeManager:
        def __init__(self, num_envs, device):
            self.goal_states = torch.zeros(num_envs, 12, device=device)
            # Object 1 (target_object): stays near default
            self.goal_states[:, 0] = -0.25  # x  — unchanged
            self.goal_states[:, 1] = 0.70  # y  — unchanged
            self.goal_states[:, 2] = 0.05  # z
            # Euler angles for obj1: all zero (no rotation goal)

            # Object 2 (cube): pushed ~0.20m in +X
            self.goal_states[:, 6] = -0.05  # x  — pushed from -0.25 to -0.05
            self.goal_states[:, 7] = 0.65  # y  — slight Y change
            self.goal_states[:, 8] = 0.05  # z
            # Euler angles for obj2: all zero (no rotation goal)

            self.goal_valid = torch.ones(num_envs, dtype=torch.bool, device=device)
            self.pos_threshold = 0.04
            self.rot_threshold = 0.5

    base_env.episode_manager = _GoalEpisodeManager(2, device)

    goal_mgr = base_env.episode_manager
    cube_disp = float(
        torch.norm(
            goal_mgr.goal_states[0, 6:9]
            - torch.tensor([-0.25, 0.70, 0.05], device=device)
        )
    )
    print(f"\n  Goal: cube displaced {cube_disp:.3f} m from default")
    print(
        f"    Default: [-0.25, 0.70]  →  Goal: [{goal_mgr.goal_states[0,6]:.2f}, {goal_mgr.goal_states[0,7]:.2f}]"
    )
    print(
        f"    Valid goal? {'YES ✓' if cube_disp > 0.07 else 'NO ✗'} (threshold: 0.07m)"
    )

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
            # Env 1 (right): demo push
            demo_act_rec = demo_env_actions[t].unsqueeze(0)
            # Env 0 (left): untrained Bob
            with torch.no_grad():
                bob_bins_rec, _, _, _, _ = ac.act(bob_obs_rec[0:1], None)
                bob_act_rec, bob_gripper_rec = bins_to_env_action(
                    bob_bins_rec, bob_gripper_rec
                )
            # Action order: [env0 (Bob), env1 (demo)]
            combined_rec = torch.cat([bob_act_rec, demo_act_rec], dim=0)
            obs_dict_rec, _, term_rec, trunc_rec, _ = base_env.step(combined_rec)
            recorder.capture()
            bob_obs_rec = obs_dict_rec["bob_policy"]
            if (term_rec | trunc_rec).any():
                obs_dict_rec, _ = base_env.reset()
                bob_obs_rec = obs_dict_rec["bob_policy"]
        recorder.stop_and_save(os.path.join(video_dir, "start_random_shaping_push.mp4"))

    # ══════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════
    print(
        f"\n  Training for {args.num_iterations} iterations "
        f"(raw NLL + aux_coef={aux_coef})..."
    )
    print(
        f"\n  {'Iter':>6} | {'NLL':>8} | {'Aux':>8} | {'X md':>6} | "
        f"{'Y md':>6} | {'Gr md':>6} | {'Match%':>8}"
    )
    print("  " + "-" * 65)

    nll_history = []
    aux_history = []

    for it in range(1, args.num_iterations + 1):
        # ── Phase A: Replay episode ──
        obs_dict, info = base_env.reset()
        bob_obs_all = obs_dict["bob_policy"]

        bob_gripper = torch.ones(1, 1, device=device)
        demo_obs_list = []  # observations from env 1 (RIGHT / demo side)
        demo_act_list = []  # push bin actions (ground truth)
        episode_corrupted = False

        for t in range(N):
            # Env 1 (right): demo push action
            demo_act = demo_env_actions[t].unsqueeze(0)

            # Env 0 (left): Bob's current policy
            bob_obs_t = bob_obs_all[0:1]
            with torch.no_grad():
                bob_bins, _, _, _, _ = ac.act(bob_obs_t, None)
                bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

            # Action tensor: [env0 (Bob), env1 (demo)]
            combined_action = torch.cat([bob_act, demo_act], dim=0)

            # Collect demo: env 1's current observation + the push bins
            if not episode_corrupted:
                demo_obs_list.append(bob_obs_all[1:2].clone())  # env 1 obs (RIGHT)
                demo_act_list.append(
                    push_bins[t : t + 1].clone()
                )  # push bins at step t

            # Step simulation
            obs_dict, _, terminated, truncated, _ = base_env.step(combined_action)
            bob_obs_all = obs_dict["bob_policy"]

            dones = terminated | truncated
            if dones.any():
                if dones[1].item():
                    # Demo env (right) terminated — trajectory incomplete
                    episode_corrupted = True
                obs_dict, _ = base_env.reset()
                bob_obs_all = obs_dict["bob_policy"]

        # ── Phase B: Train via ABC + Aux loss ──
        if episode_corrupted or len(demo_obs_list) < 10:
            nll_history.append(nll_history[-1] if nll_history else 20.0)
            aux_history.append(aux_history[-1] if aux_history else 1.0)
            if episode_corrupted:
                print(f"  [iter {it}] Demo env (right) terminated — skipping training")
            continue

        demo_obs = torch.cat(demo_obs_list, dim=0)  # (T, obs_dim)
        demo_acts = torch.cat(demo_act_list, dim=0)  # (T, num_cat_dims)
        T = demo_obs.shape[0]

        for _ in range(args.abc_epochs):
            # Sequential LSTM ABC loss — carry (h, c) through all T steps.
            # Resetting to 0 each step loses temporal context (LSTM would treat
            # every observation as episode-start, unable to track push phases).
            h = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c = torch.zeros(1, ac.lstm_hidden_size, device=device)

            seq_lps = []
            for step in range(T):
                obs_t = demo_obs[step : step + 1]  # (1, obs_dim)
                raw, (h, c) = ac._actor_forward(obs_t, (h, c))
                dist = ac._make_distribution(raw)
                lp = dist.log_prob(demo_acts[step : step + 1].long())
                seq_lps.append(lp)

            bc_loss = -torch.stack(seq_lps).mean()  # raw NLL

            # Aux loss: GoalEncoder distance prediction
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

        # ── Evaluate (sequential, for accurate NLL measurement) ──
        with torch.no_grad():
            h_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            c_eval = torch.zeros(1, ac.lstm_hidden_size, device=device)
            eval_lps = []
            eval_greedy = []
            for step in range(T):
                obs_t = demo_obs[step : step + 1]
                raw, (h_eval, c_eval) = ac._actor_forward(obs_t, (h_eval, c_eval))
                dist = ac._make_distribution(raw)
                eval_lps.append(dist.log_prob(demo_acts[step : step + 1].long()))
                logits = raw.view(1, ac.num_cat_dims, ac.num_bins)
                eval_greedy.append(logits.argmax(dim=-1).squeeze(0))

            raw_nll = -torch.stack(eval_lps).mean().item()
            greedy = torch.stack(eval_greedy)  # (T, num_cat_dims)
            x_mode = greedy[:, 0].mode().values.item()
            y_mode = greedy[:, 1].mode().values.item()
            gr_mode = greedy[:, 5].mode().values.item()
            all_match = (greedy == demo_acts).all(dim=-1).float().mean().item()

        nll_history.append(raw_nll)
        aux_val = (
            aux_loss_val.item()
            if isinstance(aux_loss_val, torch.Tensor)
            else aux_loss_val
        )
        aux_history.append(aux_val)

        if it % 5 == 0 or it == 1:
            print(
                f"  {it:>6} | {raw_nll:>+8.3f} | {aux_val:>8.4f} | "
                f"{x_mode:>6.0f} | {y_mode:>6.0f} | {gr_mode:>6.0f} | "
                f"{all_match:>7.1%}"
            )

    # ══════════════════════════════════════════════════════════════
    # FINAL VERIFICATION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  FINAL EPISODE — LEFT (Bob) should mirror RIGHT (push demo)")
    print(f"{'=' * 70}")

    obs_dict, _ = base_env.reset()
    bob_obs_all = obs_dict["bob_policy"]
    bob_gripper = torch.ones(1, 1, device=device)

    if recorder is not None:
        print("\n  Capturing END episode (converged Bob)...")
        recorder.start()

    action_diffs = []
    for t in range(N):
        demo_act = demo_env_actions[t].unsqueeze(0)

        bob_obs_t = bob_obs_all[0:1]
        with torch.no_grad():
            bob_bins = ac.act_inference(bob_obs_t)
            bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

        diff = (demo_act - bob_act).abs().mean().item()
        action_diffs.append(diff)

        combined = torch.cat([bob_act, demo_act], dim=0)
        obs_dict, _, terminated, truncated, _ = base_env.step(combined)
        if recorder is not None:
            recorder.capture()
        bob_obs_all = obs_dict["bob_policy"]

        dones = terminated | truncated
        if dones.any():
            obs_dict, _ = base_env.reset()
            bob_obs_all = obs_dict["bob_policy"]

    if recorder is not None:
        recorder.stop_and_save(
            os.path.join(video_dir, "end_converged_shaping_push.mp4")
        )

    # ── Results ──
    mean_diff = np.mean(action_diffs)
    final_nll = nll_history[-1] if nll_history else float("inf")
    nll_decreased = len(nll_history) >= 2 and nll_history[-1] < nll_history[0]

    print(
        f"\n  NLL:         {nll_history[0]:+.3f} → {final_nll:+.3f}  "
        f"({'✓ decreased' if nll_decreased else '✗ DID NOT decrease'})"
    )
    print(
        f"  Action diff: {mean_diff:.4f}  "
        f"({'✓ small' if mean_diff < 0.3 else '✗ large'})"
    )

    with torch.no_grad():
        h_f = torch.zeros(1, ac.lstm_hidden_size, device=device)
        c_f = torch.zeros(1, ac.lstm_hidden_size, device=device)
        final_greedy_list = []
        for step in range(T):
            obs_t = demo_obs[step : step + 1]
            raw, (h_f, c_f) = ac._actor_forward(obs_t, (h_f, c_f))
            logits = raw.view(1, ac.num_cat_dims, ac.num_bins)
            final_greedy_list.append(logits.argmax(dim=-1).squeeze(0))
        final_greedy = torch.stack(final_greedy_list)

        # During the push phase (steps 60-119), X should be bin 8
        push_phase_greedy = final_greedy[60:120]
        push_phase_target = demo_acts[60:120] if len(demo_acts) > 60 else demo_acts
        push_x_mode = push_phase_greedy[:, 0].mode().values.item()

        final_match = (final_greedy == demo_acts).all(dim=-1).float().mean().item()
        push_match = (
            (push_phase_greedy == push_phase_target).all(dim=-1).float().mean().item()
        )

    print(
        f"  X mode (push): {push_x_mode:.0f} (target: 8 during push phase)  "
        f"{'✓' if abs(push_x_mode - 8) < 1 else '✗'}"
    )
    print(
        f"  Push match:    {push_match:.1%}  "
        f"({'✓ >50%' if push_match > 0.5 else '✗ <50%'})"
    )
    print(
        f"  Final Match:   {final_match:.1%}  "
        f"({'✓ >50%' if final_match > 0.5 else '✗ <50%'})"
    )

    if aux_history:
        print(f"  Aux loss:    {aux_history[0]:.4f} → {aux_history[-1]:.4f}")

    passed = nll_decreased and final_match > 0.5
    print(
        f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}: "
        f"{'Bob learned to push the cube via ABC!' if passed else 'ABC did not converge on push trajectory'}"
    )

    if not passed and final_match < 0.3:
        print("  → TIP: Try --num_iterations 500 or --abc_epochs 3")
        print(
            "  → TIP: Reduce aux_coef in ppo_continuous.yaml if GoalEncoder dominates"
        )

    # ── Continuous replay for visual inspection ──────────────────
    print("\n  Replaying continuously for visual inspection...")
    print("  LEFT = Bob (trained), RIGHT = demo push")
    print("  (Close window or Ctrl+C to exit)\n")

    try:
        while simulation_app.is_running():
            obs_dict, _ = base_env.reset()
            bob_obs_all = obs_dict["bob_policy"]
            bob_gripper = torch.ones(1, 1, device=device)

            for t in range(N):
                demo_act = demo_env_actions[t].unsqueeze(0)

                bob_obs_t = bob_obs_all[0:1]
                with torch.no_grad():
                    bob_bins = ac.act_inference(bob_obs_t)
                    bob_act, bob_gripper = bins_to_env_action(bob_bins, bob_gripper)

                combined = torch.cat([bob_act, demo_act], dim=0)
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
