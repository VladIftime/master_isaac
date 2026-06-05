"""
GoalEncoder Unit Tests + DifferentialIK Smoke Test
===================================================

Combines all five GoalEncoder property tests from test_goal_encoder_unit.py
with a live DifferentialIK environment smoke test that saves a video.

GoalEncoder unit tests (pure PyTorch, no Isaac Sim physics):
  1. at_goal_is_zero   — current==goal  →  ||g|| = 0 exactly
  2. norm_decreases    — current→goal   →  ||g|| has strictly negative trend
  3. perm_invariant    — swap objects   →  identical pooled embedding
  4. distinct_goals    — diff goals     →  different embeddings
  5. aux_loss_trains   — aux MSE decreases over 50 gradient steps

DifferentialIK smoke test (Isaac Sim):
  - Creates AsyncDualPlayDiffIKEnvCfg (2 envs)
  - Verifies action dim == 7 (DiffIK arm=6 + gripper=1)
  - Verifies alice_policy / bob_policy obs groups are accessible
  - Runs --smoke_steps steps with a small forward nudge action
  - Saves video to tests/videos/diffik_smoke_test.mp4

Usage (from project root — NOT headless to capture video):
    python asyncDualPlayPPO/tests/test_goal_encoder_unit_diffik.py

Compare against RMPflow unit tests (no Isaac Sim):
    python asyncDualPlayPPO/tests/test_goal_encoder_unit.py
"""

import isaaclab.app
from isaaclab.app import AppLauncher

import argparse
import os
import sys
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ── video helpers ─────────────────────────────────────────────────────────────

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
    def __init__(self, cam_pos: tuple, look_at: tuple, fps: int = 24,
                 resolution: tuple = (1920, 1080)):
        import omni.replicator.core as rep
        self._rep = rep
        self._fps = fps
        self._frames: list = []
        self.active = False
        cam = rep.create.camera(position=cam_pos, look_at=look_at,
                                focal_length=18.0, clipping_range=(0.01, 10000.0))
        rp = rep.create.render_product(cam, resolution)
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annot.attach([rp])

    def start(self) -> None:
        self._frames = []
        self.active = True

    def capture(self) -> None:
        if not self.active:
            return
        data = self._annot.get_data()
        if data is None:
            return
        arr = data["data"] if isinstance(data, dict) else data
        if arr is None or arr.size == 0:
            return
        self._frames.append(arr[:, :, :3].copy())

    def stop_and_save(self, output_path: str) -> None:
        self.active = False
        if not self._frames:
            print(f"  [VideoRecorder] No frames captured — skipping {output_path}")
            return
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _write_mp4(self._frames, output_path, self._fps)
        print(f"  [VideoRecorder] {len(self._frames)} frames → {output_path}")


def install_noise_filter() -> None:
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


# ── GoalEncoder unit tests (pure PyTorch) ────────────────────────────────────

_NUM_OBJECTS = 2
_K_PER_OBJ   = 6
_POSE_DIM    = 6
_FLAT        = _NUM_OBJECTS * _POSE_DIM   # 12


def _enc(device: str = "cpu"):
    from asyncDualPlayPPO.algorithms.goal_encoder import GoalEncoder
    enc = GoalEncoder(
        num_objects=_NUM_OBJECTS,
        K_per_obj=_K_PER_OBJ,
        hidden_dim=64,
        variant="difference",
        use_aux_loss=True,
    ).to(device)
    enc.eval()
    return enc


def _rand_pose(batch: int = 1, device: str = "cpu"):
    import torch
    return torch.randn(batch, _FLAT, device=device)


def _test_at_goal_is_zero(device: str) -> None:
    import torch
    enc = _enc(device)
    goal = _rand_pose(4, device)
    with torch.no_grad():
        g = enc(goal, goal)
    assert g.shape == (4, _K_PER_OBJ), f"wrong shape {g.shape}"
    assert g.abs().max().item() < 1e-5, (
        f"||g|| should be 0 when current==goal, got max={g.abs().max().item():.2e}"
    )
    print("  PASS  at_goal_is_zero")


def _test_norm_decreases(device: str) -> None:
    import torch
    import numpy as np
    enc = _enc(device)
    T = 50
    goal  = _rand_pose(1, device).expand(T, -1)
    start = _rand_pose(1, device).expand(T, -1)
    alpha = torch.linspace(0, 1, T, device=device).unsqueeze(1)
    current = start + alpha * (goal - start)
    with torch.no_grad():
        g = enc(goal, current)
    norms = g.norm(dim=-1).cpu().numpy()
    t_idx = np.arange(T, dtype=float)
    corr = float(
        torch.corrcoef(
            torch.from_numpy(np.stack([norms, t_idx]))
        )[0, 1].item()
    )
    assert corr < -0.5, (
        f"||g|| should decrease as current→goal (corr expected < -0.5, got {corr:.4f}). "
        "Encoder may be receiving scrambled features."
    )
    print(f"  PASS  norm_decreases  (Pearson corr={corr:.4f})")


def _test_perm_invariant(device: str) -> None:
    import torch
    enc = _enc(device)
    goal = _rand_pose(8, device)
    curr = _rand_pose(8, device)
    goal_swap = torch.cat([goal[:, _POSE_DIM:], goal[:, :_POSE_DIM]], dim=-1)
    curr_swap = torch.cat([curr[:, _POSE_DIM:], curr[:, :_POSE_DIM]], dim=-1)
    with torch.no_grad():
        g_orig = enc(goal, curr)
        g_swap = enc(goal_swap, curr_swap)
    max_diff = (g_orig - g_swap).abs().max().item()
    assert max_diff < 1e-5, (
        f"Encoder is NOT permutation-invariant: max |g_orig - g_swap| = {max_diff:.2e}"
    )
    print(f"  PASS  permutation_invariant  (max_diff={max_diff:.2e})")


def _test_distinct_goals(device: str) -> None:
    import torch
    enc = _enc(device)
    curr   = torch.zeros(1, _FLAT, device=device)
    goal_a = torch.ones(1, _FLAT, device=device) *  1.0
    goal_b = torch.ones(1, _FLAT, device=device) * -1.0
    with torch.no_grad():
        g_a = enc(goal_a, curr)
        g_b = enc(goal_b, curr)
    diff = (g_a - g_b).norm().item()
    assert diff > 0.01, (
        f"Two opposite goals collapsed to same embedding (||g_a - g_b|| = {diff:.4f})"
    )
    print(f"  PASS  distinct_goals  (||g_a - g_b|| = {diff:.4f})")


def _test_aux_loss_trains(device: str) -> None:
    import torch
    from asyncDualPlayPPO.algorithms.goal_encoder import GoalEncoder
    enc = GoalEncoder(
        num_objects=_NUM_OBJECTS,
        K_per_obj=_K_PER_OBJ,
        hidden_dim=64,
        variant="difference",
        use_aux_loss=True,
    ).to(device)
    enc.train()
    optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)
    torch.manual_seed(0)
    goal = _rand_pose(32, device)
    curr = _rand_pose(32, device)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        loss, _, _ = enc.aux_loss(goal, curr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], (
        f"Aux loss did not decrease: start={losses[0]:.4f}  end={losses[-1]:.4f}"
    )
    print(f"  PASS  aux_loss_trains  ({losses[0]:.4f} → {losses[-1]:.4f})")


def run_unit_tests(device: str) -> tuple:
    """Run all 5 GoalEncoder unit tests; return (passed, failed)."""
    suite = [
        ("at_goal_is_zero",       lambda: _test_at_goal_is_zero(device)),
        ("norm_decreases",        lambda: _test_norm_decreases(device)),
        ("permutation_invariant", lambda: _test_perm_invariant(device)),
        ("distinct_goals",        lambda: _test_distinct_goals(device)),
        ("aux_loss_trains",       lambda: _test_aux_loss_trains(device)),
    ]
    passed = failed = 0
    for name, fn in suite:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    return passed, failed


# ── DiffIK smoke test ─────────────────────────────────────────────────────────

def run_diffik_smoke_test(simulation_app, video_dir: str, smoke_steps: int = 80) -> None:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play_diffik import AsyncDualPlayDiffIKEnvCfg

    print("\n  Creating DiffIK env (2 envs) ...")
    env_cfg = AsyncDualPlayDiffIKEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.scene.env_spacing = 3.0
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    print(f"  Device: {device}")

    # ── obs dim verification ──────────────────────────────────────────────────
    alice_dim_info = env.unwrapped.observation_manager.group_obs_dim["alice_policy"]
    bob_dim_info   = env.unwrapped.observation_manager.group_obs_dim["bob_policy"]
    alice_obs_dim = alice_dim_info[0] if isinstance(alice_dim_info, (tuple, list)) else alice_dim_info
    bob_obs_dim   = bob_dim_info[0]   if isinstance(bob_dim_info,   (tuple, list)) else bob_dim_info
    print(f"  Alice obs dim: {alice_obs_dim}  |  Bob obs dim: {bob_obs_dim}")

    # ── action dim verification ───────────────────────────────────────────────
    if len(env.action_space.shape) > 1:
        env_action_dim = env.action_space.shape[1]
    else:
        env_action_dim = env.action_space.shape[0]
    print(f"  Env action dim: {env_action_dim}  (expect 7: DiffIK arm=6 + gripper=1)")
    assert env_action_dim == 7, f"Expected action dim 7, got {env_action_dim}"

    # ── camera setup ─────────────────────────────────────────────────────────
    recorder = _VideoRecorder(
        cam_pos=(3.5, 0.0, 2.8),
        look_at=(0.0, 0.0, 0.5),
        fps=24,
    )

    # ── reset + step loop ────────────────────────────────────────────────────
    obs_dict, _ = env.reset()
    simulation_app.update()

    assert "alice_policy" in obs_dict, "alice_policy missing from obs after reset"
    assert "bob_policy"   in obs_dict, "bob_policy missing from obs after reset"

    recorder.start()
    for _ in range(smoke_steps):
        # Small constant nudge — just enough to show the controller is live
        action = torch.zeros(2, env_action_dim, device=device)
        action[:, 0] = 0.05   # dx
        obs_dict, _, _, _, _ = env.step(action)
        recorder.capture()
        simulation_app.update()

    assert "alice_policy" in obs_dict, "alice_policy missing from obs after steps"
    assert "bob_policy"   in obs_dict, "bob_policy missing from obs after steps"

    # ── save video ───────────────────────────────────────────────────────────
    video_path = os.path.join(video_dir, "diffik_smoke_test.mp4")
    recorder.stop_and_save(video_path)

    print(f"  PASS  diffik_smoke_test  ({smoke_steps} steps, action_dim=7, obs keys OK)")
    env.close()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GoalEncoder unit tests + DifferentialIK smoke test"
    )
    parser.add_argument(
        "--smoke_steps", type=int, default=80,
        help="Steps to run in the DiffIK smoke test (default: 80)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    install_noise_filter()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir  = os.path.join(script_dir, "videos")

    # ── section 1: GoalEncoder unit tests ────────────────────────────────────
    print("\nGoalEncoder unit tests (DiffIK env context, pure PyTorch)")
    print("=" * 58)
    print(f"  Device: {device}")
    unit_passed, unit_failed = run_unit_tests(device)
    print("=" * 58)
    print(f"  {unit_passed} passed  {unit_failed} failed\n")

    # ── section 2: DiffIK smoke test ─────────────────────────────────────────
    print("DifferentialIK smoke test (Isaac Sim)")
    print("=" * 58)
    smoke_passed = False
    try:
        run_diffik_smoke_test(simulation_app, video_dir, args.smoke_steps)
        smoke_passed = True
    except AssertionError as e:
        print(f"  FAIL  diffik_smoke_test: {e}")
    except Exception as e:
        print(f"  ERROR diffik_smoke_test: {type(e).__name__}: {e}")
    print("=" * 58)
    print(f"  {'1 passed' if smoke_passed else '0 passed  1 failed'}\n")

    total_failed = unit_failed + (0 if smoke_passed else 1)
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
