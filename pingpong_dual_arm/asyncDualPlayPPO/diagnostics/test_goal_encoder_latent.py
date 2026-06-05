"""
Test 4b — Goal Encoder Latent Space Analysis.

Loads a trained checkpoint, generates synthetic goal embeddings spanning
three heuristic categories (planar / lifted / rotated), then:
    1. Computes t-SNE and silhouette score (target > 0.15)
    2. Checks noise invariance: 2cm perturbation → relative change < 0.20
    3. Reads tensorboard scalars to verify embedding_norm / embedding_std health

Prerequisite: at least 500 training iterations (checkpoint must exist).

Usage:
    python -m asyncDualPlayPPO.diagnostics.test_goal_encoder_latent \\
        --ckpt asyncDualPlayPPO/runs/diag_t3/bob/model_50.pt \\
        --cfg  asyncDualPlayPPO/cfg/ppo/ppo_continuous.yaml \\
        --log_dir asyncDualPlayPPO/runs/diag_t3/summary \\
        --num_objects 1
"""

import argparse
import math
import os
import sys

import torch


def _make_synthetic_goals(num_objects: int, n_per_class: int = 300, device="cpu"):
    """
    Generate three classes of (goal_flat, curr_flat) pairs:
        planar  — object stays on table, small XY displacement
        lifted  — object raised >5 cm in Z
        rotated — object rotated >0.3 rad in one Euler axis
    Returns goal_flat, curr_flat, labels each of shape (3*n, num_obj*6).
    """
    N = n_per_class
    dims = num_objects * 6

    curr = torch.zeros(3 * N, dims, device=device)
    goal = torch.zeros(3 * N, dims, device=device)

    # planar: small XY displacement, no Z or rotation change
    goal[:N, 0] = torch.rand(N, device=device) * 0.15 + 0.02   # x 2–17 cm
    goal[:N, 1] = torch.rand(N, device=device) * 0.10 + 0.02   # y 2–12 cm

    # lifted: Z displacement 5–20 cm
    goal[N:2*N, 2] = torch.rand(N, device=device) * 0.15 + 0.05

    # rotated: rotation in yaw (euler[2])
    goal[2*N:, 5] = torch.rand(N, device=device) * (math.pi / 2) + 0.3

    labels = ["planar"] * N + ["lifted"] * N + ["rotated"] * N
    return goal, curr, labels


def check_latent_space(ckpt_path: str, cfg_path: str, log_dir: str = None,
                       num_objects: int = 1, save_plot: str = "goal_encoder_tsne.png"):
    print("\n" + "=" * 70)
    print("TEST 4b — Goal Encoder Latent Space Analysis")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    import yaml

    if not os.path.isfile(ckpt_path):
        print(f"  SKIP: checkpoint not found at {ckpt_path}")
        sys.exit(0)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["params"]["policy"]
    model_cfg["num_objects"] = num_objects
    model_cfg["use_goal_encoder"] = True   # set at runtime in train_curobo.py

    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    obs_dim = None
    for k, v in state.items():
        if "critic.0.weight" in k and v.ndim == 2:
            obs_dim = v.shape[1]
            break
    if obs_dim is None:
        print(f"  SKIP: could not infer obs_dim from checkpoint")
        sys.exit(0)

    model = ActorCritic((obs_dim,), (obs_dim,), 6, 1.0, model_cfg=model_cfg)
    model.load_state_dict(state, strict=False)
    model = model.eval().to(device)

    if not model.use_goal_encoder:
        print("  SKIP: use_goal_encoder=False")
        sys.exit(0)

    failures = []

    # ── Embedding extraction ───────────────────────────────────────────────────
    goal_flat, curr_flat, labels = _make_synthetic_goals(num_objects, n_per_class=300, device=device)

    with torch.no_grad():
        emb = model.goal_encoder(goal_flat, curr_flat)  # (900, K_per_obj)

    emb_np = emb.cpu().numpy()
    print(f"  Embeddings: shape={emb_np.shape}  norm_mean={emb.norm(dim=-1).mean():.4f}  std_mean={emb.std(dim=-1).mean():.4f}")

    # ── Embedding stats sanity ─────────────────────────────────────────────────
    norm_mean = emb.norm(dim=-1).mean().item()
    std_mean = emb.std(dim=-1).mean().item()
    if norm_mean < 0.01:
        failures.append(f"FAIL: embedding collapsed to near-zero (norm_mean={norm_mean:.4f})")
    if std_mean < 0.001:
        failures.append(f"FAIL: encoder outputs nearly constant vectors (std_mean={std_mean:.4f})")

    # ── t-SNE + silhouette ─────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
        import numpy as np

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        X_2d = tsne.fit_transform(emb_np)

        label_ids = [{"planar": 0, "lifted": 1, "rotated": 2}[l] for l in labels]
        sil = silhouette_score(X_2d, label_ids)
        print(f"  t-SNE silhouette score: {sil:.3f}  (target > 0.15)")

        if sil < 0.15:
            failures.append(
                f"FAIL: t-SNE silhouette={sil:.3f} < 0.15 — "
                "goal encoder not separating goal types; need more training"
            )

        # Save plot if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            for lbl, color in [("planar", "blue"), ("lifted", "red"), ("rotated", "green")]:
                mask = [l == lbl for l in labels]
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl, s=8, alpha=0.6)
            ax.legend()
            ax.set_title(f"GoalEncoder t-SNE  silhouette={sil:.3f}")
            plt.savefig(save_plot, dpi=150)
            plt.close(fig)
            print(f"  t-SNE plot saved → {save_plot}")
        except ImportError:
            pass

    except ImportError:
        print("  SKIP t-SNE: sklearn not installed (pip install scikit-learn)")

    # ── Noise invariance ──────────────────────────────────────────────────────
    sigma = 0.02  # 2 cm
    perturbed = goal_flat + sigma * torch.randn_like(goal_flat)
    with torch.no_grad():
        emb_p = model.goal_encoder(perturbed, curr_flat)
    delta_norm = (emb_p - emb).norm(dim=-1)
    rel_change = (delta_norm / (emb.norm(dim=-1) + 1e-6)).mean().item()
    print(f"  Noise invariance (σ=2cm): rel_change={rel_change:.4f}  (target < 0.20)")
    if rel_change > 0.20:
        failures.append(
            f"FAIL: GoalEncoder too sensitive to 2cm noise (rel_change={rel_change:.4f} > 0.20)"
        )

    # ── TB scalars health check ────────────────────────────────────────────────
    if log_dir and os.path.isdir(log_dir):
        try:
            from tbparse import SummaryReader
            df = SummaryReader(log_dir).scalars
        except ImportError:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            import pandas as pd
            ea = EventAccumulator(log_dir)
            ea.Reload()
            rows = []
            for tag in ea.Tags()["scalars"]:
                for e in ea.Scalars(tag):
                    rows.append({"tag": tag, "step": e.step, "value": e.value})
            df = pd.DataFrame(rows)

        norm_vals = df[df["tag"] == "GoalEncoder/embedding_norm"]["value"]
        std_vals = df[df["tag"] == "GoalEncoder/embedding_std"]["value"]
        if len(norm_vals) >= 20:
            tail_norm = norm_vals.values[-20:].mean()
            tail_std = std_vals.values[-20:].mean() if len(std_vals) >= 20 else 0
            print(f"  TB embedding_norm tail mean: {tail_norm:.4f}")
            print(f"  TB embedding_std  tail mean: {tail_std:.4f}")
            if tail_norm < 0.1:
                failures.append(f"FAIL: embedding_norm collapsed (TB tail mean={tail_norm:.4f})")
            if tail_std < 0.01:
                failures.append(f"FAIL: embedding_std too low (TB tail mean={tail_std:.4f})")

    if failures:
        print()
        for f in failures:
            print(f"  {f}")
        print("TEST 4b FAIL")
        sys.exit(1)
    else:
        print("TEST 4b PASS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--cfg",
        default=os.path.join(os.path.dirname(__file__), "..", "cfg", "ppo", "ppo_continuous.yaml"),
    )
    parser.add_argument("--log_dir", default=None, help="TB summary dir for embedding health check")
    parser.add_argument("--num_objects", type=int, default=1)
    parser.add_argument("--save_plot", default="goal_encoder_tsne.png")
    args = parser.parse_args()
    check_latent_space(args.ckpt, args.cfg, args.log_dir, args.num_objects, args.save_plot)
