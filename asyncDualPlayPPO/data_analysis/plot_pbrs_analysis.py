#!/usr/bin/env python3
"""
PBRS reward-function justification plots for thesis.

Generates two suites of plots:
  Part A — "Why PBRS is the right reward function"
  Part B — "Why these specific hyperparameters were chosen"

All analytical plots require zero training data — they are computed from the
reward formulas themselves.  Empirical learning-curve plots are added when
TensorBoard log directories are supplied with --tb-dirs.

Usage:
  # Analytical-only (runs immediately, no training logs needed):
  python -m asyncDualPlayPPO.data_analysis.plot_pbrs_analysis -o analysis/pbrs

  # With TensorBoard data from completed training runs:
  python -m asyncDualPlayPPO.data_analysis.plot_pbrs_analysis \
      --tb-dirs runs/push_pbrs_a/summary runs/push_pbrs_b/summary \
      --labels "A: PBRS only" "B: PBRS+Curriculum" \
      -o analysis/pbrs

  # Compare PBRS run(s) against old fractional-improvement runs:
  python -m asyncDualPlayPPO.data_analysis.plot_pbrs_analysis \
      --tb-dirs runs/push_pbrs_a/summary runs/push_ppo_rel_full/summary \
      --labels "PBRS" "Old Fractional" \
      -o analysis/pbrs_vs_old
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("[ERROR] matplotlib is required.  Install with: pip install matplotlib")
    sys.exit(1)

# ── PBRS constants (mirrors reward_pbrs.py) ───────────────────────────────────
K_P = 30.0
K_R = 5.0
W_POS = 10.0
W_ROT = 10.0
POS_THRESHOLD = 0.05
COS_ROT_THRESHOLD = 0.01
COMPLETION_BONUS = 5.0
ROTATION_BONUS = 2.0

# Old fractional-improvement formula constants (mirrors wrapper_push.py)
OLD_ALPHA = 3.0
OLD_BETA = 0.5
OLD_BETA_ROT = 0.25
OLD_COMPLETION = 5.0
OLD_ROT_SUB = 2.0
OLD_POS_THRESH = 0.05
OLD_ROT_THRESH = 0.2

# Workspace range
D_MAX = 0.45
YAW_MAX = np.pi

DEFAULT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
}


def potential_pos(d, k_p=K_P):
    """Φ_pos(s) = exp(−k_p · ||obj_xy − goal_xy||²)"""
    return np.exp(-k_p * d**2)


def potential_rot(yaw_err, k_r=K_R):
    """Φ_rot(s) = exp(−k_r · (1 − cos Δθ) / 2)"""
    c = (1.0 - np.cos(yaw_err)) / 2.0
    return np.exp(-k_r * c)


def old_fractional_reward(d_prev, d_now):
    """Old formula: α·(d_prev−d_now)/d_prev − β·d_now (clamped)."""
    denom = np.clip(d_prev, 0.01, None)
    imp = OLD_ALPHA * (d_prev - d_now) / denom
    imp = np.clip(imp, -5.0, 5.0)
    pen = -OLD_BETA * d_now
    pen = np.clip(pen, -2.0, 0.0)
    return imp + pen


def pbrs_reward(d_prev, d_now, k_p=K_P, w=W_POS):
    """PBRS: w·(Φ_pos(s') − Φ_pos(s))."""
    phi_now = potential_pos(d_now, k_p)
    phi_prev = potential_pos(d_prev, k_p)
    return w * (phi_now - phi_prev)


# ── Smoothing ─────────────────────────────────────────────────────────────────
def smooth(values, weight=0.6):
    s = np.zeros_like(values)
    if len(values) == 0:
        return s
    last = values[0]
    for i, v in enumerate(values):
        sv = last * weight + (1 - weight) * v
        s[i] = sv
        last = sv
    return s


def ema(values, alpha=0.9):
    s = np.zeros_like(values)
    if len(values) == 0:
        return s
    last = values[0]
    for i, v in enumerate(values):
        last = alpha * last + (1 - alpha) * v
        s[i] = last
    return s


# ── TensorBoard loading ───────────────────────────────────────────────────────
def load_tensorboard_data(summary_dir: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[ERROR] tensorboard is required for --tb-dirs.  Install with: pip install tensorboard")
        sys.exit(1)

    summary_dir = _resolve_summary_dir(summary_dir)
    ea = EventAccumulator(str(summary_dir))
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": np.array([e.step for e in events]),
            "values": np.array([e.value for e in events]),
        }
    return data


def _resolve_summary_dir(path: Path) -> Path:
    if path.is_dir():
        for f in path.iterdir():
            if f.name.startswith("events.out.tfevents"):
                return path
    summary_sub = path / "summary"
    if summary_sub.is_dir():
        return summary_sub
    for child in path.iterdir():
        if child.is_dir():
            for f in child.iterdir():
                if f.name.startswith("events.out.tfevents"):
                    return child
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Why PBRS is the right reward function
# ═══════════════════════════════════════════════════════════════════════════════

def plot_a1_potential_landscape(out_dir: Path):
    """A1: 2D heatmap of Φ_pos(s) across the workspace."""
    print("  [A1] Potential landscape heatmap ...", flush=True)

    x = np.linspace(-0.50, 0.50, 200)
    y = np.linspace(0.25, 0.70, 200)
    X, Y = np.meshgrid(x, y)
    goal = np.array([0.0, 0.50])
    D = np.sqrt((X - goal[0])**2 + (Y - goal[1])**2)
    Z = potential_pos(D, k_p=K_P)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax.plot(goal[0], goal[1], "r*", markersize=14, markeredgecolor="white", markeredgewidth=1.5,
            label="Goal")
    contour_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95]
    cs = ax.contour(X, Y, Z, levels=contour_levels, colors="white", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%.2f", fontsize=7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("PBRS Position Potential  Φ_pos(s) = exp(−k_p · d_xy²)  (k_p=30)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Φ_pos(s)")
    fig.tight_layout()
    fig.savefig(out_dir / "a1_potential_landscape.png", dpi=200)
    plt.close(fig)


def plot_a2_gradient_comparison(out_dir: Path):
    """A2: Gradient magnitude |∂R/∂d| for PBRS vs old fractional vs old raw delta."""
    print("  [A2] Gradient magnitude comparison ...", flush=True)

    d_prev = np.linspace(0.01, 0.45, 500)
    delta = 0.01  # small improvement

    # PBRS gradient: (Φ(d_prev-δ) − Φ(d_prev)) / δ  (approximate ∂R/∂d)
    d_next = np.clip(d_prev - delta, 0.001, None)
    pbrs_grad = np.abs(pbrs_reward(d_prev, d_next) / delta)

    # Old fractional gradient
    old_frac_grad = np.abs(old_fractional_reward(d_prev, d_next) / delta)

    # Old raw delta gradient (α only, pre-Fix P63)
    old_raw_grad = np.full_like(d_prev, 12.0)  # α_pos=12

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(d_prev, pbrs_grad, linewidth=2, label=f"PBRS  w·∂Φ/∂d  (k_p={int(K_P)}, w={int(W_POS)})",
            color="#1f77b4")
    ax.plot(d_prev, old_frac_grad, linewidth=2, label=f"Old fractional  α·∂(Δd/d_prev)/∂d  (α={OLD_ALPHA:.0f})",
            color="#d62728", linestyle="--")
    ax.plot(d_prev, old_raw_grad, linewidth=2, label="Old raw delta  α=12  (pre-Fix P63)",
            color="#ff7f0e", linestyle=":", alpha=0.7)

    ax.axvline(x=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")
    ax.axvline(x=0.13, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.4)
    ax.annotate("Peak gradient\nat d=0.13 m", xy=(0.13, 22), xytext=(0.20, 25),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.7),
                fontsize=8, color="#1f77b4")

    ax.set_xlabel("Distance from goal  d (m)")
    ax.set_ylabel("|∂R/∂d|  (gradient magnitude)")
    ax.set_title("Reward Gradient Magnitude vs Distance — PBRS vs Old Fractional Formula")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 0.45)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "a2_gradient_comparison.png", dpi=200)
    plt.close(fig)


def plot_a3_episode_simulation(out_dir: Path):
    """A3: Per-push reward comparison for a simulated 5-push episode."""
    print("  [A3] Simulated episode reward comparison ...", flush=True)

    start_d = 0.20
    pushes = [0.08, 0.05, 0.03, 0.02, 0.01]
    n = len(pushes)

    d = np.zeros(n + 1)
    d[0] = start_d
    for i in range(n):
        d[i + 1] = max(d[i] - pushes[i], 0.001)

    pbrs_rews = np.array([pbrs_reward(d[i], d[i+1]) for i in range(n)])
    old_rews = np.array([old_fractional_reward(d[i], d[i+1]) for i in range(n)])

    # Add completion bonus to old formula on last push if within threshold
    if d[-1] < OLD_POS_THRESH:
        old_rews[-1] += OLD_COMPLETION
    # For PBRS, completion bonus is separate — show it as stacked
    pbrs_completion = 0.0
    for i in range(n):
        if d[i+1] < POS_THRESHOLD and d[i] >= POS_THRESHOLD:
            pbrs_completion = COMPLETION_BONUS
            break

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(1, n + 1)
    width = 0.35

    # Top-left: PBRS rewards
    ax = axes[0, 0]
    bars = ax.bar(x, pbrs_rews, width, color="#1f77b4", alpha=0.85, label="PBRS dense  w·(Φ′−Φ)")
    # Add completion bar if applicable
    if pbrs_completion > 0:
        ax.bar([5], [pbrs_completion], width, bottom=[pbrs_rews[-1]], color="#2ca02c", alpha=0.7,
               label=f"+{COMPLETION_BONUS:.0f} completion bonus")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Push")
    ax.set_ylabel("Reward")
    ax.set_title("PBRS Rewards Per Push  (w=10, k_p=30)")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Top-right: Old fractional rewards
    ax = axes[0, 1]
    bars_old = ax.bar(x, old_rews, width, color="#d62728", alpha=0.85, label="Old fractional  α·Δd/d_prev − β·d")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Push")
    ax.set_ylabel("Reward")
    ax.set_title(f"Old Fractional Rewards Per Push  (α={OLD_ALPHA:.0f}, β={OLD_BETA})")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Bottom: distance progression
    ax = axes[1, 0]
    ax.plot(np.arange(0, n + 1), d, "ko-", linewidth=2, markersize=6)
    ax.axhline(y=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")
    ax.set_xlabel("Pushes completed")
    ax.set_ylabel("Distance to goal  d (m)")
    ax.set_title("Distance Progression Over Episode")
    ax.set_xticks(np.arange(0, n + 1))
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Bottom-right: key comparison table
    ax = axes[1, 1]
    ax.axis("off")
    table_text = (
        f"Start distance: {start_d:.2f} m\n"
        f"Final distance:  {d[-1]:.3f} m\n"
        f"PBRS total dense reward:  {pbrs_rews.sum():+.2f}\n"
        f"Old fractional total:     {old_rews.sum():+.2f}\n\n"
        f"PBRS: diminishing returns near goal,\n"
        f"      no penalty terms\n\n"
        f"Old formula: amplified rewards near goal,\n"
        f"      −β·d penalty shifts optimal policy"
    )
    ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

    fig.subplots_adjust(top=0.88)
    fig.suptitle("Simulated 5-Push Episode — PBRS vs Old Fractional Formula",
                 fontsize=13, y=0.98)
    fig.savefig(out_dir / "a3_episode_simulation.png", dpi=200)
    plt.close(fig)


def plot_a4_cosine_distance(out_dir: Path):
    """A4: Cosine angular distance vs yaw_distance_rad."""
    print("  [A4] Cosine vs modular angular distance ...", flush=True)

    dtheta = np.linspace(-np.pi, np.pi, 600)
    cos_err = (1.0 - np.cos(dtheta)) / 2.0
    yaw_dist = np.abs(dtheta)
    yaw_dist = np.where(yaw_dist > np.pi, 2 * np.pi - yaw_dist, yaw_dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: angular error functions
    ax = axes[0]
    ax.plot(dtheta, cos_err, linewidth=2, color="#1f77b4",
            label="cos_rot_err = (1 − cos Δθ) / 2  (C∞ smooth)")
    ax.plot(dtheta, yaw_dist / np.pi, linewidth=2, color="#d62728", linestyle="--",
            label="yaw_err = min(|Δθ|, 2π − |Δθ|) / π  (cusp at ±π)")
    ax.axvline(x=-np.pi, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(x=np.pi, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate("Cusp — gradient\nundefined at ±π",
                xy=(np.pi, 1.0), xytext=(np.pi - 1.5, 0.7),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=8, color="#d62728")
    ax.set_xlabel("Δθ  (rad)")
    ax.set_ylabel("Error metric")
    ax.set_title("Angular Error Metrics")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-0.02, 1.05)

    # Right: gradient of the error functions
    ax = axes[1]
    dcos_dtheta = np.gradient(cos_err, dtheta[1] - dtheta[0])
    dyaw_dtheta = np.gradient(yaw_dist / np.pi, dtheta[1] - dtheta[0])
    ax.plot(dtheta, dcos_dtheta, linewidth=2, color="#1f77b4",
            label="∂(cos_rot_err)/∂θ = sin(Δθ) / 2  (smooth)")
    ax.plot(dtheta, dyaw_dtheta, linewidth=2, color="#d62728", linestyle="--",
            label="∂(yaw_err)/∂θ  (step at ±π)")
    ax.axvline(x=-np.pi, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(x=np.pi, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate("Discontinuity", xy=(np.pi - 0.01, 1/np.pi), xytext=(1.5, 0.35),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=8, color="#d62728")
    ax.set_xlabel("Δθ  (rad)")
    ax.set_ylabel("Gradient")
    ax.set_title("Gradient of Angular Error Metrics")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-np.pi, np.pi)

    fig.subplots_adjust(top=0.88)
    fig.suptitle("Cosine Angular Distance vs yaw_distance_rad",
                 fontsize=13, y=0.98)
    fig.savefig(out_dir / "a4_cosine_distance.png", dpi=200)
    plt.close(fig)


def _find_tag(data: dict, aliases: list) -> str | None:
    for tag in aliases:
        if tag in data:
            return tag
    return None


def plot_a5_tb_learning_curves(all_data: list, labels: list, out_dir: Path):
    """A5: Learning curves from TensorBoard data."""
    if not all_data:
        return

    print("  [A5] Learning curves from TensorBoard ...", flush=True)

    metrics = [
        (["Metrics/SuccessRate", "Metrics/Bob/SuccessRate"], "Success Rate", "Success Rate"),
        (["Metrics/RotationSR", "Metrics/Bob/RotationSR"], "Rotation SR", "Rotation SR"),
        (["Metrics/PosError", "Metrics/Bob/PosError"], "Position Error (m)", "Position Error"),
        (["Metrics/RotError", "Metrics/Bob/RotError"], "Rotation Error (rad)", "Rotation Error"),
        (["Reward/Mean", "Reward/Bob"], "Mean Reward", "Mean Reward"),
        (["Loss/Agent/Value", "Loss/Bob/Value"], "Value Loss", "Value Loss"),
        (["Loss/Agent/Surrogate", "Loss/Bob/Surrogate"], "Policy Loss", "Policy Loss"),
        (["Reward/EMA"], "EMA Reward", "EMA Reward"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_data), 10)))

    for aliases, ylabel, title in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        any_found = False
        for i, (data, lbl) in enumerate(zip(all_data, labels)):
            tag = _find_tag(data, aliases)
            if tag is None:
                continue
            any_found = True
            steps = data[tag]["steps"]
            values = data[tag]["values"]
            ax.plot(steps, values, alpha=0.15, color=colors[i])
            ax.plot(steps, smooth(values, 0.6), color=colors[i], linewidth=2, label=lbl)
        if not any_found:
            plt.close(fig)
            continue
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_name = title.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(out_dir / f"a5_{safe_name}.png", dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Why these specific hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════

def plot_b1_kp_sensitivity(out_dir: Path):
    """B1: Potential shape and gradient for k_p ∈ {15, 30, 50}."""
    print("  [B1] k_p sensitivity ...", flush=True)

    d = np.linspace(0, 0.45, 500)
    k_values = [15, 30, 50]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    styles = ["--", "-", ":"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: potential
    ax = axes[0]
    for k, c, ls in zip(k_values, colors, styles):
        phi = potential_pos(d, k_p=k)
        ax.plot(d, phi, linewidth=2, color=c, linestyle=ls, label=f"k_p = {k}")
        ax.axhline(y=0.01, color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(x=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")
    ax.set_xlabel("Distance from goal  d (m)")
    ax.set_ylabel("Φ_pos(s)")
    ax.set_title("Position Potential Shape  Φ_pos(d) = exp(−k_p · d²)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)

    # Right: gradient magnitude
    ax = axes[1]
    for k, c, ls in zip(k_values, colors, styles):
        delta = 0.001
        d_next = np.clip(d - delta, 0.0001, None)
        grad = np.abs(pbrs_reward(d, d_next, k_p=k) / delta)
        ax.plot(d, grad, linewidth=2, color=c, linestyle=ls, label=f"k_p = {k}")
        peak_d = 1 / np.sqrt(2 * k)
        ax.axvline(x=peak_d, color=c, linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axvline(x=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")
    ax.set_xlabel("Distance from goal  d (m)")
    ax.set_ylabel("|∂R/∂d|  (gradient magnitude)")
    ax.set_title(f"Reward Gradient Magnitude  |∂R/∂d|  (w={int(W_POS)})")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.subplots_adjust(top=0.88)
    fig.suptitle("Hyperparameter Sensitivity: k_p  (position potential sharpness)",
                 fontsize=13, y=0.98)
    fig.savefig(out_dir / "b1_kp_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_b2_kr_sensitivity(out_dir: Path):
    """B2: Rotation potential shape for k_r ∈ {3, 5, 10}."""
    print("  [B2] k_r sensitivity ...", flush=True)

    dtheta = np.linspace(-np.pi, np.pi, 600)
    k_values = [3, 5, 10]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    styles = ["--", "-", ":"]
    cos_threshold = np.arccos(1 - 2 * COS_ROT_THRESHOLD)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: potential
    ax = axes[0]
    for k, c, ls in zip(k_values, colors, styles):
        phi = potential_rot(dtheta, k_r=k)
        ax.plot(dtheta, phi, linewidth=2, color=c, linestyle=ls, label=f"k_r = {k}")
    ax.axvline(x=cos_threshold, color="green", linestyle=":", linewidth=1.5,
               label=f"Rot threshold (cos_err < {COS_ROT_THRESHOLD})")
    ax.axvline(x=-cos_threshold, color="green", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Δθ  (rad)")
    ax.set_ylabel("Φ_rot(s)")
    ax.set_title("Rotation Potential Shape  Φ_rot(Δθ) = exp(−k_r · (1−cos Δθ)/2)")
    ax.legend(fontsize=9)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-0.02, 1.05)

    # Right: gradient
    ax = axes[1]
    delta = 0.005
    for k, c, ls in zip(k_values, colors, styles):
        dtheta_next = dtheta - delta
        phi_now = potential_rot(dtheta, k_r=k)
        phi_next = potential_rot(dtheta_next, k_r=k)
        grad = np.abs(W_ROT * (phi_next - phi_now) / delta)
        ax.plot(dtheta, grad, linewidth=2, color=c, linestyle=ls, label=f"k_r = {k}")
    ax.axvline(x=cos_threshold, color="green", linestyle=":", linewidth=1.5,
               label=f"Rot threshold")
    ax.axvline(x=-cos_threshold, color="green", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Δθ  (rad)")
    ax.set_ylabel("|∂R/∂θ|  (gradient magnitude)")
    ax.set_title(f"Rotation Reward Gradient Magnitude  (w_rot={int(W_ROT)})")
    ax.legend(fontsize=9)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(bottom=0)

    fig.subplots_adjust(top=0.88)
    fig.suptitle("Hyperparameter Sensitivity: k_r  (rotation potential sharpness)",
                 fontsize=13, y=0.98)
    fig.savefig(out_dir / "b2_kr_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_b3_weight_scaling(out_dir: Path):
    """B3: Expected per-push reward at different w values."""
    print("  [B3] Weight scaling ...", flush=True)

    d = np.linspace(0.001, 0.45, 500)
    delta = 0.03  # typical push improvement
    d_next = np.clip(d - delta, 0.001, None)

    w_values = [1, 5, 10, 20]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(w_values)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: reward per push at different distances
    ax = axes[0]
    for w, c in zip(w_values, colors):
        r = pbrs_reward(d, d_next, w=w)
        ax.plot(d, r, linewidth=2, color=c, label=f"w = {w}")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.axvline(x=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")
    ax.set_xlabel("Distance from goal  d (m)")
    ax.set_ylabel(f"Reward for {delta*100:.0f} cm push")
    ax.set_title(f"PBRS Reward vs Distance for a {delta*100:.0f} cm improvement")
    ax.legend(fontsize=9)

    # Right: bar chart comparing key ranges
    ax = axes[1]
    scenarios = [
        ("Far-field\n(d=0.30→0.27 m)", 0.30, 0.27),
        ("Mid-range\n(d=0.15→0.12 m)", 0.15, 0.12),
        ("Near-goal\n(d=0.06→0.03 m)", 0.06, 0.03),
    ]
    x = np.arange(len(w_values))
    width = 0.2
    for idx, (label, d_p, d_n) in enumerate(scenarios):
        rewards = [pbrs_reward(d_p, d_n, w=w) for w in w_values]
        bars = ax.bar(x + idx * width, rewards, width, label=label, alpha=0.85)
    ax.set_xlabel("Scaling weight  w")
    ax.set_ylabel(f"Reward for {delta*100:.0f} cm push")
    ax.set_title("Reward Magnitude vs Scaling Weight  w")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(w) for w in w_values])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(y=0, color="gray", linewidth=0.8)

    fig.subplots_adjust(top=0.88)
    fig.suptitle("Hyperparameter Scaling: w_pos / w_rot  (any scalar multiple is valid PBRS)",
                 fontsize=13, y=0.98)
    fig.savefig(out_dir / "b3_weight_scaling.png", dpi=200)
    plt.close(fig)


def plot_b4_gamma_shaping(out_dir: Path):
    """B4: gamma_shaping = 1.0 vs 0.95 — near-goal sign inversion."""
    print("  [B4] gamma_shaping comparison ...", flush=True)

    d = np.linspace(0.005, 0.15, 300)
    delta = 0.01

    d_next = np.clip(d - delta, 0.001, None)
    phi_now = potential_pos(d, K_P)
    phi_next = potential_pos(d_next, K_P)

    # γ = 1.0: F = Φ(s') − Φ(s)
    r_gamma_1 = phi_next - phi_now

    # γ = 0.95: F = γ·Φ(s') − Φ(s)
    r_gamma_095 = 0.95 * phi_next - phi_now

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: raw rewards
    ax = axes[0]
    ax.plot(d, r_gamma_1, linewidth=2, color="#1f77b4",
            label="γ_shaping = 1.0  (Grzes & Kudenko 2009)")
    ax.plot(d, r_gamma_095, linewidth=2, color="#d62728", linestyle="--",
            label="γ_shaping = 0.95")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.axvline(x=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")
    ax.set_xlabel("Distance from goal  d (m)")
    ax.set_ylabel(f"PBRS reward for {delta*100:.0f} cm improvement")
    ax.set_title(f"PBRS Reward: γ=1.0 vs γ=0.95  (k_p={int(K_P)}, w={int(W_POS)})")
    ax.legend(fontsize=9)

    # Right: zoomed near-goal
    ax = axes[1]
    d_zoom = np.linspace(0.005, 0.06, 200)
    d_next_z = np.clip(d_zoom - delta, 0.001, None)
    phi_now_z = potential_pos(d_zoom, K_P)
    phi_next_z = potential_pos(d_next_z, K_P)
    r1_z = phi_next_z - phi_now_z
    r095_z = 0.95 * phi_next_z - phi_now_z

    ax.plot(d_zoom, r1_z, linewidth=2, color="#1f77b4",
            label="γ_shaping = 1.0  (always positive for progress)")
    ax.plot(d_zoom, r095_z, linewidth=2, color="#d62728", linestyle="--",
            label="γ_shaping = 0.95  (negative reward for progress near goal)")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.axvline(x=POS_THRESHOLD, color="green", linestyle=":", linewidth=1.5,
               label=f"Success threshold ({POS_THRESHOLD:.2f} m)")

    # Find sign-change point for γ=0.95
    sign_change_idx = np.argmax(r095_z < 0)
    if sign_change_idx > 0:
        d_sign = d_zoom[sign_change_idx]
        ax.axvline(x=d_sign, color="#d62728", linestyle=":", linewidth=1, alpha=0.6)
        ax.annotate(f"Sign flips at d≈{d_sign:.3f} m",
                    xy=(d_sign, 0), xytext=(0.035, -0.003),
                    arrowprops=dict(arrowstyle="->", color="#d62728"),
                    fontsize=8, color="#d62728")

    ax.set_xlabel("Distance from goal  d (m)")
    ax.set_ylabel(f"PBRS reward for {delta*100:.0f} cm improvement")
    ax.set_title("Zoom: Near-Goal — γ=0.95 Punishes Progress When d < ~0.02 m")
    ax.legend(fontsize=9)

    fig.subplots_adjust(top=0.88)
    fig.suptitle("Why gamma_shaping = 1.0: Near-Goal Sign Inversion with gamma < 1",
                 fontsize=13, y=0.98)
    fig.savefig(out_dir / "b4_gamma_shaping.png", dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Empirical PBRS component analysis (Step 4)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_empirical_pbrs_breakdown(all_data: list, labels: list, out_dir: Path):
    """Step 4: PBRS component decomposition and curriculum tracking."""
    if not all_data:
        return

    print("  [Empirical] PBRS component breakdown ...", flush=True)

    # PBRS component tags with aliases for different naming conventions
    pbrs_tag_aliases = [
        (["PBRS/DensePos", "Reward/Dense/Pos", "Reward/Bob/Dense/Pos"], "Dense Position", "#1f77b4"),
        (["PBRS/DenseRot", "Reward/Dense/Rot", "Reward/Bob/Dense/Rot"], "Dense Rotation", "#ff7f0e"),
        (["PBRS/CompletionRate"], "Completion Rate", "#2ca02c"),
        (["PBRS/RotBonusRate"], "Rotation Bonus Rate", "#d62728"),
        (["PBRS/TipRate"], "Tip Rate", "#9467bd"),
    ]

    curriculum_tag_aliases = [
        (["Curriculum/w_rot"], "w_rot", "#1f77b4"),
        (["Curriculum/phase"], "Phase", "#d62728"),
        (["Curriculum/pos_term_threshold"], "Pos Term Threshold", "#2ca02c"),
    ]

    colors_model = plt.cm.tab10(np.linspace(0, 1, max(len(all_data), 10)))

    # --- PBRS components ---
    for aliases, ylabel, _ in pbrs_tag_aliases:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        any_found = False
        for i, (data, lbl) in enumerate(zip(all_data, labels)):
            tag = _find_tag(data, aliases)
            if tag is None:
                continue
            any_found = True
            steps = data[tag]["steps"]
            values = data[tag]["values"]
            ax.plot(steps, values, alpha=0.15, color=colors_model[i])
            ax.plot(steps, smooth(values, 0.6), color=colors_model[i], linewidth=2, label=lbl)
        if not any_found:
            plt.close(fig)
            continue
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_name = ylabel.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(out_dir / f"empirical_{safe_name}.png", dpi=150)
        plt.close(fig)

    # --- Curriculum tracking ---
    for aliases, ylabel, _ in curriculum_tag_aliases:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        any_found = False
        for i, (data, lbl) in enumerate(zip(all_data, labels)):
            tag = _find_tag(data, aliases)
            if tag is None:
                continue
            any_found = True
            steps = data[tag]["steps"]
            values = data[tag]["values"]
            ax.plot(steps, values, alpha=0.15, color=colors_model[i])
            ax.plot(steps, smooth(values, 0.6), color=colors_model[i], linewidth=2, label=lbl)
        if not any_found:
            plt.close(fig)
            continue
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_name = ylabel.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(out_dir / f"empirical_{safe_name}.png", dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary markdown
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary_md(out_dir: Path, tb_loaded: bool):
    """Write a summary README describing every plot."""
    md = out_dir / "README.md"
    lines = [
        "# PBRS Analysis Plot Suite",
        "",
        "## Part A: Why PBRS is the right reward function",
        "",
        "| Plot | File | Description |",
        "|------|------|-------------|",
        "| A1 | `a1_potential_landscape.png` | 2D heatmap of Φ_pos(s) = exp(−k_p·d²) across the workspace. Shows smooth, bounded (0,1] potential with concentric gradients — no dead zones, no singularities. |",
        "| A2 | `a2_gradient_comparison.png` | Gradient magnitude dR/dd for PBRS vs old fractional formula vs old raw delta. PBRS gradient peaks at d=0.13 m (the sweet spot for learning), falls to 0 at d=0 (stable at goal, no noise amplification). Old formula 1/d_prev amplifies noise near goal. |",
        "| A3 | `a3_episode_simulation.png` | Side-by-side per-push reward comparison for a simulated 5-push episode (d: 0.20→0.08→...→0.01 m). PBRS produces smoothly diminishing returns; old formula amplifies near-goal rewards and includes penalty terms that shift the optimal policy. |",
        "| A4 | `a4_cosine_distance.png` | Cosine angular distance (1−cos Δθ)/2 vs _yaw_distance_rad. Cosine is C∞ smooth everywhere; yaw_distance has a cusp/gradient-discontinuity at Δθ=±π. PBRS uses the smooth cosine metric to eliminate gradient cliffs in the Euler-angle space. |",
    ]

    if tb_loaded:
        lines += [
            "| A5 | `a5_*.png` | Empirical learning curves from TensorBoard event files: Success Rate, Rotation SR, Position Error, Rotation Error, Mean Reward, Value Loss. One PNG per metric, all models overlaid. |",
            "",
            "## Empirical PBRS component analysis",
            "",
            "| Plot | File | Description |",
            "|------|------|-------------|",
            "| E1 | `empirical_PBRS_DensePos.png` | Mean dense position reward per iteration. Should be positive and stable, confirming PBRS provides consistent directional signal. |",
            "| E2 | `empirical_PBRS_DenseRot.png` | Mean dense rotation reward per iteration. |",
            "| E3 | `empirical_PBRS_CompletionRate.png` | Rate at which the position completion bonus (+5) fires. Should increase as position error decreases. |",
            "| E4 | `empirical_PBRS_RotBonusRate.png` | Rate at which the rotation bonus (+2) fires (both thresholds met). |",
            "| E5 | `empirical_PBRS_TipRate.png` | Rate of tip-over detections. Should be low and stable. |",
            "| E6 | `empirical_Curriculum_w_rot.png` | (Model B) Rotation weight ramp: 0 → 10 over curriculum iterations. |",
            "| E7 | `empirical_Curriculum_phase.png` | (Model B) Curriculum phase: 1 (position-only) → 2 (ramp active). |",
            "| E8 | `empirical_Curriculum_pos_term_threshold.png` | (Model B) Position termination threshold fade: 0.05 → 0.02 during curriculum ramp. |",
        ]

    lines += [
        "",
        "## Part B: Why these specific hyperparameters were chosen",
        "",
        "| Plot | File | Description |",
        "|------|------|-------------|",
        "| B1 | `b1_kp_sensitivity.png` | Potential shape and gradient for k_p ∈ {15, 30, 50}. k_p=15 is too flat (weak gradient beyond 0.15 m). k_p=50 is too sharp (Φ≈0.01 at d>0.30 m — dead zone). k_p=30 balances far-field signal (~0.07 at 0.30 m) with strong mid-range gradient (peak at d=0.13 m). |",
        "| B2 | `b2_kr_sensitivity.png` | Rotation potential shape and gradient for k_r ∈ {3, 5, 10}. k_r=5 maps the cos_rot_err=0.01 success threshold to Φ_rot≈0.95, matching the position potential at its threshold. Good gradient across typical rotation range (±36°). |",
        "| B3 | `b3_weight_scaling.png` | Expected per-push reward for w ∈ {1, 5, 10, 20}. w=10 yields ~[−2, +3] per push for typical distances — comparable magnitude to sparse bonuses (+5), so neither term dominates. Any scalar multiple is valid PBRS (policy invariance holds for all w). |",
        "| B4 | `b4_gamma_shaping.png` | γ_shaping = 1.0 vs γ_shaping = 0.95. With γ=0.95, the 5% discount tax exceeds the marginal potential improvement near the goal, causing sign inversion — the agent is penalized for approaching. γ=1.0 (Grzes & Kudenko 2009) preserves policy invariance for episodic MDPs. |",
        "",
        "## Reference",
        "",
        "- Ng et al. (1999): *Policy invariance under reward transformations: Theory and application to reward shaping*",
        "- Grzes & Kudenko (2009): *Theoretical and empirical analysis of reward shaping in reinforcement learning*",
        "",
        f"PBRS parameters: k_p={K_P}, k_r={K_R}, w_pos={W_POS}, w_rot={W_ROT}, γ_shaping=1.0",
        f"Old formula parameters: α={OLD_ALPHA}, β={OLD_BETA}, β_rot={OLD_BETA_ROT}",
    ]

    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] Summary written to {md}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PBRS reward-function justification plots for thesis."
    )
    parser.add_argument("--tb-dirs", nargs="*", type=str, default=None,
                        help="TensorBoard summary directories for empirical learning curves.")
    parser.add_argument("--labels", nargs="*", type=str, default=None,
                        help="Labels for each --tb-dirs (same order).")
    parser.add_argument("-o", "--out-dir", type=str, default="analysis/pbrs",
                        help="Output directory for plots.")
    parser.add_argument("--analytical-only", action="store_true", default=False,
                        help="Skip TensorBoard loading (analytical plots only).")
    parser.add_argument("--no-empirical", action="store_true", default=False,
                        help="Skip empirical PBRS component breakdowns.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(DEFAULT_STYLE)

    # ── Load TensorBoard data if requested ────────────────────────────────────
    all_data = []
    labels = []
    tb_loaded = False

    if args.tb_dirs and not args.analytical_only:
        for p in args.tb_dirs:
            sd = Path(p)
            if not sd.exists():
                print(f"[WARN] TB dir not found: {sd}")
                continue
            print(f"[TB] Loading {sd} ...")
            data = load_tensorboard_data(sd)
            if data:
                all_data.append(data)
                print(f"      Found {len(data)} scalar tags")
            else:
                print(f"      No scalar data found")

        if args.labels:
            labels = args.labels[:len(all_data)]
        else:
            labels = [f"Run {i+1}" for i in range(len(all_data))]

        tb_loaded = len(all_data) > 0
        if tb_loaded:
            print(f"[TB] Loaded {len(all_data)} run(s)\n")
    elif args.tb_dirs and args.analytical_only:
        print("[INFO] --analytical-only set — skipping TensorBoard\n")

    # ── Part A: Why PBRS ──────────────────────────────────────────────────────
    print("=" * 72)
    print("PART A: Why PBRS is the right reward function")
    print("=" * 72)

    plot_a1_potential_landscape(out_dir)
    plot_a2_gradient_comparison(out_dir)
    plot_a3_episode_simulation(out_dir)
    plot_a4_cosine_distance(out_dir)
    if tb_loaded:
        plot_a5_tb_learning_curves(all_data, labels, out_dir)

    # ── Part B: Why these hyperparameters ─────────────────────────────────────
    print("=" * 72)
    print("PART B: Why these specific hyperparameters were chosen")
    print("=" * 72)

    plot_b1_kp_sensitivity(out_dir)
    plot_b2_kr_sensitivity(out_dir)
    plot_b3_weight_scaling(out_dir)
    plot_b4_gamma_shaping(out_dir)

    # ── Empirical PBRS breakdown ──────────────────────────────────────────────
    if tb_loaded and not args.no_empirical:
        print("=" * 72)
        print("Empirical: PBRS component analysis")
        print("=" * 72)
        plot_empirical_pbrs_breakdown(all_data, labels, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary_md(out_dir, tb_loaded)

    print(f"\n[INFO] All plots saved to: {out_dir.resolve()}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
