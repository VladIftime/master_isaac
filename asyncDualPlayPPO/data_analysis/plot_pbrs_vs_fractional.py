"""
PBRS vs Fractional Improvement reward comparison plots.

Generates 4 figures demonstrating why Potential-Based Reward Shaping (PBRS)
is the correct dense reward choice compared to the fractional improvement
formula.  All constants match the codebase exactly.

Fractional improvement (wrapper_push.py):
    R = alpha * (d_prev - d_now) / clamp(d_prev, 0.01)
      - beta  * d_now
    alpha=3.0, beta=0.5

PBRS (reward_pbrs.py):
    Phi(d)  = exp(-k * d^2),   k_pos=30, k_rot=5
    F       = w * (Phi(d_now) - Phi(d_prev)),   w_pos=w_rot=10.0

Usage:
    python -m asyncDualPlayPPO.data_analysis.plot_pbrs_vs_fractional
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ALPHA = 3.0
BETA = 0.5
CLAMP_MIN = 0.01
POS_IMP_CLAMP = (-5.0, 5.0)
PEN_CLAMP = (-2.0, 0.0)

K_POS = 30.0
K_ROT = 5.0
W_POS = 10.0
W_ROT = 10.0


def phi_pos(d):
    return np.exp(-K_POS * d ** 2)


def phi_rot(r):
    return np.exp(-K_ROT * r ** 2)


def pbrs_pos(d_prev, d_now):
    return W_POS * (phi_pos(d_now) - phi_pos(d_prev))


def frac_pos_imp(d_prev, d_now):
    denom = np.maximum(d_prev, CLAMP_MIN)
    raw = ALPHA * (d_prev - d_now) / denom
    return np.clip(raw, *POS_IMP_CLAMP)


def frac_penalty(d_now):
    return np.clip(-BETA * d_now, *PEN_CLAMP)


def frac_total(d_prev, d_now):
    return frac_pos_imp(d_prev, d_now) + frac_penalty(d_now)


def _style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
    })


def plot1_potential_functions():
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    d = np.linspace(0, 0.50, 500)
    r = np.linspace(0, 2.0, 500)

    ax1.plot(d, phi_pos(d), color="#2196F3", label=r"$\Phi_{\mathrm{pos}}(d) = e^{-30\,d^2}$")
    ax1.plot(r, phi_rot(r), color="#FF9800", label=r"$\Phi_{\mathrm{rot}}(r) = e^{-5\,r^2}$")
    ax1.set_xlabel("Distance  (m for pos, rad for rot)")
    ax1.set_ylabel(r"Potential  $\Phi$")
    ax1.set_title("(a)  PBRS Potential Functions")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    grad_pos = 2 * K_POS * d * np.exp(-K_POS * d ** 2)
    grad_rot = 2 * K_ROT * r * np.exp(-K_ROT * r ** 2)
    ax2.plot(d, grad_pos, color="#2196F3",
             label=r"$|\partial\Phi_{\mathrm{pos}}/\partial d|$  (pos)")
    ax2.plot(r, grad_rot, color="#FF9800",
             label=r"$|\partial\Phi_{\mathrm{rot}}/\partial r|$  (rot)")
    ax2.set_xlabel("Distance  (m for pos, rad for rot)")
    ax2.set_ylabel("Gradient magnitude")
    ax2.set_title("(b)  Potential Gradient (reward sensitivity)")
    ax2.legend()

    fig.suptitle("Plot 1:  PBRS Potential Functions and Gradients",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "potential_functions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot2_reward_signal_comparison():
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    d_prev = np.linspace(0.02, 0.50, 500)
    delta_d = 0.01
    d_now = d_prev - delta_d

    r_pbrs = pbrs_pos(d_prev, d_now)
    r_frac_only = frac_pos_imp(d_prev, d_now)
    r_frac_total = frac_total(d_prev, d_now)

    ax.plot(d_prev, r_pbrs, color="#2196F3", linewidth=2.5,
            label="PBRS dense")
    ax.plot(d_prev, r_frac_only, color="#4CAF50", linestyle="--",
            label="Fractional improvement only")
    ax.plot(d_prev, r_frac_total, color="#F44336", linewidth=2.5,
            label="Fractional + distance penalty")

    neg_mask = r_frac_total < 0
    if neg_mask.any():
        ax.fill_between(d_prev, r_frac_total, 0, where=neg_mask,
                         color="#F44336", alpha=0.15,
                         label="Negative reward despite progress")
        cross_idx = np.where(np.diff(neg_mask.astype(int)))[0]
        if len(cross_idx) > 0:
            cx = d_prev[cross_idx[0]]
            ax.axvline(cx, color="#F44336", linestyle=":", alpha=0.5)
            ax.annotate(f"Crossover\nd = {cx:.2f} m",
                        xy=(cx, 0), xytext=(cx + 0.04, 0.15),
                        arrowprops=dict(arrowstyle="->", color="#F44336"),
                        fontsize=9, color="#F44336")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r"Starting distance  $d_{\mathrm{prev}}$  (m)")
    ax.set_ylabel("Reward for 1 cm improvement")
    ax.set_title("Plot 2:  Reward Signal for Equal Progress "
                 r"($\Delta d = 0.01$ m)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "reward_signal_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot3_near_goal_singularity():
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    d_prev = np.linspace(0.002, 0.15, 1000)
    delta_d = 0.002
    d_now = np.maximum(d_prev - delta_d, 0.0)

    r_pbrs = pbrs_pos(d_prev, d_now)
    r_frac = frac_pos_imp(d_prev, d_now)

    ax.plot(d_prev, r_pbrs, color="#2196F3", linewidth=2.5,
            label="PBRS dense")
    ax.plot(d_prev, r_frac, color="#F44336", linewidth=2.5,
            label="Fractional improvement")

    ax.axvline(CLAMP_MIN, color="gray", linestyle="--", alpha=0.7,
               linewidth=1.5, label=f"Clamp boundary ({CLAMP_MIN} m)")

    ax.fill_betweenx([ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else 0,
                       max(r_frac.max(), r_pbrs.max()) * 1.1],
                      0, CLAMP_MIN, color="gray", alpha=0.08)

    ax.annotate("Clamp active:\nreward saturates",
                xy=(CLAMP_MIN / 2, r_frac[d_prev < CLAMP_MIN].mean()),
                xytext=(0.03, r_frac.max() * 0.8),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    ax.set_xlabel(r"Starting distance  $d_{\mathrm{prev}}$  (m)")
    ax.set_ylabel(r"Reward for 2 mm improvement")
    ax.set_title("Plot 3:  Near-Goal Behaviour "
                 r"($\Delta d = 0.002$ m)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "near_goal_singularity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot4_cycle_invariance():
    _style()
    trajectory = [0.30, 0.20, 0.10, 0.20, 0.30]
    step_labels = [f"{trajectory[i]:.2f} -> {trajectory[i+1]:.2f}"
                   for i in range(len(trajectory) - 1)]
    n_steps = len(trajectory) - 1

    pbrs_rewards = []
    frac_rewards = []
    for i in range(n_steps):
        dp, dn = trajectory[i], trajectory[i + 1]
        pbrs_rewards.append(pbrs_pos(dp, dn))
        frac_rewards.append(frac_total(dp, dn))

    pbrs_cum = np.cumsum(pbrs_rewards)
    frac_cum = np.cumsum(frac_rewards)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(n_steps)
    w = 0.35
    bars_p = ax1.bar(x - w / 2, pbrs_rewards, w, color="#2196F3", label="PBRS")
    bars_f = ax1.bar(x + w / 2, frac_rewards, w, color="#F44336",
                     label="Fractional + penalty")
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(step_labels, fontsize=9)
    ax1.set_xlabel("Transition  (d_prev -> d_now)")
    ax1.set_ylabel("Per-step reward")
    ax1.set_title("(a)  Per-Step Reward")
    ax1.legend()
    for bar, val in zip(bars_p, pbrs_rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:+.2f}", ha="center", va="bottom", fontsize=8,
                 color="#2196F3")
    for bar, val in zip(bars_f, frac_rewards):
        y_off = 0.05 if val >= 0 else -0.15
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                 f"{val:+.2f}", ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=8, color="#F44336")

    steps = np.arange(1, n_steps + 1)
    ax2.plot(steps, pbrs_cum, "o-", color="#2196F3", markersize=8,
             linewidth=2.5, label="PBRS cumulative")
    ax2.plot(steps, frac_cum, "s-", color="#F44336", markersize=8,
             linewidth=2.5, label="Fractional cumulative")
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    ax2.annotate(f"PBRS = {pbrs_cum[-1]:+.4f}\n(zero-sum)",
                 xy=(n_steps, pbrs_cum[-1]),
                 xytext=(n_steps - 0.8, pbrs_cum.max() * 0.5),
                 arrowprops=dict(arrowstyle="->", color="#2196F3"),
                 fontsize=10, color="#2196F3", fontweight="bold")
    ax2.annotate(f"Frac = {frac_cum[-1]:+.2f}\n(nonzero residual)",
                 xy=(n_steps, frac_cum[-1]),
                 xytext=(n_steps - 1.5, frac_cum[-1] - 0.8),
                 arrowprops=dict(arrowstyle="->", color="#F44336"),
                 fontsize=10, color="#F44336", fontweight="bold")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative reward")
    ax2.set_title("(b)  Cumulative Reward over Round Trip")
    ax2.set_xticks(steps)
    ax2.legend()

    traj_str = " -> ".join(f"{t:.2f}" for t in trajectory)
    fig.suptitle(f"Plot 4:  Cycle Invariance  (trajectory: d = {traj_str} m)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "cycle_invariance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    print(f"Saving plots to {PLOT_DIR}/\n")
    plot1_potential_functions()
    plot2_reward_signal_comparison()
    plot3_near_goal_singularity()
    plot4_cycle_invariance()
    print("\nDone.")
