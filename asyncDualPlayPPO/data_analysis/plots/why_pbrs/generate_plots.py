#!/usr/bin/env python3
"""Generate 4 thesis plots: Why PBRS is the right choice for robotic pushing.

Plot 1: Position Error — Classic Dense (rel) vs PBRS-A (cropped to common budget)
Plot 2: Position + Rotation Error (dual panel) — multi-objective convergence
Plot 3: Mean Reward — reward signal quality comparison
Plot 4: Validation Aggregate SR — held-out test performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")

TB_DIRS = {
    "classic_rel": os.path.join(RUNS, "ppo_classic_reward/hpc_push_2048env_rel_full/summary"),
    "pbrs_A": os.path.join(RUNS, "long_runs/hpc_pbrs_simp_528env/summary"),
}

ENV_COUNTS = {"classic_rel": 2048, "pbrs_A": 528}
PUSHES_PER_ROLLOUT = {"classic_rel": 15, "pbrs_A": 15}

COMMON_BUDGET = 16.1e6

VALIDATION_CSVS = {
    "PBRS-A\n(Simple)": os.path.join(
        RUNS, "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_simp_528env/results_simp.csv"),
    "PBRS-B\n(Curriculum)": os.path.join(
        RUNS, "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_curr_528env/results_curr.csv"),
    "PBRS-C\n(ASP)": os.path.join(
        RUNS, "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_asp_528env/results_asp.csv"),
}

COLORS = {
    "classic_rel": "#d62728",
    "pbrs_A": "#2ca02c",
}

LABELS = {
    "classic_rel": "Ad-hoc Dense Reward",
    "pbrs_A": "PBRS",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_scalar(run_key, tag):
    ea = EventAccumulator(TB_DIRS[run_key])
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def ema_smooth(vals, alpha=0.03):
    s = np.empty_like(vals)
    s[0] = vals[0]
    for i in range(1, len(vals)):
        s[i] = alpha * vals[i] + (1 - alpha) * s[i - 1]
    return s


def to_env_steps(steps, run_key):
    return steps.astype(np.float64) * ENV_COUNTS[run_key] * PUSHES_PER_ROLLOUT[run_key]


def crop_to_budget(steps, vals, run_key, budget=COMMON_BUDGET):
    env_steps = to_env_steps(steps, run_key)
    mask = env_steps <= budget
    return env_steps[mask], vals[mask]


def _plot_pair(ax, tag, crop=True):
    for key in ["classic_rel", "pbrs_A"]:
        steps, vals = load_scalar(key, tag)
        if steps is None:
            continue
        if crop:
            env_s, v = crop_to_budget(steps, vals, key)
        else:
            env_s = to_env_steps(steps, key)
            v = vals
        smoothed = ema_smooth(v, alpha=0.03)
        ax.plot(env_s / 1e6, v, alpha=0.10, color=COLORS[key], lw=0.8)
        ax.plot(env_s / 1e6, smoothed, color=COLORS[key], label=LABELS[key])


def plot1_position_error():
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_pair(ax, "Metrics/PosError")

    ax.axhline(y=0.05, color="grey", ls="--", lw=1, alpha=0.6, label="Success threshold (0.05 m)")

    s_c, v_c = load_scalar("classic_rel", "Metrics/PosError")
    s_p, v_p = load_scalar("pbrs_A", "Metrics/PosError")
    es_c, vc = crop_to_budget(s_c, v_c, "classic_rel")
    es_p, vp = crop_to_budget(s_p, v_p, "pbrs_A")
    final_c = np.mean(ema_smooth(vc, 0.03)[-20:])
    final_p = np.mean(ema_smooth(vp, 0.03)[-20:])
    ax.annotate(f"{final_c:.3f} m", xy=(es_c[-1]/1e6, final_c),
                xytext=(15, 8), textcoords="offset points",
                fontsize=10, color=COLORS["classic_rel"],
                arrowprops=dict(arrowstyle="-", color=COLORS["classic_rel"], lw=0.8))
    ax.annotate(f"{final_p:.3f} m", xy=(es_p[-1]/1e6, final_p),
                xytext=(15, -12), textcoords="offset points",
                fontsize=10, color=COLORS["pbrs_A"],
                arrowprops=dict(arrowstyle="-", color=COLORS["pbrs_A"], lw=0.8))

    ax.set_xlabel("Environment Pushes (millions)")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_title("Position Error: Ad-hoc Dense vs. PBRS\n(equal compute budget: 16.1M pushes)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    out = os.path.join(OUT_DIR, "1_position_error.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot2_pos_rot_error():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    _plot_pair(ax, "Metrics/PosError")
    ax.axhline(y=0.05, color="grey", ls="--", lw=1, alpha=0.6, label="Threshold (0.05 m)")
    ax.set_xlabel("Environment Pushes (millions)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Error")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    _plot_pair(ax, "Metrics/RotError")
    ax.set_xlabel("Environment Pushes (millions)")
    ax.set_ylabel("Rotation Error")
    ax.set_title("Rotation Error")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=0)

    s_c, v_c = load_scalar("classic_rel", "Metrics/RotError")
    s_p, v_p = load_scalar("pbrs_A", "Metrics/RotError")
    _, vc = crop_to_budget(s_c, v_c, "classic_rel")
    _, vp = crop_to_budget(s_p, v_p, "pbrs_A")
    fc = np.mean(ema_smooth(vc, 0.03)[-20:])
    fp = np.mean(ema_smooth(vp, 0.03)[-20:])
    ratio = fc / max(fp, 1e-6)
    axes[1].text(0.50, 0.92, f"Ad-hoc: {fc:.2f}  |  PBRS: {fp:.2f}  ({ratio:.1f}x gap)",
                 transform=axes[1].transAxes, fontsize=10, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6))

    fig.suptitle("Multi-Objective Convergence: Ad-hoc Dense vs. PBRS\n"
                 "(equal compute budget: 16.1M pushes)", fontsize=14, y=1.04)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "2_pos_rot_error.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot3_reward_signal():
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_pair(ax, "Reward/Mean")

    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xlabel("Environment Pushes (millions)")
    ax.set_ylabel("Mean Reward per Push")
    ax.set_title("Reward Signal Quality: Ad-hoc Dense vs. PBRS\n"
                 "(equal compute budget: 16.1M pushes)")
    ax.legend(loc="upper left")

    s_c, v_c = load_scalar("classic_rel", "Reward/Mean")
    s_p, v_p = load_scalar("pbrs_A", "Reward/Mean")
    _, vc = crop_to_budget(s_c, v_c, "classic_rel")
    _, vp = crop_to_budget(s_p, v_p, "pbrs_A")
    std_c = np.std(vc[-100:])
    std_p = np.std(vp[-100:])
    mean_c = np.mean(ema_smooth(vc, 0.03)[-20:])
    mean_p = np.mean(ema_smooth(vp, 0.03)[-20:])
    ax.text(0.98, 0.15,
            f"Ad-hoc: mean={mean_c:.2f}, std={std_c:.2f}\n"
            f"PBRS:   mean={mean_p:.2f}, std={std_p:.2f}",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    out = os.path.join(OUT_DIR, "3_reward_signal.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot4_validation():
    dfs = {}
    for label, path in VALIDATION_CSVS.items():
        if os.path.isfile(path):
            dfs[label] = pd.read_csv(path)

    if not dfs:
        print("  [SKIP] No validation CSVs found.")
        return

    model_colors = {
        "PBRS-A\n(Simple)": "#2ca02c",
        "PBRS-B\n(Curriculum)": "#1f77b4",
        "PBRS-C\n(ASP)": "#e377c2",
    }

    totals = {}
    for mname, df in dfs.items():
        pos_only = df[df["test_type"] == "pos_only"]
        pos_rot = df[df["test_type"] == "pos_rot"]
        totals[mname] = {
            "Overall": df["success"].mean(),
            "Position-Only": pos_only["success"].mean() if len(pos_only) > 0 else 0.0,
            "Position+Rotation": pos_rot["success"].mean() if len(pos_rot) > 0 else 0.0,
            "Avg Pos Error (m)": df["pos_err"].mean(),
        }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    categories = ["Overall", "Position-Only", "Position+Rotation"]
    x = np.arange(len(categories))
    bar_w = 0.25
    for i, (mname, t) in enumerate(totals.items()):
        vals = [t[c] for c in categories]
        offset = (i - len(totals) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=mname,
                      color=model_colors.get(mname, f"C{i}"),
                      edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.02, f"{v:.0%}",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Success Rate")
    ax.set_title("Validation SR by Test Type\n(20 held-out scenarios)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[1]
    test_names = list(dfs[list(dfs.keys())[0]]["test_name"])
    n_tests = len(test_names)
    xt = np.arange(n_tests)
    for i, (mname, df) in enumerate(dfs.items()):
        success = df["success"].values.astype(float)
        offset = (i - len(dfs) / 2 + 0.5) * bar_w
        ax.bar(xt + offset, success, bar_w, label=mname,
               color=model_colors.get(mname, f"C{i}"),
               edgecolor="white", linewidth=0.5)

    bg = {"E_": "#e8f5e9", "M_": "#fff3e0", "H_": "#ffebee"}
    for j, tn in enumerate(test_names):
        for prefix, c in bg.items():
            if tn.startswith(prefix):
                ax.axvspan(j - 0.4, j + 0.4, alpha=0.18, color=c, zorder=0)
                break

    ax.set_xticks(xt)
    ax.set_xticklabels(test_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Success (0/1)")
    ax.set_title("Per-Scenario Success\n(green=easy, orange=medium, red=hard)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, 1.35)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "4_validation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating 4 thesis plots: Why PBRS\n")

    print("[1/4] Position Error: Ad-hoc Dense vs PBRS")
    plot1_position_error()

    print("[2/4] Position + Rotation Error (dual panel)")
    plot2_pos_rot_error()

    print("[3/4] Reward Signal Quality")
    plot3_reward_signal()

    print("[4/4] Validation Results")
    plot4_validation()

    print(f"\nAll plots saved to: {OUT_DIR}")
