#!/usr/bin/env python3
"""Generate thesis plots: Why PBRS is the right reward for 1-arm push-primitive RL.

Data vintage: 26.06.20 analysis bundle (anal_26.06.18) for the PBRS models, plus
the original ad-hoc-dense classic baseline TensorBoard run.

Headline claim — PBRS wins for a *plain single-agent PPO agent with no curriculum*:
  Plot 1: Position Error    — Ad-hoc Dense PPO vs PBRS PPO (Model A, single-agent)
  Plot 2: Position+Rotation — multi-objective convergence, same ablation
  Plot 3: Mean Reward       — reward-signal quality (mean + variance), same ablation

Supporting / scoping evidence:
  Plot 4: Validation SR     — held-out generalization of PBRS models (A/B/C)
  Plot 5: Curriculum-axis scoping contrast — SuccessRate & RotationSR across
          A (single-agent), B (forced curriculum), C (ASP), D (ASP, GE-ablated).
          Shows the reward is sound (A/B learn) while the self-play curriculum
          is the separate hard problem (C/D Bob SR stays near zero).

The only variable changed between the Plot 1-3 curves is the reward function:
same UR5e + T-block scene, same 4D push primitive, same PPO+LSTM agent, no
curriculum in either arm of the ablation.
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

# ── Data sources ──────────────────────────────────────────────────────────────
# Ad-hoc-dense classic baseline: original TensorBoard summary (no 26.06.20 rerun).
TB_DIRS = {
    "classic_rel": os.path.join(RUNS, "ppo_classic_reward/hpc_push_2048env_rel_full/summary"),
}

# PBRS models: pre-extracted CSVs from the 26.06.20 analysis bundle.
ANAL_CSV = os.path.join(
    RUNS, "ppo_pbrs_reward/26.06.20/runs/anal_26.06.18/csv")

PBRS_CSV_PREFIX = {
    "pbrs_A": "Push_PPO-Si",       # Model A: single-agent PPO + PBRS, NO curriculum
    "pbrs_B": "Push_PPO-Cr",       # Model B: single-agent PPO + PBRS + forced curriculum
    "pbrs_C": "Push_PPO-ASP",      # Model C: ASP self-play, Bob uses PBRS
    "pbrs_D": "Push_PPO-ASP-NGE",  # Model D: Model C with GoalEncoder ablated
}
SINGLE_AGENT = {"pbrs_A", "pbrs_B"}  # use Metrics_* tags; ASP models use Metrics_Bob_*

ENV_COUNTS = {"classic_rel": 2048, "pbrs_A": 528, "pbrs_B": 528, "pbrs_C": 528, "pbrs_D": 528}
PUSHES_PER_ROLLOUT = 15  # same LSTM temporal window across all runs

COMMON_BUDGET = 16.1e6   # equal-compute crop for the headline ablation (env-pushes)

# Validation CSVs — A/B/C held-out generalization (no newer A/B validation in 26.06.20).
VALIDATION_CSVS = {
    "PBRS-A\n(Single-agent)": os.path.join(
        RUNS, "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_simp_528env/results_simp.csv"),
    "PBRS-B\n(Curriculum)": os.path.join(
        RUNS, "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_curr_528env/results_curr.csv"),
    "PBRS-C\n(ASP)": os.path.join(
        RUNS, "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_asp_528env/results_asp.csv"),
}

# Canonical metric -> TensorBoard tag (classic baseline only).
TB_TAG = {
    "pos": "Metrics/PosError",
    "rot": "Metrics/RotError",
    "reward": "Reward/Mean",
    "sr": "Metrics/SuccessRate",
    "rotsr": "Metrics/RotationSR",
}


def _csv_suffix(key, canonical):
    bob = key not in SINGLE_AGENT
    table = {
        "pos":    "Metrics_Bob_PosError"    if bob else "Metrics_PosError",
        "rot":    "Metrics_Bob_RotError"    if bob else "Metrics_RotError",
        "reward": "Reward_Bob"              if bob else "Reward_Mean",
        "sr":     "Metrics_Bob_SuccessRate" if bob else "Metrics_SuccessRate",
        "rotsr":  "Metrics_Bob_RotationSR"  if bob else "Metrics_RotationSR",
    }
    return table[canonical]


COLORS = {
    "classic_rel": "#d62728",
    "pbrs_A": "#2ca02c",
    "pbrs_B": "#1f77b4",
    "pbrs_C": "#e377c2",
    "pbrs_D": "#9467bd",
}

LABELS = {
    "classic_rel": "Ad-hoc Dense Reward (single-agent, no curriculum)",
    "pbrs_A": "PBRS (single-agent, no curriculum)",
    "pbrs_B": "PBRS-B (curriculum)",
    "pbrs_C": "PBRS-C (ASP)",
    "pbrs_D": "PBRS-D (ASP, GE-ablated)",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
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


# ── Loaders ───────────────────────────────────────────────────────────────────
def _load_tb(run_dir, tag):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def load_series(key, canonical):
    """Unified loader: classic baseline from TensorBoard, PBRS models from CSV."""
    if key == "classic_rel":
        return _load_tb(TB_DIRS[key], TB_TAG[canonical])
    path = os.path.join(ANAL_CSV, f"{PBRS_CSV_PREFIX[key]}_{_csv_suffix(key, canonical)}.csv")
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    return df["step"].to_numpy(), df["value"].to_numpy()


def ema_smooth(vals, alpha=0.03):
    s = np.empty_like(vals, dtype=np.float64)
    s[0] = vals[0]
    for i in range(1, len(vals)):
        s[i] = alpha * vals[i] + (1 - alpha) * s[i - 1]
    return s


def to_env_steps(steps, run_key):
    return steps.astype(np.float64) * ENV_COUNTS[run_key] * PUSHES_PER_ROLLOUT


def crop_to_budget(steps, vals, run_key, budget=COMMON_BUDGET):
    env_steps = to_env_steps(steps, run_key)
    mask = env_steps <= budget
    return env_steps[mask], vals[mask]


def _final(vals, n=20):
    return float(np.mean(ema_smooth(vals, 0.03)[-n:]))


# ── Headline ablation: classic ad-hoc vs PBRS single-agent (Model A) ──────────
def _plot_headline_pair(ax, canonical, crop=True):
    for key in ["classic_rel", "pbrs_A"]:
        steps, vals = load_series(key, canonical)
        if steps is None:
            print(f"  [WARN] missing series {key}/{canonical}")
            continue
        if crop:
            env_s, v = crop_to_budget(steps, vals, key)
        else:
            env_s, v = to_env_steps(steps, key), vals
        ax.plot(env_s / 1e6, v, alpha=0.10, color=COLORS[key], lw=0.8)
        ax.plot(env_s / 1e6, ema_smooth(v, 0.03), color=COLORS[key], label=LABELS[key])


def plot1_position_error():
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_headline_pair(ax, "pos")
    ax.axhline(y=0.05, color="grey", ls="--", lw=1, alpha=0.6, label="Success threshold (0.05 m)")

    s_c, v_c = load_series("classic_rel", "pos")
    s_p, v_p = load_series("pbrs_A", "pos")
    es_c, vc = crop_to_budget(s_c, v_c, "classic_rel")
    es_p, vp = crop_to_budget(s_p, v_p, "pbrs_A")
    final_c, final_p = _final(vc), _final(vp)
    ax.annotate(f"{final_c:.3f} m", xy=(es_c[-1] / 1e6, final_c),
                xytext=(15, 8), textcoords="offset points", fontsize=10,
                color=COLORS["classic_rel"],
                arrowprops=dict(arrowstyle="-", color=COLORS["classic_rel"], lw=0.8))
    ax.annotate(f"{final_p:.3f} m", xy=(es_p[-1] / 1e6, final_p),
                xytext=(15, -12), textcoords="offset points", fontsize=10,
                color=COLORS["pbrs_A"],
                arrowprops=dict(arrowstyle="-", color=COLORS["pbrs_A"], lw=0.8))

    ax.set_xlabel("Environment Pushes (millions)")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_title("Position Error: Ad-hoc Dense vs. PBRS\n"
                 "(plain single-agent PPO, no curriculum; equal budget 16.1M pushes)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    out = os.path.join(OUT_DIR, "1_position_error.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved {out}  (ad-hoc {final_c:.3f} m | PBRS {final_p:.3f} m)")
    return final_c, final_p


def plot2_pos_rot_error():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    _plot_headline_pair(axes[0], "pos")
    axes[0].axhline(y=0.05, color="grey", ls="--", lw=1, alpha=0.6, label="Threshold (0.05 m)")
    axes[0].set_xlabel("Environment Pushes (millions)")
    axes[0].set_ylabel("Position Error (m)")
    axes[0].set_title("Position Error")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].set_ylim(bottom=0)

    _plot_headline_pair(axes[1], "rot")
    axes[1].set_xlabel("Environment Pushes (millions)")
    axes[1].set_ylabel("Rotation Error")
    axes[1].set_title("Rotation Error")
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].set_ylim(bottom=0)

    s_c, v_c = load_series("classic_rel", "rot")
    s_p, v_p = load_series("pbrs_A", "rot")
    _, vc = crop_to_budget(s_c, v_c, "classic_rel")
    _, vp = crop_to_budget(s_p, v_p, "pbrs_A")
    fc, fp = _final(vc), _final(vp)
    ratio = fc / max(fp, 1e-6)
    axes[1].text(0.50, 0.92, f"Ad-hoc: {fc:.2f}  |  PBRS: {fp:.2f}  ({ratio:.1f}x gap)",
                 transform=axes[1].transAxes, fontsize=10, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6))

    fig.suptitle("Multi-Objective Convergence: Ad-hoc Dense vs. PBRS\n"
                 "(plain single-agent PPO, no curriculum; equal budget 16.1M pushes)",
                 fontsize=14, y=1.04)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "2_pos_rot_error.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved {out}  (rot ad-hoc {fc:.2f} | PBRS {fp:.2f} | {ratio:.1f}x)")
    return fc, fp, ratio


def plot3_reward_signal():
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_headline_pair(ax, "reward")
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xlabel("Environment Pushes (millions)")
    ax.set_ylabel("Mean Reward per Push")
    ax.set_title("Reward Signal Quality: Ad-hoc Dense vs. PBRS\n"
                 "(plain single-agent PPO, no curriculum; equal budget 16.1M pushes)")
    ax.legend(loc="upper left")

    s_c, v_c = load_series("classic_rel", "reward")
    s_p, v_p = load_series("pbrs_A", "reward")
    _, vc = crop_to_budget(s_c, v_c, "classic_rel")
    _, vp = crop_to_budget(s_p, v_p, "pbrs_A")
    std_c, std_p = float(np.std(vc[-100:])), float(np.std(vp[-100:]))
    mean_c, mean_p = _final(vc), _final(vp)
    ax.text(0.98, 0.15,
            f"Ad-hoc: mean={mean_c:.2f}, std={std_c:.2f}\n"
            f"PBRS:   mean={mean_p:.2f}, std={std_p:.2f}",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    out = os.path.join(OUT_DIR, "3_reward_signal.png")
    fig.savefig(out); plt.close(fig)
    print(f"  Saved {out}  (ad-hoc mean={mean_c:.2f} std={std_c:.2f} | "
          f"PBRS mean={mean_p:.2f} std={std_p:.2f})")
    return (mean_c, std_c), (mean_p, std_p)


def plot4_validation():
    dfs = {}
    for label, path in VALIDATION_CSVS.items():
        if os.path.isfile(path):
            dfs[label] = pd.read_csv(path)
    if not dfs:
        print("  [SKIP] No validation CSVs found.")
        return None

    model_colors = {
        "PBRS-A\n(Single-agent)": "#2ca02c",
        "PBRS-B\n(Curriculum)": "#1f77b4",
        "PBRS-C\n(ASP)": "#e377c2",
    }

    totals = {}
    for mname, df in dfs.items():
        pos_only = df[df["test_type"] == "pos_only"]
        pos_rot = df[df["test_type"] == "pos_rot"]
        totals[mname] = {
            "Overall": df["success"].mean(),
            "Position-Only": pos_only["success"].mean() if len(pos_only) else 0.0,
            "Position+Rotation": pos_rot["success"].mean() if len(pos_rot) else 0.0,
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
        ax.bar(x + offset, vals, bar_w, label=mname,
               color=model_colors.get(mname, f"C{i}"), edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Success Rate")
    ax.set_title("Validation SR by Test Type\n(20 held-out scenarios)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[1]
    test_names = list(dfs[list(dfs.keys())[0]]["test_name"])
    xt = np.arange(len(test_names))
    for i, (mname, df) in enumerate(dfs.items()):
        success = df["success"].values.astype(float)
        offset = (i - len(dfs) / 2 + 0.5) * bar_w
        ax.bar(xt + offset, success, bar_w, label=mname,
               color=model_colors.get(mname, f"C{i}"), edgecolor="white", linewidth=0.5)
    bg = {"E_": "#e8f5e9", "M_": "#fff3e0", "H_": "#ffebee"}
    for j, tn in enumerate(test_names):
        for prefix, c in bg.items():
            if tn.startswith(prefix):
                ax.axvspan(j - 0.4, j + 0.4, alpha=0.18, color=c, zorder=0)
                break
    ax.set_xticks(xt); ax.set_xticklabels(test_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Success (0/1)")
    ax.set_title("Per-Scenario Success\n(green=easy, orange=medium, red=hard)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, 1.35)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "4_validation.png")
    fig.savefig(out); plt.close(fig)
    print("  Saved " + out + "  " +
          " | ".join(f"{m.splitlines()[0]} overall={t['Overall']:.0%}" for m, t in totals.items()))
    return totals


# ── Scoping contrast: reward is sound (A/B) vs curriculum is hard (C/D) ────────
def plot5_scoping_contrast():
    keys = ["pbrs_A", "pbrs_B", "pbrs_C", "pbrs_D"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    finals = {}

    for canonical, ax, title in [("sr", axes[0], "Success Rate (pos AND rot)"),
                                 ("rotsr", axes[1], "Rotation Success Rate")]:
        for key in keys:
            steps, vals = load_series(key, canonical)
            if steps is None:
                print(f"  [WARN] missing {key}/{canonical}")
                continue
            env_s = to_env_steps(steps, key)
            ax.plot(env_s / 1e6, vals, alpha=0.12, color=COLORS[key], lw=0.8)
            ax.plot(env_s / 1e6, ema_smooth(vals, 0.03), color=COLORS[key], label=LABELS[key])
            if canonical == "sr":
                finals[key] = _final(vals)
        ax.set_xlabel("Environment Pushes (millions)")
        ax.set_ylabel(title.split(" (")[0])
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(bottom=0)

    if "pbrs_C" in finals:
        axes[0].text(0.97, 0.05,
                     f"PBRS-C (ASP) plateaus at SR={finals['pbrs_C']:.4f}\n"
                     "PBRS active -> reward is sound;\nself-play curriculum is the open problem",
                     transform=axes[0].transAxes, fontsize=9, ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", alpha=0.85))

    fig.suptitle("Scoping Contrast: the Reward Works (A/B) — the Self-Play Curriculum Does Not (C/D)\n"
                 "all four use the identical PBRS reward; only the curriculum differs",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "5_scoping_contrast.png")
    fig.savefig(out); plt.close(fig)
    print("  Saved " + out + "  finals(SR): " +
          ", ".join(f"{PBRS_CSV_PREFIX[k]}={v:.4f}" for k, v in finals.items()))
    return finals


if __name__ == "__main__":
    print("Generating thesis plots: Why PBRS (26.06.20 data)\n")

    print("[1/5] Position Error: Ad-hoc Dense vs PBRS (single-agent, no curriculum)")
    plot1_position_error()
    print("[2/5] Position + Rotation Error (dual panel)")
    plot2_pos_rot_error()
    print("[3/5] Reward Signal Quality")
    plot3_reward_signal()
    print("[4/5] Validation Results (A/B/C held-out)")
    plot4_validation()
    print("[5/5] Scoping Contrast (A/B/C/D, identical PBRS reward)")
    plot5_scoping_contrast()

    print(f"\nAll plots saved to: {OUT_DIR}")
