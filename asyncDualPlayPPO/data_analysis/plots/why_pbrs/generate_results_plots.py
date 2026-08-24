#!/usr/bin/env python3
"""Generate the two Push-T results charts for the discussion slides.

Reads the validation CSVs in results/pbrs_asp/ (produced by validate_push.py /
validate_push_asp.py), computes success rate per category (no area-coverage),
and renders:

  results_dp_vs_pbrs.png        Diffusion Policy vs PBRS single-agent
                                across [Overall, Disc, Pos-only, Pos+Rot]
  results_selfplay_collapse.png Overall SR for all 5 runs (self-play collapse)

Each test CSV has 30 tests: 10 disc_pos + 10 pos_only + 10 pos_rot.
Success column is binary; SR = mean(success) per test_type.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "..", ".."))
RESULTS = next(
    (os.path.join(_ROOT, d, "pbrs_asp")
     for d in ("results", "results_asp_tasp_dpose")
     if os.path.isdir(os.path.join(_ROOT, d, "pbrs_asp"))),
    os.path.join(_ROOT, "results", "pbrs_asp"),
)

RUNS = {
    "Diffusion Policy":   "dp_results.csv",
    "PBRS single-agent":  "results_valid_simp_dpose.csv",
    "ASP (disc)":         "results_valid_asp_disc.csv",
    "time-ASP (disc)":    "results_valid_tasp_disc.csv",
    "time-ASP (T-block)": "results_valid_tasp_dpose.csv",
}

CAT_ORDER = ["Overall", "Disc", "Pos-only", "Pos+Rot"]
CAT_KEY = {"Disc": "disc_pos", "Pos-only": "pos_only", "Pos+Rot": "pos_rot"}

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
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_sr(csv_name):
    df = pd.read_csv(os.path.join(RESULTS, csv_name))
    out = {"Overall": float(df["success"].mean()) * 100.0, "n": len(df)}
    for cat, key in CAT_KEY.items():
        sub = df[df["test_type"] == key]
        out[cat] = float(sub["success"].mean()) * 100.0 if len(sub) else float("nan")
    return out


def plot_dp_vs_pbrs(sr):
    series = ["Diffusion Policy", "PBRS single-agent"]
    colors = {"Diffusion Policy": "#d62728", "PBRS single-agent": "#2ca02c"}
    x = np.arange(len(CAT_ORDER))
    bw = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(series):
        vals = [sr[name][c] for c in CAT_ORDER]
        off = (i - 0.5) * bw
        ax.bar(x + off, vals, bw, label=name, color=colors[name],
               edgecolor="white", linewidth=0.6)
        for j, v in enumerate(vals):
            ax.text(x[j] + off, v + 1.5, f"{v:.0f}%", ha="center",
                    fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_title("Push-T held-out test (30 scenes): RL reward beats imitation\n"
                 "PBRS single-agent (no demos, no curriculum) vs Diffusion Policy")
    ax.legend(loc="upper right")
    out = os.path.join(OUT_DIR, "results_dp_vs_pbrs.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_selfplay_collapse(sr):
    names = ["Diffusion Policy", "PBRS single-agent",
             "ASP (disc)", "time-ASP (disc)", "time-ASP (T-block)"]
    colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    vals = [sr[n]["Overall"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), vals, color=colors,
                  edgecolor="white", linewidth=0.6, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Overall Success Rate")
    ax.set_ylim(0, 95)
    ax.set_yticks(range(0, 81, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 81, 20)])
    ax.axvspan(1.5, 4.5, alpha=0.10, color="red", zorder=0)
    ax.text(3.0, 70, "same PBRS reward\n+ self-play curriculum\n→ collapse",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", alpha=0.85))
    ax.set_title("The reward is sound; the self-play curriculum is the open problem\n"
                 "(all curricula reuse the identical PBRS reward)")
    out = os.path.join(OUT_DIR, "results_selfplay_collapse.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating Push-T results charts (success rate, no coverage)\n")
    sr = {name: load_sr(csv) for name, csv in RUNS.items()}

    print("Verified success rates (%):")
    header = f"  {'run':22s} {'n':>3s} {'Overall':>8s} {'Disc':>7s} {'Pos-only':>9s} {'Pos+Rot':>8s}"
    print(header)
    for name in RUNS:
        s = sr[name]
        print(f"  {name:22s} {s['n']:3d} {s['Overall']:7.1f}% {s['Disc']:6.1f}% "
              f"{s['Pos-only']:8.1f}% {s['Pos+Rot']:7.1f}%")
    print()

    plot_dp_vs_pbrs(sr)
    plot_selfplay_collapse(sr)
    print(f"\nDone. Charts in {OUT_DIR}")
