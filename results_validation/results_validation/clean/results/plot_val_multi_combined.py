#!/usr/bin/env python3
"""
Generate val_multi_combined.png — dual-panel validation summary:
  Left: overall SR bars with ±1 SE error bars
  Right: SR by test type (pos_only / pos_rot) with error bars

Models: PPO-Baseline, PPO-PBRS, PPO-Curriculum, TASP-dPose, ASP-dPose.
No TASP-dPose-BP.

Usage: python plot_val_multi_combined.py
Output: comparison/val_multi_combined.png
"""
import csv
import os
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BASE_DIR), "..", "..", "..",
                   "literature", "paper-async", "presentation", "figures",
                   "val_multi_combined.png")
OUT = os.path.abspath(OUT)

MODELS = [
    ("../../../orig_loss/0_orig_rew_30_isaac.csv","PPO-Baseline",       "#555555", "s"),
    ("A_simp/20_isaac_30t.csv",               "PPO-PBRS",           "#0072B2", "o"),
    ("B_curr/28_isaac_30t.csv",               "PPO-Curriculum",     "#E69F00", "s"),
    ("G_tasp_dpose/26_isaac.csv",             "TASP-dPose",         "#009E73", "^"),
    ("E_asp_dpose/26_isaac.csv",              "ASP-dPose",          "#D55E00", "D"),
]


def load_csv(rel_path):
    path = os.path.join(BASE_DIR, rel_path)
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def compute_aggregates(rows):
    """scene_solved = binary success column (1/0); aggregates by test_type."""
    per_type = defaultdict(list)
    scene_ok = 0
    n_scenes = len(rows)
    for r in rows:
        ok = int(r.get("success", "0")) == 1
        tt = r.get("test_type", "")
        if ok:
            scene_ok += 1
        if tt:
            per_type[tt].append(ok)
    p = scene_ok / n_scenes if n_scenes else 0.0
    se = math.sqrt(p * (1 - p) / n_scenes) if n_scenes > 0 else 0.0

    type_agg = {}
    for tt, vals in per_type.items():
        vals_a = np.array(vals, dtype=bool)
        n = len(vals_a)
        p_t = vals_a.mean()
        se_t = math.sqrt(p_t * (1 - p_t) / n) if n > 0 else 0.0
        type_agg[tt] = {"mean": float(p_t), "se": float(se_t), "n": n}

    return {"overall": {"mean": float(p), "se": float(se), "n": n_scenes, "solved": scene_ok},
            "by_type": type_agg}


def main():
    data = []
    for rel, label, color, marker in MODELS:
        path = os.path.join(BASE_DIR, rel)
        if not os.path.isfile(path):
            print(f"[WARN] Missing: {path}")
            continue
        rows = load_csv(rel)
        agg = compute_aggregates(rows)
        agg["label"] = label
        agg["color"] = color
        agg["marker"] = marker
        data.append(agg)
        print(f"[OK] {label:20s} {agg['overall']['solved']:2d}/{agg['overall']['n']} "
              f"SR={agg['overall']['mean']*100:.1f}% ±{agg['overall']['se']*100:.1f}% SE")

    if not data:
        print("[ERROR] No data")
        return

    n_models = len(data)
    colors = [d["color"] for d in data]
    labels = [d["label"] for d in data]

    # ── Create two-panel figure ─────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ── LEFT: overall SR bars ──
    x = np.arange(n_models)
    means = [d["overall"]["mean"] * 100 for d in data]
    ses = [d["overall"]["se"] * 100 for d in data]
    bars = ax_left.bar(x, means, 0.6, color=colors, edgecolor="black", linewidth=0.6,
                       yerr=ses, capsize=5, error_kw={"linewidth": 1.2})
    for i, (bar, mean, se) in enumerate(zip(bars, means, ses)):
        ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 1.0,
                     f"{mean:.1f}%" if mean < 15 else f"{mean:.1f}%",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_left.set_ylabel("Scene Success Rate (%)")
    ax_left.set_title("Overall SR (30 T-block scenes)", fontweight="bold", fontsize=11)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_left.set_ylim(0, 115)
    ax_left.grid(axis="y", alpha=0.25)

    # ── RIGHT: SR by test type (pos_only / pos_rot) ──
    test_types = ["pos_only", "pos_rot"]
    n_tt = len(test_types)
    x_tt = np.arange(n_tt)
    width = 0.60 / n_models

    for i, d in enumerate(data):
        tt = d["by_type"]
        t_means = [tt.get(t, {"mean": 0.0, "se": 0.0})["mean"] * 100 for t in test_types]
        t_ses = [tt.get(t, {"mean": 0.0, "se": 0.0})["se"] * 100 for t in test_types]
        offset = (i - (n_models - 1) / 2) * width
        bars_r = ax_right.bar(x_tt + offset, t_means, width, label=d["label"], color=d["color"],
                              edgecolor="black", linewidth=0.6, yerr=t_ses, capsize=5,
                              error_kw={"linewidth": 1.2})
        for bar, mean, se in zip(bars_r, t_means, t_ses):
            ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 1.0,
                          f"{mean:.0f}%" if mean >= 10 else f"{mean:.1f}%",
                          ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax_right.set_ylabel("Scene Success Rate (%)")
    ax_right.set_title("SR by Test Type", fontweight="bold", fontsize=11)
    ax_right.set_xticks(x_tt)
    ax_right.set_xticklabels(["Position Only\n(10 scenes)", "Position + Rotation\n(20 scenes)"],
                             fontsize=9)
    ax_right.legend(fontsize=9, loc="upper right")
    ax_right.set_ylim(0, 115)
    ax_right.grid(axis="y", alpha=0.25)

    fig.suptitle("Multi-Model Validation — 30 T-block Scenes, Best-of-20, 30-Push Budget\n"
                 "(Error bars = ±1 SE across scenes)",
                 fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] {OUT}")


if __name__ == "__main__":
    main()
