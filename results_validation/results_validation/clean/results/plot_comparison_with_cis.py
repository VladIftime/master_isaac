#!/usr/bin/env python3
"""
Multi-model validation comparison with confidence intervals.
Uses best-of-20 trial data (trial_count, success_count columns) to compute
binomial 95% CIs per scene, then aggregates across difficulty/test-type.

Produces 4 plots for the thesis presentation:
  val_multi_sr_overall.png      — overall SR bars with CIs
  val_multi_sr_difficulty.png   — SR by difficulty (easy/med/hard) with CIs
  val_multi_sr_testtype.png     — SR by test type (pos_only/pos_rot) with CIs
  val_multi_error_bars.png      — per-test error bars (A vs B close-up + E-H overlaid)

Usage:
  python plot_comparison_with_cis.py [--out-dir OUTPUT_DIR]
"""
import csv
import os
import sys
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from matplotlib.patches import Rectangle

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("A_simp",       "20_isaac_30t.csv", "PPO-PBRS",        "#0072B2",  "o"),
    ("B_curr",       "28_isaac_30t.csv", "PPO-Curriculum",  "#E69F00",  "s"),
    ("G_tasp_dpose", "26_isaac.csv",     "TASP-dPose",      "#009E73",  "^"),
    ("E_asp_dpose",  "26_isaac.csv",     "ASP-dPose",       "#D55E00",  "D"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi, centre)."""
    import scipy.stats as ss
    z = ss.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z*z / n
    mid = (p + z*z / (2*n)) / denom
    radius = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return max(0.0, mid - radius), min(1.0, mid + radius), p


def difficulty_from_name(test_name: str) -> str:
    name = test_name.strip().upper()
    if name.startswith("E_"):
        return "easy"
    if name.startswith("M_"):
        return "medium"
    if name.startswith("H_") or name.startswith("EDGE"):
        return "hard"
    return "unknown"


def compute_aggregated_sr(rows: list[dict]) -> dict:
    """Returns per-scene SRs + aggregate by difficulty + test_type.

    Uses two metrics:
    - scene_sr: binary `success` column (0/1) = was scene solved in best-of-20?
      This is the presentation metric (e.g., 24/30 = 80.0%).
    - trial_sr: from success_count/trial_count = per-trial hit rate.
      Used for per-test binomial error bars.
    For aggregate SE: binomial SE on scene_sr using sqrt(p*(1-p)/n).
    """
    per_scene = {}
    scene_successes = []
    for r in rows:
        tid = int(r["test_index"])
        n = int(r.get("trial_count", 20))
        s = int(r.get("success_count", 0))
        scene_solved = int(r.get("success", "0")) == 1
        trial_sr = s / n if n > 0 else 0.0
        lo, hi, _ = wilson_ci(s, n)
        scene_successes.append(scene_solved)
        per_scene[tid] = {
            "test_name": r["test_name"],
            "test_type": r.get("test_type", ""),
            "trial_count": n,
            "success_count": s,
            "scene_solved": scene_solved,
            "trial_sr": trial_sr,     # per-trial hit rate (for binomial CI bars)
            "ci_lo": lo,
            "ci_hi": hi,
            "difficulty": difficulty_from_name(r["test_name"]),
        }

    # Aggregate by difficulty — using scene_solved (binary)
    by_diff = defaultdict(list)
    for tid, d in per_scene.items():
        by_diff[d["difficulty"]].append(d["scene_solved"])
    by_diff_agg = {}
    for diff, vals in by_diff.items():
        vals = np.array(vals, dtype=bool)
        n = len(vals)
        p = vals.mean()
        se = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        by_diff_agg[diff] = {"mean_sr": float(p), "se": float(se), "n_scenes": n}

    # Aggregate by test type — using scene_solved
    by_tt = defaultdict(list)
    for tid, d in per_scene.items():
        tt = d["test_type"]
        if tt:
            by_tt[tt].append(d["scene_solved"])
    by_tt_agg = {}
    for tt, vals in by_tt.items():
        vals = np.array(vals, dtype=bool)
        n = len(vals)
        p = vals.mean()
        se = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        by_tt_agg[tt] = {"mean_sr": float(p), "se": float(se), "n_scenes": n}

    scene_successes = np.array(scene_successes, dtype=bool)
    n_total = len(scene_successes)
    p_total = scene_successes.mean()
    se_total = math.sqrt(p_total * (1 - p_total) / n_total) if n_total > 0 else 0.0
    overall = {
        "mean_sr": float(p_total),
        "se": float(se_total),
        "n_scenes": n_total,
        "n_solved": int(scene_successes.sum()),
    }

    return {
        "per_scene": per_scene,
        "by_difficulty": by_diff_agg,
        "by_test_type": by_tt_agg,
        "overall": overall,
    }


def load_all_models() -> list[dict]:
    data = []
    for model_dir, csv_name, label, color, marker in MODELS:
        path = os.path.join(BASE_DIR, model_dir, csv_name)
        if not os.path.isfile(path):
            print(f"[WARN] Missing: {path}")
            continue
        rows = load_csv(path)
        # Verify trial_count/success_count columns
        if "trial_count" not in rows[0] or "success_count" not in rows[0]:
            print(f"[WARN] {path}: missing trial_count/success_count columns, skipping")
            continue
        agg = compute_aggregated_sr(rows)
        agg["label"] = label
        agg["color"] = color
        agg["marker"] = marker
        agg["model_key"] = model_dir
        data.append(agg)
        print(f"[OK] {model_dir}: scene SR={agg['overall']['mean_sr']*100:.1f}% "
              f"({agg['overall']['n_solved']}/{agg['overall']['n_scenes']}) "
              f"±{agg['overall']['se']*100:.1f}% (SE)")
    return data


# ── Plot generators ──────────────────────────────────────────────────────────

def plot_overall_sr(data: list[dict], out_dir: str):
    """Grouped bar chart: overall SR per model with CIs."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(data)
    x = np.arange(n)
    colors = [d["color"] for d in data]
    labels = [d["label"] for d in data]
    means = [d["overall"]["mean_sr"] * 100 for d in data]
    ses   = [d["overall"]["se"] * 100 for d in data]

    bars = ax.bar(x, means, 0.6, color=colors, edgecolor="black", linewidth=0.6,
                  yerr=ses, capsize=5, error_kw={"linewidth": 1.2})

    for i, (bar, mean, se) in enumerate(zip(bars, means, ses)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 1.0,
                f"{mean:.1f}%" if mean < 20 else f"{mean:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Validation Success Rate (%)")
    ax.set_title("Overall Validation SR — 30 T-block Scenes, Best-of-20\n(Error bars = ±1 SE across scenes)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "val_multi_sr_overall.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_sr_by_difficulty(data: list[dict], out_dir: str):
    """Grouped bar: SR by difficulty (easy/med/hard) across models."""
    difficulties = ["easy", "medium", "hard"]
    n_models = len(data)
    n_diffs = len(difficulties)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(n_diffs)
    width = 0.75 / n_models

    for i, d in enumerate(data):
        diffs = d["by_difficulty"]
        means = [diffs.get(diff, {"mean_sr": 0.0, "se": 0.0})["mean_sr"] * 100 for diff in difficulties]
        ses   = [diffs.get(diff, {"mean_sr": 0.0, "se": 0.0})["se"] * 100 for diff in difficulties]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, label=d["label"], color=d["color"],
                      edgecolor="white", linewidth=0.4, yerr=ses, capsize=4)
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ses)*0.5 + 0.5,
                        f"{mean:.0f}" if mean >= 10 else f"{mean:.1f}",
                        ha="center", va="bottom", fontsize=6, fontweight="bold")

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Validation SR by Difficulty — 30 T-block Scenes, Best-of-20\n(Error bars = ±1 SE across scenes)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in difficulties])
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "val_multi_sr_difficulty.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_sr_by_testtype(data: list[dict], out_dir: str):
    """Grouped bar: SR by test type (pos_only / pos_rot) across models."""
    test_types = ["pos_only", "pos_rot"]
    n_models = len(data)
    n_tt = len(test_types)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(n_tt)
    width = 0.70 / n_models

    for i, d in enumerate(data):
        tt = d["by_test_type"]
        means = [tt.get(t, {"mean_sr": 0.0, "se": 0.0})["mean_sr"] * 100 for t in test_types]
        ses   = [tt.get(t, {"mean_sr": 0.0, "se": 0.0})["se"] * 100 for t in test_types]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, label=d["label"], color=d["color"],
                      edgecolor="white", linewidth=0.4, yerr=ses, capsize=4)
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ses) + 1.5,
                        f"{mean:.0f}" if mean >= 10 else f"{mean:.1f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                        rotation=90 if mean < 15 else 0)

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Validation SR by Test Type — 30 T-block Scenes, Best-of-20\n(Error bars = ±1 SE across scenes)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Position Only\n(10 tests)", "Position + Rotation\n(20 tests)"])
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "val_multi_sr_testtype.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_per_test_errorbars(data: list[dict], out_dir: str):
    """Per-test SR with binomial CIs. Two panels: (A vs B close-up) + (all ASP overlaid)."""
    # Collect per-scene data
    test_ids = sorted(data[0]["per_scene"].keys())
    n_tests = len(test_ids)
    test_names = [data[0]["per_scene"][tid]["test_name"].split(" #")[0] for tid in test_ids]
    test_types = [data[0]["per_scene"][tid].get("test_type", "") for tid in test_ids]

    # Use nicer model keys
    short_keys = {d["model_key"]: d["label"].split(":")[0].strip() for d in data}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n_tests * 0.55), 10), sharex=True)

    # ── Panel 1: A vs B only ──
    a_data = next(d for d in data if d["model_key"] == "A_simp")
    b_data = next(d for d in data if d["model_key"] == "B_curr")
    series = [("A_simp", a_data, "#0072B2"), ("B_curr", b_data, "#E69F00")]
    w = 0.35
    x_t = np.arange(n_tests)

    for j, (key, d, color) in enumerate(series):
        srs = []
        ci_lows = []
        ci_his = []
        for tid in test_ids:
            sc = d["per_scene"].get(tid, {"trial_sr": 0, "ci_lo": 0, "ci_hi": 0})
            srs.append(sc["trial_sr"])
            ci_lows.append(sc["trial_sr"] - sc["ci_lo"])
            ci_his.append(sc["ci_hi"] - sc["trial_sr"])
        offset = (j - 0.5) * w
        ci_errs = [ci_lows, ci_his]
        bars = ax1.bar(x_t + offset, [s * 100 for s in srs], w,
                       color=color, edgecolor="white", linewidth=0.4,
                       yerr=[np.array(ci_lows) * 100, np.array(ci_his) * 100],
                       capsize=2, error_kw={"linewidth": 0.8, "alpha": 0.6},
                       label=short_keys[key])

    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Per-Scene SR with 95% Binomial CIs — PPO-PBRS vs PPO-Curriculum\n(30 T-block scenes, 20 trials each)",
                  fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.25)

    type_colors = {"pos_only": "#4CAF50", "pos_rot": "#9C27B0"}
    # Panel 1: tick labels are empty (sharex with panel 2) — no color needed here

    # ── Panel 2: all models ──
    w2 = 0.75 / len(data)
    for i, d in enumerate(data):
        srs = []
        ci_lows = []
        ci_his = []
        for tid in test_ids:
            sc = d["per_scene"].get(tid, {"trial_sr": 0, "ci_lo": 0, "ci_hi": 0})
            srs.append(sc["trial_sr"])
            ci_lows.append(sc["trial_sr"] - sc["ci_lo"])
            ci_his.append(sc["ci_hi"] - sc["trial_sr"])
        offset_i = (i - (len(data) - 1) / 2) * w2
        ax2.bar(x_t + offset_i, [s * 100 for s in srs], w2,
                color=d["color"], edgecolor="white", linewidth=0.3,
                yerr=[np.array(ci_lows) * 100, np.array(ci_his) * 100],
                capsize=1.5, error_kw={"linewidth": 0.5, "alpha": 0.4},
                label=short_keys[d["model_key"]])

    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Per-Scene SR — All 4 Models (95% Binomial CIs)",
                  fontweight="bold")
    ax2.set_xticks(x_t)
    ax2.set_xticklabels(test_names, rotation=45, ha="right", fontsize=7)
    # Color labels by test type
    for i, (label, tt) in enumerate(zip(ax2.get_xticklabels(), test_types)):
        label.set_color(type_colors.get(tt, "black"))
    ax2.legend(fontsize=7, loc="upper right", ncol=3)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.25)

    # Add legend for test type colors
    from matplotlib.patches import Patch
    legend_tt = [Patch(facecolor=type_colors[t], edgecolor="black", linewidth=0.5,
                        label=f"{t.replace('pos_only','Pos-Only').replace('pos_rot','Pos+Rot')}")
                 for t in type_colors]
    fig.legend(handles=legend_tt, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=0.5, rect=[0, 0.04, 1, 1])
    path = os.path.join(out_dir, "val_multi_error_bars.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


def write_ci_summary_table(data: list[dict], out_dir: str):
    """Write a text summary table with CIs."""
    path = os.path.join(out_dir, "ci_summary.txt")
    with open(path, "w") as f:
        f.write("Model                                                                   Scene SR    SE     95% CI (±2SE)     N scenes\n")
        f.write("-" * 110 + "\n")
        for d in data:
            ov = d["overall"]
            f.write(f"{d['label']:<60s} {ov['n_solved']:2d}/{ov['n_scenes']} "
                    f"{ov['mean_sr']*100:5.1f}%    {ov['se']*100:5.1f}%   [{max(0,ov['mean_sr']*100 - 2*ov['se']*100):5.1f}% – {min(100,ov['mean_sr']*100 + 2*ov['se']*100):5.1f}%]\n")
        f.write("\nA_simp vs B_curr gap (scene-level): ")
        a_ov = next(d for d in data if d["model_key"] == "A_simp")["overall"]
        b_ov = next(d for d in data if d["model_key"] == "B_curr")["overall"]
        diff = (a_ov["mean_sr"] - b_ov["mean_sr"]) * 100
        se_diff = math.sqrt(a_ov["se"]**2 + b_ov["se"]**2) * 100
        f.write(f"{diff:+.1f}pp ± {se_diff:.1f}pp (SE of difference, independent scenes)\n")
        if abs(diff) > 2 * se_diff:
            f.write(f"→ Significant at 95% level\n")
        else:
            f.write(f"→ NOT statistically significant at 95% level — the 3.3pp gap is within noise for 30 scenes\n")
    print(f"[Text] {path}")
    # Print to stdout too
    with open(path) as f:
        print(f.read())


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "comparison"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model data...")
    data = load_all_models()
    if not data:
        print("[ERROR] No models loaded!")
        sys.exit(1)

    print("\nGenerating plots...")
    plot_overall_sr(data, args.out_dir)
    plot_sr_by_difficulty(data, args.out_dir)
    plot_sr_by_testtype(data, args.out_dir)
    plot_per_test_errorbars(data, args.out_dir)
    write_ci_summary_table(data, args.out_dir)

    print(f"\n[Done] All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
