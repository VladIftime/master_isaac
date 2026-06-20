#!/usr/bin/env python3
"""Compare multiple validation CSV files with side-by-side plots.

Reads CSV files produced by validate_throw.py and generates two plots:
  1. comparison_distances.png  — grouped bar chart (best + mean per test)
  2. comparison_birdseye.png   — top-down mean-landing scatter

Usage:
    python scripts/compare_validations.py \
        logs/run_A.csv logs/run_B.csv \
        --labels "Checkpoint 200k" "Checkpoint 500k" \
        --output_dir logs/comparisons/
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

CB_PALETTE = [
    (0.000, 0.447, 0.698),
    (0.902, 0.624, 0.000),
    (0.000, 0.620, 0.451),
    (0.800, 0.475, 0.655),
    (0.337, 0.706, 0.914),
    (0.835, 0.369, 0.000),
    (0.941, 0.894, 0.259),
]

HATCH_PATTERNS = ["/", "\\", "x", "+", ".", "o", "|"]

TABLE_X_RANGE = (-1.0, 1.0)
TABLE_Y_RANGE = (0.4, 1.85)


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    tests = defaultdict(list)
    for row in rows:
        tests[int(row["test_no"])].append(row)
    return tests


MIN_LANDING_Z = 0.1


def compute_stats(test_rows):
    valid = [
        r for r in test_rows
        if float(r.get("landing_z", "1.0")) >= MIN_LANDING_Z
    ]
    if not valid:
        valid = test_rows

    dists = [float(r["distance"]) for r in valid]
    landings = [(float(r["landing_x"]), float(r["landing_y"])) for r in valid]
    aabb_hits = [int(r["aabb_hit"]) for r in valid]

    in_ws = [
        (lx, ly) for lx, ly in landings
        if TABLE_X_RANGE[0] <= lx <= TABLE_X_RANGE[1]
        and TABLE_Y_RANGE[0] <= ly <= TABLE_Y_RANGE[1]
    ]

    return {
        "best": min(dists) if dists else float("inf"),
        "mean": np.mean(dists) if dists else float("inf"),
        "std": np.std(dists) if dists else 0.0,
        "aabb_rate": sum(aabb_hits) / len(aabb_hits) if aabb_hits else 0.0,
        "n_attempts": len(dists),
        "mean_landing": (
            (np.mean([p[0] for p in in_ws]), np.mean([p[1] for p in in_ws]))
            if in_ws else None
        ),
        "target_x": float(test_rows[0]["target_x"]),
        "target_y": float(test_rows[0]["target_y"]),
        "test_name": test_rows[0]["test_name"],
    }


def plot_distances(all_stats, common_tests, labels, save_path):
    n_runs = len(labels)
    n_tests = len(common_tests)

    fig, (ax_best, ax_mean) = plt.subplots(2, 1, figsize=(max(10, n_tests * 1.2), 5.5), sharex=True)
    fig.suptitle("Validation Comparison — Distance to Target", fontsize=14, y=0.98)

    x = np.arange(n_tests)
    total_w = 0.75
    bar_w = total_w / n_runs

    for run_idx, label in enumerate(labels):
        color = CB_PALETTE[run_idx % len(CB_PALETTE)]
        hatch = HATCH_PATTERNS[run_idx % len(HATCH_PATTERNS)]
        offset = -total_w / 2 + bar_w * (run_idx + 0.5)

        bests = [all_stats[run_idx][t]["best"] * 100 for t in common_tests]
        means = [all_stats[run_idx][t]["mean"] * 100 for t in common_tests]
        stds = [all_stats[run_idx][t]["std"] * 100 for t in common_tests]

        ax_best.bar(
            x + offset, bests, bar_w,
            color=color, alpha=0.75, edgecolor="black", linewidth=0.6,
            hatch=hatch, label=label,
        )
        ax_mean.bar(
            x + offset, means, bar_w,
            color=color, alpha=0.75, edgecolor="black", linewidth=0.6,
            hatch=hatch, label=label,
            yerr=stds, capsize=2, error_kw={"linewidth": 0.8},
        )

    ax_best.set_ylabel("Best Distance (cm)")
    ax_best.set_title("Best Distance per Test No.", fontsize=11)
    ax_best.grid(True, alpha=0.3, axis="y")
    ax_best.legend(fontsize=9, loc="upper right")

    ax_mean.set_ylabel("Mean Distance (cm)")
    ax_mean.set_title("Mean Distance per Test No.", fontsize=11)
    ax_mean.set_xlabel("Test No.")
    ax_mean.grid(True, alpha=0.3, axis="y")

    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels(common_tests)

    best_vals = []
    mean_vals = []
    for run_idx in range(n_runs):
        for t in common_tests:
            best_vals.append(all_stats[run_idx][t]["best"] * 100)
            mean_vals.append((all_stats[run_idx][t]["mean"] + all_stats[run_idx][t]["std"]) * 100)
    ax_best.set_ylim(0, max(best_vals) * 1.15 if best_vals else 1.0)
    ax_mean.set_ylim(0, max(mean_vals) * 1.15 if mean_vals else 1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Distance plot saved to: {save_path}")
    plt.close(fig)


def plot_birdseye(all_stats, common_tests, labels, save_path):
    n_runs = len(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Validation Comparison — Bird's Eye View", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    table_rect = plt.Rectangle(
        (TABLE_X_RANGE[0], TABLE_Y_RANGE[0]),
        TABLE_X_RANGE[1] - TABLE_X_RANGE[0],
        TABLE_Y_RANGE[1] - TABLE_Y_RANGE[0],
        linewidth=1, edgecolor="grey", facecolor="lightgrey", alpha=0.3,
    )
    ax.add_patch(table_rect)

    ax.plot(0, 0, "ks", markersize=10)

    targets_plotted = set()
    for t in common_tests:
        s = all_stats[0][t]
        tx, ty = s["target_x"], s["target_y"]
        if t not in targets_plotted:
            ax.plot(tx, ty, "s", color="steelblue", markersize=10, alpha=0.8)
            ax.annotate(str(t), (tx, ty), textcoords="offset points", xytext=(5, 5), fontsize=8)
            targets_plotted.add(t)

    markers = ["o", "D", "^", "v", "P", "X", "h"]
    for run_idx, label in enumerate(labels):
        color = CB_PALETTE[run_idx % len(CB_PALETTE)]
        marker = markers[run_idx % len(markers)]

        for t in common_tests:
            s = all_stats[run_idx][t]
            tx, ty = s["target_x"], s["target_y"]
            ml = s["mean_landing"]
            if ml is None:
                continue
            mx, my = ml
            ax.plot(
                mx, my, marker, color=color, markersize=8, alpha=0.85,
                markeredgecolor="black", markeredgewidth=0.5,
                label=label if t == common_tests[0] else None,
            )
            ax.plot(
                [tx, mx], [ty, my],
                "--", color=color, alpha=0.45, linewidth=1.0,
            )

    ax.set_xlim(-0.6, 0.8)
    ax.set_ylim(-0.2, 1.8)

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="steelblue", linestyle="None", markersize=8, alpha=0.8, label="Target"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=8, label="Robot"),
    ]
    for run_idx, label in enumerate(labels):
        color = CB_PALETTE[run_idx % len(CB_PALETTE)]
        marker = markers[run_idx % len(markers)]
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color=color, linestyle="None",
                       markersize=8, markeredgecolor="black", markeredgewidth=0.5, label=label)
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Bird's eye plot saved to: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare validation CSV files")
    parser.add_argument("csvs", nargs="+", help="CSV files from validate_throw.py")
    parser.add_argument("--labels", nargs="+", default=None, help="Display name per run (default: filenames)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: logs/comparisons/)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.splitext(os.path.basename(f))[0] for f in args.csvs]
    if len(args.labels) != len(args.csvs):
        print(f"[ERROR] Number of labels ({len(args.labels)}) must match number of CSVs ({len(args.csvs)})")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "logs", "comparisons")

    if not args.show:
        matplotlib.use("Agg")

    runs_data = []
    for csv_path in args.csvs:
        if not os.path.isfile(csv_path):
            print(f"[ERROR] File not found: {csv_path}")
            sys.exit(1)
        runs_data.append(load_csv(csv_path))

    common_tests = set(runs_data[0].keys())
    for rd in runs_data[1:]:
        common_tests &= set(rd.keys())
    common_tests = sorted(common_tests)

    if not common_tests:
        print("[ERROR] No common test_no found across all CSVs.")
        sys.exit(1)

    print(f"[INFO] Comparing {len(args.csvs)} runs across {len(common_tests)} common tests: {common_tests}")

    all_stats = []
    for run_idx, rd in enumerate(runs_data):
        stats = {}
        for t in common_tests:
            stats[t] = compute_stats(rd[t])
        all_stats.append(stats)

    for run_idx, label in enumerate(args.labels):
        bests = [all_stats[run_idx][t]["best"] for t in common_tests]
        means = [all_stats[run_idx][t]["mean"] for t in common_tests]
        rates = [all_stats[run_idx][t]["aabb_rate"] for t in common_tests]
        print(
            f"  {label:30s} | avg_best={np.mean(bests)*100:.1f}cm | avg_mean={np.mean(means)*100:.1f}cm "
            f"| aabb_rate={np.mean(rates)*100:.1f}%"
        )

    dist_path = os.path.join(args.output_dir, "comparison_distances.png")
    bird_path = os.path.join(args.output_dir, "comparison_birdseye.png")

    plot_distances(all_stats, common_tests, args.labels, dist_path)
    plot_birdseye(all_stats, common_tests, args.labels, bird_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
