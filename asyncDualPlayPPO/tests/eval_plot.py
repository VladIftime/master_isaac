"""
Model comparison plot generator.

Reads evaluation CSVs and produces a 2x2 grouped bar chart comparing
models across all 10 tests.

Run:
  python -m asyncDualPlayPPO.tests.eval_plot
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
INPUT_CSVS = [
    "results/eval_push_results.csv",
    "results/eval_curobo_results.csv",
]
OUTPUT_PNG = "results/model_comparison.png"
# ═══════════════════════════════════════════════════════════════════════════════

import csv
import os
import sys
from collections import defaultdict

import numpy as np


def load_results(csv_paths):
    rows = []
    for path in csv_paths:
        if not os.path.isfile(path):
            print(f"[WARNING] CSV not found, skipping: {path}")
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["test_id"] = int(row["test_id"])
                row["episode_idx"] = int(row["episode_idx"])
                row["seed"] = int(row["seed"])
                row["success"] = int(row["success"])
                row["pos_error"] = float(row["pos_error"])
                row["rot_error"] = float(row["rot_error"])
                row["pushes_used"] = int(row["pushes_used"])
                rows.append(row)
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["test_id"], row["test_name"])
        grouped[key].append(row)

    results = {}
    for (model_name, test_id, test_name), episodes in grouped.items():
        sr = np.mean([e["success"] for e in episodes]) * 100
        avg_pos = np.mean([e["pos_error"] for e in episodes])
        avg_rot = np.mean([e["rot_error"] for e in episodes])
        avg_push = np.mean([e["pushes_used"] for e in episodes])
        if model_name not in results:
            results[model_name] = {}
        results[model_name][test_id] = {
            "test_name": test_name,
            "sr": sr,
            "pos_error": avg_pos,
            "rot_error": avg_rot,
            "pushes_used": avg_push,
            "n_episodes": len(episodes),
        }
    return results


def print_summary(results):
    models = sorted(results.keys())
    test_ids = sorted(set(tid for m in results.values() for tid in m.keys()))

    print(f"\n{'═'*80}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'═'*80}")
    header = f"{'Model':<28s}"
    header += f"{'SR%':>7s} {'PosErr':>8s} {'RotErr':>8s} {'Actions':>8s}"
    print(header)
    print(f"{'─'*28} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")

    for model_name in models:
        data = results[model_name]
        all_sr = [data[tid]["sr"] for tid in data]
        all_pe = [data[tid]["pos_error"] for tid in data]
        all_re = [data[tid]["rot_error"] for tid in data]
        all_ap = [data[tid]["pushes_used"] for tid in data]
        print(f"{model_name:<28s}"
              f"{np.mean(all_sr):6.1f}% "
              f"{np.mean(all_pe):8.4f} "
              f"{np.mean(all_re):8.4f} "
              f"{np.mean(all_ap):7.1f}")

    print(f"\n{'─'*80}")
    print(f"PER-TEST BREAKDOWN:")
    print(f"{'─'*80}")

    for tid in test_ids:
        tname = ""
        for m in results.values():
            if tid in m:
                tname = m[tid]["test_name"]
                break
        print(f"\n  Test {tid:2d}: {tname}")
        for model_name in models:
            if tid not in results[model_name]:
                continue
            d = results[model_name][tid]
            print(f"    {model_name:<26s} SR={d['sr']:5.1f}%  "
                  f"PosErr={d['pos_error']:.4f}  RotErr={d['rot_error']:.4f}  "
                  f"Actions={d['pushes_used']:.1f}  (n={d['n_episodes']})")


def plot_comparison(results, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available. Skipping plot generation.")
        return

    models = sorted(results.keys())
    test_ids = sorted(set(tid for m in results.values() for tid in m.keys()))
    n_models = len(models)
    n_tests = len(test_ids)

    if n_models == 0 or n_tests == 0:
        print("[WARNING] No data to plot.")
        return

    test_labels = []
    for tid in test_ids:
        for m in results.values():
            if tid in m:
                test_labels.append(m[tid]["test_name"])
                break
        else:
            test_labels.append(f"test_{tid}")

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Push-T Model Comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("Success Rate (%)", "sr"),
        ("Avg Position Error (m)", "pos_error"),
        ("Avg Rotation Error (rad)", "rot_error"),
        ("Avg Actions Used", "pushes_used"),
    ]

    bar_width = 0.8 / n_models
    x = np.arange(n_tests)

    for ax_idx, (ax, (title, key)) in enumerate(zip(axes.flat, metrics)):
        for m_idx, model_name in enumerate(models):
            values = []
            for tid in test_ids:
                if tid in results[model_name]:
                    values.append(results[model_name][tid][key])
                else:
                    values.append(0)
            offset = (m_idx - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width,
                         label=model_name, color=colors[m_idx], edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Test")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        if key == "sr":
            ax.set_ylim(0, 105)
            ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved to {output_path}")
    plt.close()


def main():
    rows = load_results(INPUT_CSVS)
    if not rows:
        print("[ERROR] No data loaded from CSVs.")
        sys.exit(1)

    print(f"[Plot] Loaded {len(rows)} rows from {len(INPUT_CSVS)} CSV(s).")
    results = aggregate(rows)
    print_summary(results)
    plot_comparison(results, OUTPUT_PNG)


if __name__ == "__main__":
    main()
