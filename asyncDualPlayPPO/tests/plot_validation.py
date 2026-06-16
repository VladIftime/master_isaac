"""
Plot validation results from CSV files produced by validate_push.py
and validate_push_asp.py.

Produces:
  - Bar chart: success rate per model (overall)
  - Bar chart: success rate per model per difficulty (easy/medium/hard)
  - Bar chart: average pushes per model
  - Bar chart: average pushes per difficulty per model
  - Table: per-test comparison

Usage:
  python asyncDualPlayPPO/tests/plot_validation.py \
      --csvs results_simp.csv results_curr.csv results_asp.csv \
      --labels "A: PBRS only" "B: PBRS+Curriculum" "C: PBRS+ASP" \
      -o validation_plots
"""

import argparse
import os
import sys
from typing import List, Optional, Dict, Tuple


def load_csv(path: str) -> list:
    import csv
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows: list) -> Dict:
    total = len(rows)
    n_success = sum(1 for r in rows if r["success"] == "1")
    pushes = [int(r["pushes_used"]) for r in rows]
    pos_errs = [float(r["pos_err"]) for r in rows]
    rot_errs = [float(r["rot_err"]) for r in rows]
    covs = [float(r.get("area_coverage", 0.0)) for r in rows]
    return {
        "total": total,
        "successes": n_success,
        "sr": n_success / total * 100 if total > 0 else 0.0,
        "avg_pushes": sum(pushes) / len(pushes) if pushes else 0.0,
        "avg_pos_err": sum(pos_errs) / len(pos_errs) if pos_errs else 0.0,
        "avg_rot_err": sum(rot_errs) / len(rot_errs) if rot_errs else 0.0,
        "avg_coverage": sum(covs) / len(covs) if covs else 0.0,
        "pushes": pushes,
    }


def difficulty_from_name(test_name: str) -> str:
    name = test_name.upper()
    if name.startswith("E_"):
        return "easy"
    if name.startswith("M_"):
        return "medium"
    if name.startswith("H_") or name.startswith("EDGE"):
        return "hard"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Plot validation results")
    parser.add_argument("--csvs", type=str, nargs="+", required=True,
                        help="CSV files to compare")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for each CSV (same order)")
    parser.add_argument("-o", "--out-dir", type=str, default="validation_plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    labels = args.labels if args.labels else [f"Model {i+1}" for i in range(len(args.csvs))]

    all_data = []
    for path in args.csvs:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path}")
            continue
        rows = load_csv(path)
        metrics = compute_metrics(rows)
        all_data.append({"path": path, "rows": rows, "metrics": metrics,
                         "label": labels[len(all_data)]})
        print(f"[OK] Loaded {path}: SR={metrics['sr']:.1f}%, "
              f"avg_pushes={metrics['avg_pushes']:.1f}, "
              f"n={metrics['total']}")

    if not all_data:
        print("[ERROR] No valid CSV files loaded.")
        sys.exit(1)

    n_models = len(all_data)

    # ── Overall success rate ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    model_labels = [d["label"] for d in all_data]
    srs = [d["metrics"]["sr"] for d in all_data]
    bars = ax.bar(model_labels, srs, color=["#2196F3", "#4CAF50", "#FF9800"][:n_models])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Overall Validation Success Rate")
    for bar, val in zip(bars, srs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(srs) * 1.25 if max(srs) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "overall_sr.png"))
    plt.close(fig)

    # ── Success rate by difficulty ────────────────────────────────────────────
    difficulties = ["easy", "medium", "hard"]
    diff_colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}

    for diff in difficulties:
        fig, ax = plt.subplots(figsize=(6, 4))
        diff_srs = []
        for d in all_data:
            diff_rows = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
            diff_metrics = compute_metrics(diff_rows)
            diff_srs.append(diff_metrics["sr"])
        if all(s == 0 for s in diff_srs):
            plt.close(fig)
            continue
        bars = ax.bar(model_labels, diff_srs,
                       color=["#2196F3", "#4CAF50", "#FF9800"][:n_models])
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Success Rate — {diff.title()} Difficulty")
        for bar, val in zip(bars, diff_srs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_ylim(0, max(diff_srs) * 1.3 if max(diff_srs) > 0 else 100)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"sr_{diff}.png"))
        plt.close(fig)

    # ── Average pushes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_pushes = [d["metrics"]["avg_pushes"] for d in all_data]
    bars = ax.bar(model_labels, avg_pushes, color=["#2196F3", "#4CAF50", "#FF9800"][:n_models])
    ax.set_ylabel("Average Pushes")
    ax.set_title("Average Pushes per Test")
    for bar, val in zip(bars, avg_pushes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "avg_pushes.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    avg_covs = [d["metrics"]["avg_coverage"] for d in all_data]
    bars = ax.bar(model_labels, avg_covs, color=["#2196F3", "#4CAF50", "#FF9800"][:n_models])
    ax.set_ylabel("Area Coverage (%)")
    ax.set_title("Average Target Area Coverage")
    for bar, val in zip(bars, avg_covs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(avg_covs) * 1.25 if max(avg_covs) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "avg_coverage.png"))
    plt.close(fig)

    # SR by difficulty grouped
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(difficulties))
    width = 0.8 / n_models
    for i, d in enumerate(all_data):
        diff_srs_list = []
        for diff in difficulties:
            dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
            dm = compute_metrics(dr)
            diff_srs_list.append(dm["sr"])
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], diff_srs_list, width,
                       label=d["label"],
                       color=["#2196F3", "#4CAF50", "#FF9800"][i])
        for bar, val in zip(bars, diff_srs_list):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Difficulty")
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.title() for d in difficulties])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_by_difficulty_grouped.png"))
    plt.close(fig)

    # ── Success rate by test type (pos_only vs pos_rot) ───────────────────────
    test_types = ["pos_only", "pos_rot"]
    fig, ax = plt.subplots(figsize=(6, 4))
    type_srs = []
    for d in all_data:
        t_rows = [r for r in d["rows"] if r.get("test_type", "") == "pos_only"]
        t_m = compute_metrics(t_rows)
        type_srs.append(t_m["sr"])
    bars = ax.bar(model_labels, type_srs, color=["#2196F3", "#4CAF50", "#FF9800"][:n_models])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate — Position Only (10 tests)")
    for bar, val in zip(bars, type_srs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(type_srs) * 1.3 if max(type_srs) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_pos_only.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    type_srs_rot = []
    for d in all_data:
        t_rows = [r for r in d["rows"] if r.get("test_type", "") == "pos_rot"]
        t_m = compute_metrics(t_rows)
        type_srs_rot.append(t_m["sr"])
    bars = ax.bar(model_labels, type_srs_rot, color=["#2196F3", "#4CAF50", "#FF9800"][:n_models])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate — Position + Rotation (10 tests)")
    for bar, val in zip(bars, type_srs_rot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(type_srs_rot) * 1.3 if max(type_srs_rot) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_pos_rot.png"))
    plt.close(fig)

    # Grouped: SR by test type across models
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(test_types))
    width = 0.8 / n_models
    for i, d in enumerate(all_data):
        type_srs_list = []
        for tt in test_types:
            tr = [r for r in d["rows"] if r.get("test_type", "") == tt]
            tm = compute_metrics(tr)
            type_srs_list.append(tm["sr"])
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], type_srs_list, width,
                       label=d["label"],
                       color=["#2196F3", "#4CAF50", "#FF9800"][i])
        for bar, val in zip(bars, type_srs_list):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Test Type")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Position Only", "Position + Rotation"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_by_test_type_grouped.png"))
    plt.close(fig)

    # ── Per-test comparison table as text ─────────────────────────────────────
    table_path = os.path.join(args.out_dir, "per_test_comparison.txt")
    with open(table_path, "w") as f:
        header = f"{'Test':>4s} {'Name':15s} {'Type':8s} " + "  ".join(f"{l:>18s}" for l in model_labels)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for test_idx in range(1, get_test_count() + 1):
            from asyncDualPlayPPO.tasks.utils.validation_configs import get_test_config
            cfg = get_test_config(test_idx)
            if cfg is None:
                continue
            line = f"{test_idx:4d} {cfg.name:15s} {cfg.test_type:8s} "
            for d in all_data:
                r = [r for r in d["rows"] if int(r["test_index"]) == test_idx]
                if r:
                    r = r[0]
                    s = "PASS" if r["success"] == "1" else "FAIL"
                    _cov = float(r.get("area_coverage", 0.0))
                    line += f"  {s:>6s} P={r['pushes_used']:>2s} E={float(r['pos_err']):.3f}m C={_cov:.0f}%"
                else:
                    line += f"  {'---':>20s}"
            f.write(line + "\n")
    print(f"[OK] Per-test comparison saved to {table_path}")

    # ── Summary markdown table ────────────────────────────────────────────────
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Validation Summary\n\n")
        f.write("| Model | SR | Avg Pushes | Avg PosErr | Avg RotErr | Avg Cov | Tests |\n")
        f.write("|-------|----|-----------|------------|----------|--------|-------|\n")
        for d in all_data:
            m = d["metrics"]
            f.write(f"| {d['label']} | {m['sr']:.1f}% | {m['avg_pushes']:.1f} | "
                    f"{m['avg_pos_err']:.3f} m | {m['avg_rot_err']:.3f} rad | {m['avg_coverage']:.1f}% | {m['total']} |\n")
        f.write("\n## By Difficulty\n\n")
        f.write("| Model | Easy SR | Medium SR | Hard SR | Easy Pushes | Medium Pushes | Hard Pushes |\n")
        f.write("|-------|---------|-----------|---------|-------------|---------------|------------|\n")
        for d in all_data:
            parts = [d["label"]]
            for diff in ["easy", "medium", "hard"]:
                dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
                dm = compute_metrics(dr)
                parts.append(f"{dm['sr']:.1f}%")
            for diff in ["easy", "medium", "hard"]:
                dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
                dm = compute_metrics(dr)
                parts.append(f"{dm['avg_pushes']:.1f}")
            f.write("| " + " | ".join(parts) + " |\n")
        f.write("\n## By Test Type\n\n")
        f.write("| Model | Pos-Only SR | Pos+Rot SR | Pos-Only Pushes | Pos+Rot Pushes |\n")
        f.write("|-------|-------------|-----------|-----------------|---------------|\n")
        for d in all_data:
            parts = [d["label"]]
            for tt in ["pos_only", "pos_rot"]:
                tr = [r for r in d["rows"] if r.get("test_type", "") == tt]
                tm = compute_metrics(tr)
                parts.append(f"{tm['sr']:.1f}%")
            for tt in ["pos_only", "pos_rot"]:
                tr = [r for r in d["rows"] if r.get("test_type", "") == tt]
                tm = compute_metrics(tr)
                parts.append(f"{tm['avg_pushes']:.1f}")
            f.write("| " + " | ".join(parts) + " |\n")
    print(f"[OK] Summary markdown saved to {md_path}")

    print(f"\n[Done] All plots saved to {args.out_dir}/")


def get_test_count() -> int:
    from asyncDualPlayPPO.tasks.utils.validation_configs import get_test_count as _gtc
    return _gtc()


if __name__ == "__main__":
    main()
