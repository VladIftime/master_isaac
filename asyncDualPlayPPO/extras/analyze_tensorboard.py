#!/usr/bin/env python3
"""
Analyze TensorBoard summary logs from train_push.py / train_push_asp.py.

Reads TensorBoard event files from one or more summary directories,
auto-detects training type, and generates comparison plots.

Usage:
  python asyncDualPlayPPO/extras/analyze_tensorboard.py \
    --summary-dirs runs/exp1/summary runs/exp2/summary \
    --labels "PPO Baseline" "ASP" \
    -o analysis/comparison

  python asyncDualPlayPPO/extras/analyze_tensorboard.py \
    --summary-dirs runs/exp1/summary \
    -o analysis/single_run
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("[ERROR] tensorboard is required. Install with: pip install tensorboard")
    sys.exit(1)


PUSH_TAGS = {
    "Loss/Agent/Surrogate": ("Surrogate Loss", "Policy Loss"),
    "Loss/Agent/Value": ("Value Loss", "Value Loss"),
    "Reward/Mean": ("Mean Reward", "Mean Reward"),
    "Reward/EMA": ("EMA Reward", "EMA Reward"),
    "Metrics/SuccessRate": ("Success Rate", "Success Rate"),
    "Metrics/RotationSR": ("Rotation SR", "Rotation Success Rate"),
    "Metrics/PosError": ("Position Error (m)", "Position Error"),
    "Metrics/RotError": ("Rotation Error (rad)", "Rotation Error"),
    "Metrics/IKFailRate": ("IK Fail Rate", "IK Fail Rate"),
    "Metrics/EpisodicSR": ("Episodic SR", "Episodic Success Rate"),
    "Metrics/AvgPushesPerEpisode": ("Avg Pushes/Ep", "Avg Pushes per Episode"),
    "Metrics/Episodes": ("Episodes", "Episodes per Iteration"),
}

ASP_TAGS = {
    "Loss/Alice/Surrogate": ("Surrogate Loss", "Alice Policy Loss"),
    "Loss/Alice/Value": ("Value Loss", "Alice Value Loss"),
    "Reward/Alice": ("Reward", "Alice Reward"),
    "Loss/Bob/Surrogate": ("Surrogate Loss", "Bob Policy Loss"),
    "Loss/Bob/Value": ("Value Loss", "Bob Value Loss"),
    "Reward/Bob": ("Reward", "Bob Reward"),
    "Metrics/Bob/SuccessRate": ("Success Rate", "Bob Success Rate"),
    "Metrics/Bob/PosError": ("Position Error (m)", "Bob Position Error"),
    "Metrics/Bob/RotError": ("Rotation Error (rad)", "Bob Rotation Error"),
    "Metrics/Bob/PositionSR": ("Position SR", "Bob Position SR"),
    "Metrics/Bob/RotationSR": ("Rotation SR", "Bob Rotation SR"),
    "Alice/EntropyCoef": ("Entropy Coef", "Alice Entropy Coefficient"),
    "Alice/LearningRate": ("Learning Rate", "Alice Learning Rate"),
    "Metrics/Alice/ValidGoals": ("Count", "Alice Valid Goals"),
    "Metrics/Alice/InvalidGoals": ("Count", "Alice Invalid Goals"),
    "Metrics/Alice/GoalValidityRate": ("Rate", "Alice Goal Validity Rate"),
    "Metrics/Alice/MeanDisp3D": ("Displacement (m)", "Alice Mean 3D Displacement"),
    "Metrics/Alice/EMAReward": ("EMA Reward", "Alice EMA Reward"),
    "Metrics/IKFailRate": ("IK Fail Rate", "IK Fail Rate"),
    "GoalEncoder/embedding_norm": ("Norm", "Goal Encoder Embedding Norm"),
}

COMPARISON_METRICS = [
    {
        "title": "Success Rate",
        "ylabel": "Success Rate",
        "tags": ["Metrics/SuccessRate", "Metrics/Bob/SuccessRate"],
    },
    {
        "title": "Position SR",
        "ylabel": "Position SR",
        "tags": ["Metrics/Bob/PositionSR"],
    },
    {
        "title": "Rotation SR",
        "ylabel": "Rotation SR",
        "tags": ["Metrics/RotationSR", "Metrics/Bob/RotationSR"],
    },
    {
        "title": "Mean Reward",
        "ylabel": "Reward",
        "tags": ["Reward/Mean", "Reward/Bob"],
    },
    {
        "title": "EMA Reward",
        "ylabel": "EMA Reward",
        "tags": ["Reward/EMA"],
    },
    {
        "title": "Position Error",
        "ylabel": "Position Error (m)",
        "tags": ["Metrics/PosError", "Metrics/Bob/PosError"],
    },
    {
        "title": "Rotation Error",
        "ylabel": "Rotation Error (rad)",
        "tags": ["Metrics/RotError", "Metrics/Bob/RotError"],
    },
    {
        "title": "Policy Loss",
        "ylabel": "Surrogate Loss",
        "tags": ["Loss/Agent/Surrogate", "Loss/Bob/Surrogate"],
    },
    {
        "title": "Value Loss",
        "ylabel": "Value Loss",
        "tags": ["Loss/Agent/Value", "Loss/Bob/Value"],
    },
    {
        "title": "IK Fail Rate",
        "ylabel": "IK Fail Rate",
        "tags": ["Metrics/IKFailRate"],
    },
]


def resolve_summary_dir(path: Path) -> Path:
    has_events = any(f.name.startswith("events.out.tfevents") for f in path.iterdir() if f.is_file())
    if has_events:
        return path
    summary_sub = path / "summary"
    if summary_sub.is_dir():
        return summary_sub
    for child in path.iterdir():
        if child.is_dir():
            if any(f.name.startswith("events.out.tfevents") for f in child.iterdir() if f.is_file()):
                return child
    return path


def load_summary(summary_dir: Path) -> dict:
    summary_dir = resolve_summary_dir(summary_dir)
    ea = EventAccumulator(str(summary_dir))
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    data = {}
    for tag in scalars:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        wall_times = np.array([e.wall_time for e in events])
        data[tag] = {"steps": steps, "values": values, "wall_times": wall_times}
    return data


def detect_type(data: dict) -> str:
    if "Loss/Alice/Surrogate" in data or "Reward/Alice" in data:
        return "asp"
    return "push"


def smooth(values, weight=0.6):
    smoothed = np.zeros_like(values)
    last = values[0] if len(values) > 0 else 0.0
    for i, v in enumerate(values):
        smoothed_val = last * weight + (1 - weight) * v
        smoothed[i] = smoothed_val
        last = smoothed_val
    return smoothed


ALICE_COLOR = "#d62728"  # red
BOB_COLOR = "#1f77b4"    # blue
NEUTRAL_COLOR = "#2ca02c"  # green


def _tag_color(tag: str) -> str:
    tag_lower = tag.lower()
    if "alice" in tag_lower:
        return ALICE_COLOR
    if "bob" in tag_lower:
        return BOB_COLOR
    return NEUTRAL_COLOR


def plot_single_run(data: dict, run_type: str, out_dir: Path, label: str,
                    mode: str = "both", smoothing: float = 0.6):
    tags_map = ASP_TAGS if run_type == "asp" else PUSH_TAGS
    available = [tag for tag in tags_map if tag in data]

    if not available:
        print(f"[WARN] No plottable tags found for {label}")
        return

    if mode in ("separate", "both"):
        for tag in available:
            ylabel, title = tags_map[tag]
            color = _tag_color(tag) if run_type == "asp" else "C0"
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            steps = data[tag]["steps"]
            values = data[tag]["values"]
            ax.plot(steps, values, alpha=0.3, color=color)
            ax.plot(steps, smooth(values, smoothing), color=color, linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {label}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe_tag = tag.replace("/", "_")
            fig.savefig(out_dir / f"{safe_tag}.png", dpi=150)
            plt.close(fig)

    if mode in ("combined", "both"):
        n = len(available)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, tag in enumerate(available):
            ax = axes[idx]
            ylabel, title = tags_map[tag]
            color = _tag_color(tag) if run_type == "asp" else "C0"
            steps = data[tag]["steps"]
            values = data[tag]["values"]
            ax.plot(steps, values, alpha=0.3, color=color)
            ax.plot(steps, smooth(values, smoothing), color=color, linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)

        if run_type == "asp":
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=ALICE_COLOR, linewidth=2, label="Alice"),
                Line2D([0], [0], color=BOB_COLOR, linewidth=2, label="Bob"),
                Line2D([0], [0], color=NEUTRAL_COLOR, linewidth=2, label="Shared"),
            ]
            fig.legend(handles=legend_elements, loc="upper right",
                       fontsize=10, framealpha=0.9)

        fig.suptitle(f"{label}", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Plotted {len(available)} metrics for '{label}'")


def _find_tag(data: dict, tag_aliases: list) -> str | None:
    for tag in tag_aliases:
        if tag in data:
            return tag
    return None


def plot_comparison(all_data: list, labels: list, out_dir: Path,
                    mode: str = "both", smoothing: float = 0.6):
    available_metrics = []
    for metric in COMPARISON_METRICS:
        if any(_find_tag(d, metric["tags"]) is not None for d in all_data):
            available_metrics.append(metric)

    if not available_metrics:
        print("[WARN] No common metrics found for comparison")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_data), 10)))

    if mode in ("separate", "both"):
        for metric in available_metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for i, (data, lbl) in enumerate(zip(all_data, labels)):
                tag = _find_tag(data, metric["tags"])
                if tag is None:
                    continue
                steps = data[tag]["steps"]
                values = data[tag]["values"]
                ax.plot(steps, values, alpha=0.15, color=colors[i])
                ax.plot(steps, smooth(values, smoothing), color=colors[i],
                        linewidth=2, label=lbl)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric["ylabel"])
            ax.set_title(metric["title"])
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe_name = metric["title"].replace(" ", "_")
            fig.savefig(out_dir / f"cmp_{safe_name}.png", dpi=150)
            plt.close(fig)

    if mode in ("combined", "both"):
        n = len(available_metrics)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            for i, (data, lbl) in enumerate(zip(all_data, labels)):
                tag = _find_tag(data, metric["tags"])
                if tag is None:
                    continue
                steps = data[tag]["steps"]
                values = data[tag]["values"]
                ax.plot(steps, values, alpha=0.15, color=colors[i])
                ax.plot(steps, smooth(values, smoothing), color=colors[i],
                        linewidth=2, label=lbl)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric["ylabel"])
            ax.set_title(metric["title"])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Training Comparison", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Comparison plots: {len(available_metrics)} metrics across {len(all_data)} runs")


def write_csv(data: dict, out_dir: Path, prefix: str = ""):
    csv_dir = out_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    for tag, series in data.items():
        safe_tag = tag.replace("/", "_")
        fname = f"{prefix}{safe_tag}.csv" if prefix else f"{safe_tag}.csv"
        path = csv_dir / fname
        with open(path, "w") as f:
            f.write("step,value,wall_time\n")
            for s, v, w in zip(series["steps"], series["values"], series["wall_times"]):
                f.write(f"{s},{v},{w}\n")


def print_run_summary(data: dict, run_type: str, label: str):
    print(f"\n  {label} ({run_type.upper()}):")
    key_metrics = {
        "push": ["Metrics/SuccessRate", "Metrics/RotationSR", "Reward/Mean",
                 "Metrics/PosError", "Metrics/RotError"],
        "asp": ["Metrics/Bob/SuccessRate", "Metrics/Bob/PositionSR",
                "Reward/Bob", "Reward/Alice", "Metrics/Bob/PosError"],
    }
    for tag in key_metrics.get(run_type, []):
        if tag in data:
            vals = data[tag]["values"]
            if len(vals) > 0:
                last_10 = vals[-min(10, len(vals)):]
                print(f"    {tag:40s}  last={vals[-1]:.4f}  "
                      f"avg(last10)={np.mean(last_10):.4f}  "
                      f"max={np.max(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TensorBoard summaries and generate comparison plots."
    )
    parser.add_argument("--summary-dirs", nargs="+", type=str, required=True,
                        help="One or more paths to TensorBoard summary directories.")
    parser.add_argument("--labels", nargs="+", type=str, default=None,
                        help="Labels for each summary dir (defaults to dir parent name).")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory for all plots and CSVs (default: ./tb_analysis).")
    parser.add_argument("--mode", type=str, choices=["separate", "combined", "both"],
                        default="both",
                        help="'separate': one PNG per metric. 'combined': single grid PNG. "
                             "'both': produce both (default).")
    parser.add_argument("--smoothing", type=float, default=0.6,
                        help="Exponential smoothing weight (0=none, 0.99=very smooth).")
    parser.add_argument("--csv", action="store_true", default=False,
                        help="Also export CSV files for each metric.")
    parser.add_argument("--no-individual", action="store_true", default=False,
                        help="Skip individual run plots (only produce comparison).")

    args = parser.parse_args()

    summary_dirs = [Path(p) for p in args.summary_dirs]
    for sd in summary_dirs:
        if not sd.exists():
            parser.error(f"Summary directory does not exist: {sd}")

    if args.labels:
        if len(args.labels) != len(summary_dirs):
            parser.error("Number of --labels must match number of --summary-dirs")
        labels = args.labels
    else:
        labels = [sd.parent.name for sd in summary_dirs]

    out_dir = Path(args.out_dir) if args.out_dir else Path("tb_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"

    print(f"[INFO] Loading {len(summary_dirs)} TensorBoard summary dir(s) ...")
    all_data = []
    for sd, lbl in zip(summary_dirs, labels):
        print(f"  Loading: {sd}  ({lbl})")
        data = load_summary(sd)
        print(f"    Found {len(data)} scalar tag(s)")
        all_data.append(data)

    for data, lbl in zip(all_data, labels):
        run_type = detect_type(data)
        print_run_summary(data, run_type, lbl)

    if not args.no_individual:
        for data, lbl in zip(all_data, labels):
            run_type = detect_type(data)
            safe_label = lbl.replace(" ", "_").replace("/", "_")
            run_dir = out_dir / safe_label
            run_dir.mkdir(parents=True, exist_ok=True)
            plot_single_run(data, run_type, run_dir, lbl,
                            args.mode, args.smoothing)
            if args.csv:
                write_csv(data, csv_dir, prefix=f"{safe_label}_")

    if len(all_data) > 1:
        print(f"\n[INFO] Generating comparison plots ...")
        cmp_dir = out_dir / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_data, labels, cmp_dir,
                        args.mode, args.smoothing)

    print(f"\n[INFO] Output directory: {out_dir.resolve()}")
    print(f"[INFO]")
    print(f"[INFO]   {out_dir.name}/")
    for lbl in labels:
        safe_label = lbl.replace(" ", "_").replace("/", "_")
        print(f"[INFO]   ├── {safe_label}/        (individual plots)")
    if args.csv:
        print(f"[INFO]   ├── csv/                 (CSV exports)")
    if len(all_data) > 1:
        print(f"[INFO]   └── comparison/          (cross-run comparisons)")
    else:
        safe_label = labels[0].replace(" ", "_").replace("/", "_")
        print(f"[INFO]   └── {safe_label}/        (individual plots)")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
