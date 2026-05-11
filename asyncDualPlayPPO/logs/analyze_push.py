#!/usr/bin/env python3
"""
Parse push-PPO training logs, write CSVs, and plot metrics.

Expects log files containing lines produced by train_push.py:

  [Iter     5] Loss=-0.0123↓ | Val=0.0456 | Rew=+0.1234 (EMA +0.1001) | PosErr=0.4567 | SR=0.1234 | IK_fail=0.0123 | AvgPushes=12.3 | Epi=15 | BestSR=0.3000

Usage:
  python analyze_push.py --log-file runs/push_ppo_baseline/train.log
  python analyze_push.py --log-file run1.log run2.log --out-dir analysis/
"""

import re
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Patterns ---
ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+"
    r"Loss=([-\d.]+).?\s*\|\s*Val=([-\d.]+)\s*\|\s*"
    r"Rew=([-+\d.]+)\s*\(EMA\s+([-+\d.]+)\)\s*\|\s*"
    r"PosErr=([-\d.]+)\s*\|\s*SR=([-\d.]+)\s*\|\s*"
    r"IK_fail=([-\d.]+)\s*\|\s*"
    r"AvgPushes=(nan|[-\d.]+)\s*\|\s*Epi=(\d+)\s*\|\s*"
    r"BestSR=([-\d.]+)"
)


def parse_log(path: Path) -> list[dict]:
    """Parse one log file. Returns list of iter_records."""
    text = path.read_text(errors="replace")
    iters = []
    seen = set()
    for m in ITER_RE.finditer(text):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        avg_p = float(m.group(9)) if m.group(9) != "nan" else None
        iters.append({
            "iter": n,
            "loss": float(m.group(2)),
            "val":  float(m.group(3)),
            "rew":  float(m.group(4)),
            "rew_ema": float(m.group(5)),
            "pos_err": float(m.group(6)),
            "sr":   float(m.group(7)),
            "ik_fail": float(m.group(8)),
            "avg_pushes": avg_p,
            "episodes":  int(m.group(10)),
            "best_sr":   float(m.group(11)),
        })

    iters.sort(key=lambda x: x["iter"])
    return iters


def merge_logs(paths: list[Path]) -> list[dict]:
    """Merge multiple log files; deduplicate by iter number, keep lowest-path occurrence."""
    all_iters: list[dict] = []
    seen: set[int] = set()
    for p in paths:
        iters = parse_log(p)
        for r in iters:
            if r["iter"] not in seen:
                seen.add(r["iter"])
                all_iters.append(r)
    all_iters.sort(key=lambda x: x["iter"])
    return all_iters


def write_csv(iters: list[dict], out_dir: Path):
    i_path = out_dir / "push_iters.csv"
    fields = ["iter", "loss", "val", "rew", "rew_ema", "pos_err", "sr", "ik_fail", "avg_pushes", "episodes", "best_sr"]
    with open(i_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(iters)
    print(f"[INFO] Wrote {i_path}")


def smooth(vals: list, window: int = 5) -> list:
    if len(vals) < window:
        return vals
    result = []
    for i in range(len(vals)):
        s = max(0, i - window // 2)
        e = min(len(vals), i + window // 2 + 1)
        result.append(sum(vals[s:e]) / (e - s))
    return result


def plot_metrics(iters: list[dict], out_dir: Path):
    if not iters:
        print("[WARN] No data to plot.")
        return

    def _fmt(ax, ylabel, title):
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Push-PPO Training Overview", fontsize=14, fontweight="bold")

    xs = [r["iter"] for r in iters]

    def _plot(ax, ys_raw, label, color="tab:blue", ref=None):
        if not xs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        ax.plot(xs, smooth(ys_raw), color=color, linewidth=1.5, label=label)
        if ref is not None:
            ax.axhline(ref, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
                       label=f"ref={ref}")

    # Row 0: Loss, Value Loss, Reward
    _plot(axes[0, 0], [r["loss"] for r in iters], "Policy Loss", "tab:blue")
    _fmt(axes[0, 0], "Loss", "Policy Loss")

    _plot(axes[0, 1], [r["val"] for r in iters], "Value Loss", "tab:orange")
    _fmt(axes[0, 1], "Loss", "Value Loss")

    _plot(axes[0, 2], [r["rew_ema"] for r in iters], "Reward (EMA)", "tab:green")
    axes[0, 2].plot(xs, smooth([r["rew"] for r in iters]), color="tab:green",
                    linewidth=0.8, alpha=0.3, linestyle="--", label="Raw")
    _fmt(axes[0, 2], "Reward", "Reward (EMA + Raw)")

    # Row 1: SR (step-level), IK fail rate, Best SR
    _plot(axes[1, 0], [r["sr"] for r in iters], "Step SR", "tab:blue")
    _fmt(axes[1, 0], "Success Rate", "Step Success Rate")

    _plot(axes[1, 1], [r["ik_fail"] for r in iters], "IK Fail Rate", "tab:red", ref=0.05)
    _fmt(axes[1, 1], "IK Fail Rate", "IK Fail Rate (5% threshold)")

    _plot(axes[1, 2], [r["best_sr"] for r in iters], "Best SR", "gold")
    _fmt(axes[1, 2], "Success Rate", "Best SR (cumulative)")

    # Row 2: Position Error, Avg Pushes, Episodes
    _plot(axes[2, 0], [r["pos_err"] for r in iters], "Mean Pos Error", "tab:magenta")
    _fmt(axes[2, 0], "Distance (m)", "Mean Position Error to Goal")

    avg_p_xs = [r["iter"] for r in iters if r["avg_pushes"] is not None]
    avg_p_ys = [r["avg_pushes"] for r in iters if r["avg_pushes"] is not None]
    _plot(axes[2, 1], avg_p_ys if avg_p_xs else [], "Avg Pushes / Episode", "tab:cyan")
    if avg_p_xs:
        axes[2, 1].set_xticks(avg_p_xs)
    _fmt(axes[2, 1], "Pushes", "Avg Pushes per Episode")

    _plot(axes[2, 2], [r["episodes"] for r in iters], "Episodes Completed", "tab:brown")
    _fmt(axes[2, 2], "Count", "Episodes Completed per Iteration")

    plt.tight_layout()
    _save(fig, "plot_push_overview.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze push-PPO training logs.")
    parser.add_argument(
        "--log-file", type=str, nargs="+", required=True,
        help="One or more log files to parse (stitched in order).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (defaults to directory of first log file).",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.log_file]
    out_dir = Path(args.out_dir) if args.out_dir else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Parsing {len(paths)} log file(s)...")
    iters = merge_logs(paths)
    print(f"[INFO] Iter records: {len(iters)}")
    if iters:
        print(f"[INFO] Iter range: {iters[0]['iter']} → {iters[-1]['iter']}")

    write_csv(iters, out_dir)
    plot_metrics(iters, out_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
