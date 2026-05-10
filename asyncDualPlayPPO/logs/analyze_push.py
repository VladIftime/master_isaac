#!/usr/bin/env python3
"""
Parse push-PPO training logs, write CSVs, and plot metrics.

Expects log files containing lines produced by train_push.py:

  [Push Update    5] Loss:  0.0123 | Val:  0.0456 | Rew:  0.1234 | SR:  0.2500
  [Iter     5] SR=0.2500 | IK_fail=0.0123 | Rew=0.1234 | AvgPushes=12.3 | Episodes=15 | BestSR=0.3000

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
UPDATE_RE = re.compile(
    r"\[Push Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*"
    r"\|\s*Rew:\s*([-\d.]+)\s*\|\s*SR:\s*([-\d.]+)"
)
ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+SR=([-\d.]+)\s*\|\s*IK_fail=([-\d.]+)\s*"
    r"\|\s*Rew=([-\d.]+)\s*\|\s*AvgPushes=(nan|[-\d.]+)\s*"
    r"\|\s*Episodes=(\d+)\s*\|\s*BestSR=([-\d.]+)"
)


def parse_log(path: Path) -> tuple[list[dict], list[dict]]:
    """Parse one log file. Returns (update_records, iter_records)."""
    text = path.read_text(errors="replace")
    updates = []
    seen_u = set()
    for m in UPDATE_RE.finditer(text):
        n = int(m.group(1))
        if n in seen_u:
            continue
        seen_u.add(n)
        updates.append({
            "iter": n,
            "loss": float(m.group(2)),
            "val": float(m.group(3)),
            "rew": float(m.group(4)),
            "sr": float(m.group(5)),
        })

    iters = []
    seen_i = set()
    for m in ITER_RE.finditer(text):
        n = int(m.group(1))
        if n in seen_i:
            continue
        seen_i.add(n)
        avg_p = float(m.group(5)) if m.group(5) != "nan" else None
        iters.append({
            "iter": n,
            "sr": float(m.group(2)),
            "ik_fail": float(m.group(3)),
            "rew": float(m.group(4)),
            "avg_pushes": avg_p,
            "episodes": int(m.group(6)),
            "best_sr": float(m.group(7)),
        })

    updates.sort(key=lambda x: x["iter"])
    iters.sort(key=lambda x: x["iter"])
    return updates, iters


def merge_logs(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Merge multiple log files; deduplicate by iter number, keep lowest-path occurrence."""
    all_updates: list[dict] = []
    all_iters: list[dict] = []
    seen_u: set[int] = set()
    seen_i: set[int] = set()
    for p in paths:
        updates, iters = parse_log(p)
        for r in updates:
            if r["iter"] not in seen_u:
                seen_u.add(r["iter"])
                all_updates.append(r)
        for r in iters:
            if r["iter"] not in seen_i:
                seen_i.add(r["iter"])
                all_iters.append(r)
    all_updates.sort(key=lambda x: x["iter"])
    all_iters.sort(key=lambda x: x["iter"])
    return all_updates, all_iters


def write_csv(updates: list[dict], iters: list[dict], out_dir: Path):
    u_path = out_dir / "push_updates.csv"
    with open(u_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "loss", "val", "rew", "sr"])
        w.writeheader()
        w.writerows(updates)
    print(f"[INFO] Wrote {u_path}")

    i_path = out_dir / "push_iters.csv"
    with open(i_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "sr", "ik_fail", "rew", "avg_pushes", "episodes", "best_sr"])
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


def plot_metrics(updates: list[dict], iters: list[dict], out_dir: Path):
    if not updates and not iters:
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

    u_xs = [r["iter"] for r in updates]
    i_xs = [r["iter"] for r in iters]

    def _plot(ax, xs, ys_raw, label, color="tab:blue", ref=None):
        if not xs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        ax.plot(xs, smooth(ys_raw), color=color, linewidth=1.5, label=label)
        if ref is not None:
            ax.axhline(ref, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
                       label=f"ref={ref}")

    # Row 0: losses and step reward
    _plot(axes[0, 0], u_xs, [r["loss"] for r in updates], "Policy Loss", "tab:blue")
    _fmt(axes[0, 0], "Loss", "Policy Loss")

    _plot(axes[0, 1], u_xs, [r["val"] for r in updates], "Value Loss", "tab:orange")
    _fmt(axes[0, 1], "Loss", "Value Loss")

    _plot(axes[0, 2], u_xs, [r["rew"] for r in updates], "Step Reward", "tab:green")
    _fmt(axes[0, 2], "Reward", "Mean Step Reward")

    # Row 1: SR (step-level and episodic), IK fail rate
    _plot(axes[1, 0], u_xs, [r["sr"] for r in updates], "Step SR", "tab:blue")
    if iters:
        i_sr = [r["sr"] for r in iters]
        axes[1, 0].plot(i_xs, smooth(i_sr), color="tab:purple", linewidth=1.5,
                        linestyle="--", label="Iter SR")
    _fmt(axes[1, 0], "Success Rate", "Success Rate (step & iter)")

    _plot(axes[1, 1], i_xs, [r["ik_fail"] for r in iters], "IK Fail Rate", "tab:red",
          ref=0.05)
    _fmt(axes[1, 1], "IK Fail Rate", "IK Fail Rate (5% threshold)")

    _plot(axes[1, 2], i_xs, [r["best_sr"] for r in iters], "Best SR", "gold")
    _fmt(axes[1, 2], "Success Rate", "Best SR (cumulative)")

    # Row 2: episode-level stats
    avg_p_xs = [r["iter"] for r in iters if r["avg_pushes"] is not None]
    avg_p_ys = [r["avg_pushes"] for r in iters if r["avg_pushes"] is not None]
    _plot(axes[2, 0], avg_p_xs, avg_p_ys, "Avg Pushes / Episode", "tab:cyan")
    _fmt(axes[2, 0], "Pushes", "Avg Pushes per Episode")

    _plot(axes[2, 1], i_xs, [r["episodes"] for r in iters], "Episodes Completed", "tab:brown")
    _fmt(axes[2, 1], "Count", "Episodes Completed per Iteration")

    # Reward from iter summary
    _plot(axes[2, 2], i_xs, [r["rew"] for r in iters], "Iter Mean Reward", "tab:green")
    _fmt(axes[2, 2], "Reward", "Iter Mean Reward")

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
    updates, iters = merge_logs(paths)
    print(f"[INFO] Update records: {len(updates)},  Iter records: {len(iters)}")
    if updates:
        print(f"[INFO] Iter range: {updates[0]['iter']} → {updates[-1]['iter']}")

    write_csv(updates, iters, out_dir)
    plot_metrics(updates, iters, out_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
