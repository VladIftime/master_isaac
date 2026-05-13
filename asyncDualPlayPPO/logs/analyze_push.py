#!/usr/bin/env python3
"""
Parse push-PPO training logs (or SLURM .out files), write CSVs, and plot metrics.

Parses [Iter N], [Episode], and [Push N] lines produced by train_push.py.

Usage:
  python analyze_push.py --log-file logs/ppo_push.log
  python analyze_push.py --log-file slurm-12345.out --out-dir analysis/
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

EPISODE_RE = re.compile(
    r"\[Episode\]\s+pushes=(\d+)\s+(SUCCESS|fail)\s+"
    r"goal=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+orient=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
    r"final=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
    r"rot=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
    r"err_pos=([-\d.]+)m\s+err_rot=([-\d.]+)rad"
)

PUSH_RE = re.compile(
    r"\[Push\s+(\d+)\]\s+"
    r"rew=([-+\d.]+)\s+\([^)]+\)\s+"
    r"pos_err=([-\d.]+)\s+"
    r"rot_err=([-\d.]+)\s+"
    r"at_goal=(\d+)/(\d+)"
)


def parse_file(path: Path) -> dict:
    """Parse one log / SLURM .out file. Returns dict of parsed records."""
    text = path.read_text(errors="replace")

    iters = []
    seen: set[int] = set()
    for m in ITER_RE.finditer(text):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        avg_p = float(m.group(9)) if m.group(9) != "nan" else None
        iters.append({
            "iter":       n,
            "loss":       float(m.group(2)),
            "val":        float(m.group(3)),
            "rew":        float(m.group(4)),
            "rew_ema":    float(m.group(5)),
            "pos_err":    float(m.group(6)),
            "sr":         float(m.group(7)),
            "ik_fail":    float(m.group(8)),
            "avg_pushes": avg_p,
            "episodes":   int(m.group(10)),
            "best_sr":    float(m.group(11)),
        })

    episodes = []
    for m in EPISODE_RE.finditer(text):
        episodes.append({
            "pushes":   int(m.group(1)),
            "success":  m.group(2) == "SUCCESS",
            "goal_x":   float(m.group(3)),
            "goal_y":   float(m.group(4)),
            "goal_z":   float(m.group(5)),
            "goal_yaw": float(m.group(8)),
            "final_x":  float(m.group(9)),
            "final_y":  float(m.group(10)),
            "final_z":  float(m.group(11)),
            "err_pos":  float(m.group(15)),
            "err_rot":  float(m.group(16)),
        })

    pushes = []
    for m in PUSH_RE.finditer(text):
        pushes.append({
            "push_step": int(m.group(1)),
            "rew":       float(m.group(2)),
            "pos_err":   float(m.group(3)),
            "rot_err":   float(m.group(4)),
            "at_goal":   int(m.group(5)),
            "n_envs":    int(m.group(6)),
        })

    iters.sort(key=lambda x: x["iter"])
    return {"iters": iters, "episodes": episodes, "pushes": pushes}


def merge_files(paths: list[Path]) -> dict:
    """Merge multiple log files; deduplicate iters by number."""
    all_iters:    list[dict] = []
    all_episodes: list[dict] = []
    all_pushes:   list[dict] = []
    seen_i: set[int] = set()
    for p in paths:
        data = parse_file(p)
        for r in data["iters"]:
            if r["iter"] not in seen_i:
                seen_i.add(r["iter"])
                all_iters.append(r)
        all_episodes.extend(data["episodes"])
        all_pushes.extend(data["pushes"])
    all_iters.sort(key=lambda x: x["iter"])
    return {"iters": all_iters, "episodes": all_episodes, "pushes": all_pushes}


def write_csv(data: dict, out_dir: Path):
    i_path = out_dir / "push_iters.csv"
    fields = ["iter", "loss", "val", "rew", "rew_ema", "pos_err", "sr",
              "ik_fail", "avg_pushes", "episodes", "best_sr"]
    with open(i_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data["iters"])
    print(f"[INFO] Wrote {i_path}")

    if data["episodes"]:
        e_path = out_dir / "push_episodes.csv"
        with open(e_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "pushes", "success", "goal_x", "goal_y", "goal_z", "goal_yaw",
                "final_x", "final_y", "final_z", "err_pos", "err_rot"])
            w.writeheader()
            w.writerows(data["episodes"])
        print(f"[INFO] Wrote {e_path} ({len(data['episodes'])} episodes)")


def smooth(vals: list, window: int = 5) -> list:
    if len(vals) < window:
        return vals
    result = []
    for i in range(len(vals)):
        s = max(0, i - window // 2)
        e = min(len(vals), i + window // 2 + 1)
        result.append(sum(vals[s:e]) / (e - s))
    return result


def plot_metrics(data: dict, out_dir: Path):
    iters    = data["iters"]
    episodes = data["episodes"]
    if not iters and not episodes:
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

    # Row 1: SR, IK fail rate, Best SR
    _plot(axes[1, 0], [r["sr"] for r in iters], "Step SR", "tab:blue")
    _fmt(axes[1, 0], "Success Rate", "Step Success Rate")

    _plot(axes[1, 1], [r["ik_fail"] for r in iters], "IK Fail Rate", "tab:red", ref=0.05)
    _fmt(axes[1, 1], "IK Fail Rate", "IK Fail Rate (5% threshold)")

    _plot(axes[1, 2], [r["best_sr"] for r in iters], "Best SR", "gold")
    _fmt(axes[1, 2], "Success Rate", "Best SR (cumulative)")

    # Row 2: Pos Error, Avg Pushes, Episodes
    _plot(axes[2, 0], [r["pos_err"] for r in iters], "Mean Pos Error", "magenta")
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

    # Rolling success rate over episodes
    if episodes:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        window = 50
        sr_rolling = []
        for i in range(len(episodes)):
            s = max(0, i - window)
            chunk = episodes[s:i + 1]
            sr_rolling.append(sum(1 for e in chunk if e["success"]) / len(chunk))
        ax2.plot(range(len(sr_rolling)), sr_rolling, color="tab:blue", linewidth=1.5,
                 label=f"Rolling SR (window={window})")
        ax2.set_title("Episode-Level Success Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Success Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        _save(fig2, "plot_episode_sr.png")


def print_summary(data: dict):
    iters    = data["iters"]
    episodes = data["episodes"]
    sep = "=" * 60
    if iters:
        last = iters[-1]
        print(f"\n{sep}")
        print("TRAINING SUMMARY")
        print(sep)
        print(f"  Iterations:     {len(iters)}")
        print(f"  Iter range:     {iters[0]['iter']} → {last['iter']}")
        print(f"  Last Loss:      {last['loss']:.4f}")
        print(f"  Last SR:        {last['sr']:.4f}")
        print(f"  Best SR:        {last['best_sr']:.4f}")
        print(f"  Last PosErr:    {last['pos_err']:.4f} m")
        print(f"  Last IK fail:   {last['ik_fail']:.4f}")
    if episodes:
        n_success = sum(1 for e in episodes if e["success"])
        errs      = [e["err_pos"] for e in episodes]
        rot_errs  = [e["err_rot"] for e in episodes]
        pushes    = [e["pushes"]  for e in episodes]
        print(f"  Episodes:       {len(episodes)}")
        print(f"  Successes:      {n_success} ({100*n_success/len(episodes):.1f}%)")
        print(f"  Mean pos_err:   {sum(errs)/len(errs):.4f} m")
        print(f"  Mean rot_err:   {sum(rot_errs)/len(rot_errs):.4f} rad")
        print(f"  Mean pushes:    {sum(pushes)/len(pushes):.1f}")
    if iters or episodes:
        print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze push-PPO training logs.")
    parser.add_argument(
        "--log-file", type=str, nargs="+", required=True,
        help="One or more log or SLURM .out files to parse.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (defaults to directory of first log file).",
    )
    args = parser.parse_args()

    paths   = [Path(p) for p in args.log_file]
    out_dir = Path(args.out_dir) if args.out_dir else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Parsing {len(paths)} file(s)...")
    data = merge_files(paths)
    print(f"[INFO] Iter records: {len(data['iters'])},  "
          f"Episodes: {len(data['episodes'])},  "
          f"Push steps: {len(data['pushes'])}")

    write_csv(data, out_dir)
    plot_metrics(data, out_dir)
    print_summary(data)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
