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
    r"Loss=([-\d.]+)[^\|]*\|\s*Val=([-\d.]+)\s*\|\s*"
    r"Rew=([-+\d.]+)\s*\(EMA\s+([-+\d.]+)\)\s*\|\s*"
    r"PosErr=([-\d.]+)\s*\|\s*RotErr=([-\d.]+)\s*\|\s*"
    r"SR=([-\d.]+)\s*\|\s*RotSR=([-\d.]+)\s*\|\s*"
    r"IK_fail=([-\d.]+)\s*\|\s*"
    r"AvgPushes=(nan|[-\d.]+)\s*\|\s*Epi=(\d+)\s*\|\s*"
    r"BestSR=([-\d.]+)"
)

EPISODE_RE = re.compile(
    r"\[Episode\]\s+pushes=(\d+)\s+(SUCCESS|fail)\s+"
    r"rew=[-+\d.]+\s+"
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
        avg_p = float(m.group(11)) if m.group(11) != "nan" else None
        iters.append({
            "iter":       n,
            "loss":       float(m.group(2)),
            "val":        float(m.group(3)),
            "rew":        float(m.group(4)),
            "rew_ema":    float(m.group(5)),
            "pos_err":    float(m.group(6)),
            "rot_err":    float(m.group(7)),
            "sr":         float(m.group(8)),
            "rot_sr":     float(m.group(9)),
            "ik_fail":    float(m.group(10)),
            "avg_pushes": avg_p,
            "episodes":   int(m.group(12)),
            "best_sr":    float(m.group(13)),
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
    fields = ["iter", "loss", "val", "rew", "rew_ema", "pos_err", "rot_err",
              "sr", "rot_sr", "ik_fail", "avg_pushes", "episodes", "best_sr"]
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
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Push-PPO Training Overview", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    xs = [r["iter"] for r in iters]

    def _plot(ax, ys_raw, label, color="tab:blue", ref=None):
        if not xs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        ax.plot(xs, smooth(list(ys_raw)), color=color, linewidth=1.5, label=label)
        if ref is not None:
            ax.axhline(ref, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
                       label=f"ref={ref}")

    # Row 0 — Loss, Value Loss, Reward
    ax_loss = fig.add_subplot(gs[0, 0])
    _plot(ax_loss, [r["loss"] for r in iters], "Policy Loss", "tab:blue")
    _fmt(ax_loss, "Loss", "Policy Loss")

    ax_val = fig.add_subplot(gs[0, 1])
    _plot(ax_val, [r["val"] for r in iters], "Value Loss", "tab:orange")
    _fmt(ax_val, "Loss", "Value Loss")

    ax_rew = fig.add_subplot(gs[0, 2])
    _plot(ax_rew, [r["rew_ema"] for r in iters], "Reward (EMA)", "tab:green")
    ax_rew.plot(xs, smooth([r["rew"] for r in iters]), color="tab:green",
                linewidth=0.8, alpha=0.3, linestyle="--", label="Raw")
    _fmt(ax_rew, "Reward", "Reward (EMA + Raw)")

    # Row 1 — Combined SR (PosSR + RotSR), Best SR, Avg Pushes + Episodes
    ax_sr = fig.add_subplot(gs[1, 0])
    _plot(ax_sr, [r["sr"] for r in iters], "Position SR", "tab:blue")
    _plot(ax_sr, [r["rot_sr"] for r in iters], "Rotation SR", "tab:purple")
    ax_sr.set_ylim(0, 1)
    _fmt(ax_sr, "Success Rate", "Success Rate (Position + Rotation)")

    ax_best = fig.add_subplot(gs[1, 1])
    best_vals = [(r["iter"], r["best_sr"]) for r in iters if r["best_sr"] >= 0.0]
    if best_vals:
        bx, by = zip(*best_vals)
        ax_best.plot(bx, by, color="gold", linewidth=1.5, label="Best SR")
    ax_best.set_ylim(0, 1)
    _fmt(ax_best, "Success Rate", "Best SR (cumulative)")

    ax_pushes = fig.add_subplot(gs[1, 2])
    avg_p_xs = [r["iter"] for r in iters if r["avg_pushes"] is not None]
    avg_p_ys = [r["avg_pushes"] for r in iters if r["avg_pushes"] is not None]
    if avg_p_xs:
        ax_pushes.plot(avg_p_xs, smooth(avg_p_ys), color="tab:cyan",
                       linewidth=1.5, label="Avg Pushes / Epi")
    ax2 = ax_pushes.twinx()
    ax2.plot(xs, smooth([r["episodes"] for r in iters]), color="tab:brown",
             linewidth=1.0, linestyle="--", label="Episodes")
    ax_pushes.set_title("Avg Pushes & Episodes Completed")
    ax_pushes.set_xlabel("Iteration")
    ax_pushes.set_ylabel("Pushes", color="tab:cyan")
    ax_pushes.tick_params(axis="y", labelcolor="tab:cyan")
    ax2.set_ylabel("Count", color="tab:brown")
    ax2.tick_params(axis="y", labelcolor="tab:brown")
    lines1, lab1 = ax_pushes.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax_pushes.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax_pushes.grid(True, alpha=0.3)

    # Row 2 — Pos+Rot Error (dual y-axis), Episodes, (empty)
    ax_err = fig.add_subplot(gs[2, 0:2])
    ax_pos = ax_err
    ax_rot = ax_err.twinx()
    _plot(ax_pos, [r["pos_err"] for r in iters], "Position Error", "tab:red")
    _plot(ax_rot, [r["rot_err"] for r in iters], "Rotation Error", "tab:blue")
    ax_pos.set_title("Mean Error to Goal")
    ax_pos.set_xlabel("Iteration")
    ax_pos.set_ylabel("Position (m)", color="tab:red")
    ax_pos.tick_params(axis="y", labelcolor="tab:red")
    ax_rot.set_ylabel("Rotation (rad)", color="tab:blue")
    ax_rot.tick_params(axis="y", labelcolor="tab:blue")
    lines1, lab1 = ax_pos.get_legend_handles_labels()
    lines2, lab2 = ax_rot.get_legend_handles_labels()
    ax_pos.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    ax_spare = fig.add_subplot(gs[2, 2])
    ax_spare.axis("off")

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
        ax2.set_ylim(0, 1)
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
