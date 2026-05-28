#!/usr/bin/env python3
"""
Analyze SLURM / log output from train_push.py.

Parses [Iter N], [Episode], and [Push N] lines from any *.out or *.log file,
writes CSVs, and generates training metric plots.

Usage:
  python -m asyncDualPlayPPO.logs.analyze_push --log slurm-*.out
  python -m asyncDualPlayPPO.logs.analyze_push --log push_prim_test.log -o analysis/
"""

import re
import csv
import argparse
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Flexible iter line parser ────────────────────────────────────────────
# Matches key=value pairs in any order from an [Iter N] line.
ITER_LINE_RE = re.compile(r"\[Iter\s+(\d+)\]\s+(.*)")
KV_RE = re.compile(r"([A-Za-z_]+)=([-+\d.]+(?:e[+-]\d+)?|nan)\s*")


def _parse_iter_line(line: str) -> dict | None:
    m = ITER_LINE_RE.match(line)
    if not m:
        return None
    it = int(m.group(1))
    tail = m.group(2)
    rec = {"iter": it}
    for km in KV_RE.finditer(tail):
        k = km.group(1)
        v = km.group(2)
        rec[k] = float(v) if v != "nan" else math.nan
    return rec


# ── Episode line parser ──────────────────────────────────────────────────
EPISODE_RE = re.compile(
    r"\[Episode\]\s+pushes=(\d+)\s+(SUCCESS|fail)\s+"
    r"rew=([-+\d.]+)\s+"
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


def parse_log(path: Path) -> dict:
    """Parse a log file. Returns {iters, episodes, pushes}."""
    text = path.read_text(errors="replace")
    lines = text.split("\n")

    iters: dict[int, dict] = {}
    for line in lines:
        rec = _parse_iter_line(line)
        if rec is not None:
            i = rec["iter"]
            if i not in iters:
                iters[i] = rec

    episodes = []
    for m in EPISODE_RE.finditer(text):
        episodes.append({
            "pushes": int(m.group(1)),
            "success": m.group(2) == "SUCCESS",
            "rew": float(m.group(3)),
            "goal_x": float(m.group(4)),
            "goal_y": float(m.group(5)),
            "goal_z": float(m.group(6)),
            "goal_rx": float(m.group(7)),
            "goal_ry": float(m.group(8)),
            "goal_rz": float(m.group(9)),
            "final_x": float(m.group(10)),
            "final_y": float(m.group(11)),
            "final_z": float(m.group(12)),
            "rot_rx": float(m.group(13)),
            "rot_ry": float(m.group(14)),
            "rot_rz": float(m.group(15)),
            "err_pos": float(m.group(16)),
            "err_rot": float(m.group(17)),
        })

    pushes = []
    for m in PUSH_RE.finditer(text):
        pushes.append({
            "push_step": int(m.group(1)),
            "rew": float(m.group(2)),
            "pos_err": float(m.group(3)),
            "rot_err": float(m.group(4)),
            "at_goal": int(m.group(5)),
            "n_envs": int(m.group(6)),
        })

    return {
        "iters": sorted(iters.values(), key=lambda r: r["iter"]),
        "episodes": episodes,
        "pushes": pushes,
    }


def merge_paths(paths: list[Path]) -> dict:
    """Merge multiple log files; deduplicate iters by number."""
    seen_i: set[int] = set()
    all_iters: list[dict] = []
    all_episodes: list[dict] = []
    all_pushes: list[dict] = []
    for p in paths:
        data = parse_log(p)
        for r in data["iters"]:
            if r["iter"] not in seen_i:
                seen_i.add(r["iter"])
                all_iters.append(r)
        all_episodes.extend(data["episodes"])
        all_pushes.extend(data["pushes"])
    all_iters.sort(key=lambda r: r["iter"])
    return {"iters": all_iters, "episodes": all_episodes, "pushes": all_pushes}


def write_csv(data: dict, out_dir: Path):
    iters = data["iters"]
    if iters:
        # Collect all keys across all iter records
        all_keys = set()
        for r in iters:
            all_keys.update(r.keys())
        fields = sorted(all_keys)
        i_path = out_dir / "push_iters.csv"
        with open(i_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(iters)
        print(f"[INFO] Wrote {i_path} ({len(iters)} rows)")

    ep = data["episodes"]
    if ep:
        e_path = out_dir / "push_episodes.csv"
        with open(e_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "pushes", "success", "rew",
                "goal_x", "goal_y", "goal_z", "goal_rx", "goal_ry", "goal_rz",
                "final_x", "final_y", "final_z",
                "rot_rx", "rot_ry", "rot_rz",
                "err_pos", "err_rot",
            ])
            w.writeheader()
            w.writerows(ep)
        print(f"[INFO] Wrote {e_path} ({len(ep)} episodes)")

    pu = data["pushes"]
    if pu:
        p_path = out_dir / "push_steps.csv"
        with open(p_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["push_step", "rew", "pos_err", "rot_err", "at_goal", "n_envs"])
            w.writeheader()
            w.writerows(pu)
        print(f"[INFO] Wrote {p_path} ({len(pu)} push summaries)")


# ── Plotting ─────────────────────────────────────────────────────────────

def smooth(arr, window=7):
    """Simple moving average."""
    arr = list(arr)
    if len(arr) < window:
        return arr if isinstance(arr, np.ndarray) else np.array(arr)
    y = np.array(arr)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _safe_get(rec: dict, key: str, default=np.nan) -> float:
    v = rec.get(key, default)
    return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else np.nan


def plot_metrics(data: dict, out_dir: Path):
    iters = data["iters"]
    episodes = data["episodes"]
    if not iters:
        print("[WARN] No iter data to plot.")
        return

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    xs = [r["iter"] for r in iters]
    x_arr = np.array(xs)

    def _axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # ── Main overview (4×2 grid) ────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle("Push-PPO Training Overview", fontsize=14, fontweight="bold")

    # Rewards
    ax = axes[0, 0]
    rew = [_safe_get(r, "Rew") for r in iters]
    rew_ema = [_safe_get(r, "EMA") for r in iters]
    ax.plot(x_arr, smooth(rew), color="green", alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x_arr, smooth(rew_ema), color="green", linewidth=1.5, label="EMA")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    _axis(ax, "Mean Reward per Push", "Reward")
    ax.legend(fontsize=8)

    # Loss / Val
    ax = axes[0, 1]
    loss = [_safe_get(r, "Loss") for r in iters]
    val = [_safe_get(r, "Val") for r in iters]
    ax.plot(x_arr, smooth(loss), color="blue", alpha=0.3, linewidth=0.8, label="Loss (raw)")
    ax.plot(x_arr, smooth(val), color="orange", linewidth=1.5, label="Value Loss")
    _axis(ax, "Policy & Value Loss", "Loss")
    ax.legend(fontsize=8)

    # PosErr / RotErr
    ax = axes[1, 0]
    pe = [_safe_get(r, "PosErr") for r in iters]
    re = [_safe_get(r, "RotErr") for r in iters]
    ax.plot(x_arr, smooth(pe), color="magenta", linewidth=1.5, label="PosErr (m)")
    ax.set_ylabel("Position Error (m)")
    ax2r = ax.twinx()
    ax2r.plot(x_arr, smooth(re), color="purple", linewidth=1.5, label="RotErr (rad)")
    ax2r.set_ylabel("Rotation Error (rad)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    _axis(ax, "Position & Rotation Error", "")

    # SR / RotSR
    ax = axes[1, 1]
    sr = [_safe_get(r, "SR") for r in iters]
    rs = [_safe_get(r, "RotSR") for r in iters]
    ax.plot(x_arr, smooth(sr), color="blue", linewidth=1.5, label="Pos SR")
    ax.plot(x_arr, smooth(rs), color="cyan", linewidth=1.5, label="Rot SR")
    _axis(ax, "Success Rate (Position & Rotation)", "Success Rate")
    ax.legend(fontsize=8)

    # IK fail
    ax = axes[2, 0]
    ik = [_safe_get(r, "IK_fail") for r in iters]
    ax.plot(x_arr, smooth(ik), color="red", linewidth=1.5, label="IK Fail Rate")
    ax.axhline(0.01, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, label="1% threshold")
    _axis(ax, "IK Fail Rate", "Rate")
    ax.legend(fontsize=8)

    # Avg pushes / episode
    ax = axes[2, 1]
    ap = [_safe_get(r, "AvgPushes") for r in iters]
    ax.plot(x_arr, smooth(ap), color="cyan", linewidth=1.5, label="Avg Pushes/Ep")
    _axis(ax, "Avg Pushes per Episode", "Pushes")
    ax.legend(fontsize=8)

    # Episodes per iteration
    ax = axes[3, 0]
    ep = [_safe_get(r, "Epi") for r in iters]
    ax.plot(x_arr, smooth(ep), color="brown", linewidth=1.5, label="Episodes")
    _axis(ax, "Episodes per Iteration", "Count")
    ax.legend(fontsize=8)

    # Best SR
    ax = axes[3, 1]
    bs = [_safe_get(r, "BestSR") for r in iters]
    bs_clean = [(x, b) for x, b in zip(xs, bs) if b > 0]
    if bs_clean:
        ax.plot([b[0] for b in bs_clean], [b[1] for b in bs_clean],
                color="gold", linewidth=1.5, label="Best SR")
    _axis(ax, "Best Cumulative Success Rate", "Success Rate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "plot_push_overview.png")

    # ── Episode-level rolling SR ─────────────────────────────────────────
    if episodes:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        window = 100
        srs = [1.0 if e["success"] else 0.0 for e in episodes]
        rolling = np.convolve(srs, np.ones(window) / window, mode="valid")
        ax2.plot(range(len(rolling)), rolling, color="blue", linewidth=1.5,
                 label=f"Episode SR (window={window})")
        ax2.set_title("Episode-Level Position Success Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Success Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        _save(fig2, "plot_episode_sr.png")

    # ── Episode reward histogram ──────────────────────────────────────────
    if episodes:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        rews = [e["rew"] for e in episodes]
        bins = np.linspace(min(rews), max(rews), 80)
        ax3.hist(rews, bins=bins, color="blue", alpha=0.7, edgecolor="white")
        ax3.axvline(np.mean(rews), color="red", linewidth=1.5, linestyle="--",
                    label=f"Mean = {np.mean(rews):+.2f}")
        ax3.axvline(-10, color="orange", linewidth=1, linestyle=":", alpha=0.7, label="-10 penalty")
        ax3.set_title(f"Episode Reward Distribution ({len(episodes)} episodes)")
        ax3.set_xlabel("Episode Reward")
        ax3.set_ylabel("Count")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        _save(fig3, "plot_reward_histogram.png")

    # ── Final object positions scatter ────────────────────────────────────
    if episodes and len(episodes) > 1000:
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        sample = episodes[-5000:] if len(episodes) > 5000 else episodes
        fx = [e["final_x"] for e in sample]
        fy = [e["final_y"] for e in sample]
        gx = [e["goal_x"] for e in sample]
        gy = [e["goal_y"] for e in sample]
        moved = sum(1 for e in sample if abs(e["final_x"]) > 0.04 or abs(e["final_y"] - 0.5) > 0.04)
        ax4.scatter(fx, fy, s=1, alpha=0.3, color="red", label=f"Final obj pos ({moved}/{len(sample)} moved)")
        ax4.scatter(gx, gy, s=1, alpha=0.1, color="orange", label="Goal pos")
        # Workspace rectangle
        ws_x = [-0.50, 0.50, 0.50, -0.50, -0.50]
        ws_y = [0.25, 0.25, 0.70, 0.70, 0.25]
        ax4.plot(ws_x, ws_y, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, label="WS bounds")
        ax4.set_xlim(-0.6, 0.6)
        ax4.set_ylim(0.15, 0.75)
        ax4.set_aspect("equal")
        ax4.set_title(f"Object Final Positions (last {len(sample)} episodes)")
        ax4.set_xlabel("X (m)")
        ax4.set_ylabel("Y (m)")
        ax4.legend(fontsize=8, markerscale=5)
        ax4.grid(True, alpha=0.3)
        _save(fig4, "plot_final_positions.png")


def print_summary(data: dict):
    iters = data["iters"]
    episodes = data["episodes"]
    pushes = data["pushes"]

    print(f"\n{'='*60}")
    print("PUSH-PPO TRAINING REPORT")
    print(f"{'='*60}")

    if iters:
        print(f"\n── Iterations ──")
        print(f"  Total:       {len(iters)}")
        print(f"  Range:       {iters[0]['iter']} → {iters[-1]['iter']}")
        last = iters[-1]
        print(f"  Last Rew:    {_safe_get(last, 'Rew'):+.3f}  (EMA: {_safe_get(last, 'EMA'):+.3f})")
        print(f"  Last SR:     {_safe_get(last, 'SR'):.4f}  RotSR: {_safe_get(last, 'RotSR'):.4f}")
        print(f"  Last PosErr: {_safe_get(last, 'PosErr'):.4f}m  RotErr: {_safe_get(last, 'RotErr'):.4f}rad")
        print(f"  Last IK fail:{_safe_get(last, 'IK_fail'):.4f}")
        print(f"  Best SR:     {_safe_get(last, 'BestSR'):.4f}")

        early = iters[0]
        print(f"\n  Change (iter {early['iter']} → {last['iter']}):")
        print(f"    SR:        {_safe_get(early,'SR'):.4f} → {_safe_get(last,'SR'):.4f}  (Δ{_safe_get(last,'SR') - _safe_get(early,'SR'):+.4f})")
        print(f"    PosErr:    {_safe_get(early,'PosErr'):.4f} → {_safe_get(last,'PosErr'):.4f}m")
        print(f"    RotErr:    {_safe_get(early,'RotErr'):.4f} → {_safe_get(last,'RotErr'):.4f}rad")
        print(f"    Rew:       {_safe_get(early,'Rew'):+.3f} → {_safe_get(last,'Rew'):+.3f}")

    if episodes:
        print(f"\n── Episodes ──")
        n = len(episodes)
        n_success = sum(1 for e in episodes if e["success"])
        print(f"  Total:       {n}")
        print(f"  Pos succes:  {n_success} ({100*n_success/n:.2f}%)")
        rews = [e["rew"] for e in episodes]
        print(f"  Mean rew:    {np.mean(rews):+.3f}")
        print(f"  Rew std:     {np.std(rews):.3f}")
        print(f"  Min rew:     {min(rews):+.3f}")
        print(f"  Max rew:     {max(rews):+.3f}")
        # −10 penalty episodes
        neg10 = sum(1 for e in episodes if e["rew"] <= -10.0)
        print(f"  −10 penalty: {neg10} ({100*neg10/n:.1f}%)")
        # Object movement
        moved = sum(1 for e in episodes if abs(e["final_x"]) > 0.04 or abs(e["final_y"] - 0.5) > 0.04 or abs(e["final_z"]) > 0.1)
        print(f"  Obj moved:   {moved} ({100*moved/n:.1f}%)")
        print(f"  Mean pos err:{np.mean([e['err_pos'] for e in episodes]):.4f}m")
        print(f"  Mean rot err:{np.mean([e['err_rot'] for e in episodes]):.4f}rad")
        print(f"  Mean pushes: {np.mean([e['pushes'] for e in episodes]):.1f}")

    if pushes:
        print(f"\n── Push Steps ──")
        print(f"  Total:       {len(pushes)}")
        print(f"  Mean rew:    {np.mean([p['rew'] for p in pushes]):+.3f}")
        print(f"  Mean at_goal:{np.mean([p['at_goal'] for p in pushes])/pushes[0]['n_envs']*100:.1f}%")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze push-PPO log output.")
    parser.add_argument(
        "--log", type=str, nargs="+", required=True,
        help="One or more *.out / *.log files to parse.",
    )
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory (default: parent dir of first log file).",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.log]
    out_dir = Path(args.out_dir) if args.out_dir else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Parsing {len(paths)} log file(s)...")
    data = merge_paths(paths)
    print(f"[INFO] Iter records: {len(data['iters'])},  Episodes: {len(data['episodes'])},  Pushes: {len(data['pushes'])}")

    write_csv(data, out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(data, plots_dir)
    print_summary(data)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
