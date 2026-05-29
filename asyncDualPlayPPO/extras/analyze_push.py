#!/usr/bin/env python3
"""
Analyze SLURM log output from train_push.py (Push-PPO baseline).

Scans a log directory for slurm-*.out files, traces job chains across
preemptions, stitches together contiguous training runs, writes CSVs/TXTs,
and generates Push-PPO training metric plots.

Parses [Iter N] compact summary lines and [Episode] / [Push N] per-event lines.

Usage:
  python asyncDualPlayPPO/extras/analyze_push.py --log-dir logs/push_ppo
  python asyncDualPlayPPO/extras/analyze_push.py --log-dir logs/push_ppo --merge-chains --separate-plots
"""

import re
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Patterns ────────────────────────────────────────────────────────────────
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
GLOBAL_START_RE = re.compile(r"Global iteration start:\s*(\d+)")
JOB_ID_RE = re.compile(r"slurm-(\d+)(?:-(.*?))?\.(?:out|txt)")

PUSH_ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+Loss=([-+\d.]+)[^\|]*\|\s+Val=([-+\d.]+)\s*\|\s+"
    r"Rew=([-+\d.]+)\s+\(EMA\s+([-+\d.]+)\)\s*\|\s+"
    r"PosErr=([-+\d.]+)\s*\|\s+RotErr=([-+\d.]+)\s*\|\s+"
    r"SR=([-+\d.]+)\s*\|\s+RotSR=([-+\d.]+)\s*\|\s+"
    r"IK_fail=([-+\d.]+)\s*\|\s+"
    r"AvgPushes=([^\s|]+)\s*\|\s+Epi=(\d+)\s*\|\s+"
    r"BestSR=([-+\d.]+)"
)

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


def parse_logs(log_dir: Path) -> dict:
    """Parse all slurm log files in log_dir (recursively). Returns {job_id: job_data}."""
    jobs = {}
    for f in list(log_dir.rglob("slurm-*-*.out")) + list(log_dir.rglob("slurm-*-*.txt")):
        if "chain_" in str(f.parent):
            continue
        m = JOB_ID_RE.match(f.name)
        if not m:
            continue
        job_id = int(m.group(1))
        suffix = m.group(2) if m.group(2) else "default"
        text = f.read_text(errors="replace")

        resume_iter = None
        gm = GLOBAL_START_RE.search(text)
        if gm:
            resume_iter = int(gm.group(1))
        else:
            rm = RESUME_RE.search(text)
            if rm:
                resume_iter = int(rm.group(1))

        chain_next = None
        cm = CHAIN_RE.search(text)
        if cm:
            chain_next = int(cm.group(1))

        push_updates = []
        for pm in PUSH_ITER_RE.finditer(text):
            push_updates.append({
                "local_iter": int(pm.group(1)),
                "loss": float(pm.group(2)),
                "val": float(pm.group(3)),
                "rew": float(pm.group(4)),
                "rew_ema": float(pm.group(5)),
                "pos_err": float(pm.group(6)),
                "rot_err": float(pm.group(7)),
                "sr": float(pm.group(8)),
                "rot_sr": float(pm.group(9)),
                "ik_fail_rate": float(pm.group(10)),
                "avg_pushes": pm.group(11),
                "episodes": int(pm.group(12)),
                "best_sr": float(pm.group(13)),
            })
        push_updates.sort(key=lambda x: x["local_iter"])

        jobs[job_id] = {
            "path": f,
            "resume_iter": resume_iter,
            "chain_next": chain_next,
            "suffix": suffix,
            "push": push_updates,
        }
    return jobs


def trace_chains(jobs: dict) -> list[list[int]]:
    """Find root jobs and trace forward chains. Link broken chains by suffix."""
    all_ids = set(jobs.keys())
    pointed_to = {v["chain_next"] for v in jobs.values() if v["chain_next"] is not None}
    roots = sorted(all_ids - pointed_to)

    explicit_chains = []
    for root in roots:
        chain = []
        jid = root
        visited = set()
        while jid is not None and jid not in visited:
            visited.add(jid)
            if jid in jobs:
                chain.append(jid)
                jid = jobs[jid]["chain_next"]
            else:
                break
        if chain:
            explicit_chains.append(chain)

    explicit_chains.sort(key=lambda c: c[0])

    final_chains = []
    chains_by_suffix = {}

    for chain in explicit_chains:
        first_job = jobs[chain[0]]
        suffix = first_job["suffix"]
        ri = first_job["resume_iter"] or 0

        if suffix not in chains_by_suffix:
            chains_by_suffix[suffix] = []

        if ri > 0 and chains_by_suffix[suffix]:
            best_prev_chain = None
            best_max_iter = -1
            for prev_c in chains_by_suffix[suffix]:
                c_max = -1
                for jid in prev_c:
                    if jobs[jid]["push"]:
                        c_max = max(c_max, jobs[jid]["push"][-1]["local_iter"])
                if c_max > best_max_iter:
                    best_max_iter = c_max
                    best_prev_chain = prev_c
            if best_prev_chain is not None:
                best_prev_chain.extend(chain)
            else:
                chains_by_suffix[suffix][-1].extend(chain)
        else:
            chains_by_suffix[suffix].append(chain)
            final_chains.append(chain)

    return final_chains


def merge_all_chains(chains: list[list[int]]) -> list[list[int]]:
    """Collapse all chains into a single chain ordered by job ID."""
    flat = sorted(jid for chain in chains for jid in chain)
    return [flat]


def assign_global_iters(chains: list[list[int]], jobs: dict) -> list[dict]:
    """Assign global iteration numbers. Deduplicates per-chain."""
    push_records = []
    for chain_idx, chain in enumerate(chains):
        seen: set[int] = set()
        for job_id in chain:
            job = jobs[job_id]
            for upd in job["push"]:
                g = upd["local_iter"]
                if g in seen:
                    continue
                seen.add(g)
                push_records.append({
                    "chain": chain_idx,
                    "job_id": job_id,
                    "global_iter": g,
                    **{k: v for k, v in upd.items() if k != "local_iter"},
                })
    push_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    return push_records


# ── CSV/TXT output ──────────────────────────────────────────────────────────

def write_csv(push_records: list[dict], out_dir: Path):
    """Write combined push training CSV."""
    if not push_records:
        return
    out_path = out_dir / "push_training.csv"
    fieldnames = [
        "chain", "job_id", "global_iter",
        "loss", "val", "rew", "rew_ema",
        "pos_err", "rot_err", "sr", "rot_sr",
        "ik_fail_rate", "avg_pushes", "episodes", "best_sr",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in push_records:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[INFO] Wrote {out_path} ({len(push_records)} rows)")


def write_raw_csv(chain_idx: int, chain: list[int], jobs: dict, out_dir: Path):
    """Write all raw parsed local-iter push records."""
    out_path = out_dir / "raw_parsed.csv"
    fields = [
        "chain", "job_id", "local_iter",
        "loss", "val", "rew", "rew_ema",
        "pos_err", "rot_err", "sr", "rot_sr",
        "ik_fail_rate", "avg_pushes", "episodes", "best_sr",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for job_id in chain:
            job = jobs[job_id]
            for upd in job["push"]:
                w.writerow({
                    "chain": chain_idx,
                    "job_id": job_id,
                    "local_iter": upd["local_iter"],
                    "loss": upd["loss"], "val": upd["val"],
                    "rew": upd["rew"], "rew_ema": upd["rew_ema"],
                    "pos_err": upd["pos_err"], "rot_err": upd["rot_err"],
                    "sr": upd["sr"], "rot_sr": upd["rot_sr"],
                    "ik_fail_rate": upd["ik_fail_rate"],
                    "avg_pushes": upd["avg_pushes"],
                    "episodes": upd["episodes"],
                    "best_sr": upd["best_sr"],
                })
    print(f"[INFO] Wrote {out_path}")


def write_raw_logs(chain: list[int], jobs: dict, out_dir: Path):
    """Concatenate raw slurm .out files for all jobs in the chain."""
    out_path = out_dir / "raw_logs.txt"
    with open(out_path, "w") as fout:
        for job_id in chain:
            job_path = jobs[job_id]["path"]
            fout.write(f"{'='*72}\n")
            fout.write(f" Job {job_id}  ({job_path.name})\n")
            fout.write(f"{'='*72}\n")
            fout.write(job_path.read_text(errors="replace"))
            fout.write("\n")
    print(f"[INFO] Wrote {out_path}")


def write_summary_txt(chain_idx: int, chain: list[int], push_records: list[dict],
                      out_dir: Path):
    """Write human-readable per-chain summary."""
    out_path = out_dir / "training_updates.txt"
    with open(out_path, "w") as f:
        f.write(f"=== PUSH-PPO BASELINE SUMMARY (Chain {chain_idx}) ===\n\n")
        f.write(f"Jobs in chain: {' → '.join(str(j) for j in chain)}\n\n")
        for pr in push_records:
            f.write(
                f"  Iter {pr['global_iter']:5d} | "
                f"Loss={pr['loss']:+.4f}  Val={pr['val']:.4f}  "
                f"Rew={pr['rew']:+.4f} (EMA {pr['rew_ema']:+.4f})  "
                f"SR={pr['sr']:.4f}  RotSR={pr['rot_sr']:.4f}  "
                f"PosErr={pr['pos_err']:.4f}  RotErr={pr['rot_err']:.4f}  "
                f"BestSR={pr['best_sr']:.4f}\n"
            )
    print(f"[INFO] Wrote {out_path}")


# ── Plotting ─────────────────────────────────────────────────────────────────

def smooth(arr, window=7):
    """Simple moving average."""
    arr = list(arr)
    if len(arr) < window:
        return np.array(arr)
    y = np.array(arr)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _plot_push_metrics(push_records: list[dict], out_dir: Path,
                       title_suffix: str = "", log_paths: list[Path] = None,
                       separate: bool = False):
    """Render Push-PPO baseline plots from iter records and re-parsed episode data."""
    push_colors = ["tab:green", "mediumseagreen", "darkgreen", "lightgreen"]

    all_chain_indices = sorted(set(r["chain"] for r in push_records))
    p_by_chain = [[r for r in push_records if r["chain"] == c] for c in all_chain_indices]
    p_labels = [f"Push C{c}" for c in all_chain_indices]

    def _draw_p(ax, key, ylabel, title):
        for records, label, color in zip(p_by_chain, p_labels, push_colors):
            pts = [(r["global_iter"], r[key]) for r in records if r.get(key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, smooth(list(ys)), color=color, label=label, linewidth=1.5)
        ax.set_title(title + title_suffix)
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    if separate:
        for key, ylabel, title, fname in [
            ("loss", "Surrogate Loss", "Push-PPO — Policy Loss", "plot_loss.png"),
            ("val", "Value Loss", "Push-PPO — Value Loss", "plot_value_loss.png"),
            ("rew", "Mean Reward", "Push-PPO — Mean Reward", "plot_reward.png"),
            ("rew_ema", "EMA Reward", "Push-PPO — EMA Reward", "plot_reward_ema.png"),
            ("sr", "Success Rate", "Push-PPO — Success Rate", "plot_sr.png"),
            ("rot_sr", "Rotation SR", "Push-PPO — Rotation SR", "plot_rot_sr.png"),
            ("pos_err", "Position Error (m)", "Push-PPO — Position Error", "plot_pos_err.png"),
            ("rot_err", "Rotation Error (rad)", "Push-PPO — Rotation Error", "plot_rot_err.png"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_p(ax, key, ylabel, title)
            plt.tight_layout()
            _save(fig, fname)

        has_ik = any(r.get("ik_fail_rate") is not None for r in push_records)
        if has_ik:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_p(ax, "ik_fail_rate", "IK Fail Rate", "Push-PPO — IK Fail Rate")
            ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            _save(fig, "plot_ik_fail.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _draw_p(ax, "best_sr", "Best SR", "Push-PPO — Best Success Rate")
        plt.tight_layout()
        _save(fig, "plot_best_sr.png")

        has_epi = any(r.get("episodes") is not None for r in push_records)
        if has_epi:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_p(ax, "episodes", "Episodes", "Push-PPO — Episodes per Iteration")
            plt.tight_layout()
            _save(fig, "plot_episodes.png")
        return

    # Combined overview grid
    from matplotlib.gridspec import GridSpec

    n_rows = 4
    has_ik = any(r.get("ik_fail_rate") is not None for r in push_records)
    has_epi = any(r.get("episodes") is not None for r in push_records)
    if has_ik:
        n_rows += 1
    if has_epi:
        n_rows += 1

    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.32)
    _row = 0

    ax_loss = fig.add_subplot(gs[_row, 0])
    ax_val = fig.add_subplot(gs[_row, 1])
    ax_rew = fig.add_subplot(gs[_row, 2])
    _row += 1

    ax_sr = fig.add_subplot(gs[_row, 0])
    ax_rotsr = fig.add_subplot(gs[_row, 1])
    ax_rew_ema = fig.add_subplot(gs[_row, 2])
    _row += 1

    ax_poserr = fig.add_subplot(gs[_row, 0])
    ax_roterr = fig.add_subplot(gs[_row, 1])
    ax_best = fig.add_subplot(gs[_row, 2])
    _row += 1

    _draw_p(ax_loss, "loss", "Surrogate Loss", "Push-PPO — Policy Loss")
    _draw_p(ax_val, "val", "Value Loss", "Push-PPO — Value Loss")
    _draw_p(ax_rew, "rew", "Mean Reward", "Push-PPO — Mean Reward")
    _draw_p(ax_sr, "sr", "Success Rate", "Push-PPO — Success Rate")
    _draw_p(ax_rotsr, "rot_sr", "Rotation SR", "Push-PPO — Rotation Success Rate")
    _draw_p(ax_rew_ema, "rew_ema", "EMA Reward", "Push-PPO — EMA Reward")
    _draw_p(ax_poserr, "pos_err", "Position Error (m)", "Push-PPO — Position Error")
    _draw_p(ax_roterr, "rot_err", "Rotation Error (rad)", "Push-PPO — Rotation Error")
    _draw_p(ax_best, "best_sr", "Best SR", "Push-PPO — Best Success Rate")

    if has_ik or has_epi:
        ax_extra1 = fig.add_subplot(gs[_row, 0])
        ax_extra2 = fig.add_subplot(gs[_row, 1])
        ax_extra3 = fig.add_subplot(gs[_row, 2])
        _row += 1

        if has_ik:
            _draw_p(ax_extra1, "ik_fail_rate", "IK Fail Rate", "Push-PPO — IK Fail Rate")
            ax_extra1.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
            ax_extra1.legend(fontsize=8)
        else:
            ax_extra1.axis("off")

        if has_epi:
            _draw_p(ax_extra2, "episodes", "Episodes", "Push-PPO — Episodes per Iteration")
        else:
            ax_extra2.axis("off")
        ax_extra3.axis("off")

    fig.suptitle(f"Push-PPO Training Overview{title_suffix}", fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout()
    _save(fig, "plot_overview.png")

    # ── Episode-level plots (re-parse slurm logs for [Episode] lines) ────
    if log_paths:
        episodes = []
        for lp in log_paths:
            if lp.exists():
                for m in EPISODE_RE.finditer(lp.read_text(errors="replace")):
                    episodes.append({
                        "pushes": int(m.group(1)),
                        "success": m.group(2) == "SUCCESS",
                        "rew": float(m.group(3)),
                        "goal_x": float(m.group(4)), "goal_y": float(m.group(5)),
                        "final_x": float(m.group(10)), "final_y": float(m.group(11)),
                        "err_pos": float(m.group(16)),
                        "err_rot": float(m.group(17)),
                    })
        if episodes:
            # Episode rolling SR
            fig2, ax2 = plt.subplots(figsize=(14, 4))
            w = min(200, len(episodes) // 10) if len(episodes) >= 10 else min(200, len(episodes))
            w = max(1, w)
            srs = [1.0 if e["success"] else 0.0 for e in episodes]
            if len(srs) >= w:
                rolling = np.convolve(srs, np.ones(w) / w, mode="valid")
                ax2.plot(range(len(rolling)), rolling, color="blue", linewidth=1.5,
                         label=f"Episode SR (window={w})")
            ax2.set_title(f"Push-PPO — Episode-Level Success Rate{title_suffix}")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Success Rate")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            _save(fig2, "plot_episode_sr.png")

            # Reward histogram
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            rews = [e["rew"] for e in episodes]
            ax3.hist(rews, bins=80, color="blue", alpha=0.7, edgecolor="white")
            ax3.axvline(np.mean(rews), color="red", linewidth=1.5, linestyle="--",
                        label=f"Mean = {np.mean(rews):+.2f}")
            ax3.set_title(f"Push-PPO — Episode Reward Distribution{title_suffix} ({len(episodes)} eps)")
            ax3.set_xlabel("Episode Reward")
            ax3.set_ylabel("Count")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            _save(fig3, "plot_reward_histogram.png")

            # Final positions scatter
            fig4, ax4 = plt.subplots(figsize=(8, 8))
            sample = episodes[-10000:] if len(episodes) > 10000 else episodes
            fx = [e["final_x"] for e in sample]
            fy = [e["final_y"] for e in sample]
            gx = [e["goal_x"] for e in sample]
            gy = [e["goal_y"] for e in sample]
            ax4.scatter(fx, fy, s=1, alpha=0.3, color="red", label="Final obj pos")
            ax4.scatter(gx, gy, s=1, alpha=0.15, color="green", label="Goal pos")
            ax4.set_xlim(-0.6, 0.6)
            ax4.set_ylim(0.15, 0.75)
            ax4.set_aspect("equal")
            ax4.set_title(f"Push-PPO — Object Final Positions{title_suffix} (last {len(sample)} eps)")
            ax4.set_xlabel("X (m)")
            ax4.set_ylabel("Y (m)")
            ax4.legend(fontsize=8, markerscale=5)
            ax4.grid(True, alpha=0.3)
            _save(fig4, "plot_final_positions.png")


def print_summary(push_records: list[dict]):
    """Print summary statistics to stdout."""
    if not push_records:
        print("[WARN] No push records to summarize.")
        return

    print(f"\n{'='*60}")
    print("PUSH-PPO TRAINING REPORT")
    print(f"{'='*60}")

    print(f"\n── Iterations ──")
    print(f"  Total:       {len(push_records)}")
    print(f"  Range:       {push_records[0]['global_iter']} → {push_records[-1]['global_iter']}")

    last = push_records[-1]
    print(f"  Last Rew:    {last['rew']:+.3f}  (EMA: {last['rew_ema']:+.3f})")
    print(f"  Last SR:     {last['sr']:.4f}  RotSR: {last['rot_sr']:.4f}")
    print(f"  Last PosErr: {last['pos_err']:.4f}m  RotErr: {last['rot_err']:.4f}rad")
    print(f"  Last IK fail:{last['ik_fail_rate']:.4f}")
    print(f"  Best SR:     {last['best_sr']:.4f}")

    first = push_records[0]
    print(f"\n  Change (iter {first['global_iter']} → {last['global_iter']}):")
    print(f"    SR:        {first['sr']:.4f} → {last['sr']:.4f}  (Δ{last['sr'] - first['sr']:+.4f})")
    print(f"    PosErr:    {first['pos_err']:.4f} → {last['pos_err']:.4f}m")
    print(f"    RotErr:    {first['rot_err']:.4f} → {last['rot_err']:.4f}rad")
    print(f"    Rew:       {first['rew']:+.3f} → {last['rew']:+.3f}")
    print(f"    BestSR:    {first['best_sr']:.4f} → {last['best_sr']:.4f}")

    rews = [r["rew"] for r in push_records]
    srs = [r["sr"] for r in push_records]
    pos_errs = [r["pos_err"] for r in push_records]
    print(f"\n── Overall ──")
    print(f"  Mean Rew:    {np.mean(rews):+.4f}")
    print(f"  Mean SR:     {np.mean(srs):.4f}")
    print(f"  Max SR:      {max(srs):.4f}")
    print(f"  Mean PosErr: {np.mean(pos_errs):.4f}m")
    print(f"  Mean RotErr: {np.mean([r['rot_err'] for r in push_records]):.4f}rad")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and stitch Push-PPO baseline training logs across SLURM job chains."
    )
    parser.add_argument(
        "--log-dir", type=str, required=True,
        help="Directory containing slurm log files (slurm-*-*.out / slurm-*-*.txt).",
    )
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory (defaults to --log-dir).",
    )
    parser.add_argument(
        "--merge-chains", action="store_true", default=False,
        help="Collapse all discovered chains into one, ordered by job ID. "
             "Use when a job was killed before its EXIT trap ran and the "
             "successor appears as an unlinked root.",
    )
    parser.add_argument(
        "--separate-plots", action="store_true", default=False,
        help="Save one PNG per metric instead of a single combined overview PNG.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir

    if not log_dir.exists():
        parser.error(f"--log-dir does not exist: {log_dir}")

    print(f"[INFO] Scanning {log_dir} for slurm log files ...")
    jobs = parse_logs(log_dir)
    print(f"[INFO] Found {len(jobs)} job log file(s)")

    chains = trace_chains(jobs)
    print(f"[INFO] Found {len(chains)} chain(s):")
    for i, ch in enumerate(chains):
        print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    if args.merge_chains and len(chains) > 1:
        chains = merge_all_chains(chains)
        print(f"[INFO] --merge-chains: collapsed to {len(chains)} chain(s):")
        for i, ch in enumerate(chains):
            print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    push_records = assign_global_iters(chains, jobs)
    print(f"[INFO] Push iter records: {len(push_records)}")
    if push_records:
        print(f"[INFO] Global iter range: {push_records[0]['global_iter']} → {push_records[-1]['global_iter']}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each chain separately
    for i, ch in enumerate(chains):
        chain_dir = out_dir / f"chain_{i}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing Chain {i} in {chain_dir} ...")

        for job_id in ch:
            job_path = jobs[job_id]["path"]
            dest = chain_dir / job_path.name
            if not dest.exists():
                shutil.copy2(job_path, dest)

        p_c = [r for r in push_records if r["chain"] == i]

        write_raw_csv(i, ch, jobs, chain_dir)
        write_raw_logs(ch, jobs, chain_dir)
        write_summary_txt(i, ch, p_c, chain_dir)

        plots_dir = chain_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_push_metrics(
            p_c, plots_dir,
            title_suffix=f" (Chain {i})",
            log_paths=[jobs[jid]["path"] for jid in ch],
            separate=args.separate_plots,
        )

    # Combined CSV at top level
    write_csv(push_records, out_dir)
    print_summary(push_records)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
