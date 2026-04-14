#!/usr/bin/env python3
"""
Parses all slurm log files in long_trainining/, traces the two job chains,
extracts training updates, writes a clean summary CSV, and plots metrics.
"""

import re
import csv
import shutil
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = None
OUT_DIR = None

# --- Patterns ---
ALICE_RE = re.compile(
    r"\[Alice\] Entropy Coef:\s*([-\d.]+).*?\[Alice Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)",
    re.DOTALL,
)
ALICE_NO_ENT_RE = re.compile(
    r"\[Alice Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)"
)
BOB_RE = re.compile(
    r"\[Bob Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)\s*\|\s*ABC:\s*([-\d.]+)\s*\|\s*SR:\s*([-\d.]+)"
)
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
JOB_ID_RE = re.compile(r"slurm-(\d+)(?:-.*)?\.out")


def parse_logs():
    """Parse all log files and return per-job data."""
    jobs = {}
    for f in LOG_DIR.glob("slurm-*-*.out"):
        m = JOB_ID_RE.match(f.name)
        if not m:
            continue
        job_id = int(m.group(1))
        text = f.read_text(errors="replace")

        resume_iter = None
        rm = RESUME_RE.search(text)
        if rm:
            resume_iter = int(rm.group(1))

        chain_next = None
        cm = CHAIN_RE.search(text)
        if cm:
            chain_next = int(cm.group(1))

        alice_updates = []
        matched_iters = set()
        for am in ALICE_RE.finditer(text):
            it = int(am.group(2))
            matched_iters.add(it)
            alice_updates.append({
                "local_iter": it,
                "entropy_coef": float(am.group(1)),
                "loss": float(am.group(3)),
                "val": float(am.group(4)),
                "rew": float(am.group(5)),
            })
        # Fallback for lines without an entropy header
        for am in ALICE_NO_ENT_RE.finditer(text):
            it = int(am.group(1))
            if it not in matched_iters:
                alice_updates.append({
                    "local_iter": it,
                    "entropy_coef": None,
                    "loss": float(am.group(2)),
                    "val": float(am.group(3)),
                    "rew": float(am.group(4)),
                })
        alice_updates.sort(key=lambda x: x["local_iter"])

        bob_updates = []
        for bm in BOB_RE.finditer(text):
            bob_updates.append({
                "local_iter": int(bm.group(1)),
                "loss": float(bm.group(2)),
                "val": float(bm.group(3)),
                "rew": float(bm.group(4)),
                "abc": float(bm.group(5)),
                "sr": float(bm.group(6)),
            })

        jobs[job_id] = {
            "path": f,
            "resume_iter": resume_iter,
            "chain_next": chain_next,
            "alice": alice_updates,
            "bob": bob_updates,
        }
    return jobs


def trace_chains(jobs):
    """
    Find root jobs (not pointed to by any other job) and trace forward chains.
    Returns list of chains, each chain is an ordered list of job_ids.
    """
    all_ids = set(jobs.keys())
    pointed_to = {v["chain_next"] for v in jobs.values() if v["chain_next"] is not None}
    roots = sorted(all_ids - pointed_to)

    chains = []
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
            chains.append(chain)
    return chains


def assign_global_iters(chains, jobs):
    """
    Assign global iteration numbers.
    Uses resume_iter if available; otherwise increments sequentially from 0.
    Returns list of records with global_iter assigned.
    """
    alice_records = []
    bob_records = []

    for chain_idx, chain in enumerate(chains):
        global_iter = 0
        for job_id in chain:
            job = jobs[job_id]

            # If this job resumed from a checkpoint, use that as base
            if job["resume_iter"] is not None:
                global_iter = job["resume_iter"]

            for upd in job["alice"]:
                alice_records.append({
                    "chain": chain_idx,
                    "job_id": job_id,
                    "global_iter": global_iter + upd["local_iter"],
                    **{k: v for k, v in upd.items() if k != "local_iter"},
                })

            for upd in job["bob"]:
                bob_records.append({
                    "chain": chain_idx,
                    "job_id": job_id,
                    "global_iter": global_iter + upd["local_iter"],
                    **{k: v for k, v in upd.items() if k != "local_iter"},
                })

            # Advance by number of local updates done
            n = max(len(job["alice"]), len(job["bob"]), 1)
            global_iter += n

    # Sort by chain then iter
    alice_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    bob_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    return alice_records, bob_records


def write_csv(alice_records, bob_records, out_dir):
    out_path = out_dir / "training_updates.csv"
    fieldnames = ["agent", "chain", "job_id", "global_iter", "loss", "val", "rew", "entropy_coef", "abc", "sr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in alice_records:
            writer.writerow({"agent": "alice", "abc": "", "sr": "", **r})
        for r in bob_records:
            writer.writerow({"agent": "bob", "entropy_coef": "", **r})
    print(f"[INFO] Wrote {out_path}")
    return out_path


def write_raw_logs(chain, jobs, out_dir):
    """Concatenate the raw slurm .out files for all jobs in the chain into raw_logs.txt."""
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


def write_raw_csv(chain_idx, chain, jobs, out_dir):
    """Write all raw parsed update records (local_iter, before global assignment) to raw_parsed.csv."""
    out_path = out_dir / "raw_parsed.csv"
    alice_fields = ["agent", "chain", "job_id", "local_iter", "loss", "val", "rew", "entropy_coef"]
    bob_fields   = ["agent", "chain", "job_id", "local_iter", "loss", "val", "rew", "abc", "sr"]
    all_fields   = ["agent", "chain", "job_id", "local_iter", "loss", "val", "rew", "entropy_coef", "abc", "sr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for job_id in chain:
            job = jobs[job_id]
            for upd in job["alice"]:
                writer.writerow({
                    "agent": "alice",
                    "chain": chain_idx,
                    "job_id": job_id,
                    "local_iter": upd["local_iter"],
                    "loss": upd["loss"],
                    "val": upd["val"],
                    "rew": upd["rew"],
                    "entropy_coef": upd.get("entropy_coef", "") if upd.get("entropy_coef") is not None else "",
                    "abc": "",
                    "sr": "",
                })
            for upd in job["bob"]:
                writer.writerow({
                    "agent": "bob",
                    "chain": chain_idx,
                    "job_id": job_id,
                    "local_iter": upd["local_iter"],
                    "loss": upd["loss"],
                    "val": upd["val"],
                    "rew": upd["rew"],
                    "entropy_coef": "",
                    "abc": upd["abc"],
                    "sr": upd["sr"],
                })
    print(f"[INFO] Wrote {out_path}")


def smooth(vals, window=5):
    if len(vals) < window:
        return vals
    result = []
    for i in range(len(vals)):
        start = max(0, i - window // 2)
        end = min(len(vals), i + window // 2 + 1)
        result.append(sum(vals[start:end]) / (end - start))
    return result


def plot_metrics(alice_records, bob_records, out_dir, title_suffix=""):
    n_chains = max(
        (max(r["chain"] for r in alice_records) if alice_records else 0),
        (max(r["chain"] for r in bob_records) if bob_records else 0),
    ) + 1

    # Alice = blues, Bob = reds/oranges per chain
    alice_colors = ["tab:blue", "cornflowerblue", "navy", "steelblue"]
    bob_colors   = ["tab:red", "tomato", "darkred", "salmon"]

    def plot_single(ax, records_list, labels, colors, key, ylabel, title):
        for records, label, color in zip(records_list, labels, colors):
            if not records:
                continue
            xs = [r["global_iter"] for r in records]
            ys = smooth([r[key] for r in records])
            ax.plot(xs, ys, color=color, label=label, linewidth=1.5)
        ax.set_title(title + title_suffix)
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Build per-chain record lists
    all_chain_indices = sorted(list(set([r["chain"] for r in alice_records] + [r["chain"] for r in bob_records])))
    a_by_chain = [[r for r in alice_records if r["chain"] == c] for c in all_chain_indices]
    b_by_chain = [[r for r in bob_records  if r["chain"] == c] for c in all_chain_indices]
    a_labels = [f"Alice C{c}" for c in all_chain_indices]
    b_labels = [f"Bob C{c}"   for c in all_chain_indices]

    # --- Figure: Loss ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "loss", "Loss", "Policy Loss — Alice & Bob")
    plot_single(ax, b_by_chain, b_labels, bob_colors,   "loss", "Loss", "Policy Loss — Alice & Bob")
    plt.tight_layout()
    p = out_dir / "plot_loss.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Value Loss ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "val", "Value Loss", "Value Loss — Alice & Bob")
    plot_single(ax, b_by_chain, b_labels, bob_colors,   "val", "Value Loss", "Value Loss — Alice & Bob")
    plt.tight_layout()
    p = out_dir / "plot_value_loss.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Reward ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "rew", "Reward", "Mean Episode Reward — Alice & Bob")
    plot_single(ax, b_by_chain, b_labels, bob_colors,   "rew", "Reward", "Mean Episode Reward — Alice & Bob")
    plt.tight_layout()
    p = out_dir / "plot_reward.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Bob SR ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, b_by_chain, b_labels, bob_colors, "sr", "Success Rate", "Bob — Success Rate")
    plt.tight_layout()
    p = out_dir / "plot_bob_sr.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Bob ABC ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, b_by_chain, b_labels, bob_colors, "abc", "ABC", "Bob — ABC Metric")
    plt.tight_layout()
    p = out_dir / "plot_bob_abc.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Alice Entropy Coef ---
    a_ent_by_chain = [
        [r for r in recs if r.get("entropy_coef") is not None]
        for recs in a_by_chain
    ]
    if any(a_ent_by_chain):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_single(ax, a_ent_by_chain, a_labels, alice_colors,
                    "entropy_coef", "Entropy Coef", "Alice — Entropy Coefficient")
        plt.tight_layout()
        p = out_dir / "plot_alice_entropy.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"[INFO] Saved {p}")

    # --- Overview: all metrics in one figure ---
    specs = [
        (a_by_chain, a_labels, alice_colors, "loss",         "Loss",         "Policy Loss (Alice)"),
        (b_by_chain, b_labels, bob_colors,   "loss",         "Loss",         "Policy Loss (Bob)"),
        (a_by_chain, a_labels, alice_colors, "val",          "Value Loss",   "Value Loss (Alice)"),
        (b_by_chain, b_labels, bob_colors,   "val",          "Value Loss",   "Value Loss (Bob)"),
        (a_by_chain, a_labels, alice_colors, "rew",          "Reward",       "Reward (Alice)"),
        (b_by_chain, b_labels, bob_colors,   "rew",          "Reward",       "Reward (Bob)"),
        (b_by_chain, b_labels, bob_colors,   "sr",           "SR",           "Success Rate (Bob)"),
        (b_by_chain, b_labels, bob_colors,   "abc",          "ABC",          "ABC (Bob)"),
        (a_ent_by_chain, a_labels, alice_colors, "entropy_coef", "Ent Coef", "Entropy Coef (Alice)"),
    ]

    ncols = 3
    nrows = (len(specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()
    fig.suptitle("Training Overview" + title_suffix, fontsize=15)

    for i, (recs, labels, colors, key, ylabel, title) in enumerate(specs):
        plot_single(axes[i], recs, labels, colors, key, ylabel, title)

    for j in range(len(specs), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    p = out_dir / "plot_overview.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[INFO] Saved {p}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log-dir", type=str, default=str(Path(__file__).parent / "train_130426"),
                        help="Directory containing the slurm log files (default: logs/train_130426)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (defaults to log_dir)")
    args = parser.parse_args()

    global LOG_DIR, OUT_DIR
    LOG_DIR = Path(args.log_dir)
    OUT_DIR = Path(args.out_dir) if args.out_dir else LOG_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Scanning {LOG_DIR} ...")
    jobs = parse_logs()
    print(f"[INFO] Found {len(jobs)} job log files")

    chains = trace_chains(jobs)
    print(f"[INFO] Found {len(chains)} chain(s):")
    for i, ch in enumerate(chains):
        print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    alice_records, bob_records = assign_global_iters(chains, jobs)
    print(f"[INFO] Alice updates: {len(alice_records)}, Bob updates: {len(bob_records)}")

    # Process each chain separately
    for i, ch in enumerate(chains):
        chain_dir = OUT_DIR / f"chain_{i}"
        chain_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Processing Chain {i} in {chain_dir} ...")

        # Copy original slurm logs for this chain
        for job_id in ch:
            job_path = jobs[job_id]["path"]
            shutil.copy2(job_path, chain_dir / job_path.name)

        # Filter records for this chain
        a_c = [r for r in alice_records if r["chain"] == i]
        b_c = [r for r in bob_records if r["chain"] == i]

        # Write raw outputs for this chain
        write_raw_logs(ch, jobs, chain_dir)
        write_raw_csv(i, ch, jobs, chain_dir)

        # Write processed CSV for this chain
        write_csv(a_c, b_c, chain_dir)

        # Write human-readable summary for this chain
        summary_path = chain_dir / "training_updates.txt"
        with open(summary_path, "w") as f:
            f.write(f"=== TRAINING UPDATES SUMMARY (Chain {i}) ===\n\n")
            f.write(f"--- Chain {i} ({len(ch)} jobs) ---\n")
            for ar, br in zip(a_c, b_c):
                ent = ar.get("entropy_coef")
                ent_str = f"  Ent={ent:.4f}" if ent is not None else ""
                f.write(
                    f"  Iter {ar['global_iter']:4d} | "
                    f"[Alice] Loss={ar['loss']:+.4f}  Val={ar['val']:.4f}  Rew={ar['rew']:.4f}{ent_str}  || "
                    f"[Bob]   Loss={br['loss']:+.4f}  Val={br['val']:.4f}  Rew={br['rew']:.4f}  "
                    f"ABC={br['abc']:.4f}  SR={br['sr']:.4f}\n"
                )
        print(f"[INFO] Wrote {summary_path}")

        # Plot metrics for this chain
        plot_metrics(a_c, b_c, chain_dir, title_suffix=f" (Chain {i})")

    # Final overall overview in the root output dir (optional but good for comparison)
    print(f"[INFO] Generating overall comparison overview in {OUT_DIR} ...")
    plot_metrics(alice_records, bob_records, OUT_DIR, title_suffix=" (Overview)")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
