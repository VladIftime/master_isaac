#!/usr/bin/env python3
"""
Parses all slurm log files in long_trainining/, traces the two job chains,
extracts training updates, writes a clean summary CSV, and plots metrics.
"""

import re
import os
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).parent / "long_trainining"
OUT_DIR = Path(__file__).parent / "long_trainining"

# --- Patterns ---
ALICE_RE = re.compile(
    r"\[Alice Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)"
)
BOB_RE = re.compile(
    r"\[Bob Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)\s*\|\s*ABC:\s*([-\d.]+)\s*\|\s*SR:\s*([-\d.]+)"
)
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
JOB_ID_RE = re.compile(r"slurm-(\d+)-high\.out")


def parse_logs():
    """Parse all log files and return per-job data."""
    jobs = {}
    for f in LOG_DIR.glob("slurm-*-high.out"):
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
        for am in ALICE_RE.finditer(text):
            alice_updates.append({
                "local_iter": int(am.group(1)),
                "loss": float(am.group(2)),
                "val": float(am.group(3)),
                "rew": float(am.group(4)),
            })

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


def write_csv(alice_records, bob_records):
    out_path = OUT_DIR / "training_updates.csv"
    fieldnames = ["agent", "chain", "job_id", "global_iter", "loss", "val", "rew", "abc", "sr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in alice_records:
            writer.writerow({"agent": "alice", "abc": "", "sr": "", **r})
        for r in bob_records:
            writer.writerow({"agent": "bob", **r})
    print(f"[INFO] Wrote {out_path}")
    return out_path


def smooth(vals, window=5):
    if len(vals) < window:
        return vals
    result = []
    for i in range(len(vals)):
        start = max(0, i - window // 2)
        end = min(len(vals), i + window // 2 + 1)
        result.append(sum(vals[start:end]) / (end - start))
    return result


def plot_metrics(alice_records, bob_records):
    n_chains = max(
        (max(r["chain"] for r in alice_records) if alice_records else 0),
        (max(r["chain"] for r in bob_records) if bob_records else 0),
    ) + 1

    chain_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # --- Figure 1: Loss ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Policy Loss (Alice & Bob)", fontsize=14)

    for c in range(n_chains):
        a_c = [r for r in alice_records if r["chain"] == c]
        b_c = [r for r in bob_records if r["chain"] == c]
        color = chain_colors[c % len(chain_colors)]
        label = f"Chain {c}"
        if a_c:
            xs = [r["global_iter"] for r in a_c]
            ys = smooth([r["loss"] for r in a_c])
            axes[0].plot(xs, ys, color=color, label=label, linewidth=1.5)
        if b_c:
            xs = [r["global_iter"] for r in b_c]
            ys = smooth([r["loss"] for r in b_c])
            axes[1].plot(xs, ys, color=color, label=label, linewidth=1.5)

    axes[0].set_title("Alice — Policy Loss")
    axes[0].set_xlabel("Global Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Bob — Policy Loss")
    axes[1].set_xlabel("Global Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    loss_path = OUT_DIR / "plot_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {loss_path}")

    # --- Figure 2: Value Loss ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Value Loss (Alice & Bob)", fontsize=14)

    for c in range(n_chains):
        a_c = [r for r in alice_records if r["chain"] == c]
        b_c = [r for r in bob_records if r["chain"] == c]
        color = chain_colors[c % len(chain_colors)]
        label = f"Chain {c}"
        if a_c:
            xs = [r["global_iter"] for r in a_c]
            ys = smooth([r["val"] for r in a_c])
            axes[0].plot(xs, ys, color=color, label=label, linewidth=1.5)
        if b_c:
            xs = [r["global_iter"] for r in b_c]
            ys = smooth([r["val"] for r in b_c])
            axes[1].plot(xs, ys, color=color, label=label, linewidth=1.5)

    axes[0].set_title("Alice — Value Loss")
    axes[0].set_xlabel("Global Iteration")
    axes[0].set_ylabel("Value Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Bob — Value Loss")
    axes[1].set_xlabel("Global Iteration")
    axes[1].set_ylabel("Value Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    val_path = OUT_DIR / "plot_value_loss.png"
    fig.savefig(val_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {val_path}")

    # --- Figure 3: Mean Reward ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mean Episode Reward (Alice & Bob)", fontsize=14)

    for c in range(n_chains):
        a_c = [r for r in alice_records if r["chain"] == c]
        b_c = [r for r in bob_records if r["chain"] == c]
        color = chain_colors[c % len(chain_colors)]
        label = f"Chain {c}"
        if a_c:
            xs = [r["global_iter"] for r in a_c]
            ys = smooth([r["rew"] for r in a_c])
            axes[0].plot(xs, ys, color=color, label=label, linewidth=1.5)
        if b_c:
            xs = [r["global_iter"] for r in b_c]
            ys = smooth([r["rew"] for r in b_c])
            axes[1].plot(xs, ys, color=color, label=label, linewidth=1.5)

    axes[0].set_title("Alice — Mean Reward")
    axes[0].set_xlabel("Global Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Bob — Mean Reward")
    axes[1].set_xlabel("Global Iteration")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    rew_path = OUT_DIR / "plot_reward.png"
    fig.savefig(rew_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {rew_path}")

    # --- Figure 4: Bob Success Rate & ABC ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bob — Success Rate & ABC Metric", fontsize=14)

    for c in range(n_chains):
        b_c = [r for r in bob_records if r["chain"] == c]
        color = chain_colors[c % len(chain_colors)]
        label = f"Chain {c}"
        if b_c:
            xs = [r["global_iter"] for r in b_c]
            sr = smooth([r["sr"] for r in b_c])
            abc = smooth([r["abc"] for r in b_c])
            axes[0].plot(xs, sr, color=color, label=label, linewidth=1.5)
            axes[1].plot(xs, abc, color=color, label=label, linewidth=1.5)

    axes[0].set_title("Bob — Success Rate (SR)")
    axes[0].set_xlabel("Global Iteration")
    axes[0].set_ylabel("SR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Bob — ABC Metric")
    axes[1].set_xlabel("Global Iteration")
    axes[1].set_ylabel("ABC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    sr_path = OUT_DIR / "plot_bob_sr_abc.png"
    fig.savefig(sr_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {sr_path}")

    # --- Figure 5: Combined overview ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Overview — All Metrics", fontsize=15)

    metrics_alice = [("loss", "Alice Loss"), ("val", "Alice Value Loss"), ("rew", "Alice Reward")]
    metrics_bob_sr = [("sr", "Bob Success Rate"), ("rew", "Bob Reward"), ("abc", "Bob ABC")]

    for col, (key, title) in enumerate(metrics_alice):
        ax = axes[0][col]
        for c in range(n_chains):
            a_c = [r for r in alice_records if r["chain"] == c]
            color = chain_colors[c % len(chain_colors)]
            if a_c:
                xs = [r["global_iter"] for r in a_c]
                ys = smooth([r[key] for r in a_c])
                ax.plot(xs, ys, color=color, label=f"Chain {c}", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Global Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for col, (key, title) in enumerate(metrics_bob_sr):
        ax = axes[1][col]
        for c in range(n_chains):
            b_c = [r for r in bob_records if r["chain"] == c]
            color = chain_colors[c % len(chain_colors)]
            if b_c:
                xs = [r["global_iter"] for r in b_c]
                ys = smooth([r[key] for r in b_c])
                ax.plot(xs, ys, color=color, label=f"Chain {c}", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Global Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    overview_path = OUT_DIR / "plot_overview.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {overview_path}")


def main():
    print(f"[INFO] Scanning {LOG_DIR} ...")
    jobs = parse_logs()
    print(f"[INFO] Found {len(jobs)} job log files")

    chains = trace_chains(jobs)
    print(f"[INFO] Found {len(chains)} chain(s):")
    for i, ch in enumerate(chains):
        print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    alice_records, bob_records = assign_global_iters(chains, jobs)
    print(f"[INFO] Alice updates: {len(alice_records)}, Bob updates: {len(bob_records)}")

    csv_path = write_csv(alice_records, bob_records)

    # Also write a human-readable summary
    summary_path = OUT_DIR / "training_updates.txt"
    with open(summary_path, "w") as f:
        f.write("=== TRAINING UPDATES SUMMARY ===\n\n")
        for i, ch in enumerate(chains):
            f.write(f"--- Chain {i} ({len(ch)} jobs) ---\n")
            a_c = [r for r in alice_records if r["chain"] == i]
            b_c = [r for r in bob_records if r["chain"] == i]
            for j, (ar, br) in enumerate(zip(a_c, b_c)):
                f.write(
                    f"  Iter {ar['global_iter']:4d} | "
                    f"[Alice] Loss={ar['loss']:+.4f}  Val={ar['val']:.4f}  Rew={ar['rew']:.4f}  || "
                    f"[Bob]   Loss={br['loss']:+.4f}  Val={br['val']:.4f}  Rew={br['rew']:.4f}  "
                    f"ABC={br['abc']:.4f}  SR={br['sr']:.4f}\n"
                )
            f.write("\n")
    print(f"[INFO] Wrote {summary_path}")

    plot_metrics(alice_records, bob_records)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
