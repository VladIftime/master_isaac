#!/usr/bin/env python3
"""
Parses slurm log files across one or more training-run directories,
traces job chains, stitches them together, deduplicates overlapping
iterations at run boundaries, writes clean CSVs/TXTs, and plots metrics.

Usage (single dir):
    python analyze_training.py --log-dir logs/train_140426

Usage (stitch prior run + current):
    python analyze_training.py \
        --log-dir logs/train_140426 \
        --prior-dirs logs/train_130426 \
        --out-dir logs/combined
"""

import re
import csv
import shutil
import bisect
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    r"(?:\s*\|\s*ABCCoef:\s*([-\d.]+))?"
)
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
# Explicit anchor emitted by train_high.slurm after the checkpoint detection block
GLOBAL_START_RE = re.compile(r"Global iteration start:\s*(\d+)")
JOB_ID_RE = re.compile(r"slurm-(\d+)(?:-(.*?))?\.out")
# New-format summary lines (replaces per-event [AliceEnd] lines)
ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+SR=([-\d.]+)\s*\|\s*Goals valid=(\d+) invalid=(\d+)\s*\|\s*Bob succ=(\d+) fail=(\d+)"
)
ALICE_DISP_RE = re.compile(
    r"\[AliceDisp\]\s+\d+/\d+ valid\s*\|\s*avg 3D=[-\d.]+m\s+avg XY=([-\d.]+)m\s+max XY=([-\d.]+)m"
)


def parse_logs(log_dir: Path) -> dict:
    """Parse all slurm log files in log_dir (recursively) and return per-job data dict."""
    jobs = {}
    for f in log_dir.rglob("slurm-*-*.out"):
        if "chain_" in str(f.parent):
            continue
        m = JOB_ID_RE.match(f.name)
        if not m:
            continue
        job_id = int(m.group(1))
        suffix = m.group(2) if m.group(2) else "default"
        text = f.read_text(errors="replace")

        resume_iter = None
        # Prefer the explicit anchor line emitted by train_high.slurm
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

        # New-format: parse [Iter N] summary lines (one per iteration boundary)
        # [Iter N] is printed after Update N-1, so its goals belong to Update N-1.
        iter_stats: dict[int, dict] = {}
        for im in ITER_RE.finditer(text):
            n = int(im.group(1))
            iter_stats[n] = {
                "sr": float(im.group(2)),
                "valid": int(im.group(3)),
                "invalid": int(im.group(4)),
                "bob_succ": int(im.group(5)),
                "bob_fail": int(im.group(6)),
                "pos": im.start(),
                "avg_xy": None,
                "max_xy": None,
            }

        # Attach [AliceDisp] displacement data to the preceding [Iter N]
        iter_positions = sorted((d["pos"], n) for n, d in iter_stats.items())
        for dm in ALICE_DISP_RE.finditer(text):
            dp = dm.start()
            n = None
            for ip, in_ in iter_positions:
                if ip < dp:
                    n = in_
                else:
                    break
            if n is not None and iter_stats[n]["avg_xy"] is None:
                iter_stats[n]["avg_xy"] = float(dm.group(1))
                iter_stats[n]["max_xy"] = float(dm.group(2))

        # Old-format fallback: per-event [AliceEnd] lines
        valid_pos = []
        invalid_pos = []
        if not iter_stats:
            valid_pos = [m.start() for m in re.finditer(r"\[AliceEnd\].*?outcome=valid", text)]
            invalid_pos = [m.start() for m in re.finditer(r"\[AliceEnd\].*?outcome=invalid", text)]

        alice_all_matches = []
        for am in ALICE_RE.finditer(text):
            alice_all_matches.append((am.start(), am.end(), am, True))
        for am in ALICE_NO_ENT_RE.finditer(text):
            alice_all_matches.append((am.start(), am.end(), am, False))
        alice_all_matches.sort(key=lambda x: x[0])

        alice_updates = []
        matched_iters = set()
        prev_update_end = 0

        for start, end, am, has_ent in alice_all_matches:
            it = int(am.group(2)) if has_ent else int(am.group(1))
            if it in matched_iters:
                continue
            matched_iters.add(it)

            if iter_stats:
                # [Iter it+1] holds the goals generated in the rollout that fed Update it
                ist = iter_stats.get(it + 1, {})
                v_count = ist.get("valid", 0)
                inv_count = ist.get("invalid", 0)
                avg_xy = ist.get("avg_xy")
                max_xy = ist.get("max_xy")
            else:
                v_count = bisect.bisect_left(valid_pos, end) - bisect.bisect_left(valid_pos, prev_update_end)
                inv_count = bisect.bisect_left(invalid_pos, end) - bisect.bisect_left(invalid_pos, prev_update_end)
                avg_xy = None
                max_xy = None

            if has_ent:
                alice_updates.append(
                    {
                        "local_iter": it,
                        "entropy_coef": float(am.group(1)),
                        "loss": float(am.group(3)),
                        "val": float(am.group(4)),
                        "rew": float(am.group(5)),
                        "valid_goals": v_count,
                        "invalid_goals": inv_count,
                        "avg_xy": avg_xy,
                        "max_xy": max_xy,
                    }
                )
            else:
                alice_updates.append(
                    {
                        "local_iter": it,
                        "entropy_coef": None,
                        "loss": float(am.group(2)),
                        "val": float(am.group(3)),
                        "rew": float(am.group(4)),
                        "valid_goals": v_count,
                        "invalid_goals": inv_count,
                        "avg_xy": avg_xy,
                        "max_xy": max_xy,
                    }
                )
            prev_update_end = end

        alice_updates.sort(key=lambda x: x["local_iter"])

        bob_updates = []
        for bm in BOB_RE.finditer(text):
            bob_updates.append(
                {
                    "local_iter": int(bm.group(1)),
                    "loss": float(bm.group(2)),
                    "val": float(bm.group(3)),
                    "rew": float(bm.group(4)),
                    "abc": float(bm.group(5)),
                    "sr": float(bm.group(6)),
                    "abc_coef": float(bm.group(7)) if bm.group(7) is not None else None,
                }
            )
            
        # Backward compatibility for old logs (scaling step rewards to episodic returns)
        if bob_updates:
            max_rew = max(u["rew"] for u in bob_updates)
            if max_rew < 0.1:
                # Old logs: mean step reward. Scale by 200 to approximate episodic return.
                for u in bob_updates:
                    u["rew"] *= 200.0

        jobs[job_id] = {
            "path": f,
            "resume_iter": resume_iter,
            "chain_next": chain_next,
            "suffix": suffix,
            "alice": alice_updates,
            "bob": bob_updates,
        }
    return jobs


def parse_all_dirs(log_dirs: list[Path]) -> dict:
    """Parse multiple directories and merge jobs into one dict."""
    all_jobs = {}
    for d in log_dirs:
        jobs = parse_logs(d)
        overlap = set(jobs.keys()) & set(all_jobs.keys())
        if overlap:
            print(f"[WARN] Job IDs {overlap} appear in multiple dirs — keeping first occurrence.")
        for jid, data in jobs.items():
            if jid not in all_jobs:
                all_jobs[jid] = data
    return all_jobs


def trace_chains(jobs: dict) -> list[list[int]]:
    """
    Find root jobs (not pointed to by any other job) and trace forward chains.
    Then, attempt to logically link broken chains that share the same suffix
    and resume from >0 iterations.
    Returns list of chains, each chain is an ordered list of job_ids.
    """
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

    # Sort chains by the ID of their first job
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
            # This chain resumes an existing experiment, find the best previous chain to attach to.
            # We look for the previous chain whose max global iter is closest to (but ideally <=) ri.
            best_prev_chain = None
            best_prev_chain_idx = -1
            best_max_iter = -1
            
            for idx, prev_c in enumerate(chains_by_suffix[suffix]):
                # Find max iter in prev_c
                c_max = -1
                for jid in prev_c:
                    if jobs[jid]["alice"]:
                        c_max = max(c_max, jobs[jid]["alice"][-1]["local_iter"])
                    if jobs[jid]["bob"]:
                        c_max = max(c_max, jobs[jid]["bob"][-1]["local_iter"])
                
                # We simply pick the one with the highest max_iter so far.
                if c_max > best_max_iter:
                    best_max_iter = c_max
                    best_prev_chain = prev_c
            
            if best_prev_chain is not None:
                best_prev_chain.extend(chain)
            else:
                # Fallback, just append to the last one
                chains_by_suffix[suffix][-1].extend(chain)
        else:
            chains_by_suffix[suffix].append(chain)
            final_chains.append(chain)

    return final_chains


def merge_all_chains(chains: list[list[int]]) -> list[list[int]]:
    """
    Collapse all chains into a single chain ordered by job ID.

    Use when a job was killed before its EXIT trap could run (so no
    'chained next job' line was printed), causing the next job to appear
    as an unlinked root.  Ordering by job ID preserves submission order.
    """
    flat = sorted(jid for chain in chains for jid in chain)
    return [flat]


def cross_check_job(job_id: int, job: dict):
    """
    Sanity-check a single job: compare the slurm-reported resume_iter against
    the first iteration numbers actually printed in the update logs.
    Since train.py now initialises both alice_updates and bob_updates to
    resume_iteration, the first logged iter should equal resume_iter exactly.
    Prints warnings for any mismatch so problems are visible without stopping
    the analysis.
    """
    ri = job["resume_iter"]
    if ri is None:
        ri = 0  # fresh start assumed when no anchor is present

    issues = []

    if job["alice"]:
        first_a = job["alice"][0]["local_iter"]
        if first_a != ri:
            issues.append(
                f"Alice first iter={first_a} but resume_iter={ri} "
                f"(delta={first_a - ri:+d})"
            )

    if job["bob"]:
        first_b = job["bob"][0]["local_iter"]
        if first_b != ri:
            issues.append(
                f"Bob first iter={first_b} but resume_iter={ri} "
                f"(delta={first_b - ri:+d})"
            )

    if issues:
        print(f"[WARN] Job {job_id} iteration mismatch:")
        for msg in issues:
            print(f"       {msg}")
    else:
        if job["alice"] or job["bob"]:
            print(f"[OK]   Job {job_id} iteration anchor matches resume_iter={ri}")


def assign_global_iters(
    chains: list[list[int]], jobs: dict
) -> tuple[list[dict], list[dict]]:
    """
    Assign global iteration numbers.

    Since train.py now keeps alice_updates and bob_updates both initialised to
    resume_iteration, every update line already contains the *global* iteration
    number.  We therefore use local_iter directly — no resume-offset arithmetic
    needed.  Deduplication via seen-sets handles any overlap at job boundaries
    (e.g. when a job is killed before checkpointing and its successor replays
    some iterations from the last saved checkpoint).
    """
    alice_records = []
    bob_records = []

    for chain_idx, chain in enumerate(chains):
        seen_alice_iters: set[int] = set()
        seen_bob_iters: set[int] = set()

        for job_id in chain:
            job = jobs[job_id]

            # Cross-check: slurm anchor vs. actual log numbers
            cross_check_job(job_id, job)

            for upd in job["alice"]:
                g = upd["local_iter"]   # already the global iteration number
                if g in seen_alice_iters:
                    continue
                seen_alice_iters.add(g)
                alice_records.append(
                    {
                        "chain": chain_idx,
                        "job_id": job_id,
                        "global_iter": g,
                        **{k: v for k, v in upd.items() if k != "local_iter"},
                    }
                )

            for upd in job["bob"]:
                g = upd["local_iter"]   # already the global iteration number
                if g in seen_bob_iters:
                    continue
                seen_bob_iters.add(g)
                bob_records.append(
                    {
                        "chain": chain_idx,
                        "job_id": job_id,
                        "global_iter": g,
                        **{k: v for k, v in upd.items() if k != "local_iter"},
                    }
                )

    alice_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    bob_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    return alice_records, bob_records


def write_csv(alice_records: list[dict], bob_records: list[dict], out_dir: Path):
    out_path = out_dir / "training_updates.csv"
    fieldnames = [
        "agent",
        "chain",
        "job_id",
        "global_iter",
        "loss",
        "val",
        "rew",
        "entropy_coef",
        "abc",
        "abc_coef",
        "sr",
        "valid_goals",
        "invalid_goals",
        "avg_xy",
        "max_xy",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in alice_records:
            writer.writerow({"agent": "alice", "abc": "", "abc_coef": "", "sr": "", **r})
        for r in bob_records:
            writer.writerow({"agent": "bob", "entropy_coef": "", "valid_goals": "", "invalid_goals": "", "avg_xy": "", "max_xy": "", **r})
    print(f"[INFO] Wrote {out_path}")
    return out_path


def write_raw_logs(chain: list[int], jobs: dict, out_dir: Path):
    """Concatenate the raw slurm .out files for all jobs in the chain."""
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


def write_raw_csv(chain_idx: int, chain: list[int], jobs: dict, out_dir: Path):
    """Write all raw parsed update records (local_iter) to raw_parsed.csv."""
    out_path = out_dir / "raw_parsed.csv"
    all_fields = [
        "agent",
        "chain",
        "job_id",
        "local_iter",
        "loss",
        "val",
        "rew",
        "entropy_coef",
        "abc",
        "abc_coef",
        "sr",
        "valid_goals",
        "invalid_goals",
        "avg_xy",
        "max_xy",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for job_id in chain:
            job = jobs[job_id]
            for upd in job["alice"]:
                writer.writerow(
                    {
                        "agent": "alice",
                        "chain": chain_idx,
                        "job_id": job_id,
                        "local_iter": upd["local_iter"],
                        "loss": upd["loss"],
                        "val": upd["val"],
                        "rew": upd["rew"],
                        "entropy_coef": upd.get("entropy_coef") or "",
                        "abc": "",
                        "sr": "",
                        "valid_goals": upd.get("valid_goals", ""),
                        "invalid_goals": upd.get("invalid_goals", ""),
                        "avg_xy": upd.get("avg_xy") if upd.get("avg_xy") is not None else "",
                        "max_xy": upd.get("max_xy") if upd.get("max_xy") is not None else "",
                    }
                )
            for upd in job["bob"]:
                writer.writerow(
                    {
                        "agent": "bob",
                        "chain": chain_idx,
                        "job_id": job_id,
                        "local_iter": upd["local_iter"],
                        "loss": upd["loss"],
                        "val": upd["val"],
                        "rew": upd["rew"],
                        "entropy_coef": "",
                        "abc": upd["abc"],
                        "abc_coef": upd["abc_coef"] if upd.get("abc_coef") is not None else "",
                        "sr": upd["sr"],
                        "valid_goals": "",
                        "invalid_goals": "",
                        "avg_xy": "",
                        "max_xy": "",
                    }
                )
    print(f"[INFO] Wrote {out_path}")


def smooth(vals: list, window: int = 5) -> list:
    if len(vals) < window:
        return vals
    result = []
    for i in range(len(vals)):
        start = max(0, i - window // 2)
        end = min(len(vals), i + window // 2 + 1)
        result.append(sum(vals[start:end]) / (end - start))
    return result


def plot_metrics(
    alice_records: list[dict],
    bob_records: list[dict],
    out_dir: Path,
    title_suffix: str = "",
):
    if not alice_records and not bob_records:
        print("[WARN] No records to plot.")
        return

    # Alice = blues, Bob = reds/oranges per chain
    alice_colors = ["tab:blue", "cornflowerblue", "navy", "steelblue"]
    bob_colors = ["tab:red", "tomato", "darkred", "salmon"]

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

    all_chain_indices = sorted(
        set(
            [r["chain"] for r in alice_records] + [r["chain"] for r in bob_records]
        )
    )
    a_by_chain = [[r for r in alice_records if r["chain"] == c] for c in all_chain_indices]
    b_by_chain = [[r for r in bob_records if r["chain"] == c] for c in all_chain_indices]
    a_labels = [f"Alice C{c}" for c in all_chain_indices]
    b_labels = [f"Bob C{c}" for c in all_chain_indices]

    # --- Figure: Loss ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "loss", "Loss", "Policy Loss — Alice & Bob")
    plot_single(ax, b_by_chain, b_labels, bob_colors, "loss", "Loss", "Policy Loss — Alice & Bob")
    plt.tight_layout()
    p = out_dir / "plot_loss.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Value Loss ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "val", "Value Loss", "Value Loss — Alice & Bob")
    plot_single(ax, b_by_chain, b_labels, bob_colors, "val", "Value Loss", "Value Loss — Alice & Bob")
    plt.tight_layout()
    p = out_dir / "plot_value_loss.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Reward ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "rew", "Reward", "Mean Episode Reward — Alice & Bob")
    plot_single(ax, b_by_chain, b_labels, bob_colors, "rew", "Reward", "Mean Episode Reward — Alice & Bob")
    plt.tight_layout()
    p = out_dir / "plot_reward.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Bob SR ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, b_by_chain, b_labels, bob_colors, "sr", "Success Rate", "Bob — Success Rate")
    plt.tight_layout()
    p = out_dir / "plot_bob_sr.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Bob ABC ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, b_by_chain, b_labels, bob_colors, "abc", "ABC", "Bob — ABC Metric")
    plt.tight_layout()
    p = out_dir / "plot_bob_abc.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Alice Entropy Coef ---
    a_ent_by_chain = [
        [r for r in recs if r.get("entropy_coef") is not None] for recs in a_by_chain
    ]
    if any(a_ent_by_chain):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_single(
            ax,
            a_ent_by_chain,
            a_labels,
            alice_colors,
            "entropy_coef",
            "Entropy Coef",
            "Alice — Entropy Coefficient",
        )
        plt.tight_layout()
        p = out_dir / "plot_alice_entropy.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    # --- Figure: Alice Valid Goals ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "valid_goals", "Goals", "Alice — Valid Goals")
    plt.tight_layout()
    p = out_dir / "plot_alice_valid_goals.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Alice Invalid Goals ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_single(ax, a_by_chain, a_labels, alice_colors, "invalid_goals", "Goals", "Alice — Invalid Goals")
    plt.tight_layout()
    p = out_dir / "plot_alice_invalid_goals.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # --- Figure: Alice Goal Displacement (avg_xy / max_xy) ---
    a_disp_by_chain = [
        [r for r in recs if r.get("avg_xy") is not None] for recs in a_by_chain
    ]
    if any(a_disp_by_chain):
        fig, axes_disp = plt.subplots(1, 2, figsize=(14, 5))
        plot_single(axes_disp[0], a_disp_by_chain, a_labels, alice_colors, "avg_xy", "Avg XY (m)", "Alice — Avg Goal Displacement XY")
        plot_single(axes_disp[1], a_disp_by_chain, a_labels, alice_colors, "max_xy", "Max XY (m)", "Alice — Max Goal Displacement XY")
        plt.tight_layout()
        p = out_dir / "plot_alice_goal_displacement.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")



def write_summary_txt(chain_idx: int, chain: list[int], a_c: list[dict], b_c: list[dict], out_dir: Path):
    """Write human-readable per-chain summary."""
    summary_path = out_dir / "training_updates.txt"
    # Build a lookup: global_iter → bob record
    bob_by_iter = {r["global_iter"]: r for r in b_c}
    with open(summary_path, "w") as f:
        f.write(f"=== TRAINING UPDATES SUMMARY (Chain {chain_idx}) ===\n\n")
        f.write(f"Jobs in chain: {' → '.join(str(j) for j in chain)}\n\n")
        for ar in a_c:
            g = ar["global_iter"]
            ent = ar.get("entropy_coef")
            ent_str = f"  Ent={ent:.4f}" if ent is not None else ""
            br = bob_by_iter.get(g)
            if br:
                bob_str = (
                    f"[Bob]   Loss={br['loss']:+.4f}  Val={br['val']:.4f}  "
                    f"Rew={br['rew']:.4f}  ABC={br['abc']:.4f}  SR={br['sr']:.4f}"
                )
            else:
                bob_str = "[Bob]   —"
            f.write(
                f"  Iter {g:5d} | "
                f"[Alice] Loss={ar['loss']:+.4f}  Val={ar['val']:.4f}  Rew={ar['rew']:.4f}  "
                f"Valid={ar.get('valid_goals', 0)}  Invalid={ar.get('invalid_goals', 0)}{ent_str}  || "
                f"{bob_str}\n"
            )
        # Any Bob-only iters (when Bob counter > Alice)
        alice_iters = {r["global_iter"] for r in a_c}
        for br in b_c:
            if br["global_iter"] not in alice_iters:
                g = br["global_iter"]
                f.write(
                    f"  Iter {g:5d} | [Alice] —  || "
                    f"[Bob]   Loss={br['loss']:+.4f}  Val={br['val']:.4f}  "
                    f"Rew={br['rew']:.4f}  ABC={br['abc']:.4f}  SR={br['sr']:.4f}\n"
                )
    print(f"[INFO] Wrote {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and stitch training logs across SLURM job chains."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).parent / "train_140426"),
        help="Primary directory containing slurm log files.",
    )
    parser.add_argument(
        "--prior-dirs",
        type=str,
        nargs="*",
        default=[],
        help="One or more earlier run directories to prepend (oldest first). "
             "E.g. --prior-dirs logs/train_130426 logs/train_100426",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (defaults to --log-dir).",
    )
    parser.add_argument(
        "--merge-chains",
        action="store_true",
        default=False,
        help="Collapse all discovered chains into one, ordered by job ID. "
             "Use when a job was killed before its EXIT trap ran (so no "
             "'chained next job' link was printed) and the successor job "
             "appears as an unlinked root.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    prior_dirs = [Path(d) for d in args.prior_dirs]
    out_dir = Path(args.out_dir) if args.out_dir else log_dir

    # Parse all directories — prior dirs first so chain-next links work
    all_dirs = prior_dirs + [log_dir]
    print(f"[INFO] Scanning {len(all_dirs)} director(ies):")
    for d in all_dirs:
        print(f"       {d}")

    jobs = parse_all_dirs(all_dirs)
    print(f"[INFO] Found {len(jobs)} job log files total")

    chains = trace_chains(jobs)
    print(f"[INFO] Found {len(chains)} chain(s) before merging:")
    for i, ch in enumerate(chains):
        print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    if args.merge_chains and len(chains) > 1:
        chains = merge_all_chains(chains)
        print(f"[INFO] --merge-chains: collapsed to {len(chains)} chain(s):")
        for i, ch in enumerate(chains):
            print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    alice_records, bob_records = assign_global_iters(chains, jobs)
    print(
        f"[INFO] Alice updates: {len(alice_records)}, Bob updates: {len(bob_records)}"
    )
    if alice_records:
        print(
            f"[INFO] Alice global iter range: {alice_records[0]['global_iter']} "
            f"→ {alice_records[-1]['global_iter']}"
        )
    if bob_records:
        print(
            f"[INFO] Bob   global iter range: {bob_records[0]['global_iter']} "
            f"→ {bob_records[-1]['global_iter']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each chain separately
    for i, ch in enumerate(chains):
        chain_dir = out_dir / f"chain_{i}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing Chain {i} in {chain_dir} ...")

        # Copy original slurm logs for this chain
        for job_id in ch:
            job_path = jobs[job_id]["path"]
            dest = chain_dir / job_path.name
            if not dest.exists():
                shutil.copy2(job_path, dest)

        a_c = [r for r in alice_records if r["chain"] == i]
        b_c = [r for r in bob_records if r["chain"] == i]

        write_raw_csv(i, ch, jobs, chain_dir)
        write_csv(a_c, b_c, chain_dir)
        write_summary_txt(i, ch, a_c, b_c, chain_dir)
        plot_metrics(a_c, b_c, chain_dir, title_suffix=f" (Chain {i})")

    write_csv(alice_records, bob_records, out_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
