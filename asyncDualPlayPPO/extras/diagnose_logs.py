#!/usr/bin/env python3
"""
Diagnostic log analyser for asyncDualPlayPPO Slurm runs.

Answers: "why are there no training updates?"
Extracts all event types with counts, termination reasons, Alice/Bob
funnel rates, early-termination step distributions, and SLURM exit metadata.

Usage:
    python extras/diagnose_logs.py [LOG_DIR]

LOG_DIR defaults to logs/long_trainining_080426 relative to this file.
"""

import re
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
LOG_DIR = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else _HERE.parent / "logs" / "long_trainining_080426"
)
OUT_DIR = LOG_DIR


# ---------------------------------------------------------------------------
# Regex patterns  (match exactly what the code prints)
# ---------------------------------------------------------------------------

# [EarlyEnd] Env 1361 [TERMINATED] Alice step=72/200 goal=0/5 reason=robot_through_table
EARLYEND_RE = re.compile(
    r"\[EarlyEnd\] Env (\d+) \[(\w+(?:\s+\w+)?)\] (\w+) step=(\d+)/(\d+) goal=(\d+)/5 reason=(.+)"
)

# [Alice Reward] Env 27: Valid Goal (+1.0) | XY: 0.124m Z: 0.000m | Local target: ...
ALICE_REWARD_RE = re.compile(
    r"\[Alice Reward\] Env \d+: (.+?) \(([\+\-\d.]+)\) \| XY: ([\d.]+)m Z: ([\d.]+)m"
)

# [Phase] Env 27: Alice→Bob | Goal 1/5 | Alice completed 200/200 steps
ALICE_TO_BOB_RE = re.compile(r"\[Phase\] Env \d+: Alice→Bob \| Goal (\d+)/5")

# [Phase] Env 1369: Bob→Alice | Goal 1/5 SUCCESS | Bob used 2/400 steps
BOB_TO_ALICE_RE = re.compile(
    r"\[Phase\] Env \d+: Bob→Alice \| Goal (\d+)/5 (SUCCESS|FAILURE) \| Bob used (\d+)/(\d+) steps"
)

# [Phase] Env 490: Bob SUCCESS | Steps: 83/400 | Alice Outcome Reward: 5.0
BOB_DONE_RE = re.compile(
    r"\[Phase\] Env \d+: Bob (SUCCESS|FAILURE) \| Steps: (\d+)/(\d+)"
)

# [BobSuccess] Env 1369 step=2: target dist=0.0254 ...
BOB_SUCCESS_RE = re.compile(
    r"\[BobSuccess\] Env \d+ step=(\d+): target dist=([\d.]+).*cube dist=([\d.]+)"
)

# [Alice Update 5] Loss: -0.0123 | Val: 0.4321 | Rew: 0.1234
ALICE_UPDATE_RE = re.compile(
    r"\[Alice Update\s+(\d+)\] Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)"
)

# [Bob Update 5] Loss: ... | SR: 0.12
BOB_UPDATE_RE = re.compile(r"\[Bob Update\s+(\d+)\] Loss:\s*([-\d.]+).*SR:\s*([\d.]+)")

# [Bob Update N] SKIPPED
BOB_SKIP_RE = re.compile(r"\[Bob Update\s+(\d+)\] SKIPPED")

# [Termination] Robot out of bounds: N envs
TERM_BATCH_RE = re.compile(r"\[Termination\] (.+?):\s*(\d+) envs")

# SLURM footer
SLURM_STATE_RE = re.compile(r"State\s*:\s*(.+)")
SLURM_WALLTIME_RE = re.compile(r"Used walltime\s*:\s*(.+)")
SLURM_MAXMEM_RE = re.compile(r"Maximum memory used\s*:\s*([\d.]+\s*\S+)")
SLURM_GPUUTIL_RE = re.compile(r"Max GPU utilization\s*:\s*(\d+)%")
SLURM_GPUMEM_RE = re.compile(r"Max GPU memory used\s*:\s*([\d.]+\s*\S+)")
OOM_RE = re.compile(r"oom_kill|OUT_OF_MEMORY", re.IGNORECASE)

# Config header
CFG_ENVS_RE = re.compile(r"Envs × nsteps:\s*(\d+)\s*×\s*(\d+)")
CFG_ALICE_RE = re.compile(r"alice_timesteps=(\d+)")
CFG_BOB_RE = re.compile(r"bob_timesteps=(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_file(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    d = dict(
        path=path,
        num_envs=None,
        nsteps=None,
        alice_timesteps=None,
        bob_timesteps=None,
        resume_iter=None,
        chain_next=None,
        # event counters
        earlyend_alice=[],
        earlyend_bob=[],  # list of (step, max_step, reason)
        term_batches=Counter(),  # reason -> total env count
        alice_reward_types=Counter(),  # label -> count
        alice_xy_displacements=[],  # floats
        alice_to_bob=0,
        bob_to_alice_success=0,
        bob_to_alice_failure=0,
        bob_success_steps=[],  # step numbers when bob succeeds
        bob_success_target_dist=[],
        bob_success_cube_dist=[],
        alice_updates=0,
        bob_updates=0,
        bob_skipped=0,
        alice_update_losses=[],
        bob_update_srs=[],
        # SLURM metadata
        slurm_state=None,
        slurm_walltime=None,
        slurm_maxmem=None,
        slurm_gpuutil=None,
        slurm_gpumem=None,
        oom=False,
    )

    for line in lines:
        # Config
        m = CFG_ENVS_RE.search(line)
        if m:
            d["num_envs"] = int(m.group(1))
            d["nsteps"] = int(m.group(2))
        m = CFG_ALICE_RE.search(line)
        if m:
            d["alice_timesteps"] = int(m.group(1))
        m = CFG_BOB_RE.search(line)
        if m:
            d["bob_timesteps"] = int(m.group(1))
        m = RESUME_RE.search(line)
        if m:
            d["resume_iter"] = int(m.group(1))
        m = CHAIN_RE.search(line)
        if m:
            d["chain_next"] = int(m.group(1))

        # EarlyEnd
        m = EARLYEND_RE.search(line)
        if m:
            phase = m.group(3)  # Alice / Bob
            step = int(m.group(4))
            max_step = int(m.group(5))
            reason = m.group(7).strip()
            if phase == "Alice":
                d["earlyend_alice"].append((step, max_step, reason))
            else:
                d["earlyend_bob"].append((step, max_step, reason))

        # Termination batches
        m = TERM_BATCH_RE.search(line)
        if m:
            d["term_batches"][m.group(1).strip()] += int(m.group(2))

        # Alice Reward
        m = ALICE_REWARD_RE.search(line)
        if m:
            label = m.group(1).strip()
            xy = float(m.group(3))
            d["alice_reward_types"][label] += 1
            d["alice_xy_displacements"].append(xy)

        # Phase transitions
        if ALICE_TO_BOB_RE.search(line):
            d["alice_to_bob"] += 1
        m = BOB_TO_ALICE_RE.search(line)
        if m:
            if m.group(2) == "SUCCESS":
                d["bob_to_alice_success"] += 1
            else:
                d["bob_to_alice_failure"] += 1
        m = BOB_DONE_RE.search(line)
        if m:
            if m.group(1) == "SUCCESS":
                d["bob_to_alice_success"] += 1

        # BobSuccess detail
        m = BOB_SUCCESS_RE.search(line)
        if m:
            d["bob_success_steps"].append(int(m.group(1)))
            d["bob_success_target_dist"].append(float(m.group(2)))
            d["bob_success_cube_dist"].append(float(m.group(3)))

        # Training updates
        m = ALICE_UPDATE_RE.search(line)
        if m:
            d["alice_updates"] += 1
            d["alice_update_losses"].append(float(m.group(2)))
        m = BOB_UPDATE_RE.search(line)
        if m:
            d["bob_updates"] += 1
            d["bob_update_srs"].append(float(m.group(3)))
        m = BOB_SKIP_RE.search(line)
        if m:
            d["bob_skipped"] += 1

        # SLURM footer
        m = SLURM_STATE_RE.search(line)
        if m:
            d["slurm_state"] = m.group(1).strip()
        m = SLURM_WALLTIME_RE.search(line)
        if m:
            d["slurm_walltime"] = m.group(1).strip()
        m = SLURM_MAXMEM_RE.search(line)
        if m:
            d["slurm_maxmem"] = m.group(1).strip()
        m = SLURM_GPUUTIL_RE.search(line)
        if m:
            d["slurm_gpuutil"] = int(m.group(1))
        m = SLURM_GPUMEM_RE.search(line)
        if m:
            d["slurm_gpumem"] = m.group(1).strip()
        if OOM_RE.search(line):
            d["oom"] = True

    return d


# ---------------------------------------------------------------------------
# Aggregate & report
# ---------------------------------------------------------------------------


def _bar(label, val, total, width=40):
    pct = val / total * 100 if total else 0
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:<38} {val:>7,}  [{bar}] {pct:5.1f}%"


def report(jobs: list[dict], out_dir: Path):
    # --- aggregate ---
    agg = dict(
        earlyend_alice=[],
        earlyend_bob=[],
        term_batches=Counter(),
        alice_reward_types=Counter(),
        alice_xy_displacements=[],
        alice_to_bob=0,
        bob_to_alice_success=0,
        bob_to_alice_failure=0,
        bob_success_steps=[],
        bob_success_target_dist=[],
        bob_success_cube_dist=[],
        alice_updates=0,
        bob_updates=0,
        bob_skipped=0,
        alice_update_losses=[],
        bob_update_srs=[],
        oom_count=0,
        total_walltime_s=0,
        gpu_utils=[],
    )

    for j in jobs:
        agg["earlyend_alice"].extend(j["earlyend_alice"])
        agg["earlyend_bob"].extend(j["earlyend_bob"])
        agg["term_batches"].update(j["term_batches"])
        agg["alice_reward_types"].update(j["alice_reward_types"])
        agg["alice_xy_displacements"].extend(j["alice_xy_displacements"])
        agg["alice_to_bob"] += j["alice_to_bob"]
        agg["bob_to_alice_success"] += j["bob_to_alice_success"]
        agg["bob_to_alice_failure"] += j["bob_to_alice_failure"]
        agg["bob_success_steps"].extend(j["bob_success_steps"])
        agg["bob_success_target_dist"].extend(j["bob_success_target_dist"])
        agg["bob_success_cube_dist"].extend(j["bob_success_cube_dist"])
        agg["alice_updates"] += j["alice_updates"]
        agg["bob_updates"] += j["bob_updates"]
        agg["bob_skipped"] += j["bob_skipped"]
        agg["alice_update_losses"].extend(j["alice_update_losses"])
        agg["bob_update_srs"].extend(j["bob_update_srs"])
        if j["oom"]:
            agg["oom_count"] += 1
        if j["slurm_gpuutil"] is not None:
            agg["gpu_utils"].append(j["slurm_gpuutil"])

    lines = []
    W = 80

    def section(title):
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    # ── Header ──────────────────────────────────────────────────────────────
    section("JOB INVENTORY")
    lines.append(f"  Log directory : {LOG_DIR}")
    lines.append(f"  Jobs parsed   : {len(jobs)}")
    lines.append("")
    lines.append(
        f"  {'File':<45} {'State':<22} {'Walltime':<12} {'MaxRAM':<10} {'GPU%'}"
    )
    lines.append(f"  {'-'*45} {'-'*22} {'-'*12} {'-'*10} {'-'*5}")
    for j in sorted(jobs, key=lambda x: x["path"].name):
        fname = j["path"].name
        state = j["slurm_state"] or "?"
        wt = j["slurm_walltime"] or "?"
        mem = j["slurm_maxmem"] or "?"
        gpu = f"{j['slurm_gpuutil']}%" if j["slurm_gpuutil"] is not None else "?"
        lines.append(f"  {fname:<45} {state:<22} {wt:<12} {mem:<10} {gpu}")

    oom_pct = agg["oom_count"] / len(jobs) * 100 if jobs else 0
    avg_gpu = sum(agg["gpu_utils"]) / len(agg["gpu_utils"]) if agg["gpu_utils"] else 0
    lines.append("")
    lines.append(
        f"  OOM jobs: {agg['oom_count']}/{len(jobs)} ({oom_pct:.0f}%)   "
        f"Avg GPU util: {avg_gpu:.1f}%"
    )

    # ── Training update funnel ───────────────────────────────────────────────
    section("TRAINING FUNNEL  (why there are no updates)")

    alice_completions = agg["alice_to_bob"]  # how many times Alice finished 200 steps
    alice_rewards_total = sum(agg["alice_reward_types"].values())
    alice_valid = agg["alice_reward_types"].get("Valid Goal", 0)
    bob_started = agg["alice_to_bob"]
    bob_done_total = agg["bob_to_alice_success"] + agg["bob_to_alice_failure"]

    funnel = [
        ("Alice EarlyEnd (terminated before 200 steps)", len(agg["earlyend_alice"])),
        ("Alice completed 200 steps (→ goal evaluated)", alice_completions),
        ("  of which: Valid Goal (→ Bob gets to train)", alice_valid),
        (
            "  of which: Invalid / not moved / XY too small",
            alice_rewards_total - alice_valid,
        ),
        ("Bob phases started", bob_started),
        ("Bob phases completed (success + failure)", bob_done_total),
        ("  Bob succeeded", agg["bob_to_alice_success"]),
        ("  Bob failed", agg["bob_to_alice_failure"]),
        ("Alice PPO updates", agg["alice_updates"]),
        ("Bob PPO updates", agg["bob_updates"]),
        ("Bob updates SKIPPED (no transitions)", agg["bob_skipped"]),
    ]
    ref = max(len(agg["earlyend_alice"]) + alice_completions, 1)
    for label, val in funnel:
        lines.append(_bar(label, val, ref))

    # ── Termination reasons ──────────────────────────────────────────────────
    section("TERMINATION REASONS  (from [Termination] batches)")
    total_term = sum(agg["term_batches"].values())
    lines.append(f"  Total env-terminations logged: {total_term:,}")
    lines.append("")
    for reason, cnt in agg["term_batches"].most_common():
        lines.append(_bar(reason, cnt, total_term))

    # ── EarlyEnd step distribution ───────────────────────────────────────────
    section("EARLY TERMINATION — STEP AT WHICH ROBOTS CRASH")

    alice_steps = [s for s, _, _ in agg["earlyend_alice"]]
    bob_steps = [s for s, _, _ in agg["earlyend_bob"]]

    def _step_buckets(steps, max_step, n_buckets=10):
        if not steps:
            return []
        bucket_size = max(1, max_step // n_buckets)
        buckets = Counter()
        for s in steps:
            buckets[s // bucket_size * bucket_size] += 1
        return sorted(buckets.items())

    alice_max = agg["earlyend_alice"][0][1] if agg["earlyend_alice"] else 200
    bob_max = agg["earlyend_bob"][0][1] if agg["earlyend_bob"] else 400

    lines.append(
        f"  Alice early-ends: {len(alice_steps):,}  "
        f"(median step: {sorted(alice_steps)[len(alice_steps)//2] if alice_steps else '?'}/"
        f"{alice_max})"
    )
    for bucket_start, cnt in _step_buckets(alice_steps, alice_max):
        label = f"Alice step {bucket_start:>3}-{bucket_start + alice_max//10 - 1}"
        lines.append(_bar(label, cnt, max(len(alice_steps), 1)))

    lines.append("")
    lines.append(
        f"  Bob early-ends: {len(bob_steps):,}  "
        f"(median step: {sorted(bob_steps)[len(bob_steps)//2] if bob_steps else '?'}/"
        f"{bob_max})"
    )
    for bucket_start, cnt in _step_buckets(bob_steps, bob_max):
        label = f"Bob step {bucket_start:>3}-{bucket_start + bob_max//10 - 1}"
        lines.append(_bar(label, cnt, max(len(bob_steps), 1)))

    # Early-end reason breakdown
    alice_reasons = Counter(r for _, _, r in agg["earlyend_alice"])
    bob_reasons = Counter(r for _, _, r in agg["earlyend_bob"])
    lines.append("")
    lines.append("  Alice early-end reasons:")
    for r, c in alice_reasons.most_common():
        lines.append(_bar(f"    {r}", c, max(len(alice_steps), 1)))
    lines.append("  Bob early-end reasons:")
    for r, c in bob_reasons.most_common():
        lines.append(_bar(f"    {r}", c, max(len(bob_steps), 1)))

    # ── Alice reward breakdown ───────────────────────────────────────────────
    section("ALICE GOAL EVALUATION BREAKDOWN")
    lines.append(f"  Total goal evaluations: {alice_rewards_total:,}")
    lines.append("")
    for label, cnt in agg["alice_reward_types"].most_common():
        lines.append(_bar(label, cnt, max(alice_rewards_total, 1)))

    if agg["alice_xy_displacements"]:
        xyd = sorted(agg["alice_xy_displacements"])
        p25 = xyd[len(xyd) // 4]
        p50 = xyd[len(xyd) // 2]
        p75 = xyd[3 * len(xyd) // 4]
        lines.append("")
        lines.append(f"  Alice XY displacement (of completed phases):")
        lines.append(
            f"    min={xyd[0]:.3f}m  p25={p25:.3f}m  median={p50:.3f}m  "
            f"p75={p75:.3f}m  max={xyd[-1]:.3f}m"
        )

    # ── Bob success detail ───────────────────────────────────────────────────
    section("BOB SUCCESS DETAIL")
    lines.append(f"  Total Bob successes: {len(agg['bob_success_steps'])}")
    if agg["bob_success_steps"]:
        st = sorted(agg["bob_success_steps"])
        lines.append(
            f"  Steps to success: min={st[0]} median={st[len(st)//2]} max={st[-1]}"
        )
        td = agg["bob_success_target_dist"]
        cd = agg["bob_success_cube_dist"]
        lines.append(
            f"  Target dist at success: "
            f"avg={sum(td)/len(td):.4f}m  max={max(td):.4f}m"
        )
        lines.append(
            f"  Cube dist at success:   "
            f"avg={sum(cd)/len(cd):.4f}m  max={max(cd):.4f}m"
        )

    # ── Per-job summary ──────────────────────────────────────────────────────
    section("PER-JOB SUMMARY")
    lines.append(
        f"  {'File':<45} {'EarlyA':>7} {'EarlyB':>7} {'A→B':>6} "
        f"{'B✓':>5} {'AliceUpd':>8} {'BobUpd':>7} {'BobSkip':>8}"
    )
    lines.append(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*8}")
    for j in sorted(jobs, key=lambda x: x["path"].name):
        lines.append(
            f"  {j['path'].name:<45} "
            f"{len(j['earlyend_alice']):>7,} "
            f"{len(j['earlyend_bob']):>7,} "
            f"{j['alice_to_bob']:>6,} "
            f"{j['bob_to_alice_success']:>5,} "
            f"{j['alice_updates']:>8,} "
            f"{j['bob_updates']:>7,} "
            f"{j['bob_skipped']:>8,}"
        )

    # ── Diagnosis ────────────────────────────────────────────────────────────
    section("DIAGNOSIS")

    issues = []

    if agg["oom_count"] == len(jobs):
        issues.append(
            (
                "CRITICAL — All jobs OOM-killed",
                f"100% of jobs hit the RAM ceiling (127.97 GB) and were killed by SLURM "
                f"before any training updates. The training loop never completes a full rollout. "
                f"This is the primary blocker. Fix the memory leak first.",
            )
        )
    elif agg["oom_count"] > 0:
        issues.append(
            (
                f"CRITICAL — {agg['oom_count']}/{len(jobs)} jobs OOM-killed",
                "Most jobs hit the RAM ceiling. See memory leak note.",
            )
        )

    if avg_gpu < 30:
        issues.append(
            (
                f"LOW GPU utilization ({avg_gpu:.0f}%)",
                "The GPU is mostly idle. CPU-side work (Python loops, print/flush, "
                "observation recomputes) is blocking the pipeline. The GPU waits for "
                "the CPU to dispatch the next step.",
            )
        )

    early_alice_total = len(agg["earlyend_alice"])
    if early_alice_total > 0 and alice_completions > 0:
        crash_rate = early_alice_total / (early_alice_total + alice_completions)
        if crash_rate > 0.5:
            issues.append(
                (
                    f"HIGH Alice crash rate ({crash_rate:.0%})",
                    f"{early_alice_total:,} Alice phases terminated early vs "
                    f"{alice_completions:,} completed. Most Alice phases never reach "
                    f"goal evaluation, so the rollout buffer fills with crash transitions "
                    f"instead of useful goals.",
                )
            )

    top_reason = agg["term_batches"].most_common(1)
    if top_reason and "robot_through_table" in top_reason[0][0]:
        cnt = top_reason[0][1]
        issues.append(
            (
                f"DOMINANT termination: robot_through_table ({cnt:,} envs)",
                "RMPFlow is driving the end-effector below the table surface. "
                "Tighten the workspace bounds in the termination config or add a "
                "floor constraint in RMPFlow. This wastes the majority of rollout "
                "steps on crash recovery instead of goal-directed behaviour.",
            )
        )

    valid_rate = alice_valid / alice_rewards_total if alice_rewards_total else 0
    if valid_rate < 0.5:
        issues.append(
            (
                f"LOW valid-goal rate ({valid_rate:.1%}  {alice_valid:,}/{alice_rewards_total:,})",
                "Less than half of Alice's completed phases produce a valid goal for Bob. "
                "Even when Alice doesn't crash, she mostly fails to move objects far enough "
                "(Not Moved / XY Disp Too Small). Bob gets very few training episodes.",
            )
        )

    if agg["alice_updates"] == 0 and agg["bob_updates"] == 0:
        issues.append(
            (
                "NO TRAINING UPDATES in entire log set",
                "The rollout loop never completes a full nsteps-worth of transitions "
                "because the OOM kill arrives first. Reduce num_envs or nsteps to fit "
                "within the memory budget, or profile the memory growth rate.",
            )
        )

    for title, detail in issues:
        lines.append(f"\n  ⚠  {title}")
        # word-wrap detail at 76 chars
        words = detail.split()
        row = "     "
        for w in words:
            if len(row) + len(w) + 1 > 76:
                lines.append(row)
                row = "     " + w
            else:
                row += (" " if row.strip() else "") + w
        lines.append(row)

    lines.append("")

    # ── Write text report ────────────────────────────────────────────────────
    report_path = out_dir / "diagnosis_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Report → {report_path}")
    print("\n".join(lines))

    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_all(agg: dict, jobs: list[dict], out_dir: Path):
    # 1. Funnel bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Training Event Funnel", fontsize=13)
    labels = [
        "Alice\nEarlyEnd",
        "Alice\nCompleted",
        "Valid\nGoals",
        "Bob\nStarted",
        "Bob\nSuccess",
        "Alice\nUpdates",
        "Bob\nUpdates",
    ]
    values = [
        len(agg["earlyend_alice"]),
        agg["alice_to_bob"],
        agg["alice_reward_types"].get("Valid Goal", 0),
        agg["alice_to_bob"],
        agg["bob_to_alice_success"],
        agg["alice_updates"],
        agg["bob_updates"],
    ]
    colors = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#3498db",
        "#2ecc71",
        "#9b59b6",
        "#9b59b6",
    ]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Count (all jobs combined)")
    ax.set_yscale("log" if max(values) > 100 else "linear")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / "plot_funnel.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {p}")

    # 2. Alice goal breakdown pie
    if agg["alice_reward_types"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle("Alice Goal Evaluation Breakdown", fontsize=13)
        labels_p = list(agg["alice_reward_types"].keys())
        sizes = list(agg["alice_reward_types"].values())
        ax.pie(sizes, labels=labels_p, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        plt.tight_layout()
        p = out_dir / "plot_alice_goals.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    # 3. Early-termination step histograms
    alice_steps = [s for s, _, _ in agg["earlyend_alice"]]
    bob_steps = [s for s, _, _ in agg["earlyend_bob"]]
    if alice_steps or bob_steps:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Early Termination — Step Distribution", fontsize=13)
        if alice_steps:
            alice_max = agg["earlyend_alice"][0][1]
            axes[0].hist(
                alice_steps,
                bins=20,
                range=(0, alice_max),
                color="#e74c3c",
                edgecolor="white",
            )
            axes[0].set_title(f"Alice (max={alice_max})")
            axes[0].set_xlabel("Step at crash")
            axes[0].set_ylabel("Count")
            axes[0].grid(alpha=0.3)
        if bob_steps:
            bob_max = agg["earlyend_bob"][0][1]
            axes[1].hist(
                bob_steps,
                bins=20,
                range=(0, bob_max),
                color="#3498db",
                edgecolor="white",
            )
            axes[1].set_title(f"Bob (max={bob_max})")
            axes[1].set_xlabel("Step at crash")
            axes[1].set_ylabel("Count")
            axes[1].grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / "plot_earlyend_steps.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    # 4. Per-job walltime vs RAM (to spot memory growth)
    wt_labels, wt_vals = [], []
    for j in sorted(jobs, key=lambda x: x["path"].name):
        if j["slurm_walltime"]:
            # parse HH:MM:SS into minutes
            parts = j["slurm_walltime"].strip().split(":")
            try:
                mins = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
            except Exception:
                mins = 0
            wt_labels.append(
                j["path"].name.replace("slurm-", "").replace("-high_1647.out", "")
            )
            wt_vals.append(mins)
    if wt_vals:
        fig, ax = plt.subplots(figsize=(14, 4))
        colors_bar = [
            "#e74c3c" if j.get("oom") else "#2ecc71"
            for j in sorted(jobs, key=lambda x: x["path"].name)
            if j["slurm_walltime"]
        ]
        ax.bar(wt_labels, wt_vals, color=colors_bar, edgecolor="white")
        ax.set_title("Job Walltime (red = OOM-killed)")
        ax.set_ylabel("Minutes run")
        ax.set_xlabel("Job ID")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = out_dir / "plot_walltime.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    # 5. Alice XY displacement distribution
    if agg["alice_xy_displacements"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Alice XY Displacement at Phase End", fontsize=13)
        ax.hist(
            agg["alice_xy_displacements"], bins=40, color="#2ecc71", edgecolor="white"
        )
        ax.axvline(0.07, color="red", linestyle="--", label="min valid (0.07m)")
        ax.set_xlabel("XY displacement (m)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / "plot_alice_xy.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not LOG_DIR.exists():
        print(f"[ERROR] Log directory not found: {LOG_DIR}")
        sys.exit(1)

    log_files = sorted(LOG_DIR.glob("slurm-*.out"))
    if not log_files:
        print(f"[ERROR] No slurm-*.out files in {LOG_DIR}")
        sys.exit(1)

    print(f"[INFO] Parsing {len(log_files)} log files from {LOG_DIR} ...")
    jobs = [parse_file(f) for f in log_files]
    print(f"[INFO] Done. Building report...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = report(jobs, OUT_DIR)
    plot_all(agg, jobs, OUT_DIR)
    print(f"\n[INFO] All output written to {OUT_DIR}")


if __name__ == "__main__":
    main()
