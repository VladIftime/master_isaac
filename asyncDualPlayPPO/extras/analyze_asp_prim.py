#!/usr/bin/env python3
"""
Parses slurm log files across one or more training-run directories,
traces job chains, stitches them together, deduplicates overlapping
iterations at run boundaries, writes clean CSVs/TXTs, and plots metrics.

Compatible with push-primitive ASP (train_push_asp.py), cuRobo (train_curobo.py),
and legacy (train.py / train_diffik.py) logs.

Usage (single dir):
    python asyncDualPlayPPO/extras/analyze_asp_prim.py --log-dir logs/push_asp

Usage (stitch prior run + current):
    python asyncDualPlayPPO/extras/analyze_asp_prim.py \
        --log-dir logs/push_asp \
        --prior-dirs logs/prev_run \
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
import numpy as np

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
# Push-ASP Bob updates: no ABC field (push-primitive ASP from train_push_asp.py)
PUSH_ASP_BOB_RE = re.compile(
    r"\[Bob Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)\s*\|\s*SR:\s*([-\d.]+)"
)
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
# Explicit anchor emitted by train_high.slurm after the checkpoint detection block
GLOBAL_START_RE = re.compile(r"Global iteration start:\s*(\d+)")
JOB_ID_RE = re.compile(r"slurm-(\d+)(?:-(.*?))?\.(?:out|txt)")
# Summary lines — IK_fail and ABC buf/warm are curobo-only optional fields
ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+SR=([-\d.]+)"
    r"(?:\s*\|\s*IK_fail=([-\d.]+))?"
    r"\s*\|\s*Goals valid=(\d+) invalid=(\d+)"
    r"\s*\|\s*Bob succ=(\d+) fail=(\d+)"
    r"(?:.*?\|\s*ABC buf:\s*(\d+))?"
    r"(?:.*?ABC warm:\s*(YES|NO))?"
)
# Push-ASP iteration summary (train_push_asp.py): different field ordering,
# IK_fail after Bob succ/fail, plus ObjLifted/RobotTable/Term
PUSH_ASP_ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+SR=([-\d.]+)"
    r"\s*\|\s*Goals valid=(\d+) invalid=(\d+)"
    r"\s*\|\s*Bob succ=(\d+) fail=(\d+)"
    r"\s*\|\s*IK_fail=([-\d.]+)"
    r"\s*\|\s*ObjLifted=(\d+) RobotTable=(\d+) Term=(\d+)"
)
ALICE_DISP_RE = re.compile(
    r"\[AliceDisp\]\s+(\d+)/(\d+) valid\s*\|\s*avg 3D=[-\d.]+m\s+avg XY=([-\d.]+)m\s+max XY=([-\d.]+)m"
    r"(?:\s+avg Y=[-\d.]+m)?"
    r"(?:\s+avg Z=([-\d.]+)m)?"
    r"(?:.*?not-moved[^:]+:\s*(\d+)/\d+)?"
)
ALICE_ROT_RE = re.compile(
    r"\[AliceRot\]\s+roll=([-\d.]+)rad\s+pitch=([-\d.]+)rad\s+yaw=([-\d.]+)rad\s+\(n=(\d+)\)"
)
BOB_SR_RE = re.compile(
    r"\[BobSR\]\s+PosSR=([-\d.]+)\s+RotSR=([-\d.]+)\s+PosErr=([-\d.]+)m\s+RotErr=([-\d.]+)rad(.*?)\(n=(\d+)\)"
)
# Push-PPO baseline log format — single-agent compact iteration line
PUSH_ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+Loss=([-+\d.]+)[^\|]*\|\s+Val=([-+\d.]+)\s*\|\s+"
    r"Rew=([-+\d.]+)\s+\(EMA\s+([-+\d.]+)\)\s*\|\s+"
    r"PosErr=([-+\d.]+)\s*\|\s+RotErr=([-+\d.]+)\s*\|\s+"
    r"SR=([-+\d.]+)\s*\|\s+RotSR=([-+\d.]+)\s*\|\s+"
    r"IK_fail=([-+\d.]+)\s*\|\s+"
    r"AvgPushes=([^\s|]+)\s*\|\s+Epi=(\d+)\s*\|\s+"
    r"BestSR=([-+\d.]+)"
)


def parse_logs(log_dir: Path) -> dict:
    """Parse all slurm log files in log_dir (recursively) and return per-job data dict."""
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
        # Try curobo/legacy format first, then push-ASP variant.
        iter_stats: dict[int, dict] = {}
        for im in ITER_RE.finditer(text):
            n = int(im.group(1))
            iter_stats[n] = {
                "sr": float(im.group(2)),
                "ik_fail_rate": float(im.group(3)) if im.group(3) is not None else None,
                "valid": int(im.group(4)),
                "invalid": int(im.group(5)),
                "bob_succ": int(im.group(6)),
                "bob_fail": int(im.group(7)),
                "abc_buf": int(im.group(8)) if im.group(8) is not None else None,
                "abc_warm": (im.group(9) == "YES") if im.group(9) is not None else None,
                "obj_lifted": None,
                "robot_table": None,
                "terminated": None,
                "pos": im.start(),
                "avg_xy": None,
                "max_xy": None,
                "avg_z": None,
                "not_moved_frac": None,
                "alice_rot_roll": None,
                "alice_rot_pitch": None,
                "alice_rot_yaw": None,
                "bob_pos_sr": None,
                "bob_rot_sr": None,
                "bob_pos_err": None,
                "bob_rot_err": None,
            }

        if not iter_stats:
            # Fallback: push-ASP format (train_push_asp.py)
            for im in PUSH_ASP_ITER_RE.finditer(text):
                n = int(im.group(1))
                iter_stats[n] = {
                    "sr": float(im.group(2)),
                    "ik_fail_rate": float(im.group(7)),
                    "valid": int(im.group(3)),
                    "invalid": int(im.group(4)),
                    "bob_succ": int(im.group(5)),
                    "bob_fail": int(im.group(6)),
                    "abc_buf": None,
                    "abc_warm": None,
                    "obj_lifted": int(im.group(8)),
                    "robot_table": int(im.group(9)),
                    "terminated": int(im.group(10)),
                    "pos": im.start(),
                    "avg_xy": None,
                    "max_xy": None,
                    "avg_z": None,
                    "not_moved_frac": None,
                    "alice_rot_roll": None,
                    "alice_rot_pitch": None,
                    "alice_rot_yaw": None,
                    "bob_pos_sr": None,
                    "bob_rot_sr": None,
                    "bob_pos_err": None,
                    "bob_rot_err": None,
                }

        # Attach [AliceDisp] displacement data to the preceding [Iter N]
        # Groups: (valid_n, total_n, avg_xy, max_xy, avg_z?, not_moved_n?)
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
                iter_stats[n]["avg_xy"] = float(dm.group(3))
                iter_stats[n]["max_xy"] = float(dm.group(4))
                iter_stats[n]["avg_z"] = float(dm.group(5)) if dm.group(5) is not None else None
                if dm.group(6) is not None:
                    total = int(dm.group(2))
                    not_moved = int(dm.group(6))
                    iter_stats[n]["not_moved_frac"] = not_moved / total if total > 0 else None

        # Attach [AliceRot] rotation-change data to the preceding [Iter N]
        for rm in ALICE_ROT_RE.finditer(text):
            rp = rm.start()
            n = None
            for ip, in_ in iter_positions:
                if ip < rp:
                    n = in_
                else:
                    break
            if n is not None and iter_stats[n]["alice_rot_roll"] is None:
                iter_stats[n]["alice_rot_roll"]  = float(rm.group(1))
                iter_stats[n]["alice_rot_pitch"] = float(rm.group(2))
                iter_stats[n]["alice_rot_yaw"]   = float(rm.group(3))

        # Attach [BobSR] position/rotation SR data to the preceding [Iter N]
        for bm in BOB_SR_RE.finditer(text):
            bp = bm.start()
            n = None
            for ip, in_ in iter_positions:
                if ip < bp:
                    n = in_
                else:
                    break
            if n is not None and iter_stats[n]["bob_pos_sr"] is None:
                iter_stats[n]["bob_pos_sr"] = float(bm.group(1))
                iter_stats[n]["bob_rot_sr"] = float(bm.group(2))
                iter_stats[n]["bob_pos_err"] = float(bm.group(3))
                iter_stats[n]["bob_rot_err"] = float(bm.group(4))
                
                # Parse optional object-specific metrics
                extra_str = bm.group(5)
                for om in re.finditer(r"Obj(\d+)_pos=([-\d.]+)\s+Obj\1_rot=([-\d.]+)", extra_str):
                    idx = int(om.group(1))
                    iter_stats[n][f"bob_obj{idx}_pos_err"] = float(om.group(2))
                    iter_stats[n][f"bob_obj{idx}_rot_err"] = float(om.group(3))

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
                avg_z = ist.get("avg_z")
                ik_fail_rate = ist.get("ik_fail_rate")
                not_moved_frac = ist.get("not_moved_frac")
                alice_rot_roll = ist.get("alice_rot_roll")
                alice_rot_pitch = ist.get("alice_rot_pitch")
                alice_rot_yaw = ist.get("alice_rot_yaw")
                bob_pos_sr = ist.get("bob_pos_sr")
                bob_rot_sr = ist.get("bob_rot_sr")
                bob_pos_err = ist.get("bob_pos_err")
                bob_rot_err = ist.get("bob_rot_err")
            else:
                v_count = bisect.bisect_left(valid_pos, end) - bisect.bisect_left(valid_pos, prev_update_end)
                inv_count = bisect.bisect_left(invalid_pos, end) - bisect.bisect_left(invalid_pos, prev_update_end)
                avg_xy = None
                max_xy = None
                avg_z = None
                ik_fail_rate = None
                not_moved_frac = None

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
                        "avg_z": avg_z,
                        "ik_fail_rate": ik_fail_rate,
                        "not_moved_frac": not_moved_frac,
                        "alice_rot_roll": alice_rot_roll,
                        "alice_rot_pitch": alice_rot_pitch,
                        "alice_rot_yaw": alice_rot_yaw,
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
                        "avg_z": avg_z,
                        "ik_fail_rate": ik_fail_rate,
                        "not_moved_frac": not_moved_frac,
                        "alice_rot_roll": alice_rot_roll,
                        "alice_rot_pitch": alice_rot_pitch,
                        "alice_rot_yaw": alice_rot_yaw,
                    }
                )
            prev_update_end = end

        alice_updates.sort(key=lambda x: x["local_iter"])

        bob_updates = []
        for bm in BOB_RE.finditer(text):
            it = int(bm.group(1))
            ist = iter_stats.get(it + 1, {}) if iter_stats else {}
            bob_updates.append(
                {
                    "local_iter": it,
                    "loss": float(bm.group(2)),
                    "val": float(bm.group(3)),
                    "rew": float(bm.group(4)),
                    "abc": float(bm.group(5)),
                    "sr": float(bm.group(6)),
                    "abc_coef": float(bm.group(7)) if bm.group(7) is not None else None,
                    "pos_sr": ist.get("bob_pos_sr"),
                    "rot_sr": ist.get("bob_rot_sr"),
                    "pos_err": ist.get("bob_pos_err"),
                    "rot_err": ist.get("bob_rot_err"),
                    "obj0_pos_err": ist.get("bob_obj0_pos_err"),
                    "obj0_rot_err": ist.get("bob_obj0_rot_err"),
                    "obj1_pos_err": ist.get("bob_obj1_pos_err"),
                    "obj1_rot_err": ist.get("bob_obj1_rot_err"),
                    "ik_fail_rate": ist.get("ik_fail_rate"),
                    "obj_lifted": ist.get("obj_lifted"),
                    "robot_table": ist.get("robot_table"),
                    "terminated_count": ist.get("terminated"),
                }
            )

        if not bob_updates:
            # Fallback: push-ASP Bob format (train_push_asp.py) — no ABC field
            for bm in PUSH_ASP_BOB_RE.finditer(text):
                it = int(bm.group(1))
                ist = iter_stats.get(it + 1, {}) if iter_stats else {}
                bob_updates.append(
                    {
                        "local_iter": it,
                        "loss": float(bm.group(2)),
                        "val": float(bm.group(3)),
                        "rew": float(bm.group(4)),
                        "abc": float("nan"),
                        "sr": float(bm.group(5)),
                        "abc_coef": None,
                        "pos_sr": ist.get("bob_pos_sr"),
                        "rot_sr": ist.get("bob_rot_sr"),
                        "pos_err": ist.get("bob_pos_err"),
                        "rot_err": ist.get("bob_rot_err"),
                        "obj0_pos_err": ist.get("bob_obj0_pos_err"),
                        "obj0_rot_err": ist.get("bob_obj0_rot_err"),
                        "obj1_pos_err": ist.get("bob_obj1_pos_err"),
                        "obj1_rot_err": ist.get("bob_obj1_rot_err"),
                        "ik_fail_rate": ist.get("ik_fail_rate"),
                        "obj_lifted": ist.get("obj_lifted"),
                        "robot_table": ist.get("robot_table"),
                        "terminated_count": ist.get("terminated"),
                    }
                )
            
        # Backward compatibility for old logs (scaling step rewards to episodic returns)
        if bob_updates:
            max_rew = max(u["rew"] for u in bob_updates)
            if max_rew < 0.1:
                # Old logs: mean step reward. Scale by 200 to approximate episodic return.
                for u in bob_updates:
                    u["rew"] *= 200.0

        # Push-PPO baseline detection: single-agent compact iteration line
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
                "avg_pushes": pm.group(11),  # may be "nan"
                "episodes": int(pm.group(12)),
                "best_sr": float(pm.group(13)),
            })
        push_updates.sort(key=lambda x: x["local_iter"])

        is_push_log = len(push_updates) > 0

        jobs[job_id] = {
            "path": f,
            "resume_iter": resume_iter,
            "chain_next": chain_next,
            "suffix": suffix,
            "alice": [] if is_push_log else alice_updates,
            "bob": [] if is_push_log else bob_updates,
            "push": push_updates,
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
                    for k in ("alice", "bob", "push"):
                        if jobs[jid][k]:
                            c_max = max(c_max, jobs[jid][k][-1]["local_iter"])
                
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

    if job["push"]:
        first_p = job["push"][0]["local_iter"]
        if first_p != ri:
            issues.append(
                f"Push first iter={first_p} but resume_iter={ri} "
                f"(delta={first_p - ri:+d})"
            )

    if issues:
        print(f"[WARN] Job {job_id} iteration mismatch:")
        for msg in issues:
            print(f"       {msg}")
    else:
        if job["alice"] or job["bob"] or job["push"]:
            print(f"[OK]   Job {job_id} iteration anchor matches resume_iter={ri}")


def assign_global_iters(
    chains: list[list[int]], jobs: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Assign global iteration numbers.
    """
    alice_records = []
    bob_records = []
    push_records = []

    for chain_idx, chain in enumerate(chains):
        seen_alice_iters: set[int] = set()
        seen_bob_iters: set[int] = set()
        seen_push_iters: set[int] = set()

        for job_id in chain:
            job = jobs[job_id]

            # Cross-check: slurm anchor vs. actual log numbers
            cross_check_job(job_id, job)

            for upd in job["alice"]:
                g = upd["local_iter"]
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
                g = upd["local_iter"]
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

            for upd in job["push"]:
                g = upd["local_iter"]
                if g in seen_push_iters:
                    continue
                seen_push_iters.add(g)
                push_records.append(
                    {
                        "chain": chain_idx,
                        "job_id": job_id,
                        "global_iter": g,
                        **{k: v for k, v in upd.items() if k != "local_iter"},
                    }
                )

    alice_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    bob_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    push_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    return alice_records, bob_records, push_records


def write_csv(alice_records: list[dict], bob_records: list[dict], out_dir: Path,
              push_records: list[dict] = None):
    out_path = out_dir / "training_updates.csv"
    fieldnames = [
        "agent",
        "chain",
        "job_id",
        "global_iter",
        "loss",
        "val",
        "rew",
        "rew_ema",
        "entropy_coef",
        "abc",
        "abc_coef",
        "sr",
        "valid_goals",
        "invalid_goals",
        "avg_xy",
        "max_xy",
        "avg_z",
        "ik_fail_rate",
        "not_moved_frac",
        "alice_rot_roll",
        "alice_rot_pitch",
        "alice_rot_yaw",
        "pos_sr",
        "rot_sr",
        "pos_err",
        "rot_err",
        "avg_pushes",
        "episodes",
        "best_sr",
        "obj0_pos_err",
        "obj0_rot_err",
        "obj1_pos_err",
        "obj1_rot_err",
        "obj_lifted",
        "robot_table",
        "terminated_count",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in alice_records:
            writer.writerow({"agent": "alice", "abc": "", "abc_coef": "", "sr": "",
                             "rew_ema": "", "pos_sr": "", "rot_sr": "", "pos_err": "", "rot_err": "",
                             "avg_pushes": "", "episodes": "", "best_sr": "", 
                             "obj0_pos_err": "", "obj0_rot_err": "", "obj1_pos_err": "", "obj1_rot_err": "",
                             "obj_lifted": "", "robot_table": "", "terminated_count": "", **r})
        for r in bob_records:
            writer.writerow({
                "agent": "bob",
                "entropy_coef": "", "valid_goals": "", "invalid_goals": "",
                "avg_xy": "", "max_xy": "", "avg_z": "",
                "ik_fail_rate": "", "not_moved_frac": "",
                "alice_rot_roll": "", "alice_rot_pitch": "", "alice_rot_yaw": "",
                "rew_ema": "", "avg_pushes": "", "episodes": "", "best_sr": "",
                **r,
            })
        if push_records:
            for r in push_records:
                writer.writerow({
                    "agent": "push",
                    "chain": r["chain"],
                    "job_id": r["job_id"],
                    "global_iter": r["global_iter"],
                    "loss": r.get("loss"),
                    "val": r.get("val"),
                    "rew": r.get("rew"),
                    "rew_ema": r.get("rew_ema"),
                    "entropy_coef": "",
                    "abc": "",
                    "abc_coef": "",
                    "sr": r.get("sr"),
                    "valid_goals": "",
                    "invalid_goals": "",
                    "avg_xy": "",
                    "max_xy": "",
                    "avg_z": "",
                    "ik_fail_rate": r.get("ik_fail_rate"),
                    "not_moved_frac": "",
                    "alice_rot_roll": "",
                    "alice_rot_pitch": "",
                    "alice_rot_yaw": "",
                    "pos_sr": "",
                    "rot_sr": r.get("rot_sr"),
                    "pos_err": r.get("pos_err"),
                    "rot_err": r.get("rot_err"),
                    "avg_pushes": r.get("avg_pushes"),
                    "episodes": r.get("episodes"),
                    "best_sr": r.get("best_sr"),
                    "obj0_pos_err": "",
                    "obj0_rot_err": "",
                    "obj1_pos_err": "",
                    "obj1_rot_err": "",
                    "obj_lifted": "",
                    "robot_table": "",
                    "terminated_count": "",
                })
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
        "rew_ema",
        "entropy_coef",
        "abc",
        "abc_coef",
        "sr",
        "valid_goals",
        "invalid_goals",
        "avg_xy",
        "max_xy",
        "avg_z",
        "ik_fail_rate",
        "not_moved_frac",
        "alice_rot_roll",
        "alice_rot_pitch",
        "alice_rot_yaw",
        "pos_sr",
        "rot_sr",
        "pos_err",
        "rot_err",
        "avg_pushes",
        "episodes",
        "best_sr",
        "obj0_pos_err",
        "obj0_rot_err",
        "obj1_pos_err",
        "obj1_rot_err",
        "obj_lifted",
        "robot_table",
        "terminated_count",
    ]

    def _v(d, k):
        v = d.get(k)
        return "" if v is None else v

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
                    "entropy_coef": upd.get("entropy_coef") or "",
                    "abc": "",
                    "abc_coef": "",
                    "sr": "",
                    "valid_goals": _v(upd, "valid_goals"),
                    "invalid_goals": _v(upd, "invalid_goals"),
                    "avg_xy": _v(upd, "avg_xy"),
                    "max_xy": _v(upd, "max_xy"),
                    "avg_z": _v(upd, "avg_z"),
                    "ik_fail_rate": _v(upd, "ik_fail_rate"),
                    "not_moved_frac": _v(upd, "not_moved_frac"),
                    "alice_rot_roll": _v(upd, "alice_rot_roll"),
                    "alice_rot_pitch": _v(upd, "alice_rot_pitch"),
                    "alice_rot_yaw": _v(upd, "alice_rot_yaw"),
                    "pos_sr": "",
                    "rot_sr": "",
                    "pos_err": "",
                    "rot_err": "",
                    "obj0_pos_err": "",
                    "obj0_rot_err": "",
                    "obj1_pos_err": "",
                    "obj1_rot_err": "",
                    "obj_lifted": "",
                    "robot_table": "",
                    "terminated_count": "",
                })
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
                    "abc_coef": _v(upd, "abc_coef"),
                    "sr": upd["sr"],
                    "valid_goals": "",
                    "invalid_goals": "",
                    "avg_xy": "",
                    "max_xy": "",
                    "avg_z": "",
                    "ik_fail_rate": "",
                    "not_moved_frac": "",
                    "alice_rot_roll": "",
                    "alice_rot_pitch": "",
                    "alice_rot_yaw": "",
                    "pos_sr": _v(upd, "pos_sr"),
                    "rot_sr": _v(upd, "rot_sr"),
                    "pos_err": _v(upd, "pos_err"),
                    "rot_err": _v(upd, "rot_err"),
                    "rew_ema": "",
                    "avg_pushes": "",
                    "episodes": "",
                    "best_sr": "",
                    "obj0_pos_err": _v(upd, "obj0_pos_err"),
                    "obj0_rot_err": _v(upd, "obj0_rot_err"),
                    "obj1_pos_err": _v(upd, "obj1_pos_err"),
                    "obj1_rot_err": _v(upd, "obj1_rot_err"),
                    "obj_lifted": _v(upd, "obj_lifted"),
                    "robot_table": _v(upd, "robot_table"),
                    "terminated_count": _v(upd, "terminated_count"),
                })
            for upd in job["push"]:
                writer.writerow({
                    "agent": "push",
                    "chain": chain_idx,
                    "job_id": job_id,
                    "local_iter": upd["local_iter"],
                    "loss": upd["loss"],
                    "val": upd["val"],
                    "rew": upd["rew"],
                    "rew_ema": upd["rew_ema"],
                    "entropy_coef": "",
                    "abc": "",
                    "abc_coef": "",
                    "sr": upd["sr"],
                    "valid_goals": "",
                    "invalid_goals": "",
                    "avg_xy": "",
                    "max_xy": "",
                    "avg_z": "",
                    "ik_fail_rate": upd["ik_fail_rate"],
                    "not_moved_frac": "",
                    "alice_rot_roll": "",
                    "alice_rot_pitch": "",
                    "alice_rot_yaw": "",
                    "pos_sr": "",
                    "rot_sr": upd["rot_sr"],
                    "pos_err": upd["pos_err"],
                    "rot_err": upd["rot_err"],
                    "avg_pushes": upd["avg_pushes"],
                    "episodes": upd["episodes"],
                    "best_sr": upd["best_sr"],
                    "obj0_pos_err": "",
                    "obj0_rot_err": "",
                    "obj1_pos_err": "",
                    "obj1_rot_err": "",
                    "obj_lifted": "",
                    "robot_table": "",
                    "terminated_count": "",
                })
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


def _plot_push_metrics(
    push_records: list[dict],
    out_dir: Path,
    title_suffix: str = "",
    separate: bool = False,
    log_paths: list[Path] = None,
):
    """Render Push-PPO baseline plots from iter records + re-parsed episode data."""
    # ── Iter-level metrics (same as before) ──────────────────────────────
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
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    if separate:
        # Separate PNG per metric
        for key, ylabel, title, fname in [
            ("loss", "Surrogate Loss", "Push-PPO — Policy Loss", "plot_loss.png"),
            ("val", "Value Loss", "Push-PPO — Value Loss", "plot_value_loss.png"),
            ("rew", "Mean Reward", "Push-PPO — Mean Reward", "plot_reward.png"),
            ("rew_ema", "EMA Reward", "Push-PPO — EMA Reward", "plot_reward_ema.png"),
            ("sr", "Success Rate", "Push-PPO — Success Rate", "plot_bob_sr.png"),
            ("rot_sr", "Rotation SR", "Push-PPO — Rotation SR", "plot_rot_sr.png"),
            ("pos_err", "Position Error (m)", "Push-PPO — Position Error", "plot_pos_err.png"),
            ("rot_err", "Rotation Error (rad)", "Push-PPO — Rotation Error", "plot_rot_err.png"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_p(ax, key, ylabel, title)
            plt.tight_layout()
            _save(fig, fname)

        # IK fail rate
        has_ik = any(r.get("ik_fail_rate") is not None for r in push_records)
        if has_ik:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_p(ax, "ik_fail_rate", "IK Fail Rate", "Push-PPO — IK Fail Rate")
            ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            _save(fig, "plot_ik_fail.png")

        # Best SR
        fig, ax = plt.subplots(figsize=(10, 5))
        _draw_p(ax, "best_sr", "Best SR", "Push-PPO — Best Success Rate")
        plt.tight_layout()
        _save(fig, "plot_best_sr.png")
        return

    # Combined overview
    from matplotlib.gridspec import GridSpec

    n_rows = 4  # loss/val/rew row, sr/rotsr row, poserr/roterr row, best_sr row
    has_ik = any(r.get("ik_fail_rate") is not None for r in push_records)
    if has_ik:
        n_rows += 1

    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.32)
    _row = 0

    ax_loss = fig.add_subplot(gs[_row, 0])
    ax_val = fig.add_subplot(gs[_row, 1])
    ax_rew = fig.add_subplot(gs[_row, 2]); _row += 1

    ax_sr = fig.add_subplot(gs[_row, 0])
    ax_rotsr = fig.add_subplot(gs[_row, 1])
    ax_rew_ema = fig.add_subplot(gs[_row, 2]); _row += 1

    ax_poserr = fig.add_subplot(gs[_row, 0])
    ax_roterr = fig.add_subplot(gs[_row, 1])
    ax_best = fig.add_subplot(gs[_row, 2]); _row += 1

    _draw_p(ax_loss, "loss", "Surrogate Loss", "Push-PPO — Policy Loss")
    _draw_p(ax_val, "val", "Value Loss", "Push-PPO — Value Loss")
    _draw_p(ax_rew, "rew", "Mean Reward", "Push-PPO — Mean Reward")
    _draw_p(ax_sr, "sr", "Success Rate", "Push-PPO — Success Rate")
    _draw_p(ax_rotsr, "rot_sr", "Rotation SR", "Push-PPO — Rotation Success Rate")
    _draw_p(ax_rew_ema, "rew_ema", "EMA Reward", "Push-PPO — EMA Reward")
    _draw_p(ax_poserr, "pos_err", "Position Error (m)", "Push-PPO — Position Error")
    _draw_p(ax_roterr, "rot_err", "Rotation Error (rad)", "Push-PPO — Rotation Error")
    _draw_p(ax_best, "best_sr", "Best SR", "Push-PPO — Best Success Rate")

    if has_ik:
        ax_ik = fig.add_subplot(gs[_row, 0])
        ax_spare1 = fig.add_subplot(gs[_row, 1])
        ax_spare2 = fig.add_subplot(gs[_row, 2]); _row += 1
        _draw_p(ax_ik, "ik_fail_rate", "IK Fail Rate", "Push-PPO — IK Fail Rate")
        ax_ik.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
        ax_ik.legend(fontsize=8)
        ax_spare1.axis("off")
        ax_spare2.axis("off")

    fig.suptitle(f"Push-PPO Training Overview{title_suffix}", fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout()
    _save(fig, "plot_overview.png")

    # ── Episode-level plots (re-parse slurm logs for [Episode] lines) ────
    if log_paths:
        import re as _re
        EP_RE = _re.compile(
            r"\[Episode\]\s+pushes=(\d+)\s+(SUCCESS|fail)\s+"
            r"rew=([-+\d.]+)\s+"
            r"goal=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+orient=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
            r"final=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
            r"rot=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
            r"err_pos=([-\d.]+)m\s+err_rot=([-\d.]+)rad"
        )
        episodes = []
        for lp in log_paths:
            if lp.exists():
                for m in EP_RE.finditer(lp.read_text(errors="replace")):
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
            w = min(200, len(episodes) // 10)
            srs = [1.0 if e["success"] else 0.0 for e in episodes]
            rolling = np.convolve(srs, np.ones(w) / w, mode="valid")
            ax2.plot(range(len(rolling)), rolling, color="blue", linewidth=1.5,
                     label=f"Episode SR (window={w})")
            ax2.set_title(f"Push-PPO — Episode-Level Success Rate{title_suffix}")
            ax2.set_xlabel("Episode"); ax2.set_ylabel("Success Rate")
            ax2.legend(); ax2.grid(True, alpha=0.3)
            _save(fig2, "plot_episode_sr.png")

            # Reward histogram
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            rews = [e["rew"] for e in episodes]
            ax3.hist(rews, bins=80, color="blue", alpha=0.7, edgecolor="white")
            ax3.axvline(np.mean(rews), color="red", linewidth=1.5, linestyle="--",
                        label=f"Mean = {np.mean(rews):+.2f}")
            ax3.set_title(f"Push-PPO — Episode Reward Distribution{title_suffix} ({len(episodes)} eps)")
            ax3.set_xlabel("Episode Reward"); ax3.set_ylabel("Count")
            ax3.legend(); ax3.grid(True, alpha=0.3)
            _save(fig3, "plot_reward_histogram.png")

            # Final positions scatter
            fig4, ax4 = plt.subplots(figsize=(8, 8))
            sample = episodes[-10000:] if len(episodes) > 10000 else episodes
            fx = [e["final_x"] for e in sample]; fy = [e["final_y"] for e in sample]
            gx = [e["goal_x"] for e in sample]; gy = [e["goal_y"] for e in sample]
            ax4.scatter(fx, fy, s=1, alpha=0.3, color="red", label="Final obj pos")
            ax4.scatter(gx, gy, s=1, alpha=0.15, color="green", label="Goal pos")
            ax4.set_xlim(-0.6, 0.6); ax4.set_ylim(0.15, 0.75)
            ax4.set_aspect("equal")
            ax4.set_title(f"Push-PPO — Object Final Positions{title_suffix} (last {len(sample)} eps)")
            ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)")
            ax4.legend(fontsize=8, markerscale=5); ax4.grid(True, alpha=0.3)
            _save(fig4, "plot_final_positions.png")


def plot_metrics(
    alice_records: list[dict],
    bob_records: list[dict],
    out_dir: Path,
    title_suffix: str = "",
    separate: bool = False,
    push_records: list[dict] = None,
    log_paths: list[Path] = None,
):
    """Render training plots."""
    if push_records:
        _plot_push_metrics(push_records, out_dir, title_suffix, separate, log_paths)
        return

    if not alice_records and not bob_records:
        print("[WARN] No records to plot.")
        return

    alice_colors = ["tab:blue", "cornflowerblue", "navy", "steelblue"]
    bob_colors   = ["tab:red",  "tomato",          "darkred", "salmon"]

    all_chain_indices = sorted(
        set([r["chain"] for r in alice_records] + [r["chain"] for r in bob_records])
    )
    a_by_chain = [[r for r in alice_records if r["chain"] == c] for c in all_chain_indices]
    b_by_chain = [[r for r in bob_records   if r["chain"] == c] for c in all_chain_indices]
    a_labels = [f"Alice C{c}" for c in all_chain_indices]
    b_labels = [f"Bob C{c}"   for c in all_chain_indices]

    a_disp_by_chain     = [[r for r in recs if r.get("avg_xy")         is not None] for recs in a_by_chain]
    a_z_by_chain        = [[r for r in recs if r.get("avg_z")          is not None] for recs in a_by_chain]
    a_notmov_by_chain   = [[r for r in recs if r.get("not_moved_frac") is not None] for recs in a_by_chain]
    a_roll_by_chain     = [[r for r in recs if r.get("alice_rot_roll")  is not None] for recs in a_by_chain]
    a_pitch_by_chain    = [[r for r in recs if r.get("alice_rot_pitch") is not None] for recs in a_by_chain]
    a_yaw_by_chain      = [[r for r in recs if r.get("alice_rot_yaw")   is not None] for recs in a_by_chain]
    b_pos_sr_by_chain   = [[r for r in recs if r.get("pos_sr")          is not None] for recs in b_by_chain]
    b_rot_sr_by_chain   = [[r for r in recs if r.get("rot_sr")          is not None] for recs in b_by_chain]
    has_alice_rot_data  = any(a_roll_by_chain)

    b_pos_err_by_chain  = [[r for r in recs if r.get("pos_err")         is not None] for recs in b_by_chain]
    b_rot_err_by_chain  = [[r for r in recs if r.get("rot_err")         is not None] for recs in b_by_chain]
    has_bob_err         = any(b_pos_err_by_chain)

    # Push-ASP specific metrics (train_push_asp.py)
    b_ik_fail_by_chain     = [[r for r in recs if r.get("ik_fail_rate")       is not None] for recs in b_by_chain]
    b_obj_lifted_by_chain  = [[r for r in recs if r.get("obj_lifted")         is not None] for recs in b_by_chain]
    b_robot_table_by_chain = [[r for r in recs if r.get("robot_table")        is not None] for recs in b_by_chain]
    b_terminated_by_chain  = [[r for r in recs if r.get("terminated_count")   is not None] for recs in b_by_chain]
    has_push_asp_stability = any(b_obj_lifted_by_chain)

    # ------------------------------------------------------------------ helpers
    def _draw(ax, records_list, labels, colors, key, linestyle="-"):
        """Add lines for one metric onto ax — no axis formatting."""
        for records, label, color in zip(records_list, labels, colors):
            pts = [(r["global_iter"], r[key]) for r in records if r.get(key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, smooth(list(ys)), color=color, label=label, linewidth=1.5, linestyle=linestyle)

    def _fmt(ax, ylabel, title):
        ax.set_title(title + title_suffix)
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _fill_both(ax, records_list_a, records_list_b, labels_a, labels_b, key, ylabel, title):
        """Draw both Alice and Bob series onto ax, then format."""
        _draw(ax, records_list_a, labels_a, alice_colors, key)
        _draw(ax, records_list_b, labels_b, bob_colors,   key)
        _fmt(ax, ylabel, title)

    # ---------------------------------------------------------------- save helper
    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    # ============================================================ separate mode
    if separate:
        fig, ax = plt.subplots(figsize=(10, 5))
        _draw(ax, a_by_chain, a_labels, alice_colors, "loss")
        _fmt(ax, "Loss", "Policy Loss — Alice")
        plt.tight_layout(); _save(fig, "plot_loss.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _fill_both(ax, a_by_chain, b_by_chain, a_labels, b_labels, "val", "Value Loss",
              "Value Loss — Alice & Bob")
        plt.tight_layout(); _save(fig, "plot_value_loss.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _fill_both(ax, a_by_chain, b_by_chain, a_labels, b_labels, "rew", "Reward",
              "Mean Episode Reward — Alice & Bob")
        plt.tight_layout(); _save(fig, "plot_reward.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _draw(ax, b_by_chain, b_labels, bob_colors, "sr")
        _fmt(ax, "Success Rate", "Bob — Success Rate")
        plt.tight_layout(); _save(fig, "plot_bob_sr.png")

        if any(a_disp_by_chain):
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw(ax, a_disp_by_chain, [f"XY {l}" for l in a_labels], alice_colors, "avg_xy")
            if any(a_z_by_chain):
                _draw(ax, a_z_by_chain, [f"Z {l}" for l in a_labels], ["tab:orange"]*len(a_labels), "avg_z", linestyle="--")
            _fmt(ax, "Displacement (m)", "Alice — Avg Goal Displacement")
            plt.tight_layout(); _save(fig, "plot_alice_goal_displacement.png")

        if any(a_notmov_by_chain):
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw(ax, a_notmov_by_chain, a_labels, alice_colors, "not_moved_frac")
            _fmt(ax, "Not-Moved Fraction", "Alice — Not-Moved Phase Fraction")
            plt.tight_layout(); _save(fig, "plot_not_moved.png")

        if has_alice_rot_data:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw(ax, a_roll_by_chain, [f"Roll {l}" for l in a_labels], alice_colors, "alice_rot_roll")
            _draw(ax, a_pitch_by_chain, [f"Pitch {l}" for l in a_labels], ["tab:orange"]*len(a_labels), "alice_rot_pitch", linestyle="--")
            _draw(ax, a_yaw_by_chain, [f"Yaw {l}" for l in a_labels], ["tab:green"]*len(a_labels), "alice_rot_yaw", linestyle=":")
            _fmt(ax, "Rotation (rad)", "Alice — Goal Rotation Change")
            plt.tight_layout(); _save(fig, "plot_alice_rotation.png")

        if any(b_pos_sr_by_chain) or any(b_rot_sr_by_chain):
            fig, axes_sr = plt.subplots(1, 2, figsize=(14, 5))
            _draw(axes_sr[0], b_pos_sr_by_chain, b_labels, bob_colors, "pos_sr")
            _fmt(axes_sr[0], "Position SR", "Bob — Position-Only Success Rate")
            _draw(axes_sr[1], b_rot_sr_by_chain, b_labels, bob_colors, "rot_sr")
            _fmt(axes_sr[1], "Rotation SR", "Bob — Rotation-Only Success Rate")
            plt.tight_layout(); _save(fig, "plot_bob_sr_split.png")

        if has_bob_err:
            fig, axes_err = plt.subplots(1, 2, figsize=(14, 5))
            _draw(axes_err[0], b_pos_err_by_chain, [f"Global {l}" for l in b_labels], bob_colors, "pos_err", linestyle="-")
            _draw(axes_err[1], b_rot_err_by_chain, [f"Global {l}" for l in b_labels], bob_colors, "rot_err", linestyle="-")
            
            _draw(axes_err[0], b_pos_err_by_chain, [f"Obj0 {l}" for l in b_labels], ["tab:orange"]*len(b_labels), "obj0_pos_err", linestyle="--")
            _draw(axes_err[1], b_rot_err_by_chain, [f"Obj0 {l}" for l in b_labels], ["tab:orange"]*len(b_labels), "obj0_rot_err", linestyle="--")
            
            _draw(axes_err[0], b_pos_err_by_chain, [f"Obj1 {l}" for l in b_labels], ["tab:purple"]*len(b_labels), "obj1_pos_err", linestyle=":")
            _draw(axes_err[1], b_rot_err_by_chain, [f"Obj1 {l}" for l in b_labels], ["tab:purple"]*len(b_labels), "obj1_rot_err", linestyle=":")
            
            _fmt(axes_err[0], "Position Error (m)", "Bob — Position Error")
            _fmt(axes_err[1], "Rotation Error (rad)", "Bob — Rotation Error")
            plt.tight_layout(); _save(fig, "plot_bob_errors.png")

        return

    # ============================================================ combined mode
    from matplotlib.gridspec import GridSpec
    import math

    panels = []

    def plot_loss(ax):
        _draw(ax, a_by_chain, a_labels, alice_colors, "loss")
        _fmt(ax, "Loss", "Policy Loss — Alice")
    panels.append(plot_loss)

    def plot_val(ax):
        _fill_both(ax, a_by_chain, b_by_chain, a_labels, b_labels, "val", "Value Loss", "Value Loss — Alice & Bob")
    panels.append(plot_val)

    def plot_rew(ax):
        _fill_both(ax, a_by_chain, b_by_chain, a_labels, b_labels, "rew", "Reward", "Episode Reward — Alice & Bob")
    panels.append(plot_rew)

    def plot_sr(ax):
        _draw(ax, b_by_chain, b_labels, bob_colors, "sr")
        _fmt(ax, "Success Rate", "Bob — Success Rate")
    panels.append(plot_sr)

    def plot_goals(ax):
        _draw(ax, a_by_chain, [f"Valid {l}" for l in a_labels], alice_colors, "valid_goals")
        _draw(ax, a_by_chain, [f"Invalid {l}" for l in a_labels], ["tab:orange", "darkorange", "saddlebrown", "peru"], "invalid_goals", linestyle="--")
        _fmt(ax, "Goals", "Alice — Valid & Invalid Goals")
    panels.append(plot_goals)

    if any(a_disp_by_chain):
        def plot_disp(ax):
            _draw(ax, a_disp_by_chain, [f"XY {l}" for l in a_labels], alice_colors, "avg_xy")
            if any(a_z_by_chain):
                _draw(ax, a_z_by_chain, [f"Z {l}" for l in a_labels], ["tab:orange"]*len(a_labels), "avg_z", linestyle="--")
            _fmt(ax, "Displacement (m)", "Alice — Avg Goal Displacement")
        panels.append(plot_disp)

    if has_alice_rot_data:
        def plot_rot(ax):
            _draw(ax, a_roll_by_chain, [f"Roll {l}" for l in a_labels], alice_colors, "alice_rot_roll")
            _draw(ax, a_pitch_by_chain, [f"Pitch {l}" for l in a_labels], ["tab:orange"]*len(a_labels), "alice_rot_pitch", linestyle="--")
            _draw(ax, a_yaw_by_chain, [f"Yaw {l}" for l in a_labels], ["tab:green"]*len(a_labels), "alice_rot_yaw", linestyle=":")
            _fmt(ax, "Rotation (rad)", "Alice — Rot Change")
        panels.append(plot_rot)

    if any(b_pos_sr_by_chain) or any(b_rot_sr_by_chain):
        def plot_bob_srs(ax):
            _draw(ax, b_pos_sr_by_chain, [f"Pos SR {l}" for l in b_labels], bob_colors, "pos_sr")
            _draw(ax, b_rot_sr_by_chain, [f"Rot SR {l}" for l in b_labels], ["tab:purple"]*len(b_labels), "rot_sr", linestyle="--")
            _fmt(ax, "Success Rate", "Bob — Pos/Rot Success Rate")
        panels.append(plot_bob_srs)

    if any(a_notmov_by_chain):
        def plot_notmov(ax):
            _draw(ax, a_notmov_by_chain, a_labels, alice_colors, "not_moved_frac")
            _fmt(ax, "Not-Moved Fraction", "Alice — Not-Moved Phase Fraction")
        panels.append(plot_notmov)

    if has_bob_err:
        def plot_err_pos(ax):
            _draw(ax, b_pos_err_by_chain, [f"Global {l}" for l in b_labels], bob_colors, "pos_err", linestyle="-")
            _draw(ax, b_pos_err_by_chain, [f"Obj0 {l}" for l in b_labels], ["tab:orange"]*len(b_labels), "obj0_pos_err", linestyle="--")
            _draw(ax, b_pos_err_by_chain, [f"Obj1 {l}" for l in b_labels], ["tab:purple"]*len(b_labels), "obj1_pos_err", linestyle=":")
            _fmt(ax, "Position Error (m)", "Bob — Position Error")
        panels.append(plot_err_pos)

        def plot_err_rot(ax):
            _draw(ax, b_rot_err_by_chain, [f"Global {l}" for l in b_labels], bob_colors, "rot_err", linestyle="-")
            _draw(ax, b_rot_err_by_chain, [f"Obj0 {l}" for l in b_labels], ["tab:orange"]*len(b_labels), "obj0_rot_err", linestyle="--")
            _draw(ax, b_rot_err_by_chain, [f"Obj1 {l}" for l in b_labels], ["tab:purple"]*len(b_labels), "obj1_rot_err", linestyle=":")
            _fmt(ax, "Rotation Error (rad)", "Bob — Rotation Error")
        panels.append(plot_err_rot)

    # Push-ASP specific: IK fail rate and training stability
    if has_push_asp_stability or any(b_ik_fail_by_chain):
        def plot_ik_fail(ax):
            if any(b_ik_fail_by_chain):
                _draw(ax, b_ik_fail_by_chain, b_labels, bob_colors, "ik_fail_rate")
            _fmt(ax, "IK Fail Rate", "IK Fail Rate")
            ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
            ax.legend(fontsize=8)
        panels.append(plot_ik_fail)

    if has_push_asp_stability:
        def plot_obj_lifted(ax):
            _draw(ax, b_obj_lifted_by_chain, b_labels, bob_colors, "obj_lifted")
            _fmt(ax, "Count", "Obj Lifted Count")
        panels.append(plot_obj_lifted)

        def plot_robot_table(ax):
            _draw(ax, b_robot_table_by_chain, b_labels, bob_colors, "robot_table")
            _fmt(ax, "Count", "Robot Through Table")
        panels.append(plot_robot_table)

        def plot_terminated(ax):
            _draw(ax, b_terminated_by_chain, b_labels, bob_colors, "terminated_count")
            _fmt(ax, "Count", "Terminated Count")
        panels.append(plot_terminated)

    n_cols = 3
    n_rows = math.ceil(len(panels) / n_cols)
    fig_h = max(4.5 * n_rows, 5.0)
    fig = plt.figure(figsize=(18, fig_h))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)
    
    for idx, panel_fn in enumerate(panels):
        r = idx // n_cols
        c = idx % n_cols
        ax = fig.add_subplot(gs[r, c])
        panel_fn(ax)

    fig.suptitle(f"Training Overview{title_suffix}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "plot_overview.png")



def write_summary_txt(chain_idx: int, chain: list[int], a_c: list[dict], b_c: list[dict], out_dir: Path,
                      push_c: list[dict] = None):
    """Write human-readable per-chain summary."""
    summary_path = out_dir / "training_updates.txt"
    bob_by_iter = {r["global_iter"]: r for r in b_c}
    with open(summary_path, "w") as f:
        f.write(f"=== TRAINING UPDATES SUMMARY (Chain {chain_idx}) ===\n\n")
        f.write(f"Jobs in chain: {' → '.join(str(j) for j in chain)}\n\n")

        if push_c:
            f.write("Push-PPO Baseline (single-agent):\n")
            for pr in push_c:
                f.write(
                    f"  Iter {pr['global_iter']:5d} | "
                    f"Loss={pr['loss']:+.4f}  Val={pr['val']:.4f}  "
                    f"Rew={pr['rew']:+.4f} (EMA {pr['rew_ema']:+.4f})  "
                    f"SR={pr['sr']:.4f}  RotSR={pr['rot_sr']:.4f}  "
                    f"PosErr={pr['pos_err']:.4f}  RotErr={pr['rot_err']:.4f}  "
                    f"BestSR={pr['best_sr']:.4f}\n"
                )
        else:
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
        default=None,
        help="Directory containing slurm log files (slurm-*-*.out).",
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
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        default=False,
        help="Save one PNG per metric instead of a single combined overview PNG.",
    )
    args = parser.parse_args()

    if args.log_dir is None:
        parser.error("--log-dir is required. Point it to the directory containing slurm-*.out files.")

    log_dir = Path(args.log_dir)
    prior_dirs = [Path(d) for d in args.prior_dirs]
    out_dir = Path(args.out_dir) if args.out_dir else log_dir

    if not log_dir.exists():
        parser.error(f"--log-dir does not exist: {log_dir}")

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

    alice_records, bob_records, push_records = assign_global_iters(chains, jobs)
    print(
        f"[INFO] Alice updates: {len(alice_records)}, Bob updates: {len(bob_records)}, "
        f"Push updates: {len(push_records)}"
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
    if push_records:
        print(
            f"[INFO] Push  global iter range: {push_records[0]['global_iter']} "
            f"→ {push_records[-1]['global_iter']}"
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
        p_c = [r for r in push_records if r["chain"] == i]

        write_raw_csv(i, ch, jobs, chain_dir)
        write_csv(a_c, b_c, chain_dir, push_records=p_c)
        write_summary_txt(i, ch, a_c, b_c, chain_dir, push_c=p_c)
        plots_dir = chain_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_metrics(a_c, b_c, plots_dir, title_suffix=f" (Chain {i})",
                     separate=args.separate_plots, push_records=p_c,
                     log_paths=[jobs[jid]["path"] for jid in ch])

    write_csv(alice_records, bob_records, out_dir, push_records=push_records)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
