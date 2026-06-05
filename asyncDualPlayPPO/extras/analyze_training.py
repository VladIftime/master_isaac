#!/usr/bin/env python3
"""
Unified training-log analyzer for all push/ASP experiments.

Scans a directory of slurm-*.out / slurm-*.txt files, traces job chains
across preemptions, deduplicates overlapping iterations, writes CSVs/TXTs,
and generates per-type training metric plots.

Training-type detection is based on the slurm output-filename suffix
(see the hpc/*.slurm scripts for the --output lines).  Content-based
fallback is used when the filename is ambiguous.

=== Supported training types and their log formats ===

1. Push-PPO baseline           → hpc/train_push.slurm, train_push_rel.slurm
   Suffix: -push.out  or  -push-rel.out
   Script: train_push.py
   Architecture: single-agent PPO, push-primitive macro-actions
   Key log lines:
     [Iter N] Loss=... | Val=... | Rew=... (EMA ...) | PosErr=... |
              RotErr=... | SR=... | RotSR=... | IK_fail=... |
              AvgPushes=... | Epi=... | BestSR=... | {rel|abs}
     [Episode] pushes=N  SUCCESS/fail  rew=...  goal=(...)  orient=(...)
               final=(...)  rot=(...)  err_pos=...m  err_rot=...rad
     [Push N] rew=... (pos=... rot=... dist=... bonus=...)
              pos_err=...  rot_err=...  at_goal=N/N
   Metrics plotted: loss, value loss, mean reward (EMA), SR, RotSR,
     position/rotation error, IK fail rate, best SR, episode rolling SR,
     reward histogram, final object positions scatter.

2. Push-primitive ASP          → hpc/train_push_asp.slurm
   Suffix: -push_asp.out
   Script: train_push_asp.py
   Architecture: dual-agent (Alice + Bob), push primitives, no ABC
   Key log lines:
     [Alice Update N]    Loss:... | Val:... | Rew:...
     [Bob Update N]      Loss:... | Val:... | Rew:... | SR:...
     [Alice] Entropy Coef:... (fixed)
     [Iter N] SR=... | Goals valid=... invalid=... |
              Bob succ=... fail=... | IK_fail=... |
              ObjLifted=... RobotTable=... Term=...
   Metrics plotted: Alice loss/value/reward, Bob loss/value/reward/SR,
     valid/invalid goals, IK fail rate, object-lifted count,
     robot-through-table count, terminated count.

3. cuRobo ASP                  → hpc/train_curobo.slurm
   Suffix: -curobo.out
   Script: train_curobo.py
   Architecture: dual-agent (Alice + Bob), cuRobo IK, ABC imitation
   Key log lines:
     [Alice Update N]  Loss:... | Val:... | Rew:...
     [Bob Update N]    Loss:... | Val:... | Rew:... | ABC:... | SR:... | ABCCoef:...
     [Iter N] SR=... | IK_fail=... | Goals valid=... invalid=... |
              Bob succ=... fail=... | Terminations: objects_off_table=...
              robot_through_table=... | ABC buf:... | ABC warm: YES/NO
     [BobSR]    PosSR=... RotSR=... PosErr=...m RotErr=...rad  Obj0_pos=... Obj0_rot=...  Obj1_pos=... Obj1_rot=... (n=N)
     [AliceRot] roll=...rad pitch=...rad yaw=...rad (n=N)
     [AliceDisp] N/N valid | avg 3D=...m avg XY=...m ...
   Metrics plotted: Bob SR/reward/loss/ABC, Alice reward, valid/invalid goals,
     IK fail & termination counts, Bob pos/rot error (from BobSR),
     Alice rotation change (from AliceRot).

Usage:
  python asyncDualPlayPPO/extras/analyze_training.py --log-dir logs/experiment
  python asyncDualPlayPPO/extras/analyze_training.py --log-dir logs/experiment -o analysis/ --merge-chains
"""

import re
import csv
import shutil
import bisect
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# Common patterns
# =========================================================================
CHAIN_RE = re.compile(r"chained next job:\s*(\d+)")
RESUME_RE = re.compile(r"Resuming from iteration\s+(\d+)")
GLOBAL_START_RE = re.compile(r"Global iteration start:\s*(\d+)")
JOB_ID_RE = re.compile(r"slurm-(\d+)(?:-(.*?))?\.(?:out|txt|log)")

# =========================================================================
# Push-PPO baseline patterns (train_push.py)
# =========================================================================
PUSH_ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+Loss=([-+\d.]+)[^\|]*\|\s+Val=([-+\d.]+)\s*\|\s+"
    r"Rew=([-+\d.]+)\s+\(EMA\s+([-+\d.]+)\)\s*\|\s+"
    r"PosErr=([-+\d.]+)\s*\|\s+RotErr=([-+\d.]+)\s*\|\s+"
    r"SR=([-+\d.]+)\s*\|\s+RotSR=([-+\d.]+)\s*\|\s+"
    r"IK_fail=([-+\d.]+)\s*\|\s+"
    r"AvgPushes=([^\s|]+)\s*\|\s+Epi=(\d+)\s*\|\s+"
    r"BestSR=([-+\d.]+)"
)

PUSH_EPISODE_RE = re.compile(
    r"\[Episode\]\s+pushes=(\d+)\s+(SUCCESS|fail)\s+"
    r"rew=([-+\d.]+)\s+"
    r"goal=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+orient=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
    r"final=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
    r"rot=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)\s+"
    r"err_pos=([-\d.]+)m\s+err_rot=([-\d.]+)rad"
)

# =========================================================================
# ASP patterns (curobo + push_asp) — shared with type-specific fallbacks
# =========================================================================
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
PUSH_ASP_BOB_RE = re.compile(
    r"\[Bob Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)\s*\|\s*Rew:\s*([-\d.]+)\s*\|\s*SR:\s*([-\d.]+)"
)

# cuRobo / legacy iter: IK_fail before Goals; optional ABC
CUROBO_ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+SR=([-\d.]+)"
    r"(?:\s*\|\s*IK_fail=([-\d.]+))?"
    r"\s*\|\s*Goals valid=(\d+) invalid=(\d+)"
    r"\s*\|\s*Bob succ=(\d+) fail=(\d+)"
    r"(?:.*?\|\s*ABC buf:\s*(\d+))?"
    r"(?:.*?ABC warm:\s*(YES|NO))?"
)
# Push-ASP iter: IK_fail after Bob succ/fail; plus stability counters
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

# =========================================================================
# Training-type detection
# =========================================================================

TRAIN_PUSH = "train_push"
TRAIN_PUSH_ASP = "train_push_asp"
TRAIN_CUROBO = "train_curobo"


def detect_training_type(file_path: Path, text: str) -> str:
    """Detect training type from filename suffix; fall back to content."""
    m = JOB_ID_RE.match(file_path.name)
    if m:
        suffix = (m.group(2) or "").lower()
        if suffix == "curobo" or suffix.startswith("curobo"):
            return TRAIN_CUROBO
        if suffix == "push_asp":
            return TRAIN_PUSH_ASP
        if suffix in ("push", "push-rel") or suffix.startswith("push-"):
            return TRAIN_PUSH

    # Content-based fallback: try PUSH_ITER_RE first (push baseline)
    if PUSH_ITER_RE.search(text):
        return TRAIN_PUSH
    # Try push-ASP ITER / Bob (no ABC, different ordering)
    if PUSH_ASP_ITER_RE.search(text) or PUSH_ASP_BOB_RE.search(text):
        return TRAIN_PUSH_ASP
    # Try cuRobo ITER (IK_fail before Goals, has ABC fields)
    if CUROBO_ITER_RE.search(text):
        return TRAIN_CUROBO
    # Try generic Alice/Bob (legacy or unknown ASP)
    if ALICE_RE.search(text) or ALICE_NO_ENT_RE.search(text):
        if BOB_RE.search(text):
            return TRAIN_CUROBO
        if PUSH_ASP_BOB_RE.search(text):
            return TRAIN_PUSH_ASP
        return TRAIN_PUSH_ASP
    return "unknown"


# =========================================================================
# Push-PPO baseline parser
# =========================================================================

def _parse_push_job(file_path: Path, text: str, job_id: int, suffix: str) -> dict:
    resume_iter = _extract_resume_iter(text)
    chain_next = _extract_chain_next(text)

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

    return {
        "path": file_path,
        "resume_iter": resume_iter,
        "chain_next": chain_next,
        "suffix": suffix,
        "train_type": TRAIN_PUSH,
        "push": push_updates,
        "alice": [],
        "bob": [],
    }


# =========================================================================
# ASP parser (curobo + push_asp)
# =========================================================================

def _parse_asp_job(file_path: Path, text: str, job_id: int, suffix: str) -> dict:
    resume_iter = _extract_resume_iter(text)
    chain_next = _extract_chain_next(text)

    # Detect sub-type to choose iter regex order
    sub_type = detect_training_type(file_path, text)

    # Parse [Iter N] summary lines
    iter_stats: dict[int, dict] = {}

    def _make_istat(n: int, sr: float, ik_fail: float | None,
                    valid: int, invalid: int, bob_succ: int, bob_fail: int,
                    abc_buf=None, abc_warm=None,
                    obj_lifted=None, robot_table=None, terminated=None,
                    pos: int = 0):
        return {
            "sr": sr, "ik_fail_rate": ik_fail, "valid": valid, "invalid": invalid,
            "bob_succ": bob_succ, "bob_fail": bob_fail,
            "abc_buf": abc_buf, "abc_warm": abc_warm,
            "obj_lifted": obj_lifted, "robot_table": robot_table,
            "terminated": terminated,
            "pos": pos,
            "avg_xy": None, "max_xy": None, "avg_z": None,
            "not_moved_frac": None,
            "alice_rot_roll": None, "alice_rot_pitch": None, "alice_rot_yaw": None,
            "bob_pos_sr": None, "bob_rot_sr": None,
            "bob_pos_err": None, "bob_rot_err": None,
        }

    iter_re = CUROBO_ITER_RE if sub_type == TRAIN_CUROBO else PUSH_ASP_ITER_RE
    for im in iter_re.finditer(text):
        n = int(im.group(1))
        if sub_type == TRAIN_CUROBO:
            iter_stats[n] = _make_istat(
                n, float(im.group(2)),
                float(im.group(3)) if im.group(3) is not None else None,
                int(im.group(4)), int(im.group(5)),
                int(im.group(6)), int(im.group(7)),
                int(im.group(8)) if im.group(8) is not None else None,
                (im.group(9) == "YES") if im.group(9) is not None else None,
                pos=im.start(),
            )
        else:
            iter_stats[n] = _make_istat(
                n, float(im.group(2)), float(im.group(7)),
                int(im.group(3)), int(im.group(4)),
                int(im.group(5)), int(im.group(6)),
                obj_lifted=int(im.group(8)),
                robot_table=int(im.group(9)),
                terminated=int(im.group(10)),
                pos=im.start(),
            )

    # Fallback: try the other regex
    if not iter_stats:
        fallback_re = PUSH_ASP_ITER_RE if sub_type == TRAIN_CUROBO else CUROBO_ITER_RE
        for im in fallback_re.finditer(text):
            n = int(im.group(1))
            if sub_type == TRAIN_PUSH_ASP:
                iter_stats[n] = _make_istat(
                    n, float(im.group(2)),
                    float(im.group(3)) if im.group(3) is not None else None,
                    int(im.group(4)), int(im.group(5)),
                    int(im.group(6)), int(im.group(7)),
                    pos=im.start(),
                )
            else:
                iter_stats[n] = _make_istat(
                    n, float(im.group(2)), float(im.group(7)),
                    int(im.group(3)), int(im.group(4)),
                    int(im.group(5)), int(im.group(6)),
                    pos=im.start(),
                )

    # Attach AliceDisp, AliceRot, BobSR data (curobo-specific extras)
    _attach_extras(text, iter_stats)

    # Parse Alice updates
    alice_updates = _parse_alice_updates(text, iter_stats)

    # Parse Bob updates
    bob_updates = _parse_bob_updates(text, iter_stats, sub_type)

    return {
        "path": file_path,
        "resume_iter": resume_iter,
        "chain_next": chain_next,
        "suffix": suffix,
        "train_type": sub_type,
        "alice": alice_updates,
        "bob": bob_updates,
        "push": [],
    }


def _attach_extras(text: str, iter_stats: dict):
    """Attach [AliceDisp], [AliceRot], [BobSR] data to preceding [Iter N]."""
    iter_positions = sorted((d["pos"], n) for n, d in iter_stats.items())

    for dm in ALICE_DISP_RE.finditer(text):
        dp = dm.start()
        n = _find_preceding_iter(iter_positions, dp)
        if n is not None and iter_stats[n]["avg_xy"] is None:
            iter_stats[n]["avg_xy"] = float(dm.group(3))
            iter_stats[n]["max_xy"] = float(dm.group(4))
            iter_stats[n]["avg_z"] = float(dm.group(5)) if dm.group(5) is not None else None
            if dm.group(6) is not None:
                total = int(dm.group(2))
                not_moved = int(dm.group(6))
                iter_stats[n]["not_moved_frac"] = not_moved / total if total > 0 else None

    for rm in ALICE_ROT_RE.finditer(text):
        rp = rm.start()
        n = _find_preceding_iter(iter_positions, rp)
        if n is not None and iter_stats[n]["alice_rot_roll"] is None:
            iter_stats[n]["alice_rot_roll"] = float(rm.group(1))
            iter_stats[n]["alice_rot_pitch"] = float(rm.group(2))
            iter_stats[n]["alice_rot_yaw"] = float(rm.group(3))

    for bm in BOB_SR_RE.finditer(text):
        bp = bm.start()
        n = _find_preceding_iter(iter_positions, bp)
        if n is not None and iter_stats[n]["bob_pos_sr"] is None:
            iter_stats[n]["bob_pos_sr"] = float(bm.group(1))
            iter_stats[n]["bob_rot_sr"] = float(bm.group(2))
            iter_stats[n]["bob_pos_err"] = float(bm.group(3))
            iter_stats[n]["bob_rot_err"] = float(bm.group(4))


def _find_preceding_iter(iter_positions: list, target_pos: int) -> int | None:
    n = None
    for ip, in_ in iter_positions:
        if ip < target_pos:
            n = in_
        else:
            break
    return n


def _parse_alice_updates(text: str, iter_stats: dict) -> list[dict]:
    alice_all = []
    for am in ALICE_RE.finditer(text):
        alice_all.append((am.start(), am.end(), am, True))
    for am in ALICE_NO_ENT_RE.finditer(text):
        alice_all.append((am.start(), am.end(), am, False))
    alice_all.sort(key=lambda x: x[0])

    alice_updates = []
    matched_iters = set()
    prev_update_end = 0

    for start, end, am, has_ent in alice_all:
        it = int(am.group(2)) if has_ent else int(am.group(1))
        if it in matched_iters:
            continue
        matched_iters.add(it)

        if iter_stats:
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
        else:
            v_count = inv_count = 0
            avg_xy = max_xy = avg_z = ik_fail_rate = not_moved_frac = None
            alice_rot_roll = alice_rot_pitch = alice_rot_yaw = None

        if has_ent:
            alice_updates.append({
                "local_iter": it,
                "entropy_coef": float(am.group(1)),
                "loss": float(am.group(3)), "val": float(am.group(4)), "rew": float(am.group(5)),
                "valid_goals": v_count, "invalid_goals": inv_count,
                "avg_xy": avg_xy, "max_xy": max_xy, "avg_z": avg_z,
                "ik_fail_rate": ik_fail_rate, "not_moved_frac": not_moved_frac,
                "alice_rot_roll": alice_rot_roll, "alice_rot_pitch": alice_rot_pitch,
                "alice_rot_yaw": alice_rot_yaw,
            })
        else:
            alice_updates.append({
                "local_iter": it, "entropy_coef": None,
                "loss": float(am.group(2)), "val": float(am.group(3)), "rew": float(am.group(4)),
                "valid_goals": v_count, "invalid_goals": inv_count,
                "avg_xy": avg_xy, "max_xy": max_xy, "avg_z": avg_z,
                "ik_fail_rate": ik_fail_rate, "not_moved_frac": not_moved_frac,
                "alice_rot_roll": alice_rot_roll, "alice_rot_pitch": alice_rot_pitch,
                "alice_rot_yaw": alice_rot_yaw,
            })
        prev_update_end = end

    alice_updates.sort(key=lambda x: x["local_iter"])
    return alice_updates


def _parse_bob_updates(text: str, iter_stats: dict, sub_type: str) -> list[dict]:
    bob_updates = []
    bob_re = BOB_RE if sub_type == TRAIN_CUROBO else PUSH_ASP_BOB_RE

    for bm in bob_re.finditer(text):
        it = int(bm.group(1))
        ist = iter_stats.get(it + 1, {}) if iter_stats else {}
        if sub_type == TRAIN_CUROBO:
            bob_updates.append({
                "local_iter": it,
                "loss": float(bm.group(2)), "val": float(bm.group(3)),
                "rew": float(bm.group(4)), "abc": float(bm.group(5)),
                "sr": float(bm.group(6)),
                "abc_coef": float(bm.group(7)) if bm.group(7) is not None else None,
                "pos_sr": ist.get("bob_pos_sr"), "rot_sr": ist.get("bob_rot_sr"),
                "pos_err": ist.get("bob_pos_err"), "rot_err": ist.get("bob_rot_err"),
                "obj0_pos_err": ist.get("bob_obj0_pos_err"),
                "obj0_rot_err": ist.get("bob_obj0_rot_err"),
                "obj1_pos_err": ist.get("bob_obj1_pos_err"),
                "obj1_rot_err": ist.get("bob_obj1_rot_err"),
                "ik_fail_rate": ist.get("ik_fail_rate"),
                "obj_lifted": ist.get("obj_lifted"),
                "robot_table": ist.get("robot_table"),
                "terminated_count": ist.get("terminated"),
            })
        else:
            bob_updates.append({
                "local_iter": it,
                "loss": float(bm.group(2)), "val": float(bm.group(3)),
                "rew": float(bm.group(4)), "abc": float("nan"),
                "sr": float(bm.group(5)), "abc_coef": None,
                "pos_sr": ist.get("bob_pos_sr"), "rot_sr": ist.get("bob_rot_sr"),
                "pos_err": ist.get("bob_pos_err"), "rot_err": ist.get("bob_rot_err"),
                "obj0_pos_err": ist.get("bob_obj0_pos_err"),
                "obj0_rot_err": ist.get("bob_obj0_rot_err"),
                "obj1_pos_err": ist.get("bob_obj1_pos_err"),
                "obj1_rot_err": ist.get("bob_obj1_rot_err"),
                "ik_fail_rate": ist.get("ik_fail_rate"),
                "obj_lifted": ist.get("obj_lifted"),
                "robot_table": ist.get("robot_table"),
                "terminated_count": ist.get("terminated"),
            })

    # Try fallback if primary regex didn't match
    if not bob_updates:
        fallback_bob = PUSH_ASP_BOB_RE if sub_type == TRAIN_CUROBO else BOB_RE
        for bm in fallback_bob.finditer(text):
            it = int(bm.group(1))
            ist = iter_stats.get(it + 1, {}) if iter_stats else {}
            if sub_type == TRAIN_CUROBO:
                bob_updates.append({
                    "local_iter": it,
                    "loss": float(bm.group(2)), "val": float(bm.group(3)),
                    "rew": float(bm.group(4)), "abc": float("nan"),
                    "sr": float(bm.group(5)), "abc_coef": None,
                    "pos_sr": ist.get("bob_pos_sr"), "rot_sr": ist.get("bob_rot_sr"),
                    "pos_err": ist.get("bob_pos_err"), "rot_err": ist.get("bob_rot_err"),
                    "obj0_pos_err": ist.get("bob_obj0_pos_err"),
                    "obj0_rot_err": ist.get("bob_obj0_rot_err"),
                    "obj1_pos_err": ist.get("bob_obj1_pos_err"),
                    "obj1_rot_err": ist.get("bob_obj1_rot_err"),
                    "ik_fail_rate": ist.get("ik_fail_rate"),
                    "obj_lifted": ist.get("obj_lifted"),
                    "robot_table": ist.get("robot_table"),
                    "terminated_count": ist.get("terminated"),
                })
            else:
                bob_updates.append({
                    "local_iter": it,
                    "loss": float(bm.group(2)), "val": float(bm.group(3)),
                    "rew": float(bm.group(4)), "abc": float(bm.group(5)),
                    "sr": float(bm.group(6)),
                    "abc_coef": float(bm.group(7)) if bm.group(7) is not None else None,
                    "pos_sr": ist.get("bob_pos_sr"), "rot_sr": ist.get("bob_rot_sr"),
                    "pos_err": ist.get("bob_pos_err"), "rot_err": ist.get("bob_rot_err"),
                    "obj0_pos_err": ist.get("bob_obj0_pos_err"),
                    "obj0_rot_err": ist.get("bob_obj0_rot_err"),
                    "obj1_pos_err": ist.get("bob_obj1_pos_err"),
                    "obj1_rot_err": ist.get("bob_obj1_rot_err"),
                    "ik_fail_rate": ist.get("ik_fail_rate"),
                    "obj_lifted": ist.get("obj_lifted"),
                    "robot_table": ist.get("robot_table"),
                    "terminated_count": ist.get("terminated"),
                })
        # Scale old logs if rewards are step-level
        if bob_updates:
            max_rew = max(u["rew"] for u in bob_updates)
            if max_rew < 0.1:
                for u in bob_updates:
                    u["rew"] *= 200.0

    bob_updates.sort(key=lambda x: x["local_iter"])
    return bob_updates


# =========================================================================
# Common helpers
# =========================================================================

def _extract_resume_iter(text: str) -> int | None:
    gm = GLOBAL_START_RE.search(text)
    if gm:
        return int(gm.group(1))
    rm = RESUME_RE.search(text)
    if rm:
        return int(rm.group(1))
    return None


def _extract_chain_next(text: str) -> int | None:
    cm = CHAIN_RE.search(text)
    if cm:
        return int(cm.group(1))
    return None


# =========================================================================
# Log-file scanning
# =========================================================================

def parse_logs(log_dir: Path) -> dict:
    """Scan log_dir for slurm files; parse each with type-specific parser."""
    jobs = {}
    for f in sorted(log_dir.rglob("slurm-*-*.out")) + sorted(log_dir.rglob("slurm-*-*.txt")) + sorted(log_dir.rglob("slurm-*-*.log")):
        if "chain_" in str(f.parent):
            continue
        m = JOB_ID_RE.match(f.name)
        if not m:
            continue
        job_id = int(m.group(1))
        suffix = (m.group(2) or "").lower()
        text = f.read_text(errors="replace")

        train_type = detect_training_type(f, text)
        if train_type == TRAIN_PUSH:
            job = _parse_push_job(f, text, job_id, suffix)
        elif train_type in (TRAIN_PUSH_ASP, TRAIN_CUROBO):
            job = _parse_asp_job(f, text, job_id, suffix)
        else:
            print(f"[WARN] Unknown training type in {f.name} — skipping.")
            continue

        jobs[job_id] = job
    return jobs


# =========================================================================
# Chain tracing
# =========================================================================

def trace_chains(jobs: dict) -> list[list[int]]:
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
                    for k in ("alice", "bob", "push"):
                        if jobs[jid][k]:
                            c_max = max(c_max, jobs[jid][k][-1]["local_iter"])
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
    flat = sorted(jid for chain in chains for jid in chain)
    return [flat]


# =========================================================================
# Global iter assignment
# =========================================================================

def assign_global_iters(chains: list[list[int]], jobs: dict):
    alice_records, bob_records, push_records = [], [], []

    for chain_idx, chain in enumerate(chains):
        seen_alice, seen_bob, seen_push = set(), set(), set()

        for job_id in chain:
            job = jobs[job_id]

            for upd in job.get("alice", []):
                g = upd["local_iter"]
                if g in seen_alice:
                    continue
                seen_alice.add(g)
                alice_records.append({
                    "chain": chain_idx, "job_id": job_id, "global_iter": g,
                    **{k: v for k, v in upd.items() if k != "local_iter"},
                })

            for upd in job.get("bob", []):
                g = upd["local_iter"]
                if g in seen_bob:
                    continue
                seen_bob.add(g)
                bob_records.append({
                    "chain": chain_idx, "job_id": job_id, "global_iter": g,
                    **{k: v for k, v in upd.items() if k != "local_iter"},
                })

            for upd in job.get("push", []):
                g = upd["local_iter"]
                if g in seen_push:
                    continue
                seen_push.add(g)
                push_records.append({
                    "chain": chain_idx, "job_id": job_id, "global_iter": g,
                    **{k: v for k, v in upd.items() if k != "local_iter"},
                })

    alice_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    bob_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    push_records.sort(key=lambda x: (x["chain"], x["global_iter"]))
    return alice_records, bob_records, push_records


# =========================================================================
# Smoothing helper
# =========================================================================

def smooth(arr, window=7):
    """Simple moving average with edge preservation."""
    arr = list(arr)
    if len(arr) < window:
        return np.array(arr)
    y = np.array(arr)
    kernel = np.ones(window) / window
    half = window // 2
    smoothed = np.convolve(y, kernel, mode="valid")
    return np.concatenate([y[:half], smoothed, y[-half:]])


# =========================================================================
# Plotting: Push-PPO baseline
# =========================================================================

def _plot_push(push_records: list[dict], out_dir: Path, title_suffix: str = "",
               log_paths: list[Path] = None, separate: bool = False):
    if not push_records:
        return

    push_colors = ["tab:green", "mediumseagreen", "darkgreen", "lightgreen"]
    all_cidx = sorted(set(r["chain"] for r in push_records))
    p_by_chain = [[r for r in push_records if r["chain"] == c] for c in all_cidx]
    p_labels = [f"Push C{c}" for c in all_cidx]

    def _dp(ax, key, ylabel, title):
        for records, label, color in zip(p_by_chain, p_labels, push_colors):
            pts = [(r["global_iter"], r[key]) for r in records if r.get(key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, smooth(list(ys)), color=color, label=label, linewidth=1.5)
        ax.set_title(title + title_suffix)
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    if separate:
        for key, yl, ti, fn in [
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
            _dp(ax, key, yl, ti); plt.tight_layout(); _save(fig, fn)

        has_ik = any(r.get("ik_fail_rate") is not None for r in push_records)
        if has_ik:
            fig, ax = plt.subplots(figsize=(10, 5))
            _dp(ax, "ik_fail_rate", "IK Fail Rate", "Push-PPO — IK Fail Rate")
            ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
            ax.legend(fontsize=8); plt.tight_layout(); _save(fig, "plot_ik_fail.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _dp(ax, "best_sr", "Best SR", "Push-PPO — Best Success Rate")
        plt.tight_layout(); _save(fig, "plot_best_sr.png")
        return

    # Combined overview grid
    from matplotlib.gridspec import GridSpec
    has_ik = any(r.get("ik_fail_rate") is not None for r in push_records)
    has_epi = any(r.get("episodes") is not None for r in push_records)
    n_rows = 4 + (1 if has_ik or has_epi else 0)

    fig = plt.figure(figsize=(18, 5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.38, wspace=0.30)
    _row = 0

    for col, (key, yl, ti) in enumerate([("loss", "Surrogate Loss", "Policy Loss"),
                                          ("val", "Value Loss", "Value Loss"),
                                          ("rew", "Mean Reward", "Mean Reward")]):
        ax = fig.add_subplot(gs[_row, col]); _dp(ax, key, yl, ti)
    _row += 1
    for col, (key, yl, ti) in enumerate([("sr", "Success Rate", "Success Rate"),
                                          ("rot_sr", "Rotation SR", "Rotation SR"),
                                          ("rew_ema", "EMA Reward", "EMA Reward")]):
        ax = fig.add_subplot(gs[_row, col]); _dp(ax, key, yl, ti)
    _row += 1
    for col, (key, yl, ti) in enumerate([("pos_err", "Position Error (m)", "Position Error"),
                                          ("rot_err", "Rotation Error (rad)", "Rotation Error"),
                                          ("best_sr", "Best SR", "Best Success Rate")]):
        ax = fig.add_subplot(gs[_row, col]); _dp(ax, key, yl, ti)
    _row += 1

    if has_ik or has_epi:
        ax1 = fig.add_subplot(gs[_row, 0])
        ax2 = fig.add_subplot(gs[_row, 1])
        ax3 = fig.add_subplot(gs[_row, 2])
        if has_ik:
            _dp(ax1, "ik_fail_rate", "IK Fail Rate", "IK Fail Rate")
            ax1.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
            ax1.legend(fontsize=8)
        else:
            ax1.axis("off")
        if has_epi:
            _dp(ax2, "episodes", "Episodes", "Episodes per Iteration")
        else:
            ax2.axis("off")
        ax3.axis("off")

    fig.suptitle(f"Push-PPO Training Overview{title_suffix}", fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.93, hspace=0.42, wspace=0.30)
    _save(fig, "plot_overview.png")

    # Episode-level plots
    if log_paths:
        episodes = []
        for lp in log_paths:
            if lp.exists():
                for m in PUSH_EPISODE_RE.finditer(lp.read_text(errors="replace")):
                    episodes.append({
                        "pushes": int(m.group(1)),
                        "success": m.group(2) == "SUCCESS",
                        "rew": float(m.group(3)),
                        "goal_x": float(m.group(4)), "goal_y": float(m.group(5)),
                        "final_x": float(m.group(10)), "final_y": float(m.group(11)),
                        "err_pos": float(m.group(16)), "err_rot": float(m.group(17)),
                    })
        if episodes:
            w = max(1, min(200, len(episodes) // 10))
            srs = [1.0 if e["success"] else 0.0 for e in episodes]
            if len(srs) >= w:
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                rolling = np.convolve(srs, np.ones(w) / w, mode="valid")
                ax2.plot(range(len(rolling)), rolling, color="blue", linewidth=1.5,
                         label=f"Episode SR (window={w})")
                ax2.set_title(f"Episode-Level Success Rate{title_suffix}")
                ax2.set_xlabel("Episode"); ax2.set_ylabel("Success Rate")
                ax2.legend(); ax2.grid(True, alpha=0.3)
                _save(fig2, "plot_episode_sr.png")

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            rews = [e["rew"] for e in episodes]
            ax3.hist(rews, bins=80, color="blue", alpha=0.7, edgecolor="white")
            ax3.axvline(np.mean(rews), color="red", linewidth=1.5, linestyle="--",
                        label=f"Mean = {np.mean(rews):+.2f}")
            ax3.set_title(f"Episode Reward Distribution{title_suffix} ({len(episodes)} eps)")
            ax3.set_xlabel("Episode Reward"); ax3.set_ylabel("Count")
            ax3.legend(); ax3.grid(True, alpha=0.3)
            _save(fig3, "plot_reward_histogram.png")

            fig4, ax4 = plt.subplots(figsize=(8, 8))
            sample = episodes[-10000:] if len(episodes) > 10000 else episodes
            ax4.scatter([e["final_x"] for e in sample], [e["final_y"] for e in sample],
                        s=1, alpha=0.3, color="red", label="Final obj pos")
            ax4.scatter([e["goal_x"] for e in sample], [e["goal_y"] for e in sample],
                        s=1, alpha=0.15, color="green", label="Goal pos")
            ax4.set_xlim(-0.6, 0.6); ax4.set_ylim(0.15, 0.75)
            ax4.set_aspect("equal")
            ax4.set_title(f"Object Final Positions{title_suffix} (last {len(sample)} eps)")
            ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)")
            ax4.legend(fontsize=8, markerscale=5); ax4.grid(True, alpha=0.3)
            _save(fig4, "plot_final_positions.png")


# =========================================================================
# Plotting: Push-ASP (Alice+Bob, no ABC, stability panels)
# =========================================================================

def _plot_push_asp(alice_records: list[dict], bob_records: list[dict],
                   out_dir: Path, title_suffix: str = "", separate: bool = False):
    _plot_asp_common(alice_records, bob_records, out_dir, title_suffix,
                     separate, mode="push_asp")


# =========================================================================
# Plotting: cuRobo ASP (Alice+Bob, ABC, BobSR, AliceRot, terminations)
# =========================================================================

def _plot_curobo(alice_records: list[dict], bob_records: list[dict],
                 out_dir: Path, title_suffix: str = "", separate: bool = False):
    _plot_asp_common(alice_records, bob_records, out_dir, title_suffix,
                     separate, mode="curobo")


def _plot_asp_common(alice_records: list[dict], bob_records: list[dict],
                     out_dir: Path, title_suffix: str = "",
                     separate: bool = False, mode: str = "push_asp"):
    if not alice_records and not bob_records:
        return

    alice_colors = ["tab:blue", "cornflowerblue", "navy", "steelblue"]
    bob_colors = ["tab:red", "tomato", "darkred", "salmon"]

    all_cidx = sorted(set([r["chain"] for r in alice_records] + [r["chain"] for r in bob_records]))
    a_by_chain = [[r for r in alice_records if r["chain"] == c] for c in all_cidx]
    b_by_chain = [[r for r in bob_records if r["chain"] == c] for c in all_cidx]
    a_labels = [f"Alice C{c}" for c in all_cidx]
    b_labels = [f"Bob C{c}" for c in all_cidx]

    def _draw(ax, records_list, labels, colors, key, linestyle="-"):
        for records, label, color in zip(records_list, labels, colors):
            pts = [(r["global_iter"], r[key]) for r in records if r.get(key) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, smooth(list(ys)), color=color, label=label, linewidth=1.5, linestyle=linestyle)

    def _fmt(ax, ylabel, title):
        ax.set_title(title + title_suffix)
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _fill_both(ax, key, ylabel, title):
        _draw(ax, a_by_chain, a_labels, alice_colors, key)
        _draw(ax, b_by_chain, b_labels, bob_colors, key)
        _fmt(ax, ylabel, title)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    # Data slices
    b_abc_by_chain = [[r for r in recs if r.get("abc") is not None] for recs in b_by_chain]
    has_abc = any(b_abc_by_chain)
    a_disp_chain = [[r for r in recs if r.get("avg_xy") is not None] for recs in a_by_chain]
    a_roll_chain = [[r for r in recs if r.get("alice_rot_roll") is not None] for recs in a_by_chain]
    a_pitch_chain = [[r for r in recs if r.get("alice_rot_pitch") is not None] for recs in a_by_chain]
    a_yaw_chain = [[r for r in recs if r.get("alice_rot_yaw") is not None] for recs in a_by_chain]
    has_alice_rot = any(a_roll_chain)
    b_pos_sr_c = [[r for r in recs if r.get("pos_sr") is not None] for recs in b_by_chain]
    b_rot_sr_c = [[r for r in recs if r.get("rot_sr") is not None] for recs in b_by_chain]
    b_pos_err_c = [[r for r in recs if r.get("pos_err") is not None] for recs in b_by_chain]
    b_rot_err_c = [[r for r in recs if r.get("rot_err") is not None] for recs in b_by_chain]
    has_bob_err = any(b_pos_err_c)

    b_ik_c = [[r for r in recs if r.get("ik_fail_rate") is not None] for recs in b_by_chain]
    b_obj_c = [[r for r in recs if r.get("obj_lifted") is not None] for recs in b_by_chain]
    b_rt_c = [[r for r in recs if r.get("robot_table") is not None] for recs in b_by_chain]
    b_term_c = [[r for r in recs if r.get("terminated_count") is not None] for recs in b_by_chain]
    has_stability = any(b_obj_c)

    # --- Separate mode ---
    if separate:
        fig, ax = plt.subplots(figsize=(10, 5))
        _draw(ax, a_by_chain, a_labels, alice_colors, "loss")
        _fmt(ax, "Surrogate Loss", "Policy Loss — Alice")
        plt.tight_layout(); _save(fig, "plot_loss.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _fill_both(ax, "val", "Value Loss", "Value Loss — Alice & Bob")
        plt.tight_layout(); _save(fig, "plot_value_loss.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _fill_both(ax, "rew", "Episode Reward", "Episode Reward — Alice & Bob")
        plt.tight_layout(); _save(fig, "plot_reward.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _draw(ax, b_by_chain, b_labels, bob_colors, "sr")
        _fmt(ax, "Success Rate", "Bob — Success Rate")
        plt.tight_layout(); _save(fig, "plot_bob_sr.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        _draw(ax, a_by_chain, [f"Valid {l}" for l in a_labels], alice_colors, "valid_goals")
        _draw(ax, a_by_chain, [f"Invalid {l}" for l in a_labels], ["tab:orange"] * len(a_labels),
              "invalid_goals", linestyle="--")
        _fmt(ax, "Goals", "Alice — Valid & Invalid Goals")
        plt.tight_layout(); _save(fig, "plot_goals.png")

        if has_abc:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw(ax, b_by_chain, b_labels, bob_colors, "abc")
            _fmt(ax, "ABC Loss", "Bob — ABC (Imitation) Loss")
            plt.tight_layout(); _save(fig, "plot_abc_loss.png")

        if mode == "curobo" and has_bob_err:
            fig, axes_err = plt.subplots(1, 2, figsize=(14, 5))
            _draw(axes_err[0], b_pos_err_c, [f"Global {l}" for l in b_labels], bob_colors, "pos_err")
            _draw(axes_err[0], b_pos_err_c, [f"Obj0 {l}" for l in b_labels], ["tab:orange"] * len(b_labels),
                  "obj0_pos_err", linestyle="--")
            _draw(axes_err[0], b_pos_err_c, [f"Obj1 {l}" for l in b_labels], ["tab:purple"] * len(b_labels),
                  "obj1_pos_err", linestyle=":")
            _fmt(axes_err[0], "Position Error (m)", "Bob — Position Error")
            _draw(axes_err[1], b_rot_err_c, [f"Global {l}" for l in b_labels], bob_colors, "rot_err")
            _draw(axes_err[1], b_rot_err_c, [f"Obj0 {l}" for l in b_labels], ["tab:orange"] * len(b_labels),
                  "obj0_rot_err", linestyle="--")
            _draw(axes_err[1], b_rot_err_c, [f"Obj1 {l}" for l in b_labels], ["tab:purple"] * len(b_labels),
                  "obj1_rot_err", linestyle=":")
            _fmt(axes_err[1], "Rotation Error (rad)", "Bob — Rotation Error")
            plt.tight_layout(); _save(fig, "plot_bob_errors.png")

        if has_alice_rot:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw(ax, a_roll_chain, [f"Roll {l}" for l in a_labels], alice_colors, "alice_rot_roll")
            _draw(ax, a_pitch_chain, [f"Pitch {l}" for l in a_labels], ["tab:orange"] * len(a_labels),
                  "alice_rot_pitch", linestyle="--")
            _draw(ax, a_yaw_chain, [f"Yaw {l}" for l in a_labels], ["tab:green"] * len(a_labels),
                  "alice_rot_yaw", linestyle=":")
            _fmt(ax, "Rotation (rad)", "Alice — Goal Rotation Change")
            plt.tight_layout(); _save(fig, "plot_alice_rotation.png")

        if has_stability:
            for key, yl, ti, fn in [
                ("obj_lifted", "Count", "Obj Lifted", "plot_obj_lifted.png"),
                ("robot_table", "Count", "Robot Through Table", "plot_robot_table.png"),
                ("terminated_count", "Count", "Terminated", "plot_terminated.png"),
            ]:
                chains_k = [[r for r in recs if r.get(key) is not None] for recs in b_by_chain]
                if any(chains_k):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    _draw(ax, chains_k, b_labels, bob_colors, key)
                    _fmt(ax, yl, ti)
                    plt.tight_layout(); _save(fig, fn)

            if any(b_ik_c):
                fig, ax = plt.subplots(figsize=(10, 5))
                _draw(ax, b_ik_c, b_labels, bob_colors, "ik_fail_rate")
                _fmt(ax, "IK Fail Rate", "IK Fail Rate")
                ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold")
                ax.legend(fontsize=8)
                plt.tight_layout(); _save(fig, "plot_ik_fail.png")
        return

    # --- Combined overview ---
    from matplotlib.gridspec import GridSpec
    import math

    panels = []

    panels.append(lambda ax: (_draw(ax, a_by_chain, a_labels, alice_colors, "loss"),
                               _fmt(ax, "Loss", "Policy Loss — Alice")))
    panels.append(lambda ax: (_fill_both(ax, "val", "Value Loss", "Value Loss — Alice & Bob"),))
    panels.append(lambda ax: (_fill_both(ax, "rew", "Reward", "Episode Reward — Alice & Bob"),))
    panels.append(lambda ax: (_draw(ax, b_by_chain, b_labels, bob_colors, "sr"),
                               _fmt(ax, "Success Rate", "Bob — Success Rate")))
    panels.append(lambda ax: (_draw(ax, a_by_chain, [f"Valid {l}" for l in a_labels], alice_colors, "valid_goals"),
                               _draw(ax, a_by_chain, [f"Invalid {l}" for l in a_labels],
                                     ["tab:orange"] * len(a_labels), "invalid_goals", linestyle="--"),
                               _fmt(ax, "Goals", "Alice — Valid & Invalid Goals")))

    if has_abc:
        panels.append(lambda ax: (_draw(ax, b_by_chain, b_labels, bob_colors, "abc"),
                                   _fmt(ax, "ABC Loss", "Bob — ABC (Imitation) Loss")))

    if has_alice_rot:
        panels.append(lambda ax: (
            _draw(ax, a_roll_chain, [f"Roll {l}" for l in a_labels], alice_colors, "alice_rot_roll"),
            _draw(ax, a_pitch_chain, [f"Pitch {l}" for l in a_labels], ["tab:orange"] * len(a_labels),
                  "alice_rot_pitch", linestyle="--"),
            _draw(ax, a_yaw_chain, [f"Yaw {l}" for l in a_labels], ["tab:green"] * len(a_labels),
                  "alice_rot_yaw", linestyle=":"),
            _fmt(ax, "Rotation (rad)", "Alice — Rot Change")))

    if mode == "curobo" and has_bob_err:
        panels.append(lambda ax: (
            _draw(ax, b_pos_err_c, [f"Global {l}" for l in b_labels], bob_colors, "pos_err"),
            _draw(ax, b_pos_err_c, [f"Obj0 {l}" for l in b_labels], ["tab:orange"] * len(b_labels),
                  "obj0_pos_err", linestyle="--"),
            _draw(ax, b_pos_err_c, [f"Obj1 {l}" for l in b_labels], ["tab:purple"] * len(b_labels),
                  "obj1_pos_err", linestyle=":"),
            _fmt(ax, "Position Error (m)", "Bob — Position Error")))
        panels.append(lambda ax: (
            _draw(ax, b_rot_err_c, [f"Global {l}" for l in b_labels], bob_colors, "rot_err"),
            _draw(ax, b_rot_err_c, [f"Obj0 {l}" for l in b_labels], ["tab:orange"] * len(b_labels),
                  "obj0_rot_err", linestyle="--"),
            _draw(ax, b_rot_err_c, [f"Obj1 {l}" for l in b_labels], ["tab:purple"] * len(b_labels),
                  "obj1_rot_err", linestyle=":"),
            _fmt(ax, "Rotation Error (rad)", "Bob — Rotation Error")))

    if has_stability:
        if any(b_ik_c):
            panels.append(lambda ax: (
                _draw(ax, b_ik_c, b_labels, bob_colors, "ik_fail_rate"),
                _fmt(ax, "IK Fail Rate", "IK Fail Rate"),
                ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="5% threshold"),
                ax.legend(fontsize=8)))
        panels.append(lambda ax: (_draw(ax, b_obj_c, b_labels, bob_colors, "obj_lifted"),
                                   _fmt(ax, "Count", "Obj Lifted Count")))
        panels.append(lambda ax: (_draw(ax, b_rt_c, b_labels, bob_colors, "robot_table"),
                                   _fmt(ax, "Count", "Robot Through Table")))
        panels.append(lambda ax: (_draw(ax, b_term_c, b_labels, bob_colors, "terminated_count"),
                                   _fmt(ax, "Count", "Terminated Count")))

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

    mode_label = {"push_asp": "Push-ASP", "curobo": "cuRobo ASP"}.get(mode, mode)
    fig.suptitle(f"{mode_label} Training Overview{title_suffix}", fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.93, hspace=0.42, wspace=0.30)
    _save(fig, "plot_overview.png")


# =========================================================================
# CSV / TXT output
# =========================================================================

TRAINING_CSV_FIELDS = [
    "agent", "chain", "job_id", "global_iter",
    "loss", "val", "rew", "rew_ema", "entropy_coef",
    "abc", "abc_coef", "sr",
    "valid_goals", "invalid_goals",
    "avg_xy", "max_xy", "avg_z",
    "ik_fail_rate", "not_moved_frac",
    "alice_rot_roll", "alice_rot_pitch", "alice_rot_yaw",
    "pos_sr", "rot_sr", "pos_err", "rot_err",
    "avg_pushes", "episodes", "best_sr",
    "obj0_pos_err", "obj0_rot_err", "obj1_pos_err", "obj1_rot_err",
    "obj_lifted", "robot_table", "terminated_count",
]


def _make_empty_row():
    return {k: "" for k in TRAINING_CSV_FIELDS}


def write_training_csv(alice_records, bob_records, push_records, out_dir: Path):
    """Write combined training_updates.csv for all agents."""
    if not alice_records and not bob_records and not push_records:
        return
    out_path = out_dir / "training_updates.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_CSV_FIELDS)
        w.writeheader()
        for r in alice_records:
            row = _make_empty_row()
            row["agent"] = "alice"
            for k in r:
                if k in TRAINING_CSV_FIELDS:
                    row[k] = "" if r[k] is None else r[k]
            w.writerow(row)
        for r in bob_records:
            row = _make_empty_row()
            row["agent"] = "bob"
            for k in r:
                if k in TRAINING_CSV_FIELDS:
                    row[k] = "" if r[k] is None else r[k]
            w.writerow(row)
        for r in push_records:
            row = _make_empty_row()
            row["agent"] = "push"
            for k in r:
                if k in TRAINING_CSV_FIELDS:
                    row[k] = "" if r[k] is None else r[k]
            w.writerow(row)
    print(f"[INFO] Wrote {out_path} ({len(alice_records) + len(bob_records) + len(push_records)} rows)")


def write_raw_csv(chain_idx: int, chain: list[int], jobs: dict, out_dir: Path):
    """Write raw parsed local-iter records."""
    raw_fields = ["agent", "chain", "job_id", "local_iter"] + [f for f in TRAINING_CSV_FIELDS
                  if f not in ("agent", "chain", "job_id", "global_iter")]
    out_path = out_dir / "raw_parsed.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for job_id in chain:
            job = jobs[job_id]
            for upd in job.get("alice", []):
                row = {k: "" for k in raw_fields}
                row.update({"agent": "alice", "chain": chain_idx, "job_id": job_id,
                            "local_iter": upd["local_iter"]})
                for k, v in upd.items():
                    if k in raw_fields:
                        row[k] = "" if v is None else v
                w.writerow(row)
            for upd in job.get("bob", []):
                row = {k: "" for k in raw_fields}
                row.update({"agent": "bob", "chain": chain_idx, "job_id": job_id,
                            "local_iter": upd["local_iter"]})
                for k, v in upd.items():
                    if k in raw_fields:
                        row[k] = "" if v is None else v
                w.writerow(row)
            for upd in job.get("push", []):
                row = {k: "" for k in raw_fields}
                row.update({"agent": "push", "chain": chain_idx, "job_id": job_id,
                            "local_iter": upd["local_iter"]})
                for k, v in upd.items():
                    if k in raw_fields:
                        row[k] = "" if v is None else v
                w.writerow(row)
    print(f"[INFO] Wrote {out_path}")


def write_raw_logs(chain: list[int], jobs: dict, out_dir: Path):
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


def write_summary_txt(chain_idx: int, chain: list[int], a_c: list[dict], b_c: list[dict],
                      p_c: list[dict], out_dir: Path, train_type: str):
    out_path = out_dir / "training_updates.txt"
    type_names = {TRAIN_PUSH: "Push-PPO Baseline", TRAIN_PUSH_ASP: "Push-ASP", TRAIN_CUROBO: "cuRobo ASP"}
    with open(out_path, "w") as f:
        f.write(f"=== {type_names.get(train_type, train_type)} SUMMARY (Chain {chain_idx}) ===\n\n")
        f.write(f"Jobs in chain: {' → '.join(str(j) for j in chain)}\n\n")

        if p_c:
            for pr in p_c:
                f.write(f"  Iter {pr['global_iter']:5d} | "
                        f"Loss={pr['loss']:+.4f}  Val={pr['val']:.4f}  "
                        f"Rew={pr['rew']:+.4f} (EMA {pr['rew_ema']:+.4f})  "
                        f"SR={pr['sr']:.4f}  RotSR={pr['rot_sr']:.4f}  "
                        f"PosErr={pr['pos_err']:.4f}  RotErr={pr['rot_err']:.4f}  "
                        f"BestSR={pr['best_sr']:.4f}\n")
        else:
            bob_by_iter = {r["global_iter"]: r for r in b_c}
            for ar in a_c:
                g = ar["global_iter"]
                ent = ar.get("entropy_coef")
                ent_s = f"  Ent={ent:.4f}" if ent is not None else ""
                br = bob_by_iter.get(g)
                bob_s = (f"[Bob] Loss={br['loss']:+.4f}  Val={br['val']:.4f}  "
                         f"Rew={br['rew']:.4f}  SR={br['sr']:.4f}") if br else "[Bob] —"
                f.write(f"  Iter {g:5d} | "
                        f"[Alice] Loss={ar['loss']:+.4f}  Val={ar['val']:.4f}  "
                        f"Rew={ar['rew']:.4f}  "
                        f"Valid={ar.get('valid_goals', 0)}  Invalid={ar.get('invalid_goals', 0)}{ent_s}  ||  "
                        f"{bob_s}\n")
            alice_iters = {r["global_iter"] for r in a_c}
            for br in b_c:
                if br["global_iter"] not in alice_iters:
                    g = br["global_iter"]
                    f.write(f"  Iter {g:5d} | [Alice] —  || "
                            f"[Bob] Loss={br['loss']:+.4f}  Val={br['val']:.4f}  "
                            f"Rew={br['rew']:.4f}  SR={br['sr']:.4f}\n")
    print(f"[INFO] Wrote {out_path}")


# =========================================================================
# Summary printer
# =========================================================================

def print_summary(alice_records, bob_records, push_records, train_type: str, compact: bool = False):
    type_names = {TRAIN_PUSH: "Push-PPO Baseline", TRAIN_PUSH_ASP: "Push-ASP",
                  TRAIN_CUROBO: "cuRobo ASP"}
    label = f"  {type_names.get(train_type, train_type)}"
    if compact:
        indent = "    "
    else:
        indent = ""
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

    if push_records:
        pr = push_records
        print(f"{indent}Total iters: {len(pr)}  ({pr[0]['global_iter']} → {pr[-1]['global_iter']})")
        last = pr[-1]
        print(f"{indent}Last Rew: {last['rew']:+.3f} (EMA {last['rew_ema']:+.3f})  "
              f"SR={last['sr']:.4f} RotSR={last['rot_sr']:.4f}  "
              f"PosErr={last['pos_err']:.4f}m RotErr={last['rot_err']:.4f}rad  "
              f"BestSR={last['best_sr']:.4f}")
        rews = [r["rew"] for r in pr]
        srs = [r["sr"] for r in pr]
        print(f"{indent}Mean Rew: {np.mean(rews):+.4f}  Mean SR: {np.mean(srs):.4f}  Max SR: {max(srs):.4f}")

    elif alice_records or bob_records:
        if alice_records:
            print(f"{indent}Alice updates: {len(alice_records)}  "
                  f"({alice_records[0]['global_iter']} → {alice_records[-1]['global_iter']})")
            la = alice_records[-1]
            print(f"{indent}Alice last: Rew={la['rew']:+.3f}  Valid={la.get('valid_goals',0)}  "
                  f"Invalid={la.get('invalid_goals',0)}")
        if bob_records:
            print(f"{indent}Bob updates:   {len(bob_records)}  "
                  f"({bob_records[0]['global_iter']} → {bob_records[-1]['global_iter']})")
            lb = bob_records[-1]
            print(f"{indent}Bob last:   Loss={lb['loss']:+.4f}  Rew={lb['rew']:+.3f}  SR={lb['sr']:.4f}")
            srs = [r["sr"] for r in bob_records]
            print(f"{indent}Bob mean SR: {np.mean(srs):.4f}  Max SR: {max(srs):.4f}")

    if not compact:
        print(f"{'='*60}\n")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and stitch training logs across SLURM job chains "
                    "(push-PPO, push-ASP, cuRobo ASP)."
    )
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Directory containing slurm-*.out / slurm-*.txt files.")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory (defaults to --log-dir).")
    parser.add_argument("--merge-chains", action="store_true", default=False,
                        help="Collapse all chains into one, ordered by job ID.")
    parser.add_argument("--separate-plots", action="store_true", default=False,
                        help="Save one PNG per metric instead of a combined overview.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir
    if not log_dir.exists():
        parser.error(f"--log-dir does not exist: {log_dir}")

    print(f"[INFO] Scanning {log_dir} for slurm log files ...")
    jobs = parse_logs(log_dir)
    print(f"[INFO] Found {len(jobs)} job log file(s)")

    # Determine dominant training type
    type_counts = {}
    for j in jobs.values():
        t = j.get("train_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    train_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"
    type_names = {TRAIN_PUSH: "Push-PPO Baseline", TRAIN_PUSH_ASP: "Push-ASP",
                  TRAIN_CUROBO: "cuRobo ASP"}
    print(f"[INFO] Dominant training type: {type_names.get(train_type, train_type)}")
    for t, c in type_counts.items():
        print(f"       {t}: {c} file(s)")

    chains = trace_chains(jobs)
    print(f"[INFO] Found {len(chains)} chain(s):")
    for i, ch in enumerate(chains):
        print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    if args.merge_chains and len(chains) > 1:
        chains = merge_all_chains(chains)
        print(f"[INFO] --merge-chains: collapsed to {len(chains)} chain(s)")

    alice_records, bob_records, push_records = assign_global_iters(chains, jobs)
    print(f"[INFO] Records — Alice: {len(alice_records)}, Bob: {len(bob_records)}, "
          f"Push: {len(push_records)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ch in enumerate(chains):
        # Determine this chain's training type
        ch_types = {}
        for jid in ch:
            t = jobs[jid].get("train_type", "unknown")
            ch_types[t] = ch_types.get(t, 0) + 1
        ch_type = max(ch_types, key=ch_types.get) if ch_types else train_type

        chain_dir = out_dir / f"chain_{i}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing Chain {i} → {chain_dir}  "
              f"({type_names.get(ch_type, ch_type)})")

        for job_id in ch:
            src = jobs[job_id]["path"]
            dest = chain_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)

        a_c = [r for r in alice_records if r["chain"] == i]
        b_c = [r for r in bob_records if r["chain"] == i]
        p_c = [r for r in push_records if r["chain"] == i]

        write_raw_csv(i, ch, jobs, chain_dir)
        write_raw_logs(ch, jobs, chain_dir)
        write_summary_txt(i, ch, a_c, b_c, p_c, chain_dir, ch_type)
        write_training_csv(a_c, b_c, p_c, chain_dir)

        plots_dir = chain_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        log_paths = [jobs[jid]["path"] for jid in ch]
        suffix = f" (Chain {i})"

        if ch_type == TRAIN_PUSH:
            _plot_push(p_c, plots_dir, suffix, log_paths, args.separate_plots)
        elif ch_type == TRAIN_PUSH_ASP:
            _plot_push_asp(a_c, b_c, plots_dir, suffix, args.separate_plots)
        elif ch_type == TRAIN_CUROBO:
            _plot_curobo(a_c, b_c, plots_dir, suffix, args.separate_plots)

    # Per-chain summary
    for i, ch in enumerate(chains):
        ch_types = {}
        for jid in ch:
            t = jobs[jid].get("train_type", "unknown")
            ch_types[t] = ch_types.get(t, 0) + 1
        ch_type = max(ch_types, key=ch_types.get) if ch_types else train_type

        a_c = [r for r in alice_records if r["chain"] == i]
        b_c = [r for r in bob_records if r["chain"] == i]
        p_c = [r for r in push_records if r["chain"] == i]

        if p_c or a_c or b_c:
            print(f"\n  Chain {i} ({type_names.get(ch_type, ch_type)}):", flush=True)
            print_summary(a_c, b_c, p_c, ch_type, compact=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
