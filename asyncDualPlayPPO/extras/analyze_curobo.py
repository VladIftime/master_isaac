#!/usr/bin/env python3
"""
Analyze SLURM / log output from train_curobo.py (ASP).

Parses [Iter N], [Bob Update N], [Alice Update N], [BobSR], and [AliceRot]
lines from *.out files, writes CSVs, and generates training metric plots.

Usage:
  python analyze_curobo.py --log slurm-*-curobo.out
  python analyze_curobo.py --log slurm-*-curobo.out -o analysis/
"""

import re
import csv
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Iter line ──────────────────────────────────────────────────────────
# [Iter 1] SR=0.00 | IK_fail=0.001 | Goals valid=222 invalid=573 | Bob succ=0 fail=8 | Terminations: objects_off_table=273  robot_through_table=268 | ABC buf: 8 | ABC warm: YES
ITER_RE = re.compile(
    r"\[Iter\s+(\d+)\]\s+"
    r"SR=([-\d.]+)\s*\|\s*"
    r"IK_fail=([-\d.]+)\s*\|\s*"
    r"Goals\s+valid=(\d+)\s+invalid=(\d+)\s*\|\s*"
    r"Bob\s+succ=(\d+)\s+fail=(\d+)\s*\|\s*"
    r"Terminations:\s+objects_off_table=(\d+)\s+robot_through_table=(\d+)\s*\|\s*"
    r"ABC\s+buf:\s+(\d+)\s*\|\s*"
    r"ABC\s+warm:\s+(\w+)"
)

# ── Bob Update ─────────────────────────────────────────────────────────
# [Bob Update 0] Loss: -0.1684 | Val: 0.0159 | Rew: -0.0001 | ABC: -0.9999 | SR: 0.0000 | ABCCoef: 0.5000
BOB_UPDATE_RE = re.compile(
    r"\[Bob\s+Update\s+(\d+)\]\s+"
    r"Loss:\s+([-+\d.]+)\s*\|\s*"
    r"Val:\s+([-+\d.]+)\s*\|\s*"
    r"Rew:\s+([-+\d.]+)\s*\|\s*"
    r"ABC:\s+([-+\d.]+)\s*\|\s*"
    r"SR:\s+([-+\d.]+)\s*\|\s*"
    r"ABCCoef:\s+([-+\d.]+)"
)

# ── Alice Update ───────────────────────────────────────────────────────
# [Alice Update 0] Loss: -0.0051 | Val: 0.3226 | Rew: -2.5277
ALICE_UPDATE_RE = re.compile(
    r"\[Alice\s+Update\s+(\d+)\]\s+"
    r"Loss:\s+([-+\d.]+)\s*\|\s*"
    r"Val:\s+([-+\d.]+)\s*\|\s*"
    r"Rew:\s+([-+\d.]+)"
)

# ── BobSR ──────────────────────────────────────────────────────────────
# [BobSR] PosSR=0.0000  RotSR=0.0000  PosErr=0.2875m  RotErr=1.9250rad  Obj0_pos=0.2514 Obj0_rot=1.4762  Obj1_pos=0.1167 Obj1_rot=1.4566  (n=8)
BOB_SR_RE = re.compile(
    r"\[BobSR\]\s+"
    r"PosSR=([-\d.]+)\s+"
    r"RotSR=([-\d.]+)\s+"
    r"PosErr=([-\d.]+)m\s+"
    r"RotErr=([-\d.]+)rad\s+"
    r"Obj0_pos=([-\d.]+)\s+Obj0_rot=([-\d.]+)\s+"
    r"Obj1_pos=([-\d.]+)\s+Obj1_rot=([-\d.]+)\s+"
    r"\(n=(\d+)\)"
)

# ── AliceRot ───────────────────────────────────────────────────────────
# [AliceRot] roll=0.1508rad  pitch=0.0408rad  yaw=0.4282rad  (n=1347)
ALICE_ROT_RE = re.compile(
    r"\[AliceRot\]\s+"
    r"roll=([-\d.]+)rad\s+"
    r"pitch=([-\d.]+)rad\s+"
    r"yaw=([-\d.]+)rad\s+"
    r"\(n=(\d+)\)"
)

# ── [ALICE END] / [BOB END] ────────────────────────────────────────────
ALICE_END_RE = re.compile(
    r"\[ALICE END\]\s+"
    r"Env=(\d+)\s+"
    r"Start=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
    r"\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
    r"→ Goal=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
    r"\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
    r"Disp3D=([-\d.]+)m"
)


def parse_log(path: Path) -> dict:
    """Parse a curobo SLURM log. Returns dict of parsed record lists."""
    text = path.read_text(errors="replace")

    iters: dict[int, dict] = {}
    for m in ITER_RE.finditer(text):
        i = int(m.group(1))
        if i not in iters:
            iters[i] = {
                "iter": i,
                "SR": float(m.group(2)),
                "IK_fail": float(m.group(3)),
                "goals_valid": int(m.group(4)),
                "goals_invalid": int(m.group(5)),
                "bob_succ": int(m.group(6)),
                "bob_fail": int(m.group(7)),
                "obj_off_table": int(m.group(8)),
                "robot_through_table": int(m.group(9)),
                "abc_buf": int(m.group(10)),
                "abc_warm": 1 if m.group(11) == "YES" else 0,
            }

    bob_updates = []
    for m in BOB_UPDATE_RE.finditer(text):
        bob_updates.append({
            "iter": int(m.group(1)),
            "loss": float(m.group(2)),
            "val": float(m.group(3)),
            "rew": float(m.group(4)),
            "abc": float(m.group(5)),
            "sr": float(m.group(6)),
            "abc_coef": float(m.group(7)),
        })

    alice_updates = []
    for m in ALICE_UPDATE_RE.finditer(text):
        alice_updates.append({
            "iter": int(m.group(1)),
            "loss": float(m.group(2)),
            "val": float(m.group(3)),
            "rew": float(m.group(4)),
        })

    bob_sr_records = []
    for m in BOB_SR_RE.finditer(text):
        bob_sr_records.append({
            "pos_sr": float(m.group(1)),
            "rot_sr": float(m.group(2)),
            "pos_err": float(m.group(3)),
            "rot_err": float(m.group(4)),
            "obj0_pos_err": float(m.group(5)),
            "obj0_rot_err": float(m.group(6)),
            "obj1_pos_err": float(m.group(7)),
            "obj1_rot_err": float(m.group(8)),
            "n": int(m.group(9)),
        })

    alice_rot_records = []
    for m in ALICE_ROT_RE.finditer(text):
        alice_rot_records.append({
            "roll": float(m.group(1)),
            "pitch": float(m.group(2)),
            "yaw": float(m.group(3)),
            "n": int(m.group(4)),
        })

    return {
        "iters": sorted(iters.values(), key=lambda r: r["iter"]),
        "bob_updates": bob_updates,
        "alice_updates": alice_updates,
        "bob_sr": bob_sr_records,
        "alice_rot": alice_rot_records,
    }


def merge_paths(paths: list[Path]) -> dict:
    """Merge multiple log files; deduplicate iters by number."""
    all_records: dict[str, list] = {
        "iters": [], "bob_updates": [], "alice_updates": [],
        "bob_sr": [], "alice_rot": [],
    }
    seen_i: set[int] = set()
    for p in paths:
        data = parse_log(p)
        for r in data["iters"]:
            if r["iter"] not in seen_i:
                seen_i.add(r["iter"])
                all_records["iters"].append(r)
        for k in ["bob_updates", "alice_updates", "bob_sr", "alice_rot"]:
            all_records[k].extend(data[k])
    all_records["iters"].sort(key=lambda r: r["iter"])
    return all_records


def write_csv(data: dict, out_dir: Path):
    for key, records in data.items():
        if not records:
            continue
        path = out_dir / f"curobo_{key}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"[INFO] Wrote {path} ({len(records)} rows)")


# ── Plotting ───────────────────────────────────────────────────────────

def smooth(arr, window=7):
    arr = list(arr)
    if len(arr) < window:
        return np.array(arr)
    y = np.array(arr)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_metrics(data: dict, out_dir: Path):
    iters = data["iters"]
    bobs = data["bob_updates"]
    alices = data["alice_updates"]
    bob_sr_recs = data["bob_sr"]
    alice_rot_recs = data["alice_rot"]

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {p}")

    def _axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # ── Overview plot (3×3) ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("ASP Curobo Training Overview", fontsize=14, fontweight="bold")

    xs_i = [r["iter"] for r in iters]
    xs_b = [r["iter"] for r in bobs]
    xs_a = [r["iter"] for r in alices]

    # Bob SR
    ax = axes[0, 0]
    bsr = [r["sr"] for r in bobs]
    ax.plot(xs_b, smooth(bsr), color="blue", linewidth=1.5, label="Bob SR")
    _axis(ax, "Bob Success Rate", "SR")
    ax.legend(fontsize=8)

    # Bob Reward
    ax = axes[0, 1]
    brew = [r["rew"] for r in bobs]
    ax.plot(xs_b, smooth(brew), color="green", linewidth=1.5, label="Bob Rew")
    _axis(ax, "Bob Mean Reward", "Reward")
    ax.legend(fontsize=8)

    # Alice Reward
    ax = axes[0, 2]
    arew = [r["rew"] for r in alices]
    ax.plot(xs_a, smooth(arew), color="orange", linewidth=1.5, label="Alice Rew")
    _axis(ax, "Alice Mean Reward", "Reward")
    ax.legend(fontsize=8)

    # Bob Loss / Val
    ax = axes[1, 0]
    bloss = [r["loss"] for r in bobs]
    bval = [r["val"] for r in bobs]
    ax.plot(xs_b, smooth(bloss), color="blue", alpha=0.3, linewidth=0.8, label="Loss")
    ax.plot(xs_b, smooth(bval), color="orange", linewidth=1.5, label="Val Loss")
    _axis(ax, "Bob Policy & Value Loss", "Loss")
    ax.legend(fontsize=8)

    # Bob ABC loss
    ax = axes[1, 1]
    babc = [r["abc"] for r in bobs]
    ax.plot(xs_b, smooth(babc), color="purple", linewidth=1.5, label="ABC Loss")
    _axis(ax, "Bob ABC (Imitation) Loss", "Loss")
    ax.legend(fontsize=8)

    # Goals valid / invalid
    ax = axes[1, 2]
    gv = [r["goals_valid"] for r in iters]
    gi = [r["goals_invalid"] for r in iters]
    ax.plot(xs_i, smooth(gv), color="green", linewidth=1.5, label="Valid")
    ax.plot(xs_i, smooth(gi), color="red", linewidth=1.5, label="Invalid")
    _axis(ax, "Goals (Valid / Invalid)", "Count")
    ax.legend(fontsize=8)

    # IK fail & terminations
    ax = axes[2, 0]
    ik = [r["IK_fail"] for r in iters]
    oot = [r["obj_off_table"] for r in iters]
    rtt = [r["robot_through_table"] for r in iters]
    ax.plot(xs_i, smooth(ik), color="red", linewidth=1.5, label="IK fail")
    ax.plot(xs_i, smooth(oot), color="orange", alpha=0.5, linewidth=0.8, label="Obj off-table")
    ax.plot(xs_i, smooth(rtt), color="grey", alpha=0.5, linewidth=0.8, label="Robot through table")
    _axis(ax, "IK Fail & Termination Counts", "Count / Rate")
    ax.legend(fontsize=8)

    # Bob pos/rot error from BobSR
    ax = axes[2, 1]
    if bob_sr_recs:
        # BobSR reported per iteration, line up with iter index
        sr_idx = list(range(len(bob_sr_recs)))
        pe = [r["pos_err"] for r in bob_sr_recs]
        re = [r["rot_err"] for r in bob_sr_recs]
        ax.plot(sr_idx, smooth(pe), color="magenta", linewidth=1.5, label="PosErr (Bob)")
        axr = ax.twinx()
        axr.plot(sr_idx, smooth(re), color="purple", linewidth=1.5, label="RotErr (Bob)")
        axr.set_ylabel("Rotation Error (rad)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = axr.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    _axis(ax, "Bob Pos/Rot Error (from BobSR)", "Position Error (m)")

    # Alice rotation change
    ax = axes[2, 2]
    if alice_rot_recs:
        ar_idx = list(range(len(alice_rot_recs)))
        ar_roll = [r["roll"] for r in alice_rot_recs]
        ar_pitch = [r["pitch"] for r in alice_rot_recs]
        ar_yaw = [r["yaw"] for r in alice_rot_recs]
        ax.plot(ar_idx, smooth(ar_roll), color="red", linewidth=1, label="Roll")
        ax.plot(ar_idx, smooth(ar_pitch), color="green", linewidth=1, label="Pitch")
        ax.plot(ar_idx, smooth(ar_yaw), color="blue", linewidth=1, label="Yaw")
        _axis(ax, "Alice Rotation Change per Goal", "Radians")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "plot_curobo_overview.png")

    # ── BobSR pos/rot SR ────────────────────────────────────────────────
    if bob_sr_recs:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        idx = list(range(len(bob_sr_recs)))
        psr = [r["pos_sr"] for r in bob_sr_recs]
        rsr = [r["rot_sr"] for r in bob_sr_recs]
        ax2.plot(idx, smooth(psr), color="blue", linewidth=1.5, label="Position SR")
        ax2.plot(idx, smooth(rsr), color="cyan", linewidth=1.5, label="Rotation SR")
        ax2.set_title("Bob Position & Rotation SR (from BobSR)")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Success Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        _save(fig2, "plot_bob_pos_rot_sr.png")


def print_summary(data: dict):
    iters = data["iters"]
    bobs = data["bob_updates"]
    alices = data["alice_updates"]

    print(f"\n{'='*60}")
    print("CURobo ASP TRAINING REPORT")
    print(f"{'='*60}")

    if iters:
        print(f"\n── Iterations ──")
        print(f"  Total:       {len(iters)}")
        print(f"  Range:       {iters[0]['iter']} → {iters[-1]['iter']}")
        last = iters[-1]
        first = iters[0]
        print(f"  Goals valid: {first['goals_valid']:>5d} → {last['goals_valid']:>5d}")
        print(f"  Goals inval: {first['goals_invalid']:>5d} → {last['goals_invalid']:>5d}")
        print(f"  Bob succ:    {first['bob_succ']:>5d} → {last['bob_succ']:>5d}")
        print(f"  Bob fail:    {first['bob_fail']:>5d} → {last['bob_fail']:>5d}")
        print(f"  IK fail:     {first['IK_fail']:.4f} → {last['IK_fail']:.4f}")
        print(f"  ABC buffer:  {first['abc_buf']:>5d} → {last['abc_buf']:>5d}")

    if bobs:
        print(f"\n── Bob Updates ──")
        print(f"  Total:       {len(bobs)}")
        last_b = bobs[-1]
        first_b = bobs[0]
        print(f"  SR:          {first_b['sr']:.4f} → {last_b['sr']:.4f}")
        print(f"  Rew:         {first_b['rew']:+.3f} → {last_b['rew']:+.3f}")
        print(f"  ABC loss:    {first_b['abc']:.4f} → {last_b['abc']:.4f}")

    if alices:
        print(f"\n── Alice Updates ──")
        print(f"  Total:       {len(alices)}")
        last_a = alices[-1]
        first_a = alices[0]
        print(f"  Rew:         {first_a['rew']:+.3f} → {last_a['rew']:+.3f}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze curobo ASP SLURM output.")
    parser.add_argument(
        "--log", type=str, nargs="+", required=True,
        help="One or more SLURM *.out files to parse.",
    )
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory (default: parent dir of first log file).",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.log]
    out_dir = Path(args.out_dir) if args.out_dir else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Parsing {len(paths)} curobo log file(s)...")
    data = merge_paths(paths)
    print(f"[INFO] Iter records: {len(data['iters'])},  Bob updates: {len(data['bob_updates'])}, "
          f"Alice updates: {len(data['alice_updates'])}")

    write_csv(data, out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(data, plots_dir)
    print_summary(data)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
