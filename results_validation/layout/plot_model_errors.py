"""
Per-model per-test error plots, regenerated from the EXACT 30-scene CSVs used
for the deck headline numbers (so the bars match 80.0 / 76.7 / 16.7 / 6.7%).

Top panel: best position error per test (cm) with 5 cm threshold.
Bottom panel: best rotation error per test (rad) with 0.2 rad threshold.
PASS = blue, single-metric-met = green, FAIL = vermilion.

Outputs -> presentation/figures/errors_<model>.png
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = "/home/vlad/IsaacLab/vlad/master_isaac"
RV = os.path.join(REPO, "results_validation")
OUTDIR = os.path.join(REPO, "literature/paper-async/presentation/figures")
sys.path.insert(0, os.path.join(REPO, "asyncDualPlayPPO", "tasks", "utils"))
import validation_configs as vc  # noqa: E402

CB_BLUE = "#0072B2"
CB_VERMILION = "#D55E00"
CB_GREEN = "#009E73"

MODELS = [
    ("PPO-PBRS",       os.path.join(RV, "A_simp/20_isaac_30t.csv"),    "errors_ppo_pbrs.png"),
    ("PPO-Curriculum", os.path.join(RV, "B_curr/28_isaac_30t.csv"),    "errors_ppo_curriculum.png"),
    ("ASP-dPose",      os.path.join(RV, "E_asp_dpose/26_isaac.csv"),   "errors_asp_dpose.png"),
    ("TASP-dPose",     os.path.join(RV, "G_tasp_dpose/26_isaac.csv"),  "errors_tasp_dpose.png"),
]

POS_TH_CM = 5.0
ROT_TH = 0.2


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r["test_index"]))
    return rows


def group_label(idx):
    if 1 <= idx <= 10:
        return "Rotation"
    if 11 <= idx <= 20:
        return "Position-only"
    return "Position + Rotation"


def plot_model(label, csv_path, out_name):
    rows = load(csv_path)
    n = len(rows)
    x = np.arange(n)
    pos_cm = [float(r["pos_err"]) * 100 for r in rows]
    rot = [float(r["rot_err"]) for r in rows]
    succ = [int(r["success"]) == 1 for r in rows]
    idxs = [int(r["test_index"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, n * 0.62), 6.6), sharex=True)
    fig.suptitle(f"{label} — Per-Test Position \\& Rotation Error (30 held-out scenes)",
                 fontweight="bold", fontsize=12)

    pos_colors = [CB_BLUE if succ[i] else (CB_GREEN if pos_cm[i] < POS_TH_CM else CB_VERMILION)
                  for i in range(n)]
    ax1.bar(x, pos_cm, 0.65, color=pos_colors, edgecolor="black", linewidth=0.4)
    ax1.axhline(POS_TH_CM, color=CB_GREEN, ls="--", lw=1.2, label=f"Threshold ({POS_TH_CM:.0f} cm)")
    ax1.set_ylabel("Position Error (cm)")
    ax1.set_ylim(0, max(max(pos_cm) * 1.15, 10))
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    rot_colors = [CB_BLUE if succ[i] else (CB_GREEN if rot[i] < ROT_TH else CB_VERMILION)
                  for i in range(n)]
    ax2.bar(x, rot, 0.65, color=rot_colors, edgecolor="black", linewidth=0.4)
    ax2.axhline(ROT_TH, color=CB_GREEN, ls="--", lw=1.2, label=f"Threshold ({ROT_TH:.1f} rad)")
    ax2.set_ylabel("Rotation Error (rad)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{i}" for i in idxs], fontsize=7)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    # group separators + labels
    bounds = [(0, 9, "Rotation"), (10, 19, "Position-only"), (20, 29, "Position + Rotation")]
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax2.transData, ax2.transAxes)
    for gs, ge, gl in bounds:
        if gs > 0:
            ax1.axvline(gs - 0.5, color="black", lw=0.6, alpha=0.3)
            ax2.axvline(gs - 0.5, color="black", lw=0.6, alpha=0.3)
        ax2.text((gs + ge) / 2.0, -0.16, gl, ha="center", va="top", fontsize=9,
                 fontweight="bold", transform=trans)

    legend = [
        Patch(facecolor=CB_BLUE, edgecolor="black", label="PASS (both gates)"),
        Patch(facecolor=CB_GREEN, edgecolor="black", label="this metric met"),
        Patch(facecolor=CB_VERMILION, edgecolor="black", label="FAIL"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_pass = sum(succ)
    print(f"[OK] {out}  ({label}: {n_pass}/{n} = {n_pass/n*100:.1f}%)")


def main():
    for label, path, out in MODELS:
        plot_model(label, path, out)


if __name__ == "__main__":
    main()
