"""
Validation Test Suite layout — model-independent.

Draws the 30 held-out T-block validation scenes (start -> goal) as a grid,
WITHOUT any pass/fail border or colouring. Pure scene geometry from
validation_configs.ALL_TESTS, so it does not depend on any model's results.

Output: <repo>/literature/paper-async/presentation/figures/val_test_layout.png
Usage:  python plot_validation_layout.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Patch, Circle

REPO = "/home/vlad/IsaacLab/vlad/master_isaac"
sys.path.insert(0, os.path.join(REPO, "asyncDualPlayPPO", "tasks", "utils"))
import validation_configs as vc  # noqa: E402

OUT = os.path.join(REPO, "literature/paper-async/presentation/figures/val_test_layout.png")

CB_SKY = "#56B4E9"      # start
CB_ORANGE = "#E69F00"   # goal
CB_GRAY = "#999999"
CB_BORDER = "#BBBBBB"

GROUP_LABEL = {
    "rotation": "Rotation\n(T1\u2013T10)",
    "pos_only": "Position-only\n(T11\u2013T20)",
    "pos_rot": "Position + Rotation\n(T21\u2013T30)",
}


def _t_verts(cx, cy, yaw, scale=0.06):
    raw = np.array([
        [-0.50, 0.50], [0.50, 0.50], [0.50, 0.20], [0.15, 0.20],
        [0.15, -0.50], [-0.15, -0.50], [-0.15, 0.20], [-0.50, 0.20],
    ]) * scale
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    pts = raw @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def _group_of(cfg):
    if 1 <= cfg.test_id <= 10:
        return "rotation"
    if 11 <= cfg.test_id <= 20:
        return "pos_only"
    return "pos_rot"


def main():
    tests = vc.ALL_TESTS
    n = len(tests)
    ncols = 10
    nrows = math.ceil(n / ncols)
    cell_w, cell_h = 2.2, 1.9

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, nrows * cell_h + 0.8))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx >= n:
            ax.set_visible(False)
            continue
        cfg = tests[idx]
        sx, sy = cfg.main_start.x, cfg.main_start.y
        gx, gy, gyaw = cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw

        # workspace bound (dotted)
        ax.add_patch(plt.Rectangle((-0.50, 0.25), 1.0, 0.45, fill=False,
                     edgecolor=CB_GRAY, linewidth=0.5, linestyle=":"))

        if cfg.object_type == "disc":
            ax.add_patch(Circle((sx, sy), 0.035, facecolor=CB_SKY, edgecolor="black",
                         linewidth=0.5, alpha=0.85))
            ax.add_patch(Circle((gx, gy), 0.035, facecolor=CB_ORANGE, edgecolor="black",
                         linewidth=0.5, alpha=0.85))
        else:
            ax.add_patch(MplPolygon(_t_verts(sx, sy, 0.0), closed=True, facecolor=CB_SKY,
                         edgecolor="black", linewidth=0.5, alpha=0.85))
            ax.add_patch(MplPolygon(_t_verts(gx, gy, gyaw), closed=True, facecolor=CB_ORANGE,
                         edgecolor="black", linewidth=0.5, alpha=0.85))

        if math.hypot(gx - sx, gy - sy) > 0.01:
            ax.annotate("", xy=(gx, gy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=CB_GRAY, lw=0.7))

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0.20, 0.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"T{cfg.test_id}  {cfg.name}", fontsize=6.5, pad=2)
        # neutral border — NO pass/fail colouring
        for spine in ax.spines.values():
            spine.set_edgecolor(CB_BORDER)
            spine.set_linewidth(0.8)
        # group label on the leftmost cell of each row
        if col == 0:
            ax.set_ylabel(GROUP_LABEL[_group_of(cfg)], fontsize=9, fontweight="bold")

    legend = [
        Patch(facecolor=CB_SKY, edgecolor="black", label="Start pose"),
        Patch(facecolor=CB_ORANGE, edgecolor="black", label="Goal pose"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Validation Test Suite \u2014 30 Held-Out T-block Scenes (start \u2192 goal)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
