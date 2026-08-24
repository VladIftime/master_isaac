"""
Top-10 Validation Test Suite layout — model-independent.

Draws the 5 easiest + 5 hardest T-block validation scenes (start -> goal)
in a 2-row x 5-column grid. Adapted from plot_validation_layout.py.

Output: <repo>/literature/paper-async/presentation/figures/val_test_layout_top10.png
Usage:  python plot_validation_layout_top10.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Patch

REPO = "/home/vladi/IsaacLab/master_isaac"
sys.path.insert(0, os.path.join(REPO, "asyncDualPlayPPO", "tasks", "utils"))
import validation_configs as vc  # noqa: E402

OUT = os.path.join(REPO, "literature/paper-async/presentation/figures/val_test_layout_top10.png")

CB_SKY = "#56B4E9"      # start
CB_ORANGE = "#E69F00"   # goal
CB_GRAY = "#999999"
CB_BORDER = "#BBBBBB"

# Selected test IDs, in display order:
# Row 1: 5 easiest (most models pass)
# Row 2: 5 hardest (no model passes)
SELECTED_IDS = [
    # Row 1 — Succeed Most
    11, 12, 13, 14, 21,
    # Row 2 — Fail Most
    7, 10, 23, 24, 27,
]



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


def main():
    # Build ordered test list from ALL_TESTS
    id_to_cfg = {cfg.test_id: cfg for cfg in vc.ALL_TESTS}
    tests = [id_to_cfg[test_id] for test_id in SELECTED_IDS]

    ncols = 5
    nrows = 2
    cell_w, cell_h = 1.8, 1.4

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, nrows * cell_h + 0.50))
    axes = np.atleast_2d(axes)

    for idx, cfg in enumerate(tests):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        sx, sy = cfg.main_start.x, cfg.main_start.y
        gx, gy, gyaw = cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw

        # workspace bound (dotted)
        ax.add_patch(plt.Rectangle(
            (-0.50, 0.25), 1.0, 0.45, fill=False,
            edgecolor=CB_GRAY, linewidth=0.5, linestyle=":"))

        ax.add_patch(MplPolygon(_t_verts(sx, sy, 0.0), closed=True,
                     facecolor=CB_SKY, edgecolor="black",
                     linewidth=0.5, alpha=0.85))
        ax.add_patch(MplPolygon(_t_verts(gx, gy, gyaw), closed=True,
                     facecolor=CB_ORANGE, edgecolor="black",
                     linewidth=0.5, alpha=0.85))

        if math.hypot(gx - sx, gy - sy) > 0.01:
            ax.annotate("", xy=(gx, gy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=CB_GRAY, lw=0.7))

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0.20, 0.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"T{cfg.test_id}  {cfg.name}", fontsize=7, pad=1)

        for spine in ax.spines.values():
            spine.set_edgecolor(CB_BORDER)
            spine.set_linewidth(0.8)


    legend = [
        Patch(facecolor=CB_SKY, edgecolor="black", label="Start pose"),
        Patch(facecolor=CB_ORANGE, edgecolor="black", label="Goal pose"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, 0.05,
             "Top row: 5 easiest scenes (3\u20134 of 4 models pass)  |  "
             "Bottom row: 5 hardest scenes (0 of 4 models pass)",
             ha="center", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.15, 0.98, 0.96], h_pad=0.15, w_pad=0.3)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
