#!/usr/bin/env python3
"""Compare push approach-offset coverage: max_r = 0.08 (current) vs 0.12 (proposed).

Shows why the current radial approach band cannot reach the extremities of the
T-block (which reaches ~0.102 m from the tracked object origin) and how
max_r = 0.12 m encloses the whole object.

Output: images/push_radius_compare.{pdf,png}
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

NUM_BINS = 21
MIN_R = 0.04

C_TBLOCK = "#56B4E9"
C_CONTACT = "#009E73"
C_BAND = "#666666"
C_REACH = "#D55E00"   # object reach circle / uncovered markers
C_SPOKE = "#CCCCCC"

# T-block outline (post x2 spawn scale), origin at the junction, metres.
TBLOCK_VERTS = np.array([
    [-0.08,  0.06], [ 0.08,  0.06], [ 0.08,  0.02], [ 0.02,  0.02],
    [ 0.02, -0.10], [-0.02, -0.10], [-0.02,  0.02], [-0.08,  0.02],
])

# Object reach: max distance from the origin to any outline vertex.
OBJ_REACH = float(np.linalg.norm(TBLOCK_VERTS, axis=1).max())  # ~0.102 m


def r_bins(max_r):
    return MIN_R + (np.arange(NUM_BINS) / (NUM_BINS - 1)) * (max_r - MIN_R)


def angle_bins():
    center = (NUM_BINS - 1) / 2.0
    return ((np.arange(NUM_BINS) - center) / center) * math.pi


def draw_panel(ax, max_r, title):
    phis = angle_bins()
    rs = r_bins(max_r)

    # T-block.
    ax.add_patch(Polygon(TBLOCK_VERTS, closed=True, facecolor=C_TBLOCK,
                         edgecolor="black", linewidth=1.0, alpha=0.85, zorder=2))
    ax.plot(0, 0, marker="+", color="black", markersize=9, markeredgewidth=1.4,
            zorder=6)

    # Approach spokes and min/max offset circles.
    for ph in phis:
        ax.plot([0, max_r * math.cos(ph)], [0, max_r * math.sin(ph)],
                color=C_SPOKE, linewidth=0.5, zorder=1)
    for rr in (MIN_R, max_r):
        ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor=C_BAND,
                            linestyle="--", linewidth=1.1, zorder=3))

    # Object reach circle (dotted, orange).
    ax.add_patch(Circle((0, 0), OBJ_REACH, fill=False, edgecolor=C_REACH,
                        linestyle=":", linewidth=1.4, zorder=4))
    ax.text(0, -OBJ_REACH - 0.006, f"object reach {OBJ_REACH*100:.1f} cm",
            ha="center", va="top", fontsize=7, color=C_REACH)

    # 21 x 21 candidate contact points.
    RR, PH = np.meshgrid(rs, phis)
    xs = (RR * np.cos(PH)).ravel()
    ys = (RR * np.sin(PH)).ravel()
    ax.scatter(xs, ys, s=5, color=C_CONTACT, zorder=5)

    # Mark outline vertices that lie beyond the outer ring (unreachable).
    d = np.linalg.norm(TBLOCK_VERTS, axis=1)
    outside = TBLOCK_VERTS[d > max_r + 1e-9]
    covered = d.max() <= max_r + 1e-9
    if len(outside):
        ax.scatter(outside[:, 0], outside[:, 1], s=70, facecolors="none",
                   edgecolors=C_REACH, linewidths=1.8, zorder=7,
                   label="outside the band")

    verdict = "object enclosed" if covered else "extremities NOT reachable"
    ax.set_title(f"{title}\n(max_r = {max_r:.2f} m: {verdict})", fontsize=9)

    lim = 0.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.155, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if len(outside):
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "images")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.9))
    draw_panel(ax1, 0.08, "(a) current")
    draw_panel(ax2, 0.12, "(b) proposed")
    fig.suptitle("Approach-offset coverage over the T-block: max_r = 0.08 vs 0.12 m",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    pdf_path = os.path.join(out_dir, "push_radius_compare.pdf")
    png_path = os.path.join(out_dir, "push_radius_compare.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"object reach = {OBJ_REACH:.4f} m")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
