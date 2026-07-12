#!/usr/bin/env python3
"""Render the object-relative push-primitive action bins over a top-down T-block.

Produces a two-panel figure:
  (a) Approach placement in the object frame: the radial offset band (r) and the
      21 approach-angle bins (phi), with the 21x21 candidate contact points.
  (b) Push execution in the world frame: the 21 push-direction bins (theta) and
      the 21 push-length bins from a representative contact point.

Constants mirror tasks/utils/action_push_relative.py and the T-block training
overrides (min_r=0.04, max_r=0.08, max_len=0.20, num_bins=21).

Output: images/push_primitive_bins.{pdf,png}
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

# ── Primitive constants (T-block) ────────────────────────────────────────────
NUM_BINS = 21
MIN_R = 0.04
MAX_R = 0.12
MAX_LEN = 0.20

# ── Colour palette (colourblind-safe, matches plot_validation.py) ─────────────
C_TBLOCK = "#56B4E9"   # object fill (CB sky)
C_CONTACT = "#009E73"  # contact points / start (CB green)
C_ARROW = "#0072B2"    # push direction (CB blue)
C_END = "#D55E00"      # push end (CB vermilion)
C_SPOKE = "#BBBBBB"    # approach-angle spokes
C_BAND = "#666666"     # offset-band circles

# ── T-block outline (post x2 spawn scale), origin at the junction, metres ─────
TBLOCK_VERTS = np.array([
    [-0.08,  0.06],   # top-left of top bar
    [ 0.08,  0.06],   # top-right of top bar
    [ 0.08,  0.02],   # inner-right of top bar
    [ 0.02,  0.02],   # right of stem top
    [ 0.02, -0.10],   # bottom-right of stem
    [-0.02, -0.10],   # bottom-left of stem
    [-0.02,  0.02],   # left of stem top
    [-0.08,  0.02],   # inner-left of top bar
])


def r_bins():
    """Radial approach offsets for bins 0..20."""
    return MIN_R + (np.arange(NUM_BINS) / (NUM_BINS - 1)) * (MAX_R - MIN_R)


def angle_bins():
    """Symmetric angle bins in [-pi, pi] for phi and theta (bins 0..20)."""
    center = (NUM_BINS - 1) / 2.0
    return ((np.arange(NUM_BINS) - center) / center) * math.pi


def length_bins():
    """Push-length bins for bins 0..20."""
    return (np.arange(NUM_BINS) / (NUM_BINS - 1)) * MAX_LEN


def draw_tblock(ax, alpha=0.85):
    ax.add_patch(Polygon(TBLOCK_VERTS, closed=True, facecolor=C_TBLOCK,
                         edgecolor="black", linewidth=1.0, alpha=alpha, zorder=2))
    ax.plot(0, 0, marker="+", color="black", markersize=9, markeredgewidth=1.4,
            zorder=6)


def panel_approach(ax):
    """(a) Object-frame approach placement: r band and phi bins."""
    draw_tblock(ax)

    phis = angle_bins()
    rs = r_bins()

    # Offset-band circles at r = min_r and r = max_r.
    for rr in (MIN_R, MAX_R):
        ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor=C_BAND,
                            linestyle="--", linewidth=1.0, zorder=3))

    # 21 approach-angle spokes from the object centre.
    for ph in phis:
        ax.plot([0, MAX_R * math.cos(ph)], [0, MAX_R * math.sin(ph)],
                color=C_SPOKE, linewidth=0.6, zorder=1)

    # 21 x 21 candidate contact points (polar grid, object yaw = 0).
    RR, PH = np.meshgrid(rs, phis)
    xs = (RR * np.cos(PH)).ravel()
    ys = (RR * np.sin(PH)).ravel()
    ax.scatter(xs, ys, s=6, color=C_CONTACT, zorder=5,
               label="candidate contact points")

    # Annotate the offset band.
    ax.annotate("", xy=(MAX_R, 0.0), xytext=(MIN_R, 0.0),
                arrowprops=dict(arrowstyle="<->", color=C_BAND, lw=1.0))
    ax.text(0.145, -0.145, r"$r \in [0.04, 0.12]\,$m",
            ha="right", va="bottom", fontsize=8)

    ax.set_title("(a) approach placement (object frame)", fontsize=10)
    lim = 0.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.15, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def panel_push(ax):
    """(b) World-frame push direction and length bins from one contact point."""
    draw_tblock(ax)

    # Representative contact point: phi = 0 bin at the mid radius.
    start = np.array([0.5 * (MIN_R + MAX_R), 0.0])
    thetas = angle_bins()

    # 21 push-direction arrows at the full push length.
    for th in thetas:
        end = start + MAX_LEN * np.array([math.cos(th), math.sin(th)])
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=0.7,
                                    alpha=0.75), zorder=3)

    # Length bins marked along one representative direction (theta = +pi/4).
    th0 = math.pi / 4.0
    dirv = np.array([math.cos(th0), math.sin(th0)])
    for L in length_bins():
        p = start + L * dirv
        ax.plot(p[0], p[1], marker="o", color=C_END, markersize=2.5, zorder=5)
    tip = start + MAX_LEN * dirv
    ax.text(tip[0] + 0.004, tip[1] + 0.004,
            r"21 length bins, $0\!-\!0.20\,$m", fontsize=8, color=C_END)

    # Start marker.
    ax.plot(start[0], start[1], marker="o", color=C_CONTACT, markersize=6,
            zorder=6, label="contact point")

    ax.set_title("(b) push direction and length (world frame)", fontsize=10)
    lim = 0.30
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "images")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.6))
    panel_approach(ax1)
    panel_push(ax2)
    fig.suptitle(
        "Object-relative push primitive: 4 dimensions, 21 bins each",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    pdf_path = os.path.join(out_dir, "push_primitive_bins.pdf")
    png_path = os.path.join(out_dir, "push_primitive_bins.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
