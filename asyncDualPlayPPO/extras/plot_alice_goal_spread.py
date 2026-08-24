"""Plot the spread of goals proposed by Alice (ASP) from a saved episode-manager snapshot.

Reads ``bob/episode_manager_best.pt`` from a training run.  Each snapshot holds the
current per-environment goal for all parallel envs:

    goal_states    [num_envs, 6]  = [x, y, z, roll, pitch, yaw]  (local / table frame)
    initial_states [num_envs, 6]  = object start pose (same layout)
    goal_valid     [num_envs]     = whether the goal passed validate_goal()
    bob_success    [num_envs]     = whether Bob solved the goal
    goals_attempted / goals_succeeded / alice_base_reward ...

NOTE: this is a *snapshot* of the current per-env goals at checkpoint time (not the full
history of every goal Alice ever generated - that is never accumulated to disk).  It is a
representative sample of Alice's goal distribution at (near) the end of training.

Outputs (into --out-dir):
    alice_goal_spread_xy.png       scatter + density of goal XY, valid vs invalid,
                                   placement-zone + table bounds, marginal histograms
    alice_goal_yaw.png             goal-orientation distribution (histogram + polar rose)
    alice_goal_displacement.png    init->goal displacement magnitude hist + quiver
    alice_goal_training_curves.png Metrics/Alice/MeanDisp3D + GoalValidityRate over training
                                   (from the run's TensorBoard event files; skipped if absent)

Usage:
    python -m asyncDualPlayPPO.extras.plot_alice_goal_spread \
        --run-dir asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.26/runs/hpc_pbrs_asp_dpose_528env \
        --out-dir literature/paper-async/presentation/figures
"""

import argparse
import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Alice placement zone (valid-goal region) and table bounds, from wrapper_push_asp.py.
_PLACE_X = (-0.50, 0.50)
_PLACE_Y = (0.25, 0.70)
_TABLE_X = (-0.70, 0.70)
_TABLE_Y = (-0.10, 0.90)

_VALID_C = "#1f77b4"
_INVALID_C = "#d62728"
_SUCCESS_C = "#2ca02c"


def _derive_tag(run_dir):
    """Derive a short model tag from the run directory name.

    e.g. 'hpc_pbrs_asp_dpose_528env' -> 'asp_dpose'
    """
    base = os.path.basename(os.path.normpath(run_dir))
    for pre in ("hpc_pbrs_", "hpc_"):
        if base.startswith(pre):
            base = base[len(pre):]
            break
    for suf in ("_528env", "_2048env", "_env"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base or "model"


def _load_snapshot(run_dir):
    path = os.path.join(run_dir, "bob", "episode_manager_best.pt")
    if not os.path.isfile(path):
        alt = os.path.join(run_dir, "bob", "episode_manager_latest.pt")
        if os.path.isfile(alt):
            print(f"[warn] best snapshot missing, using {alt}")
            path = alt
        else:
            raise FileNotFoundError(f"No episode_manager_best.pt or _latest.pt under {run_dir}/bob")
    d = torch.load(path, map_location="cpu", weights_only=False)

    def np_(k):
        v = d.get(k)
        return v.numpy() if torch.is_tensor(v) else v

    return {
        "path": path,
        "goal": np_("goal_states"),
        "init": np_("initial_states"),
        "valid": np_("goal_valid"),
        "bob_success": np_("bob_success"),
        "goals_attempted": np_("goals_attempted"),
        "goals_succeeded": np_("goals_succeeded"),
        "alice_reward": np_("alice_base_reward"),
    }


def _draw_zone(ax):
    ax.add_patch(Rectangle((_TABLE_X[0], _TABLE_Y[0]), _TABLE_X[1] - _TABLE_X[0],
                           _TABLE_Y[1] - _TABLE_Y[0], fill=False, ec="0.5",
                           ls="--", lw=1.0, label="table bounds"))
    ax.add_patch(Rectangle((_PLACE_X[0], _PLACE_Y[0]), _PLACE_X[1] - _PLACE_X[0],
                           _PLACE_Y[1] - _PLACE_Y[0], fill=False, ec="0.2",
                           ls="-", lw=1.4, label="placement zone"))


def _panel_displacement_hist(ax, s):
    g, ini, valid = s["goal"], s["init"], s["valid"].astype(bool)
    disp = np.sqrt((g[:, 0] - ini[:, 0]) ** 2 + (g[:, 1] - ini[:, 1]) ** 2)
    ax.hist(disp[valid], bins=30, color=_VALID_C, alpha=0.85)
    ax.axvline(disp[valid].mean(), color=_INVALID_C, ls="--", lw=1.5,
               label=f"mean = {disp[valid].mean():.3f} m")
    ax.set_xlabel("init \u2192 goal XY displacement (m)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)


# Tight view for the compact 2x2 comparison figure: crop empty table margins to the
# goal region (goals live in x~[-0.47,0.38], y~[0.26,0.70]) so panels are wide-aspect.
_VIEW_X = (-0.55, 0.55)
_VIEW_Y = (0.18, 0.78)


def _panel_displacement_quiver(fig, ax, s, xlim=_TABLE_X, ylim=_TABLE_Y):
    g, ini, valid = s["goal"], s["init"], s["valid"].astype(bool)
    dx = g[:, 0] - ini[:, 0]
    dy = g[:, 1] - ini[:, 1]
    disp = np.sqrt(dx ** 2 + dy ** 2)
    q = ax.quiver(ini[valid, 0], ini[valid, 1], dx[valid], dy[valid],
                  disp[valid], angles="xy", scale_units="xy", scale=1.0,
                  cmap="plasma", width=0.004, alpha=0.8)
    _draw_zone(ax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(q, ax=ax, label="disp (m)", fraction=0.046, pad=0.02)


def _panel_valid_density(fig, ax, s, xlim=_TABLE_X, ylim=_TABLE_Y):
    g, valid = s["goal"], s["valid"].astype(bool)
    hb = ax.hexbin(g[valid, 0], g[valid, 1], gridsize=25,
                   extent=[_TABLE_X[0], _TABLE_X[1], _TABLE_Y[0], _TABLE_Y[1]],
                   cmap="viridis", mincnt=1)
    _draw_zone(ax)
    ax.set_xlabel("goal x (m)")
    ax.set_ylabel("goal y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(hb, ax=ax, label="goals / cell", fraction=0.046, pad=0.02)


def plot_compare(runs, out_dir, out_name="asp_vs_tasp_training.png"):
    """2x2 comparison: cols = model, rows = {displacement hist, valid-goal density}.

    runs: list of (label, snapshot-dict).
    """
    ncol = len(runs)
    # Panels cropped to the goal region (_VIEW_*): ~1.1 wide : 0.6 tall (~1.8 aspect),
    # so a 2-row figure stays short and wide -> fills a 16:9 slide with no whitespace.
    fig, axes = plt.subplots(2, ncol, figsize=(4.9 * ncol, 4.9),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for j, (label, s) in enumerate(runs):
        nv = int(s["valid"].astype(bool).sum())
        _panel_displacement_quiver(fig, axes[0, j], s, xlim=_VIEW_X, ylim=_VIEW_Y)
        axes[0, j].set_title(f"{label} — init$\\rightarrow$goal displacement (n={nv})",
                             fontsize=10)
        _panel_valid_density(fig, axes[1, j], s, xlim=_VIEW_X, ylim=_VIEW_Y)
        axes[1, j].set_title(f"{label} — valid-goal density", fontsize=10)
    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def plot_xy(s, out_dir, tag):
    g, valid = s["goal"], s["valid"].astype(bool)
    gx, gy = g[:, 0], g[:, 1]
    succ = s["bob_success"].astype(bool) if s["bob_success"] is not None else np.zeros_like(valid)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 4], height_ratios=[1, 4],
                          wspace=0.28, hspace=0.05)

    # --- left: scatter with marginal histograms ---
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    inv = ~valid
    ax.scatter(gx[inv], gy[inv], s=18, c=_INVALID_C, alpha=0.25, marker="x",
               label=f"invalid ({int(inv.sum())})")
    v_fail = valid & ~succ
    v_succ = valid & succ
    ax.scatter(gx[v_fail], gy[v_fail], s=26, c=_VALID_C, alpha=0.75,
               edgecolors="none", label=f"valid, Bob failed ({int(v_fail.sum())})")
    ax.scatter(gx[v_succ], gy[v_succ], s=30, c=_SUCCESS_C, alpha=0.9,
               edgecolors="k", linewidths=0.3, label=f"valid, Bob solved ({int(v_succ.sum())})")
    _draw_zone(ax)
    ax.set_xlabel("goal x (m)")
    ax.set_ylabel("goal y (m)")
    ax.set_title("Alice goal positions")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    bins = 30
    axx.hist(gx[valid], bins=bins, range=_TABLE_X, color=_VALID_C, alpha=0.8)
    axx.tick_params(labelbottom=False)
    axx.set_ylabel("count", fontsize=8)
    axy.hist(gy[valid], bins=bins, range=_TABLE_Y, orientation="horizontal",
             color=_VALID_C, alpha=0.8)
    axy.tick_params(labelleft=False)
    axy.set_xlabel("count", fontsize=8)

    # --- right: density (valid goals) ---
    axd = fig.add_subplot(gs[1, 2])
    hb = axd.hexbin(gx[valid], gy[valid], gridsize=25,
                    extent=[_TABLE_X[0], _TABLE_X[1], _TABLE_Y[0], _TABLE_Y[1]],
                    cmap="viridis", mincnt=1)
    _draw_zone(axd)
    axd.set_xlabel("goal x (m)")
    axd.set_ylabel("goal y (m)")
    axd.set_title("Valid-goal density")
    axd.set_aspect("equal", adjustable="box")
    fig.colorbar(hb, ax=axd, label="goals / cell", shrink=0.8)

    out = os.path.join(out_dir, f"alice_goal_spread_xy_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def plot_yaw(s, out_dir, tag):
    g, valid = s["goal"], s["valid"].astype(bool)
    yaw = g[valid, 5]

    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(yaw, bins=36, range=(-np.pi, np.pi), color=_VALID_C, alpha=0.85)
    ax1.axvline(0.0, color="0.4", ls="--", lw=1)
    ax1.set_xlabel("goal yaw (rad)")
    ax1.set_ylabel("count")
    ax1.set_title(f"Goal orientation (valid, n={valid.sum()})")
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels(["-\u03c0", "-\u03c0/2", "0", "\u03c0/2", "\u03c0"])

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    counts, edges = np.histogram(yaw, bins=36, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2
    ax2.bar(centers, counts, width=(2 * np.pi / 36), color=_VALID_C,
            alpha=0.85, edgecolor="k", linewidth=0.3)
    ax2.set_theta_zero_location("E")
    ax2.set_title("Goal yaw (polar rose)")

    out = os.path.join(out_dir, f"alice_goal_yaw_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def plot_displacement(s, out_dir, tag):
    g, ini, valid = s["goal"], s["init"], s["valid"].astype(bool)
    dx = g[:, 0] - ini[:, 0]
    dy = g[:, 1] - ini[:, 1]
    disp = np.sqrt(dx ** 2 + dy ** 2)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(disp[valid], bins=30, color=_VALID_C, alpha=0.85)
    ax1.axvline(disp[valid].mean(), color=_INVALID_C, ls="--", lw=1.5,
                label=f"mean = {disp[valid].mean():.3f} m")
    ax1.set_xlabel("init \u2192 goal XY displacement (m)")
    ax1.set_ylabel("count")
    ax1.set_title(f"Goal displacement magnitude (valid, n={valid.sum()})")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    q = ax2.quiver(ini[valid, 0], ini[valid, 1], dx[valid], dy[valid],
                   disp[valid], angles="xy", scale_units="xy", scale=1.0,
                   cmap="plasma", width=0.004, alpha=0.8)
    _draw_zone(ax2)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("init \u2192 goal displacement vectors")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(_TABLE_X)
    ax2.set_ylim(_TABLE_Y)
    fig.colorbar(q, ax=ax2, label="displacement (m)", shrink=0.8)

    out = os.path.join(out_dir, f"alice_goal_displacement_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def plot_training_curves(run_dir, out_dir, tag):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:  # pragma: no cover
        print(f"[warn] tensorboard not available ({e}); skipping training curves")
        return

    tags = {
        "Metrics/Alice/MeanDisp3D": "Mean 3D displacement (m)",
        "Metrics/Alice/GoalValidityRate": "Goal validity rate",
    }
    # Alice/goal scalars are written by the summary (and bob) writers.
    ev_files = []
    for sub in ("summary", "bob", "alice"):
        ev_files += glob.glob(os.path.join(run_dir, sub, "events.out.tfevents.*"))
    if not ev_files:
        print("[warn] no tfevents found; skipping training curves")
        return

    series = {t: {} for t in tags}  # tag -> {step: value} (dedup, keep last)
    for f in sorted(ev_files):
        try:
            acc = EventAccumulator(f, size_guidance={"scalars": 0})
            acc.Reload()
        except Exception:
            continue
        avail = set(acc.Tags().get("scalars", []))
        for t in tags:
            if t in avail:
                for ev in acc.Scalars(t):
                    series[t][ev.step] = ev.value

    if not any(series[t] for t in tags):
        print("[warn] Alice goal tags not found in tfevents; skipping training curves")
        return

    fig, axes = plt.subplots(1, len(tags), figsize=(12, 4.2))
    for ax, (tb_tag, label) in zip(np.atleast_1d(axes), tags.items()):
        pts = sorted(series[tb_tag].items())
        if not pts:
            ax.set_visible(False)
            continue
        steps = np.array([p[0] for p in pts])
        vals = np.array([p[1] for p in pts])
        ax.plot(steps, vals, color=_VALID_C, lw=1.3)
        ax.set_xlabel("Bob update step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)

    fig.suptitle("Alice goal statistics over training", y=1.02)
    out = os.path.join(out_dir, f"alice_goal_training_curves_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def print_stats(s):
    g, ini, valid = s["goal"], s["init"], s["valid"].astype(bool)
    succ = s["bob_success"].astype(bool) if s["bob_success"] is not None else None
    n = g.shape[0]
    disp = np.sqrt((g[:, 0] - ini[:, 0]) ** 2 + (g[:, 1] - ini[:, 1]) ** 2)
    print("\n" + "=" * 60)
    print(f"snapshot: {s['path']}")
    print(f"envs (goals in snapshot): {n}")
    print(f"valid goals: {int(valid.sum())} ({100*valid.mean():.1f}%)")
    if succ is not None:
        print(f"Bob solved: {int(succ.sum())} ({100*succ.mean():.1f}% of all; "
              f"{100*succ[valid].mean():.1f}% of valid)")
    for i, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        col = g[valid, i]
        print(f"  goal {name:>5}: min={col.min():+.3f} max={col.max():+.3f} "
              f"mean={col.mean():+.3f} std={col.std():.3f}")
    print(f"  XY spread cov:\n{np.cov(g[valid, 0], g[valid, 1])}")
    print(f"  displacement (valid): mean={disp[valid].mean():.3f} "
          f"p50={np.percentile(disp[valid],50):.3f} p90={np.percentile(disp[valid],90):.3f} "
          f"max={disp[valid].max():.3f}")
    print("=" * 60 + "\n")


def main():
    p = argparse.ArgumentParser(description="Plot Alice's goal spread from an episode-manager snapshot")
    default_run = ("asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.26/runs/"
                   "hpc_pbrs_asp_dpose_528env")
    p.add_argument("--run-dir", default=default_run,
                   help="Training run dir containing bob/episode_manager_best.pt")
    p.add_argument("--out-dir", default="literature/paper-async/presentation/figures",
                   help="Directory to write the PNG plots")
    p.add_argument("--tag", default=None,
                   help="Model tag for output filenames (default: derived from --run-dir)")
    p.add_argument("--compare", nargs="+", default=None,
                   help="Two+ run dirs for a 2x2 ASP-vs-TASP comparison figure "
                        "(displacement hist + valid-goal density). Overrides single-run mode.")
    p.add_argument("--compare-labels", nargs="+", default=None,
                   help="Column labels for --compare (default: derived tags)")
    p.add_argument("--compare-out", default="asp_vs_tasp_training.png",
                   help="Output filename for the --compare figure")
    args = p.parse_args()

    if args.compare:
        os.makedirs(args.out_dir, exist_ok=True)
        labels = args.compare_labels or [_derive_tag(r) for r in args.compare]
        runs = [(lbl, _load_snapshot(r)) for lbl, r in zip(labels, args.compare)]
        for lbl, s in runs:
            print(f"[{lbl}]")
            print_stats(s)
        plot_compare(runs, args.out_dir, out_name=args.compare_out)
        print("[done]")
        return

    tag = args.tag or _derive_tag(args.run_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    s = _load_snapshot(args.run_dir)
    print(f"[tag] {tag}")
    print_stats(s)
    plot_xy(s, args.out_dir, tag)
    plot_yaw(s, args.out_dir, tag)
    plot_displacement(s, args.out_dir, tag)
    plot_training_curves(args.run_dir, args.out_dir, tag)
    print("[done]")


if __name__ == "__main__":
    main()
