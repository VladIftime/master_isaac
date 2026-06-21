"""
Plot validation results from CSV files produced by validate_push.py
and validate_push_asp.py.

Produces:
  - Bar chart: success rate per model (overall)
  - Bar chart: success rate per model per difficulty (easy/medium/hard)
  - Bar chart: average pushes per model
  - Bar chart: average pushes per difficulty per model
  - Table: per-test comparison

Usage:
  python asyncDualPlayPPO/tests/plot_validation.py \
      --csvs results_simp.csv results_curr.csv results_asp.csv \
      --labels "A: PBRS only" "B: PBRS+Curriculum" "C: PBRS+ASP" \
      -o validation_plots
"""

import argparse
import os
import sys
from typing import List, Optional, Dict, Tuple


def generate_single_run_plot(results, test_cfgs, save_path,
                             pos_threshold_cm=5.0, rot_threshold_rad=0.2,
                             policy_label=""):
    import math
    _suffix = f"  [{policy_label}]" if policy_label else ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon as MplPolygon, Patch, Circle
    from matplotlib.gridspec import GridSpec

    CB_BLUE = "#0072B2"
    CB_VERMILION = "#D55E00"
    CB_SKY = "#56B4E9"
    CB_ORANGE = "#E69F00"
    CB_GREEN = "#009E73"
    CB_GRAY = "#999999"

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

    n = len(results)
    if n == 0:
        return

    base, ext = os.path.splitext(save_path)
    errors_path = f"{base}_errors{ext}"
    layout_path = f"{base}_layout{ext}"

    test_labels = [f"T{r['test_index']}" for r in results]
    pos_errs_cm = [r["final_pos_error"] * 100.0 for r in results]
    rot_errs = [r["final_rot_error"] for r in results]
    colors = [CB_BLUE if r["success"] else CB_VERMILION for r in results]
    x = np.arange(n)
    bar_w = 0.65

    fig_e, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(18, n * 0.7), 7), sharex=True)

    pos_colors = []
    for i, r in enumerate(results):
        if r["success"]:
            pos_colors.append(CB_BLUE)
        elif pos_errs_cm[i] < pos_threshold_cm:
            pos_colors.append(CB_GREEN)
        else:
            pos_colors.append(CB_VERMILION)

    ax1.bar(x, pos_errs_cm, bar_w, color=pos_colors, edgecolor="black", linewidth=0.4)
    ax1.axhline(y=pos_threshold_cm, color=CB_GREEN, linestyle="--", linewidth=1.2,
                label=f"Threshold ({pos_threshold_cm:.0f} cm)")
    for i, r in enumerate(results):
        if r["success"]:
            tag, clr = "PASS", CB_BLUE
        elif pos_errs_cm[i] < pos_threshold_cm:
            tag, clr = "POS \u2713", CB_GREEN
        else:
            tag, clr = "FAIL", CB_VERMILION
        y_off = max(pos_errs_cm) * 0.02 + 0.3
        ax1.text(i, pos_errs_cm[i] + y_off, tag, ha="center", va="bottom",
                 fontsize=6, fontweight="bold", color=clr)
    ax1.set_ylabel("Position Error (cm)")
    ax1.set_title("Best Position Error per Test" + _suffix, fontweight="bold")
    ax1.set_xlim(-0.5, n - 0.5)
    _pos_max = max(max(pos_errs_cm) * 1.15, 10)
    ax1.set_yticks(np.arange(0, _pos_max + 5, 5))
    ax1.set_ylim(0, _pos_max)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    rot_colors = []
    for i, r in enumerate(results):
        if test_cfgs[i].get("object_type", "tblock") == "disc":
            rot_colors.append(CB_GRAY)
        elif r["success"]:
            rot_colors.append(CB_BLUE)
        elif rot_errs[i] < rot_threshold_rad:
            rot_colors.append(CB_GREEN)
        else:
            rot_colors.append(CB_VERMILION)

    ax2.bar(x, rot_errs, bar_w, color=rot_colors, edgecolor="black", linewidth=0.4)
    ax2.axhline(y=rot_threshold_rad, color=CB_GREEN, linestyle="--", linewidth=1.2,
                label=f"Threshold ({rot_threshold_rad:.1f} rad)")
    for i, r in enumerate(results):
        is_disc = test_cfgs[i].get("object_type", "tblock") == "disc"
        if is_disc:
            tag, clr = "N/A", CB_GRAY
        elif r["success"]:
            tag, clr = "PASS", CB_BLUE
        elif rot_errs[i] < rot_threshold_rad:
            tag, clr = "ROT \u2713", CB_GREEN
        else:
            tag, clr = "FAIL", CB_VERMILION
        y_off = max(rot_errs) * 0.02 + 0.01
        ax2.text(i, rot_errs[i] + y_off, tag, ha="center", va="bottom",
                 fontsize=6, fontweight="bold", color=clr)
    ax2.set_ylabel("Rotation Error (rad)")
    ax2.set_title("Best Rotation Error per Test" + _suffix, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([""] * n)
    ax2.tick_params(axis="x", length=0)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    def _group_label(cfg):
        if cfg.get("object_type", "tblock") == "disc":
            return "Disc"
        elif abs(cfg.get("goal_yaw", 0.0)) < 1e-6:
            return "Position Only (T-block)"
        return "Position + Rotation (T-block)"

    groups = [_group_label(c) for c in test_cfgs]
    group_spans = []
    cur_g, g_start = groups[0], 0
    for gi in range(1, n):
        if groups[gi] != cur_g:
            group_spans.append((g_start, gi - 1, cur_g))
            cur_g, g_start = groups[gi], gi
    group_spans.append((g_start, n - 1, cur_g))

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax2.transData, ax2.transAxes)
    for si, (gs, ge, gl) in enumerate(group_spans):
        if gs > 0:
            ax1.axvline(x=gs - 0.5, color="black", linewidth=0.6, alpha=0.3)
            ax2.axvline(x=gs - 0.5, color="black", linewidth=0.6, alpha=0.3)
        mid = (gs + ge) / 2.0
        ax2.text(mid, -0.10, gl, ha="center", va="top", fontsize=9,
                 fontweight="bold", transform=trans)

    legend_err = [
        Patch(facecolor=CB_BLUE, edgecolor="black", label="PASS"),
        Patch(facecolor=CB_GREEN, edgecolor="black", label="Metric met"),
        Patch(facecolor=CB_VERMILION, edgecolor="black", label="FAIL"),
        Patch(facecolor=CB_GRAY, edgecolor="black", label="N/A (disc)"),
    ]
    fig_e.legend(handles=legend_err, loc="lower center", ncol=4, fontsize=9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_e.tight_layout(pad=0.5, rect=[0, 0.03, 1, 1])
    fig_e.savefig(errors_path, dpi=150, bbox_inches="tight")
    plt.close(fig_e)
    print(f"[Plot] Saved to {errors_path}")

    ncols = min(n, 10)
    nrows = math.ceil(n / ncols)
    cell_w = 2.2
    cell_h = 1.8
    fig_l, axes = plt.subplots(nrows, ncols,
                               figsize=(ncols * cell_w, nrows * cell_h + 0.6))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx >= n:
            ax.set_visible(False)
            continue
        r, cfg = results[idx], test_cfgs[idx]
        ax.add_patch(plt.Rectangle((-0.50, 0.25), 1.0, 0.45,
                     fill=False, edgecolor=CB_GRAY, linewidth=0.5, linestyle=":"))
        obj_type = cfg.get("object_type", "tblock")
        if obj_type == "disc":
            _disc_r = 0.035
            ax.add_patch(Circle((cfg["start_x"], cfg["start_y"]), _disc_r,
                         facecolor=CB_SKY, edgecolor="black", linewidth=0.5, alpha=0.8))
            ax.add_patch(Circle((cfg["goal_x"], cfg["goal_y"]), _disc_r,
                         facecolor=CB_ORANGE, edgecolor="black", linewidth=0.5, alpha=0.8))
        else:
            sv = _t_verts(cfg["start_x"], cfg["start_y"], 0.0)
            ax.add_patch(MplPolygon(sv, closed=True, facecolor=CB_SKY,
                         edgecolor="black", linewidth=0.5, alpha=0.8))
            gv = _t_verts(cfg["goal_x"], cfg["goal_y"], cfg["goal_yaw"])
            ax.add_patch(MplPolygon(gv, closed=True, facecolor=CB_ORANGE,
                         edgecolor="black", linewidth=0.5, alpha=0.8))
        dx = cfg["goal_x"] - cfg["start_x"]
        dy = cfg["goal_y"] - cfg["start_y"]
        if math.hypot(dx, dy) > 0.01:
            ax.annotate("", xy=(cfg["goal_x"], cfg["goal_y"]),
                        xytext=(cfg["start_x"], cfg["start_y"]),
                        arrowprops=dict(arrowstyle="->", color=CB_GRAY, lw=0.7))
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0.20, 0.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(test_labels[idx], fontsize=7, pad=2)
        for spine in ax.spines.values():
            spine.set_edgecolor(CB_BLUE if r["success"] else CB_VERMILION)
            spine.set_linewidth(1.5)

    legend_layout = [
        Patch(facecolor=CB_BLUE, edgecolor="black", label="PASS"),
        Patch(facecolor=CB_VERMILION, edgecolor="black", label="FAIL"),
        Patch(facecolor=CB_SKY, edgecolor="black", label="Start"),
        Patch(facecolor=CB_ORANGE, edgecolor="black", label="Goal"),
    ]
    fig_l.legend(handles=legend_layout, loc="lower center", ncol=4, fontsize=9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_l.suptitle("Start / Goal Positions" + _suffix, fontweight="bold", fontsize=11)
    fig_l.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_l.savefig(layout_path, dpi=150, bbox_inches="tight")
    plt.close(fig_l)
    print(f"[Plot] Saved to {layout_path}")


def load_csv(path: str) -> list:
    import csv
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows: list) -> Dict:
    total = len(rows)
    n_success = sum(1 for r in rows if r["success"] == "1")
    pushes = [int(r["pushes_used"]) for r in rows]
    pos_errs = [float(r["pos_err"]) for r in rows]
    rot_errs = [float(r["rot_err"]) for r in rows]
    covs = [float(r.get("area_coverage", 0.0)) for r in rows]
    return {
        "total": total,
        "successes": n_success,
        "sr": n_success / total * 100 if total > 0 else 0.0,
        "avg_pushes": sum(pushes) / len(pushes) if pushes else 0.0,
        "avg_pos_err": sum(pos_errs) / len(pos_errs) if pos_errs else 0.0,
        "avg_rot_err": sum(rot_errs) / len(rot_errs) if rot_errs else 0.0,
        "avg_coverage": sum(covs) / len(covs) if covs else 0.0,
        "pushes": pushes,
    }


def difficulty_from_name(test_name: str) -> str:
    name = test_name.upper()
    if name.startswith("E_"):
        return "easy"
    if name.startswith("M_"):
        return "medium"
    if name.startswith("H_") or name.startswith("EDGE"):
        return "hard"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Plot validation results")
    parser.add_argument("--csvs", type=str, nargs="+", default=None,
                        help="CSV files to compare (multi-model mode)")
    parser.add_argument("--single", type=str, default=None,
                        help="Single CSV file for per-test error + layout plots")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for each CSV (same order)")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--rot-threshold", type=float, default=0.2,
                        help="Rotation threshold in rad (default 0.2)")
    args = parser.parse_args()

    if args.single:
        from asyncDualPlayPPO.tasks.utils.validation_configs import get_test_config
        rows = load_csv(args.single)
        results_data = []
        cfgs_data = []
        for row in rows:
            tid = int(row["test_index"])
            cfg = get_test_config(tid)
            results_data.append({
                "test_index": tid,
                "test_name": row["test_name"],
                "success": row["success"] == "1",
                "final_pos_error": float(row["pos_err"]),
                "final_rot_error": float(row["rot_err"]),
            })
            cfgs_data.append({
                "start_x": cfg.main_start.x,
                "start_y": cfg.main_start.y,
                "goal_x": cfg.main_goal_x,
                "goal_y": cfg.main_goal_y,
                "goal_yaw": cfg.main_goal_yaw,
                "object_type": getattr(cfg, "object_type", "tblock"),
            } if cfg else {
                "start_x": 0, "start_y": 0.4, "goal_x": 0, "goal_y": 0.5,
                "goal_yaw": 0, "object_type": "tblock",
            })
        save_path = os.path.splitext(args.single)[0] + ".png"
        generate_single_run_plot(results_data, cfgs_data, save_path,
                                 rot_threshold_rad=args.rot_threshold)
        return

    if not args.csvs:
        parser.error("Either --single or --csvs is required")

    if args.out_dir is None:
        args.out_dir = "validation_plots"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[ERROR] matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    labels = args.labels if args.labels else [f"Model {i+1}" for i in range(len(args.csvs))]

    all_data = []
    for path in args.csvs:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path}")
            continue
        rows = load_csv(path)
        metrics = compute_metrics(rows)
        all_data.append({"path": path, "rows": rows, "metrics": metrics,
                         "label": labels[len(all_data)]})
        print(f"[OK] Loaded {path}: SR={metrics['sr']:.1f}%, "
              f"avg_pushes={metrics['avg_pushes']:.1f}, "
              f"n={metrics['total']}")

    if not all_data:
        print("[ERROR] No valid CSV files loaded.")
        sys.exit(1)

    n_models = len(all_data)
    _cmap = matplotlib.colormaps["tab10"].resampled(max(n_models, 3))
    model_colors = [_cmap(i) for i in range(n_models)]

    test_indices = sorted(set(int(r["test_index"]) for d in all_data for r in d["rows"]))
    test_short_names = []
    for tid in test_indices:
        _found = False
        for d in all_data:
            for r in d["rows"]:
                if int(r["test_index"]) == tid:
                    test_short_names.append(r["test_name"].split(" #")[0])
                    _found = True
                    break
            if _found:
                break
        if not _found:
            test_short_names.append(f"T{tid}")

    # ── Overall success rate ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    model_labels = [d["label"] for d in all_data]
    srs = [d["metrics"]["sr"] for d in all_data]
    bars = ax.bar(model_labels, srs, color=model_colors[:n_models])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Overall Validation Success Rate")
    for bar, val in zip(bars, srs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(srs) * 1.25 if max(srs) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "overall_sr.png"))
    plt.close(fig)

    # ── Success rate by difficulty ────────────────────────────────────────────
    difficulties = ["easy", "medium", "hard"]
    diff_colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}

    for diff in difficulties:
        fig, ax = plt.subplots(figsize=(6, 4))
        diff_srs = []
        for d in all_data:
            diff_rows = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
            diff_metrics = compute_metrics(diff_rows)
            diff_srs.append(diff_metrics["sr"])
        if all(s == 0 for s in diff_srs):
            plt.close(fig)
            continue
        bars = ax.bar(model_labels, diff_srs,
                       color=model_colors[:n_models])
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Success Rate — {diff.title()} Difficulty")
        for bar, val in zip(bars, diff_srs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_ylim(0, max(diff_srs) * 1.3 if max(diff_srs) > 0 else 100)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"sr_{diff}.png"))
        plt.close(fig)

    # ── Average pushes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_pushes = [d["metrics"]["avg_pushes"] for d in all_data]
    bars = ax.bar(model_labels, avg_pushes, color=model_colors[:n_models])
    ax.set_ylabel("Average Pushes")
    ax.set_title("Average Pushes per Test")
    for bar, val in zip(bars, avg_pushes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "avg_pushes.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    avg_covs = [d["metrics"]["avg_coverage"] for d in all_data]
    bars = ax.bar(model_labels, avg_covs, color=model_colors[:n_models])
    ax.set_ylabel("Area Coverage (%)")
    ax.set_title("Average Target Area Coverage")
    for bar, val in zip(bars, avg_covs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(avg_covs) * 1.25 if max(avg_covs) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "avg_coverage.png"))
    plt.close(fig)

    # SR by difficulty grouped
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(difficulties))
    width = 0.8 / n_models
    for i, d in enumerate(all_data):
        diff_srs_list = []
        for diff in difficulties:
            dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
            dm = compute_metrics(dr)
            diff_srs_list.append(dm["sr"])
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], diff_srs_list, width,
                       label=d["label"],
                       color=model_colors[i])
        for bar, val in zip(bars, diff_srs_list):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Difficulty")
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.title() for d in difficulties])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_by_difficulty_grouped.png"))
    plt.close(fig)

    # ── Success rate by test type (pos_only vs pos_rot) ───────────────────────
    test_types = ["pos_only", "pos_rot"]
    fig, ax = plt.subplots(figsize=(6, 4))
    type_srs = []
    for d in all_data:
        t_rows = [r for r in d["rows"] if r.get("test_type", "") == "pos_only"]
        t_m = compute_metrics(t_rows)
        type_srs.append(t_m["sr"])
    bars = ax.bar(model_labels, type_srs, color=model_colors[:n_models])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate — Position Only (10 tests)")
    for bar, val in zip(bars, type_srs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(type_srs) * 1.3 if max(type_srs) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_pos_only.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    type_srs_rot = []
    for d in all_data:
        t_rows = [r for r in d["rows"] if r.get("test_type", "") == "pos_rot"]
        t_m = compute_metrics(t_rows)
        type_srs_rot.append(t_m["sr"])
    bars = ax.bar(model_labels, type_srs_rot, color=model_colors[:n_models])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate — Position + Rotation (10 tests)")
    for bar, val in zip(bars, type_srs_rot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(type_srs_rot) * 1.3 if max(type_srs_rot) > 0 else 100)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_pos_rot.png"))
    plt.close(fig)

    # Grouped: SR by test type across models
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(test_types))
    width = 0.8 / n_models
    for i, d in enumerate(all_data):
        type_srs_list = []
        for tt in test_types:
            tr = [r for r in d["rows"] if r.get("test_type", "") == tt]
            tm = compute_metrics(tr)
            type_srs_list.append(tm["sr"])
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], type_srs_list, width,
                       label=d["label"],
                       color=model_colors[i])
        for bar, val in zip(bars, type_srs_list):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Test Type")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Position Only", "Position + Rotation"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sr_by_test_type_grouped.png"))
    plt.close(fig)

    # ── Per-test error comparison (position + rotation) ───────────────────────
    fig, (ax_pos, ax_rot) = plt.subplots(2, 1, figsize=(max(14, len(test_indices) * 0.8), 9), sharex=True)
    x_t = np.arange(len(test_indices))
    w_t = 0.8 / n_models

    for i, d in enumerate(all_data):
        pe_vals, re_vals = [], []
        for tid in test_indices:
            match = [r for r in d["rows"] if int(r["test_index"]) == tid]
            pe_vals.append(float(match[0]["pos_err"]) if match else 0.0)
            re_vals.append(float(match[0]["rot_err"]) if match else 0.0)
        offset = (i - n_models / 2 + 0.5) * w_t
        ax_pos.bar(x_t + offset, pe_vals, w_t, label=d["label"], color=model_colors[i],
                   edgecolor="white", linewidth=0.5)
        ax_rot.bar(x_t + offset, re_vals, w_t, label=d["label"], color=model_colors[i],
                   edgecolor="white", linewidth=0.5)

    ax_pos.axhline(0.05, color="red", linestyle="--", alpha=0.6, linewidth=1.2,
                   label="Threshold (0.05 m)")
    ax_pos.set_ylabel("Position Error (m)")
    ax_pos.set_title("Final Position Error per Test")
    ax_pos.legend(fontsize=8, loc="upper right")
    ax_pos.grid(axis="y", alpha=0.3)

    ax_rot.axhline(0.2, color="red", linestyle="--", alpha=0.6, linewidth=1.2,
                   label="Threshold (0.2 rad)")
    ax_rot.set_ylabel("Rotation Error (rad)")
    ax_rot.set_title("Final Rotation Error per Test")
    ax_rot.set_xticks(x_t)
    ax_rot.set_xticklabels(test_short_names, rotation=45, ha="right", fontsize=8)
    ax_rot.legend(fontsize=8, loc="upper right")
    ax_rot.grid(axis="y", alpha=0.3)

    fig.suptitle("Distance to Target per Test", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(args.out_dir, "per_test_error.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] per_test_error.png saved")

    # ── Error scatter (pos_err vs rot_err) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    _markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    rect = Rectangle((0, 0), 0.05, 0.2, linewidth=2, edgecolor="green",
                      facecolor="green", alpha=0.12, label="Success region")
    ax.add_patch(rect)

    for i, d in enumerate(all_data):
        mk = _markers[i % len(_markers)]
        for r in d["rows"]:
            pe = float(r["pos_err"])
            re = float(r["rot_err"])
            tid = int(r["test_index"])
            ax.scatter(pe, re, c=[model_colors[i]], marker=mk, s=70,
                       edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(str(tid), (pe, re), textcoords="offset points",
                        xytext=(5, 5), fontsize=6, color="gray")

    for i, d in enumerate(all_data):
        ax.scatter([], [], c=[model_colors[i]], marker=_markers[i % len(_markers)],
                   s=70, edgecolors="black", linewidths=0.5, label=d["label"])

    ax.axvline(0.05, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(0.2, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Position Error (m)")
    ax.set_ylabel("Rotation Error (rad)")
    ax.set_title("Final Error: Position vs Rotation (numbers = test ID)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "error_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] error_scatter.png saved")

    # ── Normalized distance to target ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(test_indices) * 0.8), 5))
    x_t = np.arange(len(test_indices))
    w_t = 0.8 / n_models

    for i, d in enumerate(all_data):
        norms = []
        for tid in test_indices:
            match = [r for r in d["rows"] if int(r["test_index"]) == tid]
            if match:
                pe = float(match[0]["pos_err"])
                re = float(match[0]["rot_err"])
                norms.append(max(pe / 0.05, re / 0.2))
            else:
                norms.append(0.0)
        offset = (i - n_models / 2 + 0.5) * w_t
        bars = ax.bar(x_t + offset, norms, w_t, label=d["label"], color=model_colors[i],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, norms):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
               label="Pass threshold (1.0)")
    ax.set_ylabel("Normalized Distance (max of pos/0.05, rot/0.2)")
    ax.set_title("Normalized Distance to Target — below 1.0 = passed")
    ax.set_xticks(x_t)
    ax.set_xticklabels(test_short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "normalized_distance.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] normalized_distance.png saved")

    # ── Average error by difficulty ───────────────────────────────────────────
    fig, (ax_pos, ax_rot) = plt.subplots(1, 2, figsize=(12, 5))
    x_d = np.arange(len(difficulties))
    w_d = 0.8 / n_models

    for i, d in enumerate(all_data):
        avg_pe, avg_re = [], []
        for diff in difficulties:
            dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
            avg_pe.append(np.mean([float(r["pos_err"]) for r in dr]) if dr else 0.0)
            avg_re.append(np.mean([float(r["rot_err"]) for r in dr]) if dr else 0.0)
        offset = (i - n_models / 2 + 0.5) * w_d
        ax_pos.bar(x_d + offset, avg_pe, w_d, label=d["label"], color=model_colors[i],
                   edgecolor="white", linewidth=0.5)
        ax_rot.bar(x_d + offset, avg_re, w_d, label=d["label"], color=model_colors[i],
                   edgecolor="white", linewidth=0.5)

    ax_pos.axhline(0.05, color="red", linestyle="--", alpha=0.6)
    ax_pos.set_ylabel("Avg Position Error (m)")
    ax_pos.set_title("Avg Position Error by Difficulty")
    ax_pos.set_xticks(x_d)
    ax_pos.set_xticklabels([d.title() for d in difficulties])
    ax_pos.legend(fontsize=8)
    ax_pos.grid(axis="y", alpha=0.3)

    ax_rot.axhline(0.2, color="red", linestyle="--", alpha=0.6)
    ax_rot.set_ylabel("Avg Rotation Error (rad)")
    ax_rot.set_title("Avg Rotation Error by Difficulty")
    ax_rot.set_xticks(x_d)
    ax_rot.set_xticklabels([d.title() for d in difficulties])
    ax_rot.legend(fontsize=8)
    ax_rot.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "avg_error_by_difficulty.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] avg_error_by_difficulty.png saved")

    # ── Per-test comparison table as text ─────────────────────────────────────
    table_path = os.path.join(args.out_dir, "per_test_comparison.txt")
    with open(table_path, "w") as f:
        header = f"{'Test':>4s} {'Name':15s} {'Type':8s} " + "  ".join(f"{l:>18s}" for l in model_labels)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        _test_meta = {}
        for d in all_data:
            for r in d["rows"]:
                tid = int(r["test_index"])
                if tid not in _test_meta:
                    _test_meta[tid] = (r["test_name"].split(" #")[0], r.get("test_type", ""))
        for test_idx in sorted(_test_meta.keys()):
            tname, ttype = _test_meta[test_idx]
            line = f"{test_idx:4d} {tname:15s} {ttype:8s} "
            for d in all_data:
                r = [r for r in d["rows"] if int(r["test_index"]) == test_idx]
                if r:
                    r = r[0]
                    s = "PASS" if r["success"] == "1" else "FAIL"
                    _cov = float(r.get("area_coverage", 0.0))
                    line += f"  {s:>6s} P={r['pushes_used']:>2s} E={float(r['pos_err']):.3f}m C={_cov:.0f}%"
                else:
                    line += f"  {'---':>20s}"
            f.write(line + "\n")
    print(f"[OK] Per-test comparison saved to {table_path}")

    # ── Summary markdown table ────────────────────────────────────────────────
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Validation Summary\n\n")
        f.write("| Model | SR | Avg Pushes | Avg PosErr | Avg RotErr | Avg Cov | Tests |\n")
        f.write("|-------|----|-----------|------------|----------|--------|-------|\n")
        for d in all_data:
            m = d["metrics"]
            f.write(f"| {d['label']} | {m['sr']:.1f}% | {m['avg_pushes']:.1f} | "
                    f"{m['avg_pos_err']:.3f} m | {m['avg_rot_err']:.3f} rad | {m['avg_coverage']:.1f}% | {m['total']} |\n")
        f.write("\n## By Difficulty\n\n")
        f.write("| Model | Easy SR | Medium SR | Hard SR | Easy Pushes | Medium Pushes | Hard Pushes |\n")
        f.write("|-------|---------|-----------|---------|-------------|---------------|------------|\n")
        for d in all_data:
            parts = [d["label"]]
            for diff in ["easy", "medium", "hard"]:
                dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
                dm = compute_metrics(dr)
                parts.append(f"{dm['sr']:.1f}%")
            for diff in ["easy", "medium", "hard"]:
                dr = [r for r in d["rows"] if difficulty_from_name(r["test_name"]) == diff]
                dm = compute_metrics(dr)
                parts.append(f"{dm['avg_pushes']:.1f}")
            f.write("| " + " | ".join(parts) + " |\n")
        f.write("\n## By Test Type\n\n")
        f.write("| Model | Pos-Only SR | Pos+Rot SR | Pos-Only Pushes | Pos+Rot Pushes |\n")
        f.write("|-------|-------------|-----------|-----------------|---------------|\n")
        for d in all_data:
            parts = [d["label"]]
            for tt in ["pos_only", "pos_rot"]:
                tr = [r for r in d["rows"] if r.get("test_type", "") == tt]
                tm = compute_metrics(tr)
                parts.append(f"{tm['sr']:.1f}%")
            for tt in ["pos_only", "pos_rot"]:
                tr = [r for r in d["rows"] if r.get("test_type", "") == tt]
                tm = compute_metrics(tr)
                parts.append(f"{tm['avg_pushes']:.1f}")
            f.write("| " + " | ".join(parts) + " |\n")
    print(f"[OK] Summary markdown saved to {md_path}")

    print(f"\n[Done] All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
