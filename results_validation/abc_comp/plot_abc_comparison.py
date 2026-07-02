"""
Three cross-environment comparison plots for Models A, B, C.
Reads Isaac + Gym HPC validation CSVs, generates:
  p1_sr_bars.png          — Side-by-side SR bars (Isaac vs Gym, grouped by pos_only/pos_rot)
  p2_batch_bottleneck.png — Batch-size vs SR scatter with annotated push budgets
  p3_heatmap.png          — Per-scene pass/fail heatmap (30 scenes × 6 model×env combos)

Usage: python plot_abc_comparison.py
"""

import csv
import os
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "A Isaac":   ("../A_simp/20_isaac_30t.csv",    "Isaac 528 GPU", 7_920, 19.0, "#0072B2"),
    "B Isaac":   ("../B_curr/28_isaac_30t.csv",    "Isaac 528 GPU", 7_920, 20.6, "#D55E00"),
    "C Isaac":   ("../C_asp/26_isaac.csv",         "Isaac 528 GPU", 7_920, 60.2, "#009E73"),
    "E Isaac":   ("../E_asp_dpose/26_isaac.csv",   "Isaac 528 GPU", 7_920, 60.2, "#CC79A7"),
    "F Isaac":   ("../F_asp_disc/26_isaac.csv",    "Isaac 528 GPU", 7_920, 58.6, "#F0E442"),
    "G Isaac":   ("../G_tasp_dpose/26_isaac.csv",  "Isaac 528 GPU", 7_920, 30.1, "#56B4E9"),
    "H Isaac":   ("../H_tasp_disc/26_isaac.csv",   "Isaac 528 GPU", 7_920, 31.7, "#E69F00"),
    "A Gym":     ("../A_simp/hpc_gym_a_valid.csv",  "Gym HPC 32 CPU", 960, 18.0, "#0072B2"),
    "B Gym":     ("../B_curr/hpc_gym_b_valid.csv",  "Gym HPC 32 CPU", 960, 18.0, "#D55E00"),
    "C Gym":     ("../C_asp/hpc_gym_c_valid.csv",   "Gym HPC 32 CPU", 480, 4.2,  "#009E73"),
    "A Gym local": ("../A_simp/gym_gympusht.csv",   "Gym local 6 CPU", 90, 0.14, "#0072B2"),
    "B Gym local": ("../B_curr/gym_gympusht.csv",   "Gym local 6 CPU", 90, 0.14, "#D55E00"),
    "C Gym local": ("../C_asp/gympusht.csv",        "Gym local 1 CPU", 90, 0.10, "#009E73"),
}


def load_csv(rel_path):
    path = os.path.join(OUT_DIR, rel_path)
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def compute_model_stats(rows):
    scene_ok, trial_ok = 0, 0
    pos_only_ok, pos_rot_ok = 0, 0
    pos_only_n, pos_rot_n = 0, 0
    pos_errs, rot_errs, pushes = [], [], []
    scene_results = {}  # test_index -> (pass, pos_err, rot_err, pushes)

    for r in rows:
        pe = float(r["pos_err"])
        re = float(r["rot_err"])
        pu = int(r["pushes_used"])
        sc = int(r["success"])
        tt = r.get("test_type", "pos_rot")
        ti = int(r["test_index"])
        pos_errs.append(pe)
        rot_errs.append(re)
        pushes.append(pu)
        trial_ok += int(r.get("success_count", 0))
        if sc:
            scene_ok += 1
        if tt == "pos_only":
            pos_only_n += 1
            if sc:
                pos_only_ok += 1
        else:
            pos_rot_n += 1
            if sc:
                pos_rot_ok += 1
        scene_results[ti] = (sc == 1, pe, re, pu)

    n = len(rows)
    return {
        "scene_sr": scene_ok / n * 100 if n else 0,
        "trial_sr": trial_ok / (n * 20) * 100 if n else 0,
        "pos_only_sr": pos_only_ok / pos_only_n * 100 if pos_only_n else 0,
        "pos_rot_sr": pos_rot_ok / pos_rot_n * 100 if pos_rot_n else 0,
        "pos_err": statistics.mean(pos_errs) if pos_errs else 0,
        "rot_err": statistics.mean(rot_errs) if rot_errs else 0,
        "avg_pushes": statistics.mean(pushes) if pushes else 0,
        "scene_results": scene_results,
        "n_scenes": n,
        "n_pos_only": pos_only_n,
        "n_pos_rot": pos_rot_n,
    }


def difficulty_from_index(idx):
    if 1 <= idx <= 10:
        return "rotation"
    elif 11 <= idx <= 20:
        return "pos_only"
    else:
        return "pos_rot"


# ──────────────────────────────────────────────────────────
def plot1_sr_bars():
    """Side-by-side SR bars: Isaac vs Gym HPC, grouped by Position-only / Position+Rotation / Overall."""
    keys_a = ["A Isaac", "A Gym", "B Isaac", "B Gym", "C Gym"]

    stats = {}
    for k in keys_a:
        rows = load_csv(MODELS[k][0])
        stats[k] = compute_model_stats(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Cross-Environment Validation SR — Isaac Lab vs Gym-pusht HPC\nSame thesis gate, same 30 T-block scenes, same total push budget (~18–20M for PPO-PBRS / PPO-Curriculum)", fontsize=12, fontweight="bold")

    def draw_grouped(ax, keys_subset, xlabels, env_label):
        x = np.arange(len(keys_subset))
        w = 0.35
        pos_vals   = [stats[k]["pos_only_sr"] for k in keys_subset]
        posrot_vals = [stats[k]["pos_rot_sr"] for k in keys_subset]
        total_vals  = [stats[k]["scene_sr"] for k in keys_subset]
        colors_sub  = [MODELS[k][4] for k in keys_subset]

        b1 = ax.bar(x - w/2 - w/4, pos_vals, w/2, label="Position-only", color=colors_sub, edgecolor="white", alpha=0.85)
        for bar, val, col in zip(b1, pos_vals, colors_sub):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)

        b2 = ax.bar(x - w/2 + w/4, posrot_vals, w/2, label="Position + Rotation", color=colors_sub, edgecolor="white", alpha=0.40, hatch="//")
        for bar, val, col in zip(b2, posrot_vals, colors_sub):
            ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color=col)

        b3 = ax.bar(x + w/2, total_vals, w/2, label="Overall", color=colors_sub, edgecolor="white", alpha=0.95, linewidth=1.5)
        for bar, val, col in zip(b3, total_vals, colors_sub):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel("Scene Success Rate (%)", fontsize=10)
        ax.set_title(env_label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    draw_grouped(ax1, ["A Isaac", "B Isaac"], ["PPO-PBRS", "PPO-Curriculum"],
                 "Isaac Lab (528 GPU, batch 7,920)\nPPO-PBRS 19M / PPO-Curriculum 20.6M pushes")
    draw_grouped(ax2, ["A Gym", "B Gym", "C Gym"], ["PPO-PBRS", "PPO-Curriculum", "ASP"],
                 "Gym-pusht HPC (32 CPU, batch 960)\nPPO-PBRS 18M / PPO-Curriculum 18M / ASP 4.2M pushes")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "p1_sr_bars.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")


# ──────────────────────────────────────────────────────────
def plot2_batch_bottleneck():
    """Batch-size vs SR scatter: shows batch size drives performance, not push count."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.suptitle("Performance Bottleneck: Batch Size, Not Push Count\n(Matched total pushes, diverging SR — the batch gap explains the difference)", fontsize=12, fontweight="bold")

    pairs = [
        ("A Isaac", "A Gym", "A Gym local"),
        ("B Isaac", "B Gym", "B Gym local"),
        ("C Isaac", "C Gym", "C Gym local"),
    ]

    for isaac_key, gym_key, gymlocal_key in pairs:
        ir = load_csv(MODELS[isaac_key][0])
        gr = load_csv(MODELS[gym_key][0])
        lr = load_csv(MODELS[gymlocal_key][0])
        is_ = compute_model_stats(ir)
        gs  = compute_model_stats(gr)
        ls  = compute_model_stats(lr)

        ib = MODELS[isaac_key][2]
        gb = MODELS[gym_key][2]
        lb = MODELS[gymlocal_key][2]
        ip = MODELS[isaac_key][3]
        gp = MODELS[gym_key][3]
        lp = MODELS[gymlocal_key][3]
        col = MODELS[isaac_key][4]

        color_isaac = col
        color_gym = col

        pts_x = [lb, gb, ib]
        pts_y = [ls["scene_sr"], gs["scene_sr"], is_["scene_sr"]]
        sizes = [lp * 15, gp * 15, ip * 15]

        ax.plot(pts_x, pts_y, "o-", color=col, linewidth=2, markersize=8, alpha=0.8)
        for i, (x, y, sz) in enumerate(zip(pts_x, pts_y, sizes)):
            ax.plot(x, y, "o", color=col, markersize=max(sz, 6), alpha=0.7, markeredgecolor="white", markeredgewidth=1)

        label = f"Model {isaac_key[0]}"
        batch_str = f"{int(ib):,}"
        ax.annotate(f"{label}\n{batch_str}/update", xy=(ib, is_["scene_sr"]),
                    xytext=(ib + 200, is_["scene_sr"] + 3),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                    fontsize=9, color=col, fontweight="bold", ha="left")

        if gym_key[0] == "A":
            ax.annotate(f"18M pushes", xy=(gb, gs["scene_sr"]),
                        xytext=(gb + 80, gs["scene_sr"] + 7),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1),
                        fontsize=7, color=col, ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Batch Size (transitions per PPO update, log scale)", fontsize=10)
    ax.set_ylabel("Validation Scene Success Rate (%)", fontsize=10)
    ax.set_ylim(-5, 95)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([90, 480, 960, 7920])
    ax.set_xticklabels(["90\n(local 6CPU)", "480\n(ASP CPU)", "960\n(HPC 32CPU)", "7,920\n(Isaac 528GPU)"], fontsize=8)

    ax.axvspan(90, 480, alpha=0.05, color="red", label="insufficient batch")
    ax.axvspan(480, 960, alpha=0.05, color="orange", label="marginal batch")
    ax.axvspan(960, 7920, alpha=0.05, color="green", label="sufficient batch")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "p2_batch_bottleneck.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")


# ──────────────────────────────────────────────────────────
def plot3_heatmap():
    """Per-scene success-count heatmap: 30 scenes × model×env combos.
    Colour = continuous red→yellow→green gradient (0/20 → 20/20 trials passed).
    """
    keys = ["A Isaac", "B Isaac", "A Gym", "B Gym", "C Gym"]
    labels = ["PPO-PBRS\nIsaac", "PPO-Curriculum\nIsaac", "PPO-PBRS\nGym", "PPO-Curriculum\nGym", "ASP\nGym"]

    all_results = {}
    for k in keys:
        rows = load_csv(MODELS[k][0])
        sc = {}
        for r in rows:
            ti = int(r["test_index"])
            sc[ti] = int(r.get("success_count", 0))
        all_results[k] = sc

    n_models = len(keys)
    n_scenes = 30
    data = np.zeros((n_models, n_scenes), dtype=float)
    annotations = np.empty((n_models, n_scenes), dtype=object)

    for mi, k in enumerate(keys):
        for si in range(1, n_scenes + 1):
            val = all_results[k].get(si, 0)
            data[mi, si - 1] = val
            annotations[mi, si - 1] = f"{val}" if val > 0 else "0"

    fig, ax = plt.subplots(figsize=(22, 5.5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=20, aspect="auto", interpolation="nearest")

    for mi in range(n_models):
        for si in range(n_scenes):
            val = data[mi, si]
            text_color = "white" if val >= 15 else "black"
            ax.text(si, mi, f"{int(val)}", ha="center", va="center",
                    fontsize=6.5, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_scenes))
    scene_names = []
    ref = load_csv(MODELS["A Isaac"][0])
    for r in ref:
        ti = int(r["test_index"])
        nm = r["test_name"]
        nm = nm.replace(" #", "")
        if " " in nm:
            parts = nm.split(" ", 1)
            nm = parts[0] + " " + parts[1][:8]
        scene_names.append(nm)
    ax.set_xticklabels(scene_names, fontsize=6, rotation=45, ha="right")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=9)

    for x in [9.5, 19.5]:
        ax.axvline(x, color="black", linewidth=2.0, linestyle="-")

    fig.suptitle("Per-Scene Trial Success Rate — Isaac vs Gym HPC (30 T-block scenes, 20 trials each)\nColour: red=0 trials passed → yellow=10 → green=20",
                 fontsize=13, fontweight="bold", y=0.98)

    cbar = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.02, ticks=[0, 5, 10, 15, 20])
    cbar.set_label("Trials passed (of 20)", fontsize=9)

    ax.text(4.5, -1.0, "ROTATION SCENES", ha="center", fontsize=8, fontweight="bold",
            transform=ax.transData)
    ax.text(14.5, -1.0, "POS-ONLY SCENES", ha="center", fontsize=8, fontweight="bold",
            transform=ax.transData)
    ax.text(24.5, -1.0, "POS+ROT SCENES", ha="center", fontsize=8, fontweight="bold",
            transform=ax.transData)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    p = os.path.join(OUT_DIR, "p3_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")


def main():
    plot1_sr_bars()
    plot3_heatmap()
    print(f"\n[Done] All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
