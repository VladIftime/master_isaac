#!/usr/bin/env python3
"""Generate 5 novel plots for presentation from existing training + validation CSVs."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/vladi/IsaacLab/master_isaac/asyncDualPlayPPO/runs")
TRAIN_CSV = BASE / "ppo_pbrs_reward/26.06.20/runs/anal_26.06.18/csv"
VAL_20 = BASE / "ppo_pbrs_reward/26.06.12/runs"
VAL_30 = BASE / "ppo_pbrs_reward/26.06.20/runs"
OUT = Path("/home/vladi/IsaacLab/master_isaac/literature/paper-async/presentation/figures")

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

def load_csv(subdir, name):
    return pd.read_csv(TRAIN_CSV / f"{subdir}_{name}.csv")


# ============================================================
# PLOT 1 — PBRS Signal vs Validation Performance (Slide 5b)
# ============================================================
def plot1():
    si_pos = load_csv("Push_PPO-Si", "Reward_Dense_Pos")
    si_rot = load_csv("Push_PPO-Si", "Reward_Dense_Rot")

    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(si_pos['step']/1000, si_pos['value'], color='#1f77b4', lw=1.2, label='Dense Pos Reward')
    ax1.plot(si_rot['step']/1000, si_rot['value'], color='#ff7f0e', lw=1.2, label='Dense Rot Reward')
    ax1.set_xlabel('Training Iterations (×1000)')
    ax1.set_ylabel('PBRS Dense Reward')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.set_title('Model A: PBRS Reward Signal Over Training')

    # Mark where validation was run (~3000 iters)
    ax1.axvline(x=2.83, color='green', linestyle='--', lw=1.2)
    ax1.text(2.84, ax1.get_ylim()[1]*0.95, 'Validation\n(80% SR)', fontsize=7, color='green', va='top')
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plot1_pbrs_signal_vs_performance.png", bbox_inches='tight')
    plt.close(fig)
    print("Plot 1 saved.")


# ============================================================
# PLOT 2 — Cross-Model PBRS Comparison (Slide 7)
# ============================================================
def plot2():
    si_pos = load_csv("Push_PPO-Si", "Reward_Dense_Pos")
    cr_pos = load_csv("Push_PPO-Cr", "PBRS_DensePos")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(si_pos['step']/1000, si_pos['value'], color='#1f77b4', lw=1.2, label='Model A: PBRS Dense Pos')
    ax.plot(cr_pos['step']/1000, cr_pos['value'], color='#d62728', lw=1.2, label='Model B: PBRS Dense Pos (curriculum)')
    ax.set_xlabel('Training Iterations (×1000)')
    ax.set_ylabel('Dense Position Reward')
    ax.legend(fontsize=7)
    ax.set_title('Identical PBRS Signal — Different Outcomes')
    ax.annotate('Same reward, curriculum\nnever activates rotation', xy=(1.5, 1.5), xytext=(1.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='gray')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plot2_cross_model_pbrs_comparison.png", bbox_inches='tight')
    plt.close(fig)
    print("Plot 2 saved.")


# ============================================================
# PLOT 3 — Validation Failure Taxonomy (Slide 12)
# ============================================================
def plot3():
    # Load validation data
    simp = pd.read_csv(VAL_20 / "hpc_pbrs_simp_528env/results_simp.csv")
    curr = pd.read_csv(VAL_20 / "hpc_pbrs_curr_528env/results_curr.csv")
    simp30 = pd.read_csv(VAL_30 / "hpc_pbrs_simp_528env/results_valid_simp_dpose.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    def _scat(ax, df, title, color):
        diff = df['test_type'].map({'pos_only': 'easy', 'pos_rot': 'hard'}).fillna('med')
        markers = {'easy': 'o', 'med': 's', 'hard': '^'}
        colors_map = {'easy': '#2ca02c', 'med': '#ff7f0e', 'hard': '#d62728'}
        for d in ['easy', 'med', 'hard']:
            mask = diff == d
            if mask.any():
                ax.scatter(df.loc[mask, 'pos_err'], df.loc[mask, 'rot_err'],
                          c=colors_map[d], marker=markers[d], s=35, alpha=0.7,
                          edgecolors='k', linewidths=0.3, label=d)
        ax.axvline(x=0.05, color='green', ls='--', lw=1, alpha=0.6)
        ax.axhline(y=0.2, color='green', ls='--', lw=1, alpha=0.6)
        ax.fill_between([0, 0.05], 0, 0.2, color='green', alpha=0.08)
        ax.set_xlim(-0.02, max(df['pos_err'].max(), 0.85))
        ax.set_ylim(-0.1, max(df['rot_err'].max(), 3.2))
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Rotation Error (rad)')
        ax.set_title(title, fontsize=10)
        if title == 'Model A (PBRS Only)\n80% SR':
            ax.legend(fontsize=6, loc='upper right')

    _scat(axes[0], simp30, 'Model A (PBRS Only)\n80% SR', '#1f77b4')
    _scat(axes[1], curr, 'Model B (+Curriculum)\n40% SR', '#ff7f0e')
    _scat(axes[2], simp, 'Model A (20-test)\n55% SR', '#2ca02c')

    fig.suptitle('Validation Failure Taxonomy — Success Rectangle (pos<5cm, rot<0.2rad)', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "plot3_failure_taxonomy.png", bbox_inches='tight')
    plt.close(fig)
    print("Plot 3 saved.")


# ============================================================
# PLOT 4 — Alice Goal Quality vs Bob Success (ASP Diagnostic)
# ============================================================
def plot4():
    alice_valid = load_csv("Push_PPO-ASP", "Metrics_Alice_GoalValidityRate")
    alice_disp = load_csv("Push_PPO-ASP", "Metrics_Alice_MeanDisp3D")
    bob_sr = load_csv("Push_PPO-ASP", "Metrics_Bob_SuccessRate")

    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    color_a = '#1f77b4'
    ax1.plot(alice_valid['step']/1000, alice_valid['value']*100, color=color_a, lw=1.2, label="Alice Goal Validity Rate")
    ax1.plot(alice_disp['step']/1000, alice_disp['value']*100, color=color_a, lw=0.8, ls='--', label="Alice Mean Disp. (×100)")
    ax1.set_ylabel('Alice Metrics (%)', color=color_a)
    ax1.tick_params(axis='y', labelcolor=color_a)
    ax1.legend(loc='upper left', fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(bob_sr['step']/1000, bob_sr['value']*100, color='#d62728', lw=1.5, label='Bob Combined Success Rate')
    ax2.set_ylabel('Bob Success Rate (%)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.legend(loc='center right', fontsize=7)

    ax1.set_xlabel('Training Iterations (×1000)')
    ax1.set_title('ASP Diagnostic: Alice Proposes Good Goals — Bob Cannot Achieve Them')
    ax1.annotate('Alice: 79% validity,\nlow displacement', xy=(2.5, 75), fontsize=7, color=color_a)
    ax2.annotate('Bob: 0.07% SR\n(dead gradient)', xy=(2.5, 0.1), fontsize=7, color='#d62728',
                xytext=(1.5, 0.3), arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "plot4_alice_vs_bob_diagnostic.png", bbox_inches='tight')
    plt.close(fig)
    print("Plot 4 saved.")


# ============================================================
# PLOT 5 — Reward Diet: Stacked Decomposition (Slide 5a)
# ============================================================
def plot5():
    cr_pos = load_csv("Push_PPO-Cr", "PBRS_DensePos")
    cr_comp = load_csv("Push_PPO-Cr", "PBRS_CompletionRate")
    cr_rotbonus = load_csv("Push_PPO-Cr", "PBRS_RotBonusRate")
    cr_tip = load_csv("Push_PPO-Cr", "PBRS_TipRate")

    # Align all to same step index (they may differ slightly)
    steps = cr_pos['step']
    pos = cr_pos['value'].values
    comp = np.interp(steps, cr_comp['step'], cr_comp['value'])
    rotbonus = np.interp(steps, cr_rotbonus['step'], cr_rotbonus['value'])
    tip = np.interp(steps, cr_tip['step'], cr_tip['value'])

    # Downsample for cleaner plot
    n = len(steps) // 100
    steps = steps[::n] / 1000
    pos = pos[::n]
    comp = comp[::n]
    rotbonus = rotbonus[::n]
    tip = tip[::n]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.fill_between(steps, 0, pos, color='#1f77b4', alpha=0.7, label='Dense Position Reward')
    ax.fill_between(steps, pos, pos + comp, color='#2ca02c', alpha=0.7, label='Completion Bonus')
    ax.fill_between(steps, pos + comp, pos + comp + rotbonus, color='#ff7f0e', alpha=0.7, label='Rotation Bonus')
    ax.fill_between(steps, pos + comp + rotbonus, pos + comp + rotbonus - tip, color='#d62728', alpha=0.7, label='Tip Penalty')
    ax.set_xlabel('Training Iterations (×1000)')
    ax.set_ylabel('Reward per Iteration')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('Model A: PBRS Reward Decomposition Over Training')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "plot5_reward_diet.png", bbox_inches='tight')
    plt.close(fig)
    print("Plot 5 saved.")


# ============================================================
# NEW FIGURES — helpers
# ============================================================
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle, Ellipse

VAL = {
    'A': "ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/results_valid_simp_dpose.csv",
    'B': "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_curr_528env/results_curr.csv",
    'C': "ppo_pbrs_reward/26.06.12/runs/hpc_pbrs_asp_528env/results_asp.csv",
    'E': "ppo_pbrs_reward/26.06.24/runs/hpc_pbrs_asp_dpose_528env/results_valid_pbrs_asp_dpose.csv",
    'F': "ppo_pbrs_reward/26.06.24/runs/hpc_pbrs_asp_disc_528env/results_valid_pbrs_asp_disc.csv",
    'G': "ppo_pbrs_reward/26.06.24/runs/hpc_pbrs_tasp_dpose_528env/results_valid_pbrs_tasp_dpose.csv",
    'H': "ppo_pbrs_reward/26.06.24/runs/hpc_pbrs_tasp_disc_528env/results_valid_pbrs_tasp_disc.csv",
}
MODEL_TITLE = {
    'A': 'Model A\nPBRS Single-Agent', 'B': 'Model B\n+Curriculum', 'C': 'Model C\nASP',
    'E': 'Model E\nASP+d_pose', 'F': 'Model F\nASP disc', 'G': 'Model G\nTASP', 'H': 'Model H\nTASP disc',
}
CAT_LABEL = {'disc_pos': 'Disc', 'pos_only': 'T-pos', 'pos_rot': 'T-pos+rot'}
CAT_COLOR = {'disc_pos': '#2ca02c', 'pos_only': '#ff7f0e', 'pos_rot': '#d62728'}


def _load_train(prefix, metric):
    p = TRAIN_CSV / f"{prefix}_{metric}.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _smooth(y, w=21):
    s = pd.Series(np.asarray(y, dtype=float))
    if len(s) < 3:
        return s.values
    return s.rolling(w, min_periods=1, center=True).mean().values


def _load_val(model):
    p = BASE / VAL[model]
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# ============================================================
# MASTER SR RACE (full-bleed)
# ============================================================
def plot_master_sr():
    series = [
        ('A', 'Push_PPO-Si', 'Metrics_SuccessRate', '#1f77b4'),
        ('B', 'Push_PPO-Cr', 'Metrics_SuccessRate', '#ff7f0e'),
        ('C', 'Push_PPO-ASP', 'Metrics_Bob_SuccessRate', '#2ca02c'),
        ('D', 'Push_PPO-ASP-NGE', 'Metrics_Bob_SuccessRate', '#d62728'),
    ]
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    for name, pre, met, col in series:
        df = _load_train(pre, met)
        if df is None:
            continue
        ax.plot(df['step'], _smooth(df['value'].values) * 100, color=col, lw=2.4,
                label=f'Model {name}')
    ax.set_xlabel('Training Iteration', fontsize=15)
    ax.set_ylabel('Combined Success Rate (%)', fontsize=15)
    ax.set_title('Training Success Rate — Only Model A Climbs; Curriculum & ASP Stay Flat',
                 fontsize=18, pad=14)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "cmp_master_sr.png", bbox_inches='tight')
    plt.close(fig)
    print("Master SR saved.")


# ============================================================
# 8-MODEL VALIDATION TAXONOMY (full-bleed)
# ============================================================
def plot_val_taxonomy_8():
    models = ['A', 'B', 'C', 'E', 'F', 'G', 'H']
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.0))
    axes = axes.ravel()
    for i, m in enumerate(models):
        ax = axes[i]
        df = _load_val(m)
        if df is None:
            ax.text(0.5, 0.5, f'Model {m}\n(no data)', ha='center', va='center')
            ax.set_axis_off()
            continue
        sr = 100.0 * df['success'].mean()
        for cat in ['disc_pos', 'pos_only', 'pos_rot']:
            sub = df[df['test_type'] == cat]
            if len(sub):
                ax.scatter(sub['pos_err'], sub['rot_err'], c=CAT_COLOR[cat], s=30,
                           alpha=0.75, edgecolors='k', linewidths=0.3, label=CAT_LABEL[cat])
        ax.axvline(0.05, color='green', ls='--', lw=1, alpha=0.6)
        ax.axhline(0.2, color='green', ls='--', lw=1, alpha=0.6)
        ax.fill_between([0, 0.05], 0, 0.2, color='green', alpha=0.1)
        ax.set_xlim(-0.02, 0.9)
        ax.set_ylim(-0.1, 3.3)
        ax.set_title(f'{MODEL_TITLE[m].splitlines()[0]} — {sr:.0f}% SR', fontsize=10)
        ax.set_xlabel('Pos err (m)', fontsize=8)
        ax.set_ylabel('Rot err (rad)', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_axis_off()
    axes[-1].text(0.1, 0.6, 'Green box = success\n(pos<5cm, rot<0.2rad)\n\nModel A fills the box;\nASP variants scatter\noutside it.',
                  fontsize=11, va='center')
    fig.suptitle('Validation Failure Taxonomy — All Models', fontsize=17, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "val_taxonomy_8models.png", bbox_inches='tight')
    plt.close(fig)
    print("Taxonomy 8 saved.")


# ============================================================
# VALIDATION SR BAR CHART (by model x difficulty)
# ============================================================
def plot_val_sr_bars():
    models = ['A', 'B', 'C', 'E', 'F', 'G', 'H']
    cats = ['disc_pos', 'pos_only', 'pos_rot']
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(models))
    width = 0.26
    for j, cat in enumerate(cats):
        vals = []
        for m in models:
            df = _load_val(m)
            if df is None:
                vals.append(np.nan); continue
            sub = df[df['test_type'] == cat]
            vals.append(100.0 * sub['success'].mean() if len(sub) else np.nan)
        ax.bar(x + (j - 1) * width, np.nan_to_num(vals), width,
               color=CAT_COLOR[cat], label=CAT_LABEL[cat], edgecolor='k', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Model {m}' for m in models])
    ax.set_ylabel('Validation Success Rate (%)')
    ax.set_title('Validation SR by Object / Difficulty — Model A Dominates Every Category')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "val_sr_bars.png", bbox_inches='tight')
    plt.close(fig)
    print("Val SR bars saved.")


# ============================================================
# MODEL B — CURRICULUM NEVER FIRED
# ============================================================
def plot_modelB_curriculum_stuck():
    wrot = _load_train('Push_PPO-Cr', 'Curriculum_w_rot')
    pe = _load_train('Push_PPO-Cr', 'Metrics_PosError')
    fig, ax1 = plt.subplots(figsize=(8.5, 4.0))
    if pe is not None:
        ax1.plot(pe['step'], _smooth(pe['value'].values), color='#1f77b4', lw=1.6,
                 label='Position Error (m)')
    ax1.axhline(0.08, color='gray', ls='--', lw=1.4)
    ax1.text(ax1.get_xlim()[1] * 0.55, 0.085, 'trigger threshold 0.08 m', fontsize=8, color='gray')
    ax1.set_ylabel('Position Error (m)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xlabel('Training Iteration')
    ax2 = ax1.twinx()
    if wrot is not None:
        ax2.plot(wrot['step'], wrot['value'], color='#d62728', lw=2.0, label='$w_{rot}$ (curriculum)')
    ax2.set_ylabel('$w_{rot}$', color='#d62728')
    ax2.set_ylim(-0.5, 10.5)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_title('Model B: $w_{rot}$ Stuck at 0 — Curriculum Never Activated')
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "modelB_curriculum_stuck.png", bbox_inches='tight')
    plt.close(fig)
    print("Model B curriculum saved.")


# ============================================================
# ASP DEAD GRADIENT (full-bleed)
# ============================================================
def plot_asp_dead_gradient():
    bob = _load_train('Push_PPO-ASP', 'Loss_Bob_Surrogate')
    si = _load_train('Push_PPO-Si', 'Loss_Agent_Surrogate')
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    if si is not None:
        ax.plot(si['step'], _smooth(np.abs(si['value'].values)), color='#1f77b4', lw=2.4,
                label='Model A — |surrogate loss| (learning)')
    if bob is not None:
        ax.plot(bob['step'], _smooth(np.abs(bob['value'].values)), color='#d62728', lw=2.4,
                label='ASP Bob — |surrogate loss| ($\\approx$0, no gradient)')
    ax.set_xlabel('Training Iteration', fontsize=15)
    ax.set_ylabel('|PPO Surrogate Loss|', fontsize=15)
    ax.set_title('ASP Dead Gradient — Bob\u2019s Surrogate Loss Never Moves', fontsize=18, pad=14)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "asp_dead_gradient.png", bbox_inches='tight')
    plt.close(fig)
    print("ASP dead gradient saved.")


# ============================================================
# SPARSE REWARD STARVATION
# ============================================================
def plot_asp_sparse_starvation():
    comp = _load_train('Push_PPO-Cr', 'PBRS_CompletionRate')
    rb = _load_train('Push_PPO-Cr', 'PBRS_RotBonusRate')
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    if comp is not None:
        ax.plot(comp['step'], _smooth(comp['value'].values) * 100, color='#2ca02c', lw=1.6,
                label='Completion bonus fire-rate')
    if rb is not None:
        ax.plot(rb['step'], _smooth(rb['value'].values) * 100, color='#ff7f0e', lw=1.6,
                label='Rotation bonus fire-rate')
    ax.axhline(0.14, color='gray', ls='--', lw=1.2)
    ax.text(ax.get_xlim()[1] * 0.5, 0.16, 'sparse bonuses $\\approx$ 0.14% of pushes', fontsize=8, color='gray')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Fire-rate (% of pushes)')
    ax.set_title('Sparse-Reward Starvation — Bonuses Almost Never Fire')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "asp_sparse_starvation.png", bbox_inches='tight')
    plt.close(fig)
    print("Sparse starvation saved.")


# ============================================================
# BOB SKILLS vs COMBINED
# ============================================================
def plot_bob_skills_vs_combined():
    pos = _load_train('Push_PPO-ASP', 'Metrics_Bob_PositionSR')
    rot = _load_train('Push_PPO-ASP', 'Metrics_Bob_RotationSR')
    comb = _load_train('Push_PPO-ASP', 'Metrics_Bob_SuccessRate')
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for df, col, lab in [(pos, '#1f77b4', 'Position SR (individual)'),
                         (rot, '#ff7f0e', 'Rotation SR (individual)'),
                         (comb, '#d62728', 'Combined SR (gate)')]:
        if df is not None:
            ax.plot(df['step'], _smooth(df['value'].values) * 100, color=col, lw=1.8, label=lab)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Bob Has the Skills (66% / 65%) — But the Combined Gate Never Fires')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bob_skills_vs_combined.png", bbox_inches='tight')
    plt.close(fig)
    print("Bob skills saved.")


# ============================================================
# PUSHES-TO-SUCCESS HISTOGRAM
# ============================================================
def plot_val_pushes_hist():
    df = _load_val('A')
    if df is None:
        raise FileNotFoundError("Model A val csv missing")
    succ = df[df['success'] == 1]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bins = np.arange(0, 16) - 0.5
    for cat in ['disc_pos', 'pos_only', 'pos_rot']:
        sub = succ[succ['test_type'] == cat]
        if len(sub):
            ax.hist(sub['pushes_used'], bins=bins, alpha=0.6, color=CAT_COLOR[cat],
                    label=f'{CAT_LABEL[cat]} (median {int(sub["pushes_used"].median())})',
                    edgecolor='k', linewidth=0.3)
    ax.set_xlabel('Pushes to Success')
    ax.set_ylabel('Count (validation scenes)')
    ax.set_title('Model A: Pushes-to-Success by Difficulty')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "val_pushes_hist.png", bbox_inches='tight')
    plt.close(fig)
    print("Pushes hist saved.")


# ============================================================
# INDEPENDENCE 2x2 HEATMAP (full-bleed)
# ============================================================
def plot_independence_heatmap():
    data = np.array([[80.0, 17.3], [3.5, 0.07]])
    labels = [['80%\nModel A', '17.3%\nPush-PPO'], ['0\u20137%\nModels C\u2013H', '0.07%\nPush-ASP']]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=80, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['PBRS Reward', 'Ad-hoc Reward'], fontsize=15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Single-Agent', 'ASP'], fontsize=15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha='center', va='center', fontsize=16, fontweight='bold')
    ax.set_title('Reward \u00d7 Curriculum Are Independent Levers (Validation SR)', fontsize=18, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "independence_heatmap.png", bbox_inches='tight')
    plt.close(fig)
    print("Independence heatmap saved.")


# ============================================================
# DIAGRAM — TASK SCHEMATIC
# ============================================================
def diagram_task_schematic():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.add_patch(Rectangle((-0.70, -0.10), 1.40, 1.00, fill=True, fc='#f0ead6', ec='k', lw=1.5))
    ax.add_patch(Rectangle((-0.50, 0.25), 1.00, 0.45, fill=False, ec='#0066cc', lw=1.8, ls='--'))
    ax.text(0.0, 0.72, 'IK workspace', color='#0066cc', ha='center', fontsize=10)
    # object (T-block) and goal ghost
    obj = (-0.25, 0.40)
    goal = (0.30, 0.55)
    ax.add_patch(Rectangle((obj[0]-0.05, obj[1]-0.02), 0.10, 0.04, fc='#8B4513', ec='k'))
    ax.add_patch(Rectangle((obj[0]-0.015, obj[1]-0.06), 0.03, 0.08, fc='#8B4513', ec='k'))
    ax.add_patch(Rectangle((goal[0]-0.05, goal[1]-0.02), 0.10, 0.04, fc='#2ca02c', ec='g', alpha=0.4))
    ax.add_patch(Rectangle((goal[0]-0.015, goal[1]-0.06), 0.03, 0.08, fc='#2ca02c', ec='g', alpha=0.4))
    ax.text(goal[0], goal[1]+0.10, 'goal ghost', color='g', ha='center', fontsize=9)
    ax.text(obj[0], obj[1]-0.12, 'object', color='#8B4513', ha='center', fontsize=9)
    # approach (r, phi) and push (l, theta)
    start = (obj[0]-0.10, obj[1]-0.06)
    ax.add_patch(FancyArrowPatch(start, (obj[0]-0.05, obj[1]-0.02),
                                 arrowstyle='->', mutation_scale=16, color='#0066cc'))
    ax.text(start[0]-0.02, start[1]-0.03, 'approach $(r,\\phi)$', color='#0066cc', fontsize=9, ha='right')
    ax.add_patch(FancyArrowPatch(obj, (obj[0]+0.18, obj[1]+0.06),
                                 arrowstyle='->', mutation_scale=18, color='#d62728', lw=2))
    ax.text(obj[0]+0.20, obj[1]+0.08, 'push $(\\ell,\\theta)$', color='#d62728', fontsize=9)
    # robot base
    ax.add_patch(Circle((0.0, -0.05), 0.04, fc='gray', ec='k'))
    ax.text(0.0, -0.13, 'UR5e base', ha='center', fontsize=8)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.2, 1.0); ax.set_aspect('equal')
    ax.set_title('Planar Pushing Task — Object-Relative Action $(r,\\phi,\\ell,\\theta)$', fontsize=13)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / "diagram_task_schematic.png", bbox_inches='tight')
    plt.close(fig)
    print("Task schematic saved.")


# ============================================================
# DIAGRAM — LIMIT SURFACE COUPLING
# ============================================================
def diagram_limit_surface():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    # left: limit surface ellipse + twist normal
    axL.add_patch(Ellipse((0, 0), 1.6, 1.0, fill=False, ec='k', lw=1.8))
    wp = (0.566, 0.353)  # point on ellipse
    axL.plot(*wp, 'o', color='#d62728', ms=8)
    axL.add_patch(FancyArrowPatch((0, 0), wp, arrowstyle='->', mutation_scale=16, color='#1f77b4'))
    axL.text(0.28, 0.10, 'wrench $\\mathbf{w}$', color='#1f77b4', fontsize=10)
    nrm = (wp[0] + 0.45, wp[1] + 0.36)
    axL.add_patch(FancyArrowPatch(wp, nrm, arrowstyle='->', mutation_scale=16, color='#2ca02c', lw=2))
    axL.text(nrm[0]-0.1, nrm[1]+0.05, 'twist $\\mathbf{t}\\propto\\nabla f$', color='#2ca02c', fontsize=10)
    axL.set_xlim(-1.1, 1.4); axL.set_ylim(-0.8, 1.1); axL.set_aspect('equal'); axL.axis('off')
    axL.set_title('Limit Surface Normality\n(Goyal et al. 1991)', fontsize=12)
    # right: d_pose decomposition
    axR.add_patch(Rectangle((-0.05, -0.03), 0.10, 0.06, fc='#8B4513', ec='k'))
    axR.add_patch(Rectangle((0.45, 0.25), 0.10, 0.06, fc='#2ca02c', ec='g', alpha=0.4, angle=25))
    axR.add_patch(FancyArrowPatch((0, 0), (0.5, 0.28), arrowstyle='->', mutation_scale=16, color='#1f77b4'))
    axR.text(0.18, 0.05, '$(dx,dy)$', color='#1f77b4', fontsize=11)
    axR.text(0.5, 0.40, '$d\\theta$', color='#d62728', fontsize=11)
    axR.text(-0.05, -0.25, '$d_{pose}=\\sqrt{dx^2+dy^2+L^2 d\\theta^2}$', fontsize=12)
    axR.text(-0.05, -0.38, 'T-block $L=0.07$m  ·  Disc $L=0$m', fontsize=10, color='gray')
    axR.set_xlim(-0.2, 0.9); axR.set_ylim(-0.5, 0.6); axR.set_aspect('equal'); axR.axis('off')
    axR.set_title('SE(2) Unified Distance', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "diagram_limit_surface.png", bbox_inches='tight')
    plt.close(fig)
    print("Limit surface saved.")


# ============================================================
# DIAGRAM — ASP LOOP
# ============================================================
def diagram_asp_loop():
    fig, ax = plt.subplots(figsize=(10, 4.4))

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                                    fc=fc, ec='k', lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=11)

    box(0.02, 0.55, 0.22, 0.25, 'Alice\n(proposes goal)', '#cfe2f3')
    box(0.39, 0.55, 0.22, 0.25, 'Goal\n(SE(2) pose)', '#d9ead3')
    box(0.76, 0.55, 0.22, 0.25, 'Bob\n(achieves goal)', '#fce5cd')
    box(0.39, 0.12, 0.22, 0.22, 'Reward\n+5 fail / -1 win', '#f4cccc')
    box(0.76, 0.12, 0.22, 0.22, 'ABC buffer\n(imitation $\\beta$=0.5)', '#ead1dc')

    def arr(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16, color='k', lw=1.3))
    arr((0.24, 0.67), (0.39, 0.67))
    arr((0.61, 0.67), (0.76, 0.67))
    arr((0.87, 0.55), (0.55, 0.34))      # Bob outcome -> reward
    arr((0.39, 0.23), (0.13, 0.55))      # reward -> Alice
    arr((0.87, 0.34), (0.87, 0.55))      # ABC -> Bob (dashed)
    ax.text(0.5, 0.86, 'Asymmetric Self-Play Loop (Model C)', ha='center', fontsize=14)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.95); ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / "diagram_asp_loop.png", bbox_inches='tight')
    plt.close(fig)
    print("ASP loop saved.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot1()
    plot2()
    plot3()
    plot4()
    plot5()
    # --- new figures ---
    for fn in (plot_master_sr, plot_val_taxonomy_8, plot_val_sr_bars,
               plot_modelB_curriculum_stuck, plot_asp_dead_gradient,
               plot_asp_sparse_starvation, plot_bob_skills_vs_combined,
               plot_val_pushes_hist, plot_independence_heatmap,
               diagram_task_schematic, diagram_limit_surface, diagram_asp_loop):
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}")
    print("All plots generated.")
