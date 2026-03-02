#!/usr/bin/env python3
"""
Visualize training logs from ASP (Asymmetric Self-Play) training runs.

Parses structured update blocks from log files and generates multi-panel plots:
  - Bob: Success Rate, Reward, Position & Rotation Errors, ABC Loss
  - Alice: Reward, Value Loss, Surrogate Loss

Usage:
    python visualize_logs.py                          # defaults to all 3 log files
    python visualize_logs.py --files my_run.log
    python visualize_logs.py --files run1.log run2.log --labels "Run 1" "Run 2"
"""

import re
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.labelcolor': '#212529',
    'text.color': '#212529',
    'xtick.color': '#495057',
    'ytick.color': '#495057',
    'grid.color': '#dee2e6',
    'grid.alpha': 0.6,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#dee2e6',
    'font.size': 10,
})


def parse_log(filepath):
    """Parse a log file and extract Alice and Bob update metrics."""
    text = Path(filepath).read_text()

    alice = {'update': [], 'reward_mean': [], 'value_loss': [], 'surrogate_loss': [], 'episodes': []}
    bob = {'update': [], 'success_rate': [], 'reward_mean': [], 'value_loss': [],
           'surrogate_loss': [], 'abc_loss': [], 'pos_err': [], 'rot_err': [], 'episodes': []}

    # Parse ALICE UPDATE blocks
    alice_pattern = re.compile(
        r'ALICE UPDATE (\d+)\s*\n'
        r'={10,}\s*\n'
        r'\s*Rewards:\s*mean=([\d.eE+-]+)',
        re.MULTILINE
    )
    alice_loss_pattern = re.compile(
        r'ALICE UPDATE (\d+)\s*\n'
        r'={10,}\s*\n'
        r'\s*Rewards:.*?\n'
        r'\s*Losses:\s*value=([\d.eE+-]+)\s*\|\s*surrogate=([\d.eE+-]+)',
        re.MULTILINE
    )
    alice_eps_pattern = re.compile(
        r'ALICE UPDATE (\d+)\s*\n'
        r'={10,}\s*\n'
        r'.*?Outcomes:\s*(\d+)\s*episodes',
        re.MULTILINE | re.DOTALL
    )

    for m in alice_pattern.finditer(text):
        idx = int(m.group(1))
        alice['update'].append(idx)
        alice['reward_mean'].append(float(m.group(2)))

    for m in alice_loss_pattern.finditer(text):
        idx = int(m.group(1))
        # Only add if not already present (avoid duplicates)
        if idx not in [alice['value_loss'][i] for i in range(len(alice['value_loss'])) if i < len(alice['update']) and alice['update'][i] == idx]:
            pass
        alice['value_loss'].append(float(m.group(2)))
        alice['surrogate_loss'].append(float(m.group(3)))

    for m in alice_eps_pattern.finditer(text):
        alice['episodes'].append(int(m.group(2)))

    # Parse BOB UPDATE blocks
    bob_pattern = re.compile(
        r'BOB UPDATE (\d+)\s*\n'
        r'={10,}\s*\n'
        r'\s*Success Rate:\s*([\d.eE+-]+)\s*\((\d+)\s*eps\)\s*\n'
        r'\s*Rewards:\s*mean=([\d.eE+-]+)\s*\n'
        r'\s*Losses:\s*value=([\d.eE+-]+)\s*\|\s*surrogate=([\d.eE+-]+)\s*\|\s*ABC=([\d.eE+-]+)\s*\n'
        r'\s*Errors:\s*pos=([\d.eE+-]+)\s*\|\s*rot=([\d.eE+-]+)',
        re.MULTILINE
    )

    for m in bob_pattern.finditer(text):
        bob['update'].append(int(m.group(1)))
        bob['success_rate'].append(float(m.group(2)))
        bob['episodes'].append(int(m.group(3)))
        bob['reward_mean'].append(float(m.group(4)))
        bob['value_loss'].append(float(m.group(5)))
        bob['surrogate_loss'].append(float(m.group(6)))
        bob['abc_loss'].append(float(m.group(7)))
        bob['pos_err'].append(float(m.group(8)))
        bob['rot_err'].append(float(m.group(9)))

    # Parse [Bob Reward] events for histogram
    bob_rewards = []
    for m in re.finditer(r'\[Bob Reward\]\s*([+\-]?[\d.]+)', text):
        bob_rewards.append(float(m.group(1)))

    return alice, bob, bob_rewards


def plot_runs(runs, labels, save_path=None):
    """Plot all parsed runs on a shared figure."""
    colors = ['#d62828', '#003049', '#2a9d8f', '#f77f00', '#7209b7', '#4361ee']

    # --- Figure 1: Bob's Metrics ---
    fig_bob, axes_bob = plt.subplots(2, 3, figsize=(18, 9))
    fig_bob.suptitle("Bob's Training Progress", fontsize=16, fontweight='bold', color='#d62828')

    for i, (alice, bob, bob_rewards, label) in enumerate(zip(*zip(*[(r, l) for r, l in zip(runs, labels)])) if False else zip([r[1] for r in runs], [r[2] for r in runs], [r[0] for r in runs], labels)):
        # Wait — let me restructure this properly
        pass

    # Restructure: runs is list of (alice, bob, bob_rewards)
    for i, ((alice, bob, bob_rewards), label) in enumerate(zip(runs, labels)):
        c = colors[i % len(colors)]
        if not bob['update']:
            continue
        x = bob['update']

        axes_bob[0, 0].plot(x, bob['success_rate'], '-o', color=c, label=label, markersize=3, linewidth=1.5)
        axes_bob[0, 0].set_title('Success Rate')
        axes_bob[0, 0].set_ylabel('Rate')

        axes_bob[0, 1].plot(x, bob['reward_mean'], '-o', color=c, label=label, markersize=3, linewidth=1.5)
        axes_bob[0, 1].set_title('Mean Reward')

        axes_bob[0, 2].plot(x, bob['abc_loss'], '-o', color=c, label=label, markersize=3, linewidth=1.5)
        axes_bob[0, 2].set_title('ABC Loss')
        axes_bob[0, 2].axhline(y=-1.0, color='#d62828', linestyle='--', alpha=0.5, label='Broken (-1.0)' if i == 0 else None)

        axes_bob[1, 0].plot(x, bob['pos_err'], '-o', color=c, label=label, markersize=3, linewidth=1.5)
        axes_bob[1, 0].set_title('Position Error (m)')
        axes_bob[1, 0].set_xlabel('Bob Update')

        axes_bob[1, 1].plot(x, bob['rot_err'], '-o', color=c, label=label, markersize=3, linewidth=1.5)
        axes_bob[1, 1].set_title('Rotation Error (rad)')
        axes_bob[1, 1].set_xlabel('Bob Update')

        axes_bob[1, 2].plot(x, bob['value_loss'], '-', color=c, label=f'{label} value', linewidth=1.2, alpha=0.8)
        axes_bob[1, 2].plot(x, bob['surrogate_loss'], '--', color=c, label=f'{label} surr', linewidth=1.2, alpha=0.8)
        axes_bob[1, 2].set_title('PPO Losses')
        axes_bob[1, 2].set_xlabel('Bob Update')

    for ax in axes_bob.flat:
        ax.legend(fontsize=7)
        ax.grid(True)
    fig_bob.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Figure 2: Alice's Metrics ---
    fig_alice, axes_alice = plt.subplots(1, 3, figsize=(16, 5))
    fig_alice.suptitle("Alice's Training Progress", fontsize=16, fontweight='bold', color='#003049')

    for i, ((alice, bob, bob_rewards), label) in enumerate(zip(runs, labels)):
        c = colors[i % len(colors)]
        if not alice['update']:
            continue
        x = alice['update']

        axes_alice[0].plot(x, alice['reward_mean'], '-o', color=c, label=label, markersize=2, linewidth=1.2)
        axes_alice[0].set_title('Mean Reward')
        axes_alice[0].set_xlabel('Alice Update')

        if alice['value_loss']:
            ax_x = x[:len(alice['value_loss'])]
            axes_alice[1].plot(ax_x, alice['value_loss'], '-', color=c, label=label, linewidth=1.2)
        axes_alice[1].set_title('Value Loss')
        axes_alice[1].set_xlabel('Alice Update')

        if alice['surrogate_loss']:
            ax_x = x[:len(alice['surrogate_loss'])]
            axes_alice[2].plot(ax_x, alice['surrogate_loss'], '-', color=c, label=label, linewidth=1.2)
        axes_alice[2].set_title('Surrogate Loss')
        axes_alice[2].set_xlabel('Alice Update')

    for ax in axes_alice.flat:
        ax.legend(fontsize=8)
        ax.grid(True)
    fig_alice.tight_layout(rect=[0, 0, 1, 0.92])

    # --- Figure 3: Bob Reward Events ---
    any_rewards = any(r[2] for r in runs)
    if any_rewards:
        fig_rew, ax_rew = plt.subplots(figsize=(8, 4))
        fig_rew.suptitle("Bob Reward Events", fontsize=14, fontweight='bold', color='#2a9d8f')
        for i, ((alice, bob, bob_rewards), label) in enumerate(zip(runs, labels)):
            if bob_rewards:
                ax_rew.hist(bob_rewards, bins=20, alpha=0.6, color=colors[i % len(colors)], label=f'{label} ({len(bob_rewards)} events)', edgecolor='white', linewidth=0.5)
        ax_rew.set_xlabel('Reward Value')
        ax_rew.set_ylabel('Count')
        ax_rew.legend()
        ax_rew.grid(True)
        fig_rew.tight_layout()

    if save_path:
        fig_bob.savefig(f'{save_path}_bob.png', dpi=150, bbox_inches='tight')
        fig_alice.savefig(f'{save_path}_alice.png', dpi=150, bbox_inches='tight')
        if any_rewards:
            fig_rew.savefig(f'{save_path}_rewards.png', dpi=150, bbox_inches='tight')
        print(f"Saved plots to {save_path}_*.png")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize ASP training logs')
    parser.add_argument('--files', nargs='+', default=[
        'training_updates.log',
        '4_env.out',
        '20_env.out',
    ], help='Log files to parse')
    parser.add_argument('--labels', nargs='+', default=None, help='Labels for each file')
    parser.add_argument('--save', type=str, default=None, help='Save prefix (e.g. "plots/run")')
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(f).stem for f in args.files]

    runs = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"WARNING: {f} not found, skipping")
            continue
        alice, bob, bob_rewards = parse_log(f)
        print(f"[{p.name}] Alice updates: {len(alice['update'])}, Bob updates: {len(bob['update'])}, Bob reward events: {len(bob_rewards)}")
        runs.append((alice, bob, bob_rewards))

    if not runs:
        print("No valid log files found!")
        return

    # Adjust labels if some files were skipped
    valid_labels = []
    for f, l in zip(args.files, args.labels):
        if Path(f).exists():
            valid_labels.append(l)

    plot_runs(runs, valid_labels, save_path=args.save)


if __name__ == '__main__':
    main()
