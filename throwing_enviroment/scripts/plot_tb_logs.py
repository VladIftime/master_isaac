#!/usr/bin/env python3
"""Plot TensorBoard logs from SAC throw-primitive training.

Reads tfevents files and generates a multi-panel diagnostic figure + CSV export.
Includes reward-to-distance inversion using the known reward function.

Usage:
    python scripts/plot_tb_logs.py <log_dir_or_tfevents_file>
    python scripts/plot_tb_logs.py logs/skrl/throwing_primitive/2026-06-10_13-59-07_sac_torch/
    python scripts/plot_tb_logs.py  # uses most recent run under logs/skrl/throwing_primitive/
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "skrl", "throwing_primitive")


def reward_fn(d, alpha=0.9, sigma_narrow=0.1, sigma_wide=0.5):
    return (
        alpha * np.exp(-(d ** 2) / sigma_narrow)
        + (1.0 - alpha) * np.exp(-(d ** 2) / sigma_wide)
        + 0.5 * max(0.0, 1.0 - d)
    )


def reward_fn_old(d, alpha=0.9, sigma_narrow=0.01, sigma_wide=0.05):
    return (
        alpha * np.exp(-(d ** 2) / sigma_narrow)
        + (1.0 - alpha) * np.exp(-(d ** 2) / sigma_wide)
    )


def invert_reward_to_distance(reward, use_old=True):
    fn = reward_fn_old if use_old else reward_fn
    if reward >= fn(0.0):
        return 0.0
    if reward <= fn(5.0):
        return 5.0
    try:
        return brentq(lambda d: fn(d) - reward, 0.0, 5.0)
    except ValueError:
        return 5.0


def find_latest_run(log_root):
    if not os.path.isdir(log_root):
        return None
    runs = sorted(
        [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))],
        reverse=True,
    )
    return os.path.join(log_root, runs[0]) if runs else None


def find_tfevents(path):
    if os.path.isfile(path) and "tfevents" in path:
        return path
    if os.path.isdir(path):
        matches = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        if matches:
            return sorted(matches)[-1]
        matches = glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True)
        if matches:
            return sorted(matches)[-1]
    return None


def load_scalars(tfevents_path):
    ea = EventAccumulator(tfevents_path)
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": np.array(steps), "values": np.array(values)}
    return data


def export_csv(data, output_path, estimated_distances=None):
    all_tags = sorted(data.keys())
    all_steps = set()
    for tag_data in data.values():
        all_steps.update(tag_data["steps"].tolist())
    all_steps = sorted(all_steps)

    step_to_idx = {s: i for i, s in enumerate(all_steps)}
    rows = {s: {"step": int(s)} for s in all_steps}
    for tag in all_tags:
        col = tag.replace("/", " - ").strip()
        for step, val in zip(data[tag]["steps"], data[tag]["values"]):
            rows[step][col] = val

    if estimated_distances is not None:
        for step, dist in estimated_distances:
            if step in rows:
                rows[step]["Estimated Distance (m)"] = dist

    fieldnames = ["step"] + [t.replace("/", " - ").strip() for t in all_tags]
    if estimated_distances is not None:
        fieldnames.append("Estimated Distance (m)")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in all_steps:
            writer.writerow(rows[s])
    print(f"CSV exported to: {output_path}")


def plot_diagnostics(data, output_path, run_name=""):
    has_logged_dist = "Throw / Mean Distance" in data

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"SAC Training Diagnostics\n{run_name}", fontsize=13, y=0.98)

    ax = axes[0, 0]
    for suffix, color, label in [("mean", "blue", "Mean"), ("max", "green", "Max"), ("min", "red", "Min")]:
        tag = f"Reward / Total reward ({suffix})"
        if tag in data:
            ax.plot(data[tag]["steps"], data[tag]["values"], color=color, label=label, linewidth=1.2)
    ax.set_title("Reward")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if has_logged_dist:
        for tag, color, label in [
            ("Throw / Mean Distance", "darkorange", "Mean"),
            ("Throw / Min Distance", "green", "Min"),
            ("Throw / Max Distance", "red", "Max"),
        ]:
            if tag in data:
                ax.plot(data[tag]["steps"], data[tag]["values"], color=color, label=label, linewidth=1.2)
        ax.set_title("Throw Distance to Target")
    else:
        tag = "Reward / Total reward (mean)"
        if tag in data:
            steps = data[tag]["steps"]
            rewards = data[tag]["values"]
            distances = [invert_reward_to_distance(r, use_old=True) for r in rewards]
            ax.plot(steps, distances, color="darkorange", linewidth=1.5, label="Estimated Distance")
            ax.set_title("Estimated Throw Distance (inverted from reward)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    tag = "Coefficient / Entropy coefficient"
    if tag in data:
        ax.plot(data[tag]["steps"], data[tag]["values"], color="purple", linewidth=1.5)
        ax.set_yscale("log")
    ax.set_title("Entropy Coefficient (log scale)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("alpha")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    tag = "Loss / Policy loss"
    if tag in data:
        ax.plot(data[tag]["steps"], data[tag]["values"], color="crimson", linewidth=1.2, label="Policy Loss")
    ax.set_title("Policy Loss")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    ax = axes[2, 0]
    tag = "Loss / Critic loss"
    if tag in data:
        ax.plot(data[tag]["steps"], data[tag]["values"], color="teal", linewidth=1.2)
    ax.set_title("Critic Loss")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for tag, color, label in [
        ("Q-network / Q1 (mean)", "steelblue", "Q1 mean"),
        ("Q-network / Q2 (mean)", "coral", "Q2 mean"),
    ]:
        if tag in data:
            ax.plot(data[tag]["steps"], data[tag]["values"], color=color, label=label, linewidth=1.2)
    ax.set_title("Q-values (mean)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Q-value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot SAC training logs from TensorBoard events.")
    parser.add_argument("path", nargs="?", default=None, help="Path to log directory or tfevents file.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: <log_dir>/diagnostics.png).")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path (default: <log_dir>/scalars.csv).")
    args = parser.parse_args()

    if args.path is None:
        log_dir = find_latest_run(DEFAULT_LOG_ROOT)
        if log_dir is None:
            print(f"No runs found under {DEFAULT_LOG_ROOT}. Provide a path explicitly.")
            sys.exit(1)
        print(f"Using latest run: {log_dir}")
    else:
        log_dir = args.path

    tfevents_path = find_tfevents(log_dir)
    if tfevents_path is None:
        print(f"No tfevents file found at: {log_dir}")
        sys.exit(1)

    print(f"Reading: {tfevents_path}")
    data = load_scalars(tfevents_path)
    print(f"Found {len(data)} scalar tags, tags: {list(data.keys())}")

    if os.path.isfile(log_dir):
        base_dir = os.path.dirname(log_dir)
    else:
        base_dir = log_dir

    run_name = os.path.basename(base_dir)

    estimated_distances = None
    tag = "Reward / Total reward (mean)"
    if tag in data and "Info / mean_distance" not in data:
        estimated_distances = [
            (int(s), invert_reward_to_distance(r, use_old=True))
            for s, r in zip(data[tag]["steps"], data[tag]["values"])
        ]
        print(f"\nEstimated distances (inverted from reward):")
        for step, dist in estimated_distances[:5]:
            print(f"  step={step:>6d}  dist={dist:.3f}m")
        if len(estimated_distances) > 5:
            print(f"  ... ({len(estimated_distances)} total)")

    output_png = args.output or os.path.join(base_dir, "diagnostics.png")
    plot_diagnostics(data, output_png, run_name=run_name)

    output_csv = args.csv or os.path.join(base_dir, "scalars.csv")
    export_csv(data, output_csv, estimated_distances=estimated_distances)

    print("\nKey observations:")
    if "Coefficient / Entropy coefficient" in data:
        ent = data["Coefficient / Entropy coefficient"]["values"]
        print(f"  Entropy coeff: {ent[0]:.4f} -> {ent[-1]:.2e} ({'COLLAPSED' if ent[-1] < 1e-4 else 'OK'})")
    if "Reward / Total reward (mean)" in data:
        rew = data["Reward / Total reward (mean)"]["values"]
        print(f"  Mean reward:   {rew[0]:.6f} -> {rew[-1]:.6f} ({'FLAT' if abs(rew[-1] - rew[0]) < 0.01 else 'improving'})")
    if "Loss / Policy loss" in data:
        pl = data["Loss / Policy loss"]["values"]
        print(f"  Policy loss:   {pl[0]:.2e} -> {pl[-1]:.2e} ({'DIVERGING' if abs(pl[-1]) > 1e6 else 'OK'})")


if __name__ == "__main__":
    main()
