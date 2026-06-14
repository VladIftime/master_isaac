"""Shared utilities for Isaac Lab benchmark scripts.

Provides GPU-timed SPS measurement, CSV export, and matplotlib plotting.
"""

import csv
import os
import time
from datetime import datetime

import torch


def get_output_dir(base_dir="logs/benchmarks"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, timestamp)
    os.makedirs(out, exist_ok=True)
    return out


def measure_sps(env, num_steps, warmup_steps=10, random_actions=True):
    """Measure Steps Per Second using CUDA events for accurate GPU timing.

    Returns dict with: sps, total_seconds, num_steps, num_envs, physics_steps_per_sec
    """
    device = env.device if hasattr(env, "device") else "cuda:0"
    num_envs = env.num_envs

    if hasattr(env, "single_action_space") and hasattr(env.single_action_space, "shape"):
        action_shape = (num_envs,) + env.single_action_space.shape
    elif hasattr(env, "cfg") and isinstance(getattr(env.cfg, "action_space", None), int):
        action_shape = (num_envs, env.cfg.action_space)
    elif hasattr(env, "num_actions"):
        action_shape = (num_envs, env.num_actions)
    else:
        action_space = env.action_space
        if hasattr(action_space, "shape"):
            shape = action_space.shape
            if len(shape) == 1:
                action_shape = (num_envs,) + shape
            else:
                action_shape = shape
        else:
            action_shape = (num_envs, 4)

    decimation = env.cfg.decimation if hasattr(env.cfg, "decimation") else 1

    for _ in range(warmup_steps):
        if random_actions:
            actions = torch.randn(action_shape, device=device)
        else:
            actions = torch.zeros(action_shape, device=device)
        env.step(actions)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_steps):
        if random_actions:
            actions = torch.randn(action_shape, device=device)
        else:
            actions = torch.zeros(action_shape, device=device)
        env.step(actions)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000.0

    env_steps = num_steps * num_envs
    physics_steps = num_steps * num_envs * decimation
    sps = env_steps / elapsed_s
    physics_sps = physics_steps / elapsed_s

    return {
        "sps": sps,
        "physics_sps": physics_sps,
        "total_seconds": elapsed_s,
        "num_steps": num_steps,
        "num_envs": num_envs,
        "decimation": decimation,
        "env_steps_total": env_steps,
        "physics_steps_total": physics_steps,
    }


def get_gpu_memory_mb():
    """Return current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def write_csv(filepath, rows, fieldnames):
    """Write list of dicts to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Saved: {filepath}")


def print_results_table(rows, title="Results"):
    """Pretty-print a table of results to terminal."""
    if not rows:
        return
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    keys = list(rows[0].keys())
    header = " | ".join(f"{k:>16s}" for k in keys)
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for row in rows:
        vals = []
        for k in keys:
            v = row[k]
            if isinstance(v, float):
                vals.append(f"{v:>16.2f}")
            else:
                vals.append(f"{v!s:>16s}")
        print(f"  {' | '.join(vals)}")
    print(f"{'='*70}\n")


def plot_bar_chart(data, x_labels, y_values_key, title, ylabel, output_path,
                   y2_values_key=None, y2_label=None):
    """Generate a bar chart and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))

    y_vals = [d[y_values_key] for d in data]
    bars = ax1.bar(x_labels, y_vals, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel(ylabel, color="steelblue")
    ax1.set_title(title)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    for bar, val in zip(bars, y_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    if y2_values_key and y2_label:
        ax2 = ax1.twinx()
        y2_vals = [d[y2_values_key] for d in data]
        ax2.plot(x_labels, y2_vals, "ro-", linewidth=2, markersize=8)
        ax2.set_ylabel(y2_label, color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_line_chart(data, x_key, y_key, title, xlabel, ylabel, output_path):
    """Generate a line chart and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_vals = [d[x_key] for d in data]
    y_vals = [d[y_key] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, "b-o", linewidth=2, markersize=8, label="Measured")

    if len(x_vals) > 1:
        scale = y_vals[0] / x_vals[0]
        ideal = [x * scale for x in x_vals]
        ax.plot(x_vals, ideal, "g--", linewidth=1.5, alpha=0.6, label="Ideal Linear Scaling")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for x, y in zip(x_vals, y_vals):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_grouped_bar(data, x_labels, keys, labels, title, ylabel, output_path):
    """Generate grouped bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(x_labels))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        vals = [d[key] for d in data]
        offset = (i - len(keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Num Environments")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {output_path}")
