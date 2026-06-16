#!/usr/bin/env python3
"""Evaluate trained SAC policy on a dense grid of targets → heatmap of landing distance.

Extends validate_throw.py: instead of 10 predefined test targets, this evaluates
on a grid_size × grid_size grid over the full training XY range and plots a
heatmap.  Regions of low distance indicate the policy generalises; high-distance
regions reveal OOD failure modes.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/validate_heatmap.py \
        --checkpoint logs/skrl/throwing_primitive/2026-06-14_.../checkpoints/agent_6000.pt \
        --headless
    python scripts/validate_heatmap.py \
        --checkpoint ... --grid_size 10 --headless   # quick test (100 points)
    python scripts/validate_heatmap.py \
        --checkpoint ... --x_range 0.1,0.4 --y_range 1.1,1.5 --headless
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate SAC policy on target grid → heatmap")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to agent checkpoint (.pt)")
parser.add_argument("--grid_size", type=int, default=20,
                    help="Grid resolution (default 20 → 400 points)")
parser.add_argument("--x_range", type=str, default="0.0,0.5",
                    help="X range as 'min,max' (default: 0.0,0.5)")
parser.add_argument("--y_range", type=str, default="1.0,1.6",
                    help="Y range as 'min,max' (default: 1.0,1.6)")
parser.add_argument("--success_threshold", type=float, default=0.15,
                    help="Distance threshold for success in metres (default: 0.15)")
parser.add_argument("--output", type=str, default=None,
                    help="Output PNG path (default: logs/heatmap_generalisation.png)")
parser.add_argument("--csv", type=str, default=None,
                    help="Output CSV path (default: logs/heatmap_generalisation.csv)")
parser.add_argument("--no_plot", action="store_true",
                    help="Skip plot generation")
parser.add_argument("--no_csv", action="store_true",
                    help="Skip CSV export")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import yaml

from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg, TABLE_Z
from tasks.throwing_direct_env import ThrowingDirectEnv
from tasks.throw_validation_configs import THROW_TESTS

PLAYING_SIDE = "right"


def load_sac_agent(env_wrapped, checkpoint_path, device):
    """Load SAC agent using skrl Runner."""
    from skrl.utils.runner.torch import Runner

    agent_cfg_path = os.path.join(
        _PROJECT_ROOT, "source", "Throwing", "Throwing", "tasks",
        "throwing", "agents", "skrl_sac_cfg.yaml",
    )
    with open(agent_cfg_path, "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["seed"] = 42
    agent_cfg["trainer"]["timesteps"] = 1
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    runner = Runner(env_wrapped, agent_cfg)
    runner.agent.load(checkpoint_path)
    runner.agent.policy.eval()
    print(f"[INFO] Loaded SAC agent from: {checkpoint_path}")
    return runner.agent


def parse_range(range_str):
    parts = [float(x.strip()) for x in range_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{range_str}': expected 'min,max'")
    return parts[0], parts[1]


def make_grid(x_range, y_range, grid_size):
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xs, ys)
    points = list(zip(X.flat, Y.flat))
    return xs, ys, X, Y, points


def plot_heatmap(distance_grid, xs, ys, X, Y, results, validation_targets,
                 success_threshold, x_range, y_range, save_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    valid_mask = np.isfinite(distance_grid)
    vmin = np.min(distance_grid[valid_mask]) if valid_mask.any() else 0.0
    vmax = np.max(distance_grid[valid_mask]) if valid_mask.any() else 2.0
    if vmax < success_threshold * 3:
        vmax = success_threshold * 3

    mesh = ax1.pcolormesh(X, Y, distance_grid, cmap="RdYlGn_r", vmin=vmin,
                          vmax=vmax, shading="auto", edgecolors="face")
    cbar = fig.colorbar(mesh, ax=ax1, shrink=0.85)
    cbar.set_label("Landing Distance (m)", fontsize=10)

    contour_levels = [success_threshold]
    if vmax > success_threshold * 2:
        contour_levels.append(success_threshold * 2)
    ct = ax1.contour(X, Y, distance_grid, levels=contour_levels, colors="black",
                     linewidths=1.2, linestyles=["-", "--"])
    ax1.clabel(ct, fmt="%.2f m", fontsize=8)

    vt_x = [t.target_x for t in validation_targets]
    vt_y = [t.target_y for t in validation_targets]
    ax1.scatter(vt_x, vt_y, marker="D", color="black", s=50, zorder=5,
                edgecolors="white", linewidths=0.8, label="Val. targets")
    for t in validation_targets:
        ax1.annotate(str(t.test_id), (t.target_x, t.target_y),
                     textcoords="offset points", xytext=(4, 4), fontsize=7,
                     color="black")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"Landing Distance Heatmap ({grid_size}×{grid_size} grid)")
    ax1.set_xlim(x_range[0], x_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(False)

    distances = [r["distance"] for r in results if np.isfinite(r["distance"])]
    n_success = sum(1 for d in distances if d < success_threshold)
    ax2.hist(distances, bins=30, color="steelblue", edgecolor="black",
             alpha=0.8, linewidth=0.5)
    ax2.axvline(x=success_threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold ({success_threshold}m)")
    ax2.set_xlabel("Landing Distance (m)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distance Distribution\n{n_success}/{len(distances)} "
                  f"success ({n_success/len(distances)*100:.1f}%) "
                  f"| median={np.median(distances):.3f}m")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    total = len(results)
    fig.suptitle(
        f"Generalisation Heatmap — {grid_size}×{grid_size} = {total} targets "
        f"| X∈{x_range} Y∈{y_range} | threshold={success_threshold}m",
        fontsize=12, y=1.01,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Heatmap saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_csv(results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_x", "target_y",
                                                "distance", "landing_x",
                                                "landing_y", "success"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[INFO] CSV exported to: {csv_path}")


def main():
    headless = args_cli.headless

    x_range = parse_range(args_cli.x_range)
    y_range = parse_range(args_cli.y_range)
    grid_size = args_cli.grid_size
    total = grid_size * grid_size

    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_png = args_cli.output or os.path.join(_PROJECT_ROOT, "logs",
                                                  f"heatmap_generalisation_{ts}.png")
    output_csv = args_cli.csv or os.path.join(_PROJECT_ROOT, "logs",
                                               f"heatmap_generalisation_{ts}.csv")

    print(f"\n{'='*60}")
    print(f"  GENERALISATION HEATMAP")
    print(f"  Checkpoint  : {args_cli.checkpoint}")
    print(f"  Grid        : {grid_size}×{grid_size} = {total} points")
    print(f"  X range     : {x_range}")
    print(f"  Y range     : {y_range}")
    print(f"  Threshold   : {args_cli.success_threshold}m")
    print(f"  Output PNG  : {output_png}")
    print(f"  Output CSV  : {output_csv}")
    print(f"{'='*60}\n")

    cfg = ThrowingDirectEnvCfg()
    cfg.scene.num_envs = 1
    cfg.playing_arm_side = PLAYING_SIDE

    env = ThrowingDirectEnv(cfg=cfg)
    device = env.device
    env.reset()

    import gymnasium as gym
    from skrl.envs.wrappers.torch import wrap_env

    model_obs_dim = cfg.observation_space
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy_sd = ckpt["policy"]
        if isinstance(policy_sd, dict) and "net_container.0.weight" in policy_sd:
            model_obs_dim = policy_sd["net_container.0.weight"].shape[1]
    print(f"[INFO] Model obs dim: {model_obs_dim}, Env obs dim: {cfg.observation_space}")

    _OBS_8D_FROM_10D = [0, 1, 2, 4, 5, 7, 8, 9]

    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                     shape=(model_obs_dim,), dtype=np.float32)
    dummy_act_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0, 0.05, 0.1], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )

    class _DummyEnv:
        def __init__(self):
            self.observation_space = dummy_obs_space
            self.action_space = dummy_act_space
            self.state_space = dummy_obs_space
            self.num_envs = 1
            self.num_agents = 1
            self.device = device
        def reset(self): return torch.zeros(1, model_obs_dim, device=device), {}
        def step(self, a): return torch.zeros(1, model_obs_dim, device=device), torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
        def close(self): pass
        def render(self): pass

    dummy_wrapped = wrap_env(_DummyEnv(), wrapper="gymnasium")
    agent = load_sac_agent(dummy_wrapped, args_cli.checkpoint, device)

    xs, ys, X, Y, points = make_grid(x_range, y_range, grid_size)
    results = []

    target_obj = env.scene["target"]
    origin = env.scene.env_origins[0]
    tgt_z = TABLE_Z + 0.001
    env_id = torch.tensor([0], device=device)

    start_time = time.time()

    for i, (tx, ty) in enumerate(points):
        pos = torch.tensor([[tx + origin[0].item(),
                             ty + origin[1].item(),
                             tgt_z + origin[2].item()]], device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        target_obj.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1), env_ids=env_id,
        )

        obs = env._get_observations()["policy"]
        if model_obs_dim < obs.shape[-1]:
            obs = obs[:, _OBS_8D_FROM_10D]

        with torch.no_grad():
            act_out = agent.policy.act({"states": obs}, role="policy")
            action = act_out[0]

        action_np = action.cpu().numpy().flatten()
        action_clamped = torch.tensor(action_np, device=device).unsqueeze(0)
        action_clamped[:, 0].clamp_(-1.0, 1.0)
        action_clamped[:, 1].clamp_(-1.0, 1.0)
        action_clamped[:, 2].clamp_(0.05, 1.0)
        action_clamped[:, 3].clamp_(0.1, 1.0)

        env.step(action_clamped)

        distance = env._last_distances[0].item()
        landing = env._last_milk_pos[0].cpu().tolist()
        success = 1 if distance < args_cli.success_threshold else 0

        results.append({
            "target_x": tx,
            "target_y": ty,
            "distance": distance,
            "landing_x": landing[0],
            "landing_y": landing[1],
            "success": success,
        })

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            dists_done = [r["distance"] for r in results if np.isfinite(r["distance"])]
            mean_d = np.mean(dists_done) if dists_done else float("nan")
            n_ok = sum(1 for r in results if r["success"])
            print(f"  [{i+1:4d}/{total}] mean={mean_d:.3f}m  "
                  f"success={n_ok}/{i+1}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - start_time
    dists = [r["distance"] for r in results if np.isfinite(r["distance"])]
    n_success = sum(1 for r in results if r["success"])
    mean_d = np.mean(dists) if dists else float("nan")
    median_d = np.median(dists) if dists else float("nan")
    min_d = np.min(dists) if dists else float("nan")
    max_d = np.max(dists) if dists else float("nan")

    print(f"\n{'='*60}")
    print(f"  HEATMAP RESULTS")
    print(f"{'='*60}")
    print(f"  Points         : {total}")
    print(f"  Success        : {n_success}/{total} ({n_success/total*100:.1f}%)")
    print(f"  Mean distance  : {mean_d:.3f}m")
    print(f"  Median distance: {median_d:.3f}m")
    print(f"  Min distance   : {min_d:.3f}m")
    print(f"  Max distance   : {max_d:.3f}m")
    print(f"  Wall time      : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Avg per point  : {elapsed/total:.1f}s")
    print(f"{'='*60}")

    if not args_cli.no_csv:
        save_csv(results, output_csv)

    if not args_cli.no_plot:
        distance_grid = np.full_like(X, np.nan, dtype=np.float64)
        for i, (tx, ty) in enumerate(points):
            col = i % grid_size
            row = i // grid_size
            distance_grid[row, col] = results[i]["distance"]

        plot_heatmap(
            distance_grid=distance_grid,
            xs=xs, ys=ys, X=X, Y=Y,
            results=results,
            validation_targets=THROW_TESTS,
            success_threshold=args_cli.success_threshold,
            x_range=x_range,
            y_range=y_range,
            save_path=output_png,
            show=not headless,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
