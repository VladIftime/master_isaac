#!/usr/bin/env python3
"""Benchmark 1: Scalability and Throughput Testing.

Runs the ThrowingDirectEnv at increasing environment counts (128, 1024, 4096, 8192)
and records Steps Per Second (SPS) to demonstrate near-linear GPU scaling.

Uses subprocess isolation since Isaac Lab cannot reliably recreate envs in one process.

Usage:
    cd throwing_enviroment
    python scripts/benchmark_scalability.py --headless
    python scripts/benchmark_scalability.py --headless --env_counts 128,512,1024,2048,4096
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def run_single(n_envs, num_steps, warmup_steps):
    """Run benchmark for a single env count in a subprocess."""
    result_file = tempfile.mktemp(suffix=".json", prefix=f"bench_scale_{n_envs}_")
    child_script = os.path.join(_SCRIPT_DIR, "benchmark_scalability.py")

    cmd = [
        sys.executable, child_script,
        "--headless",
        "--_child",
        "--_n_envs", str(n_envs),
        "--_num_steps", str(num_steps),
        "--_warmup_steps", str(warmup_steps),
        "--_result_file", result_file,
    ]

    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=False, timeout=600)

    if proc.returncode != 0:
        print(f"  [ERROR] Subprocess for n_envs={n_envs} failed (exit code {proc.returncode})")
        return None

    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            data = json.load(f)
        os.unlink(result_file)
        return data
    return None


def child_main():
    """Subprocess entry: run a single benchmark config and save results."""
    sys.path.insert(0, _SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--_child", action="store_true")
    parser.add_argument("--_n_envs", type=int, required=True)
    parser.add_argument("--_num_steps", type=int, required=True)
    parser.add_argument("--_warmup_steps", type=int, required=True)
    parser.add_argument("--_result_file", type=str, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg
    from tasks.throwing_direct_env import ThrowingDirectEnv
    from benchmark_utils import measure_sps, get_gpu_memory_mb, reset_gpu_memory_stats

    reset_gpu_memory_stats()
    cfg = ThrowingDirectEnvCfg()
    cfg.scene.num_envs = args._n_envs
    cfg.sim.render_interval = cfg.decimation

    env = ThrowingDirectEnv(cfg=cfg)
    metrics = measure_sps(env, num_steps=args._num_steps, warmup_steps=args._warmup_steps)
    gpu_mem = get_gpu_memory_mb()

    result = {
        "num_envs": args._n_envs,
        "sps": metrics["sps"],
        "physics_sps": metrics["physics_sps"],
        "total_seconds": metrics["total_seconds"],
        "decimation": metrics["decimation"],
        "gpu_mem_allocated_mb": gpu_mem["allocated_mb"],
        "gpu_mem_max_mb": gpu_mem["max_allocated_mb"],
    }

    with open(args._result_file, "w") as f:
        json.dump(result, f)

    env.close()
    simulation_app.close()


def parent_main():
    """Parent process: orchestrate subprocess runs and aggregate results."""
    sys.path.insert(0, _SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)
    from benchmark_utils import (
        get_output_dir, write_csv, print_results_table, plot_line_chart,
    )

    parser = argparse.ArgumentParser(description="Benchmark: Scalability & Throughput")
    parser.add_argument("--env_counts", type=str, default="128,1024,4096,8192")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    env_counts = [int(x.strip()) for x in args.env_counts.split(",")]
    output_dir = args.output_dir or get_output_dir()

    print(f"\n{'='*60}")
    print(f"  Benchmark 1: Scalability & Throughput Testing")
    print(f"{'='*60}")
    print(f"  Env counts     : {env_counts}")
    print(f"  Steps/measure  : {args.num_steps}")
    print(f"  Warmup steps   : {args.warmup_steps}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*60}\n")

    results = []
    for n_envs in env_counts:
        print(f"\n[TEST] Running subprocess for num_envs={n_envs} ...")
        row = run_single(n_envs, args.num_steps, args.warmup_steps)
        if row:
            results.append(row)
            print(f"  SPS={row['sps']:.0f}  |  Physics SPS={row['physics_sps']:.0f}  |"
                  f"  Time={row['total_seconds']:.2f}s  |  GPU={row['gpu_mem_max_mb']:.0f} MB")
        else:
            print(f"  SKIPPED (subprocess failed)")

    if not results:
        print("[ERROR] No results collected.")
        return

    print_results_table(results, title="Scalability & Throughput Results")

    csv_path = os.path.join(output_dir, "scalability.csv")
    write_csv(csv_path, results, list(results[0].keys()))

    plot_line_chart(
        results,
        x_key="num_envs",
        y_key="physics_sps",
        title="Isaac Lab Scalability: Physics Steps/Second vs Num Environments",
        xlabel="Number of Parallel Environments",
        ylabel="Physics Steps Per Second",
        output_path=os.path.join(output_dir, "scalability.png"),
    )


if __name__ == "__main__":
    if "--_child" in sys.argv:
        child_main()
    else:
        parent_main()
