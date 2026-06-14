#!/usr/bin/env python3
"""Benchmark 5: Tiled Rendering Performance.

Compares env.step() throughput with and without TiledCamera at various
environment counts. Demonstrates Isaac Lab's batched RTX rendering capability.

Uses subprocess isolation since Isaac Lab cannot reliably recreate envs in one process.

Usage:
    cd throwing_enviroment
    python scripts/benchmark_tiled_camera.py --headless
    python scripts/benchmark_tiled_camera.py --headless --env_counts 128,512,1024
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def run_single(variant, n_envs, num_steps, warmup_steps):
    """Run a single variant (no_camera or tiled_camera) in a subprocess."""
    result_file = tempfile.mktemp(suffix=".json", prefix=f"bench_cam_{variant}_{n_envs}_")
    child_script = os.path.join(_SCRIPT_DIR, "benchmark_tiled_camera.py")

    cmd = [
        sys.executable, child_script,
        "--headless",
        "--_child",
        "--_variant", variant,
        "--_n_envs", str(n_envs),
        "--_num_steps", str(num_steps),
        "--_warmup_steps", str(warmup_steps),
        "--_result_file", result_file,
    ]
    if variant == "tiled_camera":
        cmd.append("--enable_cameras")

    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, timeout=600)

    if proc.returncode != 0:
        print(f"  [ERROR] Subprocess {variant} n={n_envs} failed (exit {proc.returncode})")
        return None

    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            data = json.load(f)
        os.unlink(result_file)
        return data
    return None


def child_main():
    """Subprocess entry: run a single config and save results."""
    sys.path.insert(0, _SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--_child", action="store_true")
    parser.add_argument("--_variant", type=str, required=True)
    parser.add_argument("--_n_envs", type=int, required=True)
    parser.add_argument("--_num_steps", type=int, required=True)
    parser.add_argument("--_warmup_steps", type=int, required=True)
    parser.add_argument("--_result_file", type=str, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args._variant == "tiled_camera":
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from benchmark_utils import measure_sps, get_gpu_memory_mb, reset_gpu_memory_stats

    reset_gpu_memory_stats()

    if args._variant == "tiled_camera":
        from tasks.throwing_direct_camera_env_cfg import ThrowingDirectCameraEnvCfg
        from tasks.throwing_direct_camera_env import ThrowingDirectCameraEnv

        cfg = ThrowingDirectCameraEnvCfg()
        cfg.scene.num_envs = args._n_envs
        cfg.sim.render_interval = 2
        env = ThrowingDirectCameraEnv(cfg=cfg)
    else:
        from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg
        from tasks.throwing_direct_env import ThrowingDirectEnv

        cfg = ThrowingDirectEnvCfg()
        cfg.scene.num_envs = args._n_envs
        cfg.sim.render_interval = cfg.decimation
        env = ThrowingDirectEnv(cfg=cfg)

    metrics = measure_sps(env, num_steps=args._num_steps, warmup_steps=args._warmup_steps)
    gpu_mem = get_gpu_memory_mb()

    result = {
        "variant": args._variant,
        "num_envs": args._n_envs,
        "physics_sps": metrics["physics_sps"],
        "sps": metrics["sps"],
        "gpu_max_mb": gpu_mem["max_allocated_mb"],
    }

    with open(args._result_file, "w") as f:
        json.dump(result, f)

    env.close()
    simulation_app.close()


def parent_main():
    """Parent process: orchestrate and aggregate."""
    sys.path.insert(0, _SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)
    from benchmark_utils import (
        get_output_dir, write_csv, print_results_table, plot_grouped_bar,
    )

    parser = argparse.ArgumentParser(description="Benchmark: Tiled Rendering Performance")
    parser.add_argument("--env_counts", type=str, default="128,512,1024,2048")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    env_counts = [int(x.strip()) for x in args.env_counts.split(",")]
    output_dir = args.output_dir or get_output_dir()

    print(f"\n{'='*60}")
    print(f"  Benchmark 5: Tiled Rendering Performance")
    print(f"{'='*60}")
    print(f"  Env counts     : {env_counts}")
    print(f"  Steps/measure  : {args.num_steps}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*60}\n")

    results = []

    for n_envs in env_counts:
        print(f"\n--- num_envs={n_envs} ---")

        print(f"  [NO CAMERA] ...")
        nocam = run_single("no_camera", n_envs, args.num_steps, args.warmup_steps)
        if nocam:
            print(f"    SPS={nocam['physics_sps']:.0f}  |  GPU={nocam['gpu_max_mb']:.0f} MB")

        print(f"  [TILED CAMERA 128x128] ...")
        cam = run_single("tiled_camera", n_envs, args.num_steps, args.warmup_steps)
        if cam:
            print(f"    SPS={cam['physics_sps']:.0f}  |  GPU={cam['gpu_max_mb']:.0f} MB")

        if nocam and cam:
            degradation = (1.0 - cam["physics_sps"] / nocam["physics_sps"]) * 100.0
            results.append({
                "num_envs": n_envs,
                "sps_no_camera": nocam["physics_sps"],
                "sps_tiled_camera": cam["physics_sps"],
                "degradation_pct": degradation,
                "gpu_mb_no_camera": nocam["gpu_max_mb"],
                "gpu_mb_tiled_camera": cam["gpu_max_mb"],
                "gpu_overhead_mb": cam["gpu_max_mb"] - nocam["gpu_max_mb"],
            })

    if not results:
        print("[ERROR] No results collected.")
        return

    print_results_table(results, title="Tiled Rendering Performance Results")

    csv_path = os.path.join(output_dir, "tiled_camera.csv")
    write_csv(csv_path, results, list(results[0].keys()))

    plot_grouped_bar(
        results,
        x_labels=[str(r["num_envs"]) for r in results],
        keys=["sps_no_camera", "sps_tiled_camera"],
        labels=["Headless (No Camera)", "TiledCamera (128x128 RGB)"],
        title="Tiled Rendering: Physics SPS with vs without Camera",
        ylabel="Physics Steps Per Second",
        output_path=os.path.join(output_dir, "tiled_camera.png"),
    )


if __name__ == "__main__":
    if "--_child" in sys.argv:
        child_main()
    else:
        parent_main()
