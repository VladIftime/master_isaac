#!/usr/bin/env python3
"""Benchmark 3: DirectRLEnv vs Manager-Based Overhead.

Compares raw env.step() throughput between ThrowingDirectEnv (DirectRLEnv)
and ThrowingEnv (ManagerBasedRLEnv) to demonstrate the performance advantage
of the direct workflow.

Uses subprocess isolation since Isaac Lab cannot reliably recreate envs in one process.

Usage:
    cd throwing_enviroment
    python scripts/benchmark_direct_vs_manager.py --headless
    python scripts/benchmark_direct_vs_manager.py --headless --num_envs 512
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def run_single(variant, num_envs, num_steps, warmup_steps):
    """Run a single variant (direct or manager) in a subprocess."""
    result_file = tempfile.mktemp(suffix=".json", prefix=f"bench_dvm_{variant}_")
    child_script = os.path.join(_SCRIPT_DIR, "benchmark_direct_vs_manager.py")

    cmd = [
        sys.executable, child_script,
        "--headless",
        "--_child",
        "--_variant", variant,
        "--_num_envs", str(num_envs),
        "--_num_steps", str(num_steps),
        "--_warmup_steps", str(warmup_steps),
        "--_result_file", result_file,
    ]

    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, timeout=600)

    if proc.returncode != 0:
        print(f"  [ERROR] Subprocess for variant={variant} failed (exit {proc.returncode})")
        return None

    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            data = json.load(f)
        os.unlink(result_file)
        return data
    return None


def child_main():
    """Subprocess entry: run a single variant and save results."""
    sys.path.insert(0, _SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--_child", action="store_true")
    parser.add_argument("--_variant", type=str, required=True)
    parser.add_argument("--_num_envs", type=int, required=True)
    parser.add_argument("--_num_steps", type=int, required=True)
    parser.add_argument("--_warmup_steps", type=int, required=True)
    parser.add_argument("--_result_file", type=str, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from benchmark_utils import measure_sps

    if args._variant == "direct":
        from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg
        from tasks.throwing_direct_env import ThrowingDirectEnv

        cfg = ThrowingDirectEnvCfg()
        cfg.scene.num_envs = args._num_envs
        cfg.sim.render_interval = cfg.decimation
        env = ThrowingDirectEnv(cfg=cfg)
    else:
        from tasks.throwing_env_cfg import ThrowingEnvCfg
        from tasks.throwing_env import ThrowingEnv

        cfg = ThrowingEnvCfg()
        cfg.scene.num_envs = args._num_envs
        cfg.sim.render_interval = cfg.decimation
        env = ThrowingEnv(cfg=cfg)

    metrics = measure_sps(env, num_steps=args._num_steps, warmup_steps=args._warmup_steps)

    result = {
        "variant": args._variant,
        "env_sps": metrics["sps"],
        "physics_sps": metrics["physics_sps"],
        "decimation": metrics["decimation"],
        "total_seconds": metrics["total_seconds"],
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
        get_output_dir, write_csv, print_results_table, plot_bar_chart,
    )

    parser = argparse.ArgumentParser(description="Benchmark: DirectRLEnv vs ManagerBased")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = args.output_dir or get_output_dir()

    print(f"\n{'='*60}")
    print(f"  Benchmark 3: DirectRLEnv vs Manager-Based Overhead")
    print(f"{'='*60}")
    print(f"  Num envs       : {args.num_envs}")
    print(f"  Steps/measure  : {args.num_steps}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*60}\n")

    print("[TEST] DirectRLEnv (ThrowingDirectEnv) ...")
    direct = run_single("direct", args.num_envs, args.num_steps, args.warmup_steps)
    if direct:
        print(f"  Direct: env_sps={direct['env_sps']:.0f}  |  "
              f"physics_sps={direct['physics_sps']:.0f}  |  dec={direct['decimation']}")

    print("\n[TEST] ManagerBasedRLEnv (ThrowingEnv) ...")
    manager = run_single("manager", args.num_envs, args.num_steps, args.warmup_steps)
    if manager:
        print(f"  Manager: env_sps={manager['env_sps']:.0f}  |  "
              f"physics_sps={manager['physics_sps']:.0f}  |  dec={manager['decimation']}")

    if not direct or not manager:
        print("[ERROR] One or both variants failed.")
        return

    speedup = direct["physics_sps"] / manager["physics_sps"]

    results = [
        {"variant": "DirectRLEnv", **direct},
        {"variant": "ManagerBasedRLEnv", **manager},
    ]

    print_results_table(results, title="DirectRLEnv vs Manager-Based Results")
    print(f"  DirectRLEnv physics throughput speedup: {speedup:.2f}x")
    print(f"  (Comparing physics steps/second — accounts for different decimation values)\n")

    csv_path = os.path.join(output_dir, "direct_vs_manager.csv")
    write_csv(csv_path, results, list(results[0].keys()))

    plot_bar_chart(
        results,
        x_labels=[f"DirectRLEnv\n(dec={direct['decimation']})",
                  f"ManagerBasedRLEnv\n(dec={manager['decimation']}, IK)"],
        y_values_key="physics_sps",
        title=f"DirectRLEnv vs Manager-Based ({args.num_envs} envs)\n"
              f"Speedup: {speedup:.2f}x (physics steps/sec)",
        ylabel="Physics Steps Per Second",
        output_path=os.path.join(output_dir, "direct_vs_manager.png"),
    )


if __name__ == "__main__":
    if "--_child" in sys.argv:
        child_main()
    else:
        parent_main()
