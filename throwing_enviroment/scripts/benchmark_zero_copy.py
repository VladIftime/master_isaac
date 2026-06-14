#!/usr/bin/env python3
"""Benchmark 2: Zero-Copy Architecture Ablation.

Compares standard GPU-resident observation pipeline vs forcing observations
through PCIe bus (CPU roundtrip) to demonstrate Isaac Lab's zero-copy advantage.

Uses subprocess isolation since Isaac Lab cannot reliably recreate envs in one process.

Usage:
    cd throwing_enviroment
    python scripts/benchmark_zero_copy.py --headless
    python scripts/benchmark_zero_copy.py --headless --num_envs 2048 --num_steps 200
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
    """Run a single variant (baseline or cpu_roundtrip) in a subprocess."""
    result_file = tempfile.mktemp(suffix=".json", prefix=f"bench_zc_{variant}_")
    child_script = os.path.join(_SCRIPT_DIR, "benchmark_zero_copy.py")

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

    import torch
    from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg
    from tasks.throwing_direct_env import ThrowingDirectEnv
    from benchmark_utils import measure_sps

    class ThrowingDirectEnvCPURoundtrip(ThrowingDirectEnv):
        def _get_observations(self):
            obs = super()._get_observations()
            obs["policy"] = obs["policy"].cpu().cuda()
            if "critic" in obs:
                obs["critic"] = obs["critic"].cpu().cuda()
            return obs

    cfg = ThrowingDirectEnvCfg()
    cfg.scene.num_envs = args._num_envs
    cfg.sim.render_interval = cfg.decimation

    if args._variant == "cpu_roundtrip":
        env = ThrowingDirectEnvCPURoundtrip(cfg=cfg)
    else:
        env = ThrowingDirectEnv(cfg=cfg)

    metrics = measure_sps(env, num_steps=args._num_steps, warmup_steps=args._warmup_steps)

    result = {
        "variant": args._variant,
        "sps": metrics["sps"],
        "physics_sps": metrics["physics_sps"],
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

    parser = argparse.ArgumentParser(description="Benchmark: Zero-Copy Ablation")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = args.output_dir or get_output_dir()

    print(f"\n{'='*60}")
    print(f"  Benchmark 2: Zero-Copy Architecture Ablation")
    print(f"{'='*60}")
    print(f"  Num envs       : {args.num_envs}")
    print(f"  Steps/measure  : {args.num_steps}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*60}\n")

    print("[TEST] Baseline: GPU-native zero-copy observations ...")
    baseline = run_single("baseline", args.num_envs, args.num_steps, args.warmup_steps)
    if baseline:
        print(f"  Baseline SPS: {baseline['sps']:.0f}  |  Physics SPS: {baseline['physics_sps']:.0f}")

    print("\n[TEST] CPU Roundtrip: forced .cpu().cuda() on observations ...")
    cpu_rt = run_single("cpu_roundtrip", args.num_envs, args.num_steps, args.warmup_steps)
    if cpu_rt:
        print(f"  CPU Roundtrip SPS: {cpu_rt['sps']:.0f}  |  Physics SPS: {cpu_rt['physics_sps']:.0f}")

    if not baseline or not cpu_rt:
        print("[ERROR] One or both variants failed.")
        return

    degradation_pct = (1.0 - cpu_rt["sps"] / baseline["sps"]) * 100.0
    speedup = baseline["sps"] / cpu_rt["sps"]

    results = [
        {"variant": "GPU Zero-Copy", "sps": baseline["sps"],
         "physics_sps": baseline["physics_sps"],
         "degradation_pct": 0.0, "speedup": 1.0},
        {"variant": "CPU Roundtrip", "sps": cpu_rt["sps"],
         "physics_sps": cpu_rt["physics_sps"],
         "degradation_pct": degradation_pct, "speedup": 1.0 / speedup},
    ]

    print_results_table(results, title="Zero-Copy Ablation Results")
    print(f"  Performance degradation: {degradation_pct:.1f}%")
    print(f"  Zero-copy speedup factor: {speedup:.2f}x\n")

    csv_path = os.path.join(output_dir, "zero_copy.csv")
    write_csv(csv_path, results, list(results[0].keys()))

    plot_bar_chart(
        results,
        x_labels=["GPU Zero-Copy\n(Baseline)", "CPU Roundtrip\n(.cpu().cuda())"],
        y_values_key="physics_sps",
        title=f"Zero-Copy Architecture Ablation ({args.num_envs} envs)\n"
              f"Degradation: {degradation_pct:.1f}% | Speedup: {speedup:.2f}x",
        ylabel="Physics Steps Per Second",
        output_path=os.path.join(output_dir, "zero_copy.png"),
    )


if __name__ == "__main__":
    if "--_child" in sys.argv:
        child_main()
    else:
        parent_main()
