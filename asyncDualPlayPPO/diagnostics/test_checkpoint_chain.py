"""
Test 3b — ABC Buffer Checkpoint Round-Trip.

Verifies that GPUDemonstrationBuffer survives a save/load cycle with
trajectory count, shapes, and key names intact.

This test does NOT require Isaac Lab or a GPU — it only needs the
asyncDualPlayPPO package to be importable and a checkpoint directory
containing an abc_buffer.pt file.

Usage:
    python -m asyncDualPlayPPO.diagnostics.test_checkpoint_chain \\
        --log_dir asyncDualPlayPPO/runs/diag_t3/bob

The test also accepts a raw buffer path:
    python -m asyncDualPlayPPO.diagnostics.test_checkpoint_chain \\
        --buf_path asyncDualPlayPPO/runs/diag_t3/abc_buffer.pt
"""

import argparse
import os
import sys
import torch


def check_checkpoint_chain(buf_path: str):
    print("\n" + "=" * 70)
    print("TEST 3b — ABC Buffer Checkpoint Round-Trip")
    print(f"Buffer: {buf_path}")
    print("=" * 70)

    from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer

    if not os.path.isfile(buf_path):
        print(f"  SKIP: buffer file not found at {buf_path}")
        print("  Run `python -m asyncDualPlayPPO.train --max_iterations 10 --save_interval 10` first.")
        sys.exit(0)

    # Load original
    buf_orig = GPUDemonstrationBuffer()
    buf_orig.load(buf_path)
    orig_size = buf_orig.size

    if orig_size == 0:
        print("  SKIP: buffer is empty — run more training iterations to populate it")
        sys.exit(0)

    print(f"  Original buffer size: {orig_size}")

    # Save to a temp file and reload
    tmp_path = buf_path + ".test_roundtrip.pt"
    try:
        buf_orig.save(tmp_path)

        buf_restored = GPUDemonstrationBuffer()
        buf_restored.load(tmp_path)

        failures = []

        # Size must match
        if buf_restored.size != orig_size:
            failures.append(
                f"FAIL: buffer size changed after round-trip ({orig_size} → {buf_restored.size})"
            )

        # Shape and keys must be intact for each trajectory
        for i, traj in enumerate(buf_restored._traj_store):
            for key in ("obs", "acts", "old_lp"):
                if key not in traj:
                    failures.append(f"FAIL: trajectory {i} missing key '{key}' after load")
            if "obs" in traj and "acts" in traj:
                if traj["obs"].ndim != 2:
                    failures.append(f"FAIL: traj {i} obs.ndim={traj['obs'].ndim} (expected 2)")
                if traj["acts"].ndim != 2:
                    failures.append(f"FAIL: traj {i} acts.ndim={traj['acts'].ndim} (expected 2)")
                if traj["acts"].shape[-1] != 6:
                    failures.append(
                        f"FAIL: traj {i} acts.shape[-1]={traj['acts'].shape[-1]} (expected 6 for 6D MC)"
                    )

        if failures:
            for f in failures:
                print(f"  {f}")
            print("TEST 3b FAIL")
            sys.exit(1)
        else:
            print(f"  Round-trip OK: {orig_size} trajectories preserved")
            print("TEST 3b PASS")
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", default=None,
        help="Training log dir — looks for abc_buffer.pt inside"
    )
    parser.add_argument(
        "--buf_path", default=None,
        help="Direct path to abc_buffer.pt"
    )
    args = parser.parse_args()

    if args.buf_path:
        buf_path = args.buf_path
    elif args.log_dir:
        buf_path = os.path.join(args.log_dir, "abc_buffer.pt")
    else:
        print("ERROR: provide --log_dir or --buf_path")
        sys.exit(1)

    check_checkpoint_chain(buf_path)
