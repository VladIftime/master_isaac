"""
Test 2 — Alice Exploration Sandbox (offline analysis).

Run the full training with --alice_sandbox to generate a tensorboard log, then
call check_sandbox_run() to assert curriculum health from the event file.

Usage:
    # 1. Generate the run (from master_isaac/ root):
    python -m asyncDualPlayPPO.train --alice_sandbox --num_envs 32 \\
        --max_iterations 200 --headless --exp_name diag_sandbox

    # 2. Analyse the run:
    python -m asyncDualPlayPPO.diagnostics.test_alice_sandbox \\
        --log_dir asyncDualPlayPPO/runs/diag_sandbox/summary

What is checked:
    - Metrics/Alice/ValidGoals trends upward (last-20 avg > first-20 avg)
    - Metrics/Alice/GoalValidityRate never NaN
    - Alice/EntropyCoef is constant 0.01 (Fix 2)
    - No NaN in any logged scalar
"""

import argparse
import sys


def check_sandbox_run(log_dir: str):
    print("\n" + "=" * 70)
    print("TEST 2 — Alice Exploration Sandbox (offline check)")
    print(f"Log dir: {log_dir}")
    print("=" * 70)

    try:
        from tbparse import SummaryReader
        reader = SummaryReader(log_dir)
        df = reader.scalars
    except ImportError:
        # Fallback: use tensorboard's own event accumulator
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()
        import pandas as pd
        rows = []
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                rows.append({"tag": tag, "step": e.step, "value": e.value})
        import pandas as pd
        df = pd.DataFrame(rows)

    failures = []

    # 1. Valid goals must trend upward
    vg = df[df["tag"] == "Metrics/Alice/ValidGoals"]["value"].values
    if len(vg) >= 40:
        early_avg = vg[:20].mean()
        late_avg = vg[-20:].mean()
        print(f"  ValidGoals: early_avg={early_avg:.2f}  late_avg={late_avg:.2f}")
        if late_avg <= early_avg:
            failures.append(
                f"FAIL: Valid Goals not climbing (early={early_avg:.2f} late={late_avg:.2f}). "
                "Alice is not learning to produce valid goals."
            )
    else:
        print(f"  ValidGoals: only {len(vg)} data points — need ≥40 for trend check")

    # 2. GoalValidityRate should be non-NaN throughout
    gvr = df[df["tag"] == "Metrics/Alice/GoalValidityRate"]["value"]
    if gvr.isna().any():
        failures.append("FAIL: GoalValidityRate contains NaN values")
    else:
        print(f"  GoalValidityRate: min={gvr.min():.3f}  mean={gvr.mean():.3f}")

    # 3. EntropyCoef must be constant 0.01 (Fix 2)
    ec = df[df["tag"] == "Alice/EntropyCoef"]["value"]
    if len(ec) > 0:
        if ec.std() > 1e-4:
            failures.append(f"FAIL: Alice/EntropyCoef is not constant (std={ec.std():.6f}) — SR-coupled entropy controller still present")
        if abs(ec.mean() - 0.01) > 0.005:
            failures.append(f"FAIL: Alice/EntropyCoef != 0.01 (mean={ec.mean():.4f})")
        print(f"  EntropyCoef: mean={ec.mean():.4f}  std={ec.std():.6f}")

    # 4. No NaN in any scalar
    for tag in df["tag"].unique():
        vals = df[df["tag"] == tag]["value"]
        if vals.isna().any():
            failures.append(f"FAIL: NaN values in {tag}")

    if failures:
        print()
        for f in failures:
            print(f"  {f}")
        print("TEST 2 FAIL")
        sys.exit(1)
    else:
        print("TEST 2 PASS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="Path to tensorboard event directory")
    args = parser.parse_args()
    check_sandbox_run(args.log_dir)
