"""
Test 2 — Alice Exploration Sandbox (offline analysis).

Run the full training with --alice_sandbox to generate a tensorboard log, then
call check_sandbox_run() to assert curriculum health from the event file.

Usage:
    # 1. Generate the run (from master_isaac/ root):
    python -m asyncDualPlayPPO.train_curobo --alice_sandbox --num_envs 32 \\
        --max_iterations 200 --headless --exp_name diag_sandbox

    # 2. Analyse the run:
    python -m asyncDualPlayPPO.diagnostics.test_alice_sandbox \\
        --log_dir asyncDualPlayPPO/runs/diag_sandbox/summary

What is checked:
    - Metrics/Alice/ValidGoals trends upward (last-20 avg > first-20 avg)
    - Metrics/Alice/GoalValidityRate never NaN
    - Alice/EntropyCoef is constant 0.05 (ent_coef in ppo_continuous.yaml, Fix 2)
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

    # 3. EntropyCoef must be constant 0.05 (ent_coef in ppo_continuous.yaml, Fix 2)
    EXPECTED_ENT_COEF = 0.05
    ec = df[df["tag"] == "Alice/EntropyCoef"]["value"]
    if len(ec) > 0:
        if ec.std() > 1e-4:
            failures.append(f"FAIL: Alice/EntropyCoef is not constant (std={ec.std():.6f}) — SR-coupled entropy controller still present")
        if abs(ec.mean() - EXPECTED_ENT_COEF) > 0.005:
            failures.append(f"FAIL: Alice/EntropyCoef != {EXPECTED_ENT_COEF} (mean={ec.mean():.4f}) — check ent_coef in ppo_continuous.yaml")
        print(f"  EntropyCoef: mean={ec.mean():.4f}  std={ec.std():.6f}  (expected {EXPECTED_ENT_COEF})")

    # 4. InvalidGoals must appear at least once (confirms OOZ penalty logic fires)
    inv_goals = df[df["tag"] == "Metrics/Alice/InvalidGoals"]["value"]
    if len(inv_goals) == 0:
        failures.append(
            "FAIL: Metrics/Alice/InvalidGoals not logged — "
            "train_curobo.py may be missing the InvalidGoals writer call"
        )
    elif inv_goals.sum() == 0:
        failures.append(
            "FAIL: Metrics/Alice/InvalidGoals is always 0 — "
            "out-of-zone penalty logic may never have fired. "
            "Check _handle_alice_completion() in wrapper.py and "
            "Alice's workspace boundary enforcement."
        )
    else:
        print(f"  InvalidGoals total: {int(inv_goals.sum())}  (OOZ penalty confirmed firing)")

    # 5. No NaN in any scalar
    for tag in df["tag"].unique():
        vals = df[df["tag"] == tag]["value"]
        if vals.isna().any():
            failures.append(f"FAIL: NaN values in {tag}")

    # 6. MeanDisp3D floor — Alice must be moving objects, not exploiting micro-displacement
    disp3d = df[df["tag"] == "Metrics/Alice/MeanDisp3D"]["value"]
    if len(disp3d) >= 20:
        late_disp = disp3d.values[-20:].mean()
        print(f"  MeanDisp3D: late_avg={late_disp:.4f}m  (floor=0.04m)")
        if late_disp <= 0.04:
            failures.append(
                f"FAIL: MeanDisp3D floor violated: {late_disp:.4f}m (need > 0.04m). "
                "Alice is exploiting micro-displacement — objects are not meaningfully moved."
            )

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
