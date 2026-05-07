"""
Test 3a — PPO and ABC Optimization Balance (offline analysis).

Reads a tensorboard event directory from a completed training run and verifies:
    1. Bob/ABCCoef is constant 0.5 ± 0.001 (Fix 1 — no SR-coupling)
    2. Loss/Bob/ABC > 0 once buffer has ≥1 trajectory (Fix 5 — combined loss)
    3. Loss/Bob/Surrogate is finite and non-zero
    4. ABC loss does not dwarf PPO surrogate by >5×

Usage:
    # 1. Run 50 iterations of full pipeline:
    python -m asyncDualPlayPPO.train --num_envs 32 --max_iterations 50 \\
        --headless --exp_name diag_t3

    # 2. Analyse:
    python -m asyncDualPlayPPO.diagnostics.test_ppo_abc_balance \\
        --log_dir asyncDualPlayPPO/runs/diag_t3/summary
"""

import argparse
import sys


def _load_df(log_dir: str):
    try:
        from tbparse import SummaryReader
        return SummaryReader(log_dir).scalars
    except ImportError:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import pandas as pd
        ea = EventAccumulator(log_dir)
        ea.Reload()
        rows = []
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                rows.append({"tag": tag, "step": e.step, "value": e.value})
        return pd.DataFrame(rows)


def check_ppo_abc_balance(log_dir: str):
    import numpy as np

    print("\n" + "=" * 70)
    print("TEST 3a — PPO and ABC Optimization Balance")
    print(f"Log dir: {log_dir}")
    print("=" * 70)

    df = _load_df(log_dir)
    failures = []

    # 1. ABCCoef must be constant 0.5
    abc_coef = df[df["tag"] == "Bob/ABCCoef"]["value"]
    if len(abc_coef) == 0:
        failures.append("FAIL: Bob/ABCCoef not found in log — check train_curobo.py logging block")
    else:
        coef_std = abc_coef.std()
        coef_mean = abc_coef.mean()
        print(f"  ABCCoef: mean={coef_mean:.4f}  std={coef_std:.6f}")
        if coef_std > 1e-3:
            failures.append(
                f"FAIL: abc_coef is not constant (std={coef_std:.6f}) — "
                "SR-coupling may still be active (Fix 1 not applied)"
            )
        if abs(coef_mean - 0.5) > 0.01:
            failures.append(f"FAIL: abc_coef mean={coef_mean:.4f} != 0.5 (paper Table 2)")

    # 2. ABC loss must be >0 once buffer non-empty
    buf_size = df[df["tag"] == "Metrics/ABC/BufferSize"]["value"]
    abc_loss_series = df[df["tag"] == "Loss/Bob/ABC"]
    if len(buf_size) > 0 and len(abc_loss_series) > 0:
        buf_steps = df[df["tag"] == "Metrics/ABC/BufferSize"]["step"].values
        buf_vals = buf_size.values
        warm_mask = buf_vals > 0
        if warm_mask.any():
            first_warm_step = buf_steps[warm_mask][0]
            abc_after = abc_loss_series[abc_loss_series["step"] >= first_warm_step]["value"]
            nonzero = (abc_after > 1e-8).sum()
            print(f"  ABC loss after buffer warm-up: {nonzero}/{len(abc_after)} steps > 0")
            if nonzero == 0:
                failures.append(
                    "FAIL: ABC loss is zero despite non-empty buffer — "
                    "combined loss injection (Fix 5) may not be working"
                )
        else:
            print("  Buffer never filled during this run — ABC loss check skipped")
    else:
        print("  ABC/BufferSize or Loss/Bob/ABC not found — skipping ABC loss check")

    # 3. Surrogate loss must be finite and non-zero
    surr = df[df["tag"] == "Loss/Bob/Surrogate"]["value"]
    if len(surr) == 0:
        failures.append("FAIL: Loss/Bob/Surrogate not found in log")
    else:
        finite_frac = np.isfinite(surr.values).mean()
        nonzero_any = (surr.abs() > 1e-6).any()
        print(f"  Surrogate: finite_frac={finite_frac:.3f}  any_nonzero={nonzero_any}")
        if finite_frac < 1.0:
            failures.append(f"FAIL: non-finite surrogate loss detected ({1-finite_frac:.1%} of steps)")
        if not nonzero_any:
            failures.append("FAIL: surrogate loss is identically zero — storage or PPO update broken")

    # 4. ABC should not dominate PPO surrogate by >5×
    if len(surr) > 0 and len(abc_loss_series) > 0:
        common_steps = min(len(surr), len(abc_loss_series["value"]))
        if common_steps > 0:
            abc_vals = abc_loss_series["value"].values[-common_steps:]
            surr_vals = surr.values[-common_steps:]
            ratio = abc_vals / (np.abs(surr_vals) + 1e-8)
            mean_ratio = ratio.mean()
            print(f"  ABC/Surrogate loss ratio: mean={mean_ratio:.2f}  (warn threshold: 5.0)")
            if mean_ratio > 5.0:
                print(f"  WARNING: ABC loss is >{mean_ratio:.1f}× surrogate — may overwrite PPO signal")

    if failures:
        print()
        for f in failures:
            print(f"  {f}")
        print("TEST 3a FAIL")
        sys.exit(1)
    else:
        print("TEST 3a PASS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="Path to tensorboard event directory")
    args = parser.parse_args()
    check_ppo_abc_balance(args.log_dir)
