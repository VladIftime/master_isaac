#!/usr/bin/env python3
"""
Analyze SLURM log output from train_push.py / train_push_asp.py / train_curobo.py.

Scans a log directory for slurm-*.out / .txt / .log files, traces job chains,
writes CSVs/TXTs, and generates per-type training metric plots.

Usage:
  python asyncDualPlayPPO/extras/analyze_push.py --log-dir logs/experiment
  python asyncDualPlayPPO/extras/analyze_push.py --log-dir logs/experiment -o analysis/ --merge-chains
"""

import os
import shutil
import sys
from pathlib import Path

# Allow running as both `python asyncDualPlayPPO/extras/analyze_push.py` and `python -m ...`
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from pathlib import Path

from asyncDualPlayPPO.extras.analyze_training import (
    parse_logs,
    trace_chains,
    merge_all_chains,
    assign_global_iters,
    detect_training_type,
    write_training_csv,
    write_raw_csv,
    write_raw_logs,
    write_summary_txt,
    print_summary,
    _plot_push,
    _plot_push_asp,
    _plot_curobo,
    TRAIN_PUSH,
    TRAIN_PUSH_ASP,
    TRAIN_CUROBO,
)


def main():
    import argparse

    type_names = {TRAIN_PUSH: "Push-PPO Baseline", TRAIN_PUSH_ASP: "Push-ASP",
                  TRAIN_CUROBO: "cuRobo ASP"}

    parser = argparse.ArgumentParser(
        description="Analyze and stitch training logs across SLURM job chains."
    )
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Directory containing slurm-*.out / .txt / .log files.")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory (defaults to --log-dir).")
    parser.add_argument("--merge-chains", action="store_true", default=False,
                        help="Collapse all discovered chains into one, ordered by job ID.")
    parser.add_argument("--separate-plots", action="store_true", default=False,
                        help="Save one PNG per metric instead of a combined overview.")

    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir

    if not log_dir.exists():
        parser.error(f"--log-dir does not exist: {log_dir}")

    print(f"[INFO] Scanning {log_dir} for slurm log files ...")
    jobs = parse_logs(log_dir)
    print(f"[INFO] Found {len(jobs)} job log file(s)")

    type_counts = {}
    for j in jobs.values():
        t = j.get("train_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in type_counts.items():
        print(f"       {type_names.get(t, t)}: {c} file(s)")

    chains = trace_chains(jobs)
    print(f"[INFO] Found {len(chains)} chain(s):")
    for i, ch in enumerate(chains):
        print(f"       Chain {i}: {len(ch)} jobs  [{ch[0]} → ... → {ch[-1]}]")

    if args.merge_chains and len(chains) > 1:
        chains = merge_all_chains(chains)
        print(f"[INFO] --merge-chains: collapsed to {len(chains)} chain(s)")

    alice_records, bob_records, push_records = assign_global_iters(chains, jobs)
    print(f"[INFO] Records — Alice: {len(alice_records)}, Bob: {len(bob_records)}, "
          f"Push: {len(push_records)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ch in enumerate(chains):
        ch_types = {}
        for jid in ch:
            t = jobs[jid].get("train_type", "unknown")
            ch_types[t] = ch_types.get(t, 0) + 1
        ch_type = max(ch_types, key=ch_types.get) if ch_types else TRAIN_PUSH

        chain_dir = out_dir / f"chain_{i}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing Chain {i} → {chain_dir}  "
              f"({type_names.get(ch_type, ch_type)})")

        for job_id in ch:
            src = jobs[job_id]["path"]
            dest = chain_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)

        a_c = [r for r in alice_records if r["chain"] == i]
        b_c = [r for r in bob_records if r["chain"] == i]
        p_c = [r for r in push_records if r["chain"] == i]

        write_raw_csv(i, ch, jobs, chain_dir)
        write_raw_logs(ch, jobs, chain_dir)
        write_summary_txt(i, ch, a_c, b_c, p_c, chain_dir, ch_type)
        write_training_csv(a_c, b_c, p_c, chain_dir)

        plots_dir = chain_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        log_paths = [jobs[jid]["path"] for jid in ch]
        suffix = f" (Chain {i})"

        if ch_type == TRAIN_PUSH:
            _plot_push(p_c, plots_dir, suffix, log_paths, args.separate_plots)
        elif ch_type == TRAIN_PUSH_ASP:
            _plot_push_asp(a_c, b_c, plots_dir, suffix, args.separate_plots)
        elif ch_type == TRAIN_CUROBO:
            _plot_curobo(a_c, b_c, plots_dir, suffix, args.separate_plots)

    # Per-chain summary
    for i, ch in enumerate(chains):
        ch_types = {}
        for jid in ch:
            t = jobs[jid].get("train_type", "unknown")
            ch_types[t] = ch_types.get(t, 0) + 1
        ch_type = max(ch_types, key=ch_types.get) if ch_types else TRAIN_PUSH
        a_c = [r for r in alice_records if r["chain"] == i]
        b_c = [r for r in bob_records if r["chain"] == i]
        p_c = [r for r in push_records if r["chain"] == i]
        if p_c or a_c or b_c:
            print(f"\n  Chain {i} ({type_names.get(ch_type, ch_type)}):", flush=True)
            print_summary(a_c, b_c, p_c, ch_type, compact=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
