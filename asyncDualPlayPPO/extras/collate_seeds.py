#!/usr/bin/env python3
"""Collate validation CSVs across seeds into mean +/- 95% CI per model.

Scans a results root (default: /scratch/$USER/final_results_thesis) for run
directories following the convention::

    <model>_e<envs>_i<iters>_s<seed>/validation_results_*.csv

Each validation CSV (from tests/validate_push.py or validate_push_asp.py) has a
per-scene ``success`` column (0/1) and a ``test_type`` difficulty column.  This
script computes, per run, the overall success rate and per-difficulty success
rate, then aggregates across seeds sharing the same (model, envs, iters) key
into mean +/- 95% confidence interval.

Outputs (written next to the results root, or to --out-dir):
    collated_summary.csv   -- one row per (model, envs, iters, validator)
    collated_summary.md    -- human-readable markdown table

Stdlib only (csv, statistics, glob, re, math, argparse).
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict

# Two-sided 95% t-values by degrees of freedom (n-1). Fallback to normal (1.96).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 30: 2.042}

# EXP_NAME conventions (both allow an optional trailing _<tag> ablation suffix,
# e.g. _noge / _abc / _nohist — the tag is folded into the model label so each
# ablation aggregates as its own row):
#   training/self : <model>_e<envs>_i<iters>_s<seed>[_<tag>]
#   gym crossover : <model>_e<envs>_p<push_nsteps>_s<seed>[_<tag>]
_RUN_RE = re.compile(r"^(?P<model>.+?)_e(?P<envs>\d+)_i(?P<xval>\d+)_s(?P<seed>\d+)(?:_(?P<tag>.+))?$")
_GYM_RE = re.compile(r"^(?P<model>.+?)_e(?P<envs>\d+)_p(?P<xval>\d+)_s(?P<seed>\d+)(?:_(?P<tag>.+))?$")


def t95(n):
    """95% two-sided t multiplier for a sample of size n."""
    if n < 2:
        return float("nan")
    df = n - 1
    if df in _T95:
        return _T95[df]
    # nearest tabulated df <= actual, else normal approx
    keys = [k for k in _T95 if k <= df]
    return _T95[max(keys)] if keys else 1.96


def parse_run_name(name):
    for rx, xkey in ((_RUN_RE, "iters"), (_GYM_RE, "push_nsteps")):
        m = rx.match(name)
        if m:
            d = m.groupdict()
            model = d["model"] + (f"_{d['tag']}" if d.get("tag") else "")
            return {"model": model, "envs": int(d["envs"]),
                    "xkey": xkey, "xval": int(d["xval"]), "seed": int(d["seed"])}
    return None


def summarize_csv(path):
    """Return dict of overall + per-difficulty success rate for one CSV."""
    by_diff = defaultdict(list)
    allrows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "success" not in row:
                continue
            try:
                s = float(row["success"])
            except (ValueError, TypeError):
                continue
            allrows.append(s)
            diff = (row.get("test_type") or "all").strip().lower()
            by_diff[diff].append(s)
    if not allrows:
        return None
    out = {"overall": statistics.mean(allrows), "n_scenes": len(allrows)}
    for diff, vals in by_diff.items():
        out[f"sr_{diff}"] = statistics.mean(vals)
    return out


def agg(values):
    """mean, ci95 (half-width), n for a list of per-seed rates."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, float("nan"), n
    sd = statistics.stdev(values)
    ci = t95(n) * sd / math.sqrt(n)
    return mean, ci, n


def main():
    default_root = os.path.expandvars("/scratch/$USER/final_results_thesis")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=default_root,
                    help=f"Root holding run dirs (default: {default_root})")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write collated_summary.{csv,md} (default: results-root)")
    args = ap.parse_args()

    root = os.path.expandvars(os.path.expanduser(args.results_root))
    out_dir = args.out_dir or root
    os.makedirs(out_dir, exist_ok=True)

    # group_key -> validator -> difficulty-metric -> [per-seed rate]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    group_meta = {}
    seen_runs = 0

    for csv_path in sorted(glob.glob(os.path.join(root, "*", "validation_results_*.csv"))):
        run_dir = os.path.basename(os.path.dirname(csv_path))
        meta = parse_run_name(run_dir)
        if meta is None:
            print(f"[skip] run dir does not match convention: {run_dir}")
            continue
        summary = summarize_csv(csv_path)
        if summary is None:
            print(f"[skip] no usable rows: {csv_path}")
            continue
        validator = os.path.basename(csv_path)[len("validation_results_"):-len(".csv")]
        gkey = (meta["model"], meta["envs"], meta["xkey"], meta["xval"])
        group_meta[gkey] = meta
        for metric, val in summary.items():
            if metric == "n_scenes":
                continue
            groups[gkey][validator][metric].append(val)
        seen_runs += 1

    if seen_runs == 0:
        print(f"[warn] no matching runs under {root}")
        return

    # Build rows
    rows = []
    for gkey in sorted(groups):
        model, envs, xkey, xval = gkey
        for validator in sorted(groups[gkey]):
            metrics = groups[gkey][validator]
            overall = metrics.get("overall", [])
            mean, ci, n = agg(overall)
            row = {"model": model, "envs": envs, "batch_var": xkey, "iters_or_pns": xval,
                   "validator": validator, "n_seeds": n,
                   "overall_mean": round(mean, 4),
                   "overall_ci95": round(ci, 4) if not math.isnan(ci) else ""}
            for diff_metric in sorted(m for m in metrics if m.startswith("sr_")):
                dmean, dci, _ = agg(metrics[diff_metric])
                row[f"{diff_metric}_mean"] = round(dmean, 4)
                row[f"{diff_metric}_ci95"] = round(dci, 4) if not math.isnan(dci) else ""
            rows.append(row)

    # Write CSV (union of all keys)
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    csv_out = os.path.join(out_dir, "collated_summary.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write markdown
    md_out = os.path.join(out_dir, "collated_summary.md")
    with open(md_out, "w") as f:
        f.write("# Validation summary (mean +/- 95% CI across seeds)\n\n")
        f.write(f"Source: `{root}`  |  runs collated: {seen_runs}\n\n")
        f.write("| model | envs | batch_var | iters/pns | validator | n | overall SR (95% CI) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            ci = r["overall_ci95"]
            ci_str = f"{r['overall_mean']:.3f} +/- {ci:.3f}" if ci != "" else f"{r['overall_mean']:.3f} (n=1)"
            f.write(f"| {r['model']} | {r['envs']} | {r['batch_var']} | {r['iters_or_pns']} | "
                    f"{r['validator']} | {r['n_seeds']} | {ci_str} |\n")

    print(f"[ok] collated {seen_runs} runs -> {csv_out}")
    print(f"[ok] markdown -> {md_out}")


if __name__ == "__main__":
    main()
