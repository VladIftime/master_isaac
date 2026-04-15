#!/usr/bin/env python3
"""
Plot valid goals + avg XY displacement per iteration from a test2 (or any ASP) log file.

Usage:
    python asyncDualPlayPPO/logs/diagnostics/plot_valid_goals.py \
        asyncDualPlayPPO/runs/diag_<TS>/test2.log

    # Compare multiple runs:
    python asyncDualPlayPPO/logs/diagnostics/plot_valid_goals.py \
        asyncDualPlayPPO/runs/diag_A/test2.log \
        asyncDualPlayPPO/runs/diag_B/test2.log
"""

import re
import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_ITER_RE = re.compile(r"\[Iter\s+(\d+)\].*?Goals valid=(\d+)\s+invalid=(\d+)")
_DISP_RE = re.compile(r"\[AliceDisp\].*?avg XY=([0-9.]+)m")


def parse_log(path: str):
    iters, valid, invalid, avg_xy = [], [], [], []
    pending_iter = None
    pending_valid = None
    pending_invalid = None
    with open(path) as f:
        for line in f:
            m = _ITER_RE.search(line)
            if m:
                if pending_iter is not None:
                    iters.append(pending_iter)
                    valid.append(pending_valid)
                    invalid.append(pending_invalid)
                    avg_xy.append(None)
                pending_iter    = int(m.group(1))
                pending_valid   = int(m.group(2))
                pending_invalid = int(m.group(3))
                continue

            d = _DISP_RE.search(line)
            if d and pending_iter is not None:
                iters.append(pending_iter)
                valid.append(pending_valid)
                invalid.append(pending_invalid)
                avg_xy.append(float(d.group(1)))
                pending_iter = None

    if pending_iter is not None:
        iters.append(pending_iter)
        valid.append(pending_valid)
        invalid.append(pending_invalid)
        avg_xy.append(None)

    return iters, valid, invalid, avg_xy


def smooth(values, window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def main():
    parser = argparse.ArgumentParser(description="Plot Alice valid goals + displacement vs iteration.")
    parser.add_argument("logs", nargs="+", help="Path(s) to .log file(s)")
    parser.add_argument("--window", type=int, default=5, help="Smoothing window (iterations)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing")
    parser.add_argument("--out", default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for log_path in args.logs:
        label = pathlib.Path(log_path).parent.name
        iters, valid, invalid, avg_xy = parse_log(log_path)
        if not iters:
            print(f"[WARN] No [Iter] lines found in {log_path}")
            continue

        x = np.array(iters)
        v = np.array(valid, dtype=float)
        total = v + np.array(invalid, dtype=float)
        rate  = np.where(total > 0, v / total, 0.0)

        rate_smoothed = rate if args.no_smooth else smooth(rate.tolist(), args.window)

        raw_line, = ax_top.plot(x, rate * 100,          alpha=0.25, linewidth=1, label=f"{label} rate (raw)")
        color = raw_line.get_color()
        ax_top.plot(x, rate_smoothed * 100, linewidth=2, color=color,
                    label=f"{label} rate (smoothed, w={args.window})")

        ax_r = ax_top.twinx()
        ax_r.plot(x, v, color=color, linestyle="--", alpha=0.4, linewidth=1.2, label=f"{label} # valid goals")
        ax_r.set_ylabel("# valid goals", color="grey")
        ax_r.tick_params(axis="y", labelcolor="grey")
        ax_r.set_ylim(bottom=0)

        # ── Displacement ──────────────────────────────────────────────────────
        xy_vals = np.array([val if val is not None else np.nan for val in avg_xy], dtype=float)
        has_xy  = ~np.isnan(xy_vals)

        if has_xy.any():
            ax_bot.plot(x, xy_vals * 100, alpha=0.25, linewidth=1,
                        color=color, label=f"{label} avg XY (raw)")
            if not args.no_smooth and has_xy.sum() >= args.window:
                xy_sm = smooth(xy_vals[has_xy].tolist(), args.window)
                ax_bot.plot(x[has_xy], xy_sm * 100, linewidth=2,
                            color=color, label=f"{label} avg XY (smoothed, w={args.window})")

        print(
            f"{label}: {len(iters)} iters | "
            f"mean rate={rate.mean()*100:.1f}% | final rate={rate[-1]*100:.1f}% | max rate={rate.max()*100:.1f}%"
        )

    ax_top.set_ylabel("Valid-goal rate (%)")
    ax_top.set_title("Alice: valid-goal rate per iteration")
    ax_top.set_ylim(0, 100)
    ax_top.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax_top.grid(alpha=0.3)

    handles, labels_ = ax_top.get_legend_handles_labels()
    hr, lr = ax_r.get_legend_handles_labels()
    ax_top.legend(handles + hr, labels_ + lr, loc="upper left", fontsize=8)

    ax_bot.axhline(2.0, color="red", linestyle=":", linewidth=1, alpha=0.6, label="2 cm threshold")
    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylabel("Avg XY displacement (cm)")
    ax_bot.set_title("Alice: avg XY displacement per iteration")
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(alpha=0.3)
    ax_bot.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
