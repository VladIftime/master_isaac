#!/usr/bin/env python3
"""
Live-updating plot of valid goals + avg XY displacement per iteration.

Polls the log file every --interval seconds and updates the existing window
in-place (no flicker, no new window).

Usage:
    python asyncDualPlayPPO/logs/diagnostics/plot_valid_goals_live.py \
        asyncDualPlayPPO/runs/diag_<TS>/test2.log

    # Faster refresh, wider smoothing:
    python ... --interval 5 --window 10
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
    pending_iter = pending_valid = pending_invalid = None
    try:
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
    except FileNotFoundError:
        pass
    return iters, valid, invalid, avg_xy


def smooth(values, window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def main():
    parser = argparse.ArgumentParser(description="Live valid-goals + displacement plot.")
    parser.add_argument("log", help="Path to .log file")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--window", type=int, default=5,
                        help="Smoothing window in iterations (default: 5)")
    parser.add_argument("--no-smooth", action="store_true")
    args = parser.parse_args()

    log_path = args.log
    label    = pathlib.Path(log_path).parent.name

    COLOR  = "#1f77b4"
    COLOR2 = "#e07b2a"

    # ── Build figure once ─────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    ax_r = ax_top.twinx()

    # Top panel lines
    (line_rate_sm,)  = ax_top.plot([], [], color=COLOR, linewidth=2,
                                   label=f"{label} rate (smoothed, w={args.window})")
    (line_count,)    = ax_r.plot(  [], [], color=COLOR, linestyle="--",
                                   alpha=0.4, linewidth=1.2, label="# valid goals")
    (line_total,)    = ax_r.plot(  [], [], color="grey", linestyle=":",
                                   alpha=0.6, linewidth=1.2, label="# total goals")

    ax_top.set_ylabel("Valid-goal rate (%)")
    ax_top.set_ylim(0, 60)
    ax_top.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax_top.grid(alpha=0.3)
    ax_r.set_ylabel("# goals", color="grey")
    ax_r.tick_params(axis="y", labelcolor="grey")
    ax_r.set_ylim(bottom=0)

    handles = [line_rate_sm, line_count, line_total]
    ax_top.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)

    title_top = ax_top.set_title("waiting for data…", fontsize=10, color="grey")

    # Bottom panel lines
    ax_bot_r = ax_bot.twinx()
    (line_xy_sm,)    = ax_bot.plot(  [], [], color=COLOR2, linewidth=2,
                                     label=f"avg XY (smoothed, w={args.window})")
    ax_bot.axhline(2.0, color="red", linestyle=":", linewidth=1, alpha=0.6,
                   label="2 cm threshold")
    (line_bot_total,)= ax_bot_r.plot([], [], color="grey", linestyle=":",
                                     alpha=0.6, linewidth=1.2, label="# total goals")
    ax_bot.set_ylabel("Avg XY displacement (cm)")
    ax_bot.set_xlabel("Iteration")
    ax_bot.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_bot.grid(alpha=0.3)
    ax_bot_r.set_ylabel("# total goals", color="grey")
    ax_bot_r.tick_params(axis="y", labelcolor="grey")
    ax_bot_r.set_ylim(bottom=0)
    handles_bot = [line_xy_sm, line_bot_total]
    ax_bot.legend(handles_bot + [ax_bot.get_lines()[1]],  # include threshold
                  [h.get_label() for h in handles_bot] + ["2 cm threshold"],
                  loc="upper left", fontsize=8)
    title_bot = ax_bot.set_title("", fontsize=10, color="grey")

    # Shared header texts
    iter_text   = fig.text(0.5,  0.98, "Iteration: —",
                           ha="center", va="top", fontsize=16, fontweight="bold")
    status_text = fig.text(0.99, 0.98, f"polling every {args.interval}s",
                           ha="right",  va="top", fontsize=8, color="grey")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)

    # ── Poll loop ─────────────────────────────────────────────────────────────
    last_iter = None
    print(f"Watching {log_path}  (refresh every {args.interval}s) — Ctrl-C to stop")

    while True:
        plt.pause(args.interval)

        iters, valid, invalid, avg_xy = parse_log(log_path)
        if not iters or iters[-1] == last_iter:
            continue

        last_iter = iters[-1]
        x     = np.array(iters)
        v     = np.array(valid, dtype=float)
        total = v + np.array(invalid, dtype=float)
        rate  = np.where(total > 0, v / total, 0.0)

        rate_sm = rate if args.no_smooth else smooth(rate.tolist(), args.window)

        # Update top panel
        line_rate_sm.set_data(x, rate_sm * 100)
        line_count.set_data(  x, v)
        line_total.set_data(  x, total)

        ax_top.set_xlim(x[0], max(x[-1], x[0] + 1))
        ax_r.set_ylim(0, max(total.max() * 1.15, 1))
        title_top.set_text(
            f"last={rate[-1]*100:.1f}%  mean={rate.mean()*100:.1f}%  max={rate.max()*100:.1f}%"
        )

        # Update bottom panel
        xy_vals = np.array([val if val is not None else np.nan for val in avg_xy], dtype=float)
        has_xy  = ~np.isnan(xy_vals)

        if not args.no_smooth and has_xy.sum() >= args.window:
            xy_sm = smooth(xy_vals[has_xy].tolist(), args.window)
            line_xy_sm.set_data(x[has_xy], xy_sm * 100)
        else:
            line_xy_sm.set_data([], [])

        line_bot_total.set_data(x, total)
        ax_bot_r.set_ylim(0, max(total.max() * 1.15, 1))

        if has_xy.any():
            xy_clean = xy_vals[has_xy]
            ax_bot.set_ylim(0, max(xy_clean.max() * 100 * 1.15, 3))
            title_bot.set_text(
                f"mean={xy_clean.mean()*100:.2f}cm  max={xy_clean.max()*100:.2f}cm"
            )

        # Shared header
        iter_text.set_text(f"Iteration: {last_iter}")
        status_text.set_text(f"polling every {args.interval}s · {len(iters)} iters read")

        fig.canvas.draw_idle()
        print(f"  [plot] iter={last_iter}  rate={rate[-1]*100:.1f}%  mean={rate.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
