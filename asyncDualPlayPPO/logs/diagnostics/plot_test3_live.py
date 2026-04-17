#!/usr/bin/env python3
"""
Live-updating plot of PPO vs ABC Tug-of-War (Test 3).

Polls the log file every --interval seconds and updates the existing window
in-place (no flicker, no new window).

Usage:
    python asyncDualPlayPPO/logs/diagnostics/plot_test3_live.py \
        asyncDualPlayPPO/runs/diag_<TS>/test3.log
"""

import re
import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_BOB_RE   = re.compile(
    r"\[Bob Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)"
    r"\s*\|\s*Rew:\s*([-\d.]+)\s*\|\s*ABC:\s*([-\d.]+)\s*\|\s*SR:\s*([\d.]+)"
)

_ITER_RE  = re.compile(
    r"\[Iter\s+(\d+)\].*?ABC buf:\s*(\d+)\s*\|\s*ABC warm:\s*(YES|NO)"
)

def parse_log(path: str):
    bob_steps, ppo_loss, abc_loss = [], [], []
    iters, buf_size, warm = [], [], []
    
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m1 = _BOB_RE.search(line)
                if m1:
                    bob_steps.append(int(m1.group(1)))
                    ppo_loss.append(float(m1.group(2)))
                    abc_loss.append(float(m1.group(5)))
                    continue
                
                m2 = _ITER_RE.search(line)
                if m2:
                    iters.append(int(m2.group(1)))
                    buf_size.append(int(m2.group(2)))
                    warm.append(1 if m2.group(3) == "YES" else 0)
    except FileNotFoundError:
        pass
    
    return bob_steps, ppo_loss, abc_loss, iters, buf_size, warm

def smooth(values, window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]

def export_csv(b_steps, ppo_loss, abc_loss, iters, buf, warm, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "bob_step", "ppo_loss", "abc_loss", "abc_buf", "abc_warm"])
        max_len = max(len(iters), len(b_steps))
        for i in range(max_len):
            it = iters[i] if i < len(iters) else ""
            b_s = b_steps[i] if i < len(b_steps) else ""
            p_l = ppo_loss[i] if i < len(ppo_loss) else ""
            a_l = abc_loss[i] if i < len(abc_loss) else ""
            b_f = buf[i] if i < len(buf) else ""
            w = warm[i] if i < len(warm) else ""
            writer.writerow([it, b_s, p_l, a_l, b_f, w])

def main():
    parser = argparse.ArgumentParser(description="Live Test 3 PPO vs ABC plot.")
    parser.add_argument("log", help="Path to .log file")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--window", type=int, default=5,
                        help="Smoothing window (default: 5)")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--static", action="store_true",
                        help="Run once and display the final plot (no polling)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional: Path to export metrics as CSV (e.g. out.csv)")
    args = parser.parse_args()

    log_path = args.log
    label    = pathlib.Path(log_path).parent.name

    C_PPO = "#e74c3c"  # Red
    C_ABC = "#e67e22"  # Orange
    C_BUF = "#3498db"  # Blue
    C_WARM = "#2ecc71" # Green

    # ── Build figure once ─────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    fig.subplots_adjust(hspace=0.25)

    # Top panel: Losses
    (line_ppo,) = ax_top.plot([], [], color=C_PPO, linewidth=1.5, label="PPO Loss")
    (line_abc,) = ax_top.plot([], [], color=C_ABC, linewidth=1.5, label="ABC Loss")
    
    ax_top.set_ylabel("Loss Magnitude (Log Scale)", color="#333333")
    ax_top.set_yscale("symlog", linthresh=1e-4) # Use symlog to handle zero and negative
    ax_top.grid(alpha=0.3)
    ax_top.set_title("Waiting for loss data...", fontsize=10, color="grey")
    
    ax_top.legend(loc="upper right", fontsize=9)

    # Bottom panel: Buffer and Warmup
    ax_bot_r = ax_bot.twinx()
    
    (line_buf,) = ax_bot.plot([], [], color=C_BUF, linewidth=2, marker=".", ms=4, label="ABC Buffer Size")
    (line_warm,) = ax_bot_r.plot([], [], color=C_WARM, alpha=0.0) # Hidden line for legend
    fill_warm = ax_bot_r.fill_between([], [], alpha=0.3, color=C_WARM, step="mid", label="ABC Warm")
    
    ax_bot.set_ylabel("Buffer Entries", color=C_BUF)
    ax_bot.tick_params(axis="y", labelcolor=C_BUF)
    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylim(bottom=0)
    ax_bot.grid(alpha=0.3)
    
    ax_bot_r.set_ylabel("Warm (1=YES)", color=C_WARM)
    ax_bot_r.tick_params(axis="y", labelcolor=C_WARM)
    ax_bot_r.set_ylim(-0.1, 1.2)
    ax_bot_r.set_yticks([0, 1])
    
    handles = [line_buf, plt.Rectangle((0,0),1,1, fc=C_WARM, alpha=0.3)]
    ax_bot.legend(handles, ["ABC Buffer Size", "ABC Warm Flag"], loc="center left", fontsize=9)
    
    title_bot = ax_bot.set_title("", fontsize=10, color="grey")

    # Shared header texts
    iter_text = fig.text(0.5, 0.96, "Test 3: PPO vs ABC Tug-of-War", ha="center", va="top", fontsize=14, fontweight="bold")
    status_text = fig.text(0.99, 0.98, f"polling every {args.interval}s", ha="right", va="top", fontsize=8, color="grey")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if not args.static:
        plt.show(block=False)

    # ── Poll loop ─────────────────────────────────────────────────────────────
    last_len = 0
    first_run = True
    if args.static:
        print(f"Reading {log_path} (static mode)")
    else:
        print(f"Watching {log_path}  (refresh every {args.interval}s) — Ctrl-C to stop")

    while True:
        if not args.static and not first_run:
            plt.pause(args.interval)

        b_steps, ppo_loss, abc_loss, iters, buf, warm = parse_log(log_path)
        
        if not args.static and len(iters) <= last_len and len(b_steps) <= last_len:
            continue
            
        last_len = max(len(iters), len(b_steps))
        
        if args.csv:
            export_csv(b_steps, ppo_loss, abc_loss, iters, buf, warm, args.csv)
        
        # Top Panel (Losses)
        if len(b_steps) > 0:
            x_b = np.array(b_steps)
            
            p_loss = np.abs(np.array(ppo_loss)) # Plot absolute magnitude for symlog
            a_loss = np.array(abc_loss) # ABC is purely positive
            
            if not args.no_smooth:
                p_loss = smooth(p_loss, args.window)
                a_loss = smooth(a_loss, args.window)
                
            line_ppo.set_data(x_b, p_loss)
            line_abc.set_data(x_b, a_loss)
            
            ax_top.set_xlim(max(0, x_b[0] - 1), x_b[-1] + 1)
            
            # Auto-scale symlog y-axis reasonably
            max_loss = max(p_loss.max() if len(p_loss) else 0, a_loss.max() if len(a_loss) else 0)
            if max_loss > 0:
                ax_top.set_ylim(-1e-4, max(max_loss * 5, 0.1))
                
            abc_active = (a_loss > 0).any()
            ratio_status = ""
            if abc_active and len(x_b) > 0:
                l_p = p_loss[-1] + 1e-9
                l_a = a_loss[-1]
                ratio = l_a / l_p
                ratio_status = f" | Ratio ABC/PPO: {ratio:.1f}x"
                if ratio > 10:
                    ratio_status += " [🚩 HIGH]"
                elif ratio < 0.1:
                    ratio_status += " [🚩 LOW]"
                    
            ax_top.set_title(
                f"PPO Loss: {p_loss[-1]:.4f}  ABC Loss: {a_loss[-1]:.4f}{ratio_status}", 
                fontsize=10, 
                color="black" if abc_active and ratio < 10 else "darkred"
            )

        # Bottom Panel (Buffer)
        if len(iters) > 0:
            x_i = np.array(iters)
            y_buf = np.array(buf)
            y_warm = np.array(warm)
            
            line_buf.set_data(x_i, y_buf)
            
            # Recreate fill_between because updating it is tricky
            for coll in list(ax_bot_r.collections):
                coll.remove()
            ax_bot_r.fill_between(x_i, y_warm, alpha=0.3, color=C_WARM, step="mid")
            
            ax_bot.set_xlim(max(0, x_i[0] - 1), x_i[-1] + 1)
            ax_bot.set_ylim(0, max(y_buf.max() * 1.2, 500))
            
            warm_status = "YES" if y_warm[-1] == 1 else "NO"
            ax_bot.set_title(f"Buffer Size: {y_buf[-1]} | Warmup: {warm_status}", fontsize=10)

        # Status
        if args.static:
            status_text.set_text(f"static mode · {len(iters)} iters read")
        else:
            status_text.set_text(f"polling every {args.interval}s · {len(iters)} iters read")
        fig.canvas.draw_idle()
        
        last_ratio = -1
        if len(ppo_loss) > 0 and len(abc_loss) > 0:
            last_ratio = abs(abc_loss[-1]) / (abs(ppo_loss[-1]) + 1e-9)
        print(f"  [plot] iter={iters[-1] if iters else 0} | PPO loss={ppo_loss[-1] if ppo_loss else 0:.4f} | " + 
              f"ABC loss={abc_loss[-1] if abc_loss else 0:.4f} | Ratio={last_ratio:.1f}x | Buf={buf[-1] if buf else 0}")

        if args.static:
            plt.show(block=True)
            break
            
        first_run = False

if __name__ == "__main__":
    main()
