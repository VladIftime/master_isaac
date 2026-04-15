#!/usr/bin/env python3
"""
plot_diagnostics.py — ASP diagnostic log plotter

Usage:
    python asyncDualPlayPPO/plot_diagnostics.py                          # default paths
    python asyncDualPlayPPO/plot_diagnostics.py --test1 path/to/test1.log ...
    python asyncDualPlayPPO/plot_diagnostics.py --out results.png

Each test gets its own figure with test-specific panels plus a shared summary
figure at the end.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; switch to TkAgg/Qt5Agg locally if you want GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─── Colours ──────────────────────────────────────────────────────────────────
C = {
    "sr":      "#2ecc71",
    "rew":     "#3498db",
    "loss":    "#e74c3c",
    "val":     "#9b59b6",
    "abc":     "#e67e22",
    "valid":   "#1abc9c",
    "invalid": "#e74c3c",
    "buf":     "#f39c12",
    "alice_rew": "#3498db",
    "alice_loss": "#e74c3c",
}

PASS_COLOR  = "#2ecc71"
WARN_COLOR  = "#f39c12"
FAIL_COLOR  = "#e74c3c"

# ─── Parsing ──────────────────────────────────────────────────────────────────
_BOB_RE   = re.compile(
    r"\[Bob Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)"
    r"\s*\|\s*Rew:\s*([-\d.]+)\s*\|\s*ABC:\s*([-\d.]+)\s*\|\s*SR:\s*([\d.]+)"
)
_ALICE_RE = re.compile(
    r"\[Alice Update\s+(\d+)\]\s+Loss:\s*([-\d.]+)\s*\|\s*Val:\s*([-\d.]+)"
    r"\s*\|\s*Rew:\s*([-\d.]+)"
)
_ITER_RE  = re.compile(
    r"\[Iter\s+(\d+)\]\s+SR=([\d.]+)\s*\|"
    r"\s*Goals valid=(\d+)\s+invalid=(\d+)\s*\|"
    r"\s*Bob succ=(\d+)\s+fail=(\d+)\s*\|"
    r".*?ABC buf:\s*(\d+|FULL)\s*\|"
    r"\s*ABC warm:\s*(YES|NO)"
)


def parse_log(path: Path) -> dict:
    """Return dicts of per-update and per-iteration metrics."""
    bob   = {"step": [], "loss": [], "val": [], "rew": [], "abc": [], "sr": []}
    alice = {"step": [], "loss": [], "val": [], "rew": []}
    itr   = {"iter": [], "sr": [], "valid": [], "invalid": [],
              "succ": [], "fail": [], "buf": [], "warm": []}

    with path.open(errors="replace") as fh:
        for line in fh:
            m = _BOB_RE.search(line)
            if m:
                bob["step"].append(int(m.group(1)))
                bob["loss"].append(float(m.group(2)))
                bob["val"].append(float(m.group(3)))
                bob["rew"].append(float(m.group(4)))
                bob["abc"].append(float(m.group(5)))
                bob["sr"].append(float(m.group(6)))
                continue

            m = _ALICE_RE.search(line)
            if m:
                alice["step"].append(int(m.group(1)))
                alice["loss"].append(float(m.group(2)))
                alice["val"].append(float(m.group(3)))
                alice["rew"].append(float(m.group(4)))
                continue

            m = _ITER_RE.search(line)
            if m:
                itr["iter"].append(int(m.group(1)))
                itr["sr"].append(float(m.group(2)))
                itr["valid"].append(int(m.group(3)))
                itr["invalid"].append(int(m.group(4)))
                itr["succ"].append(int(m.group(5)))
                itr["fail"].append(int(m.group(6)))
                buf_str = m.group(7)
                if buf_str == "FULL":
                    itr["buf"].append(2000) # placeholder for full buffer
                else:
                    itr["buf"].append(int(buf_str))
                itr["warm"].append(1 if m.group(8) == "YES" else 0)

    # convert to numpy for convenience
    for d in (bob, alice, itr):
        for k in d:
            d[k] = np.array(d[k])

    return {"bob": bob, "alice": alice, "iter": itr}


# ─── Diagnostic helpers ───────────────────────────────────────────────────────

def _pass_fail(condition: bool) -> tuple[str, str]:
    """Return (label, colour) for a pass/fail badge."""
    return ("PASS", PASS_COLOR) if condition else ("FAIL", FAIL_COLOR)


def _add_badge(ax, text, color, x=0.97, y=0.97):
    ax.text(x, y, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"))


# ─── Test 1 figure ────────────────────────────────────────────────────────────

def plot_test1(data: dict, out_path: Path | None):
    bob   = data["bob"]
    alice = data["alice"]
    itr   = data["iter"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Test 1 — PPO Reward Pipeline  (DummyBobWrapper teleport)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (0,0)  Bob SR over updates ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(bob["step"], bob["sr"], color=C["sr"], lw=1.8, marker="o", ms=3)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8, label="SR = 1.0")
    ax.set_title("Bob Success Rate")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("SR")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7)
    first_sr_pos = np.any(bob["sr"] > 0)
    txt, col = _pass_fail(bool(first_sr_pos))
    _add_badge(ax, txt, col)
    ax.grid(True, alpha=0.3)

    # ── (0,1)  Bob Mean Reward ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(bob["step"], bob["rew"], color=C["rew"], lw=1.8, marker="o", ms=3)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Bob Mean Reward")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("Mean reward")
    txt, col = _pass_fail(bool(np.any(bob["rew"] > 0)))
    _add_badge(ax, txt, col)
    ax.grid(True, alpha=0.3)

    # ── (0,2)  Bob PPO Loss ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(bob["step"], bob["loss"], color=C["loss"], lw=1.5)
    ax.set_title("Bob PPO Loss (surrogate)")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ── (1,0)  Alice Reward ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(alice["step"], alice["rew"], color=C["alice_rew"], lw=1.5, marker="s", ms=3)
    ax.set_title("Alice Mean Reward")
    ax.set_xlabel("Alice update #"); ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.3)

    # ── (1,1)  Goals valid/invalid per iter ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(itr["iter"], itr["valid"],   color=C["valid"],   label="Valid",   alpha=0.8)
    ax.bar(itr["iter"], itr["invalid"], bottom=itr["valid"],
           color=C["invalid"], label="Invalid", alpha=0.5)
    ax.set_title("Goals Valid / Invalid per Iter")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Goals")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── (1,2)  ABC warm flag ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.fill_between(itr["iter"], itr["warm"], alpha=0.6,
                    color=C["abc"], step="mid", label="ABC warm")
    ax.set_title("ABC Warmup Flag")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Warm (1=YES)")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, out_path, "test1")


# ─── Test 2 figure ────────────────────────────────────────────────────────────

def plot_test2(data: dict, out_path: Path | None):
    alice = data["alice"]
    itr   = data["iter"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Test 2 — Alice Exploration Sandbox", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (0,0)  Valid goals over time ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(itr["iter"], itr["valid"], color=C["valid"], lw=2, marker="o", ms=3)
    ax.set_title("Valid Goals per Iteration")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Valid goals")
    max_valid = itr["valid"].max() if len(itr["valid"]) else 0
    txt, col = _pass_fail(max_valid > 0)
    _add_badge(ax, txt, col)
    ax.grid(True, alpha=0.3)

    # ── (0,1)  Goals valid vs invalid stacked ────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.stackplot(itr["iter"],
                 itr["valid"], itr["invalid"],
                 labels=["Valid", "Invalid"],
                 colors=[C["valid"], C["invalid"]],
                 alpha=0.75)
    ax.set_title("Goals Stack (Valid / Invalid)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Goals")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── (0,2)  Valid goal fraction ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    total = itr["valid"] + itr["invalid"]
    frac = np.where(total > 0, itr["valid"] / total, 0.0)
    ax.plot(itr["iter"], frac, color=C["valid"], lw=2)
    ax.set_title("Valid Goal Fraction")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # ── (1,0)  Alice reward ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(alice["step"], alice["rew"], color=C["alice_rew"], lw=1.5)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Alice Mean Reward")
    ax.set_xlabel("Alice update #"); ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.3)

    # ── (1,1)  Alice loss ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(alice["step"], alice["loss"], color=C["alice_loss"], lw=1.5)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Alice PPO Loss")
    ax.set_xlabel("Alice update #"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ── (1,2)  Alice value ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(alice["step"], alice["val"], color=C["val"], lw=1.5)
    ax.set_title("Alice Value Estimate")
    ax.set_xlabel("Alice update #"); ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, out_path, "test2")


# ─── Test 3 figure ────────────────────────────────────────────────────────────

def plot_test3(data: dict, out_path: Path | None):
    bob  = data["bob"]
    itr  = data["iter"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Test 3 — PPO vs ABC Tug-of-War (Mini-ASP)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ── (0,0)  PPO loss vs ABC loss ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(bob["step"], bob["loss"], color=C["loss"], lw=1.5, label="PPO Loss")
    abc_nonzero = bob["abc"] != 0
    if abc_nonzero.any():
        ax.plot(bob["step"][abc_nonzero], bob["abc"][abc_nonzero],
                color=C["abc"], lw=1.5, label="ABC Loss", ls="--")
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Bob Loss: PPO vs ABC")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("Loss")
    ax.legend(fontsize=7)
    # Pass if ABC not completely 0 AND ratio reasonable
    abc_abs = np.abs(bob["abc"][abc_nonzero]) if abc_nonzero.any() else np.array([])
    ppo_abs = np.abs(bob["loss"][abc_nonzero]) if abc_nonzero.any() else np.array([])
    ratio_ok = (abc_abs < 10 * ppo_abs + 1e-6).all() if len(abc_abs) else True
    abc_fired = abc_nonzero.any()
    txt, col = _pass_fail(abc_fired and ratio_ok)
    if abc_fired and not ratio_ok:
        txt, col = "WARN", WARN_COLOR
    _add_badge(ax, txt, col)
    ax.grid(True, alpha=0.3)

    # ── (0,1)  Bob SR ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(bob["step"], bob["sr"], color=C["sr"], lw=1.8, marker="o", ms=3)
    ax.set_title("Bob Success Rate")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("SR")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # ── (0,2)  Bob Reward ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(bob["step"], bob["rew"], color=C["rew"], lw=1.8)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Bob Mean Reward")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # ── (1,0)  ABC buffer size ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(itr["iter"], itr["buf"], color=C["buf"], lw=2, marker=".", ms=4)
    ax.set_title("ABC Buffer Size")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Buffer entries")
    buf_growing = len(itr["buf"]) > 1 and itr["buf"][-1] > itr["buf"][0]
    txt, col = _pass_fail(buf_growing)
    _add_badge(ax, txt, col)
    ax.grid(True, alpha=0.3)

    # ── (1,1)  ABC warm flag ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(itr["iter"], itr["warm"], alpha=0.7,
                    color=C["abc"], step="mid", label="ABC warm")
    ax.set_title("ABC Warmup Events")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Warm (1=YES)")
    ax.set_ylim(-0.1, 1.2)
    txt, col = _pass_fail(itr["warm"].any())
    _add_badge(ax, txt, col)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── (1,2)  Goals valid per iter ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(itr["iter"], itr["valid"],   color=C["valid"],   label="Valid",   alpha=0.8)
    ax.bar(itr["iter"], itr["invalid"], bottom=itr["valid"],
           color=C["invalid"], label="Invalid", alpha=0.4)
    ax.set_title("Goals Valid / Invalid per Iter")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Goals")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── (2,0)  Bob value loss ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(bob["step"], bob["val"], color=C["val"], lw=1.5)
    ax.set_title("Bob Value Loss")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("Value loss")
    ax.grid(True, alpha=0.3)

    # ── (2,1)  ABC loss zoomed (only non-zero) ────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    if abc_nonzero.any():
        ax.scatter(bob["step"][abc_nonzero], bob["abc"][abc_nonzero],
                   color=C["abc"], s=25, zorder=3)
        ax.plot(bob["step"][abc_nonzero], bob["abc"][abc_nonzero],
                color=C["abc"], lw=1, alpha=0.6)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("ABC Loss (non-zero only)")
    ax.set_xlabel("Bob update #"); ax.set_ylabel("ABC loss")
    ax.grid(True, alpha=0.3)

    # ── (2,2)  Loss ratio heatmap ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    if abc_nonzero.any():
        ratio = np.abs(bob["abc"][abc_nonzero]) / (np.abs(bob["loss"][abc_nonzero]) + 1e-9)
        ax.plot(bob["step"][abc_nonzero], ratio, color=WARN_COLOR, lw=1.5)
        ax.axhline(10, color=FAIL_COLOR, ls="--", lw=1, label="|ABC/PPO|=10×")
        ax.set_title("|ABC Loss| / |PPO Loss| ratio")
        ax.set_ylabel("Ratio")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No ABC activations", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title("|ABC Loss| / |PPO Loss| ratio")
    ax.set_xlabel("Bob update #")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, out_path, "test3")


# ─── Summary figure ───────────────────────────────────────────────────────────

def plot_summary(datasets: dict[str, dict], out_path: Path | None):
    """One overlaid figure comparing SR and valid goals across all 3 tests."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Diagnostic Summary — all tests", fontsize=13, fontweight="bold")

    colors = {"test1": "#3498db", "test2": "#e74c3c", "test3": "#2ecc71"}
    labels = {"test1": "T1 Reward Pipeline",
              "test2": "T2 Alice Explore",
              "test3": "T3 PPO/ABC Balance"}

    # Panel 1 — SR
    ax = axes[0]
    for name, data in datasets.items():
        itr = data["iter"]
        if len(itr["iter"]):
            ax.plot(itr["iter"], itr["sr"],
                    color=colors[name], label=labels[name], lw=1.8)
    ax.set_title("Bob Success Rate (per iter)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("SR")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2 — Valid goals
    ax = axes[1]
    for name, data in datasets.items():
        itr = data["iter"]
        if len(itr["iter"]):
            ax.plot(itr["iter"], itr["valid"],
                    color=colors[name], label=labels[name], lw=1.8)
    ax.set_title("Valid Goals per Iter")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Valid goals")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — ABC buffer
    ax = axes[2]
    for name, data in datasets.items():
        itr = data["iter"]
        if len(itr["iter"]) and itr["buf"].max() > 0:
            ax.plot(itr["iter"], itr["buf"],
                    color=colors[name], label=labels[name], lw=1.8)
    ax.set_title("ABC Buffer Size per Iter")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Buffer entries")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, out_path, "summary")


# ─── Print diagnostic verdict ─────────────────────────────────────────────────

def print_verdict(datasets: dict[str, dict]):
    print("\n" + "="*60)
    print("  ASP DIAGNOSTIC VERDICT")
    print("="*60)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    d = datasets.get("test1")
    if d:
        bob = d["bob"]
        sr_pos = bool(np.any(bob["sr"] > 0))
        rew_pos = bool(np.any(bob["rew"] > 0))
        print("\n  Test 1 — PPO Reward Pipeline")
        _verdict_line("Bob SR > 0 within 50 iters", sr_pos)
        _verdict_line("Bob Rew > 0", rew_pos)
        if not sr_pos:
            print("    ⚠  Check DummyBobWrapper teleport and reward assignment logic.")
    else:
        print("\n  Test 1 — [no data]")

    # ── Test 2 ────────────────────────────────────────────────────────────────
    d = datasets.get("test2")
    if d:
        itr = d["iter"]
        max_valid = itr["valid"].max() if len(itr["valid"]) else 0
        late_valid = itr["valid"][len(itr["valid"])//2:].mean() if len(itr["valid"]) > 2 else 0
        print("\n  Test 2 — Alice Exploration Sandbox")
        _verdict_line("Any valid goals produced", max_valid > 0)
        _verdict_line(f"Late-run avg valid > 2  (got {late_valid:.1f})", late_valid > 2)
        if max_valid == 0:
            print("    ⚠  Alice never generates valid goals — check pos_threshold.")
    else:
        print("\n  Test 2 — [no data]")

    # ── Test 3 ────────────────────────────────────────────────────────────────
    d = datasets.get("test3")
    if d:
        bob  = d["bob"]
        itr  = d["iter"]
        abc_nonzero = bob["abc"] != 0
        abc_fired  = abc_nonzero.any()
        buf_grew   = len(itr["buf"]) > 1 and itr["buf"][-1] > itr["buf"][0]
        ratio_ok = True
        if abc_nonzero.any():
            ratio = np.abs(bob["abc"][abc_nonzero]) / (np.abs(bob["loss"][abc_nonzero]) + 1e-9)
            ratio_ok = bool((ratio < 10).all())
        print("\n  Test 3 — PPO vs ABC Tug-of-War")
        _verdict_line("ABC warmup fired at least once", abc_fired)
        _verdict_line("ABC buffer is growing", buf_grew)
        _verdict_line("|ABC/PPO| ratio < 10× when active", ratio_ok)
        if not ratio_ok:
            print("    ⚠  ABC loss dominates PPO — lower abc_coef in YAML.")
        if not abc_fired:
            print("    ⚠  ABC never activated — Alice never crossed abc_warmup_threshold.")
    else:
        print("\n  Test 3 — [no data]")

    print("\n" + "="*60 + "\n")
    return collect_verdicts(datasets)


def collect_verdicts(datasets: dict[str, dict]) -> list[tuple[str, str, bool]]:
    """Return list of (test, label, passed) for CSV export."""
    out = []
    # Test 1
    d = datasets.get("test1")
    if d:
        bob = d["bob"]
        out.append(("test1", "Bob SR > 0 within 50 iters", bool(np.any(bob["sr"] > 0))))
        out.append(("test1", "Bob Rew > 0", bool(np.any(bob["rew"] > 0))))
    # Test 2
    d = datasets.get("test2")
    if d:
        itr = d["iter"]
        max_valid = itr["valid"].max() if len(itr["valid"]) else 0
        late_valid = itr["valid"][len(itr["valid"])//2:].mean() if len(itr["valid"]) > 2 else 0
        out.append(("test2", "Any valid goals produced", max_valid > 0))
        out.append(("test2", f"Late-run avg valid > 2", late_valid > 2))
    # Test 3
    d = datasets.get("test3")
    if d:
        bob = d["bob"]; itr = d["iter"]
        abc_nonzero = bob["abc"] != 0
        abc_fired = abc_nonzero.any()
        buf_grew = len(itr["buf"]) > 1 and itr["buf"][-1] > itr["buf"][0]
        ratio_ok = True
        if abc_fired:
            ratio = np.abs(bob["abc"][abc_nonzero]) / (np.abs(bob["loss"][abc_nonzero]) + 1e-9)
            ratio_ok = bool((ratio < 10).all())
        out.append(("test3", "ABC warmup fired at least once", abc_fired))
        out.append(("test3", "ABC buffer is growing", buf_grew))
        out.append(("test3", "|ABC/PPO| ratio < 10x", ratio_ok))
    return out


def _verdict_line(label: str, passed: bool):
    icon = "✓" if passed else "✗"
    col_start = "\033[92m" if passed else "\033[91m"
    col_end   = "\033[0m"
    print(f"    {col_start}{icon}{col_end}  {label}")


# ─── CSV Export ───────────────────────────────────────────────────────────────

def export_csv(datasets: dict[str, dict], out_path: Path):
    import csv
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Header for iteration updates
        writer.writerow(["# ITERATION UPDATES"])
        writer.writerow(["test", "iter", "sr", "valid", "invalid", "succ", "fail", "buf", "warm"])
        
        for name in ["test1", "test2", "test3"]:
            if name in datasets:
                itr = datasets[name]["iter"]
                for i in range(len(itr["iter"])):
                    writer.writerow([
                        name,
                        itr["iter"][i],
                        f"{itr['sr'][i]:.4f}",
                        itr["valid"][i],
                        itr["invalid"][i],
                        itr["succ"][i],
                        itr["fail"][i],
                        itr["buf"][i],
                        itr["warm"][i]
                    ])
        
        writer.writerow([])
        # Header for Verdicts
        writer.writerow(["# DIAGNOSTIC VERDICTS"])
        writer.writerow(["test", "condition", "passed"])
        verdicts = collect_verdicts(datasets)
        for t, lbl, p in verdicts:
            writer.writerow([t, lbl, "PASS" if p else "FAIL"])
    
    print(f"  CSV exported → {out_path}")


# ─── Output helpers ───────────────────────────────────────────────────────────

def _save_or_show(fig, out_path: Path | None, tag: str):
    if out_path:
        stem   = out_path.stem
        suffix = out_path.suffix or ".png"
        dest   = out_path.parent / f"{stem}_{tag}{suffix}"
        fig.savefig(dest, dpi=130, bbox_inches="tight")
        print(f"  Saved → {dest}")
    else:
        plt.show()
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────

DEFAULT_DIR = Path(__file__).parent

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test1", type=Path, default=DEFAULT_DIR / "test1.txt",
                    help="Path to test-1 log (default: logs/diagnostics/test1.txt)")
    ap.add_argument("--test2", type=Path, default=DEFAULT_DIR / "test2.txt",
                    help="Path to test-2 log (default: logs/diagnostics/test2.txt)")
    ap.add_argument("--test3", type=Path, default=DEFAULT_DIR / "test3.txt",
                    help="Path to test-3 log (default: logs/diagnostics/test3.txt)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output file stem (e.g. results.png). "
                         "If omitted, displays interactively. "
                         "Each test gets its own file: results_test1.png etc.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Path to export consolidated CSV results.")
    args = ap.parse_args()

    datasets: dict[str, dict] = {}
    for tag, path in [("test1", args.test1), ("test2", args.test2), ("test3", args.test3)]:
        if path.exists():
            print(f"  Parsing {path} ...")
            datasets[tag] = parse_log(path)
        else:
            print(f"  [SKIP] {path} not found", file=sys.stderr)

    if not datasets:
        print("No log files found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    if "test1" in datasets:
        plot_test1(datasets["test1"], args.out)
    if "test2" in datasets:
        plot_test2(datasets["test2"], args.out)
    if "test3" in datasets:
        plot_test3(datasets["test3"], args.out)

    if len(datasets) > 1:
        plot_summary(datasets, args.out)

    verdicts = print_verdict(datasets)
    
    if args.csv:
        export_csv(datasets, args.csv)


if __name__ == "__main__":
    main()
