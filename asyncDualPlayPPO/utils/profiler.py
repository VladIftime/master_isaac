"""
TrainingProfiler — lightweight CUDA-synchronised wall-clock profiler.

Usage in train.py:
    profiler = TrainingProfiler(enabled=args.profile, device=env.device)

    # inside the rollout timestep loop:
    with profiler.section("env_step"):
        obs_full, rewards, dones, truncated, extras = env.step(env_full)

    # after each iteration:
    profiler.end_iteration(bob_updates)

    # final summary:
    profiler.print_summary()

Sections timed:
    obs_compute     — observation_manager.compute() + cat
    alice_act       — Alice act_with_hidden (all envs in phase)
    bob_act         — Bob act_with_hidden
    env_step        — env.step()
    alice_store     — alice storage.add_transitions()
    bob_store       — bob storage.add_transitions()
    reward_backfill — Alice outcome/reward backfill into storage
    abc_buffer      — construct_bob_observation + evaluate + add_trajectory
    alice_update    — compute_returns + PPO.update() for Alice
    bob_update      — compute_returns + PPOABC.update() for Bob
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Optional

import torch


class TrainingProfiler:
    def __init__(self, enabled: bool = False, device: str = "cpu"):
        self.enabled = enabled
        self.use_cuda = enabled and torch.cuda.is_available() and "cuda" in str(device)

        # Per-iteration accumulators (reset each iteration)
        self._iter_totals: dict[str, float] = defaultdict(float)
        self._iter_calls: dict[str, int] = defaultdict(int)

        # All-iterations history: section -> list of per-iter total seconds
        self._history: dict[str, list[float]] = defaultdict(list)

        self._iter_wall_start: float = 0.0
        self._iter_wall: list[float] = []
        self._pending: dict[str, float] = {}

    # ------------------------------------------------------------------
    def start_iteration(self):
        if not self.enabled:
            return
        self._iter_totals.clear()
        self._iter_calls.clear()
        if self.use_cuda:
            torch.cuda.synchronize()
        self._iter_wall_start = time.perf_counter()

    def end_iteration(self, iteration: int):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - self._iter_wall_start
        self._iter_wall.append(wall)

        for k, v in self._iter_totals.items():
            self._history[k].append(v)

        self._print_iter_table(iteration, wall)

    # ------------------------------------------------------------------
    # Explicit start/stop API for multi-line blocks that can't be
    # wrapped in a `with` statement without reindenting code.
    def mark_start(self, name: str):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        self._pending[name] = time.perf_counter()

    def mark_stop(self, name: str):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = self._pending.pop(name, None)
        if t0 is not None:
            elapsed = time.perf_counter() - t0
            self._iter_totals[name] += elapsed
            self._iter_calls[name] += 1

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            self._iter_totals[name] += elapsed
            self._iter_calls[name] += 1

    # ------------------------------------------------------------------
    def _print_iter_table(self, iteration: int, wall: float):
        # (display_label, lookup_key) — indented labels are sub-sections of env_step
        _SECTIONS = [
            ("obs_compute",       "obs_compute"),
            ("alice_act",         "alice_act"),
            ("bob_act",           "bob_act"),
            ("env_step",          "env_step"),
            ("  isaac_step",      "isaac_step"),
            ("  wrapper_reset",   "wrapper_reset"),
            ("  wrapper_rewards", "wrapper_rewards"),
            ("alice_store",       "alice_store"),
            ("bob_store",         "bob_store"),
            ("reward_backfill",   "reward_backfill"),
            ("abc_buffer",        "abc_buffer"),
            ("alice_update",      "alice_update"),
            ("bob_update",        "bob_update"),
        ]
        print(f"\n{'='*65}")
        print(f"  PROFILER  iter={iteration}   wall={wall:.1f}s")
        print(f"  {'Section':<20} {'Total(s)':>9}  {'%wall':>7}  {'calls':>6}  {'ms/call':>8}")
        print(f"  {'-'*62}")

        accounted = 0.0
        for label, key in _SECTIONS:
            t = self._iter_totals.get(key, 0.0)
            n = self._iter_calls.get(key, 0)
            pct = 100.0 * t / wall if wall > 0 else 0.0
            ms_per = 1000.0 * t / n if n > 0 else 0.0
            if key not in ("isaac_step", "wrapper_reset", "wrapper_rewards"):
                accounted += t
            print(f"  {label:<20} {t:>9.3f}  {pct:>6.1f}%  {n:>6}  {ms_per:>8.2f}")

        unaccounted = wall - accounted
        pct_un = 100.0 * unaccounted / wall if wall > 0 else 0.0
        print(f"  {'(other/overhead)':<20} {unaccounted:>9.3f}  {pct_un:>6.1f}%")
        print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    def get_section_frac(self, section: str, relative_to: str = "env_step") -> float:
        """Return this iteration's time fraction: section / relative_to.

        Returns 0.0 when the profiler is disabled or either section is missing.
        Useful for asserting that curobo_ik < 10% of env_step each iteration.
        """
        if not self.enabled:
            return 0.0
        denom = self._iter_totals.get(relative_to, 0.0)
        if denom <= 0.0:
            return 0.0
        return self._iter_totals.get(section, 0.0) / denom

    # ------------------------------------------------------------------
    def print_summary(self):
        if not self.enabled or not self._iter_wall:
            return

        import statistics

        n_iters = len(self._iter_wall)
        avg_wall = statistics.mean(self._iter_wall)

        _SECTIONS = [
            ("obs_compute",       "obs_compute"),
            ("alice_act",         "alice_act"),
            ("bob_act",           "bob_act"),
            ("env_step",          "env_step"),
            ("  isaac_step",      "isaac_step"),
            ("  wrapper_reset",   "wrapper_reset"),
            ("  wrapper_rewards", "wrapper_rewards"),
            ("alice_store",       "alice_store"),
            ("bob_store",         "bob_store"),
            ("reward_backfill",   "reward_backfill"),
            ("abc_buffer",        "abc_buffer"),
            ("alice_update",      "alice_update"),
            ("bob_update",        "bob_update"),
        ]

        print(f"\n{'='*70}")
        print(f"  PROFILER SUMMARY  ({n_iters} iterations, avg wall={avg_wall:.1f}s)")
        print(f"  {'Section':<22} {'avg(s)':>8}  {'%wall':>7}  {'stddev(s)':>10}")
        print(f"  {'-'*65}")

        for label, key in _SECTIONS:
            hist = self._history.get(key, [])
            if not hist:
                continue
            avg_t = statistics.mean(hist)
            std_t = statistics.stdev(hist) if len(hist) > 1 else 0.0
            pct = 100.0 * avg_t / avg_wall if avg_wall > 0 else 0.0
            print(f"  {label:<22} {avg_t:>8.3f}  {pct:>6.1f}%  {std_t:>10.3f}")

        print(f"{'='*70}\n")
        print("[PROFILER] Top recommendation: the largest % section is your bottleneck.")
        print("[PROFILER] env_step >70% → physics-bound; alice/bob_act >10% → network-bound;")
        print("[PROFILER] abc_buffer >5% → ABC overhead; *_store >5% → GPU memory bandwidth.")
