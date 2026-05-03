#!/usr/bin/env python3
"""
Benchmark: Current vs Optimised hot-path operations.

Simulates the key inefficiencies from the efficiency analysis WITHOUT needing
Isaac Lab or the full training pipeline.  Run on a GPU node:

    python -m asyncDualPlayPPO.tests.bench_inefficiencies          # inside container
    python asyncDualPlayPPO/tests/bench_inefficiencies.py          # or directly

Each benchmark reports wall-clock time AND peak GPU memory delta so you can
judge the ROI of each proposed fix before touching production code.
"""

import copy
import gc
import os
import sys
import time
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config — mirrors the real pipeline's sizes
# ---------------------------------------------------------------------------
NUM_ENVS = 256
ALICE_TIMESTEPS = 100   # from AsyncDualPlay.yaml
BOB_TIMESTEPS = 200     # from AsyncDualPlay.yaml
ROLLOUT_LENGTH = ALICE_TIMESTEPS + BOB_TIMESTEPS  # 300 (NOT nsteps=256, which is storage buffer size)
ALICE_OBS_DIM = 35
BOB_OBS_DIM = 51
ACTION_DIM = 6       # multi-categorical bins
NUM_BINS = 11
PI_EMB_DIM = 512
LSTM_HIDDEN = 256
HIST_POOL_SIZE = 5
ABC_CAPACITY = 50_000
NSTEPS = 256          # storage size per agent
NUM_ITERS = 5         # how many iterations to average over

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def _reset_memory():
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _peak_mb():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

@contextmanager
def measure(label: str):
    """Context manager that prints wall time + peak GPU memory delta."""
    _reset_memory()
    _sync()
    mem_before = _peak_mb()
    t0 = time.perf_counter()
    yield
    _sync()
    t1 = time.perf_counter()
    mem_after = _peak_mb()
    dt = (t1 - t0) * 1000
    dm = mem_after - mem_before
    print(f"  {label:.<55s} {dt:8.1f} ms  |  {dm:+8.2f} MB GPU")


# ===================================================================
# Dummy ActorCritic (matches real model's parameter count roughly)
# ===================================================================

class DummyActorCritic(nn.Module):
    """Lightweight stand-in for the real ActorCritic.

    Has the same approximate number of parameters and the same
    forward-pass shape so timing is representative.
    """
    def __init__(self, obs_dim, action_dim, num_bins):
        super().__init__()
        self.lstm_hidden_size = LSTM_HIDDEN
        self.use_lstm = True
        # PI encoder (shared MLP)
        self.pi_enc = nn.Sequential(
            nn.Linear(14, PI_EMB_DIM), nn.ReLU(),
            nn.Linear(PI_EMB_DIM, PI_EMB_DIM), nn.ReLU(),
        )
        # Actor trunk
        self.trunk = nn.Sequential(
            nn.Linear(7 + PI_EMB_DIM, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.lstm = nn.LSTMCell(128, LSTM_HIDDEN)
        self.head = nn.Linear(LSTM_HIDDEN, action_dim * num_bins)
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def act_with_hidden(self, obs, states, hidden):
        batch = obs.shape[0]
        robot = obs[:, :7]
        objs = obs[:, 7:].reshape(batch, -1, 14)
        enc = self.pi_enc(objs).max(dim=1).values
        feat = self.trunk(torch.cat([robot, enc], dim=-1))
        if hidden is None:
            h = torch.zeros(batch, LSTM_HIDDEN, device=obs.device)
            c = torch.zeros(batch, LSTM_HIDDEN, device=obs.device)
        else:
            h, c = hidden
        h, c = self.lstm(feat, (h, c))
        logits = self.head(h)
        value = self.critic(obs)
        acts = torch.randint(0, NUM_BINS, (batch, ACTION_DIM), device=obs.device).float()
        lp = torch.zeros(batch, device=obs.device)
        mu = logits.view(batch, ACTION_DIM, NUM_BINS).sum(-1)
        sigma = torch.zeros_like(acts)
        return acts, lp, value, mu, sigma, (h.detach(), c.detach())

    def evaluate(self, obs, states, actions):
        batch = obs.shape[0]
        robot = obs[:, :7]
        objs = obs[:, 7:].reshape(batch, -1, 14)
        enc = self.pi_enc(objs).max(dim=1).values
        feat = self.trunk(torch.cat([robot, enc], dim=-1))
        h = torch.zeros(batch, LSTM_HIDDEN, device=obs.device)
        c = torch.zeros(batch, LSTM_HIDDEN, device=obs.device)
        h, c = self.lstm(feat, (h, c))
        logits = self.head(h)
        value = self.critic(obs)
        lp = torch.zeros(batch, device=obs.device)
        ent = torch.zeros(batch, device=obs.device)
        mu = logits.view(batch, ACTION_DIM, NUM_BINS).sum(-1)
        sigma = torch.zeros_like(actions)
        return lp, ent, value, mu, sigma


# ===================================================================
# BENCHMARK 1: Per-step tensor allocation (Issue #1)
# ===================================================================

def bench_alloc_current():
    """Current: allocate tensors via torch.zeros each step."""
    for _ in range(ROLLOUT_LENGTH):
        n_alice = NUM_ENVS // 2
        n_bob = NUM_ENVS - n_alice
        a_acts = torch.zeros((n_alice, ACTION_DIM), device=DEVICE)
        a_lp   = torch.zeros(n_alice, device=DEVICE)
        a_val  = torch.zeros(n_alice, 1, device=DEVICE)
        a_mu   = torch.zeros_like(a_acts)
        a_sig  = torch.zeros_like(a_acts)

        b_acts = torch.zeros((n_bob, ACTION_DIM), device=DEVICE)
        b_lp   = torch.zeros(n_bob, device=DEVICE)
        b_val  = torch.zeros(n_bob, 1, device=DEVICE)
        b_mu   = torch.zeros_like(b_acts)
        b_sig  = torch.zeros_like(b_acts)

        a_pol  = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
        b_pol  = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)

        a_lp_f = torch.zeros(NUM_ENVS, device=DEVICE)
        a_vf   = torch.zeros(NUM_ENVS, 1, device=DEVICE)
        a_mu_f = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
        a_sf   = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)

        b_lp_f = torch.zeros(NUM_ENVS, 1, device=DEVICE)
        b_vf   = torch.zeros(NUM_ENVS, 1, device=DEVICE)
        b_mu_f = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
        b_sf   = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)

def bench_alloc_optimized():
    """Optimized: pre-allocate once, zero_ each step."""
    n_alice = NUM_ENVS // 2
    n_bob = NUM_ENVS - n_alice
    # Pre-allocate
    a_acts = torch.zeros((n_alice, ACTION_DIM), device=DEVICE)
    a_lp   = torch.zeros(n_alice, device=DEVICE)
    a_val  = torch.zeros(n_alice, 1, device=DEVICE)
    a_mu   = torch.zeros((n_alice, ACTION_DIM), device=DEVICE)
    a_sig  = torch.zeros((n_alice, ACTION_DIM), device=DEVICE)
    b_acts = torch.zeros((n_bob, ACTION_DIM), device=DEVICE)
    b_lp   = torch.zeros(n_bob, device=DEVICE)
    b_val  = torch.zeros(n_bob, 1, device=DEVICE)
    b_mu   = torch.zeros((n_bob, ACTION_DIM), device=DEVICE)
    b_sig  = torch.zeros((n_bob, ACTION_DIM), device=DEVICE)
    a_pol  = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
    b_pol  = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
    a_lp_f = torch.zeros(NUM_ENVS, device=DEVICE)
    a_vf   = torch.zeros(NUM_ENVS, 1, device=DEVICE)
    a_mu_f = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
    a_sf   = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
    b_lp_f = torch.zeros(NUM_ENVS, 1, device=DEVICE)
    b_vf   = torch.zeros(NUM_ENVS, 1, device=DEVICE)
    b_mu_f = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)
    b_sf   = torch.zeros((NUM_ENVS, ACTION_DIM), device=DEVICE)

    for _ in range(ROLLOUT_LENGTH):
        a_acts.zero_(); a_lp.zero_(); a_val.zero_(); a_mu.zero_(); a_sig.zero_()
        b_acts.zero_(); b_lp.zero_(); b_val.zero_(); b_mu.zero_(); b_sig.zero_()
        a_pol.zero_(); b_pol.zero_()
        a_lp_f.zero_(); a_vf.zero_(); a_mu_f.zero_(); a_sf.zero_()
        b_lp_f.zero_(); b_vf.zero_(); b_mu_f.zero_(); b_sf.zero_()


# ===================================================================
# BENCHMARK 2: Historical policy deep-copy (Issue #2)
# ===================================================================

def bench_histpool_current():
    """Current: deep-copy the entire model each iteration."""
    model = DummyActorCritic(BOB_OBS_DIM, ACTION_DIM, NUM_BINS).to(DEVICE)
    pool = []
    for i in range(HIST_POOL_SIZE):
        pool.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

    for _ in range(NUM_ITERS):
        snapshot = pool[0]
        hist_copy = copy.deepcopy(model)
        hist_copy.load_state_dict({k: v.to(DEVICE) for k, v in snapshot.items()})
        hist_copy.eval()

def bench_histpool_optimized():
    """Optimized: reuse a single persistent clone."""
    model = DummyActorCritic(BOB_OBS_DIM, ACTION_DIM, NUM_BINS).to(DEVICE)
    pool = []
    for i in range(HIST_POOL_SIZE):
        pool.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

    hist_clone = copy.deepcopy(model)  # one-time cost
    hist_clone.eval()

    for _ in range(NUM_ITERS):
        snapshot = pool[0]
        hist_clone.load_state_dict({k: v.to(DEVICE) for k, v in snapshot.items()})


# ===================================================================
# BENCHMARK 3: RolloutStorage states waste (Issue #3)
# ===================================================================

def bench_storage_current():
    """Current: allocates full states tensor even when unused."""
    obs_shape = (BOB_OBS_DIM,)
    states_shape = (BOB_OBS_DIM,)  # same as obs when asymmetric=False
    actions_shape = (ACTION_DIM,)
    n_trans = NSTEPS + BOB_TIMESTEPS + 10  # oversized like the real code

    observations = torch.zeros(n_trans, NUM_ENVS, *obs_shape, device=DEVICE)
    states       = torch.zeros(n_trans, NUM_ENVS, *states_shape, device=DEVICE)  # WASTE
    rewards      = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    actions      = torch.zeros(n_trans, NUM_ENVS, *actions_shape, device=DEVICE)
    dones        = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE).byte()
    masks        = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE).byte()
    actions_lp   = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    values       = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    returns      = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    advantages   = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    mu           = torch.zeros(n_trans, NUM_ENVS, *actions_shape, device=DEVICE)
    sigma        = torch.zeros(n_trans, NUM_ENVS, *actions_shape, device=DEVICE)

def bench_storage_optimized():
    """Optimized: skip states allocation when asymmetric=False."""
    obs_shape = (BOB_OBS_DIM,)
    actions_shape = (ACTION_DIM,)
    n_trans = NSTEPS + BOB_TIMESTEPS + 10

    observations = torch.zeros(n_trans, NUM_ENVS, *obs_shape, device=DEVICE)
    # states = None  ← skipped!
    rewards      = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    actions      = torch.zeros(n_trans, NUM_ENVS, *actions_shape, device=DEVICE)
    dones        = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE).byte()
    masks        = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE).byte()
    actions_lp   = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    values       = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    returns      = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    advantages   = torch.zeros(n_trans, NUM_ENVS, 1, device=DEVICE)
    mu           = torch.zeros(n_trans, NUM_ENVS, *actions_shape, device=DEVICE)
    sigma        = torch.zeros(n_trans, NUM_ENVS, *actions_shape, device=DEVICE)


# ===================================================================
# BENCHMARK 4: ABC buffer unused fields (Issue #4)
# ===================================================================

def bench_abc_buffer_current():
    """Current: allocates all 11 tensor fields."""
    obs_shape = (BOB_OBS_DIM,)
    actions_shape = (ACTION_DIM,)
    cap = ABC_CAPACITY
    observations = torch.zeros(cap, *obs_shape, device=DEVICE)
    states       = torch.zeros(cap, *obs_shape, device=DEVICE)
    actions      = torch.zeros(cap, *actions_shape, device=DEVICE)
    rewards      = torch.zeros(cap, 1, device=DEVICE)
    dones        = torch.zeros(cap, 1, device=DEVICE).byte()
    values       = torch.zeros(cap, 1, device=DEVICE)
    actions_lp   = torch.zeros(cap, 1, device=DEVICE)
    mu           = torch.zeros(cap, *actions_shape, device=DEVICE)
    sigma        = torch.zeros(cap, *actions_shape, device=DEVICE)
    returns      = torch.zeros(cap, 1, device=DEVICE)
    advantages   = torch.zeros(cap, 1, device=DEVICE)

def bench_abc_buffer_optimized():
    """Optimized: only allocate what's actually used (traj_store deque)."""
    # The real pipeline only uses _traj_store (a deque of dicts with obs, acts, old_lp)
    # No pre-allocated tensor fields needed at all.
    _traj_store = deque(maxlen=500)


# ===================================================================
# BENCHMARK 5: ABC per-env Python loop vs batched (Issue #5)
# ===================================================================

def bench_abc_loop_current():
    """Current: one forward pass per failed env."""
    model = DummyActorCritic(BOB_OBS_DIM, ACTION_DIM, NUM_BINS).to(DEVICE)
    model.eval()

    # Simulate ~170 failed envs, each with trajectory length ~100
    n_failed = 170
    traj_len = 100

    with torch.no_grad():
        for i in range(n_failed):
            obs = torch.randn(traj_len, BOB_OBS_DIM, device=DEVICE)
            acts = torch.randint(0, NUM_BINS, (traj_len, ACTION_DIM), device=DEVICE).float()
            lp, _, _, _, _ = model.evaluate(obs, None, acts)

def bench_abc_loop_optimized():
    """Optimized: single batched forward pass for all failed envs."""
    model = DummyActorCritic(BOB_OBS_DIM, ACTION_DIM, NUM_BINS).to(DEVICE)
    model.eval()

    n_failed = 170
    traj_len = 100

    with torch.no_grad():
        # Pad all trajectories and batch
        all_obs = torch.randn(n_failed * traj_len, BOB_OBS_DIM, device=DEVICE)
        all_acts = torch.randint(0, NUM_BINS, (n_failed * traj_len, ACTION_DIM), device=DEVICE).float()
        lp, _, _, _, _ = model.evaluate(all_obs, None, all_acts)


# ===================================================================
# BENCHMARK 6: searchsorted vs lookup table (Issue #6)
# ===================================================================

def bench_searchsorted_current():
    """Current: torch.searchsorted 4x per step."""
    alice_indices = torch.arange(0, NUM_ENVS, 2, device=DEVICE)  # ~128 envs
    bob_indices = torch.arange(1, NUM_ENVS, 2, device=DEVICE)

    for _ in range(ROLLOUT_LENGTH):
        # Simulate the 4 searchsorted calls per step
        perm = torch.randperm(len(alice_indices), device=DEVICE)
        n_hist = max(1, len(alice_indices) // 5)
        hist_ids = alice_indices[perm[:n_hist]]
        curr_ids = alice_indices[perm[n_hist:]]
        curr_local = torch.searchsorted(alice_indices, curr_ids)
        hist_local = torch.searchsorted(alice_indices, hist_ids)

        perm_b = torch.randperm(len(bob_indices), device=DEVICE)
        n_hist_b = max(1, len(bob_indices) // 5)
        hist_bids = bob_indices[perm_b[:n_hist_b]]
        curr_bids = bob_indices[perm_b[n_hist_b:]]
        curr_bloc = torch.searchsorted(bob_indices, curr_bids)
        hist_bloc = torch.searchsorted(bob_indices, hist_bids)

def bench_searchsorted_optimized():
    """Optimized: pre-computed lookup table, gather instead of searchsorted."""
    alice_indices = torch.arange(0, NUM_ENVS, 2, device=DEVICE)
    bob_indices = torch.arange(1, NUM_ENVS, 2, device=DEVICE)

    # Pre-compute reverse index (once per iteration, outside the step loop)
    alice_lut = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    alice_lut[alice_indices] = torch.arange(len(alice_indices), device=DEVICE)
    bob_lut = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    bob_lut[bob_indices] = torch.arange(len(bob_indices), device=DEVICE)

    for _ in range(ROLLOUT_LENGTH):
        perm = torch.randperm(len(alice_indices), device=DEVICE)
        n_hist = max(1, len(alice_indices) // 5)
        hist_ids = alice_indices[perm[:n_hist]]
        curr_ids = alice_indices[perm[n_hist:]]
        curr_local = alice_lut[curr_ids]  # O(1) gather
        hist_local = alice_lut[hist_ids]

        perm_b = torch.randperm(len(bob_indices), device=DEVICE)
        n_hist_b = max(1, len(bob_indices) // 5)
        hist_bids = bob_indices[perm_b[:n_hist_b]]
        curr_bids = bob_indices[perm_b[n_hist_b:]]
        curr_bloc = bob_lut[curr_bids]
        hist_bloc = bob_lut[hist_bids]


# ===================================================================
# Runner
# ===================================================================

def run_pair(name, fn_current, fn_optimized):
    """Run a benchmark pair and print results."""
    print(f"\n{'='*72}")
    print(f"  BENCHMARK: {name}")
    print(f"{'='*72}")

    # Warmup
    fn_current()
    fn_optimized()
    _sync()

    with measure("Current"):
        fn_current()
    with measure("Optimized"):
        fn_optimized()


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"\nConfig: {NUM_ENVS} envs, rollout_length={ROLLOUT_LENGTH}, "
          f"alice_obs={ALICE_OBS_DIM}D, bob_obs={BOB_OBS_DIM}D, action={ACTION_DIM}D")
    print(f"Each benchmark runs ONCE (no averaging) to show realistic single-iteration cost.")

    run_pair(
        "#1 — Per-Step Tensor Allocation (torch.zeros vs .zero_())",
        bench_alloc_current, bench_alloc_optimized,
    )
    run_pair(
        "#2 — Historical Pool (deepcopy vs reuse clone)",
        bench_histpool_current, bench_histpool_optimized,
    )
    run_pair(
        "#3 — RolloutStorage states (allocated vs skipped)",
        bench_storage_current, bench_storage_optimized,
    )
    run_pair(
        "#4 — ABC Buffer (11 fields vs deque only)",
        bench_abc_buffer_current, bench_abc_buffer_optimized,
    )
    run_pair(
        "#5 — ABC Demo Loop (170×1 vs 1×17000 forward pass)",
        bench_abc_loop_current, bench_abc_loop_optimized,
    )
    run_pair(
        "#6 — searchsorted vs lookup table (4× per step)",
        bench_searchsorted_current, bench_searchsorted_optimized,
    )

    # --- COMBINED: estimate total savings ---
    print(f"\n{'='*72}")
    print(f"  COMBINED ESTIMATE (all 6 fixes together)")
    print(f"{'='*72}")

    def run_all_current():
        bench_alloc_current()
        bench_histpool_current()
        bench_storage_current()
        bench_abc_buffer_current()
        bench_abc_loop_current()
        bench_searchsorted_current()

    def run_all_optimized():
        bench_alloc_optimized()
        bench_histpool_optimized()
        bench_storage_optimized()
        bench_abc_buffer_optimized()
        bench_abc_loop_optimized()
        bench_searchsorted_optimized()

    # Warmup
    run_all_current()
    run_all_optimized()
    _sync()

    with measure("All Current (combined)"):
        run_all_current()
    with measure("All Optimized (combined)"):
        run_all_optimized()

    print(f"\n{'='*72}")
    print("  Done. Compare 'Current' vs 'Optimized' rows above.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
