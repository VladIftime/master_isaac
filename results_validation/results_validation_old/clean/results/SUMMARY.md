# 26.06.26 Validation Results — Unified Summary

**Protocol**: Isaac Lab, 30 T-block scenes (tests 1-30 from `validation_configs.py`), thesis gate: `pos_err < 0.05 m AND rot_err < 0.2 rad`.  
**Key comparison**: A_simp (26.06.20 ckpt) vs B_curr (26.06.28 ckpt) — identical scenes, identical protocol.

---

## Definitive Head-to-Head (Isaac, 30 T-block, thesis gate)

| Model | Scene SR | Pos-only | Pos+rot | PosErr | RotErr | Avg Pushes |
|-------|----------|----------|---------|--------|--------|-----------|
| **A_simp** (no curriculum) | **80.0%** | 100% | 70% | 0.032 m | 0.568 rad | 23.5 |
| **B_curr** (P82 curriculum) | **76.7%** | 100% | 65% | 0.023 m | 0.663 rad | 26.7 |
| G_tasp_dpose (TASP T-block) | 16.7% | 30% | 10% | 0.158 m | 1.457 rad | 12.2 |
| H_tasp_disc (TASP disc) | 10.0% | 30% | 0% | 0.186 m | 1.260 rad | 12.6 |
| E_asp_dpose (ASP T-block) | 6.7% | 20% | 0% | 0.143 m | 1.612 rad | 12.8 |
| F_asp_disc (ASP disc) | 6.7% | 20% | 0% | 0.197 m | 1.576 rad | 11.9 |

### Key findings

1. **The single-agent model (A_simp) beats the curriculum model (B_curr) by 3.3pp** on identical scenes (80.0% vs 76.7%). Both achieve 100% pos-only SR. The curriculum's forced staging (pos → pos+rot) improves position precision (0.023m vs 0.032m PosErr) but slightly degrades rotation (0.663 vs 0.568 rad RotErr), yielding no net advantage.

2. **ASP collapses across all variants.** The best ASP model (G_tasp_dpose) reaches only 16.7% — a 4.8× gap from single-agent. Time-based Alice (G/H) outperforms outcome-based Alice (E/F), and T-block outperforms disc, but all ASP variants are severely below the no-ASP baseline.

3. **Position-only is solved (100% for A and B).** The combined pos+rot gate is the bottleneck — capping at 70% (A) / 65% (B). Rotation is the unsolved dimension across all models.

4. **The P82 fix turned the curriculum from broken to functional.** B_curr's trigger now activates (gate on episodic position-SR), producing a competent model. But it doesn't beat the simpler no-curriculum baseline — the simplest model still wins.

---

## Older Validations (different configs)

### 26.06.20 — Disc test set (legacy config, 30 scenes: 10 disc + 10 pos_only + 10 pos_rot)

| Model | SR | PosErr | RotErr | Notes |
|-------|----|--------|--------|-------|
| A_simp | 80.0% | 0.043 m | 0.312 rad | 100% disc, 80% pos-only, 60% pos+rot |
| E_asp_dpose | 0% | — | — | Near-zero; early checkpoint (it 2600) |
| F_asp_disc | 0% | — | — | Near-zero; early checkpoint (it 2400) |
| G_tasp_dpose | 0% | — | — | Near-zero; early checkpoint (it 1200) |
| H_tasp_disc | 0% | — | — | Near-zero; early checkpoint (it 1200) |

> ASP 26.06.20 models were significantly undertrained vs 26.06.26 checkpoints.

### 26.06.12 — Old 20 T-block scenes (no R_* rotation scenes, no disc)

| Model | SR | PosErr | RotErr |
|-------|----|--------|--------|
| A_simp | 55.0% | 0.138 m | 1.075 rad |
| B_curr | 35.0% | 0.124 m | 1.450 rad |
| C_asp | — | — | — |

> Old checkpoint, pre-P82 fixes. B_curr trigger was mis-specified — curriculum never activated.

---

## Gym-pusht Comparison (controlled CPU testbed, coverage gate)

See `gympusht/summary.md` — 9 models compared. Notable: E_asp_dpose and G_tasp_dpose achieved the best position control (0.097m and 0.095m PosErr) in the CPU testbed despite 0% SR under the coverage gate.

---

## Model Reference

| Model | Type | Architecture | Object |
|-------|------|-------------|--------|
| **A_simp** | Single-agent PBRS | PPO | T-block |
| **B_curr** | PBRS + forced curriculum | PPO | T-block |
| C_asp | PBRS + ASP (Alice/Bob) | PPOABC + GoalEncoder | T-block |
| E_asp_dpose | ASP + SE(2) d_pose | PPOABC + GoalEncoder | T-block |
| F_asp_disc | ASP + d_pose | PPOABC + GoalEncoder | Disc |
| G_tasp_dpose | Time-based ASP + d_pose | PPOABC + GoalEncoder | T-block |
| H_tasp_disc | Time-based ASP + d_pose | PPOABC + GoalEncoder | Disc |

---

## Output Structure

```
validation_results_260626/
  SUMMARY.md           — This file
  isaac/               — Head-to-head CSVs + comparison plots + per_test_comparison.txt + summary.md
  gympusht/            — Gym-pusht CSVs + comparison plots + summary.md
  legacy/
    26.06.12/          — Old 20-test CSVs (A, B, C)
    26.06.20/          — Old disc-test-set CSVs (A, E, F, G, H) + gym-pusht
```
