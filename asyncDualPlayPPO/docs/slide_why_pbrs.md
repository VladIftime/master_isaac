# Slide — Why PBRS for the Push-T Task

**Title:** Why PBRS? — Reward design for one-arm Push-T

## Bullets (on-slide)

- **The task couples translation + rotation.** Non-prehensile Push-T: by limit-surface mechanics, *every* push changes position **and** yaw at once — you can't fix one without disturbing the other.
- **Both naïve rewards fail.** Sparse fires on ~0.14 % of pushes (no gradient); hand-tuned dense is non-Markovian, amplifies contact noise near the goal, and lets rotation swamp position ~20× — with no guarantee it optimizes the real task.
- **PBRS gives dense signal *without* changing the goal.** Policy-invariance (Ng 1999); episodic γ_shaping = 1 → zero-sum cycles (no oscillation farming) and no near-goal sign flip; one SE(2) potential `Φ=exp(−k·d_pose²)` whose length `L` is the object's *physical* coupling constant, not a tuned weight.
- **Result — same plain PPO, no curriculum, reward swapped only:** position error 0.197 → 0.114 m, rotation error 0.69 → 0.21 (3.3×), 55 % SR on held-out scenes.

---

## Speaker notes (~1:30)

**(0:00–0:20) Task.** We push a T-shaped block to a target pose with a single arm and a simple push primitive. The defining physics is that pushing is non-prehensile: by the limit-surface theory of planar sliding, every push changes translation *and* orientation simultaneously. You cannot nudge the T without also rotating it.

**(0:20–0:45) The reward problem.** That makes reward design genuinely hard. A pure sparse success reward fires on about a tenth of a percent of pushes — there's effectively no learning signal. But the obvious fix, a hand-tuned dense reward, quietly breaks the task: ours was non-Markovian, it amplified physics noise right where precision matters near the goal, and its rotation term dominated position about twenty to one. Worst of all, nothing guarantees it's even optimizing the true objective.

**(0:45–1:15) Why PBRS.** Potential-based reward shaping solves exactly this. It supplies a dense per-push gradient, but Ng's theorem proves the optimal policy is unchanged. Because the task is episodic we set the shaping discount to one, so any back-and-forth cycle sums to exactly zero — no reward hacking — and the gradient never flips negative near the goal. And we score the *coupled* pose with a single SE(2) potential, where the length that trades rotation against translation isn't tuned — it's the object's radius of gyration, straight from the limit surface.

**(1:15–1:30) Payoff.** Same plain PPO, no curriculum — we swap only the reward. Position error drops from twenty to eleven centimetres, rotation error improves three-fold, and it generalizes to fifty-five percent on held-out scenes. The reward, on its own, is the right choice.
