# PBRS Analysis Plot Suite

## Part A: Why PBRS is the right reward function

| Plot | File | Description |
|------|------|-------------|
| A1 | `a1_potential_landscape.png` | 2D heatmap of Φ_pos(s) = exp(−k_p·d²) across the workspace. Shows smooth, bounded (0,1] potential with concentric gradients — no dead zones, no singularities. |
| A2 | `a2_gradient_comparison.png` | Gradient magnitude dR/dd for PBRS vs old fractional formula vs old raw delta. PBRS gradient peaks at d=0.13 m (the sweet spot for learning), falls to 0 at d=0 (stable at goal, no noise amplification). Old formula 1/d_prev amplifies noise near goal. |
| A3 | `a3_episode_simulation.png` | Side-by-side per-push reward comparison for a simulated 5-push episode (d: 0.20→0.08→...→0.01 m). PBRS produces smoothly diminishing returns; old formula amplifies near-goal rewards and includes penalty terms that shift the optimal policy. |
| A4 | `a4_cosine_distance.png` | Cosine angular distance (1−cos Δθ)/2 vs _yaw_distance_rad. Cosine is C∞ smooth everywhere; yaw_distance has a cusp/gradient-discontinuity at Δθ=±π. PBRS uses the smooth cosine metric to eliminate gradient cliffs in the Euler-angle space. |

## Part B: Why these specific hyperparameters were chosen

| Plot | File | Description |
|------|------|-------------|
| B1 | `b1_kp_sensitivity.png` | Potential shape and gradient for k_p ∈ {15, 30, 50}. k_p=15 is too flat (weak gradient beyond 0.15 m). k_p=50 is too sharp (Φ≈0.01 at d>0.30 m — dead zone). k_p=30 balances far-field signal (~0.07 at 0.30 m) with strong mid-range gradient (peak at d=0.13 m). |
| B2 | `b2_kr_sensitivity.png` | Rotation potential shape and gradient for k_r ∈ {3, 5, 10}. k_r=5 maps the cos_rot_err=0.01 success threshold to Φ_rot≈0.95, matching the position potential at its threshold. Good gradient across typical rotation range (±36°). |
| B3 | `b3_weight_scaling.png` | Expected per-push reward for w ∈ {1, 5, 10, 20}. w=10 yields ~[−2, +3] per push for typical distances — comparable magnitude to sparse bonuses (+5), so neither term dominates. Any scalar multiple is valid PBRS (policy invariance holds for all w). |
| B4 | `b4_gamma_shaping.png` | γ_shaping = 1.0 vs γ_shaping = 0.95. With γ=0.95, the 5% discount tax exceeds the marginal potential improvement near the goal, causing sign inversion — the agent is penalized for approaching. γ=1.0 (Grzes & Kudenko 2009) preserves policy invariance for episodic MDPs. |

## Reference

- Ng et al. (1999): *Policy invariance under reward transformations: Theory and application to reward shaping*
- Grzes & Kudenko (2009): *Theoretical and empirical analysis of reward shaping in reinforcement learning*

PBRS parameters: k_p=30.0, k_r=5.0, w_pos=10.0, w_rot=10.0, γ_shaping=1.0
Old formula parameters: α=3.0, β=0.5, β_rot=0.25
