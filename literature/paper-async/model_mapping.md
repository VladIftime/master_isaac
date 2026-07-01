# Model-Level Paper Mapping — PBRS Models A–I

Maps all 81 papers to the specific PBRS models described in `implementations.md` §1.
Each paper can support multiple models. Justifications are kept to one line.

---

## Model A — PBRS Dense Reward, Single-Agent PPO

**Description:** Single-agent PPO with PBRS dense reward `F=Φ(s')−Φ(s)`, object-relative obs+act, no curriculum. Sparse: +5 pos-only (no termination), +2 both (terminates). Penalties: −5 tip/launch/OOB/table (terminates).

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **PBRS foundations** | `ng_harada_russell_1999_policy_invariance_reward_shaping` | Original PBRS theorem — potential-based shaping preserves optimal policy |
| | `grzes_kudenko_2009_reward_shaping_analysis` | Analysis of shaping in episodic MDPs; `γ_shaping=1.0` justification |
| | `devlin2012_dynamic_potential_reward_shaping` | Extends PBRS to dynamic potentials; theoretical basis for `Φ(s)=exp(-k·d²)` |
| | `harutyunyan2015_arbitrary_reward_potential_advice` | Arbitrary reward functions expressed as potential-based advice; validates any `Φ(s)` |
| | `grzes2017_reward_shaping_episodic_rl` | Episodic-specific PBRS analysis; validates `γ_shaping=1.0` for finite-horizon tasks |
| **PPO algorithm** | `schulman2017_proximal_policy_optimization_ppo` | Clipped surrogate objective, GAE — the training algorithm |
| | `schulman2015_trust_region_policy_optimization_trpo` | PPO predecessor; KL-constrained policy updates |
| | `haarnoja2018_soft_actor_critic` | Entropy-regularised off-policy alternative (contextual comparison) |
| **Push primitive mechanics** | `mason_1986_mechanics_planning_manipulator_pushing` | Foundational push mechanics; friction cone |
| | `lynch_mason_1996_stable_pushing` | Stable push directions; centre of friction |
| | `akella_posing_polygonal_objects_pushing` | Posing objects via pushing; rotation-based reward design |
| | `goyal_ruina_papadopoulos_1991_planar_sliding_dry_friction` | Limit surface, friction modelling |
| | `howe_cutkosky_1996_force_motion_models_sliding` | Force-motion sliding models |
| | `yu2016_million_ways_to_be_pushed` | Empirical pushing dataset |
| | `stuber2020_lets_push_things_forward_survey` | Comprehensive pushing survey |
| **Action representation** | `florence2022_implicit_behavioral_cloning` | Multimodal action distributions (context for discrete push bins) |
| | `silver2018_residual_policy_learning` | Residual action corrections (context for push improvement) |

---

## Model B — PBRS + Forced pos→rot Curriculum Ramp

**Description:** Model A + deterministic curriculum controller. Phase 1: `w_rot=0`, position-only termination. Phase 2 (triggered by `ema_pos_err<0.08` for 50 iters): `w_rot` ramps 0→10 over 200 iters. Phase 3: full multi-objective.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model A papers** | (see above) | PBRS foundations, PPO, push mechanics |
| **Curriculum learning** | `narvekar2020_curriculum_learning_rl_survey` | Systematic taxonomy of CL methods in RL; positions forced curriculum vs automatic |
| | `portelas2020_automatic_curriculum_deep_rl_survey` | Automatic curriculum survey; self-paced, teacher-student, goal-generation |
| | `florensa2017_reverse_curriculum_generation_rl` | Reverse curriculum: start near goal, expand outward; analogous to pos→rot progression |
| | `luo2020_accelerating_rl_reaching_curriculum` | Precision-based continuous curriculum (PCCL); gradual requirement tightening |
| **Multi-objective RL** | (no dedicated paper in collection) | Sequential activation of objectives is implicit, not formal |
| **Hierarchical structure** | `hutsebaut2022_hierarchical_rl_survey_open_challenges` | HRL survey; relevance to staged curriculum as a form of hierarchy |
| | `sukhbaatar2018_intrinsic_motivation_asp` | Self-regulating curriculum via competition; contrast with forced curriculum |

---

## Model C — PBRS + ASP (Alice/Bob Two-Phase)

**Description:** Fork of `train_push_asp.py` with PBRS for Bob. ASP two-phase loop, GoalEncoder for Bob, ABC behavioral cloning, historical policy pool. Bob sparse: +5 pos-only gate, +2 both-threshold.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model A papers** | (see above) | PBRS, PPO, push mechanics |
| **Asymmetric Self-Play** | `plappert2021_asymmetric_self_play` | Core ASP: Alice/Bob adversarial game, ABC buffer, historical pool |
| | `sukhbaatar2018_intrinsic_motivation_asp` | Original ASP concept; self-regulating curriculum feedback |
| | `sukhbaatar2018_goal_embeddings_self_play_hierarchical_rl` | Goal embedding via self-play; Alice generates sub-goals |
| **GoalEncoder** | `sukhbaatar2018_goal_embeddings_self_play_hierarchical_rl` | φ-MLP architecture, max-pool, 8D latent, difference variant |
| **Behavioral cloning** | `torabi2018_behavioral_cloning_from_observation` | BCO: state-only imitation via inverse dynamics |
| | `florence2022_implicit_behavioral_cloning` | Energy-based BC with multimodal distributions |
| | `hester2018_deep_q_learning_from_demonstrations` | Demonstration-boosted RL (ABC: Alice demonstrates for Bob) |
| **Hierarchical RL** | `nachum2018_data_efficient_hierarchical_rl_hiro` | Off-policy HRL; sub-goal re-labelling (parallels Alice→Bob goal passing) |
| | `vezhnevets2017_feudal_networks_hierarchical_rl` | Manager/Worker HRL with differentiable latent goals |
| | `beyret2019_dot_to_dot_explainable_hierarchical_rl` | Explainable HRL for robotic manipulation |
| **Adversarial curriculum** | `dennis2020_paired_unsupervised_environment_design` | PAIRED: protagonist-antagonist regret environment design; parallels Alice/Bob |
| | `campero2021_amigo_adversarial_intrinsic_goals` | AMIGo: teacher-student adversarial goals; constructive adversary |
| | `durugkar2021_adversarial_intrinsic_motivation_rl` | AIM: adversarial intrinsic motivation via Wasserstein distance |
| **Goal-conditioned RL** | `nair2018_visual_rl_imagined_goals` | VAE-based goal encoder; retroactive goal relabeling |
| | `wang2022_goal_auxiliary_actor_critic_6d_grasping` | Goal-auxiliary actor-critic; auxiliary distance-prediction head |
| **Information bottleneck** | `goyal2019_infobot_information_bottleneck_rl` | IB for RL; discovering decision states via bottleneck; parallels GoalEncoder compression |

---

## Model D — Model C with GoalEncoder Ablated

**Description:** Bob's GoalEncoder removed (`use_goal_encoder=False`). PI-encoder sees full 22D per-object chunk (obj+goal+dist) directly, no 8D latent compression. Tests whether the 8D bottleneck helps or hurts under PBRS + ASP.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model C papers** | (see above) | Same PBRS+ASP+PPO foundation |
| **Information bottleneck theory** | `tishby2015_information_bottleneck_deep_learning` | Foundational IB principle; explains when compression helps/diminishes generalization |
| | `goyal2019_infobot_information_bottleneck_rl` | Practical IB in RL; bottleneck identifies decision states, filters distractors |
| | `sukhbaatar2018_goal_embeddings_self_play_hierarchical_rl` | States that bottleneck "forces Bob to compactly represent the goal"; Model D tests this |
| **Representation learning** | `nair2018_visual_rl_imagined_goals` | VAE-based goal representation; ablation of latent dimension |
| | `plappert2021_asymmetric_self_play` | Has ablation studies methodology (removing ABC, historical pool, etc.) |
| **Ablation methodology** | `vezhnevets2017_feudal_networks_hierarchical_rl` | Ablation study of FuN; validates design choices through systematic removal |
| | `sekkat2024_review_rl_robotic_grasping` | Survey with ablation recommendations |

---

## Model E — T-block + SE(2) d_pose Unified Metric

**Description:** Model C + SE(2) metric `d_pose = sqrt(dx² + dy² + L²·dθ²)` with characteristic length `L=0.07m` for T-block. Single PBRS potential `Φ(s)=exp(-k·d_pose²)` replaces separate position/rotation potentials. Observation: `[d_pose, bearing]`. Success: `d_pose < 0.055m`.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model C papers** | (see above) | PBRS+ASP+PPO foundation |
| **SE(2)/SE(3) geometry** | `park1995_lie_group_robot_dynamics` | Lie group formulation of rigid body dynamics; SE(3) coordinate representation |
| | `urain2023_se3_diffusion_fields_grasping` | SE(3) diffusion fields; combines position+orientation in a single cost function; uses Logmap for geodesic distance |
| | `lynch_mason_1996_stable_pushing` | References Lie groups (Jurdjevic & Sussmann 1972); planar rigid body configuration space SE(2) |
| | `mason_1986_mechanics_planning_manipulator_pushing` | Geometric constraints in planar manipulation; object pose as configuration |
| **Equivariance & symmetry** | `vanderpol2020_mdp_homomorphic_networks_symmetry` | Group-structured symmetries in RL; equivariant policies under rotations/reflections |
| | `huang2022_equivariant_transporter_network` | SE(2)-equivariant pick-and-place; immediate generalization to rotated objects |
| | `nguyen2024_equivariant_rl_partial_observability` | Equivariant RL; incorporating rotation symmetries into actor-critic networks |
| **Combined position-orientation metrics** | `plappert2021_asymmetric_self_play` | Uses separate position AND orientation checks for goal success; Model E unifies them |
| | `urain2023_se3_diffusion_fields_grasping` | SE(3) geodesic distance; directly addresses how to combine dx, dy, dθ into a single metric |
| | `goyal_ruina_papadopoulos_1991_planar_sliding_dry_friction` | Limit surface symmetry; shape of friction distribution determines feasible motions |
| **Characteristic length selection** | `park1995_lie_group_robot_dynamics` | Provides the mathematical framework for weighting translation vs rotation |
| | `akella_posing_polygonal_objects_pushing` | Object shape determines push outcomes; shape-dependent length scale implicit |

---

## Model F — Disc + Position-Only d_pose (Rotation Symmetry)

**Description:** Model E with rotationally-symmetric disc (`char_length=0.0` collapses `d_pose` to `sqrt(dx² + dy²)`). Rotation observed but not rewarded — network must learn to ignore meaningless yaw.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model E papers** | (see above) | SE(2) geometry, PBRS+ASP foundation |
| **Rotational symmetry exploitation** | `vanderpol2020_mdp_homomorphic_networks_symmetry` | Group symmetries in RL; policy/value functions invariant under SO(2) rotations |
| | `huang2022_equivariant_transporter_network` | SE(2)-equivariant manipulation; pick/place generalized across orientations |
| | `nguyen2024_equivariant_rl_partial_observability` | Equivariant actor-critic; SO(2)/SO(3) symmetries as inductive bias |
| | `goyal_ruina_papadopoulos_1991_planar_sliding_dry_friction` | Limit surface symmetry — circular slider has maximum symmetry |
| | `howe_cutkosky_1996_force_motion_models_sliding` | Circular symmetry; center of pressure simplifies force-motion models |
| **Continuous symmetry in RL** | No dedicated paper — `char_length=0.0` is a design choice, not a literature-backed method | Gap: no paper on zeroing reward weights for task-irrelevant DOFs |
| **Shape-dependent learning** | `howe_cutkosky_1996_force_motion_models_sliding` | Object shape (circular vs non-circular) affects sliding mechanics |
| | `stuber2020_lets_push_things_forward_survey` | Survey covers object shape effects on push outcomes |

---

## Model G — T-block + Time-Based ASP (Sukhbaatar Alice Reward)

**Description:** Model E + time-based Alice reward `R_A = γ_sp·max(0, t_B − t_A)`, `γ_sp=0.5`. Shallow goal penalty removed (redundant). ABC disabled. Same Bob PBRS d_pose. Self-regulating curriculum: Alice incentivized to find goals at the frontier of Bob's capability.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model E papers** | (see above) | SE(2) geometry, PBRS+ASP |
| **Time-based self-play** | `sukhbaatar2018_intrinsic_motivation_asp` | Core: `R_A ∝ max(0, t_B − t_A)` — self-regulating curriculum via effort asymmetry |
| | `sukhbaatar2018_goal_embeddings_self_play_hierarchical_rl` | Alice/Bob game where time/effort asymmetry drives exploration |
| **Adversarial curriculum design** | `dennis2020_paired_unsupervised_environment_design` | PAIRED: regret-maximizing adversary generates solvable but challenging environments; parallels `t_B − t_A` incentive |
| | `campero2021_amigo_adversarial_intrinsic_goals` | AMIGo: teacher proposes goals student finds challenging but achievable; constructively adversarial |
| | `durugkar2021_adversarial_intrinsic_motivation_rl` | AIM: adversarial intrinsic motivation; Wasserstein distance reward for goal-reaching |
| | `florensa2017_reverse_curriculum_generation_rl` | Contrast: reverse curriculum is explicit; time-based ASP is emergent |
| **Self-regulating feedback** | `sukhbaatar2018_intrinsic_motivation_asp` | "The self-regulating feedback between Alice and Bob allows them to automatically construct a curriculum" |
| **Temporal abstraction / options** | `bacon2017_option_critic_architecture` | Option-Critic: learns intra-option policies and termination conditions; each push is an option with duration `t_A, t_B` |
| | `hutsebaut2022_hierarchical_rl_survey_open_challenges` | Semi-MDPs, options framework; push primitive episodes as temporally extended actions |
| **Game theory / multi-agent dynamics** | `letcher2019_differentiable_game_mechanics` | Decomposes game Jacobian into potential and Hamiltonian components; analyzes stability of adversarial training |
| | `berner2019_dota2_large_scale_deep_rl` | Self-play at scale with time-pressure rewards; emergent strategies under competitive pressure |
| **Goal representation** | `nair2018_visual_rl_imagined_goals` | Goal-conditioned policy with VAE goal encoder; retroactive relabeling |
| | `goyal2019_infobot_information_bottleneck_rl` | IB for RL; bottleneck identifies decision states under adversarial goal generation |

---

## Model H — Disc + Time-Based ASP

**Description:** Model F (disc) + time-based Alice reward (Model G). Tests whether Sukhbaatar's self-regulation prevents toxic curriculum collapse observed in Model F (Alice creating impossibly hard goals, Bob SR 29%→6.5%).

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model F papers** | (see above) | Rotational symmetry, SE(2) d_pose |
| **All Model G papers** | (see above) | Time-based ASP, self-regulating curriculum |
| **Combined symmetry + curriculum** | `vanderpol2020_mdp_homomorphic_networks_symmetry` | Symmetry exploitation under adversarial curriculum; equivariance reduces search space |
| | `huang2022_equivariant_transporter_network` | Equivariant manipulation reduces the number of unique object configurations to learn |
| | `nguyen2024_equivariant_rl_partial_observability` | Equivariant policies are more sample-efficient — critical when curriculum is automatically generated |

---

## Model I — Model G + Bob Time Penalty

**Description:** Model G + Bob time penalty `R_B += −γ_sp·t_B`. Full Sukhbaatar symmetric reward: Alice rewarded for creating hard goals (large `t_B−t_A`), Bob penalized for taking many pushes. Creates time-urgency for Bob and zero-sum tension.

### Required Theory Components

| Component | Papers | Justification |
|-----------|--------|---------------|
| **All Model G papers** | (see above) | Time-based ASP, SE(2) geometry, PBRS |
| **Symmetric time-based rewards** | `sukhbaatar2018_intrinsic_motivation_asp` | Both agents receive time-based rewards; full competitive structure |
| | `berner2019_dota2_large_scale_deep_rl` | Competitive self-play with time/score pressure; symmetric reward structure |
| | `letcher2019_differentiable_game_mechanics` | Differentiable game decomposition; necessary to understand whether Bob's penalty creates stable or unstable training dynamics (zero-sum → potential vs Hamiltonian components) |
| | `durugkar2021_adversarial_intrinsic_motivation_rl` | Adversarial motivation where both agents' rewards depend on the other's behavior |
| **Negative time pressure** | `berner2019_dota2_large_scale_deep_rl` | Time pressure rewards in competitive self-play; emergent efficiency strategies |
| **Zero-sum game dynamics** | `letcher2019_differentiable_game_mechanics` | Hamiltonian component of game Jacobian causes cyclic/conservative dynamics; directly relevant to stability of symmetric time-based ASP |
| | `plappert2021_asymmetric_self_play` | ABC disabled in Models G/H — why? Time-based reward may make ABC redundant |
| **Temporal abstraction** | `bacon2017_option_critic_architecture` | Push count `t_B` as option termination criterion; time penalty as termination cost |
| | `nachum2018_data_efficient_hierarchical_rl_hiro` | HRL with fixed temporal abstraction (c steps); `t_B − t_A` as learned temporal horizon |

---

## Cross-Reference Matrix

Paper → Model support. ◆ = primary support, ● = supporting.

| Paper | A | B | C | D | E | F | G | H | I |
|-------|---|---|---|---|---|---|---|---|---|
| `plappert2021_asymmetric_self_play` | | | ◆ | ● | ● | ● | ● | ● | ● |
| `sukhbaatar2018_intrinsic_motivation_asp` | | | ● | | | | ◆ | ◆ | ◆ |
| `sukhbaatar2018_goal_embeddings_self_play_hierarchical_rl` | | | ◆ | ◆ | | | ● | ● | |
| `ng_harada_russell_1999_policy_invariance_reward_shaping` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `grzes_kudenko_2009_reward_shaping_analysis` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `devlin2012_dynamic_potential_reward_shaping` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `harutyunyan2015_arbitrary_reward_potential_advice` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `grzes2017_reward_shaping_episodic_rl` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `schulman2017_proximal_policy_optimization_ppo` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `schulman2015_trust_region_policy_optimization_trpo` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `haarnoja2018_soft_actor_critic` | ● | ● | | | | | | | |
| `mason_1986_mechanics_planning_manipulator_pushing` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `lynch_mason_1996_stable_pushing` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `akella_posing_polygonal_objects_pushing` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `goyal_ruina_papadopoulos_1991_planar_sliding_dry_friction` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `howe_cutkosky_1996_force_motion_models_sliding` | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ | ◆ |
| `yu2016_million_ways_to_be_pushed` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `stuber2020_lets_push_things_forward_survey` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `narvekar2020_curriculum_learning_rl_survey` | | ◆ | ● | ● | | | | | |
| `portelas2020_automatic_curriculum_deep_rl_survey` | | ◆ | ● | | | | ● | ● | |
| `florensa2017_reverse_curriculum_generation_rl` | | ◆ | ● | | | | ● | ● | |
| `luo2020_accelerating_rl_reaching_curriculum` | | ◆ | | | | | | | |
| `torabi2018_behavioral_cloning_from_observation` | | | ◆ | ◆ | ● | ● | | | |
| `florence2022_implicit_behavioral_cloning` | ● | ● | ◆ | ◆ | ● | ● | | | |
| `hester2018_deep_q_learning_from_demonstrations` | | | ◆ | ◆ | | | | | |
| `silver2018_residual_policy_learning` | ● | ● | | | | | | | |
| `nachum2018_data_efficient_hierarchical_rl_hiro` | | | ◆ | ◆ | ● | ● | ● | ● | ● |
| `vezhnevets2017_feudal_networks_hierarchical_rl` | | | ◆ | ◆ | | | | | |
| `beyret2019_dot_to_dot_explainable_hierarchical_rl` | | | ● | | | | | | |
| `hutsebaut2022_hierarchical_rl_survey_open_challenges` | | ● | ● | ● | | | ● | ● | ● |
| `nair2018_visual_rl_imagined_goals` | | | ◆ | ◆ | | | ● | ● | |
| `goyal2019_infobot_information_bottleneck_rl` | | | ◆ | ◆ | | | ● | ● | |
| `wang2022_goal_auxiliary_actor_critic_6d_grasping` | | | ● | ● | | | | | |
| `tishby2015_information_bottleneck_deep_learning` | | | | ◆ | | | | | |
| `dennis2020_paired_unsupervised_environment_design` | | | ◆ | | | | ◆ | ◆ | |
| `campero2021_amigo_adversarial_intrinsic_goals` | | | ◆ | | | | ◆ | ◆ | |
| `durugkar2021_adversarial_intrinsic_motivation_rl` | | | ◆ | | | | ◆ | ◆ | ◆ |
| `park1995_lie_group_robot_dynamics` | | | | | ◆ | ◆ | ● | ● | |
| `urain2023_se3_diffusion_fields_grasping` | | | | | ◆ | ◆ | ● | ● | |
| `vanderpol2020_mdp_homomorphic_networks_symmetry` | | | | | ◆ | ◆ | | | |
| `huang2022_equivariant_transporter_network` | | | | | ◆ | ◆ | | | |
| `nguyen2024_equivariant_rl_partial_observability` | | | | | ◆ | ◆ | | | |
| `bacon2017_option_critic_architecture` | | | | | | | ● | ● | ◆ |
| `berner2019_dota2_large_scale_deep_rl` | | | | | | | ● | ● | ◆ |
| `letcher2019_differentiable_game_mechanics` | | | | | | | ● | ● | ◆ |

*Note: Papers from categories 07 (push-grasp), 08 (dual-arm), 09 (grasping), and most of 11 (misc) are general domain background and not listed in this per-model matrix. They support the broader research context but not specific model components.*

---

*Generated 2026-06-24. ◆ = primary theoretical support, ● = contextual/secondary support.*
