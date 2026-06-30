# Literature Categories — ASP + GoalEncoder + Push-PPO Implementation

Each category maps to one or more components described in `implementations.md`.
Papers are sorted under the implementation aspect they most directly support.

---

## Category Overview

| # | Category | Implementation Component(s) | Papers |
|---|----------|----------------------------|--------|
| 1 | ASP + GoalEncoder Core | Alice↔Bob adversarial self-play loop (§3), GoalEncoder φ‑MLP latent compression (§3.2) | 6 |
| 2 | PPO & RL Algorithm Foundations | PPOABC optimizer (§3.2), PPO update (§3.5) | 3 |
| 3 | Push Primitive Mechanics | Planar pushing action space (§6–7), contact dynamics, object motion under pushing | 12 |
| 4 | PBRS Reward Shaping | Potential‑based dense rewards (§1 Models A–L), γ_shaping=1.0, k_p / k_r | 5 |
| 5 | Goal‑Conditioned & Hierarchical RL | Alice/Bob two‑level hierarchy (§3.1), GoalEncoder sub‑goal compression, HRL theory | 13 |
| 6 | Behavioral Cloning & Imitation | ABC trajectory buffer & BC loss (§3.2), demonstration‑driven policy learning | 4 |
| 7 | Push‑Grasp Synergy & Combined Manipulation | Domain context: combined prehensile / non‑prehensile manipulation for clutter | 8 |
| 8 | Dual‑Arm & Bimanual Manipulation | 2 × UR5e hardware (§1–2), multi‑arm coordination, benchmarks | 8 |
| 9 | Grasping & Dexterous Manipulation | Robotic manipulation background — grasp planning, dexterous hands, point‑cloud methods | 10 |
| 10 | Alternative Policy Architectures & Sim‑to‑Real | Future policy representations (diffusion, flow matching), physical deployment path | 4 |
| 11 | Multi‑Agent, Exploration & Sports RL | MARL, exploration theory, model‑based visual planning, sports, game theory | 8 |

**Total: 81 papers**

---

## Detailed Paper Listing

### 01 — ASP + GoalEncoder Core (3 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `sukhbaatar2018_goal_embeddings_self_play_hierarchical_rl` | Sukhbaatar et al. 2018 — *Learning Goal Embeddings via Self‑Play for Hierarchical RL* | GoalEncoder φ‑MLP architecture, Alice/Bob game concept, max‑pool embedding |
| `plappert2021_asymmetric_self_play` | Plappert et al. 2021 — *Asymmetric Self‑Play for Automatic Goal‑Conditioned RL* | ABC buffer, historical policy pool, outcome‑based Alice reward {+1,−1,+5} |
| `sukhbaatar2018_intrinsic_motivation_asp` | Sukhbaatar et al. — *Intrinsic Motivation and Automatic Curricula via Asymmetric Self‑Play* | Original ASP concept; time‑based Alice reward `R_A = γ_sp·max(0, t_B−t_A)` (used in Models G/H) |
| `dennis2020_paired_unsupervised_environment_design` | Dennis et al. 2020 — *Emergent Complexity and Zero‑Shot Transfer via Unsupervised Environment Design (PAIRED)* | Protagonist‑antagonist regret environment design; adversarial curriculum generation |
| `campero2021_amigo_adversarial_intrinsic_goals` | Campero et al. 2021 — *Learning with AMIGo: Adversarially Motivated Intrinsic Goals* | Teacher‑student adversarial goal generation; constructively adversarial curriculum |
| `durugkar2021_adversarial_intrinsic_motivation_rl` | Durugkar et al. 2021 — *Adversarial Intrinsic Motivation for Reinforcement Learning (AIM)* | Wasserstein‑distance adversarial motivation; exploration via goal difficulty |

### 02 — PPO & RL Algorithm Foundations (3 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `schulman2015_trust_region_policy_optimization_trpo` | Schulman et al. 2015 — *Trust Region Policy Optimization (TRPO)* | PPO predecessor; KL‑constrained policy updates |
| `schulman2017_proximal_policy_optimization_ppo` | Schulman et al. 2017 — *Proximal Policy Optimization Algorithms* | Clipped surrogate objective used by PPOABC; GAE integration |
| `haarnoja2018_soft_actor_critic` | Haarnoja et al. 2018 — *Soft Actor‑Critic* | Off‑policy alternative; entropy‑regularised RL |

### 03 — Push Primitive Mechanics (7 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `mason_1986_mechanics_planning_manipulator_pushing` | Mason 1986 — *Mechanics and Planning of Manipulator Pushing* | Foundational push mechanics, friction cone |
| `lynch_mason_1996_stable_pushing` | Lynch & Mason 1996 — *Stable Pushing* | Stable push directions, centre of friction |
| `akella_posing_polygonal_objects_pushing` | Akella & Mason 1998 — *Posing Polygonal Objects in the Plane by Pushing* | Cited in §2: rotation‑based push reward design |
| `goyal_ruina_papadopoulos_1991_planar_sliding_dry_friction` | Goyal et al. 1991 — *Planar Sliding with Dry Friction* | Limit surface, friction modelling |
| `howe_cutkosky_1996_force_motion_models_sliding` | Howe & Cutkosky 1996 — *Practical Force‑Motion Models for Sliding Manipulation* | Force‑motion sliding models |
| `yu2016_million_ways_to_be_pushed` | Yu et al. 2016 — *More Than a Million Ways to Be Pushed* | Empirical pushing dataset |
| `stuber2020_lets_push_things_forward_survey` | Stüber et al. 2020 — *Let's Push Things Forward: A Survey* | Comprehensive pushing survey |
| `vanderpol2020_mdp_homomorphic_networks_symmetry` | van der Pol et al. 2020 — *MDP Homomorphic Networks: Group Symmetries in RL* | Equivariant policies under rotation/reflection; group‑structured symmetry exploitation |
| `huang2022_equivariant_transporter_network` | Huang et al. 2022 — *Equivariant Transporter Network* | SE(2)‑equivariant pick‑and‑place; immediate generalisation across object orientations |
| `nguyen2024_equivariant_rl_partial_observability` | Nguyen et al. 2024 — *Equivariant RL under Partial Observability* | SO(2)/SO(3) symmetries in actor‑critic; equivariant policy design for robotic tasks |
| `urain2023_se3_diffusion_fields_grasping` | Urain et al. 2023 — *SE(3)‑DiffusionFields: Learning Smooth Cost Functions* | SE(3) geodesic distance for unified position‑orientation cost; joint grasp‑motion optimisation |
| `park1995_lie_group_robot_dynamics` | Park 1995 — *A Lie Group Formulation of Robot Dynamics* | Mathematical foundation for SE(3) coordinate representation; group metrics on rigid body configurations |

### 04 — PBRS Reward Shaping (5 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `ng_harada_russell_1999_policy_invariance_reward_shaping` | Ng, Harada & Russell 1999 — *Policy Invariance under Reward Shaping* | Foundational PBRS theorem; potential‑based shaping preserves optimal policy |
| `grzes_kudenko_2009_reward_shaping_analysis` | Grzes & Kudenko 2009 — *Reward Shaping Analysis* | Analysis of shaping in episodic MDPs (γ_shaping = 1.0) used in Models A–L |
| `devlin2012_dynamic_potential_reward_shaping` | Devlin & Kudenko 2012 — *Dynamic Potential‑Based Reward Shaping* | Extends PBRS to dynamic potentials; theoretical basis for adaptive shaping under curriculum |
| `harutyunyan2015_arbitrary_reward_potential_advice` | Harutyunyan et al. 2015 — *Expressing Arbitrary Reward Functions as Potential‑Based Advice* | Arbitrary reward functions expressed as potential‑based advice; validates general Φ(s) |
| `grzes2017_reward_shaping_episodic_rl` | Grześ 2017 — *Reward Shaping in Episodic Reinforcement Learning* | Episodic‑specific PBRS analysis; validates γ_shaping=1.0 for finite‑horizon tasks |

### 05 — Goal‑Conditioned & Hierarchical RL (13 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `nachum2018_data_efficient_hierarchical_rl_hiro` | Nachum et al. 2018 — *Data‑Efficient HRL (HIRO)* | Off‑policy HRL, sub‑goal re‑labelling |
| `vezhnevets2017_feudal_networks_hierarchical_rl` | Vezhnevets et al. 2017 — *FeUdal Networks (FuN)* | Manager/Worker HRL with differentiable latent goals |
| `beyret2019_dot_to_dot_explainable_hierarchical_rl` | Beyret et al. 2019 — *Dot‑to‑Dot: Explainable HRL for Robotic Manipulation* | Explainable hierarchical RL |
| `ji2022_hierarchical_rl_precise_soccer_quadruped` | Ji et al. 2022 — *HRL for Precise Soccer Shooting with a Quadrupedal Robot* | HRL applied; multi‑level skill hierarchy |
| `hutsebaut2022_hierarchical_rl_survey_open_challenges` | Hutsebaut-Buysse et al. 2022 — *HRL: A Survey and Open Research Challenges* | Comprehensive HRL survey |
| `wang2022_goal_auxiliary_actor_critic_6d_grasping` | Wang et al. 2022 — *Goal‑Auxiliary Actor‑Critic for 6D Robotic Grasping with Point Clouds* | Goal‑conditioned auxiliary learning |
| `florensa2017_reverse_curriculum_generation_rl` | Florensa et al. 2017 — *Reverse Curriculum Generation for RL* | Reverse curriculum: start near goal, expand outward; goal‑space difficulty progression |
| `narvekar2020_curriculum_learning_rl_survey` | Narvekar et al. 2020 — *Curriculum Learning for RL Domains: A Framework and Survey* | Systematic CL taxonomy; positions forced vs automatic curriculum approaches |
| `portelas2020_automatic_curriculum_deep_rl_survey` | Portelas et al. 2020 — *Automatic Curriculum Learning for Deep RL: A Short Survey* | ACL methods: self‑paced, teacher‑student, goal‑generation curricula |
| `nair2018_visual_rl_imagined_goals` | Nair et al. 2018 — *Visual RL with Imagined Goals* | VAE‑based goal encoder; retroactive goal relabeling; self‑supervised goal practice |
| `goyal2019_infobot_information_bottleneck_rl` | Goyal et al. 2019 — *InfoBot: Transfer and Exploration via the Information Bottleneck* | IB for RL; bottleneck identifies decision states; parallels GoalEncoder compression |
| `tishby2015_information_bottleneck_deep_learning` | Tishby & Zaslavsky 2015 — *Deep Learning and the Information Bottleneck Principle* | Foundational IB theory; explains when compression helps/hurts generalisation |
| `bacon2017_option_critic_architecture` | Bacon et al. 2017 — *The Option‑Critic Architecture* | Learns intra‑option policies and termination conditions; push primitives as temporally extended options |

### 06 — Behavioral Cloning & Imitation (4 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `torabi2018_behavioral_cloning_from_observation` | Torabi et al. 2018 — *Behavioral Cloning from Observation (BCO)* | State‑only imitation via inverse dynamics |
| `florence2022_implicit_behavioral_cloning` | Florence et al. 2022 — *Implicit Behavioral Cloning* | Energy‑based BC with multimodal action distributions |
| `hester2018_deep_q_learning_from_demonstrations` | Hester et al. 2018 — *Deep Q‑Learning from Demonstrations (DQfD)* | Demonstration‑boosted RL |
| `silver2018_residual_policy_learning` | Silver et al. 2018 — *Residual Policy Learning* | Learning residual correction policies |

### 07 — Push‑Grasp Synergy & Combined Manipulation (8 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `kasaei2024_synergy_pushing_grasping_throwing` | Kasaei et al. 2024 — *Harnessing the Synergy between Pushing, Grasping, and Throwing* (ICRA) | Three‑skill synergy RL |
| `wang2023_self_supervised_joint_pushing_grasping` | Wang et al. 2023 — *Self‑Supervised Learning for Joint Pushing and Grasping Policies* | Joint push‑grasp self‑supervision |
| `xu2021_efficient_goal_push_grasping_synergy` | Xu et al. 2021 — *Efficient Learning of Goal‑Oriented Push‑Grasping Synergy* | Goal‑oriented combined manipulation |
| `zeng2018_learning_synergies_pushing_grasping_vpg` | Zeng et al. 2018 — *Learning Synergies between Pushing and Grasping (VPG)* | Push‑grasp synergy with self‑supervision |
| `feldman2022_hybrid_shift_grasp_motion_primitives` | Feldman et al. 2022 — *A Hybrid Approach for Learning to Shift and Grasp* | Shift + grasp motion primitives (bin picking) |
| `he2021_scooping_manipulation_two_fingered_gripper` | He et al. 2021 — *Scooping Manipulation via Motion Control with a Two‑Fingered Gripper* | Non‑prehensile scooping |
| `zeng2019_pick_place_novel_objects_clutter` | Zeng et al. 2019 — *Robotic Pick‑and‑Place of Novel Objects in Clutter* | Pick‑and‑place in clutter |
| `nieuwenhuisen2013_mobile_bin_picking_anthropomorphic` | Nieuwenhuisen et al. 2013 — *Mobile Bin Picking with an Anthropomorphic Service Robot* | Mobile bin picking |

### 08 — Dual‑Arm & Bimanual Manipulation (8 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `george2025_coordinated_dual_arm_sac_poppy` | George et al. 2025 — *Coordinated Dual‑Arm Manipulation using RL (SAC on Poppy Humanoid)* | SAC dual‑arm grasping |
| `yang2024_collaborative_control_dual_arm_robots` | Yang et al. 2024 — *Collaborative Control Analysis of Dual‑Arm Robots* | Classic + intelligent control survey |
| `chen2022_human_level_bimanual_dexterous_manipulation` | Chen et al. 2022 — *Towards Human‑Level Bimanual Dexterous Manipulation* | Bimanual dexterity |
| `zhao2023_fine_grained_bimanual_aloha` | Zhao et al. 2023 — *Learning Fine‑Grained Bimanual Manipulation with Low‑Cost Hardware (ALOHA)* | Low‑cost bimanual learning |
| `cui2025_task_adaptive_dual_arm_rl` | Cui et al. 2025 — *Task‑Adaptive Dual‑Arm RL* | Adaptive dual‑arm policy |
| `ma2025_coordinated_badminton_legged_manipulators` | Ma et al. 2025 — *Learning Coordinated Badminton Skills for Legged Manipulators* | Dual‑arm sports coordination |
| `mu2025_robotwin_dual_arm_benchmark` | Mu et al. 2025 — *RobotWin: Dual‑Arm Benchmark* | Bimanual benchmark |
| `chen2025_robotwin2_scalable_data_generator_benchmark` | Chen et al. 2025 — *RoboTwin 2.0: Scalable Data Generator & Benchmark* | RobotWin v2 |

### 09 — Grasping & Dexterous Manipulation (10 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `sekkat2024_review_rl_robotic_grasping` | Sekkat et al. 2024 — *Review of RL for Robotic Grasping: Analysis and Recommendations* | RL grasping survey |
| `mohammed2022_review_robotic_manipulation_clutter` | Mohammed et al. 2022 — *Review of Learning‑Based Robotic Manipulation in Cluttered Environments* | Clutter manipulation survey |
| `murali2020_6dof_grasping_target_driven_clutter` | Murali et al. 2020 — *6‑DOF Grasping for Target‑Driven Object Manipulation in Clutter* | 6‑DOF grasping |
| `kasaei2021_mvgrasp_multi_view_3d_grasping` | Kasaei et al. 2021 — *MVGrasp: Real‑Time Multi‑View 3D Object Grasping* | Multi‑view grasping |
| `wu2020_generative_attention_general_grasping` | Wu et al. 2020 — *Generative Attention Learning: a "GenerAL" Framework for Multi‑Fingered Grasping* | Attention‑based grasping |
| `xu2021_adagrasp_adaptive_gripper_aware_grasping` | Xu et al. 2021 — *AdaGrasp: Learning an Adaptive Gripper‑Aware Grasping Policy* | Gripper‑aware grasping |
| `kumra2020_antipodal_grasping_generative_residual_cnn` | Kumra et al. 2020 — *Antipodal Robotic Grasping (GGCNN)* | Generative grasping CNN |
| `ciocarlie2007_dexterous_grasping_eigengrasps` | Ciocarlie et al. 2007 — *Dexterous Grasping via Eigengrasps* | Eigengrasps approach |
| `miller2003_automatic_grasp_planning_shape_primitives` | Miller et al. 2003 — *Automatic Grasp Planning Using Shape Primitives* | Shape‑primitive grasp planning |
| `duan2021_dexterous_grasping_point_cloud_survey` | Duan et al. 2021 — *Robotics Dexterous Grasping: Methods Based on Point Cloud and Deep Learning* | Point‑cloud dexterous grasping review |

### 10 — Alternative Policy Architectures & Sim‑to‑Real (4 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `gkanatsios2025_3d_flowmatch_actor_unified_policy` | Gkanatsios et al. 2025 — *3D FlowMatch Actor: Unified 3D Policy (3DFA)* | Flow‑matching alternative to diffusion |
| `chi2023_diffusion_policy` | Chi et al. 2023 — *Diffusion Policy* | Denoising diffusion for visuomotor policy |
| `yang2023_sim_to_real_tactile_pushing` | Yang et al. 2023 — *Sim‑to‑Real: Model‑Based and Model‑Free Deep RL for Tactile Pushing* | Sim‑to‑real transfer |
| `luo2020_accelerating_rl_reaching_curriculum` | Luo et al. 2020 — *Accelerating RL for Reaching using Continuous Curriculum Learning (PCCL)* | Sample‑efficient reaching with curriculum |

### 11 — Multi‑Agent, Exploration & Sports RL (6 papers)

| Directory | Paper | Role |
|-----------|-------|------|
| `krnjaic2024_scalable_marl_warehouse_logistics` | Krnjaic et al. 2024 — *Scalable Multi‑Agent RL for Warehouse Logistics* | MARL at scale |
| `ladosz2022_exploration_deep_rl_survey` | Ladosz et al. 2022 — *Exploration in DRL: A Survey* | Exploration strategies |
| `nematollahi2020_hindsight_foresight_structured_dynamics` | Nematollahi et al. 2020 — *Hindsight for Foresight: Unsupervised Structured Dynamics (Finn/Levine)* | Model‑based visual planning |
| `huang2021_visual_foresight_trees_object_retrieval` | Huang et al. 2021 — *Visual Foresight Trees for Object Retrieval from Clutter* | Model‑based tree search |
| `ren2025_smash_humanoid_pingpong_egocentric` | Ren et al. 2025 — *SMASH: Mastering Scalable Whole‑Body Skills for Humanoid Ping‑Pong* | Dual‑arm sports RL |
| `su2025_hitter_humanoid_table_tennis_robot` | Su et al. 2025 — *HITTER: A Humanoid Table Tennis Robot via Hierarchical Planning and Learning* | Dual‑arm table tennis |
| `berner2019_dota2_large_scale_deep_rl` | Berner et al. / OpenAI 2019 — *Dota 2 with Large Scale Deep Reinforcement Learning* | Competitive self‑play at scale; time‑pressure rewards; emergent strategies |
| `letcher2019_differentiable_game_mechanics` | Letcher et al. 2019 — *Differentiable Game Mechanics* | Decomposes game Jacobian; potential vs Hamiltonian dynamics; stability of adversarial training |

---

*Generated 2026-06-24 from implementations.md sections and literature survey. All directories follow naming convention: `[firstauthor][year]_[descriptive_snake_case]`.*
