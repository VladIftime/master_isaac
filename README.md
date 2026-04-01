# Asymmetric Self-Play with Goal Encoders for Dual-Arm Robotic Manipulation

This project combines two papers to build a hierarchical reinforcement learning system for dual-arm manipulation in Isaac Lab / Isaac Sim.

---

## Papers

### 1. Asymmetric Self-Play for Automatic Goal Discovery (OpenAI, Plappert et al. 2021)
`asymetric-self-play.pdf`

The core training framework. Two agents — **Alice** and **Bob** — play an adversarial game in a shared environment:

- **Alice** manipulates objects freely for T=100 steps to construct a goal state (a non-trivial object configuration).
- **Bob** then observes the resulting goal and must reproduce it from a fresh reset within T=200 steps.
- Goals repeat across 5 sub-goals per episode; each goal reuses the previous sub-goal's end state as the new start.

**Rewards** are sparse and symmetric:
- Bob receives **+1** for each object within the success threshold (0.04 m position, ~2° rotation).
- Bob receives **-1** if an object already at goal moves away.
- Bob receives a **+5 completion bonus** when all objects are simultaneously at goal.

**Alice Behavioral Cloning (ABC)**: When Bob fails a goal, Alice's trajectory for that sub-goal is stored in a BC replay buffer. Bob's PPO loss is augmented with a clipped imitation loss (ε=0.2, β=0.5) that clones Alice's actions, bootstrapping Bob's exploration.

**Historical policy pool**: 20% of episodes pit each agent against a randomly sampled past version of the opponent, preventing policy collapse.

---

### 2. Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning (Sukhbaatar et al. 2018)
`asymetric-self-play_charlie.pdf`

Introduces a **goal encoder** that compresses the goal description into a low-dimensional embedding, replacing naive goal-state concatenation in Bob's policy.

**Goal encoder** E: (s\*, s_t^B) → g_t ∈ ℝ^K maps the goal state s\* and Bob's current state s_t^B to a compact embedding g_t. Bob's policy becomes goal-conditioned: π'_B(s_t^B, g_t).

Two encoder architectures:
- **Difference form**: φ(s\*) − φ(s_t^B) — encodes relative progress toward goal.
- **Absolute form**: φ(s\*) — encodes goal independently of current state.

The encoder is integrated into Bob's second hidden layer:  
`h_2 = σ(W_2 σ(W_1 s_t) + W_g g_t)`

A high-level **Charlie controller** can also generate goal embeddings g_t to direct a pre-trained Bob, enabling hierarchical control without retraining the low-level policy.

---

## Project Contribution: Combining Both Papers for Dual-Arm Manipulation

This codebase merges the two frameworks and extends them to a **dual-arm robotic platform** (two UR5e robots with Robotiq grippers) simulated in **Isaac Lab / Isaac Sim**.

### Why the combination?

OpenAI's ASP provides an automatic curriculum (Alice discovers goals of increasing difficulty) and efficient exploration via ABC. However, raw goal-state concatenation in Bob's observation grows linearly with the number of objects and makes the policy sensitive to irrelevant goal dimensions.

Charlie's goal encoder solves this: the encoder compresses the goal into a fixed-size bottleneck g_t ∈ ℝ^K regardless of how many objects are in the scene. The embedding focuses Bob's attention on what still needs to change, rather than where everything currently is.

The dual-arm extension adds a second manipulator, doubling the number of objects in the workspace and making compact goal representations even more important.

### Architectural overview

```
Alice (PPO):
    Obs: EE pose(6) + gripper(1) + [obj_state(14)] × 2  = 35D
    Acts: joint velocity commands → RMPFlow controller
    Role: explore and construct interesting goal configurations

Bob (PPO + ABC + Goal Encoder):
    Obs (interleaved): Robot(7) + [obj_state(14) + goal(6) + dist(2)] × 2 = 51D
    Goal encoder:      E(goal_state, current_state) → g ∈ ℝ^K
    Acts: same action space as Alice
    Role: reproduce the goal Alice left behind

Episode Manager:
    Stores Alice's final state as the goal for Bob (12D LOCAL Euler per episode)
    Validates that Alice moved at least one object (pos_threshold or rot_threshold)
    Manages phase transitions and ABC buffer writes
```

**State format** (12D per episode, LOCAL frame, Euler):  
`[target_pos(3), target_euler(3), cube_pos(3), cube_euler(3)]`

**Bob obs per object** (14D, from `object_states()`):  
`[pos(3), euler(3), lin_vel(3), ang_vel(3), dist_to_gripper(1), contact(1)]`

**Goal encoder** (`use_goal_encoder = True`, hardcoded in `train.py`): always active for Bob.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ZYX Euler angles | Matches OpenAI paper Appendix A.2; avoids quaternion discontinuities in policy input |
| 12D goal state (pos+euler × 2 objects) | Compact; no velocities needed for goal definition |
| 51D Bob obs (interleaved) | Object state + goal + distance interleaved per-object; easier for encoder to associate |
| Alice entropy 1.0 → 0.01 over 100 iters | Annealed faster than paper (1856 envs → smaller batch) |
| ABC filter: Bob-failure only | Follows paper §3.3; avoids cloning trivial successes |
| Historical pool 20% | Exact paper ratio; sampled uniformly from past 50 checkpoints |
| Success threshold 0.04 m / ~2° | Paper Table 1 values |

---

## File Structure

```
asyncDualPlayPPO/
├── train.py                        # Main training loop (Alice+Bob PPO, ABC, historical pool)
├── tasks/
│   └── utils/
│       ├── wrapper.py              # AsyncDualPlayEnvWrapper: phase management, rewards, ABC
│       ├── observations.py         # Observation functions (EE, objects, goals, distances)
│       ├── dummy_alice_wrapper.py  # Diagnostic wrappers for testing reward/goal pipeline
│       └── ppo_abc.py              # Bob's PPO extended with ABC behavioral cloning loss
├── hpc/
│   ├── train_high.slurm            # Production HPC job (A100, 512 envs)
│   └── test_diag.slurm             # Diagnostic test suite (4 targeted checks)
├── tests/
│   └── test_abc.py                 # Step-by-step unit tests for the full pipeline
└── paper-async/
    ├── asymetric-self-play.pdf     # OpenAI ASP paper (Plappert et al. 2021)
    ├── asymetric-self-play_charlie.pdf  # Charlie/HSP paper (Sukhbaatar et al. 2018)
    └── README.md                   # This file
```

---

## Running

### Local (headless)
```bash
python train.py --num_envs 16 --max_iterations 500 --exp_name test_run --headless
```

### HPC (Apptainer / Isaac Lab container)
```bash
sbatch hpc/train_high.slurm
```

### Diagnostic tests
```bash
sbatch hpc/test_diag.slurm
# or locally:
python tests/test_abc.py --step 5 --num_envs 16 --num_iterations 10 --headless
```

Steps: 1=Pure PPO, 4=goal distance check, 5=reward pipeline, 6=movement detection.

---

## References

```bibtex
@article{plappert2021asymmetric,
  title={Asymmetric self-play for automatic goal discovery in robotic manipulation},
  author={Plappert, Matthias and Rajeswaran, Aravind and others},
  journal={arXiv preprint arXiv:2101.04882},
  year={2021}
}

@article{sukhbaatar2018learning,
  title={Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning},
  author={Sukhbaatar, Sainbayar and Lin, Zeming and Kostrikov, Ilya and Synnaeve, Gabriel and Szlam, Arthur and Fergus, Rob},
  journal={arXiv preprint arXiv:1811.09083},
  year={2018}
}
```
