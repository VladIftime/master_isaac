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

**Historical policy pool**: 20% of episodes pit each agent against a randomly sampled past version of the opponent, improving training stability. The pool holds the 5 most recent snapshots.

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
    Obs (interleaved per-object): Robot(7) + [obj_state(14) + goal(6) + dist(2)] × 2 = 51D
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
| Alice entropy 1.0 → 0.10 over min(max_iters, 250) iters | Floor raised to 0.10 so Alice retains late-stage exploration; denominator clamped at 250 so annealing always completes |
| ABC filter: Bob-failure only | Follows paper §3.3; avoids cloning trivial successes |
| Historical pool 20% | Paper ratio; pool holds last 5 snapshots (max_size=5 in HistoricalPolicyPool) |
| Success threshold 0.05 m (pos), ~2° (rot) | 0.04 m from paper; relaxed to 0.05 m in wrapper |

---

## File Structure

```
asyncDualPlayPPO/
├── train.py                            # Main training loop (Alice+Bob PPO, ABC, historical pool)
├── run_diagnostic_tests.sh             # Three-test diagnostic suite (headless, logs to runs/diag_*)
├── optuna_sweep.py                     # Hyperparameter sweep with Optuna
├── buffers.py                          # Low-level buffer utilities
├── test_checkpoint_chain.py            # Checkpoint save/load smoke test
│
├── cfg/
│   ├── ppo/ppo_continuous.yaml         # PPO + ABC hyperparameters
│   └── task/AsyncDualPlay.yaml         # Episode structure (timesteps, goals per episode)
│
├── algorithms/
│   ├── goal_encoder.py                 # GoalEncoder φ MLP + aux distance-prediction head
│   └── rl/ppo/
│       ├── module.py                   # ActorCritic, PermInvEncoder, MultiCategorical
│       ├── ppo.py                      # Base PPO (Alice)
│       ├── ppo_abc.py                  # PPOABC: PPO + Alice Behavioral Cloning (Bob)
│       └── storage.py                  # RolloutStorage + GPUDemonstrationBuffer
│
├── tasks/
│   ├── async_dual_play.py              # IsaacLab env config (scene, observations, rewards)
│   └── utils/
│       ├── wrapper.py                  # AsyncDualPlayEnvWrapper: phase management, rewards
│       ├── observations.py             # Observation functions (EE, objects, goals, distances)
│       ├── rewards.py                  # Alice reward constants + reward functions
│       ├── events.py                   # Reset events (objects, robot joints)
│       ├── terminations.py             # Episode termination conditions
│       ├── dummy_alice_wrapper.py      # Diagnostic wrappers (DummyBob, DummyGoalDistance, …)
│       └── base/events.py              # Base reset event helpers
│
├── utils/
│   ├── episode_manager.py              # EpisodeManager: phase tracking, goal storage
│   ├── goal_validator.py               # validate_goal: movement threshold check
│   └── historical_pool.py             # HistoricalPolicyPool: past-5-snapshot ring buffer
│
├── tests/
│   ├── test_abc.py                     # End-to-end ABC pipeline tests
│   └── test_abc_goal_encoder.py        # Goal encoder integration tests
│
├── hpc/
│   ├── train_high.slurm                # Production HPC job (A100, 512 envs)
│   ├── train_medium.slurm              # Medium-scale HPC job
│   ├── train_low.slurm                 # Small-scale HPC job
│   └── run_interactive.sh              # Interactive session helper
│
├── extras/                             # Offline analysis / visualisation scripts
│   ├── visualize_logs.py
│   ├── plot_results.py
│   ├── diagnose_logs.py
│   └── extract_updates.py
│
└── paper-async/
    ├── asymetric-self-play.pdf         # OpenAI ASP paper (Plappert et al. 2021)
    └── asymetric-self-play_charlie.pdf # Charlie/HSP paper (Sukhbaatar et al. 2018)
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

### Diagnostic tests (3-test suite)
```bash
# Locally (from master_isaac/):
bash asyncDualPlayPPO/run_diagnostic_tests.sh

# Individual tests:
# Test 1 — reward pipeline (DummyBobWrapper teleports target→goal, expect SR > 0)
python -m asyncDualPlayPPO.train --headless --num_envs 16 --max_iterations 50 --test_bob_reward

# Test 2 — Alice exploration sandbox (watch ValidGoals climb)
python -m asyncDualPlayPPO.train --headless --num_envs 32 --max_iterations 200

# Test 3 — PPO vs ABC balance (watch Loss/Bob/ABC vs Loss/Bob/Surrogate)
python -m asyncDualPlayPPO.train --headless --num_envs 64 --max_iterations 300
```

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
