# Asymmetric Self-Play with Goal Encoders for Robotic Manipulation

Combines two papers to train dual-arm manipulation policies via adversarial self-play
in Isaac Lab / Isaac Sim.

---

## Papers

### 1. Asymmetric Self-Play (OpenAI, Plappert et al. 2021)

Two agents — **Alice** and **Bob** — play an adversarial game:

- **Alice** (100 steps): manipulates objects freely to construct a goal configuration.
- **Bob** (100 steps): observes the goal and must reproduce it from a fresh reset.
- Goals repeat across 5 sub-goals per episode.
- **ABC**: when Bob fails, Alice's trajectory is stored as a demonstration for Bob
  to clone via clipped imitation loss (β = 0.5).
- **Historical pool**: 20% of episodes pit each agent against a past policy snapshot
  (last 5 saved every 50 iters).

**Alice rewards** (per goal, at phase end):
| Outcome | Reward |
|---|---|
| Valid goal (>0.10m displacement) | +1 |
| Shallow goal (0.05–0.10m) | −1 |
| Out-of-zone / invalid | −3 |
| Bob fails | +5 at Bob phase end |
| Bob succeeds | −1 at Bob phase end |
| Off-table / IK fail | −3 / −1 (early termination) |

**Bob rewards** (per step + phase end):
| Event | Reward |
|---|---|
| Object enters goal threshold | +1 |
| Object leaves goal threshold | −1 |
| All objects at goal simultaneously | +5 (episode ends) |
| Phase-end progress | clamp(0.6·Δpos/init + 0.4·Δrot/init, −1, +1) |

The phase-end progress reward mirrors Alice's episodic feedback structure — Bob
gets a grade on every trial, not just the 2–4% where sparse thresholds fire.
Per-step dense delta rewards were tested (v2–v5) and reverted (Fix P27): they
produced zero-mean noise that killed both agents' gradient flow.

### 2. Goal Embeddings via Self-Play (Sukhbaatar et al. 2018)

A **GoalEncoder** φ-MLP compresses `(goal_pose, current_pose)` into an 8D latent
vector, injected additively into Bob's first trunk layer:
`h₁ = ReLU(LN(W·enc + Wg·g_pooled))`. Difference variant: `g_i = φ(goal_i) − φ(current_i)`.

### 3. Single-object T-block variant (this implementation)

OpenAI's 2021 paper used 200 Bob steps for multi-block stacking tasks. For
single-object tabletop pushing, Bob's horizon is halved to 100 steps — credit
assignment is the bottleneck, not task complexity.

---

## Architecture

```
Alice (PPO, ent_coef=0.005):
    Obs: EE pose(6 Euler) + gripper(1) + obj_state(14) = 21D (1 T-block)
    Acts: MultiCategorical 6D × 11 bins → cuRobo IK → joint positions
    Role: explore and construct interesting goal configurations

Bob (PPOABC + GoalEncoder, abc_coef=0.5):
    Obs: robot(7) + obj_state(14) + goal_pose(6) + goal_dist(2) = 29D
    GoalEncoder: difference variant, K=8, max-pool
    Acts: same action space as Alice
    Role: reproduce the goal Alice left behind

Push-PPO Baseline (single-agent, no ASP no ABC):
    Obs: 29D flat vector
    Acts: MultiCategorical 6D × 21 bins → push primitive macro-action (115 substeps)
    Reward: dense improvement per push + completion bonus
```

**Kinematic pipeline** (cuRobo):
```
Policy bins → XYZ/RxRy delta → accumulated ee_target ± TCP offset → solve_batch(N) → joints
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `ent_coef` | 0.005 | Both agents (was 0.05 — Fix P23) |
| `optim_stepsize` | 3e-4 | Alice LR; cosine decay → 5e-5 |
| `cliprange` | 0.2 | PPO ε-clip |
| `gamma` | 0.998 | Discount |
| `lam` | 0.95 | GAE lambda |
| `abc_coef` | 0.5 | Fixed (paper Table 2) |
| `abc_traj_maxlen` | 500 | ABC buffer capacity |
| `abc_n_trajs` | 16 | Trajectories sampled per Bob update |
| `aux_coef` | 0.1 | GoalEncoder auxiliary distance loss |
| `alice_timesteps` | 100 | Steps per Alice phase |
| `bob_timesteps` | 100 | Steps per Bob phase (was 200 — Fix P28) |
| `max_goals_per_episode` | 5 | |
| `num_bins` | 11 | Per MultiCategorical dim |
| `lstm_hidden_size` | 256 | |
| Success threshold | 0.05 m / 0.2 rad | |

---

## Stack

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| cuRobo | 0.7.5 |
| PyTorch | 2.7.0+cu128 |
| Python | 3.11.5 |
| HPC container | `nvcr.io/nvidia/isaac-lab:2.3.0` (Apptainer) |
| HPC GPU | RTX Pro 6000 (96 GB) |

---

## Running

### Local

```bash
source /home/vlad/env_isaaclab/bin/activate
cd /home/vladi/IsaacLab/master_isaac

# Smoke test
python -m asyncDualPlayPPO.train_curobo --num_envs 16 --max_iterations 10 --headless

# Full local
python -m asyncDualPlayPPO.train_curobo --num_envs 64 --max_iterations 1000 \
    --exp_name curobo_local --headless

# Resume
python -m asyncDualPlayPPO.train_curobo --num_envs 64 --max_iterations 2000 \
    --chkpt_alice runs/curobo_local/alice/model_500.pt \
    --chkpt_bob   runs/curobo_local/bob/model_500.pt \
    --resume_iteration 500 --headless
```

### HPC

```bash
cd /home/<you>/master_isaac

# One-time setup (see implementations.md §5)
#   apptainer pull isaac-lab.sif
#   apptainer overlay create --size 8192 curobo_overlay.img
#   install cuRobo inside overlay

# Submit job (512 envs, auto-resume)
sbatch asyncDualPlayPPO/hpc/train_curobo.slurm

# Monitor
tail -f slurm-<JOBID>-curobo.out
```

### Diagnostics

```bash
bash asyncDualPlayPPO/diagnostics/run_diagnostics.sh          # all 4 tests
python -m asyncDualPlayPPO.train_curobo --test_reward_pipeline # Test 1
python -m asyncDualPlayPPO.train_curobo --alice_sandbox        # Test 2
python -m asyncDualPlayPPO.train_curobo --test_hparams          # hyperparameter audit
```

---

## File Structure

```
asyncDualPlayPPO/
├── train_curobo.py                  # Main training (cuRobo IK, ASP)
├── train_push.py                    # Push-PPO baseline
├── train_diffik.py / train.py       # Legacy controllers
├── cfg/
│   ├── ppo/ppo_continuous.yaml      # PPO + ABC hyperparameters
│   └── task/AsyncDualPlay.yaml      # Episode structure
├── algorithms/
│   ├── goal_encoder.py              # GoalEncoder φ-MLP
│   └── rl/ppo/
│       ├── module.py                # ActorCritic, PermInvEncoder
│       ├── module_push.py           # Flat MLP network (push baseline)
│       ├── ppo.py                   # Base PPO (Alice)
│       ├── ppo_abc.py               # PPOABC (Bob)
│       └── storage.py               # RolloutStorage + GPUDemonstrationBuffer
├── tasks/utils/
│   ├── wrapper.py                   # ASP env wrapper (phases, rewards)
│   ├── wrapper_push.py              # Push env wrapper
│   ├── observations.py / rewards.py # Observation & reward logic
│   └── terminations.py / events.py  # Termination & reset logic
├── utils/
│   ├── episode_manager.py           # Phase tracking, goal storage
│   ├── goal_validator.py            # Goal displacement validation
│   ├── historical_pool.py           # Policy snapshot ring buffer
│   └── profiler.py                  # Per-section timing
├── diagnostics/                     # 4-test diagnostic suite
├── tests/                           # Validation + cuRobo tests
├── hpc/                             # Slurm scripts
└── extras/                          # Log analysis / plotting
```

Full documentation: `implementations.md`, `net.md`.

---

## References

```bibtex
@article{plappert2021asymmetric,
  title={Asymmetric self-play for automatic goal discovery in robotic manipulation},
  author={Plappert, Matthias and Rajeswaran, Aravind and others},
  journal={arXiv preprint arXiv:2101.04882}, year={2021}
}
@article{sukhbaatar2018learning,
  title={Learning Goal Embeddings via Self-Play for Hierarchical RL},
  author={Sukhbaatar, Sainbayar and Lin, Zeming and others},
  journal={arXiv preprint arXiv:1811.09083}, year={2018}
}
```
