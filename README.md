# Asymmetric Self-Play with Goal Encoders and PBRS for Robotic Pushing

Tabletop-pushing policies trained in Isaac Lab / Isaac Sim, combining Asymmetric
Self-Play (Plappert et al. 2021), Goal Embeddings (Sukhbaatar et al. 2018), and
Potential-Based Reward Shaping (Ng et al. 1999) on a UR5e arm with a push-primitive
macro-action and cuRobo IK.

The research line is the **PBRS push models (A–H)**. Two push baselines and a cuRobo-IK
reference are kept; older controllers live in [`asyncDualPlayPPO/archive/`](asyncDualPlayPPO/archive).
History: [`implementations.md`](implementations.md). Network/pipeline details: [`net.md`](net.md).

## Models

All models share the push action (4D × 21 bins → object-relative `(r, φ, length, θ)`,
~72 IK substeps per push) and cuRobo IK. They differ in reward, curriculum, object, and
whether ASP is used.

| Model | Script | Description |
|---|---|---|
| A | `train_a_pbrs_simple.py` | Single-agent PPO, PBRS, no curriculum |
| B | `train_b_pbrs_curriculum.py` | A + position→rotation curriculum |
| C | `train_c_pbrs_asp.py` | PBRS + ASP (Alice/Bob), GoalEncoder |
| D | `train_d_pbrs_asp_noge.py` | C with GoalEncoder ablated |
| E | `train_e_pbrs_asp_dpose.py` | C with SE(2) `d_pose` potential (T-block) |
| F | `train_f_pbrs_asp_disc.py` | E with rotationally-symmetric disc |
| G | `train_g_tasp_dpose.py` | E + time-based Alice reward (T-block) |
| H | `train_h_tasp_disc.py` | F + time-based Alice reward (disc) |

Each model has a matching `hpc/<script>.slurm`. Baselines: `train_push.py` (Push-PPO),
`train_push_asp.py` (Push-ASP). Reference cuRobo-IK ASP: `train_curobo.py`.

## Method

- **ASP**: Alice builds a goal; Bob reproduces it from a reset. Optional ABC (Bob clones
  Alice on failure, β=0.5) and a historical-policy pool.
- **GoalEncoder**: φ-MLP compresses `(goal, current)` poses into an 8D latent injected
  into Bob's trunk (difference variant). Model D ablates it.
- **PBRS**: dense reward `F = Φ(s′) − Φ(s)`, `Φ(s) = exp(−k·d²)`, `γ_shaping = 1.0`.
  Models A–D use separate position/rotation potentials; E–H use a single SE(2)
  `d_pose = √(dx² + dy² + L²·dθ²)` (`L = 0.07 m` T-block, `0` disc).
- **Time-based Alice** (G/H): `R_A = γ_sp · max(0, t_B − t_A)`, `γ_sp = 0.5`.

## Key hyperparameters

`num_envs 528`, `max_iterations 3000`, `k_p 30`, `k_r 5`, `w_pos = w_rot 10`,
`gamma_shaping 1.0`, success `d_pose < 0.055` (E/G) / `0.05` (F/H), `ent_coef 0.005`,
`gamma 0.998`, `lam 0.95`, `abc_coef 0.5`. Full configs: `cfg/ppo/ppo_continuous.yaml`,
`cfg/task/AsyncDualPlay.yaml`.

## Stack

Isaac Sim 5.1.0 · Isaac Lab 2.3.0 · cuRobo 0.7.5 · PyTorch 2.7.0+cu128 · Python 3.11.5.
HPC: Apptainer `nvcr.io/nvidia/isaac-lab:2.3.0`, RTX Pro 6000 (96 GB).

## Running

```bash
source .venv/bin/activate

# Local smoke test (Model C)
python -m asyncDualPlayPPO.train_c_pbrs_asp \
    --num_envs 16 --max_iterations 50 --exp_name pbrs_c_smoke --headless

# HPC (auto-resumes via SIGUSR1 chaining)
sbatch asyncDualPlayPPO/hpc/train_e_pbrs_asp_dpose.slurm
```

Runs write `latest_checkpoint.pt`, `latest_iter.txt`, and `model_best.pt`. Resume:

```bash
python -m asyncDualPlayPPO.train_c_pbrs_asp --exp_name <name> \
    --chkpt runs/<name>/agent/latest_checkpoint.pt \
    --resume_iteration $(cat runs/<name>/agent/latest_iter.txt) --headless
```

Validation (`tests/validation_configs.py` scenes, plots via `tests/plot_validation.py`):

```bash
python -m asyncDualPlayPPO.tests.validate_push     --chkpt runs/<name>/agent/model_best.pt --headless  # A, B, baselines
python -m asyncDualPlayPPO.tests.validate_push_asp --chkpt runs/<name>/bob/model_best.pt   --headless  # C–H
```

## Layout

```
asyncDualPlayPPO/
├── train_a_pbrs_simple.py … train_h_tasp_disc.py   # Models A–H
├── train_push.py / train_push_asp.py / train_curobo.py
├── cfg/{ppo,task}/                                  # hyperparameters
├── algorithms/{goal_encoder.py, rl/ppo/}           # GoalEncoder, PPO/PPOABC, networks
├── tasks/{push_task_curobo*.py, utils/}            # envs, wrappers, actions, reward_pbrs.py
├── utils/                                           # episode_manager, goal_validator, historical_pool
├── hpc/                                             # one SLURM per model + baselines
├── tests/ · extras/ · data_analysis/               # validation · log tools · thesis figures
└── archive/                                         # legacy controllers (RMPFlow, DiffIK, SAC)
```

## References

```bibtex
@article{plappert2021asymmetric,
  title={Asymmetric self-play for automatic goal discovery in robotic manipulation},
  author={Plappert, Matthias and others}, journal={arXiv:2101.04882}, year={2021}}
@article{sukhbaatar2018learning,
  title={Learning Goal Embeddings via Self-Play for Hierarchical RL},
  author={Sukhbaatar, Sainbayar and others}, journal={arXiv:1811.09083}, year={2018}}
@inproceedings{ng1999policy,
  title={Policy invariance under reward transformations: Theory and application to reward shaping},
  author={Ng, Andrew Y and Harada, Daishi and Russell, Stuart}, booktitle={ICML}, year={1999}}
```
