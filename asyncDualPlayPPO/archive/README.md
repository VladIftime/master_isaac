# Archive — legacy / superseded code

These files are **frozen for provenance** and are **not part of the published research
pipeline**. They are kept so the history described in `../implementations.md` remains
reproducible. They are not maintained and their import paths point at their original
locations (before they were moved here), so they are not expected to run as-is.

| File / dir | What it was | Status |
|---|---|---|
| `train_legacy.py` | RMPFlow ASP training entry point (was `train.py`) | superseded by `train_curobo.py` |
| `train.sh` | launcher for the RMPFlow ASP run | superseded |
| `run_interactive.sh` | interactive (non-SLURM) RMPFlow launcher | superseded |
| `train_diffik_legacy.py` | DifferentialIK ASP training entry point (was `train_diffik.py`) | superseded by `train_curobo.py` |
| `train_profile_diffik.slurm` | DiffIK profiling job | superseded |
| `train_push_sac_legacy.py` | Soft Actor-Critic push baseline (was `train_push_sac.py`) | abandoned, not in paper |
| `train_push_sac.slurm` | SLURM job for the SAC baseline | abandoned |
| `sac/` | skrl SAC actor/critic models (was `algorithms/rl/sac/`) | abandoned |
| `sac_push.yaml` | SAC hyperparameters (was `cfg/sac/`) | abandoned |
| `push_primitive_sac_env.py` | per-step SAC push env wrapper (was `tasks/utils/`) | abandoned |
| `action_push_continuous.py` | continuous push-action decode for SAC (was `tasks/utils/`) | abandoned |
| `train_push_rel.slurm`, `train_push_rel_full.slurm` | early object-relative Push-PPO experiments | superseded by the PBRS models |
| `train_push_asp_new_prim.slurm` | early Push-ASP primitive experiment | superseded |

The RMPFlow / DiffIK **environment configs** (`tasks/async_dual_play.py`,
`tasks/utils/reach_dual_arm_env_cfg.py`) remain in the live tree because they are still
imported by `train_curobo.py` (DiffIK cfg is its base) and by the interactive demo tests.
