# HPC Setup & Run Guide — `train_curobo.py`

## Stack Versions

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 (commit `6c151ea`) |
| cuRobo | **0.7.x** (latest compatible with Isaac Lab 2.3 / CUDA 12.8) |
| PyTorch | 2.7.0+cu128 |
| Python | 3.11.5 |
| Container | `nvcr.io/nvidia/isaac-lab:2.3.0` (Apptainer) |
| GPU target | RTX Pro 6000 (96 GB VRAM) |

---

## Part 1 — One-Time Setup (do this once per HPC account)

### Step 1: Pull the Isaac Lab container

Run this from your project root on the login node. It takes ~20 minutes and produces a
single file `isaac-lab.sif` (~30 GB).

```bash
cd /home/<you>/master_isaac/asyncDualPlayPPO
apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0
```

> The slurm scripts expect `isaac-lab.sif` to exist in the directory where you call
> `sbatch` (the project root). Do not move it.

---

### Step 2: Build a cuRobo-patched overlay image

cuRobo is **not** included in the Isaac Lab 2.3.0 container. You must install it into
an Apptainer overlay (a writable layer on top of the base `.sif`).

#### 2a — Create the overlay

```bash
apptainer overlay create --size 8192 curobo_overlay.img
```

This creates an 8 GB writable ext3 image. 8 GB is sufficient; cuRobo + torch deps are
~3.5 GB.

#### 2b — Install cuRobo inside the overlay

```bash
apptainer exec --nv --overlay curobo_overlay.img:rw isaac-lab.sif bash
```

Inside the shell:

```bash
# Verify CUDA and torch first
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.7.0+cu128  12.8

# Clone cuRobo (tag 0.7.5 is the last release tested with Isaac Lab 2.x)
git clone https://github.com/NVlabs/curobo.git /tmp/curobo
cd /tmp/curobo
git checkout v0.7.5

# Install — skip docs, no pip upgrade (avoids breaking the container's pip)
pip install -e ".[no_dev]" --no-build-isolation

# Verify
python -c "import curobo; print(curobo.__version__)"
# Expected: 0.7.5

exit
```

> **Why `--no-build-isolation`?** The container's PyTorch is already compiled against
> CUDA 12.8. Build isolation would pull a second PyTorch, which mismatches and causes
> import errors at runtime.

> **Why tag `v0.7.5`?** cuRobo 0.7.x is the last series that uses
> `IKSolver` / `solve_batch` / `Pose` under the same API imported in
> `train_curobo.py`. The `0.8.x` series renamed several classes. If you use a newer
> tag, check that these imports still resolve:
> ```python
> from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
> from curobo.types.math import Pose as CuroboPose
> from curobo.types.robot import RobotConfig
> from curobo.types.base import TensorDeviceType
> from curobo.util_file import get_robot_configs_path, join_path, load_yaml
> ```

#### 2c — Verify the overlay works with Isaac Lab

```bash
apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    python -c "import curobo; import isaaclab; print('OK')"
```

Both imports must succeed with no errors.

---

### Step 3: Update the slurm scripts to use the overlay

The current `hpc/train_curobo.slurm` and `hpc/train_curobo_profile.slurm` call
`apptainer exec --nv ...` without `--overlay`. Add the overlay flag to both files.

In `hpc/train_curobo.slurm`, change:

```bash
apptainer exec --nv \
```

to:

```bash
apptainer exec --nv --overlay "$PROJECT_ROOT/curobo_overlay.img":ro \
```

Do the same in `hpc/train_curobo_profile.slurm`.

> Use `:ro` (read-only) at job runtime — only the one-time install step needs `:rw`.

---

### Step 4: Set up the cache directories

Isaac Sim needs writable cache folders. Create them once:

```bash
mkdir -p .cache \
         .isaac_cache/kit/data \
         .isaac_cache/kit/cache \
         .isaac_cache/kit/logs
```

These are bind-mounted by the slurm scripts into the container automatically.

---

### Step 5: Register the project as an Isaac Lab extension

The slurm scripts bind-mount the project into the container at
`/workspace/isaaclab/user_project/asyncDualPlayPPO`. Isaac Lab must be able to import
the project's tasks package. This is handled by the `asyncDualPlayPPO` package
structure (the top-level `__init__.py` is sufficient — no extra registration needed).

Verify locally:

```bash
apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train_curobo.py \
    --num_envs 16 --max_iterations 3 --headless
```

If this runs 3 iterations without error, setup is complete.

---

## Part 2 — Running Training

### Quick smoke test (interactive node)

```bash
srun --partition=gpu --gpus-per-node=rtx_pro_6000:1 --time=00:15:00 --pty bash

cd /home/<you>/master_isaac/asyncDualPlayPPO

apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    /workspace/isaaclab/isaaclab.sh -p train_curobo.py \
    --num_envs 64 --max_iterations 10 --headless --exp_name smoke_test
```

Watch for:
- `[cuRobo] IK solver created.` — solver initialised successfully
- `IK fail %` per iteration — should drop below 30% after ~50 iterations
- No `CUDA error` or `out of memory` messages

---

### Full production run

```bash
cd /home/<you>/master_isaac/asyncDualPlayPPO
sbatch hpc/train_curobo.slurm
```

Defaults: 4096 envs, 100 000 iterations, RTX Pro 6000, auto-resume on time limit.

Checkpoints are written every 10 iterations to `runs/hpc_curobo_4096env_1obj/`.
On SIGUSR1 (2 min before wall-time), the job syncs to NFS and resubmits itself.

Monitor:

```bash
tail -f slurm-<JOBID>-curobo.out
```

Key lines to watch per iteration:

```
[Iter N] SR=0.12 | IK fail%=18.3 | Alice valid=47/64 | avg XY=0.142m
```

---

### Profiling run (3 iterations, 2048 envs)

```bash
sbatch hpc/train_curobo_profile.slurm
```

Output includes a profiler table at the end:

```
Section          |  calls |   total(s) |   mean(ms) |    max(ms)
curobo_ik        |      3 |      0.042 |      14.00 |      16.2
orient_accum     |      3 |      0.001 |       0.33 |       0.4
env_step         |      3 |      1.821 |     607.00 |     621.0
alice_act        |      3 |      0.003 |       1.00 |       1.1
bob_act          |      3 |      0.003 |       0.98 |       1.0
```

`curobo_ik` should be well under 10% of `env_step` time.

---

### Resuming from checkpoint

```bash
sbatch hpc/train_curobo.slurm   # auto-detects latest checkpoint in runs/ and resumes
```

Or manually:

```bash
apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    /workspace/isaaclab/isaaclab.sh -p train_curobo.py \
    --num_envs 4096 \
    --chkpt_alice runs/hpc_curobo_4096env_1obj/alice/model_500.pt \
    --chkpt_bob   runs/hpc_curobo_4096env_1obj/bob/model_500.pt \
    --resume_iteration 500 \
    --headless
```

---

## Part 3 — Troubleshooting

### `ImportError: No module named 'curobo'`

The overlay was not passed to `apptainer exec`. Add `--overlay curobo_overlay.img:ro`
to the `apptainer exec` call in the slurm script (see Step 3).

---

### `CUDA error: device-side assert triggered`

Usually a shape mismatch in the IK batch. Check that `num_envs` matches what cuRobo
was initialised with:

```python
IKSolverConfig(..., num_seeds=1, batch_size=num_envs)
```

If you changed `--num_envs` without recreating the solver, this error appears.

---

### IK fail rate stuck above 50%

The EE target is outside the reachable workspace. Tighten the workspace bounds in
`train_curobo.py`:

```python
_WS_XY  = 0.5   # reduce if fail rate is high
_WS_Z   = (0.00, 0.60)
```

Or reduce `max_delta_m` in the policy config YAML to make per-step moves smaller.

---

### `isaac-lab.sif` not found

The slurm scripts expect the `.sif` file in the directory where `sbatch` is called
(the project root). Either `cd` to the project root before submitting, or set
`SIF_IMAGE` to an absolute path inside the slurm script.

---

### Container rebuild (if overlay approach is unavailable)

If your HPC sysadmin cannot provide Apptainer overlay support, build a new `.sif`
with cuRobo baked in using a definition file:

```singularity
Bootstrap: docker
From: nvcr.io/nvidia/isaac-lab:2.3.0

%post
    git clone https://github.com/NVlabs/curobo.git /tmp/curobo
    cd /tmp/curobo && git checkout v0.7.5
    pip install -e ".[no_dev]" --no-build-isolation
    rm -rf /tmp/curobo
```

```bash
apptainer build isaac-lab-curobo.sif isaac-lab-curobo.def
```

Then update `SIF_IMAGE="isaac-lab-curobo.sif"` in both slurm scripts.
