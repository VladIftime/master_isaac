# Cluster Training Setup — Attempts, Failures & Recommendations

**Date**: 2026-05-27
**Cluster**: fse-2a100-1, fse-4a100-1, fse-4a100-2 (RUG LWP)
**GPUs**: 4x NVIDIA A100-SXM4-80GB with MIG partitioning
**Goal**: Run `run_push.sh` and `run_curobo.sh` on the cluster

---

## Environment

| Component | Cluster (fse nodes) | Project Requirement |
|-----------|---------------------|---------------------|
| GPU | A100-SXM4-80GB | RTX Pro 6000 (96 GB) |
| CUDA driver | System driver | CUDA 12.8 (container) |
| Isaac Sim | — | 5.1.0 |
| Isaac Lab | — | 2.3.0 |
| cuRobo | — | 0.7.5 |
| PyTorch | — | 2.7.0+cu128 |
| Python | — | 3.11.5 |
| Container runtime | Apptainer (rootless) | Apptainer `.sif` |
| Docker | Rootless docker available | — |

---

## Attempt 1: Use pre-built overlay (`curobo_overlay.img` + `isaac-lab.sif`)

**Goal**: Use existing overlay image (built on original machine) directly.

**What we tried**:
1. Pull `isaac-lab.sif` via `apptainer pull docker://nvcr.io/nvidia/isaac-lab:2.3.0`
2. Use existing `curobo_overlay.img` (8 GB) as read-only overlay

**Failures**:

### 1a. Local disk full (`/dev/sda1` at 100%)
```
FATAL: short write: no space left on device
```
- `/tmp` and `/scratch` are on the local SSD (`/dev/sda1`, 434 GB, 100% full)
- Apptainer uses `/tmp` for build temp and session dirs by default
- **Fix attempted**: Point `APPTAINER_TMPDIR` to home directory (NFS, 731 GB free)
- **Result**: Pull succeeded on `fse-2a100-1` (different node with more space)

### 1b. Apptainer session directory permission denied
```
mount hook function failure: setup of overlay upper dir failed:
/var/lib/apptainer/mnt/session/overlay-images/0/upper is not writable
```
- `--fakeroot` with overlay tried to write to local disk (full/permission denied)
- `--fakeroot` needed for `apt install` inside container
- **Workaround**: Use overlay in `:rw` mode without `--fakeroot` (works for file writes but not `apt`)

### 1c. cuRobo not installed in overlay
- The overlay existed but cuRobo was never properly installed in it
- Entry: `apptainer exec --nv --overlay curobo_overlay.img:rw isaac-lab.sif bash`
- Python path needed: `source /workspace/isaaclab/_isaac_sim/setup_python_env.sh && export PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:$PATH"`
- Torch version confirmed: `2.7.0+cu128`
- cuRobo import failed: `ModuleNotFoundError: No module named 'curobo'`

---

## Attempt 2: Install cuRobo inside overlay manually

**Goal**: Use pip to install cuRobo v0.7.5 inside the writable overlay.

**What we tried**:
1. Enter container with writable overlay (no fakeroot)
2. Install cuRobo dependencies and cuRobo itself via pip

**Failures**:

### 2a. Git clone hangs (no outbound HTTPS)
```
git clone https://github.com/NVlabs/curobo.git
→ hangs indefinitely
```
- No outbound HTTPS access from inside the container (possibly cluster network restrictions)
- Same hang on the host outside the container
- **Workaround**: Bind-mount local `curobo_source/` into container

### 2b. Missing nvcc (CUDA compiler)
```
nvcc not found
```
- `isaac-lab.sif` only has CUDA runtime, not the compiler toolkit
- **Fix attempted**: `pip install nvidia-cuda-nvcc-cu12` — installed successfully but nvcc binary at non-standard path
- nvcc installed at `/usr/local/cuda-12.8/cuda_nvcc/bin/nvcc`
- Had to symlink into `CUDA_HOME/bin/`

### 2c. Missing CUDA development headers (chain of missing includes)
```
fatal error: cuda_runtime_api.h: No such file or directory
→ fixed with symlink
fatal error: crt/host_defines.h: No such file or directory
→ fixed with symlink from cuda_nvcc
fatal error: nv/target: No such file or directory
→ fixed with symlink from cuda_cccl
fatal error: thrust/complex.h: No such file or directory
→ fixed with full include merge
```

- The pip-installed CUDA packages (`nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvcc-cu12`) scatter headers across subdirectories
- `CUDA_HOME/include/` only has partial headers
- **Fix attempted**: Merge all include dirs from all CUDA sub-packages into `CUDA_HOME/include/`
- **Result**: All headers resolved, but next failure appeared

### 2d. C++ ABI incompatibility between bundled Torch headers and system compiler
```
error: command '/usr/bin/g++' failed with exit code 1
```
- Isaac Sim's bundled PyTorch C++ headers (`torch/include/torch/csrc/api/include/torch/nn/...`)
  are incompatible with the container's system `g++` compiler
- Multiple template instantiation errors (ConvTransposeNd, cloneable, etc.)
- **Root cause**: The Isaac Lab container ships precompiled PyTorch binaries but the C++
  header API signatures don't match the system toolchain
- **Result**: Cannot build cuRobo from source inside the container with the bundled pip-installed CUDA toolkit

---

## Attempt 3: Build a new SIF container with cuRobo baked in (`curobo.def`)

**Goal**: Use `apptainer build --fakeroot` to create `isaac-lab-curobo.sif` from definition file.

**Definition file** (`curobo.def` on cluster):
```singularity
Bootstrap: docker
From: nvcr.io/nvidia/isaac-lab:2.3.0
%post
    apt-get update
    apt-get install -y gnupg wget
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-compiler-12-8 cuda-cudart-dev-12-8 ninja-build
    export CUDA_HOME=/usr/local/cuda-12.8
    nvcc --version
    git clone https://github.com/NVlabs/curobo.git /tmp/curobo
    cd /tmp/curobo && git checkout v0.7.5
    /isaac-sim/python.sh -m pip install -e ".[no_dev]" --no-build-isolation
    /isaac-sim/python.sh -c "import curobo; print('cuRobo OK')"
    rm -rf /tmp/curobo
    apt-get clean
%environment
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
```

**Status**: User indicated this build succeeded on one attempt. However, the `git clone` in `%post` will hang on the cluster (same network issue). A modified `.def` using local `curobo_source/` is needed.

**Potential issues**:
- `--fakeroot` may fail due to local disk being full (session dir)
- Set `APPTAINER_TMPDIR=$HOME/tmp` before building
- Need to replace `git clone` line with `cp -r` from local source

---

## Attempt 4: Docker-based approach

**User message**: "I have managed to build curobo on a Docker container"

### 4a. Disk quota exceeded
```
docker: Error response from daemon: disk quota exceeded.
```
- Docker rootless uses `/tmp` and `/scratch` which are on the 100% full local disk
- Cannot pull images or run containers that write to disk

### 4b. USDRT Population errors (at runtime)
```
[usdrt.population.plugin] USDRT Population failed to load IFabricHierarchy interface
```
- This is an Isaac Sim Fabric/USD runtime error
- Spam loop (continuous repeat), never progresses past this point
- Indicates Isaac Sim cannot initialize its simulation backend
- **Likely cause**: Missing Vulkan/EGL/GL libraries or GPU driver incompatibility with the container

---

## Root Causes — Why These All Fail

### 1. Local disk full (operational)
`/dev/sda1` (434 GB) is at 100% on the cluster nodes. Both `/tmp` and `/scratch`
live on this disk. Apptainer needs writable temp space for:
- Pulling containers (unpacking layers)
- Session directories (overlay upper dirs)
- Docker temp space

**Fix**: Use NFS home directory (`$HOME/tmp`) for all temp dirs. Requires:
```bash
export APPTAINER_TMPDIR=$HOME/tmp
export APPTAINER_CACHEDIR=$HOME/.apptainer-cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
```

### 2. No network access from container/overlay
- `git clone` and some HTTPS downloads hang inside the container
- May work on host but not through apptainer's network namespace
- **Fix**: Use local copies of source code, bind-mount as needed

### 3. Incompatible CUDA toolkit packaging (pip vs apt)
- Pip-packaged CUDA (`nvidia-cuda-nvcc-cu12`, `nvidia-cuda-runtime-cu12`) scatters
  components across sub-package directories
- Headers not unified under a single `CUDA_HOME/include/`
- System g++ incompatible with Isaac Sim's bundled Torch C++ headers
- **Fix**: Must use `apt install cuda-compiler-12-8 cuda-cudart-dev-12-8` which
  properly installs everything under `/usr/local/cuda-12.8`

### 4. Driver/CUDA version mismatch
- Cluster GPUs use one NVIDIA driver version
- Isaac Lab container requires CUDA 12.8 toolkit and compatible driver
- USDRT Population errors suggest the GPU driver doesn't support the Fabric
  runtime interface needed by Isaac Sim 5.1.0
- **Need to verify**: `nvidia-smi` driver version on cluster vs Isaac Sim 5.1.0 requirements
- Isaac Sim 5.1.0 requires driver >= 550.x (CUDA 12.4+)

### 5. No fakeroot for overlay (apptainer limitation)
- Rootless apptainer can write files to overlay (`:rw`) but can't use `apt` or `dpkg`
- `--fakeroot` fails because session directories are on the full local disk
- **Workaround**: Build a new SIF instead of modifying the overlay

---

## Recommended Path Forward

### Option A: Build SIF on a machine with working network + disk space
1. Build `isaac-lab-curobo.sif` on a different machine (e.g., your local workstation
   or a Habrok node with GPU)
2. Copy the `.sif` file to the cluster via `rsync` or `scp`
3. This is ~35 GB, expect 20-40 min transfer
4. Then update run scripts to use `isaac-lab-curobo.sif` without the overlay

### Option B: Fix the docker disk quota + build with docker
1. Clean up `/scratch/s3426394/`, remove unused docker images/containers
2. Or ask admin to clean `/dev/sda1` on the nodes
3. Build a Docker image with cuRobo baked in, then convert to apptainer:
   ```bash
   docker build -t isaac-lab-curobo -f curobo_source/docker/isaac_sim.dockerfile .
   apptainer build isaac-lab-curobo.sif docker-daemon://isaac-lab-curobo:latest
   ```

### Option C: Use Habrok HPC (if available)
The project was originally developed on Habrok with SLURM. The HPC scripts
(`train_curobo.slurm`, etc.) are already set up for that environment with
RTX Pro 6000 GPUs, working disk space, and network access.

### Option D: Request a /project directory
As suggested in the cluster welcome message, request a `/project` directory
for more reliable storage than the local disk or home directory.

---

## What We Know Works
- Apptainer can run Isaac Lab container with overlay on `fse-2a100-1`
- MIG UUIDs in run scripts match the cluster GPUs
- Python + torch import works inside the container when PATH is set correctly
- `curobo_overlay.img` is writable in `:rw` mode (just can't install via apt)
- Home directory has sufficient space (731 GB free of 4 TB)

## What Needs Verification
- `nvidia-smi` driver version on cluster nodes
- Whether Isaac Sim 5.1.0 Fabric runtime works on A100 with cluster driver
- Whether Vulkan libraries are available (needed by Isaac Sim)
