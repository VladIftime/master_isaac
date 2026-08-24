# GIF / Video Suggestions for Presentation

Each suggestion targets a specific slide and visualizes an aspect that static plots cannot capture.  
GIFs should be 3–8 seconds, looped, with the relevant Isaac Sim viewport camera angle.

---

## Slide 2 (MDP Formulation) — Complete Push Sequence

**What:** Top-down view of a single push cycle from start to finish.

**Sequence:**
1. Object spawns at random SE(2) pose on table. Goal ghost (green translucent T-block) appears
2. Robot arm approaches at (r, φ) from object center — approach trajectory shown as dashed line
3. Gripper contacts object. Push phase: arm drives through (ℓ, θ) over 72 substeps
4. Object slides to final pose. Red arrow from start→end of push overlaid
5. Post-push: arm retracts to home. Error metrics appear (position/rotation distance to goal)

**Record from:** Isaac Sim viewport with `--num_envs=1`, debug markers (Fix P35) enabled  
**Duration:** ~4 seconds per push × 3 consecutive pushes = 12 seconds  
**Slide usage:** Replace or accompany the static placeholder on the right side

---

## Slide 3 (Push Mechanics) — T-block vs Disc Behaviour Difference

**What:** Split-screen comparison showing why the SE(2) coupling differs by object.

**Left half — T-block:** Push at 45° offset from center of mass. Object translates forward AND rotates clockwise. Superimpose green arrow (translation component) and red arc (rotation component). Show that a single push changes both pose components simultaneously.

**Right half — Disc:** Same push direction, same contact point relative to object center. Object translates forward with ZERO rotation. Red arc absent.

**Overlay:** Text annotations showing the characteristic length `L = 0.07 m` for T-block and `L = 0.0 m` for disc, plus the d_pose formula decomposing into translation and rotation components.

**Record from:** Two side-by-side Isaac Sim instances or sequential recording  
**Duration:** 5-6 seconds  
**Slide usage:** Top-right quadrant of slide 3, replacing or supplementing the gradient comparison plot

---

## Slide 6 (Model A) — Three Validation Runs (Easy → Hard)

**What:** Montage of 3 successful validation episodes showing the same model across difficulty levels.

**Scene 1 (Easy — Disc):** Disc spawns near center, goal 15cm away. Robot does 2 pushes. Goal ghost turns solid green on success. **2 seconds**

**Scene 2 (Medium — T-block pos-only):** T-block spawns at random yaw, goal 25cm away, same orientation. Robot does 5 pushes, alternating translation pushes with small rotation corrections. **4 seconds**

**Scene 3 (Hard — T-block pos+rot):** T-block spawns at 180° offset from goal. Robot does 9 pushes — first 5 correct orientation via rotation pushes, last 4 translate to position. Goal ghost turns solid green. **6 seconds**

**Overlay:** Running counter showing `pushes: 5/15`, distance-to-goal meter shrinking.

**Record from:** `validate_push.py --num_envs=1` with scene indices 1, 8, 28  
**Duration:** ~12 seconds total (play as auto-looping GIF, pause on success freeze-frame)  
**Slide usage:** Right column of slide 6, replacing the static layout plot

---

## Slide 8 (ASP Models) — Alice → Bob Phase Transition

**What:** Split-screen showing one complete ASP two-phase cycle in Model C.

**Top half — Alice Phase:**
1. Alice spawns. Table shows object at default pose
2. Alice executes 15 pushes, moving object to a new SE(2) pose
3. Green "GOAL SET" text appears. Alice's phase ends. Goal ghost snaps to Alice's final object pose

**Bottom half — Bob Phase:**
1. Objects reset to default spawn pose. Goal ghost remains at Alice's final pose
2. Bob spawns. Tries to reproduce Alice's configuration
3. After 50 pushes (or early termination), result: "BOB FAILED" in red OR "BOB SUCCEEDED" in green
4. Alice receives outcome reward (+5 for fail, -1 for success)

**Overlay:** Running push counters for both agents, reward values updating at phase end

**Record from:** `train_push_pbrs_asp.py --num_envs=1` with debug markers  
**Duration:** ~8 seconds  
**Slide usage:** Reference during Q&A (appendix) or as a supplementary animation when discussing ASP architecture

---

## Slide 11 (Computational Overhead) — ASP vs DirectRLEnv Throughput Comparison

**What:** Side-by-side accelerated playback showing the throughput difference.

**Left — ASP (528 envs):** Grid of 528 environments. Slow, staggered phase transitions visible. Alice/Bob alternating. Lots of idle time during ABC buffer updates and PPO optimization steps. **Shows ~1-2 iterations per second.**

**Right — DirectRLEnv (4096 envs, Table Tennis):** Grid of 4096 environments. Rapid, synchronized push cycles. No idle time — pure step loop with zero manager overhead. **Shows ~20 iterations per second.**

**Overlay:** Running iteration counter and wall-clock timer on both sides. At 30 real seconds, ASP shows 45 iterations, DirectRLEnv shows 600 iterations.

**Record from:** Screen capture of both training runs in headless mode with iteration logging  
**Duration:** 8-10 seconds (sped up 3× from real-time)  
**Slide usage:** Bottom of slide 11, replacing the static architecture text

---

## Slide 14 (Future Work) — Physical UR5e Sim-to-Real Clip

**What:** Short clip of the physical UR5e robot performing a push on a real T-block.

**Sequence:**
1. Camera shows UR5e + table + T-block (physical setup)
2. Robot arm moves to approach position, pushes object
3. Object slides ~10cm on table. Overhead camera confirms pose change
4. Split-screen: Isaac Sim rendering of the same push on the left, real robot on the right, showing visual sim-to-real correspondence

**Record from:** Physical lab camera + Isaac Sim playback of the same policy  
**Duration:** 5-6 seconds  
**Slide usage:** Bottom-right corner of slide 14 as future-work teaser

---

## Technical Recording Notes

| Parameter | Value |
|-----------|-------|
| **Recording tool** | Isaac Sim built-in video recorder or OBS screen capture |
| **Resolution** | 1920×1080 (16:9, matching slide aspect ratio) |
| **FPS** | 30 (smooth for physics demos) |
| **Output format** | GIF (looped, ≤15 MB each) or MP4 (for embedded video in PDF) |
| **Overlay text** | Add in post-processing (OBS or ffmpeg drawtext filter) |
| **Camera position** | Top-down (45°) for push tasks; side view for UR5e robot |

**Post-processing command (GIF from MP4):**
```bash
ffmpeg -i recording.mp4 -vf "fps=15,scale=960:-1" -loop 0 output.gif
```

**Embedded video option (beamer + okular/adobe):**
```bash
ffmpeg -i recording.mp4 -vf "scale=960:-1" -c:v libx264 -preset slow -crf 22 output.mp4
```

In the `.tex` file, replace placeholder text with:
```latex
% For GIF (most reliable across PDF viewers):
\includegraphics[width=\textwidth]{figures/push_sequence.gif}

% For embedded video (okular / Adobe Acrobat):
\usepackage{multimedia}
\movie[width=\textwidth,autostart,loop]{}{figures/push_sequence.mp4}
```

---

## Priority Order

If time is limited, record in this order:
1. Slide 2 — Complete Push Sequence (most fundamental, appears early)
2. Slide 6 — Validation Runs (strongest empirical result)
3. Slide 3 — T-block vs Disc (motivates SE(2) metric)
4. Slide 11 — Throughput Comparison (powerful architectural argument)
5. Slide 8 — Alice→Bob Transition (for Q&A)
6. Slide 14 — Sim-to-Real (future work teaser)

---

## Automated Recording Script — `tests/record_push_video.py`

A purpose-built fork of `validate_push.py` that records **top-down video** of a trained
policy directly (no Omniverse USD round-trip needed).

**Camera:** a `CameraCfg` is added to the scene at the **centre of the table**
(world XY `(0, 0.40)`, per Fix P78) and repositioned after build to the **height of the
robot's highest link** (`robot.data.body_pos_w[..., 2].max() + --cam-margin`), looking
straight down (OpenGL convention, identity quaternion → forward `-Z`, up `+Y`).

**Encoding:** frames are grabbed from `top_camera.data.output["rgb"]` each waypoint
substep (every `--capture-every`, default 3) and written to MP4 via `imageio` — which
ships its **own bundled ffmpeg** (`imageio_ffmpeg`), so no system ffmpeg is required.
Each scene is retried up to `--max_attempts` (default 3) and the first **successful**
rollout is kept.

**Run (Model A — the simple PBRS single-agent checkpoint):**
```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd /home/vladi/IsaacLab/master_isaac
python -m asyncDualPlayPPO.tests.record_push_video \
    --chkpt runs/ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/agent/model_best_simp.pt \
    --rel-obs --rel-act --headless --enable_cameras \
    --scenes 11,13,21 --fps 15 --width 1920 --height 1080
```
* `--enable_cameras` is **required** (offscreen render, works headless).
* Model A is the T-block single-agent policy → use `--rel-obs --rel-act`.
* **T-block only**, exactly like `validate_push.py`: the scene uses the single
  `target_object` (T-block). A `disc_pos` config would just run the T-block in
  position-only mode (rotation ignored) — there is no genuine disc object, since Model A
  was never trained on one. Decode uses Model A's training default (`min_r=0.02, max_r=0.08`).
* Defaults are three genuine T-block validation scenes: `11` (E_Forward, pos-only),
  `13` (E_Left, pos-only, lateral push), `21` (E_Diag, pos+rot).

**Outputs** (land in `presentation/figures/`):
`rec_push_s11.mp4`, `rec_push_s13.mp4`, `rec_push_s21.mp4` (+ matching `_key.png`).

**Slide wiring:** the *Model A — Validation Runs (overhead)* slide shows all three keyframes
3-up (`rec_push_s11/s13/s21_key.png`) with `\playclip` links, and *Model A — Validation
Results* shows `rec_push_s21_key.png` as a hero — all via `\mediabox`, which renders a
placeholder box until the file exists. Drop the recorded files in `figures/` and recompile;
the keyframes appear automatically, no `.tex` edits needed.

### Converting the existing ASP MP4s (already done)
The two pre-existing clips in `tests/videos/` were converted to keyframes + GIFs with
the same venv (cv2 + imageio):
`asp_random.mp4`/`asp_random_encoder.mp4` → `figures/asp_random{,_encoder}_key.png` +
`figures/asp_random{,_encoder}.gif`. `asp_random_key.png` is wired into the
*Models C–D — ASP Architecture* slide.

