# Network Architecture: Paper vs. Current Implementation

---

## Original OpenAI ASP Network
<!-- add reference to the paper -->


### Input Vectors

```
robot_joint_position = [joint1, joint2, joint3, joint4, joint5, joint6]
                         6 arm joint angles (radians)

gripper_position     = [tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, finger_position]
                         EE Cartesian pose (metres + radians) + finger opening

object_state         = [pos_x, pos_y, pos_z,
                         rot_roll, rot_pitch, rot_yaw,          ← Euler angles (3D)
                         vel_x, vel_y, vel_z,
                         rotvel_x, rotvel_y, rotvel_z,
                         gripper_distance, gripper_contact]     = 14D per object

goal_state (Bob only) = [desired_pos_x, desired_pos_y, desired_pos_z,
                          desired_rot_roll, desired_rot_pitch, desired_rot_yaw,  ← Euler (3D)
                          relative_distance]                    = 7D per object
```

### Forward Pass

```
robot_joint_position (6D)   →  Embedding Linear(6→256)   → LayerNorm(256)  ─┐
gripper_position     (7D)   →  Embedding Linear(7→256)   → LayerNorm(256)  ─┤
                                                                              Sum*
object_state (14D × N)      →  PI Embedding:                                 │
                                 shared Linear(14→512) → ReLU                │
                                 shared Linear(512→512) → ReLU               │
                                 sum-pool over N objects                      │
                                 LayerNorm(512)                             ──┤
                                                                              Sum*
goal_state (7D × N) [Bob]  →   PI Embedding (same structure as object)     ──┘

Sum* (256+256+512[+512 Bob]) → ReLU → MLP → LSTM → Actor head / Value head
```

---

## Current Implementation

### Input Vectors

```
ee_pose         = [ee_x, ee_y, ee_z, roll, pitch, yaw]
                    EE position (metres, env-local) + ZYX Euler angles
                    ─── Euler matches paper

gripper_state   = [finger_joint_angle]
                    Raw finger joint position (radians)

     robot_state (7D) = concat(ee_pose(6), gripper_state(1))
     ─── No joint angles: RMPFlow handles IK internally ───

object_state    = [pos_x, pos_y, pos_z,
                    roll, pitch, yaw,                          ← ZYX Euler (3D, not quat)
                    vel_x, vel_y, vel_z,
                    angvel_x, angvel_y, angvel_z,
                    gripper_distance, gripper_contact]         = 14D per object

goal_state (Bob only)
                = [desired_pos_x, desired_pos_y, desired_pos_z,
                    desired_roll, desired_pitch, desired_yaw]  = 6D per object (Euler, not quat)

goal_distance (Bob only)
                = [pos_dist,   ← L2(current_pos, goal_pos) in metres
                    rot_dist]  ← max |Euler diff| with wraparound, range [0, π]
                                                                             = 2D per object
```

**Assembled observation vectors:**

```
Alice obs = [robot_state(7) | obj1_state(14) | obj2_state(14)]          = 35D
Bob obs   = [robot_state(7) | obj1_state(14) | obj2_state(14)
                            | obj1_goal(6)   | obj2_goal(6)
                            | obj1_dist(2)   | obj2_dist(2)]             = 51D
```

### Forward Pass (Alice)

```
robot_state (7D) ───────────────────────────────────────────────────────────┐
                                                                             │
object_state (14D × 2 objects) →  PI Embedding (PermInvEncoder):           │ concat
                                     shared Linear(14→512) → LayerNorm → ReLU
                                     shared Linear(512→512) → LayerNorm → ReLU
                                     max-pool over 2 objects               │
                                     LayerNorm(512)  ← post-pool norm      │
                                                                            ─┘
                          concat [robot(7) | PI_pooled(512)] = 519D
                                     ↓
              actor_trunk: Linear(519→512) → ReLU → Linear(512→256) → ReLU → Linear(256→128)
                                     ↓
                          LSTMCell(128→256)
                                     ↓
              ┌────────────────────────────────┐
              ▼                                ▼
       Actor head                       Value head
       Linear(256→6×11=66)              Linear(35→512) → ReLU
       MultiCategorical                 → Linear(512→256) → ReLU
       (6 action dims × 11 bins)        → Linear(256→128) → ReLU → Linear(128→1)
```

**Action bins:**
```
dims 0-2: XYZ Cartesian delta   → (bin − 5) / 5 × max_delta  (default 0.05 m)
dims 3-4: Rx, Ry rotation delta → (bin − 5) / 5 × 0.5 rad
dim  5:   Gripper               → sign(normalized) ∈ {−1, 0, +1}
```

### Forward Pass (Bob)

```
robot_state (7D) ───────────────────────────────────────────────────────────┐
                                                                             │
GoalEncoder (φ MLP, shared across objects):                                 │
  input per object: current_pose(6D) + goal_pose(6D)                       │
  φ: Linear(6→64) → Tanh → Linear(64→K=8)   ← no final activation        │
  g_i = φ(goal_i) − φ(current_i)  [difference variant]                    │
  g_pooled = sum-pool(g_0, g_1)              → 8D  (additive injection)    │
                                                                             │
PI Embedding (PermInvEncoder):                                              │
  input: ONLY obj_states (14D each) — goal enters via additive injection   │ concat
  shared Linear(14→512) → LayerNorm → ReLU                                 │
  shared Linear(512→512) → LayerNorm → ReLU                                │
  max-pool over 2 objects                                                   │
  LayerNorm(512)  ← post-pool norm                                         │
                                                                            ─┘
                    concat [robot(7) | PI_pooled(512)] = 519D
                                     ↓
  h₁ = Linear(519→512)(enc) + Linear(8→512, no bias)(g_pooled)  ← additive goal injection
  h₁ = ReLU(LayerNorm(h₁))
                                     ↓
       actor_trunk_rest: Linear(512→256) → ReLU → Linear(256→128)
                                     ↓
                          LSTMCell(128→256)
                                     ↓
              ┌────────────────────────────────┐
              ▼                                ▼
       Actor head                       Value head
       Linear(256→6×11=66)              Linear(51→512) → ReLU
       MultiCategorical                 → Linear(512→256) → ReLU
                                        → Linear(256→128) → ReLU → Linear(128→1)
                                        (full raw obs, no goal encoder bottleneck)
```

---

## Differences: Paper vs. Current

| | Paper | Current |
|--|-------|---------|
| **Robot arm state** | Joint angles (6D) | ❌ Removed — RMPFlow handles IK |
| **Gripper / EE state** | EE Cartesian pose + finger (7D) | EE Euler pose + finger (7D) |
| **Object rotation** | Euler angles (3D) | **Euler angles (3D)** — matches paper |
| **Object state dims** | 14D per object | **14D** per object |
| **Goal rotation** | Euler angles (3D) | **Euler angles (3D)** — matches paper |
| **Goal state dims** | 7D (3D pos + 3D euler + 1 scalar dist) | **6D** goal pose + **2D** dist (separate term) |
| **Goal encoding** | Raw PI embedding on goal states | **GoalEncoder → K=8 latent** per object |
| **GoalEncoder φ activation** | — | **Tanh** (paper §2.4) |
| **GoalEncoder input** | — | **6D Euler pose** (pos3 + euler3) |
| **GoalEncoder pooling** | — | **Sum-pool** (g = Σ g_i; "AND" semantics) |
| **Additive goal injection** | ❌ | ✅ `h = ReLU(LN(W·enc + Wg·g))` |
| **PI encoder per-obj input** | 14D obj state | **14D obj state only** (goal separated out) |
| **Pooling (PI encoder)** | Sum-pool | **Max-pool** (more robust, standard DeepSets) |
| **Post-pool norm** | LayerNorm ✅ | LayerNorm ✅ |
| **Alice obs dim** | — | **35D** |
| **Bob obs dim** | — | **51D** |
| **Actor trunk** | MLP → LSTM | **Linear(519→512)→ReLU→(256)→(128) → LSTMCell(128→256)** |
| **Action space** | Continuous Gaussian | **MultiCategorical: 6 dims × 11 bins** |

---

## Rotation Representation Note

The current implementation uses ZYX Euler angles (roll, pitch, yaw) at observation time,
matching the paper's Appendix A.2 ("three Euler angles on three dimensions").
Quaternions are produced by IsaacSim but converted in `observations.py` before the
policy ever sees them.

The GoalEncoder's φ MLP therefore receives 6D inputs (pos3 + euler3) and computes
difference embeddings `φ(goal) − φ(current)` that are meaningful under linear arithmetic
— an advantage of Euler over quaternion for this structured subtraction.
