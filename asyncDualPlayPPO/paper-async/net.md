# Network Architecture: Paper vs. Current Implementation

---

## Original OpenAI ASP Network

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
ee_pose         = [ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz]
                    EE position (metres, env-local) + quaternion

gripper_state   = [finger_joint_angle]
                    Raw finger joint position (radians)

     robot_state (8D) = concat(ee_pose, gripper_state)
     ─── No joint angles: RMPFlow handles IK internally ───

object_state    = [pos_x, pos_y, pos_z,
                    quat_w, quat_x, quat_y, quat_z,            ← quaternion (4D not Euler 3D)
                    vel_x, vel_y, vel_z,
                    angvel_x, angvel_y, angvel_z,
                    gripper_distance, gripper_contact]         = 15D per object

goal_state (Bob only)
                = [desired_pos_x, desired_pos_y, desired_pos_z,
                    desired_quat_w, desired_quat_x, desired_quat_y, desired_quat_z]  = 7D per object

goal_distance (Bob only)
                = [pos_dist,   ← L2(current_pos, goal_pos) in metres
                    rot_dist]  ← 1 - |dot(current_quat, goal_quat)|, range [0,1]
                                                                             = 2D per object
```

### Forward Pass (Alice)

```
robot_state (8D) ───────────────────────────────────────────────────────────┐
                                                                             │
object_state (15D × 2 objects) →  PI Embedding:                            │ concat
                                     shared Linear(15→512) → LayerNorm → ReLU
                                     shared Linear(512→512) → LayerNorm → ReLU
                                     max-pool over 2 objects               │
                                     LayerNorm(512)  ← post-pool norm      │
                                                                            ─┘
                          concat [robot(8) | PI_pooled(512)] = 520D
                                     ↓
                          Linear(520→256) → ReLU
                                     ↓
                          LSTMCell(256→256)
                                     ↓
              ┌────────────────────────────────┐
              ▼                                ▼
       Actor head                       Value head
       Linear(256→6×11)                 Linear(256→1)
       MultiCategorical                 scalar V(s)
       (6D Cartesian deltas × 11 bins)
```

### Forward Pass (Bob)

```
robot_state (8D) ───────────────────────────────────────────────────────────┐
                                                                             │
per-object block (24D each = 15 obj + 7 goal + 2 dist):                    │
  ├─ GoalEncoder (per object):                                              │
  │    input: current_pose(7D) + goal_pose(7D) → difference encoding       │
  │    Linear(7→64) → ReLU → Linear(64→K=6) → g_i (6D per object)        │
  │    g_pooled = max-pool(g_0, g_1) → 6D  (additive injection later)     │
  │                                                                          │
  └─ Reassemble: [obj_state(15) | g_i(6)] = 21D per object                │
                                                                             │ concat
       PI Embedding:                                                         │
         shared Linear(21→512) → LayerNorm → ReLU                          │
         shared Linear(512→512) → LayerNorm → ReLU                         │
         max-pool over 2 objects                                            │
         LayerNorm(512)  ← post-pool norm                                  │
                                                                            ─┘
                    concat [robot(8) | PI_pooled(512)] = 520D
                                     ↓
  h₁ = Linear(520→256)(enc) + Linear(6→256)(g_pooled)  ← additive goal injection
  h₁ = ReLU(LayerNorm(h₁))
                                     ↓
                          LSTMCell(256→256)
                                     ↓
              ┌────────────────────────────────┐
              ▼                                ▼
       Actor head                       Value head
       Linear(256→6×11)                 Linear(256→1)
       MultiCategorical                 raw obs (56D) as input
                                        (no goal encoder bottleneck)
```

---

## Differences: Paper vs. Current

| | Paper | Current |
|--|-------|---------|
| **Robot arm state** | Joint angles (6D) | ❌ Removed — RMPFlow handles IK |
| **Gripper / EE state** | EE Cartesian pose + finger (7D) | EE quaternion pose + finger (8D) |
| **Object rotation** | Euler angles (3D) | **Quaternion (4D)** — avoids gimbal lock |
| **Object state dims** | 14D per object | **15D** per object (quat adds 1D) |
| **Goal rotation** | Euler angles (3D) | **Quaternion (4D)** |
| **Goal state dims** | 7D (with 1 scalar dist) | **7D** goal pose + **2D** dist (separate term) |
| **Goal encoding** | Raw PI embedding on goal states | **GoalEncoder → K=6 latent** per object |
| **Additive goal injection** | ❌ | ✅ `h = act(LN(W·enc + Wg·g))` |
| **Pooling** | Sum-pool | **Max-pool** (more robust, standard DeepSets) |
| **Post-pool norm** | LayerNorm ✅ | LayerNorm ✅ |
| **Object streams merged** | obj + goal → separate PI embeddings, summed | **Combined [obj\|g_i] → single PI embedding** |

---

## Why quaternion instead of Euler?

The paper uses Euler for goal checking (absolute difference per axis) but the observation representation
is not explicitly stated as Euler — it may also use quaternion internally.

Quaternion advantages in observations:
- No gimbal lock (Euler has singularity at pitch = ±90°)
- Smooth interpolation (SLERP)
- Consistent gradient signal for the policy network

Disadvantage: double-cover (q ≡ -q). Fixed by canonicalizing `w ≥ 0` in `ee_poses()`.