Here is the complete, step-by-step implementation plan. It is structured sequentially to ensure that environment physics and game theory are stabilized first, followed by the PPO math corrections, and finally the Hindsight Experience Replay (HER) injection.

---

### Phase 1: Environment Integrity & Deadlock Prevention

**Goal:** Stop Alice from exploiting the physics engine and ensure Bob inherits a pristine starting state.

**1. Fix Physics Reset on Phase Transition**

* **File:** `asyncDualPlayPPO/tasks/utils/wrapper.py` (and/or `events.py`)
* **Action:** In the `transition_to_bob` function, after resetting the objects to `initial_state`, explicitly call the joint reset function (e.g., `reset_robot_joints()`) to snap both arms back to the exact neutral default pose Alice started from. Clear the `contact_forces` buffer for the current step to prevent phantom explosion forces.

**2. Strict Goal Validity & Stability Bounds**

* **File:** `asyncDualPlayPPO/utils/goal_validator.py`
* **Action:** Modify the `validate_goal` logic to include a stability check. Extract `object_lin_vel` and `object_ang_vel` from the physics tensors at the exact frame Alice's turn ends.
* **Implementation:** ```python
is_stable = torch.max(torch.abs(object_lin_vel), dim=-1)[0] < 0.05
is_within_bounds = check_workspace_bounds(object_pos)
is_valid = is_stable & is_within_bounds & ... # (existing checks)
```

```


* If `is_valid` is False, Alice receives the `ALICE_INVALID_GOAL_PENALTY`, and the episode immediately terminates without Bob taking a turn.

---

### Phase 2: RMPflow Compatibility & Jitter Reduction

**Goal:** Allow Bob's RMPflow to navigate safely without fighting Euclidean gradients, and solve the rotational singularity trap.

**1. Null-Space Posture Injection (The "Singularity" Fix)**

* **File:** `asyncDualPlayPPO/tasks/utils/wrapper.py` (or where actions are routed to RMPflow).
* **Action:** When Alice finishes, store her final `dof_pos` (joint angles) as `alice_final_posture`.
* **Action:** During Bob's turn, continuously pass `alice_final_posture` into Bob's RMPflow configuration as the `posture_target` with a low-to-medium gain. Bob's RL policy remains oblivious, but the IK solver is now biased to unwind the wrists properly.

**2. Action Smoothing & Dense Shaping**

* **File:** `asyncDualPlayPPO/tasks/utils/rewards.py`
* **Action:** Add the action rate penalty: $R_{smooth} = -\lambda ||a_t - a_{t-1}||^2$.
* **Action:** Ensure the dense reward is formatted strictly as $R_{dense} = -\Delta\text{pos} - \Delta\text{rot}$ (distance to final goal, NOT a trajectory).

**3. Alpha Annealing**

* **File:** `asyncDualPlayPPO/train.py`
* **Action:** Tie the dense reward weight ($\alpha$) to the global update counter.
```python
alpha = max(0.0, initial_alpha * (1.0 - current_update / total_updates))

```


Pass this $\alpha$ into the environment wrapper or directly apply it to the reward dict so that by the end of training, Bob only optimizes the sparse $+1$ success.

---

### Phase 3: The Advantage Normalization Fix

**Goal:** Prevent Bob's surrogate loss from exploding due to `mask=0` padding steps during Alice's turn.

**1. Temporal Index Slicing**

* **File:** `asyncDualPlayPPO/algorithms/rl/ppo/storage.py`
* **Action:** In the `compute_returns` and batch sampling methods, dynamically track $T_{handoff}$ (the timestep index where Bob took control for each environment).
* **Action:** When calculating the mean and standard deviation for Advantage Normalization, **do not** use `advantages.mean()`. Instead, slice the active steps:
```python
# Pseudo-code for flattened advantage normalization
active_advantages = advantages[masks == 1] # or slice by T_handoff
mean = active_advantages.mean()
std = active_advantages.std()
advantages = (advantages - mean) / (std + 1e-8)

```


Ensure that the PPO mini-batch sampler *only* yields transitions where Bob was actively in control.

---

### Phase 4: Safe-State Filtered HER (The "Ghost Reach" Fix)

**Goal:** Provide dense success signals to Bob's pure RL policy without poisoning his Value Function with destructive collisions.

**1. Safety Filter Evaluation**

* **File:** `asyncDualPlayPPO/train.py` (After Bob's rollout concludes)
* **Action:** Evaluate the stored `contact_forces_left` and `contact_forces_right` arrays for Bob's trajectory.
* **Action:** Create a boolean mask: `is_safe_failure = (max_contact_force < THRESHOLD) & (not_success)`.

**2. Hindsight Goal Injection**

* **File:** `asyncDualPlayPPO/algorithms/rl/ppo/storage.py` (or a dedicated relabeler utility called in `train.py`)
* **Action:** For environments where `is_safe_failure == True`:
* Extract the *achieved* object state at Bob's final timestep $T_{end}$.
* Overwrite the `goal_state` vector in Bob's `obs` tensor from $T_{handoff}$ to $T_{end}$ with this new achieved state.
* Recompute the distance features inside the observation vector (if your observation space includes real-time goal distance).
* Change the reward at $T_{end}$ to $+1.0$.


* **Action:** For environments where `is_safe_failure == False` (i.e., Bob crashed), leave the trajectory exactly as it is (reward $= 0$).

---

### Execution Order Recommendation:

I recommend we tackle the implementation in the exact order above. **Phase 1** and **Phase 3** fix underlying mathematical and structural bugs that will ruin training regardless of the algorithm. **Phase 2** tunes the RMPflow mechanics, and **Phase 4** turns on the learning engine (HER).
