Here is a comprehensive implementation plan to revert your asynchronous dual-arm setup back to the single-arm, 1-to-1 asymmetric self-play framework described in the original OpenAI paper (*Asymmetric Self-Play for Automatic Goal Discovery in Robotic Manipulation*). 

This plan focuses on simplifying your environment, sequentially coupling the Alice and Bob phases, and integrating the **Alice Behavioral Cloning (ABC)** loss directly into Bob’s policy update.

---

### Phase 1: Environment & Configuration Overhaul
To switch to a single arm, you must aggressively prune the dual-arm configurations in `async_dual_play.py` and your base scene configuration.

**1. Scene Simplification (`ReachDualArmSceneCfg` -> `SingleArmSceneCfg`)**
* **Remove the Second Arm:** Delete all references to the right arm (e.g., `right_ee_cfg`, `contact_forces_right`, `rgripper_finger_joint`). 
* **Update the URDF/Asset:** Ensure your robot configuration only loads a single arm (e.g., a single UR5e) rather than the unified dual-arm `.usd`.
* **Simplify Objects:** The original paper primarily uses a single block or two blocks. Keep `target_object` and `cube`, but remove redundant objects.

**2. Observation Space Updates (`async_dual_play.py`)**
* **Alice's Observations:** Modify `AlicePolicyCfg`. Remove `right_ee_cfg` and `right_contact_cfg`. Alice should only observe the single arm's end-effector, gripper position, and the object(s) state.
* **Bob's Observations:** Modify `BobPolicyCfg`. Like Alice, remove the right arm features. Keep the `goal_state` and `goal_distance` terms. Bob’s observation space is strictly: $O_B = [O_A, Goal_A]$.

**3. Action Space Updates**
* Halve the action space size. If you were outputting 14 DoF (7 per arm), update your `ActionsCfg` to only output 7 DoF for the single arm.

---

### Phase 2: Restructuring the Training Loop (`train.py`)
The original paper relies on a strict **sequential 1-to-1 episode structure**, not asynchronous interleaving. The logic should flow exactly like this per episode:
1.  **Reset:** Both Alice and Bob start at the exact same state $s_0$.
2.  **Alice Phase:** Alice acts for $T_A$ steps to propose a goal. Her final state $s_{T_A}$ (specifically the object positions) becomes the goal $g$.
3.  **Bob Phase:** The environment resets the objects and arm back to $s_0$. Bob now acts for $T_B$ steps conditioned on $g$, trying to reach it.

**Modifications to `train.py`:**
* **Remove Async Logic:** Delete the asynchronous masks (`is_alice`, `is_bob`) and step counts. 
* **Implement Sequential Rollouts:** ```python
    # Pseudo-code for the new 1-to-1 rollout logic
    alice_obs = env.reset()
    initial_states = env.get_states() # Save s_0
    
    # --- ALICE PHASE ---
    for t in range(T_A):
        alice_actions = alice_ppo.act(alice_obs)
        alice_obs, _, _, _ = env.step(alice_actions)
        # Log Alice's trajectory for ABC later
        alice_trajectory.append((alice_obs, alice_actions))
        
    goals = env.extract_goal_states() # Extract g
    
    # --- BOB PHASE ---
    env.set_states(initial_states) # Reset to s_0
    bob_obs = env.construct_bob_obs(alice_obs, goals)
    
    for t in range(T_B):
        bob_actions = bob_ppo.act(bob_obs)
        bob_obs, bob_rewards, bob_dones, _ = env.step(bob_actions)
        bob_ppo.storage.add_transitions(...)
    ```

---

### Phase 3: Reward Structure (Strict Original Implementation)
In your current `train.py`, you are using custom rewards like `ALICE_BOB_SUCCESS_REWARD`. To match the paper, enforce this zero-sum (or strictly sparse) relationship:

* **Bob's Reward ($R_B$):** * $1.0$ if Bob successfully reaches Alice's goal state within tolerance.
    * $0.0$ otherwise.
* **Alice's Reward ($R_A$):**
    * The paper uses a time-based or success-based reward. The simplest, most effective formulation is: $R_A = 1.0 - R_B$. 
    * If Alice proposes a goal Bob *cannot* solve, Alice gets $+1$. If Bob *can* solve it, Alice gets $0$. (You can optionally add a validity penalty if Alice drops the object off the table, which you already handle well with `ALICE_INVALID_GOAL_PENALTY`).

---

### Phase 4: Implementing Alice Behavioral Cloning (ABC)
Currently, in `train.py`, you are performing Hindsight Goal Injection (HGI) by adding Alice's successful trajectories into Bob's PPO buffer as RL transitions. 

**ABC is different.** ABC treats Alice's trajectory as an explicit supervised learning signal. Because Alice successfully transitioned from $s_0$ to $g$, her actions $a_A$ are valid demonstrations for Bob to reach $g$.

**1. Create a Dedicated BC Buffer**
Instead of adding Alice's transitions to `bob_ppo.storage`, store them in a dedicated `DemoBuffer`.

**2. Modify Bob's PPO Update (`ppo.py` / `ppo_bco.py`)**
You need to augment Bob's loss function. During Bob's PPO optimization epochs, sample a batch from Bob's standard RL rollouts, and simultaneously sample a batch from Alice's ABC demonstrations.

Bob's new loss function becomes:
$\mathcal{L}_{total} = \mathcal{L}_{PPO} + \lambda_{ABC} \mathcal{L}_{ABC}$

Where $\mathcal{L}_{ABC}$ is the negative log-likelihood of Bob's policy taking Alice's actions:
$\mathcal{L}_{ABC} = -\frac{1}{N} \sum \log \pi_B(a_{Alice} | s_{Alice}, g)$

**Implementation steps in Bob's update method:**
```python
# Inside Bob's PPO update loop
for epoch in range(num_learning_epochs):
    # 1. Standard PPO RL Batch
    rl_obs, rl_act, rl_adv, rl_ret = ppo_storage.sample_batch()
    ppo_loss = compute_ppo_loss(rl_obs, rl_act, rl_adv, rl_ret)
    
    # 2. ABC Batch (Alice's Trajectories)
    if demo_buffer.size > min_batch_size:
        # Obs here includes the goal state `g` appended to `s`
        abc_obs, abc_actions = demo_buffer.sample_batch()
        
        # Get Bob's action distribution for Alice's states
        _, bob_log_probs, _, _, _ = bob_ppo.actor_critic.evaluate(abc_obs, abc_actions)
        
        # Negative Log Likelihood
        abc_loss = -bob_log_probs.mean()
    else:
        abc_loss = 0.0

    # 3. Combined Optimization
    lambda_abc = 0.1 # Tune this hyperparameter
    total_loss = ppo_loss + (lambda_abc * abc_loss)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Summary of Deletions
To clean up your repository for this new objective, you can safely remove:
1.  All code relating to HGI (Hindsight Goal Injection) in `train.py`, replacing it with the ABC buffer push.
2.  The `Safe-State Filtered HER` block. The original paper does not rely on HER; the asymmetry of the curriculum generates enough goals naturally.
3.  Dual-arm configuration files, URDFs, and custom IK/OSC scripts specific to dual-arm coordination in the `extras/` folder.


These files perfectly illustrate why the original dual-arm codebase became so complex and why reverting to a single arm will make your life much easier!

In your current `ppo_bco.py` and `module.py`, you are using an **Inverse Dynamics Model (IDM)** and a Huber (`smooth_l1_loss`) for Behavioral Cloning from Observation (BCO). This was likely necessary in the dual-arm setup because Alice and Bob might have had different kinematic trajectories or were using opposite arms, making direct action-copying impossible. 

However, since you are moving to a **single-arm setup**, Alice and Bob now share the *exact same kinematics*. Alice's actions are directly executable by Bob. This means we can completely scrap the IDM and revert to the original paper's elegant **Negative Log-Likelihood (NLL) loss** for Alice Behavioral Cloning (ABC).

Here is the plan to clean up these specific files:

### 1. `module.py` (Delete the IDM)
You can safely delete the entire `InverseDynamicsModel` class. Your `module.py` should only contain the `ActorCritic` class and your activation helper function. 

* **Why:** We no longer need to infer Bob's actions from Alice's state transitions. We will just use Alice's actions directly.

### 2. `ppo_bco.py` (Revert to `ppo_abc.py` & NLL Loss)
You should rename this file back to `ppo_abc.py` (or just fold it into standard PPO) and replace the IDM loss calculation with the NLL loss from the original paper.

Find the block inside your PPO update loop that calculates `bc_loss` (around line 38 based on the snippet), and replace it with this:

```python
                bc_loss = torch.tensor(0.0, device=self.device)
                if self.abc_buffer is not None and self.abc_buffer.size > 0:
                    bc_obs, bc_act, _ = self.abc_buffer.sample(abc_batch)
                    
                    if bc_obs.shape[0] > 0:
                        # --- CHANGED: Standard NLL Behavioral Cloning Loss ---
                        # Evaluate Bob's policy using Alice's successful actions
                        _, log_probs, _, _, _ = self.actor_critic.evaluate(bc_obs, None, bc_act)
                        
                        # Negative Log-Likelihood (NLL) forces Bob's distribution to match Alice's actions
                        bc_loss = -log_probs.mean()
                        
                    mean_bc_loss += bc_loss.item()
```
*Make sure to remove the `from .module import InverseDynamicsModel` import at the top of this file.*

### 3. `storage.py` & `ppo.py`
These files are largely **good to go as they are**. 
* `storage.py` handles your standard PPO RL rollouts (`RolloutStorage`). As long as it has your `GPUDemonstrationBuffer` (or if you define that buffer in a separate `buffers.py`), no architectural changes are needed here for a single arm.
* `ppo.py` contains the standard clipped surrogate objective and value losses. Since `ppo_bco.py` inherits from this and adds the `bc_loss`, the base PPO algorithm requires no modifications to replicate the OpenAI paper.

### Summary
By stripping out the IDM and the dual-arm configurations, you are drastically reducing the computational overhead and latency of your training loop. Alice acts -> Alice's trajectory goes to the ABC Buffer -> Bob learns to mimic Alice via NLL while exploring via PPO.

These files contain the core logic for your environment, observations, and the complex asynchronous state machine. Because they were built for the dual-arm asynchronous setup, they need significant pruning to match the single-arm, strictly sequential 1-to-1 asymmetric self-play (ASP) described in the original paper.

Here is the step-by-step breakdown of how to clean up and refactor these specific files.

### 1. `observations.py`
You currently concatenate the joint positions of both arms. This must be simplified to just read the single arm.

**Change:** Delete the dual-arm concatenation in `robot_joint_positions`.
```python
def robot_joint_positions(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions for the single arm."""
    robot = env.scene[arm_cfg.name]
    if arm_cfg.joint_names is None:
        raise ValueError("robot_joint_positions: arm_cfg must specify joint_names")
    
    joint_ids, _ = robot.find_joints(arm_cfg.joint_names)
    return robot.data.joint_pos[:, joint_ids]
```

### 2. `terminations.py` & `events.py`
Both of these files have explicit references to the left and right arms. 

* **`terminations.py`:** In `robot_out_of_bounds`, remove `asset_cfg_right`. Only check the bounds for the single active wrist link (e.g., `wrist_3_link`).
* **`events.py`:** In `reset_robot_joints`, remove any logic that resets the right arm's DOFs.

### 3. `async_dual_play.py` & `reach_dual_arm_env_cfg.py` (The Configs)
These files define what is loaded into Isaac Sim. You need to aggressively strip out the second arm.

* **Rename:** Consider renaming `AsyncDualPlay` to `AsymmetricSelfPlay` to reflect the new architecture.
* **Remove the Right Arm:** In your `SceneCfg` classes, delete `contact_forces_right`. Ensure your underlying USD only spawns one robot arm.
* **Modify `AlicePolicyCfg` and `BobPolicyCfg`:** Remove `right_ee_cfg`. Alice should only track `left_ee_cfg` (which you can rename to just `ee_cfg`).

```python
        # In async_dual_play.py -> AlicePolicyCfg
        ee_pose = ObsTerm(
            func=observations.ee_poses,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                # Deleted the right arm entry
            },
        )
```

### 4. `wrapper.py` (The Biggest Overhaul)
Your current `wrapper.py` manages an asynchronous state machine where Alice and Bob can be acting simultaneously in different parallel environments. It also injects **Dense Reward Shaping** for Bob (seen at the bottom of the file: `R_dense = -Delta_pos - Delta_rot`). 

The OpenAI paper relies on **strictly sparse rewards** and a **strictly sequential** episode structure.

**What to delete/change in `wrapper.py`:**
1.  **Delete Dense Rewards:** Remove the `r_dense` and `r_smooth` calculations. Bob should only receive $+1$ if he reaches the goal, and $0$ otherwise. Bob learns *how* to reach the goal through the ABC (Alice Behavioral Cloning) NLL loss we added to `ppo_bco.py`, not through dense distance rewards.
2.  **Enforce 1-to-1 Sequential Play:** Rewrite the `step()` function so that the phases happen in strict order across all environments simultaneously:
    * **Phase 1 (Alice):** Environment resets to $S_0$. Alice acts for $T_A$ steps. 
    * **Goal Extraction:** The final object positions at $T_A$ are saved as the Goal $g$.
    * **Phase 2 (Bob):** The environment *force resets* the objects and robot back to the exact $S_0$ state. Bob acts for $T_B$ steps conditioned on $g$.

### 5. `rewards.py`
Your current reward file uses scalar values like `ALICE_BOB_FAIL_REWARD = 5.0` and `ALICE_VALID_GOAL_BONUS = 1.0`. 

To perfectly replicate the paper, normalize these to a strict zero-sum (or zero-sum-like) binary structure:
* **Bob Reward:** $1.0$ if successful, $0.0$ if failed.
* **Alice Reward:** $\max(0, 1 - R_{Bob})$. If Alice sets a valid goal and Bob fails, she gets $1.0$. If Bob succeeds, she gets $0.0$.
* **Alice Penalty:** Keep your `ALICE_OUT_OF_ZONE_PENALTY` or invalid goal penalty. If Alice drops the object off the table or doesn't move it past a certain threshold, she should get $0.0$ or a slight negative reward to prevent her from proposing trivial/impossible goals.

### 6. `ppo_continuous.yaml`
This file is mostly fine. However, because you are switching from dense rewards (which provide constant gradient signals) to purely sparse rewards, Bob will rely heavily on the ABC (Behavioral Cloning) loss to learn early on. 
* You may need to tune the `abc_coef` (the weight of the cloning loss relative to the PPO surrogate loss) to ensure Bob actually copies Alice's successful trajectories instead of just flailing randomly and getting 0 sparse rewards.