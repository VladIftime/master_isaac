Based on a review of your provided implementation files (`async_dual_play.py`, `rewards.py`, `ppo.py`, `AsyncDualPlay.yaml`, `train.py`) and the original OpenAI "Asymmetric Self-Play for Automatic Goal Discovery" paper, there are several critical logical, mathematical, and RL-design issues preventing your agents from replicating the paper's results. 

Here are the 10 core issues hindering your implementation:

### 1. Mathematically Broken Behavioral Cloning (ABC) in PPO
In `ppo.py` (Phase 4), you are attempting to inject Alice's demonstrations into Bob's training by concatenating `demo_batch` directly into the PPO rollout tensors (`obs_batch = torch.cat([obs_batch, d_obs])`). 
* **Why it fails:** You are assigning dummy advantages (`torch.zeros`) to these demonstrations in `train.py`. PPO updates the policy using the surrogate loss $L = A \cdot \frac{\pi}{\pi_{old}}$. Because the advantage $A = 0$, the gradient for Alice's demonstration actions is exactly zero. Bob learns absolutely nothing from the ABC buffer.
* **The Fix:** Alice's Behavior Cloning (ABC) must be implemented as a separate supervised learning loss (e.g., Mean Squared Error or Negative Log-Likelihood: $L_{ABC} = - \log \pi_B(a_{demo} | s, g)$) added to the total PPO loss, completely independent of the PPO advantage buffer.

### 2. Dense Reward Leakage (YAML Config)
In `AsyncDualPlay.yaml`, you have set `use_dense_bob_rewards: true` with a weight of `0.1`.
* **Why it fails:** The fundamental premise of the Asymmetric Self-Play (ASP) paper is that Bob learns via **purely sparse rewards** (success/failure). Injecting dense distance-based rewards biases Bob's learning, creates local optima, and defeats the purpose of the unsupervised, adversarial goal discovery curriculum. 
* **The Fix:** Set `use_dense_bob_rewards: false`. Bob should only receive $+1$ for reaching the goal and $0$ otherwise.

### 3. Hardcoded Alice Outcome Overrides
In `rewards.py`, you carefully defined `ALICE_BOB_FAIL_REWARD = 5.0` and `ALICE_BOB_SUCCESS_REWARD = 0.0`. 
* **Why it fails:** In `train.py`, you completely ignore these constants and hardcode the outcome: `alice_outcome_rewards = torch.where(bob_success, torch.tensor(0.0), torch.tensor(1.0))`. 
* **The Fix:** Import and use the constants from `rewards.py`. Hardcoding $1.0$ limits Alice's incentive gradient, throwing off the intended reward scale between the valid goal bonus and the failure reward.

### 4. No Invalid Goal Constraint on Alice's Reward
In `train.py`, Alice is rewarded purely if Bob fails (`bob_success == False`). 
* **Why it fails:** If Alice simply throws the object off the table or out of reach, Bob will fail. Because you don't gate her outcome reward by goal validity in the training loop, Alice will quickly learn to generate impossible goals to maximize her reward.
* **The Fix:** Alice should only get the "Bob Failed" reward if the goal is actually valid. It should be: `alice_outcome_rewards = torch.where(~bob_success & goal_valid, ALICE_BOB_FAIL_REWARD, 0.0)`.

### 5. Missing $T_A$ (Alice Horizon) Curriculum
You hardcoded `alice_timesteps: 400` from Iteration 1.
* **Why it fails:** In the paper, Alice's time horizon ($T_A$) starts very small and slowly grows, or her action space is heavily restricted early on. Without a curriculum, a 400-step Alice will instantly push the block to the edge of the workspace in the first few iterations. Bob's success rate will drop to 0%, resulting in vanishing gradients for both agents (the "unsolvable goal trap").
* **The Fix:** Implement a curriculum that linearly increases `alice_timesteps` from e.g., 20 to 400 over the first few thousand updates based on Bob's success rate.

### 6. GAE (Generalized Advantage Estimation) Bleeding
In `train.py`, you inject Alice's terminal reward directly into the last index of her storage: `alice_ppo.storage.rewards[last_idx].copy_(alice_outcome_rewards)`.
* **Why it fails:** Because PPO relies on GAE for advantage calculation, if the environment's `dones` tensor is not explicitly overridden to `True` at that exact step for all environments, this sparse reward will bleed into the next iteration's trajectory. 
* **The Fix:** You must force `dones = True` and mask the state at `last_idx` so the GAE calculation appropriately truncates the temporal difference backup.

### 7. Bob is Forced to Mimic Alice's "Random Walk"
When executing the ABC push, Alice's raw trajectory (`traj_a`) is copied directly to the buffer.
* **Why it fails:** Alice is an exploring agent; her path to the final goal state includes random, inefficient wandering. By forcing Bob to mimic `traj_a` identically, you are teaching Bob an inefficient random walk rather than a direct path to the goal.
* **The Fix:** The paper uses Bob's RL gradients to smooth out the path over time, but to prevent catastrophic interference early on, ensure the ABC loss weight decays over time so Bob relies more on PPO to find efficient paths once he figures out the general area.

### 8. Transient / Mid-Air Goal States
When Alice finishes her 400 steps, you immediately extract the object state as the goal $g$.
* **Why it fails:** If the object is still falling, sliding, or bouncing at step 400, $g$ is recorded with those transient velocities and positions. Bob is then tasked with recreating a mid-air/sliding state, which is physically impossible to stabilize.
* **The Fix:** After Alice's rollout, run the simulation for 20-30 steps with zero-actions to allow physics to settle before extracting `goal_states`.

### 9. 1:1 Symmetric Update Ratio Bottleneck
The loop in `train.py` executes `perform_alice_update()` and `perform_bob_update()` exactly once per iteration.
* **Why it fails:** Bob is solving a much harder problem (Goal-Conditioned RL) than Alice (State-Conditioned RL). The paper highlights that Bob usually requires more gradient steps or multiple rollout attempts per Alice proposal to maintain a >20% success rate. 
* **The Fix:** Use your `args.max_alice_bob_ratio` properly. Freeze Alice's updates if Bob's success rate drops below a threshold (e.g., 10%), allowing Bob to train on the ABC buffer and current goals until he catches up.

### 10. IsaacLab Internal Horizon Clashing
In `async_dual_play.py`, you set `episode_length_s = 65.0`.
* **Why it fails:** If the manual loops in `train.py` (`for t in range(alice_timesteps)`) exceed the internal IsaacLab timeout horizon, the underlying environment will auto-reset itself mid-rollout. This will completely desynchronize Alice's physical state from the goal state Bob is attempting to reach.
* **The Fix:** Ensure `episode_length_s` in the environment configuration is set to infinity or a value strictly greater than `(alice_timesteps + bob_timesteps) * sim_dt` to ensure your manual Python loops maintain absolute control over episode termination.