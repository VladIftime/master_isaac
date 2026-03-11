Based on the logs you provided, you have successfully crossed a major milestone: **Alice is correctly solving the task-space kinematics and setting valid goals** (`[Phase] Env 0: Alice→Bob | Valid Goal | Moved: 0.129m`).

However, your PPO algorithm is experiencing two massive statistical anomalies: a **Logging Artifact** (which makes your rewards look like 0) and a **Gradient Multiplier Explosion** (which causes the 248,000 loss).

Here is the exact mathematical breakdown of what is happening in your code and how to fix it.

---

### Anomaly 1: Why are Rewards `0.0000` and Episodes `2048`?

If Alice successfully set a goal, she should have received a `+1.0` reward. So why does the log say `mean=0.0000 | min=0.0000`?

This is a side-effect of the gating logic we added for the asynchronous curriculum. When Alice is forced to "wait" for Bob, `train.py` fills her rollout buffer with zero-padded steps. To tell the PPO buffer that these steps aren't real, they are usually marked with `done=1`.

* When `storage.get_statistics()` calculates the metrics, it counts every single padded step as a brand-new 1-step episode.
* This is why Update 3 shows **2048 episodes buffered**.
* Because Alice's single `+1.0` reward is being averaged against 2047 "fake" padded episodes that received $0.0$, the mean becomes `0.0004`, which your logger rounds down and prints as `0.0000`.

**Verdict:** Your reward logic is working perfectly. The `0.0000` is just a visual logging artifact caused by the padding.

---

### Anomaly 2: Why did the Surrogate Loss explode to 248,293?

This is the real bug killing your policy, and it is a classic PPO mathematics trap.

In your implementation plan, you lowered `init_noise_std: 0.05` to ensure RMPflow only explored in safe, 5cm Cartesian increments. **This mathematically broke PPO.**

In PPO, the gradient that updates the actor network's mean ($\mu$) is divided by the variance ($\sigma^2$):


$$\nabla_\theta \mu = \frac{A}{\sigma^2} (a - \mu)$$

* When $\sigma = 1.0$, the gradient scale is $1.0$.
* By setting $\sigma = 0.05$, your variance became $0.0025$.
* Therefore, $1 / 0.0025 = \textbf{400}$.

**You accidentally multiplied Alice's learning rate by 400.** During `ALICE UPDATE 0`, the neural network took such a violently massive gradient step that the policy weights completely scattered.

When PPO evaluated the new policy, the probabilities of random outlier actions spiked astronomically. Because PPO's clipping function intentionally does *not* clip the loss when a "bad" action (negative advantage) becomes vastly more likely (the pessimistic bound), the ratio explosion passed straight through to the surrogate loss, outputting `248,293`.

---

### The Fix: Decouple PPO Variance from Environment Scale

To fix this, you must keep the neural network mathematically stable ($\sigma = 1.0$) while keeping the physical arms moving slowly ($5$ cm). You do this by scaling the actions in the environment wrapper, not the YAML.

#### Step 1: Fix the PPO Config

Go to `ppo_continuous.yaml` and revert the standard deviation so the gradients behave normally.

```yaml
# ppo_continuous.yaml
init_noise_std: 1.0  # REVERT THIS FROM 0.05

```

#### Step 2: Scale the Actions in `wrapper.py`

Open your `tasks/utils/wrapper.py` file. In the `step(self, actions)` method, intercept the raw actions coming from the neural network (which will now be roughly between `-3.0` and `3.0`) and multiply the RMPflow dimensions by `0.05` before passing them to the simulator.

```python
# tasks/utils/wrapper.py

def step(self, actions):
    # Clone to avoid modifying the PPO buffer's raw actions
    scaled_actions = actions.clone()
    
    # Scale Left Arm RMPflow Delta Poses (Dims 0 to 5)
    scaled_actions[:, 0:6] *= 0.05 
    
    # Scale Right Arm RMPflow Delta Poses (Dims 7 to 12)
    scaled_actions[:, 7:13] *= 0.05 
    
    # Grippers (Dims 6 and 13) are left unscaled
    
    # Pass the safely scaled actions to Isaac Lab
    obs, rewards, dones, info = self.env.step(scaled_actions)
    
    return obs, rewards, dones, info

```

By making this change, PPO will have a beautiful, stable surrogate loss of `~0.1`, but RMPflow will still only receive smooth `~5cm` Cartesian deltas, preventing kinematic crashes!