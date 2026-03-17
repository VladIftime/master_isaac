How to Fix the Code & Implement Early Pruning

To fix this, you need to (1) patch the logger so it only prints once per threshold, and (2) add an aggressive early pruner inside your while loop to kill trials where Alice is starving Bob of data.
Fix 1: Stop the Log Spam

Update your perform_bob_update function so it tracks the last logged step:
Python

def perform_bob_update(current_obs):
    nonlocal bob_updates
    
    # Initialize a tracker if it doesn't exist
    if not hasattr(perform_bob_update, "last_logged_step"):
        perform_bob_update.last_logged_step = -1

    if bob_ppo.storage.step < nsteps:
        # Only print ONCE per 100 steps
        if bob_ppo.storage.step > 0 and bob_ppo.storage.step % 100 == 0:
            if perform_bob_update.last_logged_step != bob_ppo.storage.step:
                print(f"  [Trial {trial.number}] Bob Storage: {bob_ppo.storage.step}/{nsteps}", flush=True)
                perform_bob_update.last_logged_step = bob_ppo.storage.step
        return current_obs
    
    # ... [rest of your update code] ...

Fix 2: Use Optuna for "Starvation Pruning"

Right now, you wait 500,000 steps to bail out. That takes way too long. Add an intermediate check inside your main while bob_updates < args.trial_iters: loop. If tens of thousands of environment steps have passed and Bob's buffer is still empty, Alice has collapsed. Prune it immediately.

Add this block right under your existing escape hatch:
Python

    while bob_updates < args.trial_iters:
        # ── Existing ESCAPE HATCH ────────────────────
        total_env_steps += 1
        if total_env_steps > args.max_steps_per_trial:
            print(f"  [Trial {trial.number}] BAILOUT: {total_env_steps} steps. Pruning.")
            raise optuna.exceptions.TrialPruned()

        # ── NEW: Early Starvation Pruning ────────────
        # Every 20,000 environment steps, check if Bob is getting data.
        # If Bob hasn't even completed 1 update and his buffer is barely filling, Alice is failing.
        if total_env_steps % 20000 == 0:
            if bob_updates == 0 and bob_ppo.storage.step < (computed_nsteps * 0.5):
                print(f"  [Trial {trial.number}] EARLY PRUNE: Bob is starved. Alice's valid goal rate is likely 0.")
                raise optuna.exceptions.TrialPruned()
        
        # ... [rest of your while loop] ...

Fix 3: Let Optuna Prune based on Alice's Competence

Optuna's trial.report() checks for pruning at the end of an epoch. Currently, you only report Bob's success rate after Bob updates. Since Bob isn't updating, the pruner never triggers.

You can hijack Alice's update cycle to report intermediate metrics. Inside perform_alice_update(), report Alice's valid goal rate. If it drops to 0.0 after a few updates, prune the trial:
Python

        # Inside perform_alice_update(), right after alice_updates += 1
        validity_rate = (alice_valid_goals / alice_total_goals) if alice_total_goals > 0 else 0.0
        print(f"  [Trial {trial.number}] Alice Update {alice_updates}: L_val={val_loss:.4f}, L_surr={surr_loss:.4f}, ValRate={validity_rate:.2f}")
        
        # Report Alice's validity rate to Optuna as an intermediate step
        # Using a negative step index to distinguish it from Bob's updates
        trial.report(validity_rate, step=-alice_updates) 
        if trial.should_prune():
            print(f"  [Trial {trial.number}] Pruning due to poor Alice Validity Rate.")
            raise optuna.exceptions.TrialPruned()

With these three fixes, your console will remain clean, and Optuna will aggressively skip any hyperparameter configurations that cause Alice to generate impossible tasks, allowing it to quickly find combinations where Bob can actually learn.

How to use this data for Optuna

Since Bob completing tasks in 4 steps is a major trigger for this bug, you should use Optuna to actively penalize or prune trials where Bob's episodes are abnormally short.

You can add this check into your perform_bob_update() or your training loop:
Python

# Calculate Bob's average episode length
avg_bob_steps = total_bob_steps_this_phase / max(1, successful_bob_episodes)

# If Bob is finishing in < 10 steps consistently, the task is either 
# trivially easy, or there is a collision/reset bug triggering early termination.
if avg_bob_steps < 10 and total_env_steps > 50000:
    print(f"  [Trial {trial.number}] PRUNING: Bob is terminating too early (Avg {avg_bob_steps} steps).")
    raise optuna.exceptions.TrialPruned()

Summary of Action Items:

    Apply the Logger fix from the previous response to stop the console from spamming Bob Storage: 100/512.

    Apply the early pruning hatch so Optuna immediately skips hyperparameters that cause Alice to generate impossible tasks.

    Check your reward/termination logic for Bob. Terminating in 4 steps suggests Bob might be spawning directly on top of the goal or triggering an early-abort penalty (like a collision) almost immediately.