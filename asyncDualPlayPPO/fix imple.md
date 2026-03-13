You have hit a classic Python "Variable Shadowing" bug!

Your RL math and PyTorch logic are completely fine. The crash is happening because you accidentally reused the variable name `a_acts` inside your Hindsight Goal Injection (HGI) loop, which overwrote the current step's actions.

### The Exact Cause

1. At the start of the step, Alice decides her actions for the 10 environments:
`a_acts = ...` *(Shape: `[10, 13]`)*
2. Then, the HGI loop runs because Alice finally reached a valid goal! Inside that loop, you extract her historical trajectory to inject into Bob's buffer using the exact same variable names:
`a_acts = alice_act_log[env_id, :s_count]` *(Shape: `[399, 13]` - this is the 399 steps she took to reach the goal)*
3. When the code reaches line 373 to log the *current* step, it tries to save `a_acts`. But because it was overwritten by the HGI loop, it tries to cram a `[399, 13]` tensor into a `[10, 13]` slot, causing the crash.

### The Fix

You just need to rename the variables inside the Hindsight Goal Injection block so they don't overwrite the current step's variables.

Open `train.py` and find the HGI block (around **line 317**). Change `a_obs` and `a_acts` to `demo_obs` and `demo_acts`.

**Change this section:**

```python
                hgi_count = 0
                for idx in success_ids:
                    env_id = idx.item()
                    s_count = min(alice_step_counts[env_id].item(), max_alice_steps)
                    if s_count == 0: continue
                    
                    a_obs = alice_obs_log[env_id, :s_count]   # <--- RENAME THESE
                    a_acts = alice_act_log[env_id, :s_count]  # <--- RENAME THESE
                    
                    _o = torch.zeros((s_count, env.bob_obs_dim), device=env.device)
                    # ... (skipping some lines) ...
                    
                    # Construct Bob's obs
                    goal_state = goal_states[env_id].unsqueeze(0).expand(s_count, -1)
                    b_obs = env.construct_bob_observation(a_obs, goal_state) # <--- UPDATE HERE
                    _o[:] = b_obs
                    
                    # ...
                    
                    # Evaluate under Bob's current policy
                    with torch.no_grad():
                        _lp, _, _v, _m, _s = bob_ppo.actor_critic.evaluate(_o, None, a_acts) # <--- UPDATE HERE
                        
                    # ... (skipping GAE math) ...
                        
                    # Add to offline demo buffer
                    none_states = torch.zeros((s_count, *env.bob_state_space.shape), device=env.device)
                    bob_ppo.demo_buffer.add_trajectory(_o, none_states, a_acts, _r, _d, _v, _lp, _m, _s, _ret, _adv) # <--- UPDATE HERE

```

**To this:**

```python
                hgi_count = 0
                for idx in success_ids:
                    env_id = idx.item()
                    s_count = min(alice_step_counts[env_id].item(), max_alice_steps)
                    if s_count == 0: continue
                    
                    # USE NEW VARIABLE NAMES TO AVOID SHADOWING
                    demo_obs = alice_obs_log[env_id, :s_count]
                    demo_acts = alice_act_log[env_id, :s_count]
                    
                    _o = torch.zeros((s_count, env.bob_obs_dim), device=env.device)
                    _r = torch.zeros((s_count,), device=env.device)
                    _d = torch.zeros((s_count,), device=env.device)
                    
                    # Construct Bob's obs
                    goal_state = goal_states[env_id].unsqueeze(0).expand(s_count, -1)
                    b_obs = env.construct_bob_observation(demo_obs, goal_state)  # UPDATED
                    _o[:] = b_obs
                    
                    _r[-1] = 5.0
                    _d[-1] = 1.0
                    
                    # Evaluate under Bob's current policy
                    with torch.no_grad():
                        _lp, _, _v, _m, _s = bob_ppo.actor_critic.evaluate(_o, None, demo_acts)  # UPDATED
                        
                    _ret = torch.zeros((s_count,), device=env.device)
                    _adv = torch.zeros((s_count,), device=env.device)
                    
                    adv = 0.0
                    gamma = bob_ppo.gamma
                    lam = bob_ppo.lam
                    
                    for step in reversed(range(s_count)):
                        next_val = 0.0 if step == s_count - 1 else _v[step + 1].item()
                        next_not_done = 0.0 if step == s_count - 1 else 1.0
                        delta = _r[step] + gamma * next_val * next_not_done - _v[step].item()
                        adv = delta + gamma * lam * next_not_done * adv
                        _adv[step] = adv
                        _ret[step] = adv + _v[step].item()
                        
                    # Add to offline demo buffer
                    none_states = torch.zeros((s_count, *env.bob_state_space.shape), device=env.device)
                    bob_ppo.demo_buffer.add_trajectory(_o, none_states, demo_acts, _r, _d, _v, _lp, _m, _s, _ret, _adv)  # UPDATED
                    hgi_count += s_count

```

Make those 5 quick renames, and your code will breeze right past this line! The fact that you hit this bug means your Hindsight Goal Injection is successfully triggering, which is fantastic news for the training run.