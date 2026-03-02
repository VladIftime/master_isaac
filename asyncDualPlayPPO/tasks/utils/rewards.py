"""
Reward functions for asymmetric dual-play.

IMPORTANT: In accordance with the Asymmetric Self-Play (ASP) paper, this
implementation uses SPARSE REWARDS ONLY.

Alice rewards: Based on Bob's success/failure (outcome-based, sparse)
Bob rewards: Based on goal achievement (sparse binary triggers, no dense shaping)

WHY SPARSE REWARDS?
-------------------
The ASP paper's core insight is that the adversarial curriculum (Alice finding
goals Bob cannot solve) drives learning. Dense distance-based rewards would
"short-circuit" this curriculum by guiding Bob too much, preventing true
unsupervised goal discovery.

Bob's actual rewards are computed directly in AsyncDualPlayEnvWrapper to ensure
integer values without dt-scaling. The functions below are mostly for reference
or potential experimentation.

See ASP_IMPLEMENTATION.md for detailed rationale.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def alice_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Alice's reward based on Bob's performance.
    
    Rewards:
    - +5: Bob failed to solve the goal
    - +1: Valid goal set (given at goal validation time)
    - -3: Goal out of bounds penalty
    - 0: Bob succeeded
    
    Returns:
        Reward tensor (num_envs,)
    """
    batch_size = env.scene[object_cfg.name].data.root_pos_w.shape[0]
    rewards = torch.zeros(batch_size, device=env.device)
    
    # Check if we're computing rewards at episode end
    if hasattr(env, 'extras') and "bob_success" in env.extras:
        bob_success = env.extras["bob_success"]
        # Alice gets +5 when Bob fails
        rewards = torch.where(bob_success, torch.zeros_like(rewards), torch.ones_like(rewards) * 5.0)
    
    return rewards


def bob_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    pos_threshold: float = 0.04,
    rot_threshold: float = 0.2,
) -> torch.Tensor:
    """
    Bob's reward based on goal achievement.
    
    If multiple objects, reward is given if ALL objects are within validation thresholds.
    
    Rewards:
    - +5: Terminal success (all objects within threshold)
    - +1: Per-object placement reward (implemented via episode manager tracking)
    - Dense: Inverse distance sum
    
    Returns:
        Reward tensor (num_envs,)
    """
    from .observations import goal_distance
    
    # Get flattened distances: (num_envs, num_objects * 2)
    # [dist_pos_1, dist_rot_1, dist_pos_2, dist_rot_2, ...]
    all_dists = goal_distance(env, object_cfg)
    batch_size = all_dists.shape[0]
    
    # Reshape to (num_envs, num_objects, 2)
    num_objects = all_dists.shape[1] // 2
    dists = all_dists.view(batch_size, num_objects, 2)
    
    pos_dists = dists[..., 0] # (num_envs, num_objects)
    rot_dists = dists[..., 1]
    
    # Calculate per-object success
    obj_success = (pos_dists < pos_threshold) & (rot_dists < rot_threshold)
    
    # Episode success if ALL objects are successful
    all_success = torch.all(obj_success, dim=1)
    
    rewards = torch.zeros(batch_size, device=env.device)
    
    # Terminal success bonus
    rewards = torch.where(all_success, torch.ones_like(rewards) * 5.0, rewards)
    
    # Per-Object Reward Logic (+1 for place, -1 for drop)
    # Requires state tracking in episode manager
    if hasattr(env, 'episode_manager'):
        prev_success = env.episode_manager.prev_obj_success
        
        # Initialize prev if None (first step of episode)
        if prev_success is None:
            prev_success = torch.zeros_like(obj_success, dtype=torch.bool)
            
        # Calculate delta
        # True -> True: 0
        # False -> False: 0
        # False -> True: +1
        # True -> False: -1
        delta = obj_success.float() - prev_success.float()
        
        # Sum rewards across objects
        step_rewards = delta.sum(dim=1)
        rewards += step_rewards
        
        # Update state in manager
        env.episode_manager.prev_obj_success = obj_success.clone()
    else:
        # Fallback if manager missing

        pass
    
    return rewards


def valid_goal_bonus(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Alice receives +1 for setting a valid goal.
    
    This is triggered when goal validation passes.
    
    Returns:
        Reward tensor (num_envs,)
    """
    batch_size = env.scene.num_envs
    rewards = torch.zeros(batch_size, device=env.device)
    
    # Check if goal validation flag is set
    if hasattr(env, 'extras') and "goal_valid" in env.extras:
        goal_valid = env.extras["goal_valid"]
        rewards = torch.where(goal_valid, torch.ones_like(rewards), rewards)
    
    return rewards


def out_of_bounds_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    x_range: tuple = (-0.6, 0.6),
    y_range: tuple = (0.3, 0.9),
) -> torch.Tensor:
    """
    Alice receives -3 if objects are outside the placement area.
    
    The placement area is defined by x_range and y_range (camera view).
    
    Returns:
        Penalty tensor (num_envs,)
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w
    
    # Handle multiple objects
    if pos.dim() == 2:
        pos = pos.unsqueeze(1)
        
    # Standardise pos and env_origins to 3D for subtraction (num_envs, num_instances, 3)
    pos = pos - env.scene.env_origins.unsqueeze(1)
        
    # Check bounds for all objects
    x_out = (pos[..., 0] < x_range[0]) | (pos[..., 0] > x_range[1])
    y_out = (pos[..., 1] < y_range[0]) | (pos[..., 1] > y_range[1])
    
    # Check if any object is out of bounds

    any_out = (x_out | y_out).any(dim=1)
    
    penalty = torch.where(any_out, torch.ones_like(any_out, dtype=torch.float) * -3.0, torch.zeros_like(any_out, dtype=torch.float))
    
    return penalty


def alice_shaping_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    debug: bool = False,
) -> torch.Tensor:
    """Dense shaping reward for Alice during goal-setting phase.
    
    Components:
    1. Reach reward: 1.0 / (1.0 + 5.0 * dist) - encourages hand approaching object
    2. Push reward: tanh(obj_velocity) - encourages moving the object
    
    This provides continuous feedback during exploration to fix paralysis from
    sparse-only rewards. Should be scaled down (e.g., 0.1x) to avoid overwhelming
    the sparse outcome rewards (+5, +1, -3).
    
    Args:
        env: Environment instance
        object_cfg: Object to track
        debug: If True, print component breakdown for first environment
    
    Returns:
        Reward tensor (num_envs,)
    """
    # Get object position
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w
    
    # Get end-effector position from left robot (Alice uses left arm)
    robot = env.scene["robot"]
    # wrist_3_link is typically the last body for end-effector
    # [FIX] Unified robot: find specific body index
    ids, _ = robot.find_bodies("left_wrist_3_link")
    ee_pos = robot.data.body_pos_w[:, ids[0], :]
    
    # Reach reward: higher as distance decreases
    # Range: [0.0, 1.0], max when dist=0
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    reach_reward = 1.0 / (1.0 + 5.0 * dist)
    
    # Push reward: object velocity magnitude
    # Range: [0.0, 1.0], saturates at high velocities via tanh
    obj_vel = torch.norm(obj.data.root_vel_w[:, :3], dim=-1)
    move_reward = torch.tanh(obj_vel)
    
    # Combine (tune weights)
    # Reach is important early (finding object)
    # Move is important later (pushing/manipulating)
    total_reward = (0.5 * reach_reward) + (1.0 * move_reward)
    
    # Optional debug output
    if debug:
        print(f"[Dense Reward Debug] Dist: {dist[0]:.3f}m | Reach: {reach_reward[0]:.4f} | "
              f"Vel: {obj_vel[0]:.3f}m/s | Push: {move_reward[0]:.4f} | Total: {total_reward[0]:.4f}")
    
    return total_reward
