import math
import torch


PBRS_K_POS = 30.0
PBRS_K_ROT = 5.0
PBRS_W_POS = 10.0
PBRS_W_ROT = 10.0

PBRS_POS_THRESHOLD = 0.05
PBRS_COS_ROT_THRESHOLD = 0.01

PBRS_COMPLETION_BONUS = 5.0
PBRS_ROTATION_BONUS = 2.0
PBRS_TIP_PENALTY = -5.0
PBRS_CATASTROPHE_PENALTY = -5.0

PBRS_K_DPOSE = 30.0
PBRS_W_DPOSE = 10.0
PBRS_DPOSE_THRESHOLD_TBLOCK = 0.055
PBRS_DPOSE_THRESHOLD_DISC = 0.05

TIP_OVER_THRESHOLD = 0.3


def cosine_rot_error(yaw_current: torch.Tensor, yaw_target: torch.Tensor) -> torch.Tensor:
    return (1.0 - torch.cos(yaw_target - yaw_current)) / 2.0


def potential_pos(obj_pos: torch.Tensor, goal_pos: torch.Tensor, k_p: float = PBRS_K_POS) -> torch.Tensor:
    d_sq = ((obj_pos[..., :2] - goal_pos[..., :2]) ** 2).sum(dim=-1)
    return torch.exp(-k_p * d_sq)


def potential_rot(yaw_current: torch.Tensor, yaw_target: torch.Tensor, k_r: float = PBRS_K_ROT) -> torch.Tensor:
    c = cosine_rot_error(yaw_current, yaw_target)
    return torch.exp(-k_r * c)


def pbrs_dense(phi_prev_pos: torch.Tensor, phi_now_pos: torch.Tensor,
               phi_prev_rot: torch.Tensor, phi_now_rot: torch.Tensor,
               w_pos: float = PBRS_W_POS, w_rot: float = PBRS_W_ROT) -> torch.Tensor:
    return w_pos * (phi_now_pos - phi_prev_pos) + w_rot * (phi_now_rot - phi_prev_rot)


def compute_pbrs_reward(
    cur_obj_pos: torch.Tensor,
    cur_obj_euler: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_euler: torch.Tensor,
    prev_phi_pos: torch.Tensor,
    prev_phi_rot: torch.Tensor,
    gave_completion: torch.Tensor,
    gave_rot_bonus: torch.Tensor,
    w_pos: float = PBRS_W_POS,
    w_rot: float = PBRS_W_ROT,
    k_p: float = PBRS_K_POS,
    k_r: float = PBRS_K_ROT,
    enable_rot_sparse: bool = True,
):
    phi_pos_now = potential_pos(cur_obj_pos, goal_pos, k_p=k_p)
    phi_rot_now = potential_rot(cur_obj_euler[..., 2], goal_euler[..., 2], k_r=k_r)

    dense = pbrs_dense(prev_phi_pos, phi_pos_now, prev_phi_rot, phi_rot_now, w_pos, w_rot)
    reward = dense.clone()

    pos_err = (cur_obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
    cos_rot_err = cosine_rot_error(cur_obj_euler[:, 2], goal_euler[:, 2])
    pos_success = pos_err < PBRS_POS_THRESHOLD
    rot_success = cos_rot_err < PBRS_COS_ROT_THRESHOLD

    new_pos = pos_success & ~gave_completion
    reward += new_pos.float() * PBRS_COMPLETION_BONUS
    gave_completion |= new_pos

    if enable_rot_sparse:
        new_both = pos_success & rot_success & ~gave_rot_bonus
        reward += new_both.float() * PBRS_ROTATION_BONUS
        gave_rot_bonus |= new_both

    tipped = (cur_obj_euler[:, 0].abs() > TIP_OVER_THRESHOLD) | \
             (cur_obj_euler[:, 1].abs() > TIP_OVER_THRESHOLD)
    reward += tipped.float() * PBRS_TIP_PENALTY

    at_goal = pos_success & rot_success
    dense_pos = w_pos * (phi_pos_now - prev_phi_pos)
    dense_rot = w_rot * (phi_rot_now - prev_phi_rot)

    return {
        "reward": reward,
        "phi_pos_now": phi_pos_now,
        "phi_rot_now": phi_rot_now,
        "pos_err": pos_err,
        "cos_rot_err": cos_rot_err,
        "pos_success": pos_success,
        "rot_success": rot_success,
        "at_goal": at_goal,
        "tipped": tipped,
        "dense_pos": dense_pos,
        "dense_rot": dense_rot,
        "gave_completion": gave_completion,
        "gave_rot_bonus": gave_rot_bonus,
    }


def check_done_pbrs(
    obs: torch.Tensor,
    terminated: torch.Tensor,
    push_count: torch.Tensor,
    max_pushes: int,
    at_goal: torch.Tensor,
    robot_dim: int = 6,
    obj_state_dim: int = 14,
    pos_term_threshold: float = 0.0,
):
    max_push_done = push_count >= max_pushes

    obj_z = obs[:, robot_dim + 2]
    launched = obj_z > 0.10
    tipped = (obs[:, robot_dim + 3].abs() > TIP_OVER_THRESHOLD) | \
             (obs[:, robot_dim + 4].abs() > TIP_OVER_THRESHOLD)
    obj_pos = obs[:, robot_dim: robot_dim + 3]
    goal_pos = obs[:, robot_dim + obj_state_dim: robot_dim + obj_state_dim + 3]
    out_of_bounds = (obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1) > 0.5

    both_success = at_goal
    done = terminated | max_push_done | both_success | launched | tipped | out_of_bounds

    pos_only = torch.zeros_like(done)
    if pos_term_threshold > 0.0:
        pos_err = (obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
        pos_only = pos_err < pos_term_threshold
        done = done | pos_only

    reasons = {
        "terminated": terminated,
        "max_pushes": max_push_done,
        "success": both_success,
        "launched": launched,
        "tipped": tipped,
        "oob": out_of_bounds,
        "pos_only": pos_only,
    }
    return done, reasons


def compute_dpose(
    obj_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    obj_yaw: torch.Tensor,
    goal_yaw: torch.Tensor,
    char_length: float,
) -> torch.Tensor:
    dx = obj_pos[..., 0] - goal_pos[..., 0]
    dy = obj_pos[..., 1] - goal_pos[..., 1]
    dtheta = ((obj_yaw - goal_yaw + math.pi) % (2.0 * math.pi)) - math.pi
    d_sq = dx ** 2 + dy ** 2 + char_length ** 2 * dtheta ** 2
    return torch.sqrt(d_sq + 1e-12)


def dpose_and_zero_yaw(
    obs: torch.Tensor,
    robot_dim: int,
    obj_state_dim: int,
    goal_dim: int,
    char_length: float,
) -> torch.Tensor:
    """Shared d_pose observation transform used identically by the training
    wrapper (`_get_push_obs`, `construct_bob_observation`) and the validator
    (`_build_obs`), so training and every eval path produce byte-identical obs.

    Writes the [d_pose, bearing] tail into the last two slots, and — when
    ``char_length == 0`` (the orientation-free disc subspace) — zeros the object
    and goal YAW obs slots so a T-block-trained policy sees a disc scene as a
    yaw-aligned in-distribution input. Roll/pitch are left intact (tip-over
    detection still reads them). Operates in place and returns ``obs``.

    obs layout: [robot(robot_dim) | obj_state(obj_state_dim) | goal_pose(goal_dim) | tail(2)]
    """
    dist_idx = robot_dim + obj_state_dim + goal_dim
    obj_pos = obs[:, robot_dim:robot_dim + 3]
    obj_yaw = obs[:, robot_dim + 5]
    goal_pos = obs[:, robot_dim + obj_state_dim:robot_dim + obj_state_dim + 3]
    goal_yaw = obs[:, robot_dim + obj_state_dim + 5]
    d_pose = compute_dpose(obj_pos, goal_pos, obj_yaw, goal_yaw, char_length)
    dx = goal_pos[:, 0] - obj_pos[:, 0]
    dy = goal_pos[:, 1] - obj_pos[:, 1]
    obs[:, dist_idx] = d_pose
    obs[:, dist_idx + 1] = torch.atan2(dy, dx)
    # Tolerant zero-test: any char_length below this is the orientation-free disc
    # subspace. Exact == 0.0 was fragile (a future 1e-9 would silently disable the
    # yaw-zeroing while d_pose stayed ~position-only → subtle train/eval mismatch).
    if char_length < 1e-6:
        obs[:, robot_dim + 5] = 0.0
        obs[:, robot_dim + obj_state_dim + 5] = 0.0
    return obs


def potential_dpose(
    obj_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    obj_yaw: torch.Tensor,
    goal_yaw: torch.Tensor,
    char_length: float,
    k: float = PBRS_K_DPOSE,
) -> torch.Tensor:
    dx = obj_pos[..., 0] - goal_pos[..., 0]
    dy = obj_pos[..., 1] - goal_pos[..., 1]
    dtheta = ((obj_yaw - goal_yaw + math.pi) % (2.0 * math.pi)) - math.pi
    d_sq = dx ** 2 + dy ** 2 + char_length ** 2 * dtheta ** 2
    return torch.exp(-k * d_sq)


def compute_pbrs_reward_dpose(
    cur_obj_pos: torch.Tensor,
    cur_obj_euler: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_euler: torch.Tensor,
    prev_phi_dpose: torch.Tensor,
    gave_completion: torch.Tensor,
    char_length: float,
    k: float = PBRS_K_DPOSE,
    w: float = PBRS_W_DPOSE,
    success_threshold: float = PBRS_DPOSE_THRESHOLD_TBLOCK,
):
    obj_yaw = cur_obj_euler[..., 2]
    goal_yaw = goal_euler[..., 2]

    phi_now = potential_dpose(cur_obj_pos, goal_pos, obj_yaw, goal_yaw, char_length, k)
    dense = w * (phi_now - prev_phi_dpose)
    reward = dense.clone()

    d_pose = compute_dpose(cur_obj_pos, goal_pos, obj_yaw, goal_yaw, char_length)
    at_goal = d_pose < success_threshold

    new_completion = at_goal & ~gave_completion
    reward += new_completion.float() * PBRS_COMPLETION_BONUS
    gave_completion = gave_completion | new_completion

    tipped = (cur_obj_euler[:, 0].abs() > TIP_OVER_THRESHOLD) | \
             (cur_obj_euler[:, 1].abs() > TIP_OVER_THRESHOLD)
    reward += tipped.float() * PBRS_TIP_PENALTY

    return {
        "reward": reward,
        "phi_now": phi_now,
        "d_pose": d_pose,
        "at_goal": at_goal,
        "tipped": tipped,
        "dense": dense,
        "gave_completion": gave_completion,
    }
