"""
Cross-environment evaluation: validate Isaac-trained push models in gym-pusht.

Loads PPO/ASP/SAC checkpoints, bridges observations/actions to the 2D pymunk
PushT environment, and runs the same 30 test scenes from validation_configs.py.
Uses Isaac success thresholds (pos<0.05m, rot<0.2rad) and best-of-3 retries.

No Isaac Lab or cuRobo required — runs entirely in the gym-pusht conda env.

Usage:
  # PPO baseline (Model A):
  python -m asyncDualPlayPPO.tests.validate_push_pusht \
      --model-type ppo --rel-obs --rel-act \
      --chkpt runs/ppo_pbrs_reward/26.06.18/runs/hpc_pbrs_simp_528env/agent/model_best.pt \
      --num_tests 20 --csv results_pusht.csv

  # ASP Bob (Model C):
  python -m asyncDualPlayPPO.tests.validate_push_pusht \
      --model-type asp --rel-obs --rel-act \
      --chkpt runs/ppo_pbrs_reward/26.06.18/runs/hpc_pbrs_asp_528env/bob/model_best.pt \
      --num_tests 20 --csv results_pusht_asp.csv

  # SAC baseline:
  python -m asyncDualPlayPPO.tests.validate_push_pusht \
      --model-type sac \
      --chkpt runs/sac/26.06.26/.../latest_checkpoint.zip \
      --num_tests 20 --csv results_pusht_sac.csv
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asyncDualPlayPPO.tasks.utils.validation_configs import get_test_config, get_test_count

# ============================================================================
# Coordinate transform (Isaac metres <-> pusht pixels)
# ============================================================================

class PushtBridge:
    """Maps Isaac metre coordinates to pusht pixel coordinates.

    Isaac workspace:  X in [-0.50, 0.50], Y in [0.25, 0.70]
    Isaac centre:      X=0.0, Y=0.475 (midpoint of Y range)
    Pusht:             origin at top-left, Y-down

    Transform: px = arena_width/2  + (ix - cx) * ppm       (X: no flip)
               py = arena_height/2 - (iy - cy) * ppm       (Y: flipped)
    """

    def __init__(self, arena_width=800, arena_height=800, m_per_px=1.0 / 750.0,
                 isaac_center_x=0.0, isaac_center_y=0.475,
                 yaw_sign=1.0, yaw_offset=0.0):
        self.aw = arena_width
        self.ah = arena_height
        self.ppm = 1.0 / m_per_px
        self.mpp = m_per_px
        self.cx = isaac_center_x
        self.cy = isaac_center_y
        self.yaw_sign = yaw_sign
        self.yaw_offset = yaw_offset

    def i2p(self, ix, iy):
        """Isaac metres -> pusht pixels."""
        px = self.aw / 2.0 + (ix - self.cx) * self.ppm
        py = self.ah / 2.0 - (iy - self.cy) * self.ppm
        return float(px), float(py)

    def p2i(self, px, py):
        """Pusht pixels -> Isaac metres."""
        ix = self.cx + (px - self.aw / 2.0) / self.ppm
        iy = self.cy - (py - self.ah / 2.0) / self.ppm
        return float(ix), float(iy)

    def i2p_yaw(self, iyaw):
        return iyaw * self.yaw_sign + self.yaw_offset

    def p2i_yaw(self, pyaw):
        return (pyaw - self.yaw_offset) / self.yaw_sign


# Isaac observation constants
_EE_Z_APPROACH = 0.50
_TABLE_Z = 0.02
_EE_ROLL = math.pi          # tool-down (quat [0,1,0,0] -> Euler ZYX = [pi, 0, 0])
_EE_PITCH = 0.0
_EE_YAW = 0.0

# Observation slot dimensions
_OBS_ROBOT_DIM = 6
_OBS_OBJ_STATE_DIM = 14
_OBS_GOAL_DIM = 6
_OBS_DIST_DIM = 2
_OBS_TOTAL = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + _OBS_GOAL_DIM + _OBS_DIST_DIM  # 28

# Isaac success thresholds (in metres/radians)
_SUCCESS_POS = 0.05
_SUCCESS_ROT = 0.20

# ============================================================================
# Validation result
# ============================================================================

@dataclass
class ValidationResult:
    test_index: int
    test_name: str
    test_type: str
    success: bool
    pushes_used: int
    final_pos_error: float
    final_rot_error: float
    area_coverage: float


# ============================================================================
# Utility functions
# ============================================================================

def _rot_distance_rad(yaw_a, yaw_b):
    diff = abs(yaw_a - yaw_b)
    diff = min(diff, 2.0 * math.pi - diff)
    return diff


def _area_coverage(pos_err, rot_err):
    pc = max(0.0, 1.0 - pos_err / 0.10)
    rc = max(0.0, 1.0 - rot_err / 0.40)
    return pc * rc * 100.0


# ============================================================================
# Isaac observation builder
# ============================================================================

def _build_rest_isaac_obs(block_px, block_py, block_angle,
                          goal_im, goal_jm, goal_yaw_i,
                          bridge: PushtBridge, rel_obs=False):
    """Build Isaac observation (28D or 30D) from pusht state, agent at rest position."""
    bx, by = bridge.p2i(block_px, block_py)
    byaw = bridge.p2i_yaw(block_angle)
    gyaw = goal_yaw_i

    # Robot EE (6D Euler ZYX) at arena centre
    rx, ry = bridge.p2i(bridge.aw / 2.0, bridge.ah / 2.0)
    robot = torch.tensor([rx, ry, _EE_Z_APPROACH, _EE_ROLL, _EE_PITCH, _EE_YAW])

    # Object state (14D)
    obj_pos = torch.tensor([bx, by, _TABLE_Z])
    obj_euler = torch.tensor([0.0, 0.0, byaw])
    obj_vel = torch.zeros(6)
    d_agent = math.sqrt((rx - bx) ** 2 + (ry - by) ** 2)
    obj_extra = torch.tensor([d_agent, 0.0])

    # Goal state (6D)
    goal_obs = torch.tensor([goal_im, goal_jm, _TABLE_Z, 0.0, 0.0, gyaw])

    # Distance (2D)
    pos_dist = math.sqrt((bx - goal_im) ** 2 + (by - goal_jm) ** 2)
    rot_dist = _rot_distance_rad(byaw, gyaw)
    dist_obs = torch.tensor([pos_dist, rot_dist])

    obs = torch.cat([robot, obj_pos, obj_euler, obj_vel, obj_extra, goal_obs, dist_obs])

    if rel_obs:
        obs = torch.cat([obs, torch.tensor([goal_im - bx, goal_jm - by])])

    return obs.unsqueeze(0)  # (1, obs_dim)


# ============================================================================
# Pusht push executor
# ============================================================================

def _execute_push(env, Xs_px, Ys_px, Xf_px, Yf_px, n_steps=15):
    """Execute one push: teleport agent to Xs, sweep to Xf, rest at arena centre."""
    # Teleport to approach point
    env.unwrapped.agent.position = (Xs_px, Ys_px)
    env.unwrapped.agent.velocity = (0.0, 0.0)
    env.unwrapped.space.step(env.unwrapped.dt)

    # Sweep along push line
    for i in range(1, n_steps + 1):
        t = i / n_steps
        tx = Xs_px + t * (Xf_px - Xs_px)
        ty = Ys_px + t * (Yf_px - Ys_px)
        action = np.array([tx, ty], dtype=np.float32)
        env.step(action)

    # Move agent to arena centre (rest)
    rest_x = env.unwrapped.arena_width / 2.0
    rest_y = env.unwrapped.arena_height / 2.0
    env.unwrapped.agent.position = (rest_x, rest_y)
    env.unwrapped.agent.velocity = (0.0, 0.0)
    env.unwrapped.space.step(env.unwrapped.dt)


# ============================================================================
# Model loading
# ============================================================================

def _load_ppo(chkpt_path, obs_dim, device):
    """Load single-agent PPO ActorCriticPush."""
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush

    ckpt_raw = torch.load(chkpt_path, map_location=device)
    ckpt = ckpt_raw["model_state_dict"] if "model_state_dict" in ckpt_raw else ckpt_raw

    # Auto-detect obs_dim from critic first-layer weight shape
    critic_w0 = ckpt.get("critic.0.weight")
    if critic_w0 is not None:
        detected_obs_dim = int(critic_w0.shape[1])
        if detected_obs_dim != obs_dim:
            print(f"[_load_ppo] obs_dim {obs_dim} -> auto-detected {detected_obs_dim} from checkpoint")
            obs_dim = detected_obs_dim

    N = 4  # num_cat_dims
    B = 21  # num_bins
    cfg = {
        "policy": {
            "use_multicategorical": True, "num_cat_dims": N, "num_bins": B,
            "use_lstm": True, "lstm_hidden_size": 256,
            "pi_hid_sizes": [512, 256, 128],
            "vf_hid_sizes": [512, 256, 128],
            "activation": "relu",
        },
    }
    ac = ActorCriticPush(
        obs_shape=(obs_dim,), states_shape=(0,),
        actions_shape=(N,), init_noise_std=0.3, model_cfg=cfg["policy"],
        asymmetric=False,
    ).to(device)
    ckpt = torch.load(chkpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    ac.load_state_dict(ckpt, strict=False)
    ac.eval()
    return ac, N, B, obs_dim


def _load_asp(chkpt_path, obs_dim, device):
    """Load ASP Bob ActorCritic (with or without GoalEncoder)."""
    from asyncDualPlayPPO.algorithms.rl.ppo.module import ActorCritic

    ckpt = torch.load(chkpt_path, map_location=device)
    ckpt_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Auto-detect obs_dim from critic first-layer weight shape
    critic_w0 = ckpt_state.get("critic.0.weight")
    if critic_w0 is not None:
        detected_obs_dim = int(critic_w0.shape[1])
        if detected_obs_dim != obs_dim:
            print(f"[_load_asp] obs_dim {obs_dim} -> auto-detected {detected_obs_dim} from checkpoint")
            obs_dim = detected_obs_dim

    # Auto-detect GoalEncoder presence from pi-encoder first-layer weight shape
    pi_w0 = ckpt_state.get("pi_encoder.obj_encoder.0.weight")
    is_noge = pi_w0 is not None and pi_w0.shape[1] == 22
    has_ge = not is_noge

    # Auto-detect model dimensions from checkpoint weight shapes
    if has_ge:
        ge_K = int(ckpt_state["_goal_proj.weight"].shape[1])       # ge_raw_dim
        trunk_in = int(ckpt_state["actor_trunk_layer1.weight"].shape[1])
        ge_hidden = int(ckpt_state["goal_encoder.phi.2.weight"].shape[1])
        pi_obj_dim = 14
        robot_state_dim = 6
        pi_emb_dim = trunk_in - robot_state_dim
    else:
        ge_K = 0
        ge_hidden = 0
        pi_obj_dim = 22
        robot_state_dim = 6
        pi_emb_dim = 512

    N = 4
    B = 21
    cfg = {
        "policy": {
            "use_multicategorical": True, "num_cat_dims": N, "num_bins": B,
            "use_lstm": True, "lstm_hidden_size": 256,
            "pi_hid_sizes": [512, 256, 128],
            "vf_hid_sizes": [512, 256, 128],
            "activation": "relu",
            "use_goal_encoder": has_ge,
            "ge_raw_per_obj": 22,
            "ge_raw_dim": ge_K,
            "goal_embed_dim": ge_K,
            "goal_encoder_hidden_dim": ge_hidden,
            "goal_encoder_variant": "difference",
            "num_objects": 1,
            "robot_state_dim": robot_state_dim,
            "pi_obj_dim": pi_obj_dim,
            "pi_emb_dim": pi_emb_dim,
            "use_pi_encoder": True,
        },
    }
    ac = ActorCritic(
        obs_shape=(obs_dim,), states_shape=(0,),
        actions_shape=(N,), initial_std=0.3, model_cfg=cfg["policy"],
        asymmetric=False,
    ).to(device)
    ac.load_state_dict(ckpt_state, strict=False)
    ac.eval()
    return ac, N, B, obs_dim


def _load_sac(chkpt_path, device):
    """Load SAC+HER model via stable-baselines3."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecEnv

    obs_fdim = 22
    goal_dim = 3
    obs_sp = gym.spaces.Dict({
        "observation":  gym.spaces.Box(-np.inf, np.inf, shape=(obs_fdim,), dtype=np.float32),
        "achieved_goal": gym.spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        "desired_goal":  gym.spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        "policy":        gym.spaces.Box(-np.inf, np.inf, shape=(obs_fdim,), dtype=np.float32),
    })
    act_sp = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    class _DV(VecEnv):
        def __init__(self): super().__init__(1, obs_sp, act_sp)
        def step_async(self, a): pass
        def step_wait(self): return np.zeros((1, 0)), np.zeros(1), np.ones(1, bool), [{}]
        def reset(self): return {}, {}
        def close(self): pass
        def get_attr(self, n, i=None):
            if n == "render_mode": return ["rgb_array"]
            return []
        def set_attr(self, n, v, i=None): pass
        def env_method(self, m, *a, i=None, **kw): return []
        def env_is_wrapped(self, w, i=None): return [False]
        def seed(self, s=None): return None
        def get_images(self): return []

    model = SAC.load(chkpt_path, env=_DV())
    model.policy.set_training_mode(False)
    return model


# ============================================================================
# SAC observation builder
# ============================================================================

def _build_sac_dict(block_px, block_py, block_angle,
                    goal_im, goal_jm, goal_yaw_i,
                    bridge: PushtBridge):
    """Build SAC dict observation (22D feature, 3D achieved/desired goal)."""
    bx, by = bridge.p2i(block_px, block_py)
    byaw = bridge.p2i_yaw(block_angle)
    gyaw = goal_yaw_i
    rx, ry = bridge.p2i(bridge.aw / 2.0, bridge.ah / 2.0)

    pos_dist = math.sqrt((bx - goal_im) ** 2 + (by - goal_jm) ** 2)
    rot_dist = _rot_distance_rad(byaw, gyaw)

    obs_vec = np.array([
        rx, ry, _EE_Z_APPROACH, _EE_ROLL, _EE_PITCH, _EE_YAW,
        bx, by, _TABLE_Z, 0.0, 0.0, byaw,
        goal_im, goal_jm, _TABLE_Z, 0.0, 0.0, gyaw,
        pos_dist, rot_dist,
        goal_im - bx, goal_jm - by,
    ], dtype=np.float32)

    ach = np.array([bx, by, byaw], dtype=np.float32)
    des = np.array([goal_im, goal_jm, gyaw], dtype=np.float32)

    return {
        "observation": obs_vec.reshape(1, -1),
        "achieved_goal": ach.reshape(1, -1),
        "desired_goal": des.reshape(1, -1),
        "policy": obs_vec.reshape(1, -1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Environment Validation (Isaac -> pusht)")
    parser.add_argument("--chkpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["ppo", "asp", "sac"], help="Model family")
    parser.add_argument("--num_tests", type=int, default=30, help="Test scenes (T-block only)")
    parser.add_argument("--max_pushes", type=int, default=15, help="Max pushes per episode")
    parser.add_argument("--rot_threshold", type=float, default=_SUCCESS_ROT,
                        help="Rotation success threshold (rad)")
    parser.add_argument("--rel-obs", action="store_true", dest="rel_obs",
                        help="Append relative goal delta to observation (28->30D)")
    parser.add_argument("--rel-act", action="store_true", dest="rel_act",
                        help="Decode actions object-relative")
    parser.add_argument("--arena-width", type=int, default=800, help="pusht arena width (px)")
    parser.add_argument("--arena-height", type=int, default=800, help="pusht arena height (px)")
    parser.add_argument("--m-per-px", type=float, default=1.0 / 750.0,
                        help="Metres per pixel (default 1/750 for T-size matching)")
    parser.add_argument("--yaw-sign", type=float, default=1.0, help="Yaw sign multiplier (+/-1)")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Yaw offset (rad)")
    parser.add_argument("--push-steps", type=int, default=15,
                        help="pusht physics steps per push")
    parser.add_argument("--csv", type=str, default=None, help="CSV output path")
    parser.add_argument("--render", action="store_true", help="Render in human mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # Bridge
    bridge = PushtBridge(
        arena_width=args.arena_width, arena_height=args.arena_height,
        m_per_px=args.m_per_px,
        yaw_sign=args.yaw_sign, yaw_offset=args.yaw_offset,
    )
    print(f"[Setup] Arena {args.arena_width}x{args.arena_height} px  "
          f"scale {1.0 / args.m_per_px:.0f} px/m  "
          f"yaw_sign={args.yaw_sign}  yaw_offset={args.yaw_offset}")

    # Create pusht environment
    render_mode = "human" if args.render else "rgb_array"
    import gym_pusht  # noqa: F401
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode=render_mode,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
    )
    print(f"[Setup] gym-pusht: arena={env.unwrapped.arena_width}x{env.unwrapped.arena_height}")

    obs_dim = _OBS_TOTAL + (2 if args.rel_obs else 0)
    print(f"[Setup] Observation dim: {obs_dim}")

    # Load model
    if args.model_type == "ppo":
        ac, num_cat_dims, num_bins, obs_dim = _load_ppo(args.chkpt, obs_dim, device)
    elif args.model_type == "asp":
        ac, num_cat_dims, num_bins, obs_dim = _load_asp(args.chkpt, obs_dim, device)
    elif args.model_type == "sac":
        sac_model = _load_sac(args.chkpt, device)

    # Auto-detect whether rel_obs is needed based on detected obs_dim
    detected_rel_obs = obs_dim > _OBS_TOTAL
    if detected_rel_obs != args.rel_obs:
        print(f"[Validate] rel_obs flag overridden: {args.rel_obs} -> {detected_rel_obs} (obs_dim={obs_dim})")
        args.rel_obs = detected_rel_obs

    # Action decoders (isaaclab-free)
    from asyncDualPlayPPO.tasks.utils.action_push import decode_push_action
    from asyncDualPlayPPO.tasks.utils.action_push_relative import decode_push_action_relative
    if args.model_type == "sac":
        from asyncDualPlayPPO.tasks.utils.action_push_continuous import \
            decode_push_action_relative_continuous

    print(f"[Validate] model_type={args.model_type}  rel_obs={args.rel_obs}  "
          f"rel_act={args.rel_act}  rot_threshold={args.rot_threshold:.3f} rad  "
          f"max_pushes={args.max_pushes}  push_steps={args.push_steps}")

    results: List[ValidationResult] = []
    test_cfgs: List[dict] = []
    n_tests = min(args.num_tests, get_test_count())

    for test_idx in range(1, n_tests + 1):
        cfg = get_test_config(test_idx)
        if cfg is None:
            continue

        obj_type = getattr(cfg, "object_type", "tblock")
        if obj_type == "disc":
            print(f"\n[Test {test_idx}/{n_tests}] {cfg.name} #{cfg.test_id}  SKIP (disc not in pusht)")
            continue

        # Count only T-block tests for display
        tblock_idx = sum(1 for c in test_cfgs if c["object_type"] == "tblock") + 1
        print(f"\n[Test {test_idx}/{n_tests}] [{tblock_idx}] {cfg.name} #{cfg.test_id}")

        best_pos_err = float("inf")
        best_rot_err = float("inf")
        best_stop = "max_pushes"
        best_pushes = 0
        best_success = False
        retry_count = 0
        MAX_RETRIES = 3

        for retry in range(MAX_RETRIES):
            retry_count = retry + 1

            # Reset pusht with block at start
            spx, spy = bridge.i2p(cfg.main_start.x, cfg.main_start.y)
            syaw = bridge.i2p_yaw(getattr(cfg.main_start, "yaw", 0.0))
            state = np.array([spx, spy, spx, spy, syaw], dtype=np.float64)
            env.reset(options={"reset_to_state": state})

            # Set goal pose
            gpx, gpy = bridge.i2p(cfg.main_goal_x, cfg.main_goal_y)
            gyaw_p = bridge.i2p_yaw(cfg.main_goal_yaw)
            env.unwrapped.goal_pose = np.array([gpx, gpy, gyaw_p])

            # Read initial state
            bp_x = env.unwrapped.block.position.x
            bp_y = env.unwrapped.block.position.y
            bp_a = env.unwrapped.block.angle

            init_pos_err = math.sqrt(
                (bridge.p2i(bp_x, bp_y)[0] - cfg.main_goal_x) ** 2
                + (bridge.p2i(bp_x, bp_y)[1] - cfg.main_goal_y) ** 2
            )
            init_rot_err = _rot_distance_rad(
                bridge.p2i_yaw(bp_a), cfg.main_goal_yaw
            )
            init_oob_2d = init_pos_err

            rtag = f"[R{retry_count}] " if retry > 0 else ""
            print(f"  {rtag}[{cfg.test_type}] "
                  f"goal=({cfg.main_goal_x:+.3f},{cfg.main_goal_y:+.3f}) yaw={cfg.main_goal_yaw:+.3f}  "
                  f"start=({cfg.main_start.x:+.3f},{cfg.main_start.y:+.3f})  "
                  f"init_pos_err={init_pos_err:.4f}m  init_rot_err={init_rot_err:.3f}rad")

            # Build initial Isaac observation
            obs = _build_rest_isaac_obs(
                bp_x, bp_y, bp_a,
                cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw,
                bridge, rel_obs=args.rel_obs,
            ).to(device)

            # LSTM hidden state
            if args.model_type in ("ppo", "asp"):
                hidden = [
                    torch.zeros(1, ac.lstm_hidden_size, device=device),
                    torch.zeros(1, ac.lstm_hidden_size, device=device),
                ]

            test_success = False
            pushes_used = 0
            stop_reason = "max_pushes"
            pos_err = init_pos_err
            rot_err = init_rot_err
            prev_pos_err = init_pos_err
            prev_rot_err = init_rot_err

            for push_i in range(args.max_pushes):
                # ---- policy inference ----
                if args.model_type in ("ppo", "asp"):
                    with torch.no_grad():
                        result = ac.act_with_hidden(obs, None, (hidden[0], hidden[1]))
                        if len(result) == 7:
                            actions, _, _, _, _, _, new_h = result
                        else:
                            actions, _, _, _, _, new_h = result
                        if new_h is not None:
                            hidden[0], hidden[1] = new_h[0], new_h[1]

                    if args.rel_act:
                        obj_x = obs[0, _OBS_ROBOT_DIM]
                        obj_y = obs[0, _OBS_ROBOT_DIM + 1]
                        obj_yaw = obs[0, _OBS_ROBOT_DIM + 5]
                        Xs, Ys, length, theta = decode_push_action_relative(
                            actions,
                            torch.stack([obj_x, obj_y]).unsqueeze(0),
                            obj_yaw.unsqueeze(0),
                            num_bins=num_bins,
                        )
                    else:
                        Xs, Ys, length, theta = decode_push_action(actions, num_bins=num_bins)
                    bins_str = ", ".join(f"{int(actions[0, i].item()):2d}" for i in range(4))
                else:
                    sac_obs = _build_sac_dict(
                        env.unwrapped.block.position.x,
                        env.unwrapped.block.position.y,
                        env.unwrapped.block.angle,
                        cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw,
                        bridge,
                    )
                    a_np, _ = sac_model.predict(sac_obs, deterministic=True)
                    a_t = torch.from_numpy(a_np).float().to(device)

                    obj_x = obs[0, _OBS_ROBOT_DIM]
                    obj_y = obs[0, _OBS_ROBOT_DIM + 1]
                    obj_yaw = obs[0, _OBS_ROBOT_DIM + 5]
                    Xs, Ys, length, theta = decode_push_action_relative_continuous(
                        a_t,
                        torch.stack([obj_x, obj_y]).unsqueeze(0),
                        obj_yaw.unsqueeze(0),
                    )
                    bins_str = f"{a_np[0,0]:+.3f},{a_np[0,1]:+.3f},{a_np[0,2]:+.3f},{a_np[0,3]:+.3f}"

                # ---- decode to world coords ----
                Xs_f = float(Xs.item())
                Ys_f = float(Ys.item())
                length_f = float(length.item())
                theta_f = float(theta.item())

                # Clamp to Isaac IK workspace (same as validate_push.py)
                Xs_f = max(-0.50, min(0.50, Xs_f))
                Ys_f = max(0.25, min(0.70, Ys_f))

                Xf = Xs_f + length_f * math.cos(theta_f)
                Yf = Ys_f + length_f * math.sin(theta_f)
                Xf = max(-0.50, min(0.50, Xf))
                Yf = max(0.25, min(0.70, Yf))

                recalc_len = math.sqrt((Xf - Xs_f) ** 2 + (Yf - Ys_f) ** 2)
                if recalc_len < 0.001:
                    length_f = 0.0

                # Convert to pusht pixels
                Xs_px, Ys_px = bridge.i2p(Xs_f, Ys_f)
                Xf_px, Yf_px = bridge.i2p(Xf, Yf)

                # Clamp to arena bounds (10 px margin from walls)
                Xs_px = max(15, min(bridge.aw - 15, Xs_px))
                Ys_px = max(15, min(bridge.ah - 15, Ys_px))
                Xf_px = max(15, min(bridge.aw - 15, Xf_px))
                Yf_px = max(15, min(bridge.ah - 15, Yf_px))

                # Execute push
                _execute_push(env, Xs_px, Ys_px, Xf_px, Yf_px, n_steps=args.push_steps)

                pushes_used = push_i + 1

                # Read state after push
                bp_x = env.unwrapped.block.position.x
                bp_y = env.unwrapped.block.position.y
                bp_a = env.unwrapped.block.angle

                bx_i, by_i = bridge.p2i(bp_x, bp_y)
                byaw_i = bridge.p2i_yaw(bp_a)

                pos_err = math.sqrt((bx_i - cfg.main_goal_x) ** 2
                                    + (by_i - cfg.main_goal_y) ** 2)
                rot_err = _rot_distance_rad(byaw_i, cfg.main_goal_yaw)
                oob_2d = math.sqrt((bx_i - cfg.main_goal_x) ** 2
                                   + (by_i - cfg.main_goal_y) ** 2)

                # Logging
                if args.rel_act:
                    r_dec = math.sqrt(
                        (Xf - float(obs[0, _OBS_ROBOT_DIM])) ** 2
                        + (Yf - float(obs[0, _OBS_ROBOT_DIM + 1])) ** 2
                    )
                else:
                    r_dec = Xs_f
                cov = _area_coverage(pos_err, rot_err)
                print(f"  push {push_i:2d}: bins=({bins_str})  "
                      f"r={r_dec:.3f} len={length_f:.3f} theta={math.degrees(theta_f):.0f}deg  "
                      f"pos={pos_err:.4f}m rot={rot_err:.3f}rad cov={cov:.1f}%")

                # Success check
                if pos_err < _SUCCESS_POS and rot_err < args.rot_threshold:
                    test_success = True

                # Early termination
                if oob_2d > init_oob_2d + 0.20:
                    stop_reason = "oob"
                    break

                # Track best
                if pos_err < best_pos_err:
                    best_pos_err = pos_err
                    best_rot_err = rot_err

                prev_pos_err = pos_err
                prev_rot_err = rot_err

                if test_success:
                    break

                # Build next observation
                obs = _build_rest_isaac_obs(
                    bp_x, bp_y, bp_a,
                    cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw,
                    bridge, rel_obs=args.rel_obs,
                ).to(device)

            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_rot_err = rot_err
            if test_success:
                stop_reason = "success"

            best_success = best_success or test_success
            if best_pushes == 0:
                best_pushes = pushes_used
            best_stop = stop_reason

            if test_success:
                break
            if stop_reason in ("max_pushes",):
                break

        if retry_count > 1:
            best_stop += f"_r{retry_count}"

        result = ValidationResult(
            test_index=test_idx,
            test_name=f"{cfg.name} #{cfg.test_id}",
            test_type=cfg.test_type,
            success=best_success,
            pushes_used=best_pushes,
            final_pos_error=best_pos_err,
            final_rot_error=best_rot_err,
            area_coverage=_area_coverage(best_pos_err, best_rot_err),
        )
        results.append(result)
        test_cfgs.append({
            "start_x": cfg.main_start.x, "start_y": cfg.main_start.y,
            "goal_x": cfg.main_goal_x, "goal_y": cfg.main_goal_y,
            "goal_yaw": cfg.main_goal_yaw,
            "object_type": obj_type,
        })
        status = "PASS" if best_success else "FAIL"
        print(f"  {status} | pushes: {best_pushes} | reason: {best_stop} | "
              f"pos_err: {best_pos_err:.4f} | rot_err: {best_rot_err:.4f} | "
              f"cov: {_area_coverage(best_pos_err, best_rot_err):.1f}%")

    # ----- Summary -----
    n_ok = sum(1 for r in results if r.success)
    sr = n_ok / len(results) * 100 if results else 0
    avg_p = np.mean([r.pushes_used for r in results]) if results else 0
    po = [r for r in results if r.test_type == "pos_only"]
    pr = [r for r in results if r.test_type == "pos_rot"]
    sr_po = sum(1 for r in po if r.success) / len(po) * 100 if po else 0
    sr_pr = sum(1 for r in pr if r.success) / len(pr) * 100 if pr else 0
    avg_c = np.mean([r.area_coverage for r in results]) if results else 0

    print(f"\n{'=' * 60}")
    print(f"PUSHT VALIDATION  ({args.model_type.upper()})")
    print(f"{'=' * 60}")
    print(f"  Total tests:   {len(results)}")
    print(f"  Successes:     {n_ok}")
    print(f"  Success rate:  {sr:.1f}%")
    print(f"  Pos-only SR:   {sr_po:.1f}%  ({len(po)} tests)")
    print(f"  Pos+rot SR:    {sr_pr:.1f}%  ({len(pr)} tests)")
    print(f"  Avg pushes:    {avg_p:.1f}")
    print(f"  Avg coverage:  {avg_c:.1f}%")
    print(f"{'=' * 60}")

    for r in results:
        s = "PASS" if r.success else "FAIL"
        print(f"  {s:5s} | Test {r.test_index:2d} | {r.test_name:30s} | pushes={r.pushes_used:2d}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test_index", "test_name", "test_type", "success",
                         "pushes_used", "pos_err", "rot_err", "area_coverage"])
            for r in results:
                w.writerow([r.test_index, r.test_name, r.test_type, int(r.success),
                             r.pushes_used, r.final_pos_error, r.final_rot_error, r.area_coverage])
        print(f"\n[CSV] Saved: {args.csv}")

    env.close()


if __name__ == "__main__":
    main()
