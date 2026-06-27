"""
Validate trained push models against the gym-pusht 2D environment.

Maps observations from gym-pusht's 5D state to the network's expected
format, decodes push actions, and executes them via PD-controlled agent
movement.  Uses gym-pusht's native coverage metric for success.

Usage:
  # PPO single-agent:
  python -m asyncDualPlayPPO.tests.validate_pusht_gym \
      --chkpt runs/.../agent/model_best.pt --num-tests 30 --csv results.csv

  # ASP Bob:
  python -m asyncDualPlayPPO.tests.validate_pusht_gym \
      --chkpt-bob runs/.../bob/model_best.pt --num-tests 30 --csv results.csv

  # SAC:
  python -m asyncDualPlayPPO.tests.validate_pusht_gym \
      --chkpt-sac runs/.../latest_checkpoint.zip --num-tests 30 --csv results.csv
"""

import argparse
import copy
import math
import os
import signal
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

SCALE = 750.0
ARENA_WIDTH = 1050
ARENA_HEIGHT = 750
APPROACH_STEPS = 40
PUSH_STEPS = 60
APPROACH_THRESHOLD_PX = 5.0

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


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
    gym_coverage: float
    trial_count: int = 1
    success_count: int = 0


def meter_to_pixel(mx: float, my: float) -> Tuple[float, float]:
    px = (mx + 0.70) * SCALE
    py = (my + 0.10) * SCALE
    return px, py


def pixel_to_meter(px: float, py: float) -> Tuple[float, float]:
    mx = px / SCALE - 0.70
    my = py / SCALE - 0.10
    return mx, my


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def _area_coverage(pos_err, rot_err):
    pc = max(0.0, 1.0 - abs(pos_err) / 0.10)
    rc = max(0.0, 1.0 - abs(rot_err) / 0.40)
    return pc * rc * 100.0


def build_ppo_obs(agent_px, agent_py, block_px, block_py, block_angle,
                  goal_px, goal_py, goal_angle, device):
    eex, eey = pixel_to_meter(agent_px, agent_py)
    obj_x, obj_y = pixel_to_meter(block_px, block_py)
    gx, gy = pixel_to_meter(goal_px, goal_py)

    dist_to_ee = math.sqrt((eex - obj_x) ** 2 + (eey - obj_y) ** 2)
    pos_dist = math.sqrt((obj_x - gx) ** 2 + (obj_y - gy) ** 2)
    rot_diff = abs(block_angle - goal_angle)
    rot_diff = min(rot_diff, 2 * math.pi - rot_diff)
    rel_dx = gx - obj_x
    rel_dy = gy - obj_y

    ee_pose = torch.tensor([[eex, eey, 0.05, 0.0, 0.0, 0.0]], device=device)
    obj_state = torch.tensor([[
        obj_x, obj_y, 0.05, 0.0, 0.0, block_angle,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dist_to_ee, 1.0,
    ]], device=device)
    goal_pose = torch.tensor([[gx, gy, 0.02, 0.0, 0.0, goal_angle]], device=device)
    goal_dist = torch.tensor([[pos_dist, rot_diff]], device=device)
    rel_goal = torch.tensor([[rel_dx, rel_dy]], device=device)
    return torch.cat([ee_pose, obj_state, goal_pose, goal_dist, rel_goal], dim=-1)


def build_asp_obs(agent_px, agent_py, block_px, block_py, block_angle,
                  goal_px, goal_py, goal_angle, device):
    eex, eey = pixel_to_meter(agent_px, agent_py)
    obj_x, obj_y = pixel_to_meter(block_px, block_py)
    gx, gy = pixel_to_meter(goal_px, goal_py)

    dist_to_ee = math.sqrt((eex - obj_x) ** 2 + (eey - obj_y) ** 2)
    pos_dist = math.sqrt((obj_x - gx) ** 2 + (obj_y - gy) ** 2)
    rot_diff = abs(block_angle - goal_angle)
    rot_diff = min(rot_diff, 2 * math.pi - rot_diff)

    ee_pose = torch.tensor([[eex, eey, 0.05, 0.0, 0.0, 0.0]], device=device)
    obj_state = torch.tensor([[
        obj_x, obj_y, 0.05, 0.0, 0.0, block_angle,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dist_to_ee, 1.0,
    ]], device=device)
    goal_pose = torch.tensor([[gx, gy, 0.02, 0.0, 0.0, goal_angle]], device=device)
    goal_dist = torch.tensor([[pos_dist, rot_diff]], device=device)
    return torch.cat([ee_pose, obj_state, goal_pose, goal_dist], dim=-1)  # 28D


def build_sac_dict_obs(agent_px, agent_py, block_px, block_py, block_angle,
                       goal_px, goal_py, goal_angle, device):
    eex, eey = pixel_to_meter(agent_px, agent_py)
    obj_x, obj_y = pixel_to_meter(block_px, block_py)
    gx, gy = pixel_to_meter(goal_px, goal_py)

    dist_to_ee = math.sqrt((eex - obj_x) ** 2 + (eey - obj_y) ** 2)
    rel_dx = gx - obj_x
    rel_dy = gy - obj_y

    ee_pose = torch.tensor([eex, eey, 0.05, 0.0, 0.0, 0.0])
    obj_pos = torch.tensor([obj_x, obj_y, 0.05])
    obj_euler = torch.tensor([0.0, 0.0, float(block_angle)])
    vel = torch.zeros(6)
    dist_contact = torch.tensor([dist_to_ee, 1.0])
    rel_goal = torch.tensor([rel_dx, rel_dy])

    observation = torch.cat([ee_pose, obj_pos, obj_euler, vel, dist_contact, rel_goal], dim=-1)
    achieved_goal = torch.cat([obj_pos[:2], obj_euler[2:3]], dim=-1)
    desired_goal = torch.tensor([gx, gy, float(goal_angle)])

    return {
        "observation": observation.unsqueeze(0).cpu().numpy().astype(np.float32),
        "achieved_goal": achieved_goal.unsqueeze(0).cpu().numpy().astype(np.float32),
        "desired_goal": desired_goal.unsqueeze(0).cpu().numpy().astype(np.float32),
        "policy": observation.unsqueeze(0).cpu().numpy().astype(np.float32),
    }


def execute_push(env, Xs_m, Ys_m, length_m, theta_rad, cv=None, render_delay=5):
    Xs_px, Ys_px = meter_to_pixel(float(Xs_m), float(Ys_m))
    Xf_px = Xs_px + float(length_m) * SCALE * math.cos(float(theta_rad))
    Yf_px = Ys_px + float(length_m) * SCALE * math.sin(float(theta_rad))

    Xf_px = max(15.0, min(ARENA_WIDTH - 15, float(Xf_px)))
    Yf_px = max(15.0, min(ARENA_HEIGHT - 15, float(Yf_px)))
    Xs_px = max(15.0, min(ARENA_WIDTH - 15, float(Xs_px)))
    Ys_px = max(15.0, min(ARENA_HEIGHT - 15, float(Ys_px)))

    obs = None
    for step in range(APPROACH_STEPS):
        obs, _, _, _, _ = env.step(np.array([Xs_px, Ys_px], dtype=np.float32))
        if cv and step % 3 == 0:
            env.render()
        agent_x, agent_y = float(obs[0]), float(obs[1])
        dist = math.sqrt((agent_x - Xs_px) ** 2 + (agent_y - Ys_px) ** 2)
        if dist < APPROACH_THRESHOLD_PX:
            break

    for step in range(PUSH_STEPS):
        obs, _, _, _, _ = env.step(np.array([Xf_px, Yf_px], dtype=np.float32))
        if cv and step % 3 == 0:
            env.render()

    block_x, block_y = float(obs[2]), float(obs[3])
    block_angle = float(obs[4])

    obj_x_m, obj_y_m = pixel_to_meter(block_x, block_y)
    goal_x_m, goal_y_m = pixel_to_meter(float(env.goal_pose[0]), float(env.goal_pose[1]))
    pos_err_m = math.sqrt((obj_x_m - goal_x_m) ** 2 + (obj_y_m - goal_y_m) ** 2)

    rot_diff = abs(block_angle - float(env.goal_pose[2]))
    rot_diff = min(rot_diff, 2 * math.pi - rot_diff)
    rot_err = max(0.0, float(rot_diff))

    coverage = 0.0
    try:
        coverage = env._get_coverage() if hasattr(env, '_get_coverage') else 0.0
    except Exception:
        pass

    return obs, coverage, pos_err_m, rot_err


def main():
    parser = argparse.ArgumentParser(description="Validate Push Model in gym-pusht")
    parser.add_argument("--chkpt", type=str, default=None, help="PPO single-agent checkpoint")
    parser.add_argument("--chkpt-bob", type=str, default=None, help="ASP Bob checkpoint")
    parser.add_argument("--chkpt-sac", type=str, default=None, help="SAC checkpoint (.zip)")
    parser.add_argument("--num-tests", type=int, default=30, help="Number of test scenes")
    parser.add_argument("--max-pushes", type=int, default=30, help="Max pushes per trial")
    parser.add_argument("--max-tries", type=int, default=20, help="Trials per test")
    parser.add_argument("--rot-threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--render", action="store_true", help="Visualize pushes in real-time")
    parser.add_argument("--render-delay", type=int, default=5, help="Ms delay between render frames")
    args = parser.parse_args()

    if not any([args.chkpt, args.chkpt_bob, args.chkpt_sac]):
        parser.error("One of --chkpt, --chkpt-bob, or --chkpt-sac is required")

    if args.chkpt_bob:
        model_type = "asp"
        chkpt_path = os.path.abspath(args.chkpt_bob)
    elif args.chkpt_sac:
        model_type = "sac"
        chkpt_path = os.path.abspath(args.chkpt_sac)
    else:
        model_type = "ppo"
        chkpt_path = os.path.abspath(args.chkpt)

    if not os.path.isfile(chkpt_path):
        for cand in [
            os.path.join(chkpt_path, "bob", "model_best.pt"),
            os.path.join(chkpt_path, "bob", "latest_checkpoint.pt"),
            os.path.join(chkpt_path, "agent", "model_best.pt"),
            os.path.join(chkpt_path, "agent", "latest_checkpoint.pt"),
            os.path.join(chkpt_path, "latest_checkpoint.zip"),
        ]:
            if os.path.isfile(cand):
                chkpt_path = cand
                break
        else:
            print(f"[ERROR] Checkpoint not found: {args.chkpt or args.chkpt_bob or args.chkpt_sac}")
            sys.exit(1)

    if chkpt_path.endswith(".zip"):
        model_type = "sac"
    elif chkpt_path.endswith(".pt"):
        try:
            _peek = torch.load(chkpt_path, map_location="cpu", weights_only=False)
            _state = _peek.get("model_state_dict", _peek)
            _keys = list(_state.keys())
            if any("pi_encoder" in k for k in _keys) or any("goal_encoder" in k for k in _keys):
                model_type = "asp"
            elif any("actor_trunk" in k for k in _keys):
                model_type = "ppo"
            elif any("policy.actor" in k for k in _keys):
                model_type = "sac"
            del _peek, _state
            torch.cuda.empty_cache()
        except Exception:
            pass

    chkpt_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(chkpt_path)))
    if args.csv is None:
        args.csv = os.path.join(chkpt_run_dir, "validation_pusht_gym.csv")
    elif not os.path.isabs(args.csv):
        args.csv = os.path.join(chkpt_run_dir, args.csv)

    from asyncDualPlayPPO.tasks.utils.validation_configs import (
        get_test_config, get_test_count,
    )
    from asyncDualPlayPPO.tasks.utils.action_push import decode_push_action
    from asyncDualPlayPPO.tasks.utils.action_push_relative import decode_push_action_relative
    import gymnasium as gym_mc

    # ── Create gym-pusht environment ───────────────────────────────────────────
    from gym_pusht.envs.pusht import PushTEnv
    env = PushTEnv(arena_width=ARENA_WIDTH, arena_height=ARENA_HEIGHT, obs_type="state",
                   render_mode="human" if args.render else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    # ── Load checkpoint ────────────────────────────────────────────────────────
    num_cat_dims = 4
    num_bins = 21

    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    # ── Dummy vector env for PPO/PPOABC constructors ───────────────────────────
    class _DummyVecEnv:
        def __init__(self, obs_space, state_space, act_space):
            self.observation_space = obs_space
            self.state_space = state_space
            self.action_space = act_space
            self.num_envs = 1
            self.device = device if device != "cpu" else torch.device("cpu")

    _obs_ppo_d = gym_mc.spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float32)
    _obs_asp_d = gym_mc.spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)

    if model_type == "sac":
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import VecEnv

        class _SB3DummyVecEnv(VecEnv):
            def __init__(self, obs_sp, act_sp, num_envs=1):
                super().__init__(num_envs, obs_sp, act_sp)
            def step_async(self, a): pass
            def step_wait(self):
                return np.zeros((self.num_envs, 0)), np.zeros(self.num_envs), np.ones(self.num_envs, dtype=bool), [{}]
            def reset(self): return {}, {}
            def close(self): pass
            def get_attr(self, a, i=None): return ["rgb_array"] if a == "render_mode" else []
            def set_attr(self, a, v, i=None): pass
            def env_method(self, m, *a, i=None, **kw): return []
            def env_is_wrapped(self, w, i=None): return [False]
            def seed(self, s=None): return None
            def get_images(self): return []

        obs_feat_dim = 22
        goal_dim = 3
        obs_space = gym_mc.spaces.Dict({
            "observation": gym_mc.spaces.Box(-np.inf, np.inf, shape=(obs_feat_dim,), dtype=np.float32),
            "achieved_goal": gym_mc.spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            "desired_goal": gym_mc.spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            "policy": gym_mc.spaces.Box(-np.inf, np.inf, shape=(obs_feat_dim,), dtype=np.float32),
        })
        act_space = gym_mc.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        model = SAC.load(chkpt_path, env=_SB3DummyVecEnv(obs_space, act_space), seed=42)
        model.policy.set_training_mode(False)

        from asyncDualPlayPPO.tasks.utils.action_push_continuous import (
            decode_push_action_relative_continuous,
        )
        hidden = None
        print(f"[Validate] Loaded SAC from {chkpt_path}")

    elif model_type == "asp":
        ppo_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cfg/ppo/ppo_continuous.yaml")
        import yaml
        with open(ppo_cfg_path, "r") as f:
            ppo_cfg = yaml.safe_load(f)

        _chkpt_raw = torch.load(chkpt_path, map_location="cpu", weights_only=False)
        _chkpt_state = _chkpt_raw.get("model_state_dict", _chkpt_raw)
        _pi_w0 = _chkpt_state.get("pi_encoder.obj_encoder.0.weight")
        _is_noge = _pi_w0 is not None and _pi_w0.shape[1] == 22
        _has_goal_encoder = False if _is_noge else True
        del _chkpt_raw, _chkpt_state
        torch.cuda.empty_cache()

        from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC

        bob_cfg = copy.deepcopy(ppo_cfg["params"])
        bob_cfg["policy"]["use_pi_encoder"] = True
        bob_cfg["policy"]["use_multicategorical"] = True
        bob_cfg["policy"]["use_lstm"] = True
        bob_cfg["policy"]["use_goal_encoder"] = _has_goal_encoder
        bob_cfg["policy"]["num_cat_dims"] = num_cat_dims
        bob_cfg["policy"]["num_bins"] = num_bins
        bob_cfg["policy"]["robot_state_dim"] = 6
        if _has_goal_encoder:
            bob_cfg["policy"]["num_objects"] = 1
            bob_cfg["policy"]["goal_embed_dim"] = 8
        else:
            bob_cfg["policy"]["pi_obj_dim"] = 22

        bob_ppo = PPOABC(
            vec_env=_DummyVecEnv(_obs_asp_d, _obs_asp_d, _mc_space),
            cfg_train=bob_cfg, device=device,
            sampler="sequential", log_dir="/tmp/validate_pusht_gym",
            asymmetric=False,
        )
        bob_ppo.observation_space = _obs_asp_d
        bob_ppo.state_space = _obs_asp_d
        bob_ppo.action_space = _mc_space
        bob_ppo.desired_kl = None

        bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
            bob_ppo.observation_space.shape,
            bob_ppo.state_space.shape,
            bob_ppo.action_space.shape,
            bob_ppo.init_noise_std,
            bob_ppo.model_cfg,
            asymmetric=False,
        ).to(device)

        if hasattr(bob_ppo.actor_critic, "_goal_proj") and bob_ppo.actor_critic._goal_proj is not None:
            with torch.no_grad():
                bob_ppo.actor_critic._goal_proj.weight.mul_(0.1)

        bob_ppo.load(chkpt_path)
        bob_ppo.actor_critic.eval()
        _lsz = bob_ppo.actor_critic.lstm_hidden_size
        hidden = [torch.zeros(1, _lsz, device=device), torch.zeros(1, _lsz, device=device)]
        print(f"[Validate] Loaded ASP Bob from {chkpt_path}  GoalEncoder={'ON' if _has_goal_encoder else 'OFF'}")

    else:
        from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
        from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush

        agent_cfg = {
            "learn": {
                "nsteps": 32, "noptepochs": 3, "nminibatches": 4,
                "cliprange": 0.2, "ent_coef": 0.002, "gamma": 0.95, "lam": 0.95,
                "optim_stepsize": 3e-4, "init_noise_std": 0.3,
                "value_loss_coef": 1.0, "max_grad_norm": 1.0,
            },
            "policy": {
                "use_multicategorical": True, "num_cat_dims": 4, "num_bins": 21,
                "use_lstm": True, "lstm_hidden_size": 256,
                "pi_hid_sizes": [512, 256, 128],
                "vf_hid_sizes": [512, 256, 128],
                "activation": "relu",
            },
        }

        agent = PPO(
            vec_env=_DummyVecEnv(_obs_ppo_d, _obs_ppo_d, _mc_space),
            cfg_train=agent_cfg, device=device,
            sampler="sequential", log_dir="/tmp/validate_pusht_gym",
            asymmetric=False,
        )
        agent.observation_space = _obs_ppo_d
        agent.state_space = _obs_ppo_d
        agent.action_space = _mc_space
        agent.desired_kl = None

        agent.actor_critic = ActorCriticPush(
            agent.observation_space.shape, agent.state_space.shape,
            agent.action_space.shape, agent.init_noise_std, agent.model_cfg,
            asymmetric=False,
        ).to(device)

        agent.load(chkpt_path)
        agent.actor_critic.eval()
        _lsz = agent.actor_critic.lstm_hidden_size
        hidden = [torch.zeros(1, _lsz, device=device), torch.zeros(1, _lsz, device=device)]
        print(f"[Validate] Loaded PPO from {chkpt_path}")

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: List[ValidationResult] = []
    test_cfgs_data: List[dict] = []
    n_tests = min(args.num_tests, get_test_count())

    for test_idx in range(1, n_tests + 1):
        cfg = get_test_config(test_idx)
        if cfg is None:
            continue

        print(f"\n[Test {test_idx}/{n_tests}] {cfg.name} #{cfg.test_id}")
        print(f"  [{cfg.test_type}] goal=({cfg.main_goal_x:+.3f},{cfg.main_goal_y:+.3f}) yaw={cfg.main_goal_yaw:+.3f}  "
              f"start=({cfg.main_start.x:+.3f},{cfg.main_start.y:+.3f})")

        start_px, start_py = meter_to_pixel(cfg.main_start.x, cfg.main_start.y)
        goal_px, goal_py = meter_to_pixel(cfg.main_goal_x, cfg.main_goal_y)
        goal_angle = float(cfg.main_goal_yaw)

        TRIAL_COUNT = args.max_tries
        trial_successes = 0
        trial_pushes = []
        best_pos_err = float('inf')
        best_rot_err = float('inf')
        best_coverage = 0.0

        for trial in range(TRIAL_COUNT):
            agent_spawn_x = min(start_px + 80, ARENA_WIDTH - 20)
            agent_spawn_y = max(20.0, min(ARENA_HEIGHT - 20,
                                          start_py + 120 if start_py < ARENA_HEIGHT / 2 else start_py - 120))
            state = np.array([agent_spawn_x, agent_spawn_y, start_px, start_py, 0.0])
            obs, _ = env.reset(options={"reset_to_state": state})
            env.goal_pose = np.array([goal_px, goal_py, goal_angle])
            if hidden is not None:
                hidden[0].zero_()
                hidden[1].zero_()

            trial_ok = False
            pushes_used = 0
            pos_err = 0.0
            rot_err = 0.0
            trial_coverage = 0.0

            for push_i in range(args.max_pushes):
                agent_px, agent_py = float(obs[0]), float(obs[1])
                block_px, block_py = float(obs[2]), float(obs[3])
                block_angle = float(obs[4])

                if model_type == "sac":
                    sac_obs = build_sac_dict_obs(
                        agent_px, agent_py, block_px, block_py, block_angle,
                        goal_px, goal_py, goal_angle, device,
                    )
                    action_np, _states = model.predict(sac_obs, deterministic=True)
                    action_tensor = torch.from_numpy(action_np).float().to(device)
                    obj_x_m, _ = pixel_to_meter(block_px, block_py)
                    Xs, Ys, length, theta = decode_push_action_relative_continuous(
                        action_tensor,
                        torch.tensor([[obj_x_m, pixel_to_meter(block_px, block_py)[1]]], device=device),
                        torch.tensor([block_angle], device=device),
                    )
                elif model_type == "asp":
                    full_obs = build_asp_obs(
                        agent_px, agent_py, block_px, block_py, block_angle,
                        goal_px, goal_py, goal_angle, device,
                    )
                    with torch.no_grad():
                        h_in = (hidden[0], hidden[1])
                        (b_acts, _, _, _, _, new_bh) = bob_ppo.actor_critic.act_with_hidden(
                            full_obs, None, h_in,
                        )
                        if new_bh is not None:
                            hidden[0] = new_bh[0]
                            hidden[1] = new_bh[1]
                    obj_x_m, obj_y_m = pixel_to_meter(block_px, block_py)
                    Xs, Ys, length, theta = decode_push_action_relative(
                        b_acts,
                        torch.tensor([[obj_x_m, obj_y_m]], device=device),
                        torch.tensor([block_angle], device=device),
                        num_bins=num_bins, min_r=0.03, max_r=0.08,
                    )
                else:
                    flat_obs = build_ppo_obs(
                        agent_px, agent_py, block_px, block_py, block_angle,
                        goal_px, goal_py, goal_angle, device,
                    )
                    with torch.no_grad():
                        actions, _, _, _, _, _, new_h = agent.actor_critic.act_with_hidden(
                            flat_obs, None, (hidden[0], hidden[1]),
                        )
                        if new_h is not None:
                            hidden[0] = new_h[0]
                            hidden[1] = new_h[1]
                    obj_x_m, obj_y_m = pixel_to_meter(block_px, block_py)
                    Xs, Ys, length, theta = decode_push_action_relative(
                        actions,
                        torch.tensor([[obj_x_m, obj_y_m]], device=device),
                        torch.tensor([block_angle], device=device),
                        num_bins=num_bins,
                    )

                Xf = Xs + length * torch.cos(theta)
                Yf = Ys + length * torch.sin(theta)

                obs, coverage, pos_err, rot_err = execute_push(
                    env, Xs, Ys, length, theta, cv=args.render, render_delay=args.render_delay,
                )

                pushes_used += 1
                trial_coverage = max(trial_coverage, coverage)

                if model_type == "sac":
                    print(f"  push {push_i:2d}: "
                          f"act=({action_np[0,0]:+.3f},{action_np[0,1]:+.3f},{action_np[0,2]:+.3f},{action_np[0,3]:+.3f})  "
                          f"len={float(length.item()):.3f} th={math.degrees(float(theta.item())):.0f}deg  "
                          f"pos={pos_err:.4f}m rot={rot_err:.3f}rad cov={coverage:.2f}")
                elif model_type == "asp":
                    print(f"  push {push_i:2d}: bins=({', '.join(f'{int(b_acts[0,i].item()):2d}' for i in range(4))})  "
                          f"len={float(length.item()):.3f} θ={math.degrees(float(theta.item())):.0f}°  "
                          f"pos={pos_err:.4f}m rot={rot_err:.3f}rad cov={coverage:.2f}")
                else:
                    print(f"  push {push_i:2d}: bins=({', '.join(f'{int(actions[0,i].item()):2d}' for i in range(4))})  "
                          f"len={float(length.item()):.3f} θ={math.degrees(float(theta.item())):.0f}°  "
                          f"pos={pos_err:.4f}m rot={rot_err:.3f}rad cov={coverage:.2f}")

                block_px, block_py = float(obs[2]), float(obs[3])
                block_mx, block_my = pixel_to_meter(block_px, block_py)
                if block_mx < -0.55 or block_mx > 0.55 or block_my < 0.20 or block_my > 0.75:
                    print(f"  [OOB] block=({block_mx:.3f},{block_my:.3f})m outside workspace")
                    break

                if coverage >= 0.95:
                    trial_ok = True

            if trial_ok:
                trial_successes += 1
            trial_pushes.append(pushes_used)
            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_rot_err = rot_err
                best_coverage = trial_coverage

        avg_pushes = int(np.mean(trial_pushes)) if trial_pushes else 0
        sr_pct = trial_successes / TRIAL_COUNT * 100

        result = ValidationResult(
            test_index=test_idx,
            test_name=f"{cfg.name} #{cfg.test_id}",
            test_type=cfg.test_type,
            success=(trial_successes > 0),
            pushes_used=avg_pushes,
            final_pos_error=best_pos_err,
            final_rot_error=best_rot_err,
            area_coverage=_area_coverage(best_pos_err, best_rot_err),
            gym_coverage=best_coverage,
            trial_count=TRIAL_COUNT,
            success_count=trial_successes,
        )
        results.append(result)
        test_cfgs_data.append({
            "start_x": cfg.main_start.x, "start_y": cfg.main_start.y,
            "goal_x": cfg.main_goal_x, "goal_y": cfg.main_goal_y,
            "goal_yaw": cfg.main_goal_yaw,
            "object_type": "tblock",
        })
        status = "PASS" if trial_successes > 0 else "FAIL"
        print(f"  {status} | {trial_successes}/{TRIAL_COUNT} = {sr_pct:.0f}% | avg_pushes: {avg_pushes} | "
              f"best_pos_err: {best_pos_err:.4f} | best_rot_err: {best_rot_err:.4f} | cov: {best_coverage:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_trials = sum(r.trial_count for r in results)
    total_successes = sum(r.success_count for r in results)
    sr = total_successes / total_trials * 100 if total_trials > 0 else 0
    n_tests_passed = sum(1 for r in results if r.success_count > 0)
    avg_pushes = np.mean([r.pushes_used for r in results]) if results else 0

    pos_only = [r for r in results if r.test_type == "pos_only"]
    pos_rot = [r for r in results if r.test_type == "pos_rot"]
    po_trials = sum(r.trial_count for r in pos_only)
    pr_trials = sum(r.trial_count for r in pos_rot)
    po_successes = sum(r.success_count for r in pos_only)
    pr_successes = sum(r.success_count for r in pos_rot)
    sr_po = po_successes / po_trials * 100 if po_trials > 0 else 0
    sr_pr = pr_successes / pr_trials * 100 if pr_trials > 0 else 0

    avg_cov = np.mean([r.gym_coverage for r in results]) if results else 0

    print(f"\n{'='*60}")
    print(f"GYM-PUSHT VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Test configs:  {len(results)}")
    print(f"  Trials:        {total_trials}")
    print(f"  Successes:     {total_successes}")
    print(f"  Success rate:  {sr:.1f}%")
    print(f"  Tests passed:  {n_tests_passed}/{len(results)} ({n_tests_passed/len(results)*100:.0f}% of configs)")
    print(f"  Pos-only SR:   {sr_po:.1f}% ({po_successes}/{po_trials} trials)")
    print(f"  Pos+rot SR:    {sr_pr:.1f}% ({pr_successes}/{pr_trials} trials)")
    print(f"  Avg pushes:    {avg_pushes:.1f}")
    print(f"  Avg coverage:  {avg_cov:.2f}")
    print(f"{'='*60}")

    for r in results:
        sr_pct = r.success_count / r.trial_count * 100 if r.trial_count > 0 else 0
        status = "PASS" if r.success_count > 0 else "FAIL"
        print(f"  {status:5s} | Test {r.test_index:2d} | {r.test_name:30s} | {r.success_count}/{r.trial_count} ({sr_pct:.0f}%) pushes={r.pushes_used:2d}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as _f:
            writer = _csv.writer(_f)
            writer.writerow(["test_index", "test_name", "test_type", "success", "pushes_used",
                             "pos_err", "rot_err", "area_coverage", "gym_coverage",
                             "trial_count", "success_count"])
            for r in results:
                writer.writerow([r.test_index, r.test_name, r.test_type, int(r.success),
                                 r.pushes_used, r.final_pos_error, r.final_rot_error,
                                 r.area_coverage, r.gym_coverage,
                                 r.trial_count, r.success_count])
        print(f"\n[CSV] Results saved to {args.csv}")

        try:
            from asyncDualPlayPPO.tests.plot_validation import generate_single_run_plot
            plot_data = [{"test_index": r.test_index, "test_name": r.test_name,
                          "success": r.success, "final_pos_error": r.final_pos_error,
                          "final_rot_error": r.final_rot_error} for r in results]
            plot_path = os.path.splitext(args.csv)[0] + ".png"
            generate_single_run_plot(plot_data, test_cfgs_data, plot_path,
                                    rot_threshold_rad=args.rot_threshold)
        except Exception as _e:
            print(f"[WARN] Plot generation failed: {_e}")

    env.close()
    if args.render:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
