"""
Push-PPO baseline training script.

Single-agent PPO with push primitive macro-actions.  No ASP, no ABC, no
historical pool — just a single PPO agent learning to push objects to goals.

Architecture:
  Agent predicts 6D push parameters → push waypoints → cuRobo IK per
  waypoint → physics step → dense reward after push completes → PPO update.

Run locally:
  python -m asyncDualPlayPPO.train_push --num_envs 16 --max_iterations 500 --exp_name push_test
"""

# ── cuRobo MUST be imported before AppLauncher ────────────────────────────────
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print(
        "\n[ERROR] cuRobo not found. Install it in the active venv before training.\n"
        "  Check: apptainer exec --nv isaac-lab.sif python -c 'import curobo'\n"
    )
    import sys
    sys.exit(1)

import torch
import torch._dynamo    # noqa: F401
import torch._C         # noqa: F401
import torch.optim      # noqa: F401

from isaaclab.app import AppLauncher

import os
import signal
import sys
import yaml
import argparse
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SuppressAllOutput:
    def __enter__(self):
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)
        self.devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.devnull, self.stdout_fd)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout, self.stdout_fd)
        os.dup2(self.saved_stderr, self.stderr_fd)
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)
        os.close(self.devnull)


def main():
    parser = argparse.ArgumentParser(description="Push-PPO Baseline Training")
    parser.add_argument("--exp_name", type=str, default="push_ppo_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--chkpt", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write terminal output to this file as well")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── Log file: opened once, written to alongside every print() ────────────
    _log_fh = None
    if args.log_file:
        _log_fh = open(args.log_file, "a", buffering=1)
        print(f"[Init] Logging to {args.log_file}", flush=True)

    def _pr(msg: str = "", end: str = "\n"):
        """Print to stdout AND log file (if active)."""
        sys.stdout.write(msg + end)
        sys.stdout.flush()
        if _log_fh:
            _log_fh.write(msg + end)
            _log_fh.flush()

    import torch
    import numpy as np
    import copy
    import gymnasium as gym_mc
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints, total_push_substeps,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import RolloutStorage

    ppo_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg = load_cfg(ppo_cfg_path)

    # ── Push hyperparameters ─────────────────────────────────────────────────
    max_pushes_per_episode = 3
    push_nsteps = 32          # pushes per PPO rollout (per env)
    noptepochs = 3
    nminibatches = 4
    cliprange = 0.2
    ent_coef = 0.01
    gamma = 0.998
    lam = 0.95
    learning_rate = 3e-4

    _policy_cfg = ppo_cfg["params"]["policy"]

    num_cat_dims = 6
    num_bins = _policy_cfg.get("num_bins", 11)
    use_lstm = _policy_cfg.get("use_lstm", True)

    print(f"[Push-PPO] Config: {push_nsteps} pushes/rollout, {num_cat_dims}D×{num_bins} bins, LSTM={use_lstm}")

    # ── Environment config ────────────────────────────────────────────────────
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )

    print("Creating environment...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)

    env = PushEnvWrapper(
        env=base_env,
        device=base_env.device,
        num_objects=1,
        max_pushes_per_episode=max_pushes_per_episode,
        headless=args.headless,
    )
    print("Environment ready.")

    # ── cuRobo IK solver ──────────────────────────────────────────────────────
    print("[cuRobo] Initialising IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(env.device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    ik_solver = IKSolver(_ik_config)
    print("[cuRobo] IK solver created.")

    # Warm-up CUDA graph
    print(f"[cuRobo] Warming up CUDA graph for N={env.num_envs} envs...")
    _wup_pos = torch.zeros(env.num_envs, 3, device=env.device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32).expand(env.num_envs, 4)
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(env.num_envs, 1, 6, device=env.device),
        retract_config=torch.zeros(env.num_envs, 6, device=env.device),
    )
    print("[cuRobo] Warm-up done.")

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)

    # Workspace clamp limits (local / env-origin-relative frame, metres)
    _WS_X = (-0.50, 0.50)
    _WS_Y = (0.25,  0.70)
    _WS_Z = ( 0.00, 0.55)

    # ── Robot body/joint indices ──────────────────────────────────────────────
    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _wrist3_ids, _ = _robot_scene.find_bodies("wrist_3_link")
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    # Fixed TCP→wrist3 offset in tool-down orientation.
    # Drive all envs to a neutral tool-down pose, settle 30 PD steps, then
    # freeze the offset measured from env 0 (identical geometry across envs).
    print("[Setup] Calibrating TCP→wrist3 offset...")
    _calib_pos = torch.zeros(env.num_envs, 3, device=env.device)
    _calib_pos[:, 1] = 0.50
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN.expand(env.num_envs, 4)),
        seed_config=_calib_cur.unsqueeze(1),
        retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(env.num_envs, 6)
    _calib_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
    _lf_w = _robot_scene.data.body_pos_w[0, _lf_ids[0]]
    _rf_w = _robot_scene.data.body_pos_w[0, _rf_ids[0]]
    _w3_w = _robot_scene.data.body_pos_w[0, _wrist3_ids[0]]
    _FIXED_TCP_OFFSET = ((_lf_w + _rf_w) / 2.0 - _w3_w).unsqueeze(0).clone()
    print(
        f"[Setup] Fixed offset = ({float(_FIXED_TCP_OFFSET[0,0]):+.3f}, "
        f"{float(_FIXED_TCP_OFFSET[0,1]):+.3f}, {float(_FIXED_TCP_OFFSET[0,2]):+.3f})"
    )

    # ── PPO agent ─────────────────────────────────────────────────────────────
    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    agent_cfg = copy.deepcopy(ppo_cfg["params"])
    agent_cfg["learn"]["nsteps"] = push_nsteps
    agent_cfg["learn"]["noptepochs"] = noptepochs
    agent_cfg["learn"]["nminibatches"] = nminibatches
    agent_cfg["learn"]["cliprange"] = cliprange
    agent_cfg["learn"]["ent_coef"] = ent_coef
    agent_cfg["learn"]["gamma"] = gamma
    agent_cfg["learn"]["lam"] = lam
    agent_cfg["learn"]["optim_stepsize"] = learning_rate

    agent = PPO(
        vec_env=env,
        cfg_train=agent_cfg,
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/agent",
        asymmetric=False,
    )
    agent.observation_space = env.observation_space
    agent.state_space = env.state_space
    agent.action_space = _mc_space
    agent.desired_kl = None

    agent.actor_critic = ActorCriticPush(
        agent.observation_space.shape,
        agent.state_space.shape,
        agent.action_space.shape,
        agent.init_noise_std,
        agent.model_cfg,
        asymmetric=False,
    ).to(env.device)

    agent.storage = RolloutStorage(
        agent.vec_env.num_envs,
        push_nsteps,
        agent.observation_space.shape,
        agent.state_space.shape,
        agent.action_space.shape,
        agent.device,
        "sequential",
    )

    agent.optimizer = torch.optim.Adam(
        agent.actor_critic.parameters(), lr=agent.learning_rate,
    )

    if args.chkpt and os.path.isfile(args.chkpt):
        agent.load(args.chkpt)
        print(f"[Resume] Loaded agent from {args.chkpt}")

    # ── LSTM hidden state ─────────────────────────────────────────────────────
    if use_lstm:
        _lsz = agent.actor_critic.lstm_hidden_size
        hidden_state = [
            torch.zeros(env.num_envs, _lsz, device=env.device),
            torch.zeros(env.num_envs, _lsz, device=env.device),
        ]
    else:
        hidden_state = None

    # ── Training state ────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")
    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    best_success_rate = -1.0
    iteration = 0
    rew_buf     = deque(maxlen=push_nsteps * env.num_envs)
    sr_buf      = deque(maxlen=200)
    pos_err_buf = deque(maxlen=push_nsteps * env.num_envs)
    ema_rew     = 0.0

    print(f"\n{'='*80}\nPUSH-PPO BASELINE: {args.exp_name}\nLOG DIR: {run_dir}\n{'='*80}\n")

    # ── Init environment ──────────────────────────────────────────────────────
    print("Initialising environment...")
    with SuppressAllOutput():
        obs = env.reset()
    print("Training loop starting...")
    sys.stdout.flush()

    # Per-env state accumulators
    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    ee_pos_local = _tcp_pos_local()
    ee_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()
    prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

    # ── SIGTERM handler ──────────────────────────────────────────────────────
    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        print("[INFO] SIGTERM received — checkpoint after current iteration.", flush=True)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── TRAINING LOOP ────────────────────────────────────────────────────────
    while iteration < args.max_iterations:
        agent.storage.clear()

        total_ik_fails = 0
        total_ik_steps = 0
        env.episode_push_counts.clear()
        env.episode_successes.clear()

        obs_pre_push = obs.clone()
        env.capture_pre_push(obs)

        for push_step in range(push_nsteps):
            # ── Agent predicts push action ────────────────────────────────────
            with torch.no_grad():
                # Zero LSTM hidden before each push so evaluate() (which uses
                # zero context) matches collection — avoids GAE ratio corruption.
                if hidden_state is not None:
                    hidden_state[0].zero_()
                    hidden_state[1].zero_()
                h_in = (hidden_state[0], hidden_state[1]) if hidden_state else None
                actions, log_prob, value, mu, sigma, new_h = agent.actor_critic.act_with_hidden(
                    obs, None, h_in,
                )
                if hidden_state is not None and new_h is not None:
                    hidden_state[0] = new_h[0]
                    hidden_state[1] = new_h[1]

            push_params = decode_push_action(actions, num_bins=num_bins)

            # ── Compute push waypoints ────────────────────────────────────────
            obj_pos = env._get_obj_pos(obs)

            waypoints = compute_push_waypoints(
                offset_x=push_params[:, 0],
                offset_y=push_params[:, 1],
                push_dx=push_params[:, 2],
                push_dy=push_params[:, 3],
                push_dz=push_params[:, 5],
                obj_pos=obj_pos,
                current_ee_pos=ee_pos_local,
                current_ee_quat=ee_quat_w,
                device=env.device,
            )

            # ── Execute push trajectory ───────────────────────────────────────
            terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            prev_grip = torch.ones(env.num_envs, device=env.device)
            for wp_idx, (wp_pos, wp_quat, wp_grip) in enumerate(waypoints):
                ik_target = wp_pos - _FIXED_TCP_OFFSET
                ik_target[:, 0].clamp_(_WS_X[0], _WS_X[1])
                ik_target[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
                ik_target[:, 2].clamp_(_WS_Z[0], _WS_Z[1])

                result = ik_solver.solve_batch(
                    CuroboPose(position=ik_target, quaternion=wp_quat),
                    seed_config=prev_joint_cmd.unsqueeze(1),
                    retract_config=prev_joint_cmd,
                )

                ik_ok = result.success.squeeze(-1)
                cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]

                total_ik_steps += env.num_envs
                total_ik_fails += int((~ik_ok).sum().item())

                solved = result.solution.view(env.num_envs, 6)
                raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, prev_joint_cmd)
                prev_joint_cmd = raw_cmd.detach().clone()

                # Gripper-change pre-step: hold joints while toggling gripper
                if (wp_grip != prev_grip).any():
                    grip_hold = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
                    grip_hold[:, :6] = cur_joints
                    grip_hold[:, 6] = wp_grip
                    obs, _, step_term2, _, _ = env.step(grip_hold)
                    terminated |= step_term2
                    prev_grip = wp_grip.clone()

                env_full = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
                env_full[:, :6] = raw_cmd
                env_full[:, 6] = wp_grip

                obs, _, step_terminated, truncated, _ = env.step(env_full)
                # terminated envs are auto-reset by the base env inside step();
                # obs already contains post-reset observations for those envs.
                terminated |= step_terminated

            # ── After push: compute reward & done ────────────────────────────
            reward = env.compute_push_reward(obs)
            done = env.check_done(obs, terminated)

            # ── Record transition ────────────────────────────────────────────
            agent.storage.add_transitions(
                obs_pre_push, obs_pre_push, actions, reward, done,
                value, log_prob, mu, sigma,
                masks=(~done).float(),
            )

            rew_buf.extend(reward.cpu().tolist())
            cur_at_goal = env.at_goal.float()
            sr_buf.extend(cur_at_goal.cpu().tolist())
            pos_err_buf.extend(env._last_pos_err.cpu().tolist())

            # Per-push compact summary (every 4 pushes to avoid spam)
            if push_step % 4 == 0:
                _pr(
                    f"  [Push {push_step:3d}] "
                    f"rew={reward.mean().item():+.3f}  pos_err={env._last_pos_err.mean().item():.3f}  "
                    f"at_goal={cur_at_goal.sum().item():.0f}/{env.num_envs}"
                )

            # ── Handle done envs ──────────────────────────────────────────────
            if done.any():
                done_ids = torch.where(done)[0]
                # Snapshot goal/object positions before reset clears them
                obj_pos_done  = obs[done_ids, env.robot_dim:env.robot_dim + 3]
                goal_pos_done = obs[done_ids, env.robot_dim + env.obj_state_dim:
                                     env.robot_dim + env.obj_state_dim + 3]
                pos_err_done  = (obj_pos_done - goal_pos_done).norm(dim=-1)
                ep_pushes_pre = len(env.episode_push_counts)
                env.reset_done_envs(done)
                ep_pushes_post = len(env.episode_push_counts)
                n_new = ep_pushes_post - ep_pushes_pre
                if n_new > 0:
                    new_pushes = env.episode_push_counts[-n_new:]
                    new_successes = env.episode_successes[-n_new:]
                    for i, (p, s) in enumerate(zip(new_pushes, new_successes)):
                        status = "SUCCESS" if s else "fail"
                        gi = min(i, len(done_ids) - 1)
                        g = goal_pos_done[gi]
                        o = obj_pos_done[gi]
                        e = pos_err_done[gi]
                        _pr(
                            f"  [Episode] pushes={p}  {status}  "
                            f"goal=({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f})  "
                            f"final=({o[0]:+.3f},{o[1]:+.3f},{o[2]:+.3f})  "
                            f"err={e:.3f}m"
                        )
                if hidden_state is not None:
                    hidden_state[0][done] = 0.0
                    hidden_state[1][done] = 0.0
                # Only at_goal / max_pushes envs need explicit reset; terminated
                # ones were already auto-reset by the base env inside step().
                needs_reset = done & ~terminated
                if needs_reset.any():
                    reset_ids = torch.where(needs_reset)[0]
                    obs_dict_r, _ = env.env.reset(env_ids=reset_ids)
                    obs_new = env._build_obs(obs_dict_r)
                    obs[needs_reset] = obs_new[needs_reset]
                ee_pos_local[done] = _tcp_pos_local()[done]
                ee_quat_w[done] = _QUAT_TOOL_DOWN.expand(done.sum().item(), 4).to(env.device)
                prev_joint_cmd[done] = _robot_scene.data.joint_pos[:, _arm_jids][done]

            # ── Update EE position tracker for next push ─────────────────────
            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()

            obs_pre_push = obs.clone()
            env.capture_pre_push(obs)

        # ── PPO UPDATE ────────────────────────────────────────────────────────
        with torch.no_grad():
            last_val = agent.actor_critic.critic(obs)
        agent.storage.compute_returns(last_val, agent.gamma, agent.lam)
        loss_val, loss_surr = agent.update()
        agent.storage.clear()

        mean_rew    = np.mean(rew_buf) if rew_buf else 0.0
        mean_pos_err = np.mean(pos_err_buf) if pos_err_buf else 0.0
        sr = np.mean(sr_buf) if sr_buf else 0.0
        ik_fail_rate = total_ik_fails / max(1, total_ik_steps)

        ema_rew = 0.9 * ema_rew + 0.1 * mean_rew

        ep_push_counts = list(env.episode_push_counts)
        ep_successes = list(env.episode_successes)
        avg_pushes = np.mean(ep_push_counts) if ep_push_counts else float("nan")
        ep_sr = np.mean(ep_successes) if ep_successes else float("nan")
        n_episodes = len(ep_push_counts)

        loss_delta = loss_surr - getattr(agent, "_last_loss_surr", 0.0)
        agent._last_loss_surr = loss_surr

        writer.add_scalar("Loss/Agent/Value",        loss_val,      iteration)
        writer.add_scalar("Loss/Agent/Surrogate",    loss_surr,     iteration)
        writer.add_scalar("Reward/Mean",             mean_rew,      iteration)
        writer.add_scalar("Reward/EMA",              ema_rew,       iteration)
        writer.add_scalar("Metrics/SuccessRate",     sr,            iteration)
        writer.add_scalar("Metrics/PosError",        mean_pos_err,  iteration)
        writer.add_scalar("Metrics/IKFailRate",      ik_fail_rate,  iteration)
        writer.add_scalar("Metrics/EpisodicSR",      ep_sr if not np.isnan(ep_sr) else 0.0, iteration)
        writer.add_scalar("Metrics/AvgPushesPerEpisode", avg_pushes if not np.isnan(avg_pushes) else 0.0, iteration)
        writer.add_scalar("Metrics/Episodes",        n_episodes,    iteration)

        # Single compact iteration line — machine-parseable
        avg_pushes_str = f"{avg_pushes:.1f}" if not np.isnan(avg_pushes) else "nan"
        trend = "↓" if loss_delta < -0.01 else ("↑" if loss_delta > 0.01 else "→")
        _pr(
            f"[Iter {iteration:5d}] "
            f"Loss={loss_surr:.4f}{trend} | Val={loss_val:.4f} | "
            f"Rew={mean_rew:+.4f} (EMA {ema_rew:+.4f}) | "
            f"PosErr={mean_pos_err:.4f} | SR={sr:.4f} | "
            f"IK_fail={ik_fail_rate:.3f} | "
            f"AvgPushes={avg_pushes_str} | Epi={n_episodes} | "
            f"BestSR={best_success_rate:.4f}"
        )
        sys.stdout.flush()

        if sr > best_success_rate:
            best_success_rate = sr
            agent.save(os.path.join(agent.log_dir, "model_best.pt"))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(agent.log_dir, f"model_{iteration}.pt"))
            print(f"  [Checkpoint] Saved model_{iteration}.pt")

        rew_buf.clear()
        sr_buf.clear()
        pos_err_buf.clear()
        iteration += 1

        if _shutdown_requested:
            agent.save(os.path.join(agent.log_dir, f"model_{iteration}_emergency.pt"))
            print("[INFO] Emergency checkpoint saved. Shutting down.")
            break

    print(f"\nTraining complete. Best SR: {best_success_rate:.4f}")
    agent.save(os.path.join(agent.log_dir, "model_final.pt"))
    simulation_app.close()


if __name__ == "__main__":
    main()
