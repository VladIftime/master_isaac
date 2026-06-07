#!/usr/bin/env python3
import csv
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from geometry_msgs.msg import PoseStamped
from race_basic_motion_control.srv import Behaviors
from rclpy.node import Node
from stable_baselines3 import SAC

gym = None
spaces = None
for _mod_name in ("gymnasium", "gym"):
    try:
        _mod = importlib.import_module(_mod_name)
        gym = _mod
        spaces = _mod.spaces
        if _mod_name == "gymnasium":
            sys.modules.setdefault("gym", _mod)
            sys.modules.setdefault("gym.spaces", _mod.spaces)
            for _space_module in (
                "box",
                "discrete",
                "multi_binary",
                "multi_discrete",
                "dict",
                "tuple",
            ):
                try:
                    sys.modules.setdefault(
                        f"gym.spaces.{_space_module}",
                        importlib.import_module(f"gymnasium.spaces.{_space_module}"),
                    )
                except ImportError:
                    pass
        break
    except ImportError:
        continue

if gym is None or spaces is None:
    raise ImportError(
        "Neither 'gymnasium' nor 'gym' is installed. Please install one of them."
    )

np.random.seed(10)


def find_package_path(package_name: str) -> Path:
    current_file = Path(__file__).resolve()
    for parent in [current_file.parent, *current_file.parents]:
        if parent.name == package_name and (parent / "package.xml").exists():
            return parent

    workspace_roots = set()
    for env_name in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
        for prefix in os.environ.get(env_name, "").split(os.pathsep):
            if not prefix:
                continue
            parts = Path(prefix).resolve().parts
            if "install" in parts:
                workspace_roots.add(Path(*parts[: parts.index("install")]))

    cwd = Path.cwd().resolve()
    workspace_roots.update([cwd, *cwd.parents])
    for root in workspace_roots:
        src = root / "src"
        for candidate in (
            src / package_name,
            src / "intuition_rl_packages" / package_name,
            src / package_name / package_name,
        ):
            if (candidate / "package.xml").exists():
                return candidate

    try:
        return Path(get_package_share_directory(package_name))
    except PackageNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find ROS2 package '{package_name}'"
        ) from exc


class TossingObjectWithObstacleAvoidanceV3(gym.Env):
    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node
        self.robot = self.node.declare_parameter("robot", "left").value
        self.robot_obs = -1.0 if self.robot == "left" else 1.0

        self.sim_or_real = self.node.declare_parameter("sim_or_real", "sim").value
        self.pose_query_frame = self.node.declare_parameter(
            "pose_query_frame", "local_world"
        ).value

        self.object_name = self.node.declare_parameter("object_name", "milk").value
        self.basket_name = self.node.declare_parameter(
            "basket_name", "wooden_box"
        ).value
        self.obstacle_name = self.node.declare_parameter(
            "obstacle_name", "obstacle_box"
        ).value

        self.object_spawn_frame = self.node.declare_parameter(
            "object_spawn_frame", "local_world"
        ).value
        self.object_spawn_x = float(
            self.node.declare_parameter("object_spawn_x", -0.11).value
        )
        self.object_spawn_y = float(
            self.node.declare_parameter("object_spawn_y", 0.374).value
        )
        self.object_spawn_z = float(
            self.node.declare_parameter("object_spawn_z", 0.03).value
        )

        self.basket_spawn_x = float(
            self.node.declare_parameter("basket_spawn_x", 0.75).value
        )
        self.basket_spawn_y = float(
            self.node.declare_parameter("basket_spawn_y", 1.2).value
        )
        self.basket_spawn_z = float(
            self.node.declare_parameter("basket_spawn_z", 0.03).value
        )

        self.obstacle_spawn_x = float(
            self.node.declare_parameter("obstacle_spawn_x", 0.75).value
        )
        self.obstacle_spawn_y = float(
            self.node.declare_parameter("obstacle_spawn_y", 1.2).value
        )
        self.obstacle_spawn_z = float(
            self.node.declare_parameter("obstacle_spawn_z", 0.03).value
        )
        self.obstacle_y_min_offset_m = float(
            self.node.declare_parameter("obstacle_y_min_offset_m", 0.05).value
        )

        self.settle_timeout_s = float(
            self.node.declare_parameter("settle_timeout_s", 2.0).value
        )
        self.settle_sample_period_s = float(
            self.node.declare_parameter("settle_sample_period_s", 0.1).value
        )
        self.settle_position_epsilon_m = float(
            self.node.declare_parameter("settle_position_epsilon_m", 0.003).value
        )
        self.settle_required_samples = int(
            self.node.declare_parameter("settle_required_samples", 3).value
        )

        self.behaviors_service = self.node.create_client(
            Behaviors, "/behaviors_service"
        )
        self._wait_for_service(self.behaviors_service, "/behaviors_service")

        self._send_behavior(robot="dual", behavior="go_to_initial_top_pose")
        self._spawn_all_objects_once()

        self.last_initial_joint_value = 0.0
        self.last_final_joint_value = 0.0
        self.last_releasing_time = 0.0
        self.last_duration = 0.0
        self.last_object_label = self.object_name

        self.dist_obs = 0.0
        self.dist_obs_x = 0.0
        self.dist_obs_y = 0.0
        self.observation_mode = "obstacle_16"

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.positive_samples_dir = Path("positiveSamples")
        self.positive_samples_dir.mkdir(parents=True, exist_ok=True)
        file_name = (
            self.positive_samples_dir / f"record_positive_sample_csv_{timestr}.csv"
        )
        self.samples_file = file_name.open("w", newline="")
        self.record_samples = csv.writer(self.samples_file)
        self.record_samples.writerow(
            [
                "robot",
                "basket_x",
                "basket_y",
                "basket_z",
                "shoulder_initial_joint_value",
                "shoulder_final_joint_value",
                "releasing_time",
                "duration_of_trajectory",
                "object_label",
                "obj_x",
                "obj_y",
                "obj_z",
                "dist",
                "dist_x",
                "dist_y",
                "reward",
            ]
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.05, 0.1]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype="float32",
        )
        observation_bound = np.array([np.inf] * 16)
        self.observation_space = spaces.Box(
            low=-observation_bound, high=observation_bound, dtype="float32"
        )

    def set_observation_mode_for_model(self, model_observation_space) -> None:
        shape = getattr(model_observation_space, "shape", None)
        if not shape:
            return

        obs_size = int(shape[0])
        if obs_size == 16:
            self.observation_mode = "obstacle_16"
        elif obs_size == 8:
            self.observation_mode = "dual_arm_8"
        elif obs_size == 7:
            self.observation_mode = "single_arm_7"
        else:
            self.node.get_logger().warn(
                f"Loaded model expects unsupported observation size {obs_size}; keeping obstacle_16 observations"
            )
            return

        observation_bound = np.array([np.inf] * obs_size)
        self.observation_space = spaces.Box(
            low=-observation_bound, high=observation_bound, dtype="float32"
        )
        self.node.get_logger().info(
            f"Using {self.observation_mode} observation layout for loaded model shape {shape}"
        )

    def _wait_for_service(self, client, name: str, timeout_s: float = 1.0) -> None:
        while rclpy.ok() and not client.wait_for_service(timeout_sec=timeout_s):
            self.node.get_logger().info(f"Waiting for service {name}...")

    def _send_behavior(
        self,
        robot: str = "left",
        behavior: str = "",
        object_name: str = "",
        initial_joint_value: float = 0.0,
        final_joint_value: float = 0.0,
        releasing_time: float = 0.0,
        duration: float = 0.0,
        target_pose: Optional[PoseStamped] = None,
        target_frame_id: str = "",
        timeout_sec: float = 10.0,
    ) -> Optional[Behaviors.Response]:
        req = Behaviors.Request()
        req.robot = robot
        req.behavior = behavior
        req.object_name = object_name
        req.initial_joint_value = float(initial_joint_value)
        req.final_joint_value = float(final_joint_value)
        req.releasing_time = float(releasing_time)
        req.duration = float(duration)
        req.target_frame_id = target_frame_id
        if target_pose is not None:
            req.object_target_pose = target_pose
        if hasattr(req, "sim_or_real"):
            req.sim_or_real = self.sim_or_real

        future = self.behaviors_service.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            self.node.get_logger().warn(f"Behavior call failed/timed out: {behavior}")
            return None
        return future.result()

    def _make_pose(self, x: float, y: float, z: float, frame_id: str) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0
        return pose

    def _spawn_gazebo_object(self, object_name: str, pose: PoseStamped) -> bool:
        response = self._send_behavior(
            robot="dual",
            behavior="spawn_object_in_gazebo",
            object_name=object_name,
            target_pose=pose,
            timeout_sec=4.0,
        )
        return bool(response and response.result)

    def _set_gazebo_object_pose(self, object_name: str, pose: PoseStamped) -> bool:
        response = self._send_behavior(
            robot="dual",
            behavior="set_object_pose_in_gazebo",
            object_name=object_name,
            target_pose=pose,
            timeout_sec=4.0,
        )
        return bool(response and response.result)

    def _get_object_position_from_gazebo(self, object_name: str) -> np.ndarray:
        response = self._send_behavior(
            robot="dual",
            behavior="get_object_pose_from_gazebo",
            object_name=object_name,
            target_frame_id=self.pose_query_frame,
            timeout_sec=5.0,
        )
        if response is None or not response.result:
            raise RuntimeError(f"get_object_pose_from_gazebo failed for {object_name}")

        pose = response.pose_in_target_frame.pose
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=float
        )

    def _wait_for_objects_to_settle(self, object_names, label: str) -> bool:
        object_names = list(object_names)
        previous_positions = {}
        stable_samples = 0
        deadline = time.monotonic() + self.settle_timeout_s

        while rclpy.ok() and time.monotonic() < deadline:
            time.sleep(self.settle_sample_period_s)
            max_delta = 0.0
            all_positions_available = True

            for name in object_names:
                try:
                    position = self._get_object_position_from_gazebo(name)
                except RuntimeError as exc:
                    self.node.get_logger().warn(
                        f"Could not read pose while waiting for {label}: {exc}"
                    )
                    all_positions_available = False
                    break

                if name in previous_positions:
                    max_delta = max(
                        max_delta,
                        float(np.linalg.norm(position - previous_positions[name])),
                    )
                previous_positions[name] = position

            if not all_positions_available or len(previous_positions) != len(
                object_names
            ):
                stable_samples = 0
                continue

            if max_delta <= self.settle_position_epsilon_m:
                stable_samples += 1
                if stable_samples >= self.settle_required_samples:
                    self.node.get_logger().info(
                        f"{label}: objects settled, max_delta={max_delta:.4f} m"
                    )
                    return True
            else:
                stable_samples = 0

        self.node.get_logger().warn(
            f"{label}: objects did not fully settle within {self.settle_timeout_s:.2f}s"
        )
        return False

    def _spawn_all_objects_once(self) -> None:
        spawn_data = [
            (
                self.object_name,
                self.object_spawn_x,
                self.object_spawn_y,
                self.object_spawn_z,
            ),
            (
                self.basket_name,
                self.basket_spawn_x,
                self.basket_spawn_y,
                self.basket_spawn_z,
            ),
            (
                self.obstacle_name,
                self.obstacle_spawn_x,
                self.obstacle_spawn_y,
                self.obstacle_spawn_z,
            ),
        ]
        for name, x, y, z in spawn_data:
            pose = self._make_pose(x, y, z, self.object_spawn_frame)
            spawned = self._spawn_gazebo_object(name, pose)
            placed = self._set_gazebo_object_pose(name, pose)
            if not placed:
                self.node.get_logger().warn(f"Failed to place Gazebo object '{name}'")
            elif not spawned:
                self.node.get_logger().info(
                    f"Gazebo object '{name}' already existed, reused by set pose"
                )
        self._wait_for_objects_to_settle(
            [self.object_name, self.basket_name, self.obstacle_name],
            "initial spawn",
        )

    def init_env(self) -> None:
        object_pose = self._make_pose(
            self.object_spawn_x,
            self.object_spawn_y,
            self.object_spawn_z,
            self.object_spawn_frame,
        )
        self._set_gazebo_object_pose(self.object_name, object_pose)

        basket_x = float(np.random.uniform(low=-1.2, high=1.2))
        basket_y = float(np.random.uniform(low=0.8, high=1.5))
        basket_pose = self._make_pose(
            basket_x, basket_y, self.basket_spawn_z, self.object_spawn_frame
        )
        self._set_gazebo_object_pose(self.basket_name, basket_pose)

        obstacle_x = float(
            (basket_x * 2.0 / 3.0) + np.random.uniform(low=-0.05, high=0.05)
        )
        obstacle_y = float((basket_y * 2.0 / 3.0) + self.obstacle_y_min_offset_m)
        obstacle_pose = self._make_pose(
            obstacle_x, obstacle_y, self.obstacle_spawn_z, self.object_spawn_frame
        )
        self._set_gazebo_object_pose(self.obstacle_name, obstacle_pose)

        self.dist_obs = 0.0
        self.dist_obs_x = 0.0
        self.dist_obs_y = 0.0
        self._wait_for_objects_to_settle(
            [self.object_name, self.basket_name, self.obstacle_name],
            "episode reset",
        )

    def action(
        self,
        robot: str,
        shoulder_initial_joint_value: float,
        shoulder_final_joint_value: float,
        releasing_time: float,
        duration_of_trajectory: float,
        object_label: str,
    ) -> None:
        self.last_initial_joint_value = -(
            (0.5 * (1.0 + shoulder_initial_joint_value) * 2.4) + 0.001
        )
        self.last_final_joint_value = -(
            (0.5 * (1.0 + shoulder_final_joint_value) * 2.4) + 0.001
        )
        self.last_releasing_time = float(releasing_time * duration_of_trajectory)
        self.last_duration = float(duration_of_trajectory)
        self.last_object_label = object_label

        self._send_behavior(robot=robot, behavior="initial_tossing_pose")
        time.sleep(0.5)

        self._send_behavior(
            robot=robot,
            behavior="tossing_object",
            object_name=object_label,
            initial_joint_value=self.last_initial_joint_value,
            final_joint_value=self.last_final_joint_value,
            releasing_time=self.last_releasing_time,
            duration=self.last_duration,
        )

        self._send_behavior(robot=robot, behavior="initial_tossing_pose")
        time.sleep(0.1)

    def get_distance(
        self, obj: str, basket: str
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        object_pose = self._get_object_position_from_gazebo(obj)
        basket_pose = self._get_object_position_from_gazebo(basket)

        dist = float(np.linalg.norm(object_pose - basket_pose))
        dist_x = float(np.linalg.norm(object_pose[0] - basket_pose[0]))
        dist_y = float(np.linalg.norm(object_pose[1] - basket_pose[1]))
        return basket_pose, object_pose, dist, dist_x, dist_y

    def observe(self):
        max_norm = 3.0
        basket_pose, object_pose, dist, dist_x, dist_y = self.get_distance(
            self.object_name, self.basket_name
        )
        if self.observation_mode == "single_arm_7":
            return [
                basket_pose[0] / max_norm,
                basket_pose[1] / max_norm,
                object_pose[0] / max_norm,
                object_pose[1] / max_norm,
                dist / max_norm,
                dist_x / max_norm,
                dist_y / max_norm,
            ]

        if self.observation_mode == "dual_arm_8":
            return [
                self.robot_obs,
                basket_pose[0] / max_norm,
                basket_pose[1] / max_norm,
                object_pose[0] / max_norm,
                object_pose[1] / max_norm,
                dist / max_norm,
                dist_x / max_norm,
                dist_y / max_norm,
            ]

        obstacle_pose = self._get_object_position_from_gazebo(self.obstacle_name)
        return [
            basket_pose[0] / max_norm,
            basket_pose[1] / max_norm,
            object_pose[0] / max_norm,
            object_pose[1] / max_norm,
            dist / max_norm,
            dist_x / max_norm,
            dist_y / max_norm,
            obstacle_pose[0] / max_norm,
            obstacle_pose[1] / max_norm,
            self.dist_obs / max_norm,
            self.dist_obs_x / max_norm,
            self.dist_obs_y / max_norm,
            self.last_initial_joint_value,
            self.last_final_joint_value,
            self.last_releasing_time,
            self.last_duration,
        ]

    def step(self, action):
        obstacle_init_pose = self._get_object_position_from_gazebo(self.obstacle_name)[
            :2
        ]

        self.action(
            robot=self.robot,
            shoulder_initial_joint_value=action[0],
            shoulder_final_joint_value=action[1],
            releasing_time=action[2],
            duration_of_trajectory=action[3],
            object_label=self.object_name,
        )
        settled_after_throw = self._wait_for_objects_to_settle(
            [self.object_name, self.basket_name, self.obstacle_name],
            "post throw reward",
        )
        basket_pose, object_pose, dist, dist_x, dist_y = self.get_distance(
            self.object_name, self.basket_name
        )

        obstacle_current_pose = self._get_object_position_from_gazebo(
            self.obstacle_name
        )[:2]
        self.dist_obs = float(
            np.linalg.norm(obstacle_current_pose - obstacle_init_pose)
        )
        self.dist_obs_x = float(
            np.linalg.norm(obstacle_current_pose[0] - obstacle_init_pose[0])
        )
        self.dist_obs_y = float(
            np.linalg.norm(obstacle_current_pose[1] - obstacle_init_pose[1])
        )

        if self.dist_obs > 0.02:
            reward = 0.0
        elif dist < 0.15:
            reward = 1.0
            self.record_samples.writerow(
                [
                    self.robot,
                    basket_pose[0],
                    basket_pose[1],
                    basket_pose[2],
                    self.last_initial_joint_value,
                    self.last_final_joint_value,
                    self.last_releasing_time,
                    self.last_duration,
                    self.last_object_label,
                    object_pose[0],
                    object_pose[1],
                    object_pose[2],
                    dist,
                    dist_x,
                    dist_y,
                    reward,
                ]
            )
            self.samples_file.flush()
        else:
            alpha = 0.9
            reward = float(
                alpha * np.exp(-(dist**2) / 0.01)
                + (1.0 - alpha) * np.exp(-(dist**2) / 0.05)
            )

        terminated = True
        truncated = False
        observation = self.observe()
        info = {"rew": reward, "settled_after_throw": settled_after_throw}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        self.init_env()
        obs = self.observe()
        return obs, {}

    def close(self):
        if hasattr(self, "samples_file") and not self.samples_file.closed:
            self.samples_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = Node("tossing_object_with_obstacle_avoidance_v3")
    env = TossingObjectWithObstacleAvoidanceV3(node)

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("START Time =", start_time)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    logdir = Path("learnedPolicies") / "log_" / f"obs_{timestr}"
    logdir.mkdir(parents=True, exist_ok=True)

    test_mode = bool(node.declare_parameter("test_mode", False).value)
    eval_episodes = int(node.declare_parameter("eval_episodes", 50).value)
    model_path_param = str(node.declare_parameter("model_path", "").value).strip()

    def normalize_model_path(path: Path) -> Path:
        normalized = Path(str(path).replace(".zip.zip", ".zip")).expanduser()
        if normalized.suffix == ".zip":
            normalized = normalized.with_suffix("")
        return normalized

    def model_exists(path: Path) -> bool:
        return path.exists() or path.with_suffix(".zip").exists()

    def display_model_path(path: Path) -> Path:
        archive_path = path.with_suffix(".zip")
        return archive_path if archive_path.exists() else path

    def latest_model_base(pattern: str) -> Optional[Path]:
        matches = sorted(
            package_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not matches:
            return None
        return normalize_model_path(matches[0])

    if model_path_param:
        model_path = normalize_model_path(Path(model_path_param))
    else:
        package_root = find_package_path("intuition_throwing_service")
        candidate_paths = []
        latest_v3_model = latest_model_base("learnedPolicies/tossing_model_obs_*.zip")
        if latest_v3_model is not None:
            candidate_paths.append(latest_v3_model)
        candidate_paths.extend(
            [
                package_root / "learnedPolicies" / "tossing_model_20230808-061831",
                package_root
                / "learnedPolicies"
                / "throwing_model_dual_arm"
                / "tossing_model_dual_arm20230810-071513",
            ]
        )
        model_path = normalize_model_path(candidate_paths[0])
        for candidate in candidate_paths:
            normalized = normalize_model_path(candidate)
            if model_exists(normalized):
                model_path = normalized
                break
        if not model_exists(model_path):
            searched = ", ".join(
                str(display_model_path(normalize_model_path(path)))
                for path in candidate_paths
            )
            raise FileNotFoundError(f"No trained SAC model found. Searched: {searched}")

    if test_mode:
        print("TEST mode enabled")
        print("Loading trained model:", str(display_model_path(model_path).resolve()))
        model = SAC.load(str(model_path))
        env.set_observation_mode_for_model(model.observation_space)

        successes = 0
        total_reward = 0.0
        for episode_idx in range(eval_episodes):
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, _ = env.step(action)
            total_reward += float(reward)
            if reward >= 1.0:
                successes += 1
            if (episode_idx + 1) % 10 == 0 or episode_idx == eval_episodes - 1:
                print(
                    f"eval episode {episode_idx + 1}/{eval_episodes}, "
                    f"reward={reward:.3f}, success_rate={successes / float(episode_idx + 1):.3f}"
                )

        mean_reward = total_reward / float(max(eval_episodes, 1))
        success_rate = successes / float(max(eval_episodes, 1))
        print("Evaluation completed")
        print(f"mean_reward={mean_reward:.4f}")
        print(f"success_rate={success_rate:.4f}")
    else:
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))
        max_timesteps = int(node.declare_parameter("max_timesteps", 35000).value)
        model.learn(total_timesteps=max_timesteps, log_interval=1)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_dir = Path("learnedPolicies")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_dir / f"tossing_model_obs_{timestr}"

        print("saving the learned policy")
        model.save(str(model_name))
        model.save_replay_buffer(str(model_name) + "_buffer")
        print("model saved successfully!! model name =", str(model_name))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START Time =", start_time)
    print("END Time =", current_time)

    env.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
