import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import carb
from isaacsim.core.api import World
from utils_robots.tasks.pick_place_task import UR5ePickPlace
from utils_robots.controllers.pick_place_controller_rmpflow import RMPFlowPickPlaceController
from scipy.spatial.transform import Rotation as R
from isaacsim.core.utils.rotations import euler_angles_to_quat

class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, headless=True, render_mode=None):
        self.render_mode = render_mode
        self.headless = headless
        
        # Initialize World
        self.world = World(physics_dt=0.01, stage_units_in_meters=1.0)
        
        # Initialize Task
        # We spawn max_objects=5 to allow curriculum learning
        self.task = UR5ePickPlace(number_of_objects=1, max_objects=5, randomize_position=True)
        self.world.add_task(self.task)
        self.world.reset()
        
        # Get Robot and Controller
        task_params = self.task.get_params()
        self.robot = self.world.scene.get_object(task_params["robot_name"]["value"])
        self.controller = RMPFlowPickPlaceController(
            name="pick_place_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
            events_dt=[3.0, 3.0, 2.0, 3.0, 0.5, 1.0, 2.0, 1.0, 3.0, 3.0, 2.0, 3.0],
            end_effector_offset=np.array([0, 0, 0.22]),
            end_effector_initial_height=0.55,
        )
        self.articulation_controller = self.robot.get_articulation_controller()
        
        # Define Spaces
        # RGB: 84x84x3
        # Depth: 84x84x1
        # Heightmap: 84x84x1 (Rasterized)
        self.img_size = 84
        
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=255, shape=(self.img_size, self.img_size, 1), dtype=np.uint8),
            "heightmap": spaces.Box(low=-np.inf, high=np.inf, shape=(self.img_size, self.img_size, 1), dtype=np.float32),
            "goal_mask": spaces.Box(low=0, high=255, shape=(self.img_size, self.img_size, 1), dtype=np.uint8),
        })
        
        # Action: X, Y, Theta, GraspProb
        # X: [-1, 1] -> [0.2, 0.8]
        # Y: [-1, 1] -> [-0.25, 0.35]
        # Theta: [-1, 1] -> [-pi, pi]
        # GraspProb: [-1, 1] -> [0, 1] (Sigmoid)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.workspace_limits = np.asarray([[0.2, 0.8], [-0.25, 0.35]])
        self.goal_object_name = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        # Step a few times to ensure physics and rendering are stable
        for _ in range(5):
            self.world.step(render=True)
        self.task.post_reset() # Randomize objects and hide unused ones
        
        # Select a random goal object from the active objects
        # The task stores active objects in self.task.objects_list (which are all objects)
        # But we only placed self.task.number_of_objects
        # We need to pick one of the active ones.
        # In post_reset, it places objects 0 to number_of_objects-1.
        # So we can pick an index from 0 to number_of_objects-1.
        if self.task.number_of_objects > 0:
            goal_idx = np.random.randint(0, self.task.number_of_objects)
            self.goal_object_name = self.task.get_object_name(goal_idx)
            # print(f"DEBUG: Selected goal object: {self.goal_object_name}")
        else:
            self.goal_object_name = None
            
        self.controller.reset()
        
        # Wait for scene to settle
        for _ in range(60):
            self.world.step(render=False)
            
        return self._get_obs(), {}
        
    def step(self, action):
        # Decode Action
        x_norm, y_norm, theta_norm, grasp_prob_logit = action
        
        # Decode Grasp Probability
        # Map [-1, 1] to probability [0, 1]
        # We can use sigmoid on the logit (which is just the raw output)
        # But the action is clipped to [-1, 1] by the wrapper usually? 
        # Actually PPO outputs raw actions, but the Env expects them in action_space.
        # Let's assume the network outputs values in [-1, 1].
        # We can map [-1, 1] linearly to [0, 1] for simplicity.
        predicted_grasp_prob = (grasp_prob_logit + 1) / 2.0
        
        # Map to workspace
        x = (x_norm + 1) / 2 * (self.workspace_limits[0, 1] - self.workspace_limits[0, 0]) + self.workspace_limits[0, 0]
        y = (y_norm + 1) / 2 * (self.workspace_limits[1, 1] - self.workspace_limits[1, 0]) + self.workspace_limits[1, 0]
        theta = theta_norm * np.pi
        
        picking_position = np.array([x, y, 0.05]) # Assume object height ~5cm
        
        # Calculate Target Orientation
        # Default Down: [pi, 0, pi]
        q_default_isaac = euler_angles_to_quat(np.array([np.pi, 0, np.pi]))
        r_default = R.from_quat([q_default_isaac[1], q_default_isaac[2], q_default_isaac[3], q_default_isaac[0]])
        r_z = R.from_euler('z', theta)
        r_target = r_z * r_default
        q_target_scipy = r_target.as_quat()
        target_orientation = np.array([q_target_scipy[3], q_target_scipy[0], q_target_scipy[1], q_target_scipy[2]])
        
        # Execute Pick and Place Sequence
        # We only care about the pick part for now, but the controller does the whole thing.
        # We will run until the controller is done or grasp is checked.
        
        grasp_success = False
        done = False
        
        # Run simulation loop
        while not self.controller.is_done():
            self.world.step(render=self.render_mode is not None)
            
            current_joint_positions = self.robot.get_joint_positions()
            actions = self.controller.forward(
                picking_position=picking_position,
                current_joint_positions=current_joint_positions,
                end_effector_orientation=target_orientation,
            )
            self.articulation_controller.apply_action(actions)
            
            # Check grasp status after pick attempt (Phase 8 is Turned Intermediate, after Grasp Check)
            if self.controller._event == 8 and self.controller.get_grasp():
                grasp_success = True
                
            # Break early if we just want to test picking
            # But let's let it finish to be safe
            
        # Calculate Reward
        # Minimum-Time Specification
        # Reward = -Time Taken (if successful)
        # Reward = -Max Time * 2 (if failed)
        
        total_time = self.controller.get_total_time()
        # Estimate max time based on events_dt sum + buffers
        # Sum of events_dt is approx 23s. Let's say max allowed is 30s.
        MAX_TIME = 30.0
        
        if grasp_success:
            task_reward = -total_time
        else:
            task_reward = -MAX_TIME * 2.0
        
        # 2. Prediction Reward: Binary Cross Entropy
        # Reward is negative log loss (maximized)
        # We clamp probabilities to avoid log(0)
        p = np.clip(predicted_grasp_prob, 1e-6, 1.0 - 1e-6)
        y = 1.0 if grasp_success else 0.0
        prediction_reward = y * np.log(p) + (1 - y) * np.log(1 - p)
        
        # Scale prediction reward to be comparable to task reward
        # BCE is usually negative small number. Task reward is now ~ -10 to -60.
        # We want prediction to be a small guidance.
        reward = task_reward + 0.1 * prediction_reward
        
        # print(f"DEBUG: Task Reward: {task_reward}, Pred Reward: {prediction_reward:.4f}, Total: {reward:.4f}")
        done = True
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {"grasp_success": grasp_success}
        
    def set_num_objects(self, n: int):
        """Set the number of active objects for curriculum learning."""
        if n > self.task.max_objects:
            # print(f"Warning: Requested {n} objects, but max is {self.task.max_objects}. Setting to max.")
            n = self.task.max_objects
        self.task.number_of_objects = n
        # print(f"Set number of objects to {n}")
        
    def _get_obs(self):
        obs = self.task.get_observations(goal_object_name=self.goal_object_name)
        
        # Process RGB
        rgb = obs["rgb_image"]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        
        # Process Depth
        depth = obs["depth_image"]
        depth = cv2.resize(depth, (self.img_size, self.img_size))
        depth = np.expand_dims(depth, axis=-1)
        
        # Process Heightmap
        # The task returns a point cloud (Nx3) as "pointcloud"
        # We need to rasterize it to an image
        pc = obs["pointcloud"]
        heightmap_img = self._rasterize_heightmap(pc)
        
        # Process Goal Mask
        if "goal_mask" in obs:
            goal_mask = obs["goal_mask"]
            goal_mask = cv2.resize(goal_mask, (self.img_size, self.img_size))
            goal_mask = np.expand_dims(goal_mask, axis=-1)
        else:
            goal_mask = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)

        return {
            "rgb": rgb,
            "depth": depth,
            "heightmap": heightmap_img,
            "goal_mask": goal_mask
        }
        
    def _rasterize_heightmap(self, pointcloud):
        # Rasterize Nx3 point cloud to (img_size, img_size, 1) image
        # Map X, Y to pixel coordinates
        
        grid = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        if len(pointcloud) == 0:
            return np.expand_dims(grid, axis=-1)
            
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        
        # Normalize X, Y to [0, img_size]
        x_min, x_max = self.workspace_limits[0]
        y_min, y_max = self.workspace_limits[1]
        
        u = ((x - x_min) / (x_max - x_min) * self.img_size).astype(int)
        v = ((y - y_min) / (y_max - y_min) * self.img_size).astype(int)
        
        # Filter valid indices
        valid = (u >= 0) & (u < self.img_size) & (v >= 0) & (v < self.img_size)
        
        # Simple max projection
        # Iterate or use fancy indexing (might overwrite, max is better)
        # For speed, we just assign. For better results, we should take max Z.
        
        # Sort by Z so highest points overwrite lower ones
        sort_idx = np.argsort(z)
        u = u[sort_idx]
        v = v[sort_idx]
        z = z[sort_idx]
        valid = valid[sort_idx]
        
        grid[v[valid], u[valid]] = z[valid]
        
        return np.expand_dims(grid, axis=-1)
        
    def close(self):
        self.world.reset()
        # Do not close simulation app here if it's shared, but for now it's fine
