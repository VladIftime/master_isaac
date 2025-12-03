# Launch Isaac Sim before any other imports
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt
import pandas as pd

class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.
    Increases the number of objects when the success rate is high.
    """
    def __init__(self, check_freq: int, max_objects: int = 5, verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.current_num_objects = 1
        self.max_objects = max_objects
        self.success_buffer = []

    def _on_step(self) -> bool:
        # Update success buffer with current step's results
        # self.locals['infos'] is a list of info dicts from the vectorized env
        for info in self.locals['infos']:
            if "grasp_success" in info:
                self.success_buffer.append(info["grasp_success"])
        
        # Keep buffer size manageable (e.g., last 20 episodes for quick feedback)
        if len(self.success_buffer) > 20:
            self.success_buffer = self.success_buffer[-20:]

        if self.n_calls % self.check_freq == 0:
            if len(self.success_buffer) > 0:
                success_rate = np.mean(self.success_buffer)
                n_episodes = len(self.success_buffer)
                
                # Get average reward for logging purposes
                avg_reward = 0.0
                try:
                    episode_rewards = self.training_env.env_method("get_episode_rewards")
                    if episode_rewards and len(episode_rewards[0]) > 0:
                        avg_reward = np.mean(episode_rewards[0][-n_episodes:])
                except:
                    pass
                
                print(f"Step {self.n_calls}: Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.2f} ({int(sum(self.success_buffer))}/{n_episodes})")
                
                if success_rate >= 0.9 and n_episodes >= 10: # Only increase if we have enough samples
                    if self.current_num_objects < self.max_objects:
                        self.current_num_objects += 1
                        print(f"Curriculum: Increasing number of objects to {self.current_num_objects}")
                        self.training_env.env_method("set_num_objects", self.current_num_objects)
                        # Clear buffer to verify performance with new difficulty
                        self.success_buffer = []
                    else:
                        pass
                        
        return True

from pick_place_env import PickPlaceEnv

# Custom Feature Extractor for Multi-Modal Input
class GraspNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # CNN for RGB (3 channels)
        self.rgb_cnn = self._build_cnn(3)
        
        # CNN for Depth (1 channel)
        self.depth_cnn = self._build_cnn(1)
        
        # CNN for Heightmap (1 channel)
        self.heightmap_cnn = self._build_cnn(1)
        
        # CNN for Goal Mask (1 channel)
        self.mask_cnn = self._build_cnn(1)
        
        # Compute output dimension
        with th.no_grad():
            sample_rgb = th.as_tensor(observation_space["rgb"].sample()[None]).float()
            sample_depth = th.as_tensor(observation_space["depth"].sample()[None]).float()
            # Heightmap is float32, so it is NOT transposed by SB3. We must permute it manually.
            sample_heightmap = th.as_tensor(observation_space["heightmap"].sample()[None]).float().permute(0, 3, 1, 2)
            sample_mask = th.as_tensor(observation_space["goal_mask"].sample()[None]).float()
            
            n_flatten_rgb = self.rgb_cnn(sample_rgb).shape[1]
            n_flatten_depth = self.depth_cnn(sample_depth).shape[1]
            n_flatten_heightmap = self.heightmap_cnn(sample_heightmap).shape[1]
            n_flatten_mask = self.mask_cnn(sample_mask).shape[1]
            
        self.total_flatten = n_flatten_rgb + n_flatten_depth + n_flatten_heightmap + n_flatten_mask
        
        self.linear = th.nn.Sequential(
            th.nn.Linear(self.total_flatten, features_dim),
            th.nn.ReLU()
        )

    def _build_cnn(self, input_channels):
        return th.nn.Sequential(
            th.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

    def forward(self, observations):
        # Normalize images
        rgb = observations["rgb"].float() / 255.0
        depth = observations["depth"].float() / 255.0
        heightmap = observations["heightmap"].float() # Heightmap is already float
        mask = observations["goal_mask"].float() / 255.0
        
        # Heightmap is float32, so it is NOT transposed by SB3. We must permute it manually.
        heightmap = heightmap.permute(0, 3, 1, 2)
        
        rgb_feat = self.rgb_cnn(rgb)
        depth_feat = self.depth_cnn(depth)
        heightmap_feat = self.heightmap_cnn(heightmap)
        mask_feat = self.mask_cnn(mask)
        
        concat_feat = th.cat([rgb_feat, depth_feat, heightmap_feat, mask_feat], dim=1)
        return self.linear(concat_feat)

def main():
    # Create Log Directory
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create Environment
    # We wrap it in DummyVecEnv for SB3
    env = DummyVecEnv([lambda: Monitor(PickPlaceEnv(headless=True), log_dir)])
    
    # Define Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=GraspNetwork,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    # Initialize Agent
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2, # Reduced for immediate debugging
        batch_size=2, # Reduced for immediate debugging
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir
    )
    
    # Epsilon-Greedy Exploration Wrapper
    # We monkey-patch the policy's forward method to inject random actions with probability epsilon
    original_forward = model.policy.forward
    epsilon = 0.1 # Exploration rate
    
    def epsilon_greedy_forward(obs, deterministic=False):
        # Call original forward to get values and log_probs (needed for PPO update)
        actions, values, log_probs = original_forward(obs, deterministic)
        
        # Apply epsilon-greedy
        if not deterministic and np.random.rand() < epsilon:
            # Generate random actions
            # Action space is [-1, 1] for 4 dimensions
            # We assume batch size is actions.shape[0]
            random_actions = th.rand_like(actions) * 2 - 1 # [-1, 1]
            return random_actions, values, log_probs
            
        return actions, values, log_probs
        
    # Apply the patch
    model.policy.forward = epsilon_greedy_forward
    # print(f"Epsilon-Greedy Strategy applied with epsilon={epsilon}")
    
    # Train
    print("Starting training...")
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="pick_place_model")
    curriculum_callback = CurriculumCallback(check_freq=1, verbose=1) # Check every episode
    
    try:
        model.learn(total_timesteps=1000, callback=[checkpoint_callback, curriculum_callback])
        model.save("pick_place_final")
        print("Training complete.")
        
        # Plotting
        try:
            print("Plotting training results...")
            monitor_path = os.path.join(log_dir, "monitor.csv")
            # Monitor wrapper creates monitor.csv, but we need to find the exact file.
            # SB3 Monitor usually creates monitor.csv if filename is provided, or monitor.csv.
            # We passed log_dir as filename prefix? No, we passed log_dir as folder.
            # Monitor(env, filename)
            # In main: Monitor(PickPlaceEnv(...), log_dir) -> filename will be log_dir.monitor.csv
            # Wait, Monitor takes (env, filename, allow_early_resets, ...)
            # If filename is a directory, it might fail or append.
            # Let's check how we initialized Monitor.
            # env = DummyVecEnv([lambda: Monitor(PickPlaceEnv(headless=True), log_dir)])
            # If log_dir is "./logs/", Monitor will try to write to "./logs/.monitor.csv" maybe?
            # Actually, Monitor expects a filename prefix.
            
            # Let's find the monitor file.
            monitor_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
            if monitor_files:
                monitor_file = os.path.join(log_dir, monitor_files[0])
                df = pd.read_csv(monitor_file, skiprows=1) # Skip first line (metadata)
                
                plt.figure(figsize=(10, 5))
                plt.plot(df['r'].rolling(window=10).mean(), label='Rolling Mean Reward (10 eps)')
                plt.plot(df['r'], alpha=0.3, label='Episode Reward')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(log_dir, "training_curve.png"))
                print(f"Plot saved to {os.path.join(log_dir, 'training_curve.png')}")
            else:
                print("No monitor file found for plotting.")
                
        except Exception as e:
            print(f"Error plotting: {e}")
            
    except KeyboardInterrupt:
        print("Training interrupted.")
        model.save("pick_place_interrupted")
    except Exception as e:
        print(f"Exception during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
