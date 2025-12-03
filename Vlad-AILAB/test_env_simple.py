from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pick_place_env import PickPlaceEnv
import numpy as np

def main():
    print("Initializing environment...")
    env = PickPlaceEnv(headless=True)
    print("Resetting environment...")
    obs, _ = env.reset()
    print("Environment reset. Obs keys:", obs.keys())
    
    for i in range(10):
        action = env.action_space.sample()
        print(f"Step {i}, Action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        if done:
            print("Episode done, resetting...")
            env.reset()
            
    print("Closing environment...")
    env.close()
    print("Closing simulation app...")
    simulation_app.close()
    print("Done.")

if __name__ == "__main__":
    main()
