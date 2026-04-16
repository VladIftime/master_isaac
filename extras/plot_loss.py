import numpy as np
import matplotlib.pyplot as plt

# Define the distance range (e.g., 0 to 1.5 meters)
distances = np.linspace(0, 1.5, 500)

# Parameters for the bounded potential function
C = 3  # The absolute maximum cumulative reward
k = 5  # The decay rate (how fast it reaches the cap)

# Calculate the cumulative reward (the potential)
cumulative_reward = C * (1 - np.exp(-k * distances))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(distances, cumulative_reward, label=r'$\Phi(s) = 0.75 (1 - e^{-5.0 \cdot dist})$', color='blue', linewidth=2)

# Add reference lines to show it never crosses the bounds
plt.axhline(y=3, color='red', linestyle='--', label='Dense Reward Cap (0.75)')
plt.axhline(y=1.0, color='green', linestyle=':', label='Sparse Reward (Valid Goal: 1.0)')
plt.axhline(y=1.0, color='red', linestyle='--', label='Sparse Reward (Valid Goal + Succes Bob: 1)')
plt.axhline(y=6.0, color='green', linestyle=':', label='Sparse Reward (Valid Goal + Fail Bob: 6.0)')

# Formatting
plt.title("Alice's Cumulative Dense Reward vs. Object Distance", fontsize=14)
plt.xlabel('Distance from Start (meters)', fontsize=12)
plt.ylabel('Cumulative Dense Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.xlim(0, 1.5)
plt.ylim(0, 6.2)

# Show the plot
plt.show()