import re
import matplotlib.pyplot as plt
import os

log_file = '/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/logs/25_rot_NLL_dense_updates_only.out'

alice_iters = []
alice_rewards = []
alice_value_loss = []
alice_surrogate_loss = []

bob_iters = []
bob_success_rate = []
bob_rewards = []
bob_value_loss = []
bob_surrogate_loss = []
bob_abc_loss = []
bob_pos_error = []
bob_rot_error = []

current_agent = None
current_iter = None

with open(log_file, 'r') as f:
    for line in f:
        m_alice = re.search(r'ALICE UPDATE (\d+)', line)
        if m_alice:
            current_agent = 'Alice'
            current_iter = int(m_alice.group(1))
            alice_iters.append(current_iter)
            continue
            
        m_bob = re.search(r'BOB UPDATE (\d+)', line)
        if m_bob:
            current_agent = 'Bob'
            current_iter = int(m_bob.group(1))
            bob_iters.append(current_iter)
            continue
            
        if current_agent == 'Alice':
            m_rew = re.search(r'Rewards:\s+mean=([-\d.]+)', line)
            if m_rew:
                alice_rewards.append(float(m_rew.group(1)))
                
            m_loss = re.search(r'Losses:\s+value=([-\d.]+)\s+\|\s+surrogate=([-\d.]+)', line)
            if m_loss:
                alice_value_loss.append(float(m_loss.group(1)))
                alice_surrogate_loss.append(float(m_loss.group(2)))
                
        elif current_agent == 'Bob':
            m_sr = re.search(r'Success Rate:\s+([-\d.]+)', line)
            if m_sr:
                bob_success_rate.append(float(m_sr.group(1)))
                
            m_rew = re.search(r'Rewards:\s+mean=([-\d.]+)', line)
            if m_rew:
                bob_rewards.append(float(m_rew.group(1)))
                
            m_loss = re.search(r'Losses:\s+value=([-\d.]+)\s+\|\s+surrogate=([-\d.]+)\s+\|\s+ABC=([-\d.]+)', line)
            if m_loss:
                bob_value_loss.append(float(m_loss.group(1)))
                bob_surrogate_loss.append(float(m_loss.group(2)))
                bob_abc_loss.append(float(m_loss.group(3)))
                
            m_err = re.search(r'Errors:\s+pos=([-\d.]+)\s+\|\s+rot=([-\d.]+)', line)
            if m_err:
                bob_pos_error.append(float(m_err.group(1)))
                bob_rot_error.append(float(m_err.group(2)))

print(f"Extracted {len(alice_iters)} Alice updates and {len(bob_iters)} Bob updates.")

# Create plots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Alice Rewards
axs[0, 0].plot(alice_iters[:len(alice_rewards)], alice_rewards, label='Alice Mean Reward')
axs[0, 0].set_title('Alice Mean Reward')
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].grid(True)

# Alice Losses
axs[0, 1].plot(alice_iters[:len(alice_value_loss)], alice_value_loss, label='Value Loss')
axs[0, 1].plot(alice_iters[:len(alice_surrogate_loss)], alice_surrogate_loss, label='Surrogate Loss')
axs[0, 1].set_title('Alice Losses')
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Bob Success Rate
axs[1, 0].plot(bob_iters[:len(bob_success_rate)], bob_success_rate, label='Success Rate', color='green')
axs[1, 0].set_title('Bob Success Rate')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Success Rate')
axs[1, 0].grid(True)

# Bob Rewards
axs[1, 1].plot(bob_iters[:len(bob_rewards)], bob_rewards, label='Bob Mean Reward')
axs[1, 1].set_title('Bob Mean Reward')
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Reward')
axs[1, 1].grid(True)

# Bob Losses
axs[2, 0].plot(bob_iters[:len(bob_value_loss)], bob_value_loss, label='Value Loss')
axs[2, 0].plot(bob_iters[:len(bob_surrogate_loss)], bob_surrogate_loss, label='Surrogate Loss')
axs[2, 0].plot(bob_iters[:len(bob_abc_loss)], bob_abc_loss, label='ABC Loss')
axs[2, 0].set_title('Bob Losses')
axs[2, 0].set_xlabel('Iteration')
axs[2, 0].set_ylabel('Loss')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Bob Errors
axs[2, 1].plot(bob_iters[:len(bob_pos_error)], bob_pos_error, label='Pos Error (m)')
axs[2, 1].plot(bob_iters[:len(bob_rot_error)], bob_rot_error, label='Rot Error (rad)')
axs[2, 1].set_title('Bob Final Position/Rotation Errors')
axs[2, 1].set_xlabel('Iteration')
axs[2, 1].set_ylabel('Error')
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
output_path = '/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/logs/plots/training_plots_25_rot_NLL.png'
plt.savefig(output_path)
print(f"Plots saved to {output_path}")
