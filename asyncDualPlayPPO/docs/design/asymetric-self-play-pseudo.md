Algorithm 1: Asymmetric Self-Play

Require:

    θA​,θB​: Initial parameters for Alice and Bob

    η: RL learning rate

    β: Weight of BC (Behavioral Cloning) loss

for training_steps = 1, 2, ... do
    # Initialize behavior policy parameters
    theta_A_old <- theta_A
    theta_B_old <- theta_B

    # Parallel data collection
    for each rollout worker do
        # Collect replay buffers for Alice (D_A), Bob (D_B), and ABC (D_BC)
        D_A, D_B, D_BC = CollectRolloutData(theta_A_old, theta_B_old)
    end for

    # Optimize Alice: PPO loss with data popped from D_A
    theta_A <- theta_A - eta * grad(L_RL(D_A))

    # Optimize Bob: RL loss with D_B and ABC loss with D_BC
    theta_B <- theta_B - eta * grad(L_RL(D_B) + beta * L_ABC(D_BC))
end for


