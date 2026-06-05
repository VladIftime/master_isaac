Algorithm 2: CollectRolloutData

Require:

    θAold​,θBold​: Behavior policy parameters for Alice and Bob

    πA​,πB​: Policies for Alice and Bob
# Initialize empty replay buffers
D_A <- {}
D_B <- {}
D_BC <- {}

# Initialize success flag (True implies Bob is allowed to try solving)
bob_succeeded <- True

for number_of_goals = 1, ..., 5 do
    # 1. Alice generates a trajectory and a goal
    tau_A, g = GenerateAliceTrajectory(pi_A, theta_A_old)

    # If the goal is invalid (e.g., no object moved), end the episode
    if g is invalid then
        break
    end if

    # 2. Bob tries to solve the goal (only if he hasn't failed yet)
    if bob_succeeded is True then
        tau_B, bob_succeeded = GenerateBobTrajectory(pi_B, theta_B_old, g)
        D_B.append(tau_B)
    end if

    # 3. Compute Alice's reward based on Bob's performance
    # Alice gets a bonus if Bob failed; otherwise standard reward
    r_A = ComputeAliceReward(bob_succeeded)
    
    # Overwrite the last reward in Alice's trajectory with r_A
    tau_A[-1].reward = r_A
    D_A.append(tau_A)

    # 4. Alice Behavioral Cloning (ABC) Logic
    # If Bob failed, use Alice's trajectory as a demonstration for Bob
    if bob_succeeded is False then
        # Relabel Alice's trajectory to be goal-augmented (goal = final state)
        tau_BC = RelabelDemonstration(tau_A, g, pi_B, theta_B_old)
        D_BC.append(tau_BC)
    end if
end for

return D_A, D_B, D_BC