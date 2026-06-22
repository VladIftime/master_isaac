Dual-Arm Push-Grasping Learning Framework.

The architecture is divided into two main loops: the Task Loop (execution) and the Training Loop (learning).
1. The Environment & Input Processing

    Environment (Isaac Gym): The simulation environment features a large number of dual-arm UR5e robots. The goal is to declutter scenes by pushing and grasping objects. The target object is highlighted in green.

    Input Data (Heightmap-RGBD): The system captures visual data using an RGB-D camera. This is converted into top-down height maps.

        RGB Data: Sent to the Feature Extraction Backbone.

        Depth Data: Bypasses the backbone and is later fused with the extracted features to form the complete state (St​).

2. Feature Extraction Backbone

This is the "vision" component of the system.

    Architecture: It uses a ResNet-50 architecture followed by PixelShuffle and DUC (Dense Upsampling Convolution) layers to process the image features at different scales (224x224 input → feature maps).

    Pre-training: Notably, this model is pre-trained on GraspNet-1Billion, meaning it already "knows" what valid grasp features look like before this specific training begins.

    Interaction: It takes the raw RGB image and outputs a dense feature map that represents the scene's semantic information.

3. Reinforcement Learning (RL) Agents

The system uses an Actor-Critic style architecture (specifically PPO, or Proximal Policy Optimization, as noted in the caption).
A. Policy Model (The Actor)

    Role: This model decides what action to take based on the current state.

    Architecture: It consists of three convolutional layers (Conv2d x 3), followed by a Flatten layer, and a series of Fully Connected (FC 64) layers with Tanh activation functions.

    Interaction:

        Input: Receives the combined state features (St​) from the backbone 

        Output: Produces an "Output Action" feature map. This map is reshaped and sent to the decoders.

B. Value Model (The Critic)

    Role: This model estimates the "value" of being in a specific state (how good is the current situation?). This helps reduce variance during training.

    Architecture: Structurally identical to the Policy Model (3 Conv layers, Flatten, FC layers with Tanh).

    Interaction:

        Input: Takes the state (St​) or next state (St+1​) from the training batch.

        Output: Produces a single value estimate (v(st​)) which is compared against real rewards to calculate the MSE Loss (Mean Squared Error).

4. Action Decoders

The "Output Action" from the Policy Model is essentially a high-level feature map. To become a physical robot movement, we add the depth map to it and it passes through specialized decoders :

    Grasp Decoder: Converts the features into a 6 DoF (Degrees of Freedom) Grasp Pose (position and orientation for the gripper).

    Push Decoder: Converts features into a push trajectory, deciding between One Path or Two Paths (likely referring to single-arm vs. dual-arm pushing strategies).

5. Motion Planning & Execution

    Inverse Kinematics: The decoded poses/paths are geometric goals. The "Motion Planning Inverse Kinematics" module calculates the actual joint angles needed for the robot arms to reach those goals.

    Task Loop: This physical action is executed in the environment, which alters the scene (moving objects), creating a new state for the next cycle.

6. The Training Loop & Reward System

This loop is responsible for improving the models over time.

    Experience Buffer: As the robot acts, data tuples (st​,a,r,st+1​) (State, Action, Reward, Next State) are stored here.

    Reward Model (Fuzzy Reward): Instead of a simple binary reward (success/fail), the system uses a Fuzzy Reward module. This likely evaluates the quality of the action using fuzzy logic rules to provide more nuanced feedback to the agent during training.

    Update Mechanism:

        A Training Batch is sampled from the buffer.

        Generalized Advantage Estimation (GAE): This calculates the "advantage" (how much better an action was than expected). It uses inputs from the Value Model and the Reward Model.

        Loss Calculation:

            The Policy Model is updated to maximize expected reward.

            The Value Model is updated to minimize the error in its value predictions (MSE Loss).

Summary of Interactions

    See: Environment → Camera → Feature Backbone.

    Think: Features → Policy Model → Action Map.

    Translate: Action Map → Decoders (Push/Grasp) → Inverse Kinematics.

    Act: Robot moves in Environment.

    Learn: The Reward Model evaluates the result → Value Model estimates state quality → Policy Model updates its weights to perform better next time.