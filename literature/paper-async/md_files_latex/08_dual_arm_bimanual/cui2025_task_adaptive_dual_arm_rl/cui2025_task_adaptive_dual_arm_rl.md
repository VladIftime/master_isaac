# A Task-Adaptive Deep Reinforcement Learning Framework for Dual-Arm Robot Manipulation

Yuanzhe Cui, Zhipeng Xu<sup>®</sup>, Lou Zhong, Pengjie Xu, Yichao Shen, *Graduate Student Member, IEEE*, and Qirong Tang<sup>®</sup>, *Member, IEEE* 

Abstract—Closed-chain manipulation occurs when several robot arms perform tasks in cooperation. It is complex to control a dual-arm system because it requires flexible and adaptable operation ability to realize closed-chain manipulation. In this study, a deep reinforcement learning (DRL) framework based on actor-critic algorithm is proposed to drive the closed-chain manipulation of a dual-arm robotic system. The proposed framework is designed to train dual robot arms to transport a large object cooperatively. In order to sustain strict constraints of closed-chain manipulation, the actor part of the proposed framework is designed in a leader-follower mode. The leader part consists of a policy trained from the DRL algorithm and works on the leader arm. The follower part consists of an inverse kinematics solver based on Damped Least Squares (DLS) and works on the follower arm. Two experiments are designed to prove the task adaptability, one of which is manipulating an object to a random pose within a defined range, the other is manipulating a delicate structural object within a narrow space.

Note to Practitioners—In common industrial manipulation scenarios, there are requirements to employ robotic arms to transport a large object relative to the robotic arm, e.g., moving a payload onto a loader and assembling big craft parts. It is a cost-effective way to use a dual-arm system to extend the loading capacity of robotic arms while preserving the flexibility of manipulation. Moreover, the dual-arm system is expected to manipulate different objects without complicated reprogram-

Manuscript received 22 November 2023; accepted 2 January 2024. Date of publication 18 January 2024; date of current version 15 January 2025. This article was recommended for publication by Associate Editor C. Yang and Editor J. Yi upon evaluation of the reviewers' comments. This work was supported in part by the Project of the National Natural Science Foundation of China under Grant 62373285, in part by the Shanghai 2021 Science and Technology Innovation Action Plan with the Special Project of Biomedical Science and Technology Support under Grant 21S31902800, in part by the Key Pre-Research Project of the 14th-Five-Year-Plan on Common Technology, in part by the National Major Talent Plan Project under Grant 2022-XXXX-XXX-079, in part by the Fundamental Research Project under Grant XXXX2022YYYC133, in part by the Shanghai Industrial Collaborative Innovation Project (Industrial Development Category) under Grant HCXBCY-2022-051, in part by the Project of the Space Structure and Mechanism Technology Laboratory of China Aerospace Science and Technology Group Company Ltd. under Grant YY-F805202210015, and in part by the Project of the National Laboratory of Space Intelligent Control under Grant HTKJ2023KL502016. (Corresponding author: Qirong Tang.)

Yuanzhe Cui, Zhipeng Xu, Lou Zhong, Pengjie Xu, and Qirong Tang are with the Laboratory of Robotics and Multibody System, School of Mechanical Engineering, Tongji University, Shanghai 201804, China (e-mail: qirong.tang@outlook.com).

Yichao Shen is with the Laboratory of Robotics and Multibody System, School of Mechanical Engineering, Tongji University, Shanghai 201804, China, and also with the Institute of Engineering and Computational Mechanics, University of Stuttgart, 70569 Stuttgart, Germany.

This article has supplementary downloadable material available at https://doi.org/10.1109/TASE.2024.3352584, provided by the authors.

Digital Object Identifier 10.1109/TASE.2024.3352584

ming, especially in small batch production scenarios. This study proposes a task-adaptive deep reinforcement learning framework for dual-arm robot manipulation. The task adaptability includes two specific aspects, one being adaptability in targeting the pose, such as manipulating an object to a random pose within a specified range. The other is the adaptivity on the task prerequisites such as manipulating a delicate structural object within a narrow space. For future research, the dual-arm system may autonomously plan the grab positions, and additional investigations should address more common scenarios involving various object shapes.

Index Terms—Dual-arm robot manipulation, deep reinforcement learning.

#### <span id="page-0-0"></span>I. INTRODUCTION

WITH robots playing increasingly important role in industrial manipulation of large and irregular objects, single robot arms face challenges in efficiently handling such tasks. Hence, a cooperative manipulation system, such as the dual-arm system, holds substantial promise within the contemporary industrial landscape. One of the most challenging issues in cooperative manipulation is the strict closed-chain constraint, which imposes exacting demands on the operational state of a dual-arm system. Consequently, controlling dual robot arms is more demanding than controlling a single robot arm.

<span id="page-0-2"></span><span id="page-0-1"></span>Typically, motion planning for a single robot arm can be achieved through spline interpolation methods [1] or sampling-based methods [2]. However, interpolation methods struggle to satisfy the closed-chain constraints within a dual-arm system. Sampling-based methods require random sampling to approximate the connectivity of robot configurations [3]. Nevertheless, satisfying the closed-chain constraints of a dual-arm system through random configuration sampling is highly challenging due to the significantly smaller dimension of the closed-chain constrained manifold compared to the complete configuration space [4].

<span id="page-0-5"></span><span id="page-0-4"></span><span id="page-0-3"></span>To employ sampling-based methods in motion planning for closed-chain systems, Trinkle et al. transform the closed-chain structure by disconnecting a joint at one end of the base link [5]. This approach effectively addresses the motion planning challenges in closed-chain systems with spherical joints. In a different approach, the authors of [6] combine the inverse kinematics switch strategy with the classical BiRRT algorithm. Additionally, other researchers have explored methods for

1558-3783 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

<span id="page-1-3"></span>generating configurations that adhere to closed-chain constraints, as seen in works by Kim et al. [\[7\]](#page-12-6) and Lamiraux and Mirabel [\[8\]. R](#page-12-7)ecently, it is common to project sampled joint configuration to closed-chain constrained manifold [\[9\].](#page-12-8) On the basis of project operation, researches achieve to apply sampling-based methods (e.g. RRT, RRT-connect and PRM) to constrained motion planning [\[10\].](#page-12-9) Furthermore, solving closed-chain motion planning at a global level is feasible. For instance, Wu et al. compute dual-arm joint trajectories using a quadratic program-based inverse kinematic solver [\[11\].](#page-12-10) The primary concept underlying this method is solving a set of configurations based on a global *SE*(3) object trajectory. Jang et al. sample configuration nodes for multiple robot arms by extracting object poses efficiently to fulfill the closed-chain constraint. They then employ a PRM-based method to connect multiple pairs of initial and target object poses [\[9\]. H](#page-12-8)owever, sample-based methods require significant computation time prior to a manipulation process, and each result is not adaptable to changes in task states.

<span id="page-1-8"></span>Reinforcement learning (RL) is a branch of machine learning that focuses on agents' actions within an environment to maximize cumulative rewards [\[12\].](#page-12-11) Numerous applications of RL in robotics involve training policies to converge toward objectives through the influence of reward-based incentives [\[13\]. C](#page-12-12)onventional RL methods often represent policies using tabular rules, which are inadequate for high-dimensional or continuous state inputs. But in recent years, advancements in deep neural networks have enabled reinforcement learning to handle high-dimensional data [\[14\],](#page-12-13) [\[15\].](#page-12-14) Consequently, deep reinforcement learning (DRL) is well-suited for highdimensional and continuous robotic tasks. Yan et al. employ the soft Q-learning method to control the capture behavior of free-floating space robots [\[16\]. D](#page-12-15)RL methods can also tackle nonprehensile rearrangement problems [\[17\]. I](#page-12-16)n addressing the closed-chain manipulation problem of humanoid robots, Sugimoto and Morimoto introduce a model-based reinforcement learning method for manipulating a rod, focusing on the position constraint of the rod end and disregarding the attitude constraint [\[18\].](#page-12-17) Xu et al. introduce a coordinated control method based on reinforcement learning for multiple mobile manipulators with closed-chain constraints [\[19\]. H](#page-12-18)owever, the manipulation tasks are relatively straightforward. A persistent challenge is the development of a framework to extend the application of reinforcement learning from a single-arm system to one with multiple arms.

<span id="page-1-12"></span>In continuous state tasks, model-free actor-critic methods are extensively employed due to their effectiveness in updating policies and their capability to handle high-dimensional state spaces with relative ease. There are several improved actor-critic methods works well in various applications. Lillicrap et al. introduce the deep deterministic policy gradient method (DDPG), which optimizes policies to perform as effectively as controllers based on known models [\[20\].](#page-12-19) Haarnoja et al. introduce an actor-critic method that maximizes entropy while optimizing expected rewards. Consequently, their framework prioritizes extensive state sampling to accomplish tasks in a stochastic manner. This deep reinforcement learning (DRL) algorithm, also referred to as the Soft Actor-

<span id="page-1-5"></span><span id="page-1-4"></span><span id="page-1-2"></span><span id="page-1-0"></span>Fig. 1. Demonstration scene I: loading truck.

<span id="page-1-7"></span><span id="page-1-6"></span><span id="page-1-1"></span>Fig. 2. Demonstration scene II: small batch production.

<span id="page-1-15"></span><span id="page-1-9"></span>Critic (SAC), significantly mitigates the challenges related to convergence properties [\[21\].](#page-12-20)

<span id="page-1-16"></span><span id="page-1-13"></span><span id="page-1-11"></span><span id="page-1-10"></span>In this study, a task-adaptive DRL framework based on actor-critic algorithm is proposed to drive the manipulation of a dual-arm system within closed-chain constraints. The task adaptability encompasses two specific dimensions: (i) one pertains to adapting to the target pose, such as manipulating an object to a random pose within a defined range, while (ii) the other focuses on adaptability in object contact, particularly handling delicate structures in confined spaces. Furthermore, owing to the generalization capabilities of deep neural networks [\[22\], th](#page-12-21)e motion policy trained using this framework can handle tasks with varying configurations. Two representative application scenarios are depicted in Fig. [1](#page-1-0) and Fig. [2.](#page-1-1) Figure [1](#page-1-0) illustrates the transportation of a regular chair onto a truck. Figure [2](#page-1-1) demonstrates a small batch production, where a wooden frame is being manipulated through a narrow space. The method proposed in this paper is well-suited for tasks that involve occasional changes in the start-goal configuration. Another potential application is in the domain of palletizing robots, where the proposed method exhibits proficiency in loading trucks of varying sizes. Due to its relatively extended operating time, this method is not suitable for efficient pipeline tasks.

<span id="page-1-14"></span>This paper is organized as follows. In Section [II,](#page-2-0) the literature review is presented. In Section [III,](#page-3-0) a focused manipulation task is described and formulated. Besides, constraints of dual-arm system are introduced in order to design reward formation. In Section [IV,](#page-4-0) details of actor and critic parts in the proposed framework is illustrated. Section [V](#page-7-0) conducts two experiments with simulations and practical robot arms to verify the task adaptability of the proposed method. Finally, Section [VI](#page-11-0) draws the conclusion.

# II. LITERATURE REVIEW

<span id="page-2-0"></span>This section presents a review of the relevant literature, categorizing it into two sections: (i) traditional methods for cooperative manipulation, and (ii) learning methods for dualarm systems. A variety of practical and traditional methods are available for addressing cooperative manipulation scenarios. Traditional methods are derived from sampling-based algorithms. Sampling-based method requires global environmental information. This information is necessary at the outset of motion planning and should include details about all obstacles. Within the subsection on learning methods, the key issue is the methodology for defining constraints and rewards. In this study, constraints and rewards are defined based on the dual-arm manipulation process and task settlement.

# *A. Traditional Methods for Cooperative Manipulation*

A traditional method for motion planning in dual-arm systems is planning the trajectory of the target object. Throughout the manipulation process, both arms adhere to the closed-chain constraints. Luh and Zheng studies the constraint relationship in the motion control of dual-arm industrial manipulators [\[23\].](#page-12-22) The constraint relations between the two arms of the closedchain system, such as joint position, velocity and driving force are derived in this research. A common application of dual-arm manipulation is object transportation. Braun et al. introduced a distributed planning method capable of real-time trajectory modifications for dual-arm systems during object transportation [\[24\]. D](#page-12-23)ual-arm systems typically operate in a leader-follower mode, with one arm designated as the leader arm and the other as the follower arm. Leader arm takes on a primary role in guiding or setting the trajectory, while the follower arm follows or replicates the movements of the leader arm.

<span id="page-2-4"></span><span id="page-2-3"></span><span id="page-2-2"></span>When the dual-arm systems need to perform some complex tasks, such as collaborative packaging or assembly, it is also necessary to consider the problems of contact and collision, dynamic compliance, control precision. Krüger et al. study the compliant contact problem of collaborative assembly [\[25\]. D](#page-12-24)uring the assembly phase, they proposed a contact compliance operation method that enables safe coexistence of human workers and dual-arm systems within the same workspace. Makris et al. focus on program design for collaborative dual-arm assembly, aiming to reduce the complexity of programming. In the industrial production environment, pegin-hole assembly is a typical manipulation task. Suárez-Ruiz and Pham study on the collaborative dexterous peg-in-hole assembly manipulation of dual-arm systems [\[26\]. T](#page-12-25)hey introduce methods for workspace optimization, motion planning, and force-position control in the context of dual-arm peg-inhole manipulation. Building upon the method of peg-in-hole manipulation, Suárez-Ruiz et al. systematically present a method for accomplishing the assembly of IKEA chairs in <span id="page-2-5"></span>unstructured environments using dual-arm systems [\[27\]. T](#page-12-26)he research presented in [\[27\]](#page-12-26) is regarded as a powerful approach applicable in various domains, including electronics manufacturing, aircraft manufacturing, logistics, and generally in high-mix, low-volume sectors.

# *B. Learning Methods for Dual-Arm Systems*

The action space dimension for dual-arm systems is greater than that of single-arm systems. Considering the high dimensional action space, deep reinforcement learning algorithms are widely used in dual-arm system applications, e.g., space manipulators. Typically, space manipulators are designed with a free-floating base. A common task for space manipulators involves reaching and capturing an object in a state of free movement. Due to considerations of reachable space size and launch costs, the dual-arm system is a prevalent configuration for space manipulators. The DDPG algorithm is adopted to realize the complex constrained motion planning of a dual-arm space manipulator in the research [\[28\]. I](#page-12-27)n this scenario, self-collision avoidance and end-effector velocity are incorporated into the reward function. This approach enables the space manipulator to robustly capture a spinning target. The primary challenge in space manipulator operations is dealing with variations in the spinning speeds of space objects. Cao et al. integrate prior knowledge guidance into the policy of deep reinforcement learning [\[29\]. P](#page-12-28)rior knowledge guidance can assist in addressing end-effector pose constraints during manipulation and enhance planning accuracy.

<span id="page-2-12"></span><span id="page-2-11"></span><span id="page-2-10"></span><span id="page-2-9"></span><span id="page-2-8"></span><span id="page-2-7"></span><span id="page-2-6"></span><span id="page-2-1"></span>A recent popular application of dual-arm systems is in humanoid robots equipped with bi-manual manipulators. Wong et al. developed a motion planning method based on the SAC algorithm for a dual-arm robot [\[30\]. T](#page-12-29)he dual-arm robot's arms consist of 7 Degrees-of-Freedom (DOF). In this study, two agents with 7-DOF action spaces are simultaneously trained, and the motion of one arm is considered the environment for the other arm. Through a reward function designed to account for movement and avoidance components, the dual-arm robot can successfully execute self-collision avoidance maneuvers. However, there is a lack of Primarysecondary constraint relationship between two arms. Thus it is hard to manipulate a rigid body synchronously under closed-chain constraints. In the reaching phase of bi-manual manipulation, the SAC algorithm can facilitate the exploration of shorter and smoother paths compared to sampling-based methods [\[31\]. A](#page-12-30)dditionally, a technique known as Hindsight Experience Replay (HER) can enhance sample efficiency [\[32\].](#page-12-31) Learning methods, particularly DRL algorithms, when applied to real robots, may encounter a simulation-to-real gap in practical applications. Typically, the robot joints in the real world exhibit different friction and stiffness characteristics compared to simulations. This discrepancy can be mitigated by equipping tactile sensors on the end-effectors [\[33\]. I](#page-12-32)n addition to the DDPG and SAC algorithms, Jiang et al. employ the Proximal Policy Optimization (PPO) algorithm to train a dualarm agent. It is demonstrated that motion paths generated by DRL algorithms are shorter than those produced by scripted or manual methods [\[34\].](#page-12-33)

In addition to the challenge of high-dimensional dualarm system motion planning, task complexity is another significant concern. The form and requirements of some tasks are complex and difficult for dual-arm systems. For instance, in healthcare scenarios, a dual-arm robot is required to reach and assist a patient in motion [35]. Given the shape of the human body and the bed, collisions can potentially occur between the human and the robot. In this study, the DRL algorithm is employed to enable the robot to approach the patient in a complex environment. The reward function comprises four components: guidance, collision avoidance, obstacle negotiation, and time efficiency. With the effect of the proximal policy optimization, DRL algorithm is useful to deal with complex motion path.

<span id="page-3-3"></span>However, these researches primarily focus on reaching or catching processes. Due to the presence of closed-chain constraints, planning the transportation process becomes more challenging. High-dimensionality and closed-chain constraints make it difficult for learning methods in achieving efficient exploration. While the SAC algorithm exhibits strong exploration capabilities, primarily due to the entropy term [31], meeting strict constraints remains challenging. One efficient approach for addressing closed-chain constraints is the imitation learning method. Imitation learning can streamline the sample complexity by incorporating expert experience. Behavioral Cloning (BC) and Inverse Reinforcement Learning (IRL) are the primary approaches for implementing imitation learning methods [36]. Franzese et al. introduce the safe interactive movement primitive learning (SIMPLe) algorithm for transforming an initial bi-manual demonstration into robotic manipulation [37]. In this framework, a teacher separately demonstrates one of the single arms, while SIMPLe synchronizes the actions of both robot arms using kinesthetic feedback. The dual-arm system can be trained to pick up a box at varying local heights. Another practical example of imitation learning applied to dual-arm systems is robotic clothing assistance [38]. Human demonstrations can be transferred to the manipulation of compliant dual-arm robots using Dynamic Movement Primitives (DMP) and the Bayesian Gaussian Process Latent Variable Model (BGPLVM). The human demonstration serves as the initial step in imitation methods, implying that changes in goals or starting points may cause retraining in imitation learning. Imitation methods are well-suited for repetitive tasks. Nevertheless, it is still helpful to use imitation method at the very start exploration stage of the DRL algorithms.

# <span id="page-3-5"></span>III. PROBLEM FORMULATION

#### <span id="page-3-0"></span>A. Description of the Manipulation Task

This study involves the cooperative transportation of geometrically regular objects, such as chairs and wooden frames, using two UR5 manipulators, to move them from their initial pose to a specified location. Figure 3 illustrates a closed-chain constrained system comprising two arms and a grasped chair. In Figure 3, the solid area represents the initial state of the closed-chain system, and the translucent area signifies one of the target states.

<span id="page-3-2"></span><span id="page-3-1"></span>Fig. 3. The demonstration of the cooperative manipulation task.

It is assumed that  $s_{robot-i} \subseteq \mathbb{R}^{n_i}$  represents the current configuration of the i-th robot arm (i=1,2), where  $n_i$  represents the dimension of degrees of freedom. In the scene of Fig. 3, robot-1 represents the leader arm and robot-2 represents the follower arm. The symbol  $c_{tar}^{obj}$  represents the 6-dimensional target pose relative to the current object pose. Therefore, the composite state of the closed-chain constrained system can be described as  $\mathcal{S}=(s_{robot-1},s_{robot-2},c_{tar}^{obj})$ . Additionally,  $a_{robot-i}\subseteq\mathbb{R}^{n_i}$  represents the increment of each DOF position of the i-th robot. The composite action of the two robot arms is denoted as  $\mathcal{A}=(a_{robot-1},a_{robot-2})$ . At the t-th step, the composite state and action can be represented as  $\mathcal{S}_t$  and  $\mathcal{A}_t$ , respectively. Hence, a policy, denoted as  $\mathcal{A}_t \sim \pi(\cdot|\mathcal{S}_t)$ , constitutes the rules guiding manipulation from the initial state to the target state.

#### <span id="page-3-4"></span>B. Dual-Arm System Constraints and Reward

Typically, manipulation tasks entail constraints. A composite state s within the space S is subject to a constraint function represented as F(s) = 0. Consequently, the constrained state space for the manipulation of a dual-arm system can be formulated as

$$\mathcal{M} = \{ s \in \mathcal{S} | F(s) = 0 \},\tag{1}$$

where  $\mathcal{M}$  is a smooth sub-manifold of  $\mathbb{R}^{n_1+n_1+6}$ . Leveraging the constraints of a dual-arm system for shaping the reward structure of each state is an effective approach.

Within the context of reinforcement learning, the dual-arm system is expected to receive a reward or punishment (negative reward) following each action  $\mathcal{A}_t$ . The primary objective in cooperative manipulation is to derive an optimal policy capable of maximizing the cumulative total reward. During the manipulation process, each configuration of the dual-arm system must adhere to multiple constraints. Therefore, the reward function is formulated based on the constraints of the dual-arm system. Five constraints are introduced as follows to shape the reward formation.

1) Closed-Chain Constraint: Regardless of whether the dual-arm system can successfully maneuver the chair to the desired pose, the execution of every action must adhere to fundamental closed-chain constraints, encompassing both positional and angular distances between the two arm ends.

If  $s_{robot-1}$  and  $s_{robot-2}$  fail to satisfy the closed-chain constraints, the task execution for the current episode will be halted, and the dual-arm system will incur a large punishment. The reward for the closed-chain constraint, denoted as  $r_{cc}$ , is formulated as

$$r_{cc} = \begin{cases} 1.5, & \text{closed-chain constraint is satisfied,} \\ -200, & \text{closed-chain constraint is not satisfied.} \end{cases}$$
(2)

2) Target Pose Constraint: The relative configuration of the target pose with respect to the current chair pose, denoted as  $c_{tar}^{obj}$ , signifies the state of task implementation. Naturally, a smaller value of  $\|c_{tar}^{obj}\|_2$  corresponds to a larger distance reward. The manipulation task is considered completed when  $\|c_{tar}^{obj}\|_2$  falls below 0.05. The reward for the target pose constraint, denoted as  $r_{tar}$ , is formulated as

$$r_{tar} = \begin{cases} -100 \times \|c_{tar}^{obj}\|_{2}^{2} - 40, & \|c_{tar}^{obj}\|_{2} \ge 0.25, \\ -100 \times \|c_{tar}^{obj}\|_{2}^{2} - \ln(16) \\ \times \|c_{tar}^{obj}\|_{2}^{2} + 10^{-10}) - 40, & 0.05 \le \|c_{tar}^{obj}\|_{2} < 0.25, \\ -100 \times \|c_{tar}^{obj}\|_{2}^{2} - \ln(16) \\ \times \|c_{tar}^{obj}\|_{2}^{2} + 10^{-10}) + 1000, & \|c_{tar}^{obj}\|_{2} < 0.04. \end{cases}$$

For enhanced comprehension of the significance of  $\|c_{tar}^{obj}\|_2$ , the relative configuration is composed of the positional distance d (in meters) and angular distance  $\gamma$  (in radians), expressed as  $\|c_{tar}^{obj}\|_2 = d + \gamma$ .

3) Action Amplitude Constraint: To ensure the smoothness of the manipulation process, it is essential to maintain a low motion amplitude for both robot arms. In theory, a smaller value of  $\|\mathcal{A}_t\|_2$  corresponds to a greater action reward. The reward for the action amplitude constraint, denoted as  $r_{action}$ , is formulated as

$$r_{action} = -2 \times \|\mathcal{A}_t\|_2. \tag{4}$$

4) Collision Constraint: Throughout the manipulation process, if any collision occurs, the system will incur a large punishment, leading to an immediate termination of the ongoing manipulation episode. The reward for the collision constraint, denoted as  $r_{collide}$ , is formulated as

$$r_{collide} = \begin{cases} 0, & \text{no collision happened,} \\ -200, & \text{collision happened.} \end{cases}$$
 (5)

5) Time Constraint: The manipulation should be expedited. Consequently, the system incurs a negative time reward at each step until the object reaches the designated pose. The reward for the time constraint, denoted as  $r_{time}$ , is formulated as

$$r_{time} = \begin{cases} -0.1t, & t < 300, \\ -30, & t \ge 300. \end{cases}$$
 (6)

Following the execution of action  $A_t$  at t-th time step, the dual-arm system is awarded the total reward  $r_t$ , which is determined based on the prevailing constraints. The total reward  $r_t$  is determined as

$$r_t = r_{cc} + r_{tar} + r_{action} + r_{collide} + r_{time}. (7)$$

In this study, a DRL framework is modified to explore a path of composite state s in  $\mathbb{R}^{n_1+n_1+6}$ , which conforms to the requirement of the constraint function F. Notably, the proposed framework works directly in  $\mathbb{R}^{n_1+n_1+6}$  without the projection operation in sample-based methods.

#### IV. THE PROPOSED DRL FRAMEWORK

<span id="page-4-0"></span>Directly applying existing RL algorithms to dual-arm or multi-arm systems is challenging due to the stringent constraints. The states of dual-arm systems form a highdimensional space, within which agents explore to maximize their rewards. Demonstrably, two state-of-the-art DRL algorithms, Soft Actor-Critic (SAC) and Deep Deterministic Policy Gradient (DDPG), are not directly applicable to the mentioned dual-arm manipulation scenario. The maximum entropy component in SAC encourages agents to explore multiple routes to gather extensive information. However, due to the stringent constraints of the dual-arm system, it is time-consuming for the SAC algorithm to yield positive rewards, which result in huge amount of training episodes. Due to the deterministic policy employed in the actor component of the DDPG algorithm, a dual-arm system can discover effective policies. Nevertheless, this approach still entails significant costs in terms of exploration. In this study, the proposed DRL framework is augmented with a leader-follower component, ensuring that the dual-arm system consistently adheres to closed-chain constraints during behavioral exploration. Within the proposed framework, a single agent is responsible for policy exploration, with another agent (the follower) collaborating with it at each time step. Furthermore, both the leader arm and the follower arm operate in a centralized mode.

#### A. Leader-Follower Robot Actors

Within the proposed framework, the dual-arm system is driven by a leader-follower approach that complements the DRL component. The action executed by the agent in the dual-arm system is the angle increment on each joint.

It is assumed that robot - 1 is a leader and robot - 2 is a follower. Initially, the leader, robot - 1, takes action  $a_{robot-1}$ at t-th time step based on the composite state  $S_t$ , constituting the top-level policy. The follower takes the states of both robot arms and the leader's action  $a_{robot-1}$  as inputs, generating  $a_{robot-2}$  using a specifically designed inverse kinematics (IK) solver. In a word, the proposed framework operates hierarchically. The top-level policy is optimized using the actor-critic algorithm, while the low-level policy is facilitated by an efficient IK solver that adheres to closed-chain constraints. Specifically, the damped least squares method is included into the inverse kinematics solving process to prevent incorrect inverse kinematics solution in singular situations. The data flow for the follower can be represented as  $(s_{robot-1}, a_{robot-1}, s_{robot-2}) \rightarrow a_{robot-2}$ . The follower's policy comprises the forward kinematics (FK) of the leader robot and an IK solver to generate  $a_{robot-2}$ . The FK is employed to compute the target pose  $c_{tar}^{robot-2}$  for the follower arm based on  $(s_{robot-1}, a_{robot-1})$ . The IK solver component of the follower

<span id="page-5-5"></span>Fig. 4. An overview of the proposed DRL framework for closed-chain manipulation and the illustration of data flow in the method.

arm is described as follows. The forward kinematics of the follower arm can be represented as

$$x = f(q), \tag{8}$$

where x is the end-effector pose in  $R^6$ , q represents the joint angles of the follower arm of  $R^6$ . The target end-effector pose of  $c_{tar}^{robot-2}$  is denoted as  $x_d$ . The mean-square error between x and  $x_d$  can be represented as

$$MSE(\mathbf{q}) = \frac{1}{2} \|\mathbf{x}_d - \mathbf{x}\|_2^2 = \frac{1}{2} \|\mathbf{x}_d - f(\mathbf{q})\|_2^2.$$
 (9)

Denote  $e(q) = x_d - x$ , and then calculate the Taylor expansion of e(q) as

$$e(\mathbf{q}) \approx e(\mathbf{q}_0) + \nabla e(\mathbf{q}_0)(\mathbf{q} - \mathbf{q}_0),$$
 (10)

where  $q_0$  is the joint angles of the follower arm at time step t. Denote  $\Delta q = (q - q_0)$ . Let q represents the joint angles of the follower arm at time step t + 1, then  $a_{robot-2} = \Delta q$ . Considering the derivation of e(q),

$$\nabla e(\boldsymbol{q}_0) = \frac{d(\boldsymbol{x}_d - \boldsymbol{x})}{dq} \bigg|_{\boldsymbol{q} = \boldsymbol{q}_0} = -\frac{df(\boldsymbol{q})}{dq} \bigg|_{\boldsymbol{q} = \boldsymbol{q}_0} = -J(\boldsymbol{q}_0). \tag{11}$$

Eq. (11) denotes that  $\nabla e(\mathbf{q}_0)$  is the negative Jacobian matrix of  $\mathbf{q}_0$ . The objective of the inverse kinematics solver is to minimize  $MSE(\mathbf{q})$ . Therefore, Combined with Eq. (9), Eq. (10) and Eq. (11), there is

$$\Delta \mathbf{q} = \underset{\Delta \mathbf{q}}{\arg \min} MSE(\mathbf{q})$$

$$= \underset{\Delta \mathbf{q}}{\arg \min} \frac{1}{2} \| e(\mathbf{q}_0) - J(\mathbf{q}_0) \Delta \mathbf{q} \|_2^2$$

$$\approx \underset{\Delta \mathbf{q}}{\arg \min} \frac{1}{2} [e(\mathbf{q}_0) - J(\mathbf{q}_0) \Delta \mathbf{q}]^T [e(\mathbf{q}_0) - J(\mathbf{q}_0) \Delta \mathbf{q}]$$

$$= [J(\mathbf{q}_0)^T J(\mathbf{q}_0)]^{-1} J(\mathbf{q}_0)^T e(\mathbf{q}_0). \tag{12}$$

Referring to Eq. (12), when the joint angles of the follower arm at  $s_{robot-2}$  are denoted as  $q_0$ , there is  $a_{robot-2} = \Delta q = [J(q_0)^T J(q_0)]^{-1} J(q_0)^T e(q_0)$ . However, in the case of the follower arm being in a singular configuration, it may fail to compute the correct  $\Delta q$ . Therefore, the Levenberg-Marquardt (LM) algorithm, also referred to as Damped Least Squares (DLS), is utilized to address singular situations. To address such issues, a damping term is introduced in the vicinity of the singular point. This term restricts the amplitude of joint movement and enhances the stability of the solution [39]. Utilizing the DLS method,  $a_{robot-2}$  is solved iteratively as

<span id="page-5-6"></span><span id="page-5-4"></span><span id="page-5-2"></span><span id="page-5-1"></span>
$$\Delta \mathbf{q} = \arg\min_{\Delta \mathbf{q}} \frac{1}{2} \| e(\mathbf{q}_0) - J(\mathbf{q}_0) \Delta \mathbf{q} \|_2^2 + \frac{1}{2} \| \lambda \Delta \mathbf{q} \|_2^2$$
$$= \left[ J(\mathbf{q}_0)^T J(\mathbf{q}_0) + \lambda^2 I \right]^{-1} J(\mathbf{q}_0)^T e(\mathbf{q}_0), \tag{13}$$

where  $\lambda$  is the damping factor of DLS solver.

<span id="page-5-0"></span>Using Eq. (13), the low-level IK solver policy can address singular point scenarios. The process of the follower IK solver is detailed in Alg. 1, with hyperparameters  $\lambda$  (the damping factor) and  $\kappa$  (the iterative ratio). Additionally,  $n_{iter}$  represents the maximum number of iterations within a time step of the leader policy.

<span id="page-5-3"></span>An overview of the proposed DRL framework for the dual-arm system manipulation is illustrated in Fig. 4. The system processes the composite state  $\mathcal{S}$  and generates a composite action  $\mathcal{A}$  for the leader-follower robot actors. The manipulation environment executes the action  $\mathcal{A}$ , providing numerical observations and corresponding rewards in one step. The experience buffer stores data for subsequent optimization. In further detail, during the operation of the dual-arm system, the follower actor relies on a high-performance IK solver to keep pace with the leader actor. Subsequently, the top-level

#### Algorithm 1 IK Solver in Low-Level Policy

<span id="page-6-0"></span>**input**: state of the leader arm  $s_{robot-1}$ , action of the leader arm  $a_{robot-1}$  and state of the follower arm  $s_{robot-2}$  at time step t of the leader policy

output: action of the follower arm

- 1 confirm the damping factor  $\lambda$  and iterative ratio  $\kappa$
- 2 calculate the target pose of the follower arm end-effector by  $s_{robot-1}$  and  $a_{robot-1}$
- 3 calculate the joint angles  $q_0$  of the follower arm at the time step t by  $s_{robot-2}$

```
4 repeat

5 | calculate the error e(q_0)

6 | if e(q_0) satisfies the accuracy requirement then

7 | break

8 | end

9 | \Delta q = [J(q_0)^T J(q_0) + \lambda^2 I]^{-1} J(q_0)^T e(q_0)

10 | q = q_0 + \Delta q

11 | q_0 \leftarrow q
```

- 12 **until** maximum iterations  $n_{iter}$ ;
- 13 **return** action of the follower arm  $a_{robot-2}$ .

policy, which comprises the actor-critic algorithm, begins optimization as the leader explores the environment and collects experience data with the follower.

Despite the follower's passive action generation, its outputs wield substantial influence over the entire system's rewards, thereby impacting the optimization of the leader policy and the critic component. Conversely, the optimized leader policy dictates the follower's capabilities. This is due to the policy gradient part of the DRL algorithm consistently striving to maximize cumulative rewards, and as a result, the leader policy is optimized toward scenarios where the follower policy performs well.

#### B. Policy Optimization Based on Actor-Critic DRL Algorithm

The leader policy is optimized using the actor-critic algorithm. Throughout the manipulation task, the dual-arm system consistently receives a total reward upon each action. The objective of the task is to optimize the policy to achieve the maximum expected return. The standard expected return is the total rewards accumulated by the dual-arm system, discounted based on the time it takes. The standard return under a selected policy  $\pi$  can be formulated as

$$J_{std}(\pi) = \mathbb{E}_{\pi} \left[ \sum_{\mathcal{S}_{t+1} \sim P} \left[ \sum_{t=0}^{T(\infty)} \gamma^t R(\mathcal{S}_t, \mathcal{A}_t, \mathcal{S}_{t+1}) \right], \tag{14} \right]$$

where  $\mathbb{E}_{\pi}$  represents the expectation subjects to the policy  $\pi$ , P represents the state transitions distribution denoted by  $S_{t+1} \sim P(\cdot|S_t, A_t)$ , and  $\gamma \in (0, 1)$  is the discount factor. Although the SAC policy only outputs  $a_{robot-1}$ , it also exerts an impact on  $a_{robot-2}$ . Therefore, the reward calculation of the leader still relies on the composite action A.

Haarnoja et al. demonstrated that enhancing the standard RL return with an entropy maximization term substantially

enhances policy exploration and robustness [40]. This modification transforms the manipulation task into searching for an optimal policy

<span id="page-6-2"></span><span id="page-6-1"></span>
$$\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T(\infty)} \gamma^t \left[ r_t + \alpha \mathcal{H}(\pi(\cdot|\mathcal{S}_t)) \right] \right], \quad (15)$$

where  $\mathcal{H}$  represents the entropy of a random variable,  $r_t = R(\mathcal{S}_t, \mathcal{A}_t, \mathcal{S}_{t+1})$ . Suppose x is a random variable subject to Y. The entropy of the random variable is computed by  $\mathcal{H}(Y) = \mathbb{E}_{x \sim Y}[-\ln Y(x)]$ .

The temperature parameter  $\alpha$  governs the trade-off between the entropy term and the reward  $r_t$ . Leveraging this objective, the soft actor-critic (SAC) DRL algorithm was introduced for continuous state-action spaces, offering efficient sample learning and stability [21].

According to the objective illustrated in Eq. (15), the value function of composite state  $S_t$  can be represented as

$$V^{\pi}(\mathcal{S}_{t}) = \mathbb{E}_{\pi} \sum_{\mathcal{S}_{t+k+1} \sim P} \left[ \sum_{k=0}^{T(\infty)} \gamma^{k} \left[ r_{t+k} + \alpha \mathcal{H}(\pi(\cdot|\mathcal{S}_{t+k})) \right] \right]$$
(16)

Hence the action-value function of the state-action pair can be represented as

$$Q^{\pi}(\mathcal{S}_{t}, \mathcal{A}_{t}) = \underset{\mathcal{S}_{t+k} \sim P}{\mathbb{E}_{\pi}} \left[ r_{t} + \sum_{k=1}^{T(\infty)} \gamma^{k} \left[ r_{t+k} + \alpha \mathcal{H}(\pi(\cdot|\mathcal{S}_{t+k})) \right] \right].$$
(17)

Let  $a_t$  represent  $a_{robot-1}$  at the t-th step. The Bellman equation for  $Q^{\pi}$  can be given by

$$Q^{\pi}(S_t, a_t) = \mathbb{E}_{S_t \to P} [r_t + \gamma V^{\pi}(S_{t+1})].$$
 (18)

Additionally,  $V^{\pi}$  and  $Q^{\pi}$  can be connected by

<span id="page-6-3"></span>
$$V^{\pi}(\mathcal{S}_t) = \mathbb{E}_{\mathbb{C}[Q^{\pi}(\mathcal{S}_t, a_t) - \alpha \ln \pi(a_t | \mathcal{S}_t)]}.$$
 (19)

Achieving the optimal policy necessitates the utilization of multiple neural networks. In this study, the double Q-functions strategy [41] is employed to mitigate the overestimate problem. Two Q-functions  $Q_{\phi_1}(\mathcal{S}_t, a_t)$  and  $Q_{\phi_2}(\mathcal{S}_t, a_t)$  are parameterized with  $\phi_1$  and  $\phi_2$ , respectively. The state value function  $V_{\psi}(\mathcal{S}_t)$  is parameterized with  $\psi$ . The policy  $\pi_{\theta}(\cdot|\mathcal{S}_t)$  is parameterized with  $\theta$ .

1) Train  $Q_{\phi_i}$ : The two Q-functions are trained independently by mean-squared Bellman error (MSBE) minimization. The loss functions can be then represented by

$$L(\phi_{i}, \mathcal{D}) = \mathbb{E}\Big[ (Q_{\phi_{i}}(S_{t}, a_{t}) - (r_{t} + \gamma(1 - d_{t+1})V_{\psi_{tab}}(S_{t+1})))^{2} \Big]$$
s.t.  $(S_{t}, a_{t}, r_{t}, S_{t+1}, d_{t+1}) \sim \mathcal{D}, i = 1, 2$  (20)

where  $\mathcal{D}$  represents the replay buffer, and  $d_{t+1}$  denotes the flag to indicate whether  $\mathcal{S}_{t+1}$  has reached the target state.

2) Train  $V_{\psi}$ : The state value function is trained through minimization of the mean-squared error. The loss function takes the minimum Q-value between  $Q_{\phi_1}$  and  $Q_{\phi_2}$  as the target Q-value, with further details available in [41]. The loss function is given by

$$L(\psi, \mathcal{D}) = \underset{\tilde{a}_{\theta} \sim \pi_{\theta}}{\mathbb{E}} \left[ \left( V_{\psi}(\mathcal{S}_{t}) - \left( \min_{i=1,2} Q_{\phi_{i}}(\mathcal{S}_{t}, \tilde{a}_{\theta}) \right) - \alpha \ln \pi_{\theta}(\tilde{a}_{\theta} | \mathcal{S}_{t}) \right)^{2} \right].$$
(21)

Importantly, to ensure that  $V_{\psi}$  is trained under the current policy, the action  $\tilde{a}_{\theta}$  used in Eq. (21) is sampled from distribution  $\pi_{\theta}(\cdot|\mathcal{S}_t)$ , not from replay buffer.

3) Train  $\pi_{\theta}$ : The policy network  $\theta$  initially outputs the mean  $\mu_{\theta}$  and the standard deviation  $\sigma_{\theta}$  of the Gaussian distribution. Subsequently, to confine the amplitudes of actions, an invertible squashing function (tanh) is applied to Gaussian samples. Hence the actions sampled according to  $\pi_{\theta}$  are obtained by

$$\tilde{\mathcal{A}}_{\theta}(\mathcal{S}_t, \xi) = \tanh(\mu_{\theta}(\mathcal{S}_t) + \sigma_{\theta}(\mathcal{S}_t) \odot \xi), \xi \sim \mathcal{N}(0, I), \quad (22)$$

where  $\odot$  represents the Hadamard product. The policy  $\pi_{\theta}$  aims to maximize  $V^{\pi}(\mathcal{S}_t)$ . The first Q approximator  $Q_{\phi_1}$  is utilized in the policy loss. The loss function can be expressed as

$$L(\theta, \mathcal{D}) = \underset{\xi \sim \mathcal{N}}{\mathbb{E}} \left[ \alpha \ln \pi_{\theta} \left( \tilde{a}_{\theta}(\mathcal{S}_{t}, \xi) | \mathcal{S}_{t} \right) - Q_{\phi} \left( \mathcal{S}_{t}, \tilde{a}_{\theta}(\mathcal{S}_{t}, \xi) \right) \right].$$
(23)

Similarly, the action  $\tilde{a}_{\theta}$  employed in Eq. (23) is also sampled from the current policy  $\pi_{\theta}(\cdot|\mathcal{S}_t)$ . After generating the policy  $\pi_{\theta}(\cdot|\mathcal{S}_t)$  as depicted in Alg. 2, the dual-arm system can perform the manipulation task from its original configuration.

As per Eq. (22), the output action is confined within (-1,1). In practical applications, the maximum allowable action amplitude is constrained to  $\pi/36$  radians, a sufficiently small value to ensure smooth robot motion. Consequently, the effective action range for the leader arm is  $(-\pi/36, \pi/36)$  radians.

#### V. EXPERIMENTS

<span id="page-7-0"></span>To validate the proposed framework's effectiveness, closedchain manipulation experiments are carried out using a dual-arm system in the V-REP simulation environment [42], and subsequently applied to practical robot arms.

# A. Experiment I: Manipulate an Object to a Randomly Selected Pose Within a Predefined Range

In order to verify the task adaptability of the dual-arm system within the proposed framework, a chair is manipulated to a randomly selected pose within a spherical workspace with a 0.4m diameter. For this experiment, the coordinate system of the dual-arm system is fixed at the center of the robot arms platform. The target position is randomly selected from the spherical workspace. The center of the spherical workspace is located at [0, 0, 1.0]m. Figure 5 illustrates two distinct manipulation processes with varying initial and target configurations using the trained policy.

# Algorithm 2 The Execution of the Proposed Method

<span id="page-7-3"></span>**input**: initial leader actor policy parameters  $\theta$ ,

```
O-function parameters \phi_1, \phi_2, V-function
                  parameters \psi, replay buffer \mathcal{D}.
     output: leader actor policy and follower actor policy.
 1 repeat
           reset environment and observe state S
           for k in range(steps_per_episode) do
                 a_{robot-1} \leftarrow \mathcal{D}.size < start\_size ? rand():
                 a_{robot-2} \leftarrow DLS(s_{robot-1}, a_{robot-1}, s_{robot-2})
                 execute A \leftarrow (a_{robot-1}, a_{robot-2}) in
                   environment,
                 observe next state S', reward r, and flag d,
                 store(\mathcal{S}, \mathcal{A}, r, \mathcal{S}', d) in \mathcal{D},
                 S \leftarrow S'
                 if \mathcal{D}.size > start\ size then
                       sample batch B \leftarrow \{(S, a_{robot-1}, r, S', d)\}
                         from \mathcal{D}.
                       compute target Q-value and target V-value,
12
                           y_q \leftarrow r + \gamma (1 - d) V_{\psi_{tar}}(\mathcal{S}')
                           y_v \leftarrow \min Q_{\phi_i}(\mathcal{S}, \tilde{a}_{\theta}) - \alpha \ln \pi_{\theta}(\tilde{a}_{\theta}|\mathcal{S})
                       update Q-functions using
                       \nabla_{\phi_i} \frac{1}{|B|} \sum (Q_{\phi_i}(\mathcal{S}, a_{robot-1}) - y_q)^2, \ i =
                       update V-function using
14
                       \nabla_{\psi \frac{1}{|B|}} \sum (V_{\psi}(S) - y_v)^2\nif mod(k, policy\_delay) == 0 then
15
                             update leader actor policy using
                             \nabla_{\theta} \frac{1}{|B|} \sum \left( \alpha \ln \pi_{\theta} \left( \tilde{a}_{\theta} | \mathcal{S} \right) - \tilde{Q}_{\phi_1} \left( \mathcal{S}, \tilde{a}_{\theta} \right) \right) update target value-function with
17
                                   \psi_{\text{targ}} \leftarrow \rho \psi_{\text{targ}} + (1 - \rho) \psi
18
                       end
                 end
                 if d or collision then
20
                      break
21
22
                 end
           end
24 until convergence or maximum iterations;
```

<span id="page-7-6"></span><span id="page-7-5"></span><span id="page-7-2"></span>25 **return** leader actor policy  $\pi_{\theta}$ .

Fig. 5. The manipulation processes from different initial configurations.

Firstly, a functional verifiability experiment is conducted where a chair is manipulated from a fixed initial state to a

<span id="page-8-2"></span>Fig. 6. Training results of the functional verifiability experiment. (a). Curves of the dual-arm system's average reward per episode (p. f. is the abbreviation of "proposed framework" in the figure). Each solid line represents the mean of measurements over four experiments with different random seeds, while the light areas represent the 0.95 confidence interval. (b). Curves that track the run steps per episode. (c). Success rate of the task implementation.

<span id="page-8-3"></span>Fig. 7. Application results with different maximal action amplitudes. (a). The variations of the distance between the current chair pose and target pose when applying the policy with different maximal action amplitudes, without retraining. (b). The variations of the leader robot joint angles. (c). The variations of the follower robot joint angles.

<span id="page-8-0"></span>Fig. 8. The illustration of the manipulation process of functional verifiability experiment.

<span id="page-8-4"></span>Fig. 9. Snapshots of practical robot arms performing manipulation task (corresponding to the Fig. 8).

# TABLE I PARAMETERS OF THE PROPOSED METHOD

<span id="page-8-1"></span>

| parameters                                       | value                |
|--------------------------------------------------|----------------------|
| number of hidden layers of leader policy network | 5                    |
| number of hidden layers of critic networks       | 5                    |
| learning rate of leader policy network           | $1.5 \times 10^{-4}$ |
| learning rate of critic networks                 | $1 \times 10^{-4}$   |
| replay buffer size                               | $2 \times 10^{5}$    |
| discount factor $(\gamma)$                       | 0.99                 |
| temperature factor $(\alpha)$                    | 0.2                  |
| steps_per_episode in Alg. 2                      | 200                  |
| start_size in Alg. 2                             | $3 \times 10^{4}$    |
| policy_delay in Alg. 2                           | 20                   |

specific target state. The initial state of environment is shown as the non-transparent part in Fig. 3, where the initial position

and Euler attitude angle of the chair center are [0.55, 0, 0.4]m and [0, 0, 0]rad. The chair is required to be manipulated to the pose of [0, 0, 1.1]m and [0, 0, 0]rad. Whether or not the task is done is determined by  $c_{tar}^{obj}$ . Numerically, when  $\|c_{tar}^{obj}\|_2 < 0.05$ , the task is completed. Several parameters employed in the proposed framework are detailed in TABLE I.

In the simulated environment of the functional verifiability experiment, the training results of the proposed framework and the pure SAC algorithm are separately presented in Fig. 6. Specifically, the experiment involving the pure SAC algorithm takes the composite state as input and directly produces all 12-dimensional actions for the dual arms. In comparison to

<span id="page-9-0"></span><span id="page-9-1"></span>Fig. 10. Training results of the task-adaptive experiment. (a). Curves of the dual-arm system's average reward per episode. (b). Curves that track the run steps per episode. (c). Success rate of the task implementation, which is calculated by the task performing success rate of the next 100 episodes including the current episode.

TABLE II

THE RESULT OF TESTS OF 1000 Random Target Pose in the Task-Adaptive Experiment

| the proposed method and other strategies |              | successful episodes                          |          |         | all episodes               |        |                   |        |          |
|------------------------------------------|--------------|----------------------------------------------|----------|---------|----------------------------|--------|-------------------|--------|----------|
|                                          | success rate | average reward per episode steps per episode |          | episode | average reward per episode |        | steps per episode |        |          |
|                                          |              | $\mu$                                        | $\sigma$ | $\mu$   | $\sigma$                   | $\mu$  | $\sigma$          | $\mu$  | $\sigma$ |
| SAC with p. f.                           | 92.1%        | -19.38                                       | 8.54     | 29.04   | 4.07                       | -21.72 | 17.16             | 36.89  | 36.15    |
| TD3 with p. f.                           | 63.6%        | -24.28                                       | 9.80     | 25.39   | 8.16                       | -31.81 | 30.16             | 77.38  | 80.02    |
| DDPG with p. f.                          | 32.2%        | -28.19                                       | 8.68     | 34.98   | 18.46                      | -38.28 | 33.97             | 135.71 | 82.27    |

the embedded SAC algorithm within the proposed framework, additional experiments are conducted with frameworks embedded with DDPG and TD3. It is evident that the pure SAC algorithm does not consistently converge after a significant number of episodes, whereas the policy of the proposed framework reliably converges to an ideal state with relatively high rewards. Furthermore, when embedded with SAC, the proposed framework achieves a higher success rate compared to scenarios where it is embedded with the other two actorcritic algorithms, attributed to SAC's superior performance in high-dimensional problems.

While the maximum action amplitude during the training process is limited to  $\pi/36$  rad as defined in Eq. (22), it is noteworthy that the trained policy is adaptable to varying maximal amplitudes. As depicted in Fig. 7(a), the policy trained using the proposed framework is capable of operating with different maximal amplitudes without the need for retraining. The distance between the current chair pose and the target pose is quantified using  $\|c_{\text{tar}}^{obj}\|_2$ . A reduced action amplitude contributes to a smoother and more stable manipulation process. The outcomes presented in Fig. 7(b) and 7(c) demonstrate the application of the trained policy to the manipulation task with a maximal action amplitude of  $\pi/180$  rad. The variations in all joint angles exhibit a high degree of smoothness and lack abrupt fluctuations.

Figure 8 illustrates the manipulation process using the policy trained by the proposed framework. The transparent chairs represent the chair's states during the manipulation process, while the non-transparent chair at the top represents the state of the chair upon task completion. The solid lines represent the trajectories of the end grippers of both robot arms. Figure 9 displays snapshots captured when applying

<span id="page-9-2"></span>Fig. 11. Snapshots of practical robot arms performing manipulation tasks from different initial configurations. (a) Snapshots corresponding to the manipulation process in the Fig. 5(a). (b) Snapshots corresponding to the manipulation process in the Fig. 5(b).

the trained policy with practical robot arms. In practical applications, the dual-arm system follows the path in joint space generated by the trained policy. The manipulation time is approximately 100 seconds, indicative of suboptimal efficiency. This inefficiency arises from the time required for program synchronization during practical applications. However, increasing the maximum action amplitude, as depicted in Fig. 7(a), can diminish the number of operation steps and consequently decrease manipulation time costs.

Secondly, a task-adaptive experiment is conducted. The target pose is randomly set within the spherical workspace when the manipulation task begins. The leader policy does not need to be retrained even when the target pose varies. Notably, the training process in the task-adaptive experiment differs from that in the functional verifiability experiment. In the task-adaptive experiment, the target pose is randomly sampled during the environment reset phase (step 3 in Alg. 2). The training results for the task-adaptive experiment are depicted

<span id="page-10-3"></span>Fig. 12. Application results of wooden frame manipulation. (a). The variations of the relative configuration, comprising of position distance d and angle distance  $\gamma$ . The relative configuration is denoted by  $\|c_{tar}^{obj}\|_2 = d + \gamma$ . (b). The variations of the leader robot joint angles. (c). The variations of the follower robot joint angles.

in Fig. 10. Achieving convergence in the task-adaptive experiment requires more training episodes compared to the functional verifiability experiment.

Furthermore, to verify the task-adaptive capability of the trained policies, one thousand target points are randomly sampled from the spherical space for testing. The results of success rate, mean  $(\mu)$  and standard deviation  $(\sigma)$  of the average reward per episode and number of steps per episode are illustrated in TABLE II. The proposed framework embedded with SAC has a success rate of over 90%. Given the possibility of singularities with no objectively feasible paths, this level of success rate is satisfactory. Particularly, the high standard deviations  $(\sigma)$  of reward in successful episodes can be attributed to the stochastic selection of target pose, signifying the task-adaptive capability of the proposed framework. Similar to Experiment I, success rate of frameworks embedded with TD3 and DDPG are lower than the situation embedded with SAC.

Despite the fixed initial configuration of the dual-arm system during training, the trained policy adeptly manages various initial configurations without requiring retraining. This can be attributed to the occurrence of random initial states within the Markov process midway through the training phase, during which policies specifically addressing these states have been trained. The processes of practical robot arms performing such tasks are shown in Fig. 11.

# B. Experiment II: Manipulate a Delicate Structural Object Within a Narrow Space

Building upon the experimental approach of Experiment I, we proceed to evaluate dual-arm system's manipulation capabilities on a wooden frame which is a delicate structure. The manipulation process of Experiment II is shown in the Fig. 13. The non-translucent yellow wooden frame above the desk denotes the start pose of object, while the translucent green part denotes the target object pose.

There is an another wooden frame beneath the desk in Fig. 13, identical in shape to the manipulated wooden frame. Therefore, the objective of the manipulation is to relocate a wooden frame to a position above the other frame, aligning all four corners for assembly. The initial and final states for Experiment II are provided in TABLE III for reference. The

<span id="page-10-0"></span>Fig. 13. The illustration of the manipulation on a wooden frame.

<span id="page-10-1"></span>TABLE III
INITIAL STATES AND FINAL STATES OF EXPERIMENT II

| parameters                                                             | value                                                                                 |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| start joint angles of the leader arm                                   | $ \begin{array}{c} [-2.472, -0.402, -2.017, \\ 2.419, 2.476, 0.004] rad \end{array} $ |
| start joint angles of the follower arm                                 | $ \begin{bmatrix} -0.146, -0.963, -0.932, \\ 1.895, 0.139, -0.006 \end{bmatrix} rad $ |
| start position of the wooden frame target position of the wooden frame | $ \begin{bmatrix} -0.550, 0, 0.470 \end{bmatrix} m $ [0.500, 0, 0.600] m              |

<span id="page-10-2"></span>Fig. 14. Manipulate the wooden frame pass through a narrow space.

task is considered complete when  $\|c_{tar}^{obj}\|_2 < 0.05$ . The training procedure for Experiment II replicates that of Experiment I.

In contrast to the manipulation process in Experiment I, the object's trajectory is extended, and the passage space for the object is significantly narrower, as depicted in Figure 14.

<span id="page-11-1"></span><span id="page-11-2"></span>Fig. 15. Position-based force controller for the follower arm.

Fig. 16. Force on the end-effector of the follower arm.

<span id="page-11-3"></span>Fig. 17. Snapshots of practical robot arms manipulating the wooden frame pass through a narrow space (corresponding to the Fig. [14\)](#page-10-2).

Additionally, the joint angle trajectories generated by the policy trained in the simulated environment are validated on the practical robot arms. It is worth noting that the wooden frame utilized in the experiment is delicate, making it prone to falling apart if subjected to excessive force from the robot arm's end-effector. During the experiment, the leader arm operates under pure position control, while the follower arm employs a position-based force control approach, as illustrated in Fig. [15.](#page-11-1)

In the Fig. [15,](#page-11-1) *x <sup>f</sup>* <sup>−</sup>*ctrl* represents the target position of the follower arm's end-effector in Cartesian space, *q <sup>f</sup>* <sup>−</sup>*ctrl* is the joint position corresponding with *x <sup>f</sup>* <sup>−</sup>*ctrl* , the value of *f <sup>d</sup>* signifies the threshold force value at the follower arm's end-effector, set at 40*N* in the experiment, *k<sup>p</sup>* and *k<sup>d</sup>* represent the proportional and differential coefficients. The force applied to the follower arm's end-effector is measured as depicted in Fig. [16.](#page-11-2) This force is determined through real-time measurements of current and torque in each joint of the robot arm, followed by calculations based on the robot arm's dynamic equation. To protect the delicate structural object in the experiment, the controller in Fig. [15](#page-11-1) activates when the total force applied to the end-effector exceeds 40*N*.

As illustrated in Fig. [12\(a\),](#page-10-3) the joint angle trajectories derived from the proposed framework effectively guide the wooden frame through a narrow space to reach the target pose. Both the leader and follower arms demonstrate smooth transitions in the Joint space, as shown in Fig. [12\(b\)](#page-10-3) and [12\(c\).](#page-10-3) Figure [17](#page-11-3) displays snapshots capturing the application of manipulating a wooden frame using practical robot arms.

# VI. CONCLUSION

<span id="page-11-0"></span>In this study, a DRL framework based on a hierarchical leader-follower policy is proposed to dri ve the manipulation of a dual-arm system under the closed-chain constraints. The primary objective of the framework is to manage continuous states in high-dimensional space, optimizing policies for selecting actions in complex manipulation tasks to achieve high-level rewards while adhering to closed-chain constraints. The proposed framework is composed of an actor-critic structure, especially the actor part of the proposed framework is driven in a leader-follower mode. The leader part consists of a policy trained from the DRL algorithm and works on the leader arm. While the follower part consists of an IK solver based on DLS and works on the follower arm. This paper conducts two experiments involving manipulation tasks of varying complexities using both the proposed framework and pure DRL algorithms. The results indicate that the pure DRL algorithm is not directly suitable for closed-chain manipulation tasks. In contrast, the proposed framework demonstrates task adaptability in dual-arm cooperative transportation. It performs effectively in experiments involving the manipulation of an object to a randomly chosen pose within a specific range and the manipulation of a delicate structural object in a narrow space. Different from traditional sampling-based methods, a notable feature of the proposed framework is its ability to explore joint configurations without factoring in the constrained manifold and projection operation. Moreover, whether the maximum action amplitude or the initial configurations of the manipulation task are altered, the dual-arm system can still perform tasks without requiring retraining, thereby highlighting the adaptability and scalability of the framework.

In Experiment I, the dual-arm system, operating within the proposed framework, successfully manipulates an object to a random pose within a spherical workspace, resulting in a substantial standard deviation in the average episode reward. The big standard deviation is caused by the variety of manipulation process from random start states to random target states. In the Experiment II, the dual-arm system under the proposed framework manipulates a delicate structural object within a narrow space while ensuring that the force on the follower arm's end-effector remains below 40*N*, confirming the smoothness of the trajectory generated by the proposed framework. Subsequent research will delve into the planning of grasping positions by the dual-arm system, expanding the scope to enhance task adaptability for irregular objects.

# ACKNOWLEDGMENT

All these supports are highly appreciated. The authors thank Dr. Wenrui Wang for the text correction of this paper.

# REFERENCES

- <span id="page-12-0"></span>[\[1\] C](#page-0-0). Yuan, W. Zhang, G. Liu, X. Pan, and X. Liu, "A heuristic rapidlyexploring random trees method for manipulator motion planning," *IEEE Access*, vol. 8, pp. 900–910, 2020.
- <span id="page-12-1"></span>[\[2\] M](#page-0-1). S. Phoon, P. S. Schmitt, and G. V. Wichert, "Constraint-based task specification and trajectory optimization for sequential manipulation," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Oct. 2022, pp. 197–202.
- <span id="page-12-2"></span>[\[3\] I.](#page-0-2) A. Sucan, M. Moll, and L. E. Kavraki, "The open motion planning library," *IEEE Robot. Autom. Mag.*, vol. 19, no. 4, pp. 72–82, Dec. 2012.
- <span id="page-12-3"></span>[\[4\] J](#page-0-3). H. Yakey, S. M. LaValle, and L. E. Kavraki, "Randomized path planning for linkages with closed kinematic chains," *IEEE Trans. Robot. Autom.*, vol. 17, no. 6, pp. 951–958, Dec. 2001.
- <span id="page-12-4"></span>[\[5\] J](#page-0-4). C. Trinkle and R. J. Milgram, "Complete path planning for closed kinematic chains with spherical joints," *Int. J. Robot. Res.*, vol. 21, no. 9, pp. 773–789, Sep. 2002.
- <span id="page-12-5"></span>[\[6\] Z](#page-0-5). Xian, P. Lertkultanon, and Q.-C. Pham, "Closed-chain manipulation of large objects by multi-arm robotic systems," *IEEE Robot. Autom. Lett.*, vol. 2, no. 4, pp. 1832–1839, Oct. 2017.
- <span id="page-12-6"></span>[\[7\] B](#page-1-2). Kim, T. T. Um, C. Suh, and F. C. Park, "Tangent bundle RRT: A randomized algorithm for constrained motion planning," *Robotica*, vol. 34, no. 1, pp. 202–225, Jan. 2016.
- <span id="page-12-7"></span>[\[8\] F](#page-1-3). Lamiraux and J. Mirabel, "Prehensile manipulation planning: Modeling, algorithms and implementation," *IEEE Trans. Robot.*, vol. 38, no. 4, pp. 2370–2388, Aug. 2022.
- <span id="page-12-8"></span>[\[9\] K](#page-1-4). Jang, J. Baek, S. Park, and J. Park, "Motion planning for closed-chain constraints based on probabilistic roadmap with improved connectivity," *IEEE/ASME Trans. Mechatronics*, vol. 27, no. 4, pp. 2035–2043, Aug. 2022.
- <span id="page-12-9"></span>[\[10\]](#page-1-5) Z. Kingston, M. Moll, and L. E. Kavraki, "Exploring implicit spaces for constrained sampling-based planning," *Int. J. Robot. Res.*, vol. 38, nos. 10–11, pp. 1151–1178, Sep. 2019.
- <span id="page-12-10"></span>[\[11\]](#page-1-6) Y. Wu, Y. Fu, and S. Wang, "Global motion planning and redundancy resolution for large objects manipulation by dual redundant robots with closed kinematics," *Robotica*, vol. 40, no. 4, pp. 1125–1150, Apr. 2022.
- <span id="page-12-11"></span>[\[12\]](#page-1-7) R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*. London, U.K.: MIT Press, 2018.
- <span id="page-12-12"></span>[\[13\]](#page-1-8) J. Kober, J. A. Bagnell, and J. Peters, "Reinforcement learning in robotics: A survey," *Int. J. Robot. Res.*, vol. 32, no. 11, pp. 1238–1274, Sep. 2013.
- <span id="page-12-13"></span>[\[14\]](#page-1-9) V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015.
- <span id="page-12-14"></span>[\[15\]](#page-1-9) A. Goyal et al., "Retrieval-augmented reinforcement learning," in *Proc. Mach. Learn. Res. (ICML)*, Jul. 2022, pp. 7740–7765.
- <span id="page-12-15"></span>[\[16\]](#page-1-10) C. Yan, Q. Zhang, Z. Liu, X. Wang, and B. Liang, "Control of freefloating space robots to capture targets using soft Q-learning," in *Proc. IEEE Int. Conf. Robot. Biomimetics (ROBIO)*, Dec. 2018, pp. 654–660.
- <span id="page-12-16"></span>[\[17\]](#page-1-11) W. Yuan, K. Hang, D. Kragic, M. Y. Wang, and J. A. Stork, "End-toend nonprehensile rearrangement with deep reinforcement learning and simulation-to-reality transfer," *Robot. Auto. Syst.*, vol. 119, pp. 119–134, Sep. 2019.
- <span id="page-12-17"></span>[\[18\]](#page-1-12) N. Sugimoto and J. Morimoto, "Trajectory-model-based reinforcement learning: Application to bimanual humanoid motor learning with a closed-chain constraint," in *Proc. 13th IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, Oct. 2013, pp. 429–434.
- <span id="page-12-18"></span>[\[19\]](#page-1-13) P. Xu et al., "Reinforcement learning compensated coordination control of multiple mobile manipulators for tight cooperation," *Eng. Appl. Artif. Intell.*, vol. 123, Aug. 2023, Art. no. 106281.
- <span id="page-12-19"></span>[\[20\]](#page-1-14) T. Lillicrap et al., "Continuous control with deep reinforcement learning," 2019, *arXiv:1509.02971*.
- <span id="page-12-20"></span>[\[21\]](#page-1-15) T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor," in *Proc. Int. Conf. Mach. Learn.*, Jul. 2018, pp. 1861–1870.
- <span id="page-12-21"></span>[\[22\]](#page-1-16) C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning (still) requires rethinking generalization," *Commun. ACM*, vol. 64, no. 3, pp. 107–115, Mar. 2021.
- <span id="page-12-22"></span>[\[23\]](#page-2-1) J. Y. S. Luh and Y. F. Zheng, "Constrained relations between two coordinated industrial robots for motion control," *Int. J. Robot. Res.*, vol. 6, no. 3, pp. 60–70, Sep. 1987.

- <span id="page-12-23"></span>[\[24\]](#page-2-2) B. M. Braun, G. P. Starr, J. E. Wood, and R. Lumia, "A framework for implementing cooperative motion on industrial controllers," *IEEE Trans. Robot. Autom.*, vol. 20, no. 3, pp. 583–589, Jun. 2004.
- <span id="page-12-24"></span>[\[25\]](#page-2-3) J. Krüger, G. Schreck, and D. Surdilovic, "Dual arm robot for flexible and cooperative assembly," *CIRP Ann.*, vol. 60, no. 1, pp. 5–8, 2011.
- <span id="page-12-25"></span>[\[26\]](#page-2-4) F. Suárez-Ruiz and Q.-C. Pham, "A framework for fine robotic assembly," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, May 2016, pp. 421–426.
- <span id="page-12-26"></span>[\[27\]](#page-2-5) F. Suárez-Ruiz, X. Zhou, and Q.-C. Pham, "Can robots assemble an IKEA chair?" *Sci. Robot.*, vol. 3, no. 17, Apr. 2018, Art. no. eaat6385.
- <span id="page-12-27"></span>[\[28\]](#page-2-6) Y. Li, X. Hao, Y. She, S. Li, and M. Yu, "Constrained motion planning of free-float dual-arm space manipulator via deep reinforcement learning," *Aerosp. Sci. Technol.*, vol. 109, Feb. 2021, Art. no. 106446.
- <span id="page-12-28"></span>[\[29\]](#page-2-7) Y. Cao, S. Wang, X. Zheng, W. Ma, X. Xie, and L. Liu, "Reinforcement learning with prior policy guidance for motion planning of dual-arm free-floating space robot," *Aerosp. Sci. Technol.*, vol. 136, May 2023, Art. no. 108098.
- <span id="page-12-29"></span>[\[30\]](#page-2-8) C.-C. Wong, S.-Y. Chien, H.-M. Feng, and H. Aoyama, "Motion planning for dual-arm robot based on soft actor-critic," *IEEE Access*, vol. 9, pp. 26871–26885, 2021.
- <span id="page-12-30"></span>[\[31\]](#page-2-9) E. Prianto, M. Kim, J.-H. Park, J.-H. Bae, and J.-S. Kim, "Path planning for multi-arm manipulators using deep reinforcement learning: Soft actor–critic with hindsight experience replay," *Sensors*, vol. 20, no. 20, p. 5911, Oct. 2020.
- <span id="page-12-31"></span>[\[32\]](#page-2-10) S. Wang, Y. Cao, X. Zheng, and T. Zhang, "A learning system for motion planning of free-float dual-arm space manipulator towards non-cooperative object," *Aerosp. Sci. Technol.*, vol. 131, Dec. 2022, Art. no. 107980.
- <span id="page-12-32"></span>[\[33\]](#page-2-11) Y. Lin et al., "Bi-touch: Bimanual tactile manipulation with sim-to-real deep reinforcement learning," *IEEE Robot. Autom. Lett.*, vol. 8, no. 9, pp. 5472–5479, Sep. 2023.
- <span id="page-12-33"></span>[\[34\]](#page-2-12) D. Jiang, H. Wang, and Y. Lu, "Mastering the complex assembly task with a dual-arm robot based on deep reinforcement learning: A novel reinforcement learning method," *IEEE Robot. Autom. Mag.*, vol. 30, no. 2, pp. 57–66, Jun. 2023.
- <span id="page-12-34"></span>[\[35\]](#page-3-2) W. Tang, C. Cheng, H. Ai, and L. Chen, "Dual-arm robot trajectory planning based on deep reinforcement learning under complex environment," *Micromachines*, vol. 13, no. 4, p. 564, Mar. 2022.
- <span id="page-12-35"></span>[\[36\]](#page-3-3) B. Fang, S. Jia, D. Guo, M. Xu, S. Wen, and F. Sun, "Survey of imitation learning for robotic manipulation," *Int. J. Intell. Robot. Appl.*, vol. 3, no. 4, pp. 362–369, Dec. 2019.
- <span id="page-12-36"></span>[\[37\]](#page-3-4) G. Franzese, L. D. S. Rosa, T. Verburg, L. Peternel, and J. Kober, "Interactive imitation learning of bimanual movement primitives," *IEEE/ASME Trans. Mechatronics*, 2023, doi: [10.1109/TMECH.2023.](http://dx.doi.org/10.1109/TMECH.2023.3295249) [3295249.](http://dx.doi.org/10.1109/TMECH.2023.3295249)
- <span id="page-12-37"></span>[\[38\]](#page-3-5) R. P. Joshi, N. Koganti, and T. Shibata, "A framework for robotic clothing assistance by imitation learning," *Adv. Robot.*, vol. 33, no. 22, pp. 1156–1174, Nov. 2019.
- <span id="page-12-38"></span>[\[39\]](#page-5-6) T. Sugihara, "Solvability-unconcerned inverse kinematics by the Levenberg–Marquardt method," *IEEE Trans. Robot.*, vol. 27, no. 5, pp. 984–991, Oct. 2011.
- <span id="page-12-39"></span>[\[40\]](#page-6-2) T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, "Reinforcement learning with deep energy-based policies," in *Proc. Mach. Learn. Res. (ICML)*, Aug. 2017, pp. 1352–1361.
- <span id="page-12-40"></span>[\[41\]](#page-6-3) S. Fujimoto, H. van Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," in *Proc. Mach. Learn. Res. (ICML)*, Jul. 2018, pp. 1587–1596.
- <span id="page-12-41"></span>[\[42\]](#page-7-6) E. Rohmer, S. P. N. Singh, and M. Freese, "V-REP: A versatile and scalable robot simulation framework," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.*, Nov. 2013, pp. 1321–1326.

Yuanzhe Cui received the B.Sc. degree in mechanical engineering from Jilin University, China, in 2020. He is currently pursuing the Ph.D. degree in robotics and mechatronics with the Laboratory of Robotics and Multibody System, Tongji University. His research interests include motion planning and control algorithms of multiple robot arms based on reinforcement learning, cooperative operation and force conflict resolution of multiple aerial robotic manipulators, and bionic swarm robots.

Zhipeng Xu received the B.Sc. and M.Sc. degrees in mechanical engineering from Tongji University, China, in 2018 and 2021, respectively. He made contribution on the practical experiments with UR5 robot arms for this paper during his study with Tongji University. His research interests include simulation of multi-robot cooperative manipulation, motion planning of dual-arm system based on reinforcement learning, and distributed cooperation of swarm robots based on stigmergy mechanism.

Yichao Shen (Graduate Student Member, IEEE) received the B.Sc. and M.Sc. degrees in mechanical engineering from Shanghai Jiao Tong University, China, in 2018 and 2021, respectively. He is currently pursuing the Dr.-Ing. degree with the University of Stuttgart, Germany. He is an Assistant Researcher with the Laboratory of Robotics and Multibody System, Tongji University. His research interests include dynamics and control algorithms of bionic swarm robotics.

Lou Zhong received the B.Sc. and M.Sc. degrees in mechanical engineering from Tongji University, China, in 2020 and 2023, respectively. He made contribution on the framework of text structure for this paper during his study with Tongji University. His research interests include space dual-arm systems and autonomous grasp of manipulators.

Pengjie Xu received the B.Sc. degree in mechanical engineering from the Shandong University of Technology, China, in 2015, the M.Sc. degree in mechanical engineering from Qingdao University, China, in 2018, and the Ph.D. degree from Tongji University, China, in 2023. He made contribution on the parameters tuning of reinforcement learning algorithm for this paper during his study with Tongji University. His research interests include kinematics modeling, reinforcement learning, and multiple mobile manipulators for cooperative transportation.

Qirong Tang (Member, IEEE) received the B.Sc. and M.Sc. degrees in mechanical engineering and mechatronics from the Harbin Institute of Technology, China, in 2006 and 2008, respectively, and the Dr.-Ing. degree from the University of Stuttgart, Germany, in 2012. From 2012 to 2014, he was a Senior Research Associate and the Robot Group Leader with the Institute of Engineering and Computational Mechanics, University of Stuttgart. He is currently a Full Professor (with distinguish) and the Founding Director of the Laboratory of Robotics

and Multibody System, as well as the Leader of the Intelligent Unmanned Systems Group, Tongji University, Shanghai, China. His research interests include swarm robotics, robotic manipulator, underwater vehicle, and space robotics.