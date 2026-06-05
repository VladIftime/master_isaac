# Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning

Sainbayar Sukhbaatar, Emily Denton

Department of Computer Science New York University {sainbar,denton}@cs.nyu.edu

Arthur Szlam, Rob Fergus Facebook AI Research New York {aszlam,robfergus}@fb.com

# Abstract

In hierarchical reinforcement learning a major challenge is determining appropriate low-level policies. We propose an unsupervised learning scheme, based on asymmetric self-play from Sukhbaatar *et al*. [\[15\]](#page-8-0), that automatically learns a good representation of sub-goals in the environment and a low-level policy that can execute them. A high-level policy can then direct the lower one by generating a sequence of continuous sub-goal vectors. We evaluate our model using Mazebase and Mujoco environments, including the challenging AntGather task. Visualizations of the sub-goal embeddings reveal a logical decomposition of tasks within the environment. Quantitatively, our approach obtains compelling performance gains over non-hierarchical approaches.

# 1 Introduction

Complex real-world tasks may require the agent to take thousands or millions of individual actions. However, tasks are usually a composition of a much smaller number of sub-tasks, reused across different objectives. This naturally suggests a hierarchical structure for the agent's controller, in which individual actions are performed by a low-level policy, while a higher-level one selects appropriate sub-goals. Correspondingly, hierarchical reinforcement learning (HRL) is a subject of much interest. Automatic learning of hierarchical policies poses many challenges, e.g. ensuring an appropriate division of labor between the levels without explicit supervision; preventing degeneracies in the low-level policies and deciding/designing how the levels should communicate with one another.

In this work we explore a form of HRL in which the high-level policy directs the low-level policy via a continuous sub-goal vector. With this approach, the challenge is to learn an appropriate representation space for the goals. Specifically, (i) the space should be general enough to cover the full range of sub-tasks within the environment; (ii) task-irrelevant details of the environment should be abstracted away, while retaining enough information to discriminate between different sub-goals and (iii) it should encode sub-goals achievable by the low-level policy (i.e. at the correct level of difficulty).

Our approach uses unsupervised asymmetric self-play [\[15\]](#page-8-0) as a pre-training phase for the low-level policy, prior to training the hierarchical model. In self-play, the agent devises tasks for itself via the goal embedding and then attempts to solve them. Since each task involves a physical change in the environment, the goal embedding learned via self-play captures aspects of the environment that are controllable, i.e. can be changed by the agent, and ignores parts that it cannot alter (e.g. static background or purely random elements). This is helpful for the high-level policy since user-specified tasks typically involve manipulations of the environment. Furthermore, the adversarial reward structure forces the agent to constantly come up with new tasks, thus ensuring a diverse goal representation. Imposing a time limit on each task limits their complexity and so ensures good coverage of goals of a given difficulty.

A key aspect of our approach is the parameterization of the low-level policy, which takes as input both the current state and a goal vector. The latter is an encoding of the target state, learned during self-play to guide the agent to complete self-imposed tasks. This provides a natural mechanism for the higher level controller to specify sub-goals that make up complex tasks. The higher policy is trained using sparse task reward as supervision and we show experimentally that it is able to learn tasks that are difficult for an agent trained at the level of atomic actions.

#### 1.1 Related Work

Options [\[17\]](#page-8-1), a formalization of temporal abstraction in an MDP, have become a popular framing of hierarchical RL. While earlier works used pre-specified option policies, there has been recent success in discovering options. For example, Bacon *et al*. [\[1\]](#page-7-0) extends the policy gradient theorem to the setting of options, and shows with the appropriate entropy regularization (to avoid collapsing to trivial one-option policies) and termination regularization (to avoid collapse onto full control of the low level actor by the high level controller), useful discrete options could be learned.

Many recent works have considered options discovery via parameterized modules operating at different timescales, where an "actor" operates at a finer timescale than a "manager", which outputs a goal or target for the actor. For example, Vezhnevets *et al*. [\[20\]](#page-8-2) trains the actor and manager together end-to-end via reward from the environment.

A line of work [\[9,](#page-7-1) [5,](#page-7-2) [4,](#page-7-3) [7,](#page-7-4) [3\]](#page-7-5) takes this approach in the context of intrinsic motivation. In Mohamed *et al*. [\[9\]](#page-7-1) a variational inference approach is used to make exploration via empowerment [\[8\]](#page-7-6) tractable. Continuing in this path [\[5,](#page-7-2) [4,](#page-7-3) [7,](#page-7-4) [3\]](#page-7-5) use an actor parameterized by state and a latent vector in such a way that the latent vector is predictable from a final state or a sequence of states the actor visits, but otherwise, the actions have high entropy. After pre-training in this way, a "manager" can learn to issue commands via the latent vector. In Haarnoja *et al*. [\[6\]](#page-7-7), a similar construction is used to train an agent end to end. Our work also uses this construction, but the unsupervised pre-training of the actor is done via asymmetric self-play as in [\[15\]](#page-8-0).

There is a large literature on goal discovery and intrinsic motivation, both independent of RL [\[12,](#page-8-3) [10\]](#page-8-4) and framed in terms of RL [\[14\]](#page-8-5). Recently, Pete *et al*. [\[11\]](#page-8-6) used a construction where the goal space is learned first by using an auto-encoder on states from the environment, and then using a goal discovery algorithm on top of the learned representation. In this work, we use an intrinsic motivation approach to learn both a low level actor and the representation of the state space. In future work, we intend to do as in [\[11\]](#page-8-6) and consider goal discovery at the level of the manager as well.

![](_page_1_Figure_6.jpeg)

<span id="page-1-0"></span>Figure 1: The Hierarchical Self-Play model in pre-training (left) and fine-tuning (right) phases.

# 2 Approach

We consider single-agent reinforcement learning in a fully-observable MDP. Let s ∈ R <sup>s</sup> be the state, a be an action. The agent has a hierarchical controller that consists of a high-level policy π<sup>C</sup> , known as "Charlie", that directs a low-level actor "Bob" πB. Both π<sup>B</sup> and π<sup>C</sup> are modeled with deep neural

nets. On a given target task, both the high and low-level policies will be learned using external reward after an unsupervised pre-training phase.

The pre-training phase consists of Bob exploring and building skills. Bob is paired with another controller  $\pi_A$  "Alice". During the pre-training phase, Bob learns a continuous goal embedding g which will be used by Charlie on the target task to communicate local sub-goals to Bob. Fig. 1 summarizes the approach.

## 2.1 Self-play Pre-training

In this phase, Alice and Bob take turns in controlling the agent. An internal reward is structured to induce a competition between them that results in the agent exploring the environment. In so doing, Bob gains an understanding of how the environment operates, i.e. how to transition from one state to one nearby. The approach is a modified form of asymmetric self-play introduced in Sukhbaatar [15].

Self-play commences with  $\pi_A$  and  $\pi_B$  randomly initialized and the agent in some initial state  $s_0$ . First, Alice takes  $T_A$  steps in the environment using policy  $\pi_A$ .

$$a_t^A = \pi_A(s_t^A, s_0)$$

Let  $s^* = s_{T_A}^A$  be Alice's final state, which becomes Bob's goal. Next, we reset the environment back to  $s_0$  and Bob takes control, taking actions according to his policy  $\pi_B$ .

$$a_t^B = \pi_B(s_t^B, s^*)$$

Bob is deemed to have succeeded if any time step his state is close to  $s^*$  under distance function D (which depends on the environment; see Section 3):

$$D(s_t^B, s^*) \le \epsilon$$

If after  $T_B$  steps this criterion has not been met, Bob has failed. Bob's reward  $R_B$  is 1 if he succeeds, 0 otherwise. Alice's reward is  $R_A = 1 - R_B$ . This reward is used to update  $\pi_A$  and  $\pi_B$  using Reinforce [21], although any other policy gradient or Q-learning based algorithm could potentially be used. The reward structure causes Alice to seek states  $s^*$  that Bob has difficulty reaching. But since we choose  $T_A \sim T_B$  and both have the same abilities (in terms of capacity & actions), Bob will quickly master a task set by Alice, forcing her to find new, unexplored ones to challenge Bob. Through many episodes of self play, Alice and Bob are thus able to explore the environment effectively.

In self-play, Bob is trained only on tasks that are achievable under  $T_A$  steps. This becomes an issue when episodes always start at the same state, because Bob never observes states that are far from the initial state  $s_0$ . A simple solution is to perform several self-play games within a single episode. An episode starts with a standard self-play game where Bob tries to reach the same state as Alice. If Bob succeeds, then the episode continues with another self-play game. However, for the next game, both Alice and Bob start from the last states they reach in the previous game. This allows them to explore far away states. An episode ends either when Bob fails or after a fixed number of games. Rewards are discounted by a hyperparameter  $\lambda$  between games, while the reward discount is 1 within each game.

Since, for every proposed task, we have the ground truth actions from Alice for achieving that task, we can train Bob's policy to imitate them. Thus, the final form of Bob's loss function is

$$L_B = \mathbb{E}_{a_t^B \sim \pi_B}[-R_B] + \alpha \mathbb{E}_{a_t^A \sim \pi_A}[-\log(\pi_B(a_t^A | s_t^A))],$$

where the hyper-parameter  $\alpha$  is for balancing the two loss terms.

For Alice, we add an entropy regularization on her policy to encourage her to propose diverse tasks. Her loss function is

$$L_A = \mathbb{E}_{a_t^A \sim \pi_A} [-R_A - \beta H(\pi_A(s_t^A))].$$

Here H is the entropy function, and  $\beta$  is the coefficient of the entropy regularization.

This scheme differs from [15] in several ways:

(i) The number of steps taken by Alice and Bob are fixed to  $T_A$  and  $T_B$  respectively, versus being dynamic as in [15]. This constrains the scope of work done by the low-level policy to be manageable and also reduces the amount of exploration needed. We choose  $T_B$  to be slightly larger than  $T_A$  so Bob has a chance of success even with few mistakes.

- (ii) The episodes are broken into multiple shorter segments with the environment reset to the beginning of the segment instead of the beginning of the episode. This allows for more exploration while keeping Bob's policy manageable for Charlie.
- (iii) Keeping with the other changes, instead of having a reward for Bob based on time, we adopt a simplified 0/1 reward.
- (iv) The structure of Bob, detailed below.

#### 2.2 Bob's Architecture

Bob's policy has two components. The first is a goal encoder E that maps the current and goal states to a low-dimensional goal embedding  $g_t \in \mathbb{R}^K$ :

$$g_t = E(s^*, s_t^B).$$

The low dimension of the space (i) acts as a bottleneck, forcing Bob to compactly represent the goal and (ii) makes Charlie's job easier, as he will be generating  $g_t$ 's to control Bob on the target task.

We consider two forms of the goal encoder E:

- 1. Compute the **difference** between current and target states:  $E(s^*, s_t^B) = \phi(s^*) \phi(s_t^B)$ , where  $\phi$  is a state embedding function.
- 2. An **absolute** representation, which just considers  $s^*$ :  $E(s^*, s_t^B) = \phi(s^*)$ .

Bob's second component is a policy conditioned on a goal embedding

$$a_t^B = \pi_B'(s_t^B, g_t).$$

Putting the two stages together, we have

$$a_t^B = \pi_B(s_t^B, s^*) = \pi_B'(s_t^B, E(s^*, s_t^B)).$$

#### 2.3 Training Charlie

After self-play pre-training, Bob is used as a low-level policy for solving the user-specified target task. A high-level policy, Charlie, is introduced that outputs a goal vector  $g_t$  which controls Bob:

$$g_t = \pi_C(s_t).$$

This vector is then fed to Bob's goal policy  $\pi'_B$ , which converts it to an action on the environment

$$a_{t+i} = \pi'_B(s_{t+i}, g_t).$$

Here, Bob will take multiple actions with the same goal for  $i = [0, \cdots, T_C - 1]$ , thus Charlie's next action occurs at  $t + T_c$ . External reward from the task is used to train Charlie, as well as fine-tuning Bob's policy  $\pi'_B$ . We choose  $T_C$  to be equal to  $T_A$  because Bob is trained on goals achievable in  $T_A$  steps.

## 2.4 Parameterization of the Components

We use a multi-layer perceptron (MLP) with tanh non-linearity for parameterizing all the components of our model. Both  $\pi_A$  and  $\pi_C$  are two-layer MLPs. For  $\phi$ , we also use a two-layer MLP, but without non-linearity at the last layer to avoid putting bounds on the goal embedding. Lastly, we use a three-layer MLP for Bob's policy  $\pi_B'$  where the second hidden layer is given by

$$h_2 = \sigma(W_2\sigma(W_1s_t) + W_gg_t).$$

 $\sigma$  is a tanh non-linearity, and bias terms are omitted for brevity. All the hidden layers have 64 units.

All the policy networks have M+1 output heads, where M is the dimension of the action space. For Charlie, M is equal to the goal embedding dimension, K. Besides M action heads, each policy network also outputs a baseline value. For a discrete action space, each head is a linear layer followed by softmax. For continuous actions, each action head outputs  $\mu$  and  $\log(\sigma)$  values, and an action is sampled from  $\mathcal{N}(\mu,\sigma)$ .

# <span id="page-4-0"></span>3 Experiments

We test our hierarchical self-play (HSP) model on two different environments. First, it is applied to a task procedurally generated in a grid-world environment, Mazebase [\[16\]](#page-8-8). The second environment is a control of an ant-like robot in Mujoco physics simulation [\[19\]](#page-8-9), where it has to collect randomly placed objects.

In all experiments, we set Alice's entropy regularization coefficient β to 0.01, and Bob's imitation coefficient α to 0.03. For training all policies, we use the REINFORCE [\[21\]](#page-8-7) algorithm with a learned baseline. However, our model can be trained by more sample efficient policy gradient algorithms such as TRPO and PPO. For optimization, we use RMSProp [\[18\]](#page-8-10) with learning rate 0.001 and α = 0.97, = 1e − 6. We run each experiment 5 times with different random seeds, and report their mean and standard deviation. In the training plots, the pre-training steps are not included as they are unsupervised, and the same pre-trained model potentially can be used for different tasks. The code is available at <https://cims.nyu.edu/~sainbar/hsp>.

# 3.1 MazeBase

We test our model in the Mazebase [\[16\]](#page-8-8) environment on the "Key-Door" task, where the grid is divided into two rooms by a wall as shown in Fig. [2\(](#page-4-1)left). The objective is to reach the treasure goal, but the agent first needs to pick up the key and then open the door.

The task is not trivial because the object locations are randomized for every episode. In addition, it has sparse reward (+1 for success and 0 otherwise), making it even more challenging.

The observation is a binary vector of size MapWidth×MapHeight×VocabularySize. The vocabulary consists of words necessary for describing objects: "agent", "door", "block", "key" and "goal". The possible actions are four one-step movement actions, pick action, and stop action. To pick an item, the agent has to be on top of it and perform the pick action. Episodes terminate after 40 steps.

During self-play, we use the `2-norm on the state as distance metric D and set = 0. This means Bob needs to exactly match everything in the grid to Alice's final state. We use a goal embedding of K = 3 dimensions.

Alice always takes T<sup>A</sup> = 5 steps, and Bob has T<sup>B</sup> = 7 steps to reach the goal state. However, we allow up to 4 self-play games per episode with discount of λ = 0.7 between games. This effectively allows Alice and Bob to take up to 4T<sup>A</sup> = 20 and 4T<sup>B</sup> = 28 steps respectively, which is usually enough for picking up the key and entering the other room. We use the absolute version of the goal embedding function E. When training Charlie, we set T<sup>C</sup> = 5.

Alice learns to propose increasingly complex tasks throughout self-play. Initially, Alice's tasks are movement based, but by the end of self-play training Alice has explored the space of tasks afforded by the KeyDoor environment and frequently unlocks the door and navigates to the second room. This is illustrated in Fig. [3](#page-5-0) (left) which plots the frequency of different types of tasks proposed by Alice. Empirically, we see that the corresponding goal space learned during self-play training reflects the controllable aspects of the environment. Fig. [3](#page-5-0) (right) visualizes the learned goal embedding by

![](_page_4_Figure_10.jpeg)

![](_page_4_Figure_11.jpeg)

<span id="page-4-1"></span>Figure 2: Left: In the KeyDoor task, the agent needs to unlock the door with the key and go to the goal. Right: Comparison of our approach HSP with other baselines: REINFORCE with no self-play (blue) and self-play from [\[15\]](#page-8-0) which has no hierarchy (red).

![](_page_5_Figure_0.jpeg)

![](_page_5_Figure_1.jpeg)

<span id="page-5-0"></span>Figure 3: Left: Plot showing probability of interacting with key and door objects during a self-play episode, as self-play training progresses. Note that the task distribution proposed by Alice changes: initially tasks are only movements (i.e. low chance of either key/door interaction), but chances of collecting the key steeply increases after just a few epochs, and going through the door gradually increases. Right: Goal embeddings learned by Bob for a particular maze. The green (red) points correspond to states where the door locked (unlocked).

plotting φ(s) for each possible state, s, of a particular maze instance. Two distinct planes are evident in the goal embedding corresponding to states in which the door is locked (green) and unlocked (red). Within each plane the the spatial structure of the grid world is evident.

We then train Charlie on the KeyDoor task, comparing against REINFORCE [\[21\]](#page-8-7) and self-play [\[15\]](#page-8-0) algorithms. Our hierarchical self-play model outperforms REINFORCE by a significant margin as shown in Fig. [2\(](#page-4-1)right). Our improvement over the self-play baseline indicates that it is not merely the unsupervised training that provides the boost in performance. Rather, the hierarchy introduced by Charlie is crucial for good performance.

### 3.2 Mujoco Ant

Next, we apply our method to the Ant environment from [\[2\]](#page-7-8), where the agent controls a four-legged ant-like robot. The agent takes as an observation a 125 dimensional vector that contains location, velocity and joint angles. The action is a 8 dimensional vector that controls the 8 joints of the ant. Although the actions are continuous, we discretize them into 5 bins.

In self-play, we define the distance function D as a physical distance in the x − y plane, and set = 0.25. This means Bob only needs to reach a location with distance less than 0.25 from the Alice's last location. Alice always takes T<sup>A</sup> = 50 steps, and Bob is allowed up to T<sup>B</sup> = 70 steps. Like the MazeBase experiments, we allow 4 games per episode and use discount λ = 0.7. We set the dimension of goal embedding K = 2, and use the difference version of the goal embedding function E.

![](_page_5_Figure_8.jpeg)

<span id="page-5-1"></span>Figure 4: AntGather task. Left: Example of the environment. Right: Comparison of our approach HSP to other baselines.

We train Charlie on the more complicated AntGather task [2]. As shown in Fig. 4 (left), the ant is placed in a small arena with green and red objects. The object locations are randomized in every episode. When the ant touches an object , the object disappears and the agent gets a reward of +1 (-1) if the object was green (red). Each episode terminates after 1000 steps. In addition, if the ant jumps too high (in the z-axis), it gets penalty of -10 and the episode terminates. This makes the task more challenging because it inhibits exploration and introduces a local minima where the ant learns to stay still to avoid this penalty. This penalty is also present in the self-play phase.

Compared to the Ant environment, the observation of AntGather task includes an additional 20 dimensional vector containing sensory values for detecting nearby objects (see [2] for more details). While Charlie takes this full observation as input, we give Bob an observation without those sensory values<sup>1</sup>. We set  $T_C = 50$ , so Charlie's single action corresponds to 50 steps in the environment, and he is allowed to take 20 actions in an episode.

![](_page_6_Figure_2.jpeg)

<span id="page-6-3"></span>Figure 5: **Left:** The average distance of tasks proposed by Alice during training. **Center:** Goal locations proposed by Alice by the end of training. **Right:** Corresponding goal embeddings. Their color comes from their location, so we can see the goal space resembles spatial space.

As a baseline, we compare our model to the REINFORCE [21], PPO [13], and self-play<sup>2</sup> [15] algorithms. We use an open-source implementation<sup>3</sup> of PPO, and run it with the recommended hyper-parameters. As validation, we run it on the Ant task, where it outperforms our REINFORCE baseline by a large margin. However, on the AntGather task, it failed to learn at all. We note that Duan *et al*. [2] also ran several other RL algorithms including TRPO on this task, and all of them failed to learn. The REINFORCE and self-play baselines also fail to obtain a positive reward. In contrast, our model obtains an average reward of 0.5 after training, outperforming the baselines as shown in Fig. 4 (right).

In Fig. 5 (left), we show the average distance of tasks proposed by Alice during self-play training. We can see that Alice and Bob are learning to travel farther as training progresses. In Fig. 5 (center), we show the actual goal XY locations proposed by Alice by the end of training. It is evident that Alice proposes diverse goal positions in all directions. Large entropy regularization on Alice's policy was important for maintaining this diversity. For every proposed task  $s^*$  in Fig. 5 (center), we plot their corresponding goal embedding  $\phi(s^*)$  in Fig. 5 (right). The points are colored by their locations from Fig. 5 (center), so we can see the goal embedding captures the spatial structure of the state (up to an arbitrary global transformation) by the end of the training.

Next, we choose 4 fixed points from Fig. 5 (right) and fed them as a goal to Bob for 10 times with  $T_C=50$ . As shown in Fig. 6, Bob travels in different directions depending on which goal is chosen. This shows that it is possible to control Bob using the goal vector. There is also diversity inside each goal due to the stochasticity of Bob's policy. Another interesting point is that Bob managed to travel much longer average distances (6 or more) with those fixed goals, even though during self-play, he only traveled distances up to 0.7 and only visited a small spatial region shown in Fig. 5 (center). This indicates that Bob's policy is capable of generalizing to unseen states in some degree.

<span id="page-6-0"></span><sup>&</sup>lt;sup>1</sup>During self-play on the Ant task, the state lacks the sensory input dimensions. Hence Bob would not know what to do with them in the AntGather task.

<span id="page-6-1"></span><sup>&</sup>lt;sup>2</sup>The self-play baseline has almost the same number of unsupervised training steps as our model.

<span id="page-6-2"></span><sup>3</sup> https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

![](_page_7_Figure_0.jpeg)

<span id="page-7-9"></span>Figure 6: Trajectories taken by Bob in XY space when given selected goals from Fig. [5\(](#page-6-3)right) (stars A,B,C,D) as input. Consider, for example, goal A. This corresponds to a target location in the upper left (cyan colored diamonds in Fig. [5](#page-6-3) (center)). We see that Bob's trajectories for goal A move in that direction also. Additionally, the total distance covered by Bob is far greater than that traversed during self-play, demonstrating Charlie's ability to reuse Bob's policy and also generalization of the state embedding φ(s).

# 4 Discussion

We have proposed a novel approach for learning goal embeddings that relies on the concept of unsupervised self-play. These can then be utilized in a hierarchical RL framework to speed exploration on complex tasks with sparse reward. Experiments on AntGather demonstrate the ability of the resulting hierarchical controller to move the Ant long distances to obtain reward, unlike non-hierarchical policy gradient methods.

One limitation of our self-play approach is that the choice of D (the distance function used to decide if the self-play task has been completed successfully or not) requires some domain knowledge. For example, in the Mujoco Ant tasks, we chose D to only care about x − y location, not height and joint angles.

Although REINFORCE was used in our experiments, more sophisticated policy gradients or Qlearning could be used instead at both the level of the unsupervised self-play pre-training and the test task reinforcement learning. Another future direction is to expand the self-play concept to the higher-level controller, where a meta-Alice and meta-Bob would play against one another, passing sub-goals to low-level Alice and Bob. Meta-Bob could be used as pre-trained version of Charlie, so reducing the supervision required on the target task.

# References

- <span id="page-7-0"></span>[1] P.-L. Bacon, J. Harb, and D. Precup. The option-critic architecture. In *AAAI*, 2017.
- <span id="page-7-8"></span>[2] Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel. Benchmarking deep reinforcement learning for continuous control. In *ICML*, 2016.
- <span id="page-7-5"></span>[3] B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine. Diversity is all you need: Learning skills without a reward function. *arXiv*, abs/1802.06070, 2018.
- <span id="page-7-3"></span>[4] C. Florensa, D. Held, M. Wulfmeier, M. Zhang, and P. Abbeel. Reverse curriculum generation for reinforcement learning. In *Proceedings of the 1st Annual Conference on Robot Learning*, pages 482–495, 2017.
- <span id="page-7-2"></span>[5] K. Gregor, D. J. Rezende, and D. Wierstra. Variational intrinsic control. *arXiv*, abs/1611.07507, 2016.
- <span id="page-7-7"></span>[6] T. Haarnoja, K. Hartikainen, P. Abbeel, and S. Levine. Latent space policies for hierarchical reinforcement learning. In *ICML*, 2018.
- <span id="page-7-4"></span>[7] K. Hausman, J. T. Springenberg, Z. Wang, N. Heess, and M. Riedmiller. Learning an embedding space for transferable robot skills. In *ICLR*, 2018.
- <span id="page-7-6"></span>[8] A. S. Klyubin, D. Polani, and C. L. Nehaniv. Empowerment: a universal agent-centric measure of control. In *Proceedings of the IEEE Congress on Evolutionary Computation, CEC*, pages 128–135, 2005.
- <span id="page-7-1"></span>[9] S. Mohamed and D. J. Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. In *NIPS*, pages 2125–2133, 2015.

- <span id="page-8-4"></span>[10] P. Oudeyer and F. Kaplan. What is intrinsic motivation? A typology of computational approaches. *Front. Neurorobot.*, 2009.
- <span id="page-8-6"></span>[11] A. Péré, S. Forestier, O. Sigaud, and P.-Y. Oudeyer. Unsupervised learning of goal spaces for intrinsically motivated goal exploration. In *ICLR*, 2018.
- <span id="page-8-3"></span>[12] J. Schmidhuber. Curious model-building control systems. In *Proc. Int. J. Conf. Neural Networks*, pages 1458–1463. IEEE Press, 1991.
- <span id="page-8-11"></span>[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. *arXiv*, abs/1707.06347, 2017.
- <span id="page-8-5"></span>[14] S. P. Singh, A. G. Barto, and N. Chentanez. Intrinsically motivated reinforcement learning. In *NIPS*, pages 1281–1288, 2004.
- <span id="page-8-0"></span>[15] S. Sukhbaatar, Z. Lin, I. Kostrikov, G. Synnaeve, A. Szlam, and R. Fergus. Intrinsic motivation and automatic curricula via asymmetric self-play. In *ICLR*, 2018.
- <span id="page-8-8"></span>[16] S. Sukhbaatar, A. Szlam, G. Synnaeve, S. Chintala, and R. Fergus. Mazebase: A sandbox for learning from games. *arXiv 1511.07401*, 2015.
- <span id="page-8-1"></span>[17] R. S. Sutton, D. Precup, and S. P. Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. *Artif. Intell.*, 112:181–211, 1999.
- <span id="page-8-10"></span>[18] T. Tieleman and G. Hinton. Lecture 6.5 - rmsprop, coursera: Neural networks for machine learning, 2012.
- <span id="page-8-9"></span>[19] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In *IROS*, pages 5026–5033. IEEE, 2012.
- <span id="page-8-2"></span>[20] A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jaderberg, D. Silver, and K. Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. In *ICML*, 2017.
- <span id="page-8-7"></span>[21] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. In *Machine Learning*, pages 229–256, 1992.