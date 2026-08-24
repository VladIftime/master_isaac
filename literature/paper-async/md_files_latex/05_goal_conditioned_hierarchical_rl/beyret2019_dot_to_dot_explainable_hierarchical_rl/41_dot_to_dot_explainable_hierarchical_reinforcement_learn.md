# Dot-to-Dot: Explainable Hierarchical Reinforcement Learning for Robotic Manipulation

Benjamin Beyret<sup>1</sup> , Ali Shafti<sup>1</sup> and A. Aldo Faisal1,2,3,<sup>4</sup>

*Abstract*— Robotic systems are ever more capable of automation and fulfilment of complex tasks, particularly with reliance on recent advances in intelligent systems, deep learning and artificial intelligence. However, as robots and humans come closer in their interactions, the matter of interpretability, or explainability of robot decision-making processes for the human grows in importance. A successful interaction and collaboration will only take place through mutual understanding of underlying representations of the environment and the task at hand. This is currently a challenge in deep learning systems. We present a hierarchical deep reinforcement learning system, consisting of a low-level agent handling the large actions/states space of a robotic system efficiently, by following the directives of a high-level agent which is learning the highlevel dynamics of the environment and task. This high-level agent forms a representation of the world and task at hand that is interpretable for a human operator. The method, which we call Dot-to-Dot, is tested on a MuJoCo-based model of the Fetch Robotics Manipulator, as well as a Shadow Hand, to test its performance. Results show efficient learning of complex actions/states spaces by the low-level agent, and an interpretable representation of the task and decision-making process learned by the high-level agent.

# I. INTRODUCTION

Robots are increasingly present in our lives, from production lines to homes, hospitals and schools, we rely more and more on their presence. Robots working directly with humans are either controlled directly by human input [1], pre-programmed to follow a pre-planned choreography of movements between the human and the robot [2] or programmed to follow a control law. Having an intelligent robot that can learn a task and adapt to situations unseen before, while interacting with a human, is a major challenge in the field.

The rapid advance of artificial intelligence has led researchers to the creation of intuitive agents representing robots in simulated environments, or on real-world robots in specific use cases. The majority of these intelligent systems are based on deep learning [3], [4]. While these systems produce impressive results in automation, they also result in what is referred to as a "blackbox" algorithm, i.e. the decision-making process, and factors affecting it are not clear to the human user. A human interacting with such a system will not realise how their actions are being interpreted by the robotic system, and how they lead to robot actions. These systems are therefore not explainable. For a human interacting with such a system, not knowing how the robot's

<span id="page-0-0"></span>Fig. 1: Dot-to-Dot in action: the agent learns to break down a complex task into simpler low-level actions. On FetchPickAndPlace: we first generates the green dot as a subgoal, using the high-level policy [\(1a\)](#page-0-0), then the low-level policy reaches that sub-goal [\(1b](#page-0-0) and [1c\)](#page-0-0), and finally the end goal [\(1d\)](#page-0-0)

actions relate to their own behaviour or the environment, can lead to reduced trust, reliability, safety and efficiency for the overall interaction.

Reinforcement learning is one of the promising solutions to the intelligent robotics problem [5], [6]. Most of these techniques use some form of optimization to solve a task in a given robotics environment [3], however very few actually look at the inherent structure of the tasks, or are concerned with creating higher level representations which are interpretable by an interacting human user.

In this work, our main goal is to create an intelligent robotic agent that can solve manipulation tasks by learning some form of curriculum [7] as well as a structure of movement, in a manner that conserves interpretability for a human operator. Our solution has a hierarchical structure where a complex task is split into multiple low-level consecutive simpler actions. This idea builds on the concept of action grammars governing human behaviour, which we previously used in a human-robot interaction scenario [8]. In our proposed hierarchy, the high-level agent will divide the full task into smaller (easier) actions that a low-level agent can learn

<sup>1</sup>Brain and Behaviour Lab: Dept. of Computing, <sup>2</sup> Dept. of Bioengineering, <sup>3</sup> Data Science Institute, <sup>4</sup> UKRI Centre for Doctoral Training in AI for Healthcare – bb1010@ic.ac.uk, a.shafti@ic.ac.uk, a.faisal@ic.ac.uk

to fulfil. In this manner the low-level agent will learn to make sense of the many degrees of freedom of the system with a curriculum learning approach, while the high-level agent will learn the overall dynamics of the environment and the task at hand, creating a high-level representation governing the decision-making process. Effectively, the high-level agent serves as an interpreter between the human, the environment and the low-level agent controlling the robot's many degrees of freedom. This will serve as a fundamental step in making intelligent robots operating through reinforcement learning that is explainable to human users.

The paper is structured as follows: Section II covers the necessary background upon which our method is built. Section III presents our proposed algorithm, *Dot-to-Dot*, with its design and implementation explained in detail. Section IV reports the results of training and task performance for Dot-to-Dot, followed by a discussion on the inner representation the high-level agent creates of its world, and its interpretability. Finally, Section V concludes the paper and covers potential future work.

#### II. BACKGROUND

In this section, we detail the building blocks of our work. Our algorithm relies on three existing algorithms and concepts, namely Deep Deterministic Policy Gradients (DDPG, [9]), Hindsight Experience Replay (HER, [10]) and Hierarchical Reinforcement Learning [11], which we describe below.

Let us first define a few notations used throughout this paper: an observation (i.e. states) space  $\mathcal{O}$ , an action space  $\mathcal{A}$  and a set of goals  $\mathcal{G} \subseteq \mathcal{O}$ . For example, in the case of a robotic arm that needs to push a cube around a table,  $\mathcal{O}$  could be the position of the gripper and position of the cube,  $\mathcal{G}$  can be a position on the table where the cube needs to be moved to, and  $\mathcal{A}$  can be a set of actions that end up moving the gripper. We define  $g \in \mathcal{G}$  as the goal of an episode, sampled at t=0 when resetting an environment,  $ag \in \mathcal{G}$  an achieved goal at time t>0 (e.g. position of the cube at time t), and  $sg \in \mathcal{G}$  a defined sub-goal which will be a waypoint to g. Finally,  $r_t$  is the reward at time t and  $R_t = \sum_{k=0}^t r_k$  the cumulative reward obtained by the agent at time t.

#### A. Deep Deterministic Policy Gradients

The Deep Deterministic Policy Gradients (DDPG) method [9] combines several techniques together in order to solve continuous control tasks. DDPG is an extension of the Deterministic Policy Gradients (DPG) algorithm [12]. DPG uses an actor-critic architecture [13] where the actor's role is to take actions in the environment while the critic is in charge of assessing how useful these actions are to complete a given task. In other terms, both the actor's and critic's policies are updated using the temporal difference error of the critic's value function. Moreover, DPG uses a parameterized function  $\mu(s|\theta^{\mu})$  as the actor and Q(s,a) as the critic function, which is learned through Q-learning. The updates of the actor consist in applying to both functions the

gradients of the expected return  $J = \mathbb{E}[R_t]$  with respect to the parameters  $\theta^{\mu}$ .

Finally, Deep DPG extends DPG by using neural networks as function approximators for  $\mu(s|\theta^{\mu})$  and Q(s,a), implementing an architecture similar to [14].

## B. Hindsight Experience Replay

In Hindsight Experience Replay (HER), Andrychowicz et al. [10] use an elegant trick to train agents faster in large states and actions spaces. Observing that very few traces actually yield a positive outcome (i.e. goal reached), the authors propose to make use of every trace no matter whether the goal was reached or not. In fact, they note that regardless of what the objective of a series of actions is, and no matter the outcome, we can still acquire valuable information from experience; i.e. we can still learn how every state of a trace was reached by looking at the previous states visited and actions taken in those states. During an episode, HER samples a batch of N traces  $(s_t^i, ag_t^i, a_t^i, s_{t+1}^i, ag_{t+1}^i, r_t^i, g^i)$ for t < T and  $i \le N$ . Then, at training, for a proportion Kof these traces, HER will replace g with a randomly selected  $ag_{t'}^i$  with  $t' \in \mathcal{U}([t+1;T]), \mathcal{U}$  being the uniform distribution; meaning that it assumes the incorrect state we ended up in, was in fact our goal. Therefore, in hindsight, we look at goals that were achieved instead of the original goal, learning from mistakes. This technique proved to be greatly successful for diverse robotics tasks in simulation [10].

#### C. Hierarchical Reinforcement Learning

Hierarchical reinforcement learning [11] is a technique that intends to address the problem of planning how to solve a task, by introducing various levels of abstraction. Most of the time it involves several policies, which interact with each other, often as a high-level policy dictating which of another set of policies to use and when. One such structure [11] involves several low-level policies called options, which each learn how to complete a specific task, while a high-level policy decides which low-level option to use when. Another approach [15], [16] is to match observations to goals using a higher-level policy commanding a lower-level policy, each with its own level of abstraction and temporal resolution.

Our contribution builds on top of these techniques, combining them to create an agent that achieves structured robotic manipulation. Closely related work is that of Nachum et al. [17] where a similar hierarchical structure is used to solve exploration tasks in a data-efficient manner and using observations as goals. In [18] another hierarchical structure is used to solve common toy environment tasks. The latter uses different low-level policies for each sub-goals while the former focuses on exploration tasks. In our work on the other hand, we focus on the inherent structure of robotic manipulation and reusing low-level skills across different steps of a task, as well as explainability.

# III. DESIGN AND IMPLEMENTATION

#### A. Design

We introduce a hierarchical reinforcement learning architecture which we call Dot-to-Dot (DtD). This comes from

<span id="page-2-0"></span>(a) Dot-to-Dot structure. Sub-goals generated by π1, fed to π<sup>0</sup>

(b) Example of an episode to train DtD

Fig. 2: Dot-to-Dot setup and example of an episode during training

the fact that our architecture is made of a high-level agent that generates sub-goals for a low-level agent, which follows them one by one to achieve the high-level task – resembling the children's game of the same name, where connecting dots creates an overall drawing.

To implement this, we define two policies: a low-level one π<sup>0</sup> : O × G → A and a high-level one π<sup>1</sup> : O × G → G. The low-level policy π<sup>0</sup> is trained using DDPG, that takes as inputs observations as well as goals generated by the highlevel policy π1. This is described in figure [2a.](#page-2-0)

To teach our agent complex sequences of action in a sparse reward setup, we note that it is easier to learn how to reach nearby goals than further ones. This defines the DtD method: in order to train the low-level agent in an efficient manner, for a given starting point (o0, ag0, g) ∈ O × G × G in an episode, π<sup>1</sup> (i.e. high-level agent) first generates sub-goals that are in the vicinity of ag0. This is done by making the first sub-goal sg<sup>1</sup> be a noisy perturbation of ag0, formally: sg<sup>0</sup> = ag<sup>0</sup> + N (0, σ), with N a Gaussian distribution and σ the noise parameter, effectively ignoring the final goal g. Doing so, the low-level policy π<sup>0</sup> can be trained easily on reaching these newly defined sub-goals. The other central idea of DtD is to ignore whether or not goals and sub-goals have been achieved during an episode, and to use HER for both π<sup>0</sup> and π<sup>1</sup> to extract as much information as possible from past experience.

Let us take an example of a trace and describe the training process. Figure [2b](#page-2-0) shows an episode with two sub-goals. In this example observations are represented in blue (oi), sub-goals (sgi) in green and the final goal (g) in red; ag<sup>i</sup> refers to achieved goals. First, we use π<sup>1</sup> to generate sg0. As mentioned above, at first sg<sup>0</sup> is a noisy perturbation of ag<sup>0</sup> such that: sg<sup>0</sup> = ag<sup>0</sup> + N (0, σ). Then π<sup>0</sup> tries to reach sg<sup>0</sup> in a certain number of steps and reaches ag1. At the beginning of training, we mostly have ag<sup>1</sup> 6= sg0, which is fine, as we ignore this and generate a new sub-goal sg<sup>1</sup> such that sg<sup>1</sup> = ag<sup>1</sup> + N (0, σ). Again, π<sup>0</sup> generates actions to reach sg<sup>1</sup> but instead reaches ag2. Finally, as we want our algorithm to learn which traces are useful for reaching a given goal, and which are not, we constrain the last sub-goal to be equal to the actual final goal g. Doing so, we can learn if a sequence of actions and sub-goals can reach a given goal or not.

Once this episode is done, we have obtained a series of states (oi) reached by π0, sub-goals (sgi) generated by π<sup>1</sup> and actually achieved goals (agi). In order to train our agents, we use DDPG in combination with HER. The low level agent is therefore easier to train, as it needs to reach goals closer to its current state. The interesting part is the training of the high level agent, resulting in π1: as HER allows for replacement of the actual goal g with some achieved goal ag, it makes training more efficient by learning which sub-goal can make π<sup>0</sup> reach which goal.

This method of training intrinsically generates a form of curriculum learning as the overall agent will learn more and more complex tasks along the way. In fact, it will first explore its surroundings, then learn how to generate useful sub-goals for a given goal and finally put both together to solve tasks.

## *B. Implementation*

For our experiments we used three robotic environments from OpenAI gym robotics suite [19]: FetchPush, FetchPickAndPlace and HandManipulateBlock. These are simulations based on MuJoCo [20]. All of them are goal oriented setups, in both Fetch environments the agent is a robotic arm based on the Fetch robot [21] which must move a black cube to a desired goal represented by a red dot which is either on a table (FetchPush) or in the air (FetchPickAndPlace). For both, the actions are 4-dimensional with 3 continuously controlling the position of the gripper and one for opening/closing the gripper. The observations are the Cartesian positions of the gripper and the object as well as both their linear and angular velocities. In the FetchPush environment however, the gripper is always set to be closed, which forces the agent to push the cube around, making the task rely heavily on physical properties of the block and table (i.e. friction), which need to be learned by the agent. The Hand environment is a Shadow hand [22] holding a cube which must be manipulated in-hand to reach a desired orientation goal; note that in figure [5,](#page-4-0) the goal and sub-goal orientations are displayed to the right of the hand. In the HandManipulateBlock, the agent directly controls the individual joints of the hand, which makes for a much more challenging task. In this case, the actions are 20-dimensional (24 DoF, 4 of which are coupled), and the observations are the position and velocities of the 24 joints, as well as the object's Cartesian position, its rotation as a quaternion as well as linear and angular velocities [23].

Algorithm [1](#page-3-0) presents more details on DtD. Note that the sub-goal selection agent π<sup>1</sup> is in charge of initially generating sub-goals as local noisy perturbations of the current achieved goal, and gradually handing over to the actor-critic agent for sub-goals inference. This can also be done in an -greedy

#### Algorithm 1: Dot-to-Dot (DtD)

```
Initialize DDPG agent π0 (low-level) and HER buffer R
Initialize sub-goal selection agent π1
for epoch = 0, Nepochs do
   for episode = 1, M do
       Reset the environment to obtain (s0, ag0, g)
       s = s0, ag = ag0
       for n = 0, N do . N=number of sub-episodes
          Sample sg ← π1(s, g)
          for t = Tn, Tn+1 do
              Sample an action at ← π0(s||sg)
                               . || denotes concatenation
              Execute at, observe (st+1, agt+1, rt)
              s = st, ag = agt
          for t = 0, TN do
              (st, agt, sgt, at, rt, st+1, agt+1, g)
                                                store −−→ R
   for training = 0, Ntrainings do
       perform an update of π0 using R
       perform an update of π1 using R
```

## manner.

There are a few technical details that we need to address before looking at the actual implementation and the results we obtained. First of all, we made the choice to always force the last sub-goal of an episode to match the actual end-goal set at the beginning of the episode. Another possibility is to fully ignore the actual goal every time we explore, and only set sub-goals as defined above. However, experiments have shown this technique to be quite inefficient due to the fact that the agent does not even try reaching the goal and we believe this leads to inefficient training of the policy as a result. Another choice we made is that of replacing goals with achieved goals during training. This results in applying HER to sub-goals when training π1's network. In practice, consider an example with 5 sub-goals. During training we sample an episode stored in our replay buffer R, from this episode we extract: (sg0, sg1, sg2, sg3, sg4) the subgoals, (o0, o1, o2, o3, o4) and (ag0, ag1, ag2, ag3, ag4) the corresponding observations and achieved goals, and g the goal. In classic experience replay training we would train the network on traces such as (sg2, o2, g). Instead, similarly to HER, we choose to replace g with a later achieved goal ag for a certain proportion of traces. For example, we could replace g with ag<sup>3</sup> and train on (sg2, o2, ag3) instead, virtually making the trace successful as ag<sup>3</sup> was reached by definition. This proved very efficient while training and will be used in all following experiments.

# IV. RESULTS AND ANALYSIS

## *A. Training performance*

Figures [3a](#page-3-1) and [3b](#page-3-1) show the evolution of the success rate of DDPG, HER and DtD over epochs, on the FetchPush and FetchPickAndPlace tasks respectively. We note that DtD is marginally slower than HER at training on the easier task FetchPush while being marginally faster on the more difficult one FetchPickAndPlace. Meanwhile, DDPG

<span id="page-3-1"></span>(b) Success rates during training for FetchPickAndPlace

Fig. 3: Training curves for DDPG, HER and DtD agents on the Fetch tasks. The curves represent the mean and the confidence intervals are 25th to 75th percentiles.

did not succeed at the tasks in the given number of epochs. Training was done on five random seeds, the figures show the mean and confidence intervals on these seeds.

## *B. Task performance*

In this section we present a few still frames of episodes using the best policy obtained after training. We are interested in assessing whether or not the sub-goals generated by the high-level agent are meaningful and make sense in terms of positioning.

We first present the results obtained on the Fetch environments. where the end goal is represented by a red dot, while sub-goals are represented by a green dot. We look at configurations with only one sub-goal as the table is rather small, note that the last sub-goal is forced to be the endgoal which is why the red dot turns green in the last frames (Figures [4c](#page-3-2) and [1d\)](#page-0-0).

In both environments, figures [4a](#page-3-2) and [1a](#page-0-0) show the sub-goals generated are indeed located approximately in the middle of the initial cube's position and end-goal. We can also see that the low-level agent does indeed succeed in reaching all sub-goals and completes the task. The FetchPush example also shows that the agent managed to learn some

<span id="page-3-2"></span>(a) starting position (b) reach sub-goal (c) goal reached

Fig. 4: DtD agent on FetchPush: the agent first generates the green dot as a sub-goal using the high-level policy [\(4a\)](#page-3-2), then the low-level agent reaches that sub-goal [\(4b\)](#page-3-2), and finally the last subgoal which is now equal to the end goal [\(4c\)](#page-3-2)

<span id="page-4-0"></span>(c) sub-goal missed (d) end goal reached

Fig. 5: DtD agent on the HandManipulateBlock environment. The agent starts with the cube in an initial position [\(5a\)](#page-4-0), rotates the cube to reach the first sub-goal [\(5b\)](#page-4-0) but doesn't manage to reach the sub-goal in the first sub-episode as the sub-goal shifts to the end goal [\(5c\)](#page-4-0), this does not prevent the agent from reaching the end goal as the sub-goal generated was useful to the low level agent [\(5d\)](#page-4-0)

form of concept and representation of its environment, the most obvious one being the fact that sub-goals have to be generated on the tabletop. In fact, during training, the highlevel agent is not constrained at all in terms of sub-goal generation. As mentioned in the previous part, sub-goals can be generated anywhere in the vicinity of the initial cube's location, and therefore can even appear inside the table or in the air. However, traces that include these types of sub-goals will bear very low rewards and therefore force the agent to generate sub-goals close to the tabletop.

Figure [5](#page-4-0) shows frames of an episode on the ShadowHand simulation where a cube must be rotated to the goal orientation. We can again see that the sub-goals are generated to be on the way to the end-goal, however it is harder to observe than in the Fetch environments due to the nature of the task. As figure [5c](#page-4-0) shows, the agent did not manage to reach the first sub-goal in the given time. Despite this miss, we can see that the agent positioned the cube closer to the target anyway, as shown by the yellow side being positioned correctly. Therefore, even though the low-level agent may sometimes miss a sub-goal due to time constraints, the generated sub-goals help reach an end-goal, as shown in Figure [5d.](#page-4-0)

# *C. High-level agent's inner representation*

In this part, we look at the way the high-level agent values different regions of the environment as candidates for the low-level agent's sub-goals. This allows us to interpret the way the agent represents the various environments internally, and makes it easier for humans to read into the decisionmaking process of the agent, improving explainability. We use a specific configuration of the FetchPush environment, where the initial position and the goal have been chosen to be at opposite corners of the table. The idea is to look at the table from above with the robot north of the table, and discretize both its X and Y axes. We then define sub-goals as pairs of the discrete axes: sg = (x, y). Finally, for each of these sub-goals, we compute π1's Q-values Qπ<sup>1</sup> (o, sg, g) which is the expected value of choosing sg as a sub-goal in the given configuration. A low value means that the sub-goal is not a good candidate, and the overall episode will yield a low cumulated reward.

In figure [6,](#page-4-1) we show the two extremes in terms of distance from the initial state to the end-goal. The FetchPush environment is ideal to look at inner representations as it is almost two dimensional (goals are always on the tabletop). Figure [6a](#page-4-1) shows this setup. We first look at the Q-values at the very beginning of training, after just one epoch, as shown in figure [6b.](#page-4-1) We can see the values are very close to each other and spread in a small interval, which shows that the high-level agent does not have a clear representation of the environment yet. Despite this lack of clear representation, the agent seems to attribute higher values to sub-goals closer to the end-goal (to the right of the table) rather than those close to the starting position (to its left). This makes sense

<span id="page-4-1"></span>Fig. 6: Setup with initial state and goal diagonally opposed on the table. The heatmaps show the highest areas (yellow) of value for the high-level agent to predict a sub-goal. The black square represents the position of the cube, the red circle is the end goal

as sub-goals close to the end-goal are most likely to allow the agent to reach its destination, and are therefore the very first sub-goals that lead to successful traces.

Finally, after training, the best policy's Q-values are represented in figure [6c.](#page-4-1) We can now see that the values are spread over a much larger interval, and therefore the higher values mean the associated sub-goals are clearly better candidates. This area of higher value is located well in the middle between the starting position and the end-goal. Note that in practice, the sub-goal will only be the point with the highest value on this heatmap. We can therefore conclude that the agent learnt a good representation of its environment as well as a notion of distance, considering the effects of friction and the dynamics involved in pushing a block around to an end-goal.

# V. CONCLUSIONS

We set out to create an agent that can learn to complete tasks in environments that are challenging by nature: high dimensional and only presenting sparse rewards. We also aimed at finding a structure that equips the agent with the ability to create a representation of its environment that can be easily understood by humans. We achieved this by combining several techniques to produce the Dot-to-Dot algorithm, learning a hierarchical structure of motion and manipulation through curriculum learning. This was implemented and tested through OpenAI Gym and MuJoCo, with the Fetch Robotics Manipulator and the Shadow Hand environments.

In terms of training times we obtained results equivalent to the current baselines, however we have shown that on top of this, we achieved to provide the agent with the ability to produce interpretable representations of its environment. The agent learnt a notion of distance, being able to create waypoints to an end-goal, splitting a complex task into several easier consecutive ones and reusing learnt behaviour across these. We believe this can serve as a fundamental first step to help make robotic agents intelligent while preserving the explainability of their actions.

Future work will focus on improving exploration for subgoals in the vicinity of a current position, one solution for this could be to use intrinsic motivation and curiosity [24], [25]. Another lead could be to produce more goals that do not necessarily need to be achieved, leading the agent towards a direction instead of having waypoints. Finally, we are interested in testing the algorithm on a real robotic system. This method complements the work we presented in [8], making that robotic setup a good candidate for realworld use of Dot-to-Dot.

## REFERENCES

- [1] H. Tanaka, K. Ohnishi, H. Nishi, T. Kawai, Y. Morikawa, S. Ozawa, and T. Furukawa, "Implementation of bilateral control system based on acceleration control using fpga for multi-dof haptic endoscopic surgery robot," *IEEE Transactions on Industrial Electronics*, vol. 56, no. 3, pp. 618–627, March 2009.
- [2] A. Bauer, D. Wollherr, and M. Buss, "Human–robot collaboration: a survey," *International Journal of Humanoid Robotics*, vol. 5, no. 01, pp. 47–66, 2008.

- [3] J. Kober, J. A. D. Bagnell, and J. Peters, "Reinforcement learning in robotics: A survey," *International Journal of Robotics Research*, July 2013.
- [4] S. Levine, C. Finn, T. Darrell, and P. Abbeel, "End-to-end training of deep visuomotor policies," *The Journal of Machine Learning Research*, vol. 17, no. 1, pp. 1334–1373, 2016.
- [5] F. Guenter, M. Hersch, S. Calinon, and A. Billard, "Reinforcement learning for imitating constrained reaching movements," *Advanced Robotics*, vol. 21, no. 13, pp. 1521–1544, 2007.
- [6] S. Gu, E. Holly, T. Lillicrap, and S. Levine, "Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates," in *Robotics and Automation (ICRA), 2017 IEEE International Conference on*. IEEE, 2017, pp. 3389–3396.
- [7] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning," in *Proceedings of the 26th Annual International Conference on Machine Learning*, ser. ICML '09. New York, NY, USA: ACM, 2009, pp. 41–48.
- [8] A. Shafti, P. Orlov, and A. A. Faisal, "Gaze-based, context-aware robotic system for assisted reaching and grasping," in *2019 IEEE International Conference on Robotics and Automation (ICRA)*, 2019.
- [9] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, "Continuous control with deep reinforcement learning," *CoRR*, vol. abs/1509.02971, 2015.
- [10] M. Andrychowicz, D. Crow, A. K. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, and W. Zaremba, "Hindsight experience replay," in *NIPS*, 2017.
- [11] A. G. Barto and S. Mahadevan, "Recent advances in hierarchical reinforcement learning," *Discrete Event Dynamic Systems*, vol. 13, no. 4, pp. 341–379, Oct 2003.
- [12] D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller, "Deterministic policy gradient algorithms," in *Proceedings of the 31st International Conference on International Conference on Machine Learning - Volume 32*, ser. ICML'14. JMLR.org, 2014, pp. I–387–I–395.
- [13] R. S. Sutton and A. G. Barto, *Introduction to Reinforcement Learning*, 1st ed. Cambridge, MA, USA: MIT Press, 1998.
- [14] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529 EP –, Feb 2015.
- [15] P. Dayan and G. E. Hinton, "Feudal reinforcement learning," in *Advances in neural information processing systems*, 1993, pp. 271– 278.
- [16] A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jaderberg, D. Silver, and K. Kavukcuoglu, "Feudal networks for hierarchical reinforcement learning," *CoRR*, vol. abs/1703.01161, 2017. [Online]. Available:<http://arxiv.org/abs/1703.01161>
- [17] O. Nachum, S. S. Gu, H. Lee, and S. Levine, "Data-efficient hierarchical reinforcement learning," in *Advances in Neural Information Processing Systems*, 2018, pp. 3307–3317.
- [18] A. Levy, R. P. Jr., and K. Saenko, "Hierarchical actor-critic," *CoRR*, vol. abs/1712.00948, 2017.
- [19] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, "Openai gym," 2016.
- [20] E. Todorov, T. Erez, and Y. Tassa, "Mujoco: A physics engine for model-based control," in *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, Oct 2012, pp. 5026–5033.
- [21] Fetch. [Online]. Available: [https://fetchrobotics.com/](https://fetchrobotics.com/robotics-platforms/) [robotics-platforms/](https://fetchrobotics.com/robotics-platforms/)
- [22] A. Kochan, "Shadow delivers first hand," *Industrial robot: an international journal*, vol. 32, no. 1, pp. 15–16, 2005.
- [23] M. Plappert, M. Andrychowicz, A. Ray, B. McGrew, B. Baker, G. Powell, J. Schneider, J. Tobin, M. Chociej, P. Welinder, V. Kumar, and W. Zaremba, "Multi-goal reinforcement learning: Challenging robotics environments and request for research," *CoRR*, vol. abs/1802.09464, 2018.
- [24] P.-Y. Oudeyer, J. Gottlieb, and M. Lopes, *Intrinsic motivation, curiosity, and learning: Theory and applications in educational technologies:*, 07 2016, vol. 229.
- [25] C. Colas, P. Fournier, O. Sigaud, and P. Oudeyer, "CURIOUS: intrinsically motivated multi-task, multi-goal reinforcement learning," *CoRR*, vol. abs/1810.06284, 2018.