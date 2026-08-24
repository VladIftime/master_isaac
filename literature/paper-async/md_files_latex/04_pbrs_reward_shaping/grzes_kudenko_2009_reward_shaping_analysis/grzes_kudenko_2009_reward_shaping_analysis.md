## Theoretical and Empirical Analysis of Reward Shaping in Reinforcement Learning

Marek Grześ and Daniel Kudenko

Department of Computer Science, University of York, Heslington, YO10 5DD, York, UK

{grzes,kudenko}@cs.york.ac.uk

## **Abstract**

Reinforcement learning suffers scalability problems due to the state space explosion and the temporal credit assignment problem. Knowledge-based approaches have received a significant attention in the area. Reward shaping is a particular approach to incorporate domain knowledge into reinforcement learning. Theoretical and empirical analysis of this paper reveals important properties of this principle, especially the influence of the reward type, MDP discount factor, and the way of evaluating the potential function on the performance.

#### 1. Introduction

Reinforcement learning (RL) is a machine leaning paradigm for solving problems in which solutions require a sequence of decisions. These decisions should be made in such a way so that some notion of a long-term reward is maximised. Thus, to solve a RL task means to find a policy which maps environment states to actions the agent ought to execute. The agent is given only an immediate reward for each action executed in the world and has to optimise the long-term score for a sequence of actions. This is known as the *temporal credit assignment problem* which makes RL time consuming [1].

The principle idea to improve the performance of machine learning techniques in general is to reduce the hypothesis space [2]. RL is a simulation-based technique where the policy is estimated from samples obtained from the simulated or real environment. The key research challenge in the RL community is how to reduce learning complexity, that is, the number of suboptimal actions in the environment required to estimate the policy. Different directions have been investigated in the area. For example, the representational bias reduces the hypothesis space to the set of solutions which can be learned with the reduced representation [3] or the procedural bias focuses the exploration process, i.e., how the agent acts in the world during learning, towards preferred regions of the state space [4]. In both cases, the bias can be in the form of either soft or hard constraints.

In this paper reward shaping is considered as a way of incorporating the procedural bias into RL algorithms. In standard circumstances RL algorithms learn only from the environment reward which refers only to the last action executed in the world. The idea of reward shaping is to provide

an additional external reward which does not change the optimal solution but which guides the agent during learning in a more controlled fashion. Reward shaping constitutes a particular method to incorporate background knowledge into RL. Different types of knowledge obtained in different ways and represented differently can be used with reward shaping [4], [5]. However, the general idea is the same. Reward shaping uses some heuristic assessment of how good or bad particular states in the environment are. Having this in mind, one can see RL with heuristic knowledge given to the agent (e.g., via reward shaping) as *informed reinforcement learning* where the difference between informed RL and uninformed RL is analogous to informed and uninformed search in artificial intelligence [6].

The underlying mathematical model of RL is the Markov Decision Process (MDP). One of the elements of the formal definition of MDPs is the discount factor,  $0 \le \gamma \le 1$ , which determines how rewards are regulated, i.e., how the long-term reward is calculated from immediate rewards. This originally comes from economic models where the same payoff has different utility now than when received in the future [7]. The discount factor is thus important and represents a part of the specification of a particular domain.

This paper conducts theoretical and empirical analysis of potential-based reward shaping. In particular, the influence of different reward models, values of the discount factor, and ways of evaluating the potential function on learning with reward shaping is investigated.

## 2. Reward Shaping

Reward shaping is a promising way of mitigating the negative impact of the temporal credit assignment problem. The idea of modifying the reward has been attempted many times in the past [8], [9]. However, without theoretically analysed solutions some attempts did not work as expected. A classical example is the RL agent which learns how to ride a bicycle. In this case, wrongly defined reward shaping caused the agent to ride in circles instead of directing it to the goal [9]. A significant advancement in formalising reward shaping was the development of the potential-based reward shaping, F(s,s'), which is evaluated as the difference of some potential function,  $\Phi$ , of two consecutive states, i.e., a source state s and a destination state s' [10], [11]:

$$F(s, s') = \gamma \Phi(s') - \Phi(s), \tag{1}$$

where  $\gamma$  is a discount factor. Ng et al. [10] proved that reward shaping defined in this way leaves the optimal behaviour unchanged while the time for attempting suboptimal actions can be reduced. Progress estimators in [12] are very similar to the potential function and represent good early findings about desired properties of reward shaping.

Ng et al. [10] noted that  $\Phi(s)=V(s)$  is a particularly convenient potential function because the value function in the process M' with such reward shaping is  $V_{M'}(s)\equiv 0$  which is a particularly easy V-function to learn. All that would remain would be to learn non-zero Q-values since the model of the world is not available. The potential function which satisfies  $\Phi(s)=V(s)$  is named a v-equivalent potential function in the remainder of the paper.

The potential function is inherently a heuristic function. Without any loss of generality we focus in this paper on the potential function estimated as the straight line distance, d(s), from a given state, s, to the goal state. This kind of the heuristic function is also common in informed search [6]. The potential function should be higher for states which are closer to the goal according to the heuristic. Thus, one can define the non-decreasing potential function as either a positive,  $\Phi^+$ , or negative,  $\Phi^-$  potential function, where  $\Phi^+(s) = [\max_{s' \in S} d(s')] - d(s)$  and  $\Phi^-(s) = -d(s)$ . The way the potential function is evaluated represents one of the dimensions of our analysis.

## 3. Reward and the Discount Factor

For a thorough introduction to MDPs the reader is referred to the relevant literature [7]. Two particular elements of the formal description of MDPs are treated as two additional dimensions of our analysis. The reward model and the discount factor are the intrinsic elements of both the formal definition of MDPs and the specification of the problem being modelled. The reward function, R(s, a, s'), is the immediate reward received when action a when taken in state s results in a transition to state s'. The reward function can have different character. Some reward models can be very sparse (e.g., in games very often the reward is given only at the end of the game with +1 for winning and -1 for loosing the game [13]) or very dense when the non-zero reward is given for each action. Without loss of generality, in this paper we consider two general types of the reward function which allow extending our findings to different specific characteristics of the reward model. In the first instance, we assume a sparse reward where the positive reward,  $R_q = 1$ , is given only upon entering the goal state. The second case, deals with non-positive step reward,  $R_s = -1$ , which has a meaning of the action cost. The discount factor,  $\gamma$ , is inextricably associated with the reward model. For example, in episodic tasks with  $R_q$  the discount factor,  $\gamma$ , should satisfy  $\gamma < 1$ .

## 4. Running Examples and Algorithms

In this section domains and algorithms which are used in the experimental part of our analysis are described. In order to have a controlled impact of domain properties on tested algorithms, three artificial tasks are evaluated. In the first domain, the impact of the path length, N, can be easily analysed without interference of other factors. Additionally, admissible heuristic functions have different quality in tested domains which yields additional dimension of our analysis. The first domain is the random walk (RW) task, which is worth considering because the straight line distance to the goal is very close to the v-equivalent potential function for this process (equation  $\Phi(s) = V(s)$  is satisfied in RW when actions are deterministic). There are N states in this domain. The agent starts in the most left position, and has to reach the most right position. There are two stochastic actions, left and right, which can fail with probability 0.2. The second domain is a particular variant of the maze task which can be found in [14]. In this domain, which we name Maze in the remainder of the paper, the heuristic which we are using is of medium quality. It is heading the agent towards the goal, however it does not take obstacles into account. The last domain is a maze which has been commonly used in the RL literature [1, Figure 9.5] and is named S-maze throughout this paper. In our case, a bigger version is used. Each grid position from the base configuration is uniformly divided into 64 squares yielding 72×48 states. There are eight actions which lead to an adjacent cell if it is not the border nor an obstacle. In such situations actions do not have any effect. Actions are stochastic. With probability 0.2 an action can fail in which case another outcome is chosen with a uniform probability. The straight line distance heuristic is the most inaccurate (when comparing to two previous domains) in this case since backtracking with a sequence of steps is required.

The SARSA algorithm is used as the experimental framework [1]. The  $\epsilon$ -greedy exploration is used with  $\epsilon=0.3$  in the first episode and decreasing linearly to 0.01 in the last episode. The learning rate,  $\alpha$ , starts with 0.1 in the first episode and is decreased linearly to 0.01 in the last episode. Other parameters are given with the description of a particular analysis. Unless explicitly specified, the Q-table is initialised to the value of 0.

# 5. Positive and Negative Potential Functions and $\gamma = 1$

The goal of this section is to check the influence of negative and positive potential functions when  $\gamma=1$  and also to analyse whether modified impact of reward shaping (i.e.,  $F(s,s')=\tau(\gamma\Phi(s')-\Phi(s))$ ) where  $\tau>0$ ) can influence the learning rate. Results from these experiments serve as an introduction to the theoretical and empirical analysis of Section 6. Because  $\gamma=1$  here, only the step reward,  $R_s$ , can be used.

Figure 1. Results on RW-64 with positive and negative potential functions.

Figure 2. Results on S-maze with positive and negative potential functions.

The first analysis compares learning rates of the SARSA algorithm with positive and negative potential functions. Results on three tested domains showed that negative and positive potential functions have exactly the same performance which is also better than SARSA. Results for the RW-64 and S-maze tasks are reported in Figures 1 and 2 (graphs in this section show the cumulative reward of the agent as a function of the episode number).

The next question we are asking is whether we can perform better assuming the same potential function as above which is very close to the v-equivalent potential function in RW and more faulty in two other domains. In this experiment the shaping reward is scaled with the multiplicative factor,  $\tau$ , in the range of  $0.1-10^4$ . Since two types of the potential function work the same in this configuration, for each domain one type of the potential function was tested. The value of  $\tau$  is reported in graphs with results ( $\tau = 1$  reflects the standard not scaled shaping reward). Figure 3 shows results on RW-64. It can be observed that the relative reduction of the shaping reward decreases performance. Scaling up led to improvement with reference to not scaled shaping. The algorithm reaches saturation point where further increasing of  $\tau$  did not bring further improvement. The performance was also not decreased by high values of  $\tau$ . Figure 4 shows scaling results on the Maze domain. For lower values of  $\tau$ , results show the same pattern as in RW-64. The performance with  $\tau < 1$  is lower than with neutral  $\tau = 1$ . Higher values of  $\tau$  show

Figure 3. Results on RW-64 with a negative potential function and scaling of the shaping reward.

Figure 4. Results on Maze with a positive potential function and scaling of the shaping reward.

improvement, however here the value of  $\tau = 2$  is the best whereas all higher tested values reduce performance. This fact can be explained by the quality of the potential function. In RW-64, this function is very close to the vequivalent function therefore even very high values of  $\tau$ did not hurt the performance. The potential function in Maze is more faulty, thus for very high scaling it leads to lower results. In S-maze heuristic is more inaccurate and the high values of  $\tau$  lead to even worse consequences (see Figure 5). Results for  $\tau = 50$  and 100 indicate that initial episodes were very long. Furthermore, with  $\tau = 10^4$  infinite trajectories were encountered and the agent was not able to reach the goal. This situation is caused by the faulty heuristic function which leads to dead ends and long backtracking sequences are required to change the direction of search. With very high values of the scaling factor,  $\tau > 50$ , the influence of the shaping reward becomes very strong. The shaping reward overshadows the reward received from the environment. Overall, the theory of potential-based reward shaping [10], [11] indicates convenient properties of the vequivalent potential function. However, the RL agent faces

Figure 5. Results on S-maze with a positive potential function and scaling of the shaping reward.

also the problem of exploration, and the higher values of the shaping reward (scaled up with  $\tau>1$ ) have positive influence on exploration as it was reported in experiments discussed in this paragraph. However, it should be noted that in all three investigated domains the heuristics, even though they may be faulty, still contain useful information. In the case of completely misleading heuristics (e.g., a heuristic which always prefers the longest path in the shortest path problem), any reward shaping would decrease performance.

# 6. Positive and Negative Potential Functions and $\gamma < 1$

This section investigates what kind of problems can be encountered when learning with positive and negative potential functions in environments in which  $\gamma < 1$ .

## 6.1. The Potential Function, Discount Factor, and the Actual Shaping Reward

In this section a detailed analysis is conducted to investigate how a different notion of the potential function (i.e., whether it is positive or negative),  $\Phi$ , influences the actual shaping reward which is given to the agent. Specifically, this analysis is conducted with respect to three types of environment transitions in MDP. When assuming that states s and s' are two states in the environment for which  $\Phi(s) < \Phi(s')$ , the agent should be rewarded (not penalised) by the shaping reward for transition  $s \to s'$  and not rewarded (penalised) by the shaping reward for  $s' \to s$ . Additionally, transitions  $s \to s$  should not be rewarded. Thus, the following properties will be investigated:

$$F(s,s') = \gamma \Phi(s') - \Phi(s) \ge 0, \tag{2}$$

$$F(s',s) = \gamma \Phi(s) - \Phi(s') < 0, \tag{3}$$

$$F(s,s) = \gamma \Phi(s) - \Phi(s) \le 0. \tag{4}$$

Without loss of generality it is enough to assume that the potential function is a linear function with discrete values in  $\mathcal{C}$  and  $|\Delta\Phi|=1$  for any pair of adjacent states. This type of the potential function is named an *additive potential function* in our further discussion. The straight line distance to the goal, d(s), can be considered as additive potential when  $\Phi^-(s)=-\lfloor d(s)\rfloor$  for the negative potential function and the positive potential function evaluated in the analogous way.

#### 6.1.1. Positive Potential Function

In the first instance the additive potential function and its impact on properties shown in Equations 2, 3 and 4 is investigated. In this case, if n represents the potential function of state s, the potential function for state s' is n+1. Thus, for additive positive potential the following quantities represent three types of transitions which we consider in our analysis:  $F(s,s') = \gamma(n+1) - n$ ,  $F(s',s) = \gamma n - (n+1)$  and  $F(s,s) = \gamma n - n$ , where  $n \in \mathcal{N}$ . From Equations 2, 3 and 4 and simple algebraic transformations we obtain accordingly:

$$\gamma \ge \frac{n}{n+1}, n \le \frac{\gamma}{1-\gamma},\tag{5}$$

$$\gamma \le \frac{n+1}{n}, n \ge \frac{1}{\gamma - 1},\tag{6}$$

$$\gamma < 1, n > 0. \tag{7}$$

When additive positive potential is used, it is enough to assume with no loss of generality that the minimum value of n is 0 for the most distant state from the goal and nobtains the maximal value in the goal state. In this case, from Equation 5 it can be read, that transitions  $s \rightarrow s'$ which happen close to the goal state will be negatively rewarded when  $n > \gamma/(1-\gamma)$  (n increases when moving towards the goal). For lower values of n, the positive reward will be given as required by Equation 2. This relationship shows that in the case of long trajectories (high n) the value of  $\gamma$  should be correspondingly high. If, for example, the maximum value of n = 1000, then  $\gamma > 0.999$ . And analogously, for example, for  $\gamma = 0.9$  the maximum value of n, which implies the maximum length of the trajectory, is  $n \leq 9$ . If this conditions are violated, the negative shaping reward will be given for those transitions  $(s \rightarrow s')$  in this case) which should be positively reward according the the potential function  $\Phi$ . Transitions  $s' \to s$  and  $s \to s$ do not impose any constraints on n and  $\gamma$  as shown in Equations 6 and 7 accordingly. Therefore transitions  $s' \to s$ and  $s \rightarrow s$  are never positively rewarded. They are always penalised regardless of the value of n and  $\gamma$  as required by Equations 3 and 4.

1. Our analysis can be naturally extended to the full continuous case.

## 6.1.2. Negative Potential Function

Firstly, the additive negative potential function and its impact on properties shown in Equations 2, 3 and 4 is investigated. In this case, if -(n+1) represents the potential function of state s, the potential function for state s' is -n. Thus, for the additive negative potential function the following quantities represent three types of transitions which are considered in our analysis:  $F(s,s') = \gamma(-n) + (n+1)$ ,  $F(s',s) = \gamma(-n-1) + n$  and  $F(s,s) = \gamma(-n) + n$ , where  $n \in \mathcal{N}$ . From Equations 2, 3 and 4 and simple algebraic transformations we obtain accordingly:

$$\gamma \le \frac{n+1}{n}, n \ge \frac{1}{\gamma - 1},\tag{8}$$

$$\gamma \ge \frac{n}{n+1}, n \le \frac{\gamma}{1-\gamma},\tag{9}$$

$$\gamma \ge 1. \tag{10}$$

Without loss of generality it is enough to assume that the maximum value of -n is 0 for the goal state and -nobtains the minimal value for the most distant state from the goal state when the additive negative potential function is considered. In this case, from Equation 8 it can be read, that transitions  $s \to s'$  are always positively rewarded as it is required. Here, problems arise with conditions expressed by Equation 9. In this case, when moving further from the goal state (i.e., when n grows), transitions  $s' \rightarrow s$  start to be positively rewarded whereas they are required to be always non-positively rewarded. For lower values of n, that is those close to the goal state, the negative reward will be appropriately given. But, when moving away from the goal state, those transitions start to be positively rewarded. This relationship and specifically Equation 9 show that in the case of long trajectories (high n) the value of  $\gamma$  should be correspondingly high. These conditions mirror what has been found for the additive positive potential function in the previous subsection. Here,  $s' \to s$  start to be positively rewarded when far from the goal state (high n), and in the previous case  $s \rightarrow s'$  receive a negative reward when the trajectory is long (high n) and close to the goal state.

## 6.1.3. Positive and Negative Potential Functions

The theoretical analysis presented in two previous subsections is summarised in Table 1. The most important outcomes of these results can be summarised as follows:

- The additive positive potential function poses problems for transitions  $s \to s'$  when close to the goal state (high values of n).
- With the additive negative potential function, transitions
   s → s' are always properly rewarded, whereas s' → s
   may be positively rewarded when far from the goal
   state. Additionally, the shaping reward for s → s is
   always positive when γ < 1.</li>

Figure 6. Results on RW-32 with  $\gamma=0.95,\,R_{step},$  and positive and negative potential functions.

## 6.2. An Empirical Comparison of Positive and Negative Potential Functions

The previous subsection demonstrates analytical results on the properties of reward shaping when  $\gamma < 1$  and the potential function can be both positive and negative. Now, empirical analysis is performed to verify the theoretical findings and further investigate the problem. Because of the discounting in experiments in this section, results in graphs contain the number of episode steps as a function of the episode number. This presentation yielded the most legible charts in this configuration.

## **6.2.1.** Evaluation with the Step Reward $R_s$

In the first instance, RW was tested with  $\gamma=0.95$  and with a different length, N, in the range of  $2^3 - 2^7$ . Two example runs are reported in Figures 6 and 7. In the first case, in Figure 6 the positive potential function performs worse than negative. On RW-8 both potential functions obtain similar speedup and the growing length of RW showed a decrease in the performance of the positive potential function (Table 1 shows that good transitions start to be negatively rewarded when N grows). Thus, with growing N the score of the positive potential function becomes closer to the no shaping baseline. However, the performance of the negative potential function does not remain superior. The experiment with N=40 (see Figure 7) captures the situation when learning with the negative potential function (though very good initially) starts going into long trajectories after around 200 episodes. In this run, it was still able to reach the goal state even though trajectories are already significantly longer. For higher values of the RW length, N, the negative potential did not converge at all (unfinished trajectories with millions of steps).

Two additional questions can, therefore, be asked in this situation to further analyse the problem: 1) why the positive potential function is weaker than negative, and 2) why the negative potential function does not converge on longer RWs (N>40).

The first question is explained by Table 1, but it may

|                    |                 | Actual reward                 |                                  |  |
|--------------------|-----------------|-------------------------------|----------------------------------|--|
| Transition         | Expected reward | Positive potential            | Negative potential               |  |
| $s \rightarrow s'$ | $F(s,s') \ge 0$ | F(s,s') < 0 close to the goal | always $F(s, s') \ge 0$          |  |
| $s' \rightarrow s$ | $F(s',s) \le 0$ | always $F(s',s) \leq 0$       | far from the goal $F(s', s) > 0$ |  |
| $s \rightarrow s$  | $F(s,s) \le 0$  | always $F(s,s) \leq 0$        | $F(s,s) > 0$ when $\gamma < 1$   |  |

Table 1. The influence of the type of the additive potential function and the discount factor,  $\gamma$ , on the actual shaping reward when conditions are violated.

Figure 7. Results on RW-40 with  $\gamma=0.95,\,R_{step},$  and positive and negative potential functions.

| $\Phi^+$ | F(s,s') | F(s',s) | $\Phi^-$ | F(s,s') | F(s',s) |
|----------|---------|---------|----------|---------|---------|
| 0        | 0.9     | -1      | -16      | 2.5     | 0.6     |
| 1        | 0.8     | -1.1    | -15      | 2.4     | 0.5     |
| 2        | 0.7     | -1.2    | -14      | 2.3     | 0.4     |
| 3        | 0.6     | -1.3    | -13      | 2.2     | 0.3     |
| 4        | 0.5     | -1.4    | -12      | 2.1     | 0.2     |
| 5        | 0.4     | -1.5    | -11      | 2       | 0.1     |
| 6        | 0.3     | -1.6    | -10      | 1.9     | 0       |
| 7        | 0.2     | -1.7    | -9       | 1.8     | -0.1    |
| 8        | 0.1     | -1.8    | -8       | 1.7     | -0.2    |
| 9        | 0       | -1.9    | -7       | 1.6     | -0.3    |
| 10       | -0.1    | -2      | -6       | 1.5     | -0.4    |
| 11       | -0.2    | -2.1    | -5       | 1.4     | -0.5    |
| 12       | -0.3    | -2.2    | -4       | 1.3     | -0.6    |
| 13       | -0.4    | -2.3    | -3       | 1.2     | -0.7    |
| 14       | -0.5    | -2.4    | -2       | 1.1     | -0.8    |
| 15       | -0.6    | -2.5    | -1       | 1       | -0.9    |
| 16       |         |         | 0        |         |         |

Table 2. Shaping rewards from positive and negative potential functions on RW-16 with  $\gamma=0.9$ .

not be easy to observe this fact there. For this reason Table 2 shows shaping rewards for both positive and negative potential functions. When moving towards the goal state (the bottom row in Table 2), the shaping reward for  $s \to s'$  is decreasing when the potential function is positive and its absolute value is additionally lower than in the case of the negative potential function. It means that the shaping reward resulting from the positive potential function is lower. Results were improved when the shaping reward in this case was scaled with  $\tau > 1$ . For  $\tau = 2$  results were significantly improved and with  $\tau = 3$  the result was as good as with the negative potential function on RW-16.

Figure 8. Results on Maze with  $\gamma = 0.95$ ,  $R_{step}$ , and positive and negative potential functions.

Table 2 helps also in explaining why the algorithm with the negative potential function does not converge on long RWs. In this case, transitions far from the goal state receive high positive shaping rewards in both directions. This reward is constantly growing when going away from the goal state. It means that for both good transitions,  $s \to s'$ , and wrong transitions,  $s' \to s$ , Q-values become higher than zero and cause the agent to mistakenly reinforce those values by following such loopy paths which involve those transitions far from the goal state. This happens because the initial value of Q(s,a)=0 of all state-action pairs represents the highest possible value only with reward r<0. Anything higher than 0 will be mistakenly preferred.

This configuration of  $R_s$  and positive and negative potential functions was tested also on Maze and S-maze domains. Results presented in Figures 8 and 9 show the same pattern in the performance where the positive potential function is significantly inferior to the negative representation. Tests with different values of  $\gamma$  also show the same pattern as in RW when appropriately changing the length of RW under a constant discount factor. In this case, the negative potential function did not converge when  $\gamma$  was too small (e.g.,  $\gamma=0.95$  on S-maze).

## **6.2.2.** Evaluation with the Goal Reward $R_q$

The first series of experiments is on RW with different values of N. Results with lower values of N are not reported in graphs. For example, on RW-8 two potential functions yield the same speedup. On RW-16 the negative potential function performs worse than either positive and the SARSA

Figure 9. Results on S-maze with  $\gamma=0.99,\,R_{step},$  and positive and negative potential functions.

Figure 10. Results on RW-128 with  $\gamma=0.95,\,R_{goal},\,$  and the positive potential function only (negative does not converge).

baseline, though initially it is better than SARSA. For N=32 and higher, the negative potential function does not converge leading to infinite trajectories. The positive potential function becomes worse as well when N increases. One experiment for N=128 is presented in Figure 10. It shows typical behaviour of the positive potential function which is much better initially when the no-shaping approach performs random exploration but later on it is significantly worse than learning without shaping.

Experiments on Maze show similar properties. Figure 11 shows results with  $\gamma=0.95$ . The positive potential function is again better only initially and has longer episodes than no-shaping after around 100 episodes. The negative potential function was unstable with this value of  $\gamma$ . A more detailed view of this run is in the internal part of Figure 11, which shows the first 300 episodes. The negative potential function, even though good initially, goes into long trajectories and stabilises again after around 3000 episodes. In not reported results with  $\gamma=0.8$  it did not converge at all, and with  $\gamma=0.99$  the graph is similar as in Figure 11.

The S-maze task was the most challenging for reward shaping in this configuration (graphs are not included in the

Figure 11. Results on Maze with  $\gamma=0.95,\,R_{goal},\,{\rm and}$  positive and negative potential functions.

paper). Learning with the negative potential function did not converge for any of tested  $\gamma$  values, i.e., 0.9, 0.95, and 0.99. The positive potential function performs in a similar way as with other domains. Its performance drops when the discount factor decreases and with  $\gamma=0.8$  it performs significantly worse even in early episodes.

The lack of convergence of learning (i.e. infinite episodes) with the negative potential function was encountered in this section as well. The explanation of this problem which was given in Section 6.2.1 applies also to this configuration. The initialisation of the Q-table with a value higher than zero was required and allowed the agent to reach the goal on both RW and S-maze.

One more solution was investigated to solve the problem of the lack of convergence of the negative potential function. The treatment of transitions  $s \to s$  was changed, i.e., F(s,s)was manually set to 0 for all states whereas everything else was left unchanged. We expected that this modification would allow the agent to avoid infinite trajectories in the problematic situation. It did not however. But, the positive potential function encountered problems with F(s, s) = 0. For example, on S-maze with  $\gamma = 0.95$  infinite trajectories arise after around 900 episodes. Our more detailed analysis revealed that when F(s,s) = 0 for each state, then there is no penalty for rebounding to the same state. This can happen when there is a wall in front of the agent and the move forward action will always fail. Transitions in such states can cause negative values of the Q-function. When this happens, the temporal difference for transitions  $s \to s$ will be positive and the agent will wrongly prefer executing actions which cause  $s \to s$ . For this reason F(s,s) should not be manually set to 0 but rather left as a negative value given by the potential function. This results in a natural penalty for transitions  $s \to s$  (see Table 1).

## 7. Conclusion

This paper presents novel theoretical and empirical insight into RL with reward shaping that every RL practitioner should be aware of. The overall contribution of this paper can be summarised as follows:

- When  $\gamma=1$ , the potential function can be both positive and negative and in both cases the performance is exactly the same.
- Even when  $\Phi(s) = V(s)$ , the learning algorithm still needs to learn effects of actions and for this reason scaling the shaping reward up  $(\tau > 1)$  improves the learning rate, because the exploration is improved.
- In domains with faulty heuristics one can not scale the
  potential function up too much because the agent may
  be heavily penalised for diverging from the shaping
  reward and this may result in failures in reaching the
  goal state. However the scaling factor, τ, with the value
  of 2 yielded best results on all tested domains with
  different quality of the heuristic function and can be
  considered in practical applications.
- The analysis of the actual shaping reward in domains with  $\gamma < 1$  was conducted (summary in Table 1) and results allow explaining the outcomes of the empirical analysis.
- When  $\gamma < 1$  and learning with  $R_s$ , the positive potential function performs worse than the negative one and the scaling factor  $\tau > 1$  improves learning with the positive potential function. The negative potential function is better in the initial configuration but breaks when conditions defined in Equations 8-10 are significantly violated. This can happen even with a very accurate heuristic function as shown on RW (see Figure 7). The re-initialisation of the Q-table to higher values (e.g., 100) allows avoiding infinite episodes.
- The goal reward,  $R_g$ , seems to be more challenging to reward shaping. Generally, both types of the potential function lead to a considerable improvement only at the very beginning of learning, when the no-shaping agent performs random exploration. For higher lengths of RW (e.g., N=128) or generally situations when conditions in Equations 2-4 are violated to a higher extent, the positive potential function, even though good initially, is significantly worse than no shaping. The negative potential function leads in these cases to a lack of convergence, and a different initialisation of the Q-table is required to avoid infinite episodes.
- Additional analysis of the previous case revealed that transitions s → s should be rewarded according to the standard evaluation of the shaping reward, F(s,s), because these transitions should be constantly penalised (Table 1).

Our findings do not violate the relevant theory on potential-based reward shaping [10], [11]. Since this kind of reward shaping has the equivalent initialisation, the same problems can be encountered with the corresponding initialisation of the value function.

The work presented in this paper is a part of a bigger ongoing project. Our aim is to propose in the near future solutions to problems reported in this paper. The theory is an important element of our current work on these improvements.

## Acknowledgment

This research was sponsored by the United Kingdom Ministry of Defence Research Programme.

## References

- [1] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. MIT Press, 1998.
- [2] T. M. Mitchell, Machine Learning. McGraw-Hill, 1997.
- [3] T. G. Dietterich, "Hierarchical reinforcement learning with the MAXQ value function decomposition," *Journal of Artificial Intelligence Research*, vol. 13, pp. 227–303, 2000.
- [4] J. Asmuth, M. L. Littman, and R. Zinkov, "Potential-based shaping in model-based reinforcement learning," in *Proc. of AAAI Conference on Artificial Intelligence*, 2008.
- [5] M. Grzes and D. Kudenko, "Plan-based reward shaping for reinforcement learning," in *Proc. of the 4th IEEE International Conference on Intelligent Systems (IS'08)*. IEEE, 2008, pp. 22–29.
- [6] S. J. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach (2nd Edition)*. Prentice Hall, 2002.
- [7] M. L. Puterman, Markov Decision Processes: Discrete Stochastic Dynamic Programming. New York, NY, USA: John Wiley & Sons, Inc., 1994.
- [8] V. Gullapalli and A. G. Barto, "Shaping as a method for accelerating reinforcement learning," in *Proc. of the 1992 IEEE International Symposium on Intelligent Control*, 1992, pp. 554–559.
- [9] J. Randløv and P. Alstrom, "Learning to drive a bicycle using reinforcement learning and shaping," in *Proc. of the 15th International Conference on Machine Learning*, 1998, pp. 463–471.
- [10] A. Y. Ng, D. Harada, and S. J. Russell, "Policy invariance under reward transformations: Theory and application to reward shaping," in *Proc. of the 16th International Conference* on Machine Learning, 1999, pp. 278–287.
- [11] E. Wiewiora, "Potential-based shaping and q-value initialisation are equivalent," *JAIR*, vol. 19, pp. 205–208, 2003.
- [12] M. J. Mataric, "Reward functions for accelerated learning," in *In Proc. of the 11th International Conference on Machine Learning*, 1994, pp. 181–189.
- [13] G. J. Tesauro, "TD-gammon, a self-teaching backgammon program, achieves master-level play," *Neural Computation*, vol. 6, no. 2, pp. 215–219, 1994.
- [14] M. Grzes and D. Kudenko, "Learning shaping rewards in model-based reinforcement learning," in *Proc. of AAMAS* 2009 Workshop on Adaptive Learning Agents (ALA'09), 2009.