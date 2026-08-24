# 3D FlowMatch Actor: Unified 3D Policy for Singleand Dual-Arm Manipulation

Nikolaos Gkanatsios†,<sup>1</sup> Jiahe Xu†,<sup>1</sup> Matthew Bronars<sup>1</sup> Arsalan Mousavian<sup>2</sup> Tsung-Wei Ke⋆,<sup>3</sup> Katerina Fragkiadaki<sup>1</sup> <sup>1</sup>Carnegie Mellon University <sup>2</sup>NVIDIA <sup>3</sup>National Taiwan University <https://3d-flowmatch-actor.github.io/>

Abstract: We present 3D FlowMatch Actor (3DFA), a 3D policy architecture for robot manipulation that combines flow matching for trajectory prediction with 3D pretrained visual scene representations for learning from demonstration. 3DFA leverages 3D relative attention between action and visual tokens during action denoising, building on prior work in 3D diffusion-based single-arm policy learning. Through a combination of flow matching and targeted system-level and architectural optimizations, 3DFA achieves over 30× faster training and inference than previous 3D diffusion-based policies, without sacrificing performance. On the bimanual PerAct2 benchmark, it establishes a new state of the art, outperforming the nextbest method by an absolute margin of 41.4%. In extensive real-world evaluations, it surpasses strong baselines with up to 1000× more parameters and significantly more pretraining. In unimanual settings, it sets a new state of the art on 74 RLBench tasks by directly predicting dense end-effector trajectories, eliminating the need for motion planning. Comprehensive ablation studies underscore the importance of our design choices for both policy effectiveness and efficiency.

# 1 Introduction

Single-arm manipulation has achieved great success in handling long-horizon and high-precision tasks [\[1,](#page-9-0) [2,](#page-9-0) [3\]](#page-9-0), even in highly cluttered environments [\[4,](#page-9-0) [5\]](#page-9-0). However, the lack of coordination between multiple end-effectors largely constrains single-arm systems to simple pick-and-place tasks, making them inadequate for addressing the more complex and diverse manipulation challenges encountered in real-world daily tasks. To overcome these challenges, bimanual systems offer a promising alternative by enabling more dexterous and coordinated interactions with the environment [\[6\]](#page-9-0).

Although bimanual setups improve the ability of a robot to perform more intricate and dexterous tasks, they also impose stricter demands on spatiotemporal precision. Both arms must operate in a tightly coordinated manner, executing actions in the correct temporal sequence and at precisely aligned spatial locations. This added complexity makes learning effective manipulation policies more difficult than in the single-arm case. Despite growing interest, existing approaches [\[7,](#page-9-0) [8,](#page-9-0) [9,](#page-9-0) [10,](#page-9-0) [11\]](#page-9-0) still fall short of achieving robust generalization across a wide range of tasks.

In parallel, recent advances in single-arm manipulation have demonstrated the power of diffusion models in capturing multimodal behaviors [\[12,](#page-9-0) [13,](#page-9-0) [14,](#page-9-0) [15\]](#page-9-0), achieving high-precision action prediction through 3D scene understanding [\[16,](#page-9-0) [17,](#page-9-0) [18,](#page-9-0) [19\]](#page-10-0) and impressive generalization to various tasks and language instructions [\[20,](#page-10-0) [21,](#page-10-0) [22\]](#page-10-0). A natural next step toward robust bimanual manipulation is to integrate these advances. In fact, we show that an adaptation of 3D Diffuser Actor (3DDA) [\[12\]](#page-9-0) a model that combines diffusion-based action generation with 3D scene representations - already establishes a new state of the art on the bimanual manipulation benchmark PerAct2 [\[10\]](#page-9-0). Next, we ask the question: what prevents 3DDA from being deployed in real-world bimanual manipulation scenarios? Our experimentation suggests two key bottlenecks: slow inference speed and long training

<sup>†</sup>Equal contribution, <sup>⋆</sup>Work done at CMU

time. For instance, on PerAct2, 3DDA requires approximately 21 days to train and operates at 0.5Hz during inference. The prolonged training time significantly limits the model's ability to adapt to new tasks, while the slow inference speed makes it unsuitable for real-time or dynamic task execution.

In light of these observations, we introduce 3D FlowMatch Actor (3DFA), a significantly more efficient extension of 3DDA that improves both training time and inference speed by an order of magnitude. On the PerAct2 bimanual manipulation benchmark [10], 3DFA reduces the training time from 21 days to 16 hours and increases the inference speed from 0.5Hz to 18.2Hz, without sacrificing performance. To enable faster inference, we replace the DDPM-based [23] formulation used in 3DDA with a Flow Matching approach [24, 25], reducing the number of denoising steps during inference from 100 to 5. To reduce training overhead, we implement a series of system-level optimizations for computational efficiency, such as faster token sampling, fewer camera inputs, optimized data loading, efficient attention implementation and mixed-precision training. While none of these techniques is novel on its own, our contribution lies in carefully integrating them into a manipulation system.

3DFA achieves a new state of the art on the PerAct2 simulation benchmark, with a success rate of 85.1%. Furthermore, it outperforms strong baselines—including  $\pi_0$  [20]—both in simulation and on a real-world 10-task benchmark we constructed using the bimanual ALOHA platform [26]. We conduct an extensive ablation study to break down the contributions of each design choice.

Notably, 3DFA is a general-purpose framework capable of predicting both sparse keyposes and dense end-effector trajectories, and is applicable to both unimanual and bimanual manipulation. On the 18 single-arm tasks of PerAct [16], 3DFA is trained to predict the next end-effector keypose and performs on par with 3DDA, while requiring significantly less training and inference time. Furthermore, on the 74-task benchmark of [27], 3DFA excels at jointly predicting the next keypose and the trajectory connecting it to the current pose in a single forward pass, outperforming the best baseline by 7.3%. These results highlight the versatility and efficiency of 3DFA as a framework for 3D manipulation.

In summary, our contributions are: (1) Adaptation of state-of-the-art single-arm 3D generative policies to bimanual manipulation, (2) Dramatic acceleration of training and inference time in 3D policies, (3) State-of-the-art bimanual manipulation results on PerAct2, with an absolute margin of 41.4% over  $\pi_0$ , (4) State-of-the-art bimanual manipulation results in the real world, outperforming foundational policies in a direct comparison, (5) State of the art on the HiveFormer unimanual 74-task benchmark, demonstrating planner-free trajectory prediction capabilities.

#### 2 3D FlowMatch Actor

The architecture of 3DFA is shown in Figure 1. It is a robot policy for single- and dual-arm manipulation that generates 3D end-effector trajectories for one or more robot arms conditioned on the task instruction, proprioception history and scene visual information, through iterative denoising. 3DFA grounds visual and action tokens to 3D locations described in a common coordinate frame, using calibration information to transfer visual tokens from the camera frame to the robot's frame.

3DFA builds upon the state-of-the-art single-arm 3D diffusion policy, 3DDA [12], which we extend to also solve bimanual tasks. To accelerate inference and training, we replace the original DDPM-based diffusion mechanism with flow matching, adopt faster point sampling methods and attention implementations, and build a highly-efficient data loading pipeline.

We denote demonstrations as sequences of observations and actions  $\{(\mathbf{o}_1, \mathbf{a}_1), (\mathbf{o}_2, \mathbf{a}_2), \ldots\}$ , accompanied by a task language instruction l, where  $\mathbf{o}_t$  denotes the visual observation and  $\mathbf{a}_t$  the robot action at timestep t. Each observation  $\mathbf{o}_t$  consists of one or more posed RGB-D images. Each action  $\mathbf{a}_t$  is a single-arm end-effector pose and is decomposed into 3D location, rotation and binary (open/close) state:  $\mathbf{a}_t = \{\mathbf{a}_t^{\mathrm{loc}} \in \mathbb{R}^3, \mathbf{a}_t^{\mathrm{rot}} \in \mathbb{R}^6, \mathbf{a}_t^{\mathrm{open}} \in \{0,1\}\}$ . Let  $\tau_t = (\mathbf{a}_{t:t+T}^{\mathrm{loc}}, \mathbf{a}_{t:t+T}^{\mathrm{rot}})$  denote the trajectory of 3D locations and rotations at timestep t, with a temporal horizon T. At each timestep t, our model predicts one of more trajectories  $\tau_t$  and binary states  $\mathbf{a}_{t:t+T}^{\mathrm{open}}$ . We first review Flow Matching in Section 2.1 and the 3DDA architecture in Section 2.2. We then detail our extensions to support bimanual manipulation and the design choices to enhance efficiency in Section 2.3.

<span id="page-2-0"></span>Figure 1: **Top:** 3DFA is a flow-matching policy built atop 3D Diffuser Actor [12]. It encodes the visual scene o, robot proprioception for left and right arms  $c_L$ ,  $c_R$ , and noised trajectories  $\tau_L^i$ ,  $\tau_R^i$  into 3D tokens. Given language tokens l, diffusion step i, and these 3D tokens, 3DFA predicts velocity fields  $v_{\theta,L}$  and  $v_{\theta,R}$  for the left and right arms, respectively. **Bottom:** During inference, 3DFA iteratively predicts straight-line velocity fields pointing toward the left- and right-hand target poses.

### 2.1 Flow Matching for Fast Action Generation

Diffusion models [23, 28] are powerful generative frameworks for modeling multimodal data distributions. They generate samples by iteratively removing Gaussian noise via a stochastic process defined by conditional probability transitions. In contrast, flow matching approaches [25, 24] generate data by solving an optimal transport problem between a source  $\mu_0$  and a target distribution  $\mu_1$ .

In particular, we adopt Rectified Flow [24], an instantiation of flow matching that transforms a sample  $X_0 \sim \mu_0$  into  $X_1 \sim \mu_1$  by following straight paths in the sample space. This significantly reduces computation during inference while retaining the expressiveness needed for high-dimensional generation, making it especially well-suited for robotics, where real-time performance is critical. The rectified flow between  $(X_0, X_1)$  defines a continuous, time-differentiable trajectory  $\mathbf{Z} = Z_t : t \in [0, 1]$  that transports  $X_0$  to  $X_1$  and is governed by the ordinary differential equation (ODE):

$$dZ_t = v_t^X(Z_t)dt, \quad t \in [0, 1], \quad Z_0 = X_0,$$
 (1)

where  $v_t^X$  is a time-dependent velocity field. The optimal velocity field that minimizes the expected discrepancy from the straight-line path between  $X_0$  and  $X_1$  can be formally defined as:  $\inf_v \int_0^1 \mathbb{E}[\|X_1 - X_0 - v(X_t, t)\|] dt$ , where  $X_t = (1-t)X_0 + tX_1$  denotes the linear interpolation at time t [24]. The model is trained to approximate this velocity field by minimizing the loss [24]:

$$\mathcal{L}_{\theta} = \mathbb{E}_{t,X}[\|v_{\theta}(X_t, t) - v_t^X\|^2]$$
(2)

In our setting, the goal is to generate robot actions from noise. We define  $\mu_0 \sim \mathcal{N}(0, I)$  and  $\mu_1$  as the distribution over real actions. During inference, the model iteratively transforms the noise sample  $X_0$  into a predicted action  $X_1$  over N steps with a fixed step size  $\Delta t = \frac{1}{N}$ :  $X_{t+\Delta t} = X_t - \Delta t \cdot v_\theta(X_t, t)$ .

#### 2.2 3D Diffuser Actor

3DDA [12] is a 3D diffusion policy for manipulation that learns from demonstrations how to predict end-effector trajectories. By framing trajectory generation as a denoising process, 3DDA learns to

<span id="page-3-0"></span>iteratively refine noisy trajectory samples into clean end-effector motions, conditioning on multi-view RGB-D observations, language instructions and proprioception history. 3DDA conditions on 3D scene feature representations derived from posed cameras and sensed depth and uses DDPM-based diffusion to predict the noise component at each diffusion step. It distills 2D foundational features on point clouds to describe the scene and uses calibration information to transform the positional encodings of the 3D visual tokens into the robot's frame and to fuse them with action tokens that eventually decode the end-effector's translation and rotation at each future timestep.

To simplify notation, we denote  $\tau^i$  as the noisy trajectory estimate at diffusion step i without specifying the trajectory timestep. The model conditions on the following inputs: (1) 3D scene tokens: 3DDA featurizes image views using a 2D image encoder and lifts each of the feature patches to 3D by calculating the average 3D location of each patch; (2) 3D proprioception tokens: 3DDA contextualizes a set of learnable embeddings with 3D scene tokens based on the proprioceptive location; (3) 3D trajectory tokens: 3DDA maps each noisy action  $\mathbf{a}^i$  of trajectory  $\tau^i$  at diffusion step i to a latent feature vector and lifts these feature vectors to 3D, based on the noisy location estimate of  $\mathbf{a}^i$ ; (4) language tokens: language instructions are encoded to latent embeddings with a text encoder. 3DDA fuses all 3D tokens using relative 3D attentions, and additionally fuses language tokens using standard attention. As a last step, the refined trajectory tokens are passed through MLPs to predict the noise added to  $\tau^0$ , as well as the end-effector opening. We refer to [12] for more details.

#### 2.3 3D FlowMatch Actor

We detail the core elements of 3DFA, as changes we have made over the model of [12].

**Extension to Bimanual Manipulation:** We first redefine the robot action in a bimanual form:  $\mathbf{a}_{t,L}$  and  $\mathbf{a}_{t,R}$  denote the robot action at timestep t, of the left and right robot arm respectively. Our goal is to predict the corresponding trajectory  $\tau_{t,L} = (\mathbf{a}_{t:t+T,L}^{\mathrm{loc}}, \mathbf{a}_{t:t+T,L}^{\mathrm{rot}})$  and  $\tau_{t,R} = (\mathbf{a}_{t:t+T,R}^{\mathrm{loc}}, \mathbf{a}_{t:t+T,r}^{\mathrm{rot}})$  of temporal horizon T for both arms. To apply 3DFA on unimanual manipulation setups, we still use the unimanual trajectory definition:  $\mathbf{a}_t$  and  $\tau_t = (\mathbf{a}_{t:t+T}^{\mathrm{loc}}, \mathbf{a}_{t:t+T}^{\mathrm{rot}})$ . We follow the same 3D tokenization procedure to map (1) the noisy estimate of pose  $\mathbf{a}_L^i$  of  $\tau_L^i$  and  $\mathbf{a}_R^i$  of  $\tau_R^i$  at denoising step i, and (2) the left- and right-hand proprioceptive information  $c_L$  and  $c_R$ , into 3D tokens. We use the same 3D Relative Denoising Transformer architecture to contextualize all these tokens and predict the translation and rotation noise and the end-effector opening for both arms.

Flow Matching Action Prediction Objective: We replace the DDPM-based diffusion method with rectified flow. The noisy left- and right-hand trajectory estimate  $\tau_L^i = (1-i)\epsilon_L + i\tau_{t,L}^0$  and  $\tau_R^i = (1-i)\epsilon_R + i\tau_{t,R}^0$  become the linear interpolation at denoising step i, where  $\tau_{t,L}^0$  and  $\tau_{t,R}^0$  denote the clean trajectory, and  $\epsilon_L$  and  $\epsilon_R$  denote the sampled noise for the left- and right-hand end effector. In particular, our model takes as input two-hand trajectory estimate tokens  $\tau^i = \{\tau_L^i, \tau_R^i\}$ , proprioception tokens  $\mathbf{c} = \{c_L, c_R\}$ , language tokens l, and scene tokens l, it outputs the left- and right-hand velocity field  $v_{\theta,L}, v_{\theta,R}$  and gripper openness  $f_{\theta,L}^{\mathrm{open}}, f_{\theta,L}^{\mathrm{open}}$ , respectively. We ignore time step t to simplify notations.

During training, we sample a time step t uniformly, denoising step  $i \sim \sigma(\mathcal{N}(0,I))$  from a logit-normal distribution and noise  $\epsilon_L \sim \mathcal{N}(0,I), \epsilon_R \sim \mathcal{N}(0,I)$  from Gaussian distribution. We use the L2 loss to supervise the velocity field and binary cross-entropy (BCE) to supervise the end-effector opening. By ignoring the notation of time step t, the objective reads:

$$\mathcal{L}_{\theta} = \|\epsilon_{L} - \boldsymbol{\tau}_{L}^{0} - v_{\theta,L}(\mathbf{o}, l, \mathbf{c}, \boldsymbol{\tau}^{i}, i)\|^{2} + \|\epsilon_{R} - \boldsymbol{\tau}_{R}^{0} - v_{\theta,R}(\mathbf{o}, l, \mathbf{c}, \boldsymbol{\tau}^{i}, i)\|^{2}$$

$$+ \operatorname{BCE}(f_{\theta,L}^{\text{open}}(\mathbf{o}, l, \mathbf{c}, \boldsymbol{\tau}^{i}, i), \mathbf{a}_{1:T,L}^{\text{open}}) + \operatorname{BCE}(f_{\theta,R}^{\text{open}}(\mathbf{o}, l, \mathbf{c}, \boldsymbol{\tau}^{i}, i), \mathbf{a}_{1:T,R}^{\text{open}}),$$
(3)

**Accelerating Dataloading:** We replace 3DDA's data loaders and optimize the keypose sampling during batching, data type conversion, and unprojection and augmentation operations. Specifically, we change the episodic loading to random keypose sampling. In more detail, the 3DDA codebase loads entire episodes, chunks them and concatenates chunks from different episodes to form a batch. We, on the other hand, sample keyposes across random episodes. To efficiently do this, we used the

Python library zarr, to lazily load and access data indices across all episodes simultaneously. This offers the following advantages: i) we avoid loading whole episodes at once to only use a chunk, as this wastes time for data that is not used; ii) we ensure higher diversity in every batch; iii) we ensure a fixed batch size, contrary to 3DDA that concatenates chunks of possibly different sizes.

We handle data types to ensure faster batch collation. Specifically, we make sure to always load uint8 images and half-precision depth maps. This significantly speeds up batch formation, especially when large batch sizes are used. The data is converted to the desired type (float32) after being loaded to the GPU, where data-type conversions are much faster. In contrast, 3DDA loads and batches float32 tensors. We found that handling data types cuts down the loading time by half.

Lastly, we move depth unprojection to point cloud and augmentations to GPU. This offers two advantages: i) loading single-channel depth is much faster than three-channel point clouds; ii) it allows for faster, batched operations that are optimized on GPUs.

Faster point sampling We adopt density-biased sampling (DBS) [\[29\]](#page-10-0) to replace farthest point sampling (FPS) [\[30\]](#page-10-0). In more detail, 3DDA employs FPS in the feature space to sparsify the scene tokens. FPS maintains a set of candidate points and a set of sampled points. Then, it iteratively samples the candidate point with maximum average distance from all sampled points. We replace this with DBS, which first estimates the sparsity of a neighborhood around a point as the average distance of the k = 8 nearest neighbors of that point. Then, it promotes sampling in the sparser neighborhoods. A fast batched version of DBS can be implemented in pure PyTorch [\[31\]](#page-10-0).

Mixed-precision training: We allow for autocasting operations to half precision when possible. This reduces the memory footprint of our model and allows for larger batch sizes.

Efficient attention implementation: We use a modern PyTorch implementation of attention that runs optimized C++ code under the hood and is faster by an order of magnitude, especially when combined with large batch sizes and half precision.

CUDA graph compilation: This technique is offered in modern PyTorch versions. The model is compiled as a graph of non-dynamic operations, allowing for optimizing all operations. Making 3DFA compilable required refactoring changes over 3DDA, such as removal of logic branches, CPU operations, in-graph language tokenization and custom kernel operations such as FPS.

# 3 Experiments

We evaluate 3DFA on learning manipulation behaviors from demonstration in simulation and the real world. Specifically, we test 3DFA on the PerAct2 bimanual manipulation benchmark (Section 3.1), on the PerAct (Section [3.2\)](#page-6-0) and HiveFormer (Section [3.3\)](#page-6-0) unimanual manipulation benchmarks, and on a suite of 10 real-world bimanual tasks using the Aloha robot platform (Section [3.4\)](#page-7-0).

### 3.1 Evaluation on the PerAct2 Bimanual Manipulation Benchmark

PerAct2 [\[10\]](#page-9-0) is a simulation benchmark for bimanual manipulation using a dual-arm setup with two Franka Emika Panda robots. The benchmark includes a suite of 13 bimanual tasks, each with 1 to 5 variations involving changes in object pose, appearance, or semantics (see Section [C.1](#page-15-0) for details). For each task, the dataset provides 100 demonstrations for training and 100 episodes for evaluation. Methods are trained to predict the next end-effector keypose [\[10\]](#page-9-0), which is then passed to an RRT-based motion planner [\[35\]](#page-11-0) to generate a feasible joint trajectory from the current configuration to the predicted pose. PerAct2 supports five calibrated RGB-D cameras—front, left wrist, right wrist, left shoulder, and right shoulder—all capturing images at a resolution of 256 × 256. We find that using only the front, left wrist and right wrist is sufficient for all tasks. Task success rate is used as the primary evaluation metric.

|                                                                                                                 | multi-task<br>training | Avg.<br>Success                        | push<br>box                               | lift<br>ball           | push<br>buttons                                   | pick up<br>plate             | put item<br>into drawer                          | put bottle<br>into fridge        |
|-----------------------------------------------------------------------------------------------------------------|------------------------|----------------------------------------|-------------------------------------------|------------------------|---------------------------------------------------|------------------------------|--------------------------------------------------|----------------------------------|
| ACT [8]                                                                                                         | Х                      | 5.9                                    | 0                                         | 36                     | 4                                                 | 0                            | 13                                               | 0                                |
| RVT-LF [18, 10]                                                                                                 | X                      | 10.5                                   | 52                                        | 17                     | 39                                                | 3                            | 10                                               | 0                                |
| PerAct-LF [16, 10]                                                                                              | ×                      | 17.5                                   | 57                                        | 40                     | 10                                                | 2                            | 27                                               | 0                                |
| PerAct <sup>2</sup> [10]                                                                                        | ×                      | 16.8                                   | 6                                         | 50                     | 47                                                | 4                            | 10                                               | 3                                |
| DP3 [15]                                                                                                        | ×                      | -                                      | 56                                        | 64                     | -                                                 | -                            | -                                                | -                                |
| KStarDiffuser [32]                                                                                              | X                      | -                                      | 83                                        | 98.7                   | -                                                 | -                            | -                                                | -                                |
| PPI [33]                                                                                                        | X                      | -                                      | 96.7                                      | 89.3                   | -                                                 | -                            | 79.7                                             | -                                |
| AnyBimanual [34]                                                                                                | /                      | 32                                     | 46                                        | 36                     | 73                                                | 8                            | -                                                | 26                               |
| $\pi_0$ -keypose [20]                                                                                           | /                      | 43.7                                   | 93                                        | 97                     | 38                                                | 41                           | 40                                               | 22                               |
| 3DFA (ours)                                                                                                     | ✓                      | 85.1                                   | $92.7_{\pm 0.47}$                         | $99.7_{\pm 0.47}$      | <b>92.7</b> $_{\pm 1.89}$                         | <b>69.7</b> <sub>±12.6</sub> | $93.0_{\pm 2.83}$                                | <b>89.3</b> <sub>±1.89</sub>     |
|                                                                                                                 |                        |                                        |                                           |                        |                                                   |                              |                                                  |                                  |
|                                                                                                                 | multi-task             | handover                               | pick up                                   | straighten             | sweep                                             | lift                         | handover                                         | take tray                        |
|                                                                                                                 | multi-task<br>training | handover<br>item                       | pick up<br>laptop                         | straighten<br>rope     | sweep<br>dust                                     | lift<br>tray                 | handover<br>item (easy)                          | take tray<br>out of oven         |
| ACT [8]                                                                                                         |                        | l                                      |                                           | _                      |                                                   |                              | l                                                | out of oven                      |
| ACT [8]<br>RVT-LF [18, 10]                                                                                      |                        | item                                   | laptop                                    | rope                   | dust                                              | tray                         | item (easy)                                      | out of oven  2 3                 |
| RVT-LF [18, 10]<br>PerAct-LF [16, 10]                                                                           |                        | item 0                                 | laptop 0                                  | rope<br>16             | dust 0                                            | tray 6                       | item (easy)                                      | out of oven 2 3 8                |
| RVT-LF [18, 10]                                                                                                 |                        | 0<br>0                                 | laptop 0 3                                | rope 16 3              | 0<br>0                                            | tray 6 6                     | item (easy) 0 0                                  | out of oven  2 3                 |
| RVT-LF [18, 10]<br>PerAct-LF [16, 10]                                                                           |                        | 0<br>0<br>0                            | laptop                                    | rope 16 3 21           | 0<br>0<br>0<br>28                                 | tray 6 6 14                  | 0<br>0<br>0<br>9                                 | out of oven 2 3 8                |
| RVT-LF [18, 10]<br>PerAct-LF [16, 10]<br>PerAct <sup>2</sup> [10]                                               |                        | 0<br>0<br>0                            | 1aptop 0 3 11 12                          | rope 16 3 21           | 0<br>0<br>0<br>28<br>0                            | tray 6 6 14                  | 0<br>0<br>0<br>9<br>41                           | out of oven 2 3 8                |
| RVT-LF [18, 10]<br>PerAct-LF [16, 10]<br>PerAct <sup>2</sup> [10]<br>DP3 [15]                                   |                        | 0<br>0<br>0<br>11<br>-                 | 0<br>3<br>11<br>12<br>6.3<br>43.7<br>46.3 | rope 16 3 21           | dust 0 0 28 0 1.7                                 | tray 6 6 14                  | 0<br>0<br>9<br>41<br>0                           | out of oven 2 3 8                |
| RVT-LF [18, 10]<br>PerAct-LF [16, 10]<br>PerAct <sup>2</sup> [10]<br>DP3 [15]<br>KStarDiffuser [32]             | training  X X X X X X  | 0 0 0 11 15                            | 1aptop  0 3 11 12 6.3 43.7 46.3 7         | rope  16 3 21 24 24 24 | 0<br>0<br>28<br>0<br>1.7<br>89<br>98.7<br>67      | tray  6 6 14 1 - 92 14       | 0<br>0<br>9<br>41<br>0<br>27<br>62.7<br>44       | 2<br>3<br>8<br>9<br>-<br>-<br>24 |
| RVT-LF [18, 10]<br>PerAct-LF [16, 10]<br>PerAct <sup>2</sup> [10]<br>DP3 [15]<br>KStarDiffuser [32]<br>PPI [33] | training  X X X X X X  | 0<br>0<br>0<br>11<br>-<br>-<br>15<br>2 | laptop  0 3 11 12 6.3 43.7 46.3 7 27      | rope  16 3 21 24       | 0<br>0<br>28<br>0<br>1.7<br>89<br>98.7<br>67<br>2 | tray  6 6 14 1 - 92          | 0<br>0<br>9<br>41<br>0<br>27<br>62.7<br>44<br>59 | 2<br>3<br>8<br>9                 |

Table 1: **Evaluation on PerAct2 bimanual benchmark**. 3DFA, AnyBimanual and  $\pi_0$ -keypose are multi-task policies and evaluate one checkpoint across all tasks, while others are single-task policies and report results from the best checkpoint per task. **3DFA outperforms all prior arts by a large margin.** 

Baselines We compare our method against several strong baselines, including: ACT [8], RVT-LF [18, 10], PerAct-LF [16, 10], and PerAct<sup>2</sup>[10] (results taken from [10]); AnyBimanual[34]; 3D Diffusion Policy (DP3)[15] and KStarDiffuser[32] (results reported in [32]); PPI [33]; and  $\pi_0$ [20]. Notably,  $\pi_0$  is a 2D robot policy pretrained on 10,000 hours of robot demonstration data and capable of both unimanual and bimanual manipulation. It takes visual input from three camera views (front and wrists) and outputs future joint angle trajectories. We adapt  $\pi_0$  to predict 3D end-effector keyposes, which is the standard output format used across all policies in this setting and significantly improves its performance. We call this variant  $\pi_0$ -keypose. Our method, along with  $\pi_0$ -keypose and AnyBimanual, uses *multitask training* and evaluates performance using the final checkpoint, whereas other baselines train separate models per task and report results from the best intermediate checkpoint for each task. Second, several baselines—including AnyBimanual, DP3, PPI, and KStarDiffuser—are evaluated on *only a subset* of the 13 benchmark tasks. For further details on all baselines, please refer to Section C.6.

**Results** We show quantitative results in Table 1. 3DFA largely outperforms all baselines, with an absolute improvement of 68.3% over PerAct<sup>2</sup>. When isolating the same 5 tasks in which both PPI and KStarDiffuser report results, 3DFA achieves 92.3%, outperforming them by 13.6% and 24% absolute respectively. Notably, our method with 3.8M parameters outperforms  $\pi_0$  which has nearly 1000 times more parameters. We show that, although large-scale pretraining is useful, explicitly incorporating 3D information into the model is a strong inductive bias. We also test 3DDA on this benchmark, which also achieves an average success rate of 85%. We discuss failure cases in Section C.10.

**Ablations** We compare different denoising methods and quantify the effect of each contribution on training time and control frequency (Figure 2). Key observations:

• **Denoising method impact**: DDPM is the least flexible, requiring 100 denoising steps to achieve 85.0% SR. DDIM [36] reduces steps but suffers a sharp performance drop. Flow

<span id="page-6-0"></span>Figure 2: Ablation study on PerAct2. Left: 3DFA's performance is stable even with as few as three denoising steps, contrary to the variants that use DDIM and DDPM. Middle: all design choices significantly contribute to lowering the training time from 504h to 16h (4 L40S GPUs). Right: Our contributions drastically improve inference speed from 0.5Hz to 18.2Hz (on one L40S GPU).

matching is the most robust, maintaining full 85.1% performance even with 5 steps, 83.9% with 3 steps, and 75.2% with just 1 step.

- Training time reduction: Starting from the original 3DDA (adapted for bimanual manipulation), replacing its data-loading scheme with ours cuts training time by two-thirds. Additional optimizations cumulatively yield a 10× speedup, for a total 30× training speedup.
- Control frequency gains: Switching from DDPM to flow matching reduces denoising steps by 20× but raises control frequency only 9×, as scene encoding dominates the forward pass. Replacing FPS with DBS halves this cost, doubling control frequency. The attention implementation has negligible effect at test time, as it is tuned for large batch sizes.

### 3.2 Evaluation on the PerAct Unimanual Manipulation Benchmark

We evaluate our method on the 18-task PerAct benchmark [\[16\]](#page-9-0) to enable a direct comparison with 3DDA [\[12\]](#page-9-0) in terms of both performance and efficiency. All policies are trained in a multi-task setting using 100 demonstrations per task and evaluated over 25 episodes per task. The setup includes four calibrated RGB-D cameras: front, wrist, left shoulder, and right shoulder (task details in Section [C.3\)](#page-17-0).

We compare two variants of 3DFA against 3DDA in Figure [3,](#page-7-0) with all models trained to predict keyposes. When using all four cameras, 3DFA achieves performance on par with 3DDA while offering a 28× faster inference speed and requiring 6× less training time. With only two cameras (front and wrist), 3DFA further reduces computational cost—14× less training time and 30× faster inference—while incurring only a minor drop in performance. Full results are provided in Section [C.7.](#page-19-0)

### 3.3 Evaluation on the HiveFormer Unimanual Manipulation Benchmark

We evaluate 3DFA on the 74-task HiveFormer benchmark [\[27\]](#page-10-0) to showcase its ability to predict dense end-effector trajectories rather than sparse keyposes. This capability enables planner-free execution, which is critical for tasks requiring continuous interaction with objects and the environment. For instance, when opening a fridge, the end-effector must follow the door's rotational arc to respect its mechanical limits—a constraint that an RRT planner does not account for.

All policies are trained on one task at a time with 100 demonstrations and evaluated on 100 test episodes. We report results across the full 74-task benchmark and also highlight performance on a subset of 8 challenging tasks [\[19,](#page-10-0) [37\]](#page-11-0) that demand continuous interaction and cannot be solved by simply predicting keyposes. Further details are provided in Sections [C.4](#page-17-0) and [C.5.](#page-18-0)

Baselines We compare against: (1) keypose-prediction models HiveFormer [\[27\]](#page-10-0), InstructRL [\[38\]](#page-11-0) and Act3D [\[17\]](#page-9-0), which use three cameras; (2) close-loop trajectory-prediction models PointFlow-Match [\[37\]](#page-11-0) and DP3 [\[15\]](#page-9-0), which use five cameras; (3) ChainedDiffuser [\[19\]](#page-10-0), a two-stage policy consisting of a keypose predictor and a keypose-conditioned trajectory predictor, using three cameras.

<span id="page-7-0"></span>Figure 3: **Single-arm manipulation results. Left**: On the PerAct 18-task benchmark, 3DFA matches 3DDA's performance while reducing training time by over 6× and achieving 28× faster inference. **Right**: 3DFA achieves a new state-of-the-art on the 74 tasks of HiveFormer.

|                      | Avg.    | Num.   | unplug  | close | open  | open   | frame off | open | books in | shoes in |
|----------------------|---------|--------|---------|-------|-------|--------|-----------|------|----------|----------|
|                      | Success | stages | charger | door  | box   | fridge | hanger    | oven | shelf    | box      |
| DP3 [15]             | 28.5    | 1      | 33.3    | 76.0  | 98.3  | 4.3    | 12.3      | 0.3  | 3.7      | 0.0      |
| PointFlowMatch [37]  | 67.8    | 1      | 83.6    | 68.3  | 99.4  | 31.9   | 38.6      | 75.9 | 68.8     | 76.0     |
| ChainedDiffuser [19] | 84.5    | 2      | 95.0    | 76.0  | 96.0  | 68.0   | 85.0      | 86.0 | 92.0     | 78.0     |
| 3DFA (ours)          | 91.3    | 1      | 99.0    | 96.0  | 100.0 | 70.0   | 96.0      | 99.0 | 99.0     | 71.0     |

Table 2: **Evaluation on 8 RLBench tasks** from [19] that require continuous interaction with the environment. 3DFA predicts dense trajectories and outperforms all baselines by a large margin.

3DFA is trained to jointly predict the next end-effector keypose and the dense trajectory until the next keypose in a non-hierarchical, single-forward-pass fashion. It relies on two camera observations, front and wrist, to maintain consistency with the rest of our experiments. More details in Section C.6.

We present results across all 74 tasks in Figure 3. 3DFA sets a new state of the art with a success rate of 90.3%, surpassing Act3D by 7.3%. On eight particularly challenging tasks (Table 2), our method outperforms the two-stage ChainedDiffuser by 6.8% and PointFlowMatch by 23.5% absolute, demonstrating its strong capability for continuous trajectory prediction. A detailed performance breakdown for all tasks is provided in Section C.8.

## 3.4 Evaluation in the Real World

We construct a challenging real-world multi-task manipulation benchmark using Mobile Aloha [39], a dual-arm mobile manipulator equipped with a front-facing ZEDX RGB-D camera and two wrist-mounted RealSense D405 RGB-D sensors. Our benchmark (Figure 4, Section C.2) comprises 10 tasks that demand precise and coordinated two-handed manipulation, examples include *open marker*, *close ziploc*, and *insert battery*, surpassing the complexity of PerAct2 tasks. We collect 40 demonstrations per task, recording visual observations and joint actions at 5 Hz. All models are trained or fine-tuned using the same demonstration set and evaluated on 20 episodes per task.

**Baselines** We compare against two strong baselines: (1)  $\pi_0$  [20], a generalist 2D manipulation policy, and (2) iDP3 [40], a variant of DP3 adapted for dual-arm humanoid systems with architectural enhancements. All models, including ours, are trained for closed-loop trajectory prediction without intermediate keypose supervision. Following their original designs, both baselines directly predict joint angles for the robot arms, whereas 3DFA predicts 3D end-effector poses, which are then converted to joint commands via inverse kinematics. To mimic human operation, all models output joint values for the Aloha leader arms, with the follower arms mirroring the motion. 3DFA and iDP3 are 3D policies that require accurate depth sensing; due to high noise from the wrist-mounted depth sensors, we exclude wrist camera inputs for both. In contrast, the 2D  $\pi_0$  policy is depth-independent and utilizes all three camera views. All image inputs are downsampled to a resolution of 256  $\times$  256.

<span id="page-8-0"></span>Figure 4: **Real-world benchmark.** Our real-world benchmark consists of 10 bimanual tasks, divided into 5 easy tasks (top row) and 5 difficult tasks (bottom row).

|                        | Avg.<br>Success |       |      | 1  |    |    |    | put marker<br>into bowl |    | 1  |    | 1  |   |
|------------------------|-----------------|-------|------|----|----|----|----|-------------------------|----|----|----|----|---|
| $\pi_0$ [20]           | 32.5            | 100ms | 3238 | 85 | 75 | 40 | 20 | 20                      | 15 | 35 | 20 | 10 | 5 |
| iDP3 <sup>†</sup> [40] | 24.5            | 420ms | 68.8 | 45 | 35 | 30 | 40 | 35                      | 25 | 15 | 10 | 10 | 0 |
| 3DFA (ours)            | 53.5            | 54ms  | 3.8  | 85 | 75 | 80 | 80 | 70                      | 60 | 45 | 25 | 10 | 5 |

Table 3: Evaluation in the real world. 3DFA and  $\pi_0$  adopt multi-task learning settings, while iDP3 uses a single-task learning setup. 3DFA outperforms  $\pi_0$  and iDP3 on all tasks by a large margin.

**Results** We show the quantitative results in Table 3. 3DFA outperforms  $\pi_0$  and iDP3 on most tasks. We find that  $\pi_0$  is sensitive to occlusion and cannot locate and reach the object if it is not visible in the wrist observations. We also find that iDP3 struggles to predict precise end-effector poses, as the robot often approaches but fails to grasp the object. In contrast, our method consistently outperforms these baselines in most tasks. We refer to Sections C.9, C.10 for more analysis.

# 3.5 Limitations and Future Directions

While 3DFA significantly reduces training and inference costs, it retains a key limitation: as a 3D policy, it depends on accurate depth sensing and camera calibration—resources often unavailable in large-scale, real-world imitation learning datasets. In particular, wrist-mounted cameras present the greatest calibration challenges. We are exploring approaches to relax these depth and calibration requirements, including architectures capable of jointly processing 2D and 3D observations [41], as well as leveraging recent advances in metric depth estimation and automatic calibration from the broader computer vision community.

The compact parameterization of 3DFA, while contributing to its efficiency, also makes it susceptible to certain errors. For instance, on PerAct, the model may struggle with unseen variations encountered at test time. In our real-world benchmark, it has difficulty with tasks demanding fine force control or extreme precision. Scaling up the model's capacity and incorporating strong vision-language models for richer visual and language understanding are promising directions for addressing these challenges.

#### 4 Conclusion

We introduced 3D FlowMatch Actor (3DFA), a fast and versatile 3D manipulation policy that combines flow matching with pretrained 3D scene representations. Through targeted architectural and system-level optimizations, 3DFA achieves over 30X faster training and inference than prior 3D diffusion-based policies, while setting new state of the art on both bimanual (PerAct2) and unimanual (RLBench-74) benchmarks. It delivers real-time performance, scales to real-world tasks, and removes the need for motion planning via direct dense trajectory prediction. These results position 3DFA as an efficient and general-purpose framework for unimanual and bimanual robot manipulation.

# <span id="page-9-0"></span>References

- [1] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. *Advances in Neural Information Processing Systems*, 36, 2024.
- [2] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison. Rlbench: The robot learning benchmark & learning environment. *IEEE Robotics and Automation Letters*, 5(2):3019–3026, 2020.
- [3] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. *arXiv preprint arXiv:2405.05941*, 2024.
- [4] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. *arXiv preprint arXiv:2403.12945*, 2024.
- [5] H.-S. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot. *arXiv preprint arXiv:2307.00595*, 2023.
- [6] C. Smith, Y. Karayiannidis, L. Nalpantidis, X. Gratal, P. Qi, D. V. Dimarogonas, and D. Kragic. Dual arm manipulation—a survey. *Robotics and Autonomous systems*, 60(10):1340–1353, 2012.
- [7] J. Grannen, Y. Wu, S. Belkhale, and D. Sadigh. Learning bimanual scooping policies for food acquisition. In *Conference on Robot Learning*, 2022.
- [8] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. *RSS*, 2023.
- [9] J. Grannen, Y. Wu, B. Vu, and D. Sadigh. Stabilize to act: Learning to coordinate for bimanual manipulation. *CoRL*, 2023.
- [10] M. Grotz, M. Shridhar, T. Asfour, and D. Fox. Peract2: A perceiver actor framework for bimanual manipulation tasks. *arXiv preprint arXiv:2407.00278*, 2024.
- [11] I.-C. A. Liu, S. He, D. Seita, and G. Sukhatme. Voxact-b: Voxel-based acting and stabilizing policy for bimanual manipulation. *CoRL*, 2024.
- [12] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. *arXiv preprint arXiv:2402.10885*, 2024.
- [13] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. *arXiv preprint arXiv:2303.04137*, 2023.
- [14] M. Reuss, M. Li, X. Jia, and R. Lioutikov. Goal-conditioned imitation learning using score-based diffusion policies. *arXiv preprint arXiv:2304.02532*, 2023.
- [15] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In *Proceedings of Robotics: Science and Systems (RSS)*, 2024.
- [16] M. Shridhar, L. Manuelli, and D. Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. In *Conference on Robot Learning*, pages 785–799. PMLR, 2023.
- [17] T. Gervet, Z. Xian, N. Gkanatsios, and K. Fragkiadaki. Act3d: 3d feature field transformers for multi-task robotic manipulation. *CoRL*, 2023.
- [18] A. Goyal, J. Xu, Y. Guo, V. Blukis, Y.-W. Chao, and D. Fox. Rvt: Robotic view transformer for 3d object manipulation. *arXiv preprint arXiv:2306.14896*, 2023.

- <span id="page-10-0"></span>[19] Z. Xian, N. Gkanatsios, T. Gervet, T.-W. Ke, and K. Fragkiadaki. Chaineddiffuser: Unifying trajectory diffusion and keypose prediction for robotic manipulation. In *Conference on Robot Learning*, pages 2323–2339. PMLR, 2023.
- [20] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π0: A vision-language-action flow model for general robot control, 2024. *arXiv preprint arXiv:2410.24164*, 2024.
- [21] Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. *arXiv preprint arXiv:2411.19650*, 2024.
- [22] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. *arXiv preprint arXiv:2501.15830*, 2025.
- [23] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. *CoRR*, abs/2006.11239, 2020. URL <https://arxiv.org/abs/2006.11239>.
- [24] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. *arXiv preprint arXiv:2209.03003*, 2022.
- [25] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.
- [26] Z. Fu, T. Z. Zhao, and C. Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation, 2024.
- [27] P.-L. Guhur, S. Chen, R. G. Pinel, M. Tapaswi, I. Laptev, and C. Schmid. Instruction-driven history-aware policies for robotic manipulations. In *6th Annual Conference on Robot Learning*, 2022. URL [https://openreview.net/forum?id=h0Yb0U\\_-Tki](https://openreview.net/forum?id=h0Yb0U_-Tki).
- [28] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*, 2020.
- [29] C. R. Palmer and C. Faloutsos. Density biased sampling: an improved method for data mining and clustering. In *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, SIGMOD '00, page 82–92, New York, NY, USA, 2000. Association for Computing Machinery. ISBN 1581132174. [doi:10.1145/342009.335384.](http://dx.doi.org/10.1145/342009.335384) URL [https:](https://doi.org/10.1145/342009.335384) [//doi.org/10.1145/342009.335384](https://doi.org/10.1145/342009.335384).
- [30] Y. Eldar, M. Lindenbaum, M. Porat, and Y. Zeevi. The farthest point strategy for progressive image sampling. *IEEE Transactions on Image Processing*, 6(9):1305–1315, 1997. [doi:10.1109/](http://dx.doi.org/10.1109/83.623193) [83.623193.](http://dx.doi.org/10.1109/83.623193)
- [31] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. Pytorch: An imperative style, high-performance deep learning library, 2019. URL [https://arxiv.org/abs/1912.](https://arxiv.org/abs/1912.01703) [01703](https://arxiv.org/abs/1912.01703).
- [32] Q. Lv, H. Li, X. Deng, R. Shao, Y. Li, J. Hao, L. Gao, M. Y. Wang, and L. Nie. Spatial-temporal graph diffusion policy with kinematic modeling for bimanual robotic manipulation. *arXiv preprint arXiv:2503.10743*, 2025.
- [33] Y. Yang, Z. Cai, Y. Tian, J. Zeng, and J. Pang. Gripper keypose and object pointflow as interfaces for bimanual robotic manipulation. *RSS*, 2025.

- <span id="page-11-0"></span>[34] G. Lu, T. Yu, H. Deng, S. S. Chen, Y. Tang, and Z. Wang. Anybimanual: Transferring unimanual policy for general bimanual manipulation. *arXiv preprint arXiv:2412.06779*, 2024.
- [35] J. J. Kuffner and S. M. LaValle. Rrt-connect: An efficient approach to single-query path planning. In *Proceedings 2000 ICRA. Millennium Conference. IEEE International Conference on Robotics and Automation. Symposia Proceedings (Cat. No. 00CH37065)*, volume 2, pages 995–1001. IEEE, 2000.
- [36] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models, 2022.
- [37] E. Chisari, N. Heppert, M. Argus, T. Welschehold, T. Brox, and A. Valada. Learning robotic manipulation policies from point clouds with conditional flow matching. *arXiv preprint arXiv:2409.07343*, 2024.
- [38] H. Liu, L. Lee, K. Lee, and P. Abbeel. Instruction-following agents with jointly pre-trained vision-language models. *arXiv preprint arXiv:2210.13431*, 2022.
- [39] Z. Fu, T. Z. Zhao, and C. Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. *arXiv preprint arXiv:2401.02117*, 2024.
- [40] Y. Ze, Z. Chen, W. Wang, T. Chen, X. He, Y. Yuan, X. B. Peng, and J. Wu. Generalizable humanoid manipulation with improved 3d diffusion policies. *arXiv preprint arXiv:2410.10803*, 2024.
- [41] A. Jain, P. Katara, N. Gkanatsios, A. W. Harley, G. Sarch, K. Aggarwal, V. Chaudhary, and K. Fragkiadaki. Odin: A single model for 2d and 3d segmentation, 2024. URL [https:](https://arxiv.org/abs/2401.02416) [//arxiv.org/abs/2401.02416](https://arxiv.org/abs/2401.02416).
- [42] Y. Mu, T. Chen, Z. Chen, S. Peng, Z. Lan, Z. Gao, Z. Liang, Q. Yu, Y. Zou, M. Xu, L. Lin, Z. Xie, M. Ding, and P. Luo. Robotwin: Dual-arm robot benchmark with generative digital twins, 2025. URL <https://arxiv.org/abs/2504.13059>.
- [43] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. *arXiv preprint arXiv:2410.07864*, 2024.
- [44] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T.-W. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023.
- [45] T. Motoda, R. Hanai, R. Nakajo, M. Murooka, F. Erich, and Y. Domae. Learning bimanual manipulation via action chunking and inter-arm coordination with transformers. *arXiv preprint arXiv:2503.13916*, 2025.
- [46] J.-J. Jiang, X.-M. Wu, Y.-X. He, L.-A. Zeng, Y.-L. Wei, D. Zhang, and W.-S. Zheng. Rethinking bimanual robotic manipulation: Learning with decoupled interaction framework. *arXiv preprint arXiv:2503.09186*, 2025.
- [47] N. Gkanatsios, A. Jain, Z. Xian, Y. Zhang, C. Atkeson, and K. Fragkiadaki. Energy-based models as zero-shot planners for compositional scene rearrangement. *arXiv preprint arXiv:2304.14391*, 2023.
- [48] H. Chen, C. Lu, C. Ying, H. Su, and J. Zhu. Offline reinforcement learning via high-fidelity generative behavior modeling, 2023.

- <span id="page-12-0"></span>[49] B. Yang, H. Su, N. Gkanatsios, T.-W. Ke, A. Jain, J. Schneider, and K. Fragkiadaki. Diffusiones: Gradient-free planning with diffusion for autonomous driving and zero-shot instruction following. *ArXiv*, abs/2402.06559, 2024.
- [50] P. Hansen-Estruch, I. Kostrikov, M. Janner, J. G. Kuba, and S. Levine. Idql: Implicit q-learning as an actor-critic method with diffusion policies. *arXiv preprint arXiv:2304.10573*, 2023.
- [51] N. Funk, J. Urain, J. Carvalho, V. Prasad, G. Chalvatzaki, and J. Peters. Actionflow: Equivariant, accurate, and efficient policies with spatially symmetric flow matching. *arXiv preprint arXiv:2409.04576*, 2024.
- [52] M. Braun, N. Jaquier, L. Rozo, and T. Asfour. Riemannian flow matching policy for robot motion learning. In *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 5144–5151. IEEE, 2024.
- [53] F. Zhang and M. Gienger. Affordance-based robot manipulation with flow matching. *arXiv preprint arXiv:2409.01083*, 2024.
- [54] H. Ding, N. Jaquier, J. Peters, and L. Rozo. Fast and robust visuomotor riemannian flow matching policy. *arXiv preprint arXiv:2412.10855*, 2024.
- [55] Q. Zhang, Z. Liu, H. Fan, G. Liu, B. Zeng, and S. Liu. Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation. *arXiv preprint arXiv:2412.04987*, 2024.
- [56] S. Wang, L. Wang, S. Zhou, J. Tian, J. Li, H. Sun, and W. Tang. Flowram: Grounding flow matching policy with region-aware mamba framework for robotic manipulation. *Conference on Computer Vision and Pattern Recognition*, 2025.
- [57] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. *arXiv preprint arXiv:2503.14734*, 2025.
- [58] M. Reuss, H. Zhou, M. Rühle, Ö. E. Yagmurlu, F. Otto, and R. Lioutikov. Flower: Democratizing ˘ generalist robot policies with efficient vision-language-action flow policies. In *7th Robot Learning Workshop: Towards Robots with Human-Level Abilities*, 2025.
- [59] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021.
- [60] S. James and A. J. Davison. Q-attention: Enabling efficient learning for vision-based robotic manipulation. *IEEE Robotics and Automation Letters*, 7(2):1612–1619, 2022.
- [61] A. Jaegle, F. Gimeno, A. Brock, A. Zisserman, O. Vinyals, and J. Carreira. Perceiver: General perception with iterative attention, 2021.
- [62] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pages 234–241. Springer, 2015.
- [63] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation, 2017. URL <https://arxiv.org/abs/1612.00593>.

# Appendix

| A                                                | Related Work                               | 15 |  |  |  |  |  |  |  |  |  |
|--------------------------------------------------|--------------------------------------------|----|--|--|--|--|--|--|--|--|--|
| B                                                | Implementation details                     |    |  |  |  |  |  |  |  |  |  |
| C<br>Additional Experimental Results and Details |                                            |    |  |  |  |  |  |  |  |  |  |
|                                                  | C.1<br>Peract2 tasks<br>                   | 16 |  |  |  |  |  |  |  |  |  |
|                                                  | C.2<br>Real-world tasks<br>                | 17 |  |  |  |  |  |  |  |  |  |
|                                                  | C.3<br>PerAct tasks                        | 18 |  |  |  |  |  |  |  |  |  |
|                                                  | C.4<br>74 HiveFormer tasks<br>             | 18 |  |  |  |  |  |  |  |  |  |
|                                                  | C.5<br>8 ChainedDiffuser tasks             | 19 |  |  |  |  |  |  |  |  |  |
|                                                  | C.6<br>Additional details on baselines<br> | 19 |  |  |  |  |  |  |  |  |  |
|                                                  | C.7<br>Detailed PerAct results<br>         | 20 |  |  |  |  |  |  |  |  |  |
|                                                  | C.8<br>Detailed HiveFormer results<br>     | 21 |  |  |  |  |  |  |  |  |  |
|                                                  | C.9<br>Detailed real-world results<br>     | 21 |  |  |  |  |  |  |  |  |  |
|                                                  | C.10 Failure cases                         | 22 |  |  |  |  |  |  |  |  |  |

# <span id="page-14-0"></span>A Related Work

Bimanual manipulation. Bimanual manipulation is challenging due to the need for precise coordination between two arms. A key bottleneck is the difficulty of collecting large-scale, high-quality bimanual demonstration data, which has historically constrained the scalability of approaches [\[7,](#page-9-0) [9\]](#page-9-0). Recent works [\[8,](#page-9-0) [26\]](#page-10-0) have introduced more cost-effective pipelines and platforms for scaling realworld data collection. However, these methods primarily rely on RGB inputs and struggle to generalize across diverse tasks, object types, scene configurations or robot embodiments. To address these limitations, several multi-task simulation benchmarks have been proposed to facilitate large-scale demonstration collection and policy evaluation. PerAct2 [\[10\]](#page-9-0) extends the RLBench [\[2\]](#page-9-0) benchmark to support multi-task bimanual manipulation, with expert demonstrations generated using sampling-based motion planners [\[35\]](#page-11-0). RoboTwin [\[42\]](#page-11-0) introduces a generative digital twin framework, leveraging 3D generative foundation models and large language models to create diverse expert datasets and real-world-aligned evaluation environments.

In parallel, the development of bimanual manipulation policies generally falls into two main categories. One line of research extends single-arm policy architectures to bimanual. For example, VoxAct-B [\[11\]](#page-9-0) builds upon the 3D voxel-based scene representations of [\[16\]](#page-9-0) to jointly predict object and end-effector poses. Similarly, RDT-1B [\[43\]](#page-11-0) follows the Vision-Language-Action (VLA) paradigm [\[44,](#page-11-0) [20,](#page-10-0) [22\]](#page-10-0), fine-tuning large language models to improve generalization to diverse task instructions. The second line of research composes multiple single-arm policies into a unified bimanual policy [\[34,](#page-11-0) [45,](#page-11-0) [46\]](#page-11-0). For instance, DIF [\[46\]](#page-11-0) observes that bimanual manipulation involves both independent and coordinated actions between the two arms, so it proposes to learn separate single-arm policies and a coordination module to combine them for coordinated tasks. Our work belongs to the first category and extends the action space of [\[12\]](#page-9-0) to predict pose trajectories for both arms simultaneously.

Diffusion and Flow Matching models in robotics. Diffusion models have emerged as powerful tools in imitation learning [\[13,](#page-9-0) [14,](#page-9-0) [47\]](#page-11-0) and offline RL [\[48,](#page-11-0) [49,](#page-12-0) [50\]](#page-12-0). DDPM [\[23\]](#page-10-0) has been the most widely adopted diffusion algorithm to iteratively add and remove Gaussian noise to and from the samples, following conditional probability paths. More recently, Flow Matching [\[24,](#page-10-0) [25\]](#page-10-0) has drawn attention in robot learning. Policies that use flow matching learn to predict the velocity field directly pointing toward the target action [\[20,](#page-10-0) [51,](#page-12-0) [52,](#page-12-0) [53,](#page-12-0) [54,](#page-12-0) [55,](#page-12-0) [37,](#page-11-0) [56,](#page-12-0) [57,](#page-12-0) [58\]](#page-12-0), resulting in better sampling efficiency and lower computational cost than DDPM-based policies. 3DFA also incorporates Flow Matching, yielding a state-of-the-art bimanual manipulation policy with real-time inference capabilities.

# B Implementation details

Architecture: We closely follow 3DDA [\[12\]](#page-9-0) in terms of architecture design. We use identical weights to 3DDA, with only the appropriate changes so that we handle and predict bimanual actions of 16 dimensions rather than 8. Specifically, this affects all layers that either project a noisy action token to a high-dimensional latent vector or predict an action token from a latent vector.

For completeness, we briefly describe the architecture here: 3DFA uses CLIP [\[59\]](#page-12-0) to encode each image separately into a feature map, then assigns 3D positions to each 2D feature token using depth and the known camera parameters. The feature maps are merged into a single feature cloud, which is subsampled using density-biased sampling (DBS) to keep 1/4th of the tokens. The distance metric for DBS is computed in the feature space, not the 3D space. We call the subsampled feature cloud "visual tokens".

The noisy trajectory estimate is encoded with a linear layer. Trajectory and visual tokens are concatenated in the sequence dimension and fed to a sequence of 6 self-attention layers. All tokens in this attention use 3D rotary positional embeddings, as used in [\[12,](#page-9-0) [17\]](#page-9-0). Additionally, these layers are modulated by a signal computed by projecting i) the current end-effector pose and ii) the denoising step, into high-dimensional features. This signal is used to compute adaptive normalization parameters.

<span id="page-15-0"></span>Lastly, MLPs are used on the output tokens to predict the translation velocity field, rotation velocity field and end-effector opening. Our model is language-conditioned, encodes the instruction with CLIP and then lets the visual and trajectory tokens attend to the language features, as in 3DDA.

Normalization of the output space: 3D relative attention requires that the positional embeddings of all action tokens and visual tokens be expressed in a common coordinate frame. The original 3DDA normalizes the 3D space using the target end-effector's statistics, which does not generalize to bimanual setups where each arm has distinct spatial distributions. Naively aggregating statistics across both arms and normalizing based on that would significantly compress the spatial distribution and harm performance. Instead, we compute positional embeddings in the global unnormalized frame, while still predicting trajectories in the per-arm normalized space. This hybrid approach preserves 3DDA's attention mechanism while accommodating the bimanual setup.

Flow hyperparameters: When training a Flow Matching model, we sample a denoising i from a continuous internal between 0 and 1, where i = 0 denotes pure noise and i = 1 indicates a perfectly clear signal. One hyperparameter is how we sample i during training. We experimented with uniform sampling, beta sampling (as in π0) and logit-normal sampling (as in [\[58\]](#page-12-0)). We found beta sampling to consistently underperform the other two options, but little difference between uniform and logitnormal. We adopt logit-normal and sample i as follows: we first sample x from a Gaussian with 0 mean and 1.5 std, then i = σ(x), where σ is the sigmoid function.

Other hyperparameters: We follow the training hyperparameters of 3DDA [\[12\]](#page-9-0), in terms of optimizer selection, learning rate, etc. We use a constant batch size of 256 keyposes (64 per GPU) for PerAct2 and PerAct, 16 trajectories (one GPU) for the HiveFormer tasks, randomly sampled from different tasks and episodes. We found that our model converges in 300000 training steps on PerAct/PerAct2, contrary to 3DDA that needs 600000 steps. We hypothesize that this is due to the optimized batching scheme we propose. For the HiveFormer, we train single-task models until convergence, which usually takes less than 100000 steps or approximately 4 hours on one GPU.

Keypose discovery We use keyposes only on RLBench, while for our real-world experiments we predict trajectories directly. For our single-arm RLBench experiments, we use the heuristics from [\[60\]](#page-12-0), while for PerAct2 we use the ones from [\[10\]](#page-9-0), that adapts those of [\[60\]](#page-12-0) for the bimanual setup. Specifically we examine each arm separately and a pose is marked as a keypose if (1) the end-effector state changes (grasp or release) or (2) the velocity magnitude approaches zero (often at pre-grasp poses or a new phase of a task). Once we compute a set keyposes for each arm, we then take the union of the two sets as the bimanual keyposes.

# C Additional Experimental Results and Details

# C.1 Peract2 tasks

We provide brief descriptions of the 13 PerAct2 tasks for completeness. All tasks vary the object pose, appearance and semantics. For more details, we refer to the PerAct2 paper [\[10\]](#page-9-0).

- 1. Push box: The scene is equipped with a heavy box and a target area. The robot needs to push the box using both arms to move it to the designated area. This task cannot be solved with one robot due to the weight of the box.
- 2. Lift ball: The scene is equipped with a large ball. The robot needs to grasp the ball using both arms and lift it to a height above 0.95 m. This task is impossible to solve with one robot due to the size of the object.
- 3. Push buttons: The scene contains three buttons of different colors. The robot needs to push two of the three buttons simultaneously, as specified in the language instruction.
- 4. Pick up plate: The robot needs to pick and securely lift a plate. The robot must coordinate both arms to lift the plate without tilting or dropping it.

- <span id="page-16-0"></span>5. Put item in drawer: The scene contains a furniture with drawers and a small item. The robot needs to open a specific drawer and place an item inside. The correct drawer is specified in the language instruction.
- 6. Put bottle in fridge: The robot needs to open a fridge and put a bottle inside.
- 7. Handover an item: There are multiple items of different colors on the table. The robot needs to pick one with one arm and hand it securely to the other arm. The correct item is specified by the language instruction.
- 8. Pick up laptop: The robot needs to pick up a laptop that is placed on top of a block. This requires first moving the laptop into a position where it can be grasped and then lifting it.
- 9. Straighten rope: The robot needs to straighten a rope so that both ends are placed in distinct target areas.
- 10. Sweep dustpan: The scene is equipped with a broom, a dust pan, supporting objects and dust. The robot needs to sweep the dust into the dust pan using the broom. The task is considered successfully completed when all the dust is inside the dust pan.
- 11. Lift tray: The robot needs to lift a tray for more than 1.2m. The tray is originally placed on a holder. An item is on top of the tray and must be balanced while the tray is lifted. The primary challenge lies in coordinating the lifting motion with both arms to maintain the balance of the item on the tray.
- 12. Handover item (easy): This is a variant of the handover item task (described above), but the same object is always picked.
- 13. Take tray out of oven: The robot needs to remove a tray that is located inside an oven. This involves opening the oven door and then grasping the tray.

### C.2 Real-world tasks

We explain the real-world tasks and their success conditions in more detail.

- 1. Lift ball: The robot needs to use both arms to stabilize and lift a small ball without grasping it.
- 2. Straighten rope: The robot grasps both ends of the rope and pulls until the rope becomes straight.
- 3. Pick up plate: The robot manipulates a small plate. The left arm needs to tilt the plate, while the right arm grasps it and lifts it up.
- 4. Stack bowls: The robot needs to stack two bowls on top of each other.
- 5. Put marker in cup: The robot needs to use the left arm to reach and fetch the cup, then use the right arm to grasp the marker and put it into the cup.
- 6. Handover block: The robot uses its left arm to pick up a block and give it to the right arm.
- 7. Stack blocks: The robot needs to stack two cubes (3cm on each side) on top of each other, which is harder than stacking bowls, since the blocks are smaller.
- 8. Open marker: The robot needs to grasp the marker with its left arm and then use the right arm to grasp the cap and remove it from the marker.
- 9. Close ziploc: The robot first grasps one end of the ziploc bag and then grasps the gray slider on the ziploc to close the bag.
- 10. Insert battery: The robot moves its left arm near the slot for holding and then the right arm grasps the battery and inserts the battery into the slot.

The tasks "stack bowls", "put marker in cup" and "stack blocks" are bimanual due to the reachability of the objects: the objects are initially placed in locations where only one of the two arms can reach them.

### <span id="page-17-0"></span>C.3 PerAct tasks

PerAct offers a suite of 18 tasks and 249 variations, making it one of the most challenging RLBenchbased benchmarks. The variations include instance references, e.g. "open the top/middle/bottom drawer", color, e.g. "push the blue/purple/etc button", counting, e.g. "stack 2/3 cups", and others such as object category, size and shape. We enumerate the 18 tasks here and refer to the PerAct paper [\[16\]](#page-9-0) for more details:

- 1. Open a drawer
- 2. Slide a block to a colored zone
- 3. Sweep the dust into a dustpan
- 4. Take the meat off the grill frame
- 5. Turn on the water tap
- 6. Put a block in the drawer
- 7. Close a jar
- 8. Drag a block with the stick
- 9. Stack blocks
- 10. Screw a light bulb
- 11. Put the cash in a safe
- 12. Place a wine bottle on the rack
- 13. Put groceries in the cardboard
- 14. Put a block in the shape sorter
- 15. Push a button
- 16. Insert onto a peg
- 17. Stack cups
- 18. Hang cups on the rack

# C.4 74 HiveFormer tasks

HiveFormer offers a suite of 74 tasks, grouped into 9 categories. The tasks come without variations, following [\[27,](#page-10-0) [17\]](#page-9-0). We enumerate the groups and tasks here and refer to [\[27,](#page-10-0) [17\]](#page-9-0) for more details:

- Planning (can be decomposed into multiple sub-goals): basketball in hoop, put rubbish in bin, meat off grill, meat on grill, change channel, tv on, tower3, push buttons, stack wine.
- Tools (need interaction with a target object): slide block to target, reach and drag, take frame off hanger, water plants, hang frame on hanger, scoop with spatula, place hanger on rack, move hanger, sweep to dustpan, take plate off colored dish rack, screw nail.
- Long term: wipe desk, stack blocks, take shoes out of box, slide cabinet open and place cups.
- Rotation-invariant: reach target, push button, lamp on, lamp off, push buttons, pick and lift, take lid off saucepan.
- Motion planner (require continuous interaction with the object/environment): toilet seat down, close laptop lid, open box, open drawer, close drawer, close box, phone on base, toilet seat up, put books on bookshelf.
- Multimodal (multiple solutions are possible): pick up cup, turn tap, lift numbered block, beat the buzz, stack cups.

- <span id="page-18-0"></span>• Precision (high-precision requirements): take usb out of computer, play jenga, insert onto square peg, take umbrella out of umbrella stand, insert usb in computer, straighten rope, pick and lift small, put knife on chopping board, place shape in shape sorter, take toilet roll off stand, put umbrella in umbrella stand, setup checkers.
- Screw (require (un)screwing an object): turn oven on, change clock, open window, open wine bottle.
- Visual Occlusion (large objects occlude certain views): close microwave, close fridge, close grill, open grill, unplug charger, press switch, take money out safe, open microwave, put money in safe, open door, close door, open fridge, open oven, plug charger in power supply

## C.5 8 ChainedDiffuser tasks

While most of the 74 HiveFormer tasks can be solved by simply predicting keyposes and employing the RRT motion planner to obtain a trajectory to execute, there are several tasks that require continuous interaction with the objects and the environment. For example, when opening a fridge, the end-effector trajectory should follow the door's rotation arc to respect its mechanical constraints. ChainedDiffuser [\[19\]](#page-10-0) identified 10 such challenging tasks. PointFlowMatch [\[37\]](#page-11-0) further investigates trajectory prediction to solve 8 of those 10 tasks. To directly compare 3DFA's ability to predict dense trajectories against those works, we evaluate on the same 8 tasks that both works consider, namely:

- 1. Unplug charger
- 2. Close door
- 3. Open box
- 4. Open fridge
- 5. Take frame off hanger
- 6. Open oven
- 7. Put books on bookshelf
- 8. Take shoes out of box

All these tasks require specific motion that cannot be modeled by an RRT planner that connects keyposes with collision-free linear segments. For example, unplugging a charger requires a smooth motion until the charger is out of the socket. Assuming that the end-effector has grasped the charger, the RRT trajectory is not guaranteed to produce a trajectory that smoothly extracts the charger from the socket.

# C.6 Additional details on baselines

On PerAct2, we compare against: (1) ACT [\[8\]](#page-9-0), a 2D transformer architecture that is trained as a conditional VAE to predict a sequence of actions; (2) RVT-LF [\[18,](#page-9-0) [10\]](#page-9-0), that unprojects 2D views to form a point cloud, renders virtual views and feeds them to a transformer to predict the 3D actions for each arm in sequence; (3) PerAct-LF [\[16,](#page-9-0) [10\]](#page-9-0), that voxelizes the 3D space and uses a Perceiver [\[61\]](#page-12-0) architecture to predict the 3D actions for each arm in sequence; (4) PerAct<sup>2</sup> [\[10\]](#page-9-0), which shares the same architecture as PerAct-LF but predicts the actions for the two arms jointly; (5) AnyBimanual [\[34\]](#page-11-0), which combines and adapts two pre-trained single-arm PerAct [\[16\]](#page-9-0) policies; (6) 3D Diffusion Policy (DP3) [\[15\]](#page-9-0), which encodes 3D scenes with a point cloud encoder and uses a diffusion UNet [\[62\]](#page-12-0) to predict the 3D actions; (7) KStar Diffuser [\[32\]](#page-10-0), a diffusion graph convolutional network that regularizes end-effector pose prediction with predicting body joint angles; (8) PPI [\[33\]](#page-10-0), a 3D diffusion policy that regularizes action prediction by tracking points sampled from objects of interest; (9) π<sup>0</sup> [\[20\]](#page-10-0), a state-of-the-art 2D robot generalist that is pre-trained on 10,000 hours of robot data and capable of performing bimanual manipulation. We adapted π<sup>0</sup> to predict end-effector keyposes and fine-tuned it using three cameras.

<span id="page-19-0"></span>

|                                           | Avg.    | open                                                        | slide                                                           | sweep to                                                    | meat off                                                  | turn                                                      | put in                                    | close                             | drag                      | stack                        |
|-------------------------------------------|---------|-------------------------------------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|-------------------------------------------|-----------------------------------|---------------------------|------------------------------|
|                                           | Success | drawer                                                      | block                                                           | dustpan                                                     | grill                                                     | tap                                                       | drawer                                    | jar                               | stick                     | blocks                       |
| 3DDA [12]                                 | 81.3    |                                                             |                                                                 | 84.0 <sub>±4.4</sub>                                        | <b>96.8</b> <sub>±1.6</sub>                               | 99.2 <sub>±1.6</sub>                                      | 96.0 <sub>±3.6</sub>                      | 96.0 <sub>±2.5</sub>              | 100.0 <sub>±0.0</sub>     | 68.3 <sub>±3.3</sub>         |
| 3DFA (2-cam)                              | 78.4    | $95.2_{\pm 3.4}$                                            | $98.4_{\pm 2.2}$                                                | $93.6_{\pm 1.8}$                                            | <b>96.8</b> $_{\pm 3.4}$                                  | 99.2 $_{\pm 1.8}$                                         | $94.4_{\pm 4.6}$                          | $92.0_{\pm 2.2}$                  | $99.2_{\pm 1.8}$          | $28.0_{\pm 3.6}$             |
| 3DFA (4-cam)                              | 80.3    |                                                             |                                                                 | 99.2 $_{\pm 1.8}$                                           | <b>96.8</b> $_{\pm 1.8}$                                  | $97.8_{\pm 1.8}$                                          | $95.2_{\pm 3.0}$                          | $98.4_{\pm 2.2}$                  | $98.0_{\pm 2.5}$          | $41.2_{\pm 3.9}$             |
|                                           |         |                                                             |                                                                 |                                                             |                                                           |                                                           |                                           |                                   |                           |                              |
|                                           |         | screw                                                       | put in                                                          | place<br>wine                                               | put in                                                    | sort<br>shape                                             | push<br>buttons                           | insert                            | stack                     | place                        |
| 2004 5121                                 |         | bulb                                                        | safe                                                            | wine                                                        | cupboard                                                  | shape                                                     | buttons                                   | peg                               | cups                      | cups                         |
| 3DDA [12]                                 |         | bulb   <b>82.4</b> ±2.0                                     | safe   97.6 <sub>±2.0</sub>                                     | wine 93.6 <sub>±4.8</sub>                                   | cupboard $85.6_{\pm 4.1}$                                 | shape   44.0 <sub>±4.4</sub>                              | buttons $98.4_{\pm 2.0}$                  | peg   <b>65.6</b> ±4.1            | cups 47.2 <sub>±8.5</sub> | cups<br>24.0 <sub>±7.6</sub> |
| 3DDA [12]<br>3DFA (2-cam)<br>3DFA (4-cam) |         | bulb<br><b>82.4</b> <sub>±2.0</sub><br>80.0 <sub>±8.0</sub> | safe<br>  97.6 <sub>±2.0</sub><br>  <b>98.4</b> <sub>±2.2</sub> | wine<br>93.6 <sub>±4.8</sub><br><b>98.4</b> <sub>±2.2</sub> | cupboard<br><b>85.6</b> $_{\pm 4.1}$<br>68.8 $_{\pm 7.7}$ | shape<br>  44.0 <sub>±4.4</sub><br>  42.0 <sub>±4.9</sub> | buttons $98.4_{\pm 2.0}$ $99.2_{\pm 1.8}$ | peg   <b>65.6</b> ±4.1   56.4±2.2 | cups 47.2 <sub>±8.5</sub> | cups                         |

Table 4: **Detailed results on the single-arm PerAct benchmark.** 3DFA performs on par with 3DDA on most tasks, even when trained and tested with only two cameras.

On the 74 HiveFormer tasks, we compare against: (1) HiveFormer [27], a 3D policy that predicts actions as offsets from the input point cloud; (2) InstructRL [38], a 2D policy that leverages pretrained vision and language encoders and regresses 6-DoF actions; (3) Act3D [17], a 3D policy that predicts the next action location by iterating between coarse-to-fine sampling and featurization; (4) ChainedDiffuser [19], a two-stage policy that employs Act3D for keypose prediction and a diffusion-based trajectory optimization model to connect the current pose to the next keypose; (5) PointFlowMatch [37], a flow-based 3D policy that encodes the input point cloud into a single vector using PointNet [63] and (6) DP3 [15], discussed in the previous paragraph.

3DFA is trained to predict a keypose-horizon trajectory: it jointly predicts the next end-effector keypose and the dense trajectory until the next keypose in a non-hierarchical, single-forward-pass fashion. HiveFormer, InstructRL and Act3D are trained to predict end-effector keyposes and then use RRT [35] to plan a trajectory. ChainedDiffuser is a two-stage model that predicts the end-effector keypose and then a keypose-conditioned trajectory. PointFlowMatch and DP3 are trained to predict closed-loop trajectories.

#### C.7 Detailed PerAct results

We show detailed results on all 18 tasks for 3DDA and 3DFA in Table 4. We observe that the average success rate is not fully informative. Although the results in many tasks align, there are statistically significant differences on tasks such as block stacking, putting groceries into cupboard, screwing bulb, sweeping to dustpan and opening drawer. Variance is larger for the two-camera 3DFA, as a single mistake may cause severe occlusions across all views, a scenario that is rarer when more cameras are used. Throughout our experimentation, we noticed different sources of large variance:

- First, we identified a very large variance in performance on those tasks across checkpoints. For example, for the same model variant, one checkpoint can achieve 80% success in screwing a bulb, while another can achieve 60%. The pattern we observed is that the variance across checkpoints on individual tasks is large, but the average performance on all tasks does not vary significantly. Our hypothesis is that 3DDA and 3DFA are very low-parameter models: during multi-task training, the network may pick a "mode", where it adapts to specific tasks more and underfits others. Understanding how low-capacity models select task-specific modes during multi-task training is an important avenue for future investigation.
- Second, we tried to fix a checkpoint and evaluate different seeds, which is how we calculate the variance in Table 4. Most tasks display relatively low variance, e.g., they may hit or miss one episode, but this is exacerbated by the small amount of test episode that PerAct offers: with 25 test episodes, a hit/miss contributes 4% to a task's performance. Interestingly, we find that this variance is not due to the non-deterministic nature of 3DDA or 3DFA, but mostly due to the interaction with the simulator, including the motion planner.

<span id="page-20-0"></span>Figure 5: **Detailed evaluation on 74 RLBench tasks.** grouped into 9 categories. 3DFA achieves a new state of the art on all groups.

| stack cups                     | 84  | beat the buzz                | 100 | lift numbered block                 | 100 |
|--------------------------------|-----|------------------------------|-----|-------------------------------------|-----|
| turn tap                       | 100 | pick up cup                  | 100 | wipe desk                           | 37  |
| stack blocks                   | 85  | take shoes out of box        | 71  | slide cabinet open and place cup    | 20  |
| close microwave                | 100 | close fridge                 | 100 | close grill                         | 100 |
| unplug charger                 | 99  | open grill                   | 100 | press switch                        | 100 |
| put money in safe              | 99  | take money out of safe       | 100 | open microwave                      | 100 |
| close door                     | 96  | open door                    | 100 | open fridge                         | 70  |
| open oven                      | 99  | plug charge in power supply  | 60  | slide block to target               | 100 |
| take frame off hanger          | 96  | reach and drag               | 100 | water plants                        | 79  |
| hang frame on hanger           | 50  | scoop with spatula           | 86  | place hanger on rack                | 88  |
| move hanger                    | 100 | sweep to dustpan             | 100 | take plate off colored dish rack    | 100 |
| screw nail                     | 26  | toilet seat down             | 100 | close laptop lid                    | 100 |
| open drawer                    | 98  | open box                     | 100 | close drawer                        | 100 |
| phone on base                  | 97  | close box                    | 100 | reach target                        | 100 |
| toilet seat up                 | 98  | push button                  | 100 | lamp on                             | 100 |
| put books on bookshelf         | 99  | lamp off                     | 100 | pick and lift                       | 100 |
| take lid off saucepan          | 100 | turn oven on                 | 100 | change clock                        | 100 |
| open window                    | 100 | open wine bottle             | 100 | basketball in hoop                  | 100 |
| put rubbish in bin             | 100 | meat off grill               | 100 | meat on grill                       | 100 |
| change channel                 | 100 | tv on                        | 100 | tower3                              | 100 |
| push buttons                   | 100 | stack wine                   | 100 | take usb out of computer            | 100 |
| insert onto square peg         | 80  | play jenga                   | 100 | take umbrella out of umbrella stand | 100 |
| insert usb in computer         | 78  | straighten rope              | 75  | pick and lift small                 | 100 |
| put knife on chopping board    | 99  | place shape in shape shorter | 40  | take toilet roll off stand          | 97  |
| out umbrella in umbrella stand | 25  | setup checkers               | 99  | <del>-</del>                        | -   |

Table 5: Detailed results of 3DFA on all 74 single-arm tasks.

• To isolate the effect of seed, we run the same checkpoint several times with a fixed seed. We still find that internal states in the RLBench simulator and the planner are not controlled by seeding, thus we still observe variance in many tasks.

### C.8 Detailed HiveFormer results

We show the results of 3DFA and baselines on the 9 groups in Figure 5 and the results of 3DFA on all 74 tasks in Table 5. 3DFA is trained to predict trajectories for all tasks and performs very well on most of them. Predicting trajectories allows 3DFA to not rely on motion planners or hand-designed collision checking that previous works employ [17]. While 3DFA solves 57 of 74 tasks with over 90% accuracy, several tasks still exhibit lower success rates. We analyze these failures in detail in Section C.10.

### C.9 Detailed real-world results

As shown in Table 3, 3DFA excels in performing easy tasks, including *lift ball*, *straighten rope*, *pick up plate*, *stack bowl* and *put marker into bowl*. For more challenging tasks such as *stack blocks*, *open marker*, *close ziploc* and *insert battern*, where objects are smaller or transparent or complex contact

<span id="page-21-0"></span>dynamics are involved, the performance of our method and baselines decreases, indicating spatial and dynamic reasoning as a clear avenue for future work.

It is worth mentioning that, when 3DFA fails to grasp the related object, it often automatically re-attempts the grasp. The model will re-attempt the task until it is successful or until the object moves out of the bounding box. This is a result of training for closed-loop trajectory prediction and is not observed when we optimize for keypose prediction.

### C.10 Failure cases

We analyze the failure modes of 3DFA in both the simulation and the real world. We also discuss the failure modes of our baselines in the real world.

On PerAct2, we notice that most of the errors come from motion planning. For example, in the "handover item" tasks, the RRT-predicted trajectories may not be collision-free when the object is placed near the receiving arm. The task with the lowest performance is "straighten rope". RRT predicts linear segments that do not respect the rope limits. Interestingly, we achieve a much higher success rate for the same task in the real world, where we predict trajectories directly. We did not train a trajectory model on PerAct2 because we found the interaction with the simulator to be particularly slow, making the evaluation of a trajectory prediction model impractical.

To validate the effect of the RRT planner in this setup, we replayed the ground-truth keyposes for the test episodes and use the planner to move between them. We get an average performance of 85.8%, which is only marginally better than our model's. "Pick up plate" and "straighten rope" are still the tasks with most errors, achieving 59% and and 39% respectively.

On PerAct, we observe that the model struggles with understanding (sometime unseen) variations at test time. For example, it may place the lid on the wrong jar in "close jar" or stack blocks of wrong colors. We verify that on the HiveFormer setup, where there are no variations, 3DFA achieves very high success rates on stacking blocks and cups (Table [5\)](#page-20-0). This could be mitigated by integrating more powerful representations from a VLM, rather than CLIP. Another source of failure is lower precision than needed in some tasks, such as in "put a block in the shape sorter" and "insert onto peg". 3DFA featurizes all visual observations and subsamples the feature cloud, resulting in a sparser representation. Reducing the number of points is necessary to manage computational resources. This could be prevented using some VLM guidance to focus around regions of interest for the next action. Lastly, when using two cameras only, occlusions can become an issue.

On the 74 HiveFormer tasks, high-precision requirements and occlusions are the main issues. 3DFA achieves its lowest performance on the "long-term" task group. However, this is due to heavy occlusions caused by using the front and wrist cameras only. Specifically, for "slide cabinet open and place cup", the cup is most of the times hidden from the front camera, as it is occluded by the cabinet. 3DFA always opens the cabinet but then struggles to find the cup. To verify, we trained 3DFA using the overhead camera for this task and achieved 74%. Another example of heavy occlusion is "open fridge". Some examples of high-precision tasks that 3DFA struggles are "hang frame on hanger", "place shape in shape shorter" and "put umbrella in umbrella stand".

In the real world experiments, most failure cases of 3DFA in simple tasks are caused by lastcentimeter errors that move the objects towards an undesired direction. Examples include slipping the ball or not raising the other edge of the plate high enough. For hard tasks, the success rate is much lower; we think this is caused by occlusion from the robot or objects in the "handover block", "open marker" and "insert battery" tasks. We also discovered a 2cm error in the robot's forward kinematics, which significantly reduces the performance on the aforementioned tasks that require precise operation. This error is exacerbated when the arm leans forward too much. We show videos of different failure cases for our model on our website.

For baselines, we found that π<sup>0</sup> [\[20\]](#page-10-0) requires the relevant object to always remain visible in the wrist camera view, otherwise it cannot finish the grasping part; this problem is obvious in "handover block", "open marker" and "insert battery" when the wrist camera view is blocked by the object in hand or the robot itself. Most of π<sup>0</sup> failures stem from difficulty in understanding 3D relationships between objects. For example, in the two stacking tasks, the object cannot be placed directly above the other, and in the "insert marker into the cup" task, the model cannot find the correct position to drop the marker. On the other hand, iDP3 [\[40\]](#page-11-0) has a high success rate in finding the 3D relationship between objects, while most of its failure cases are caused by failed grasps.