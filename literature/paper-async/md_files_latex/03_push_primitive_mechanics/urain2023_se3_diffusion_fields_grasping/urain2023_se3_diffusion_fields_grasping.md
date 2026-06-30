# SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion

Julen Urain\*1, Niklas Funk\*1, Jan Peters<sup>1,2,3,4</sup>, Georgia Chalvatzaki<sup>1</sup>

![](_page_0_Picture_3.jpeg)

Fig. 1: Pick and place task in which the robot has to pick a mug and move it to the target pose (in the shelves) without colliding. We exploit diffusion models for jointly optimizing both grasp and motion and show the successful trajectory from left to right.

Abstract—Multi-objective optimization problems are ubiquitous in robotics, e.g., the optimization of a robot manipulation task requires a joint consideration of grasp pose configurations, collisions and joint limits. While some demands can be easily hand-designed, e.g., the smoothness of a trajectory, several task-specific objectives need to be learned from data. This work introduces a method for learning data-driven SE(3) cost functions as diffusion models. Diffusion models can represent highly-expressive multimodal distributions and exhibit proper gradients over the entire space due to their score-matching training objective. Learning costs as diffusion models allows their seamless integration with other costs into a single differentiable objective function, enabling joint gradient-based motion optimization. In this work, we focus on learning SE(3) diffusion models for 6DoF grasping, giving rise to a novel framework for joint grasp and motion optimization without needing to decouple grasp selection from trajectory generation. We evaluate the representation power of our SE(3) diffusion models w.r.t. classical generative models, and we showcase the superior performance of our proposed optimization framework in a series of simulated and real-world robotic manipulation tasks against representative baselines. Videos, code and additional details are available at: https://sites.google.com/view/se3dif

#### I. INTRODUCTION

Autonomous robot manipulation tasks usually involve complex actions requiring a set of sequential or recurring subtasks to be achieved while satisfying certain constraints, thus, casting robot manipulation into a multi-objective motion optimization problem [1]–[3]. Let us consider the pick-and-place task in Fig. 1, for which the motion optimization should consider the possible set of grasping and placing poses, the trajectories' smoothness, collision avoidance with the environment, and the robot's joint limits. While some objectives are easy to model (e.g., joint limits, smoothness),

This work received funding by the DFG Emmy Noether Programme (CH 2676/1-1), by the AICO grant by the Nexplore/Hochtief Collaboration with TU Darmstadt, and the EU project ShareWork.

<sup>1</sup> Technische Universität Darmstadt (Germany), <sup>2</sup> German Research Center for AI (DFKI), <sup>3</sup> Hessian.AI, <sup>4</sup> Centre for Cognitive Science {julen.urain, niklas.funk, jan.peters, georgia.chalvatzaki} @tu-darmstadt.de

others (e.g., collision avoidance, grasp pose selection) are more expensive to model and are therefore commonly approximated by learning-based approaches [4]–[8].

Data-driven models are usually integrated into motion optimization either as sampling functions (explicit generators) [6], [9], or cost functions (scalar fields) [4], [10]. When facing multi-objective optimization scenarios, the explicit generators do not allow a direct composition with other objectives, requiring two or even more separate phases during optimization [11]. Looking back at the example of Fig. 1, a common practice is to learn a grasp generator as an explicit model, sample top-k grasps, and then find the trajectory that, initialized by a grasp candidate, solves the task with a minimum cost. Given the grasp sampling is decoupled from the trajectory planning, it might happen the sampled grasps to be unfeasible for the problem, leading to an unsolvable trajectory optimization problem. On the other hand, learned scalar fields represent task-specific costs that can be combined with other learned or heuristic cost functions to form a single objective function for a joint optimization process. However, these cost functions are often learned through cross-entropy optimization [6], [12] or contrastive divergence [10], [13], creating hard discriminative regions in the learned model that lead to large plateaus in the learned field with zero or noisy slope regions [14], [15], thereby making them unsuitable for pure gradient-based optimization. Thus, it is a common strategy to rely on taskspecific samplers that first generate samples close to low-cost regions before optimizing [6], [12].

In this work, we propose learning *smooth* data-driven cost functions, drawing inspiration from state-of-the-art diffusion generative models [16]–[20]. By *smoothness*, we refer to the cost function exposing informative gradients in the entire space. We propose learning these smooth cost functions in the SE(3) robot's workspace, thus defining task-specific SE(3) cost functions. In particular, in this work, we show how to learn diffusion models for 6DoF grasping, leveraging open-source vastly annotated 6DoF grasp pose datasets like Acronym [21]. SE(3) diffusion models allow moving initially

<sup>\*</sup> Authors contributed equally.

random samples to low-cost regions (regions of good grasping poses on objects) by evolving a gradient-based inverse diffusion process [22] (cf. Fig. 2). SE(3) diffusion models come with two benefits. First, we get smooth cost functions in SE(3) that can be directly used in motion optimization. Second, they better cover and represent multimodal distributions, like in a 6DoF grasp generation scenario, leading to better and more sample efficient performance of the subsequent robot planning.

Consequently, we propose a joint grasp and motion optimization framework using the learned 6DoF grasp diffusion model as cost function and combining it with other differentiable costs (trajectory smoothness, collision avoidance, etc.). All costs combined (learned and hand-designed) form a single, smooth objective function that optimizing it enables the generation of good robot trajectories for complex robot manipulation tasks. This work shows how our framework enables facing grasp generation and classical trajectory optimization as a joint gradient-based optimization loop.

Our contributions are threefold: (1) we show how to learn smooth cost functions in SE(3) as diffusion models. While score-based generative modeling has been previously introduced for arbitrary Riemannian manifolds [19], we focus on the particular requirements for the Lie group SE(3). (2) we use the SE(3) diffusion models to learn 6DoF grasp pose distributions as cost functions. Our experiments show that our learned models generate more diverse and successful grasp poses w.r.t. state-of-the-art grasp generative models. Once the model is trained, (3) we introduce a gradientbased optimization framework for jointly resolving grasp and motion generation, in which we integrate our learned 6DoF grasp diffusion model with additional task-related cost terms. To properly integrate diffusion models in the motion optimization problem, we rewrite the optimization as an inverse diffusion process, similarly to [23]. In contrast with previous methods that decouple the grasp pose selection and the motion planning, our framework resolves the grasp and motion planning problem by iteratively improving the trajectory to jointly minimize the learned object-grasp cost term and the task-related costs. We remark that this joint optimization is only possible thanks to the smoothness of our learned diffusion model and using instead a grasp classifier, trained with cross-entropy loss, as cost won't resolve the problem due to its lack of smoothness. Our quantitative and qualitative results in simulation and the real-world robotic manipulation experiments suggest that our proposed method for learning costs as SE(3) diffusion models enables efficiently finding good grasp and motion solutions against baseline approaches and resolves complex pick-and-place tasks as in Fig. 1.

### II. PRELIMINARIES

**Diffusion Models.** Unlike common deep generative models (Variational Autoencoders (VAE), generative adversarial networks (GAN)) that explicitly generate a sample from a noise signal, diffusion models learn to generate samples by iteratively moving noisy random samples towards a learned

distribution [16], [24]. A common approach to train diffusion models is by *Denoising Score Matching (DSM)* [25], [26]. To apply DSM [24], [27], we first perturb the data distribution  $\rho_{\mathcal{D}}(x)$  with Gaussian noise on L noise scales  $\mathcal{N}(\mathbf{0}, \sigma_k \mathbf{I})$  with  $\sigma_1 < \sigma_2 < \cdots < \sigma_L$ , to obtain a noise perturbed distribution  $q_{\sigma_k}(\hat{x}) = \int_x \mathcal{N}(\hat{x}|x, \sigma_k \mathbf{I}) \rho_{\mathcal{D}}(x) dx$ . To sample from the perturbed distribution,  $q_{\sigma_k}(\hat{x})$  we first sample from the data distribution  $x \sim \rho_{\mathcal{D}}(x)$  and then add white noise  $\hat{x} = x + \epsilon$  with  $\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma_k \mathbf{I})$ . Next, we estimate the score function of each noise perturbed distribution  $\nabla_x \log q_{\sigma_k}(x)$  by training a noise-conditioned vector field  $s_{\theta}(x, k)$ , by score matching  $s_{\theta}(x, k) \approx \nabla_x \log q_{\sigma_k}(x)$  for all  $k = 1, \dots, L$ . The training objective of DSM [26] is

$$\mathcal{L}_{dsm} = \frac{1}{L} \sum_{k=0}^{L} \mathbb{E}_{\boldsymbol{x}, \hat{\boldsymbol{x}}} \left[ \left\| \boldsymbol{s}_{\boldsymbol{\theta}}(\hat{\boldsymbol{x}}, k) - \nabla_{\hat{\boldsymbol{x}}} \log \mathcal{N}(\hat{\boldsymbol{x}} | \boldsymbol{x}, \sigma_k^2 \boldsymbol{I}) \right\| \right], \quad (1)$$

with  $x \sim \rho_D(x)$  and  $\hat{x} \sim \mathcal{N}(x, \sigma_k I)$  To generate samples from the trained model, we apply Annealed Langevin Markov Chain Monte Carlo (MCMC) [28]. We first draw an initial set of samples from a distribution  $x_L \sim \rho_L(x)$  and then, simulate an inverse Langevin diffusion process for L steps, from k = L to k = 1

$$x_{k-1} = x_k + \frac{\alpha_k^2}{2} s_{\theta}(x_k, k) + \alpha_k \epsilon, \ \epsilon \sim \mathcal{N}(0, I),$$
 (2)

with  $\alpha_k > 0$  a step dependent coefficient. Overall, DSM Eq. (1) learns models that output vectors pointing towards the samples of the training dataset  $\rho_{\mathcal{D}}(x)$  [22].

SE(3) Lie group. The SE(3) Lie group is prevalent in robotics. A point  $H = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in SE(3)$  represents the full pose (position and orientation) of an object or robot link with  $R \in SO(3)$  the rotation matrix and  $t \in \mathbb{R}^3$  the 3D position. A Lie group encompasses the concepts of group and smooth manifold in a unique body. Lie groups are smooth manifolds whose elements have to fulfil certain constraints. Moving along the constrained manifold is achieved by selecting any velocity withing the space tangent to the manifold at H (i.e., the so-called tangent space). The tangent space at the identity is called *Lie algebra* and noted  $\mathfrak{se}(3)$ . The Lie algebra has a non-trivial structure, but is isomorphic to the vector space  $\mathbb{R}^6$  in which we can apply linear algebra. As in [29], we work in the vector space  $\mathbb{R}^6$  instead of the Lie algebra  $\mathfrak{se}(3)$ . We can move the elements between the Lie group and the vector space with the logarithmic and exponential maps, Logmap : SE(3)  $\to \mathbb{R}^6$  and Expmap :  $\mathbb{R}^6 \to SE(3)$  respectively [29]. A Gaussian distribution on Lie groups can be defined as

$$q(\boldsymbol{H}|\boldsymbol{H}_{\mu}, \boldsymbol{\Sigma}) \propto \exp\left(-0.5 \|\text{Logmap}(\boldsymbol{H}_{\mu}^{-1}\boldsymbol{H})\|_{\boldsymbol{\Sigma}^{-1}}^{2}\right),$$
 (3)

with  $H_{\mu} \in SE(3)$  the mean and  $\Sigma \in \mathbb{R}^{6 \times 6}$  the covariance matrix [30]. This special form is required as the distance between two Lie group elements is not represented in Euclidean space. Following the notation of [29], given a function  $f : SE(3) \to \mathbb{R}$ , the derivative w.r.t. a SE(3) element,  $Df(H)/DH \in \mathbb{R}^6$  is a vector of dimension 6. We refer the reader to [29] and the Appendix in project site for an extended presentation of the SE(3) Lie group.

![](_page_2_Figure_0.jpeg)

**Fig. 2:** Generating high quality SE(3) grasp poses by iteratively refining random initial samples (k=L) with an inverse Langevin diffusion process over SE(3) elements (Eq. (6)).

# III. SE(3)-DIFFUSION FIELDS

In this section, we show how to adapt diffusion models to the Lie group SE(3) [29], as it is a crucial space for robot manipulation. The SE(3) space is not Euclidean, hence, multiple design choices need to be considered for adapting Euclidean diffusion models. In the following, we first explain the required modifications (Section III-A). Then, we propose a neural network architecture for learning SE(3) diffusion models that represent 6DoF grasp pose distributions and show how we train it (Sec. III-B). Finally, we show how to integrate the learned diffusion models into a grasp and motion optimization problem and show how to optimize it jointly considering the grasp and the motion (Sec. III-C).

### A. From Euclidean diffusion to diffusion in SE(3)

A diffusion model in SE(3) is a vector field that outputs a vector  $v \in \mathbb{R}^6$  for an arbitrary query point  $H \in SE(3)$ , i.e.,  $v = s_{\theta}(H, k)$  with a scalar conditioning variable k determining the current noise scale [24].

**Denoising Score Matching in SE(3).** Similar to the Euclidean space version (cf. Sec. II), DSM is applied in two phases. We first generate a perturbed data point in SE(3), i.e., sample from the Gaussian on Lie groups Eq. (3),  $\hat{H} \sim q(\hat{H}|H, \sigma_k I)$  with mean  $H \in \rho_D(H)$  and standard deviation  $\sigma_k$  for noise scale k. Practically, we sample from this distribution using a white noise vector  $\epsilon \in \mathbb{R}^6$ ,

$$\hat{\boldsymbol{H}} = \boldsymbol{H} \operatorname{Expmap}(\boldsymbol{\epsilon}), \ \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_k^2 \boldsymbol{I}).$$
 (4)

Following the idea of DSM, the model is trained to match the score of the perturbed training data distribution. Thus, DSM in SE(3) requires computing the derivatives of the perturbed distribution w.r.t. a Lie group element. Hence, the new DSM loss function on Lie groups equates to

$$\mathcal{L}_{dsm} = \frac{1}{L} \sum_{k=0}^{L} \mathbb{E}_{\boldsymbol{H}, \hat{\boldsymbol{H}}} \left[ \left\| \boldsymbol{s}_{\boldsymbol{\theta}}(\hat{\boldsymbol{H}}, k) - \frac{D \log q(\hat{\boldsymbol{H}} | \boldsymbol{H}, \sigma_{k} \boldsymbol{I})}{D \hat{\boldsymbol{H}}} \right\| \right], (5)$$

with  $H \sim \rho_{\mathcal{D}}(H)$  and  $\hat{H} \sim q(\hat{H}|H,\sigma_k I)$ . Note that, as introduced in Sec. II, the derivatives w.r.t. a SE(3) element  $\hat{H}$  outputs a vector on  $\mathbb{R}^6$ . In practice, we compute this derivative by automatic differentiation using Theseus [31] library along with PyTorch.

Sampling with Langevin MCMC in SE(3). Evolving the inverse Langevin diffusion process for SE(3) elements (cf. Fig. 2 for visualization) requires adapting the previously presented Euclidean Langevin MCMC approach Eq. (2). In particular, we have to ensure staying on the SE(3) manifold throughout the inverse diffusion process. Thus, we adapt the

inverse diffusion in SE(3) as

$$\bm{H}_{k-1} = \operatorname{Expmap}\left(\frac{\alpha_k^2}{2}\bm{s}_{\bm{\theta}}(\bm{H}_k,k) + \alpha_k\bm{\epsilon}\right)\bm{H}_k, \qquad (6)$$

with  $\epsilon \in \mathbb{R}^6$  sampled from  $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  and the step dependent coefficient  $\alpha_k > 0$ . By iteratively applying Eq. (6), we move a set of randomly sampled SE(3) poses to the data distribution  $\rho_D(\mathbf{H})$  (See Fig. 2).

From the score function to energy model. While most of the works in learning diffusion models learn a vector field representing the score  $s_{\theta}$ , in our work, we learn a scalar field that represents the energy of the distribution  $E_{\theta}$ . In contrast with learning a score function, learning an Energy Based Models (EBM) allow us evaluating the quality of the generated samples and compose it with other cost functions for multi-objective motion optimization. To learn an EBM with denoising score matching, we model our score function  $s_{\theta}(\boldsymbol{H},k) = -DE_{\theta}(\boldsymbol{H},k)/D\boldsymbol{H}$ , as the derivative of the EBM  $E_{\theta}$ .

### B. Architecture & training of Grasp SE(3)-DiffusionFields

Even though we can represent any data-driven cost in SE(3) with SE(3)-DiffusionFields (SE(3)-DiF), in this work, we focus on cost functions that capture 6DoF grasp pose distributions conditioned on the object we aim to grasp. In this work, we assume to have access to the object pose, a reasonable assumption thanks to the impressive results in 6DoF object pose estimation and segmentation [32]. We defer studying the perception aspect of encoding point clouds into object pose and shape as in [6], [33] for a future work. We illustrate the architecture for our grasp SE(3)-DiF model in Fig. 3 and the training pipeline in Algorithm 1. The proposed model maps an object (represented by its id and pose) and a 6DoF grasp pose  $H \in SE(3)$  to an energy  $e \in \mathbb{R}$ , that measures the grasp quality for the particular object.

We train the model to jointly match the Signed Distance Field (SDF) of the object we aim to grasp and predict the grasp energy level by the DSM loss Eq. (5). Learning jointly the SDF of the object and the grasp pose improves the quality of the grasp generation [33], [34]. During the training, we assume the object's id m and pose  $H_w^o \in SE(3)$  are available, and we retrieve a learnable object shape code  $z_m$  given the index m as in [35]. For training the SDF loss, we apply a supervised learning pipeline. Given a dataset of 3D points  $x_w \in \mathbb{R}^3$  and  $sdf \in \mathbb{R}$  for a particular object m,  $\mathcal{D}_{sdf}^m: (x_w, sdf)$ , we first map the points to the object's reference frame  $x_o = H_w^o x_w$  and then predict the SDF given the feature encoder  $F_\theta$  (See Algorithm 1).

As previously introduced in Eq. (5), to apply the DSM loss, we compute the energy  $e \in \mathbb{R}$  over the grasp poses  $\hat{H}$ . These grasp poses have been previously obtained by perturbing grasp poses from the dataset  $H \in \rho_{\mathcal{D}}(H)$  with a noise level k Eq. (4). In our problem, we consider  $\rho_{\mathcal{D}}(H)$  to be a distribution of successful grasp poses for a particular object, and learn the energy to approximate the log-probability of this distribution under noise. We compute the energy e given a grasp pose  $\hat{H}$  in three steps. (I) We transform the

![](_page_3_Figure_0.jpeg)

Fig. 3: SE(3)-DiF's architecture for learning 6D grasp pose distributions. We train the model to jointly learn the objects' sdf and to minimize the denoising loss. Given grasp pose  $H \in SE(3)$  we transform it to a set of 3D points  $x_w \in \mathbb{R}^{N \times 3}$  (I). Next, we transform the points into the object's local frame, using the object's pose  $H_w^o$ . Given the resulting points  $x_o$  and the object's shape code z we apply the feature encoder  $F_\theta$  (II) to obtain a object and grasp-related features (sdf,  $\psi$ )  $\in \mathbb{R}^{N \times (\psi+1)}$ . Finally, (III) we flatten the features and compute the energy e through the decoder  $D_\theta$ . We provide a point-cloud-based implementation in our code repository: https://github.com/TheCamusean/grasp\_diffusion.

grasp pose to a fixed set of N 3D-points around the gripper  $x_g \in \mathbb{R}^{N \times 3}$  in the world frame  $x_w = Hx_g$ . We thereby express the grasp pose through a set of 3D points' positions, similar to [34]. Then, we move the points to the object's local frame,  $x_{o_m} = H_w^{o_m} x_w$ . (II) We apply the feature encoding network  $F_\theta$  which is also conditioned on  $z_m$  and k to inform about the object shape and noise level, respectively. The encoding network outputs both the SDF predictions for the query points,  $\mathrm{sdf} \in \mathbb{R}^{N \times 1}$ , and a set of additional features  $\psi \in \mathbb{R}^{N \times \psi}$ . Thus, the feature encoder's output is of size  $N \times (1 + \psi)$ . (III) We flatten the features and pass them through the decoder  $D_\theta$  to obtain the scalar energy value e. Given the energy, we compute the DSM loss Eq. (5). During training, we jointly learn the objects' latent codes  $z_m$ , and the parameters  $\theta$  of the feature encoder  $F_\theta$  and decoder  $D_\theta$ .

# **Algorithm 1:** Grasp SE(3)-DiF Training **Given:** $\theta_0$ : initial params for z, $F_{\theta}$ , $D_{\theta}$ ;

```
Datasets: \mathcal{D}_o: \{m, \mathbf{H}_w^o\}, object ids and poses,
     \mathcal{D}_{sdf}^m: \{x, sdf\}, 3D positions x and sdf for object m,
     \mathcal{D}_g^{m^*}: \{H\} successful grasp poses for object m;
1 for s \leftarrow 0 to S-1 do
             k, \sigma_k \leftarrow [0, \dots, L];

m, \mathbf{H}_w^o \in \mathcal{D}_o;
                                                                        // sample objects ids and poses
              z = \text{shape codes}(m);
             SDF train
              \boldsymbol{x}, sdf \in \mathcal{D}^m_{sdf};
                                                                // get 3D points and sdf for obj. m
              \hat{sdf}, _{-} = F_{\boldsymbol{\theta}}(\boldsymbol{H}_{w}^{o}\boldsymbol{x}, \boldsymbol{z}, k);
                                                                                         // get predicted sdf
              L_{sdf} = \mathcal{L}_{mse}(s\hat{d}f, sdf);
                                                                                          // compute sdf error
             Grasp diffusion train
             H \sim \mathcal{D}_a^m;
                                                              Sample success grasp poses for obj. \boldsymbol{m}
              \epsilon \sim \mathcal{N}(\mathbf{0}, \sigma_k \mathbf{I});
11
                                                                      // sample white noise on k scale
             \hat{\boldsymbol{H}} = \boldsymbol{H} \operatorname{Expmap}(\boldsymbol{\epsilon});
                                                                         // perturb grasp pose Eq. (4)
12
              \boldsymbol{x}_{n}^{o}=\hat{\boldsymbol{H}}\boldsymbol{x}_{n};
                                                      // Transform to N 3d points (see Figure 3)
             \hat{sdf}_n, \boldsymbol{\psi}_n = F_{\boldsymbol{\theta}}(\boldsymbol{x}_n^o, \boldsymbol{z}_b, k);
14
                                                                                                 // get features
              \Psi = \text{Flatten}(\hat{sdf}_n, \boldsymbol{\psi}_n);
15
                                                                                     // Flatten the features
              e = D_{\boldsymbol{\theta}}(\Psi);
                                                                                             // compute energy
17
              L_{dsm} = \mathcal{L}_{dsm}(e, \hat{\boldsymbol{H}}, \boldsymbol{H}, \sigma_k);
                                                                               // Compute dsm loss Eq. (5)
             Parameter update
18
              L = L_{\rm dsm} + L_{\rm sdf};
                                                                                                     // Sum losses
19
             \boldsymbol{\theta}_{s+1} = \boldsymbol{\theta}_s - \alpha \nabla_{\boldsymbol{\theta}} L;
                                                                                        // Update parameters
21 return \theta^*;
```

### C. Grasp and motion optimization with diffusion models

Given a trajectory  $\tau: \{q_t\}_{t=1}^T$ , consisting of T waypoints, with  $q_t \in \mathbb{R}^{d_q}$  the robot's joint positions at time instant t; in motion optimization, we aim to find the minimum cost trajectory  $\tau^* = \arg\min_{\boldsymbol{\tau}} \mathcal{J}(\boldsymbol{\tau}) = \arg\min_{\boldsymbol{\tau}} \sum_j \omega_j c_j(\boldsymbol{\tau})$ , where the objective function  $\mathcal{J}$  is a weighted sum of costs  $c_j$ ,

with weights  $\omega_i > 0$ . Herein, we integrate the learned SE(3)-DiF for grasp generation as one cost term of the objective function. It is, thus, combined with other heuristic costs, e.g., collision avoidance or trajectory smoothness. Optimizing over the whole set of costs enables obtaining optimal trajectories jointly taking into account grasping, as well as motionrelated objectives. This differs from classic grasp and motion planning approaches in which the grasp pose sampling and trajectory planning are treated separately [36], by first sampling the grasp pose, and, then, searching for a trajectory that satisfies the selected grasp. In classic approaches, given the grasp sampling is decoupled from the trajectory planning, it might happen the sampled grasps to be unfeasible for the problem, leading to an unsolvable trajectory planning problem. We hypothesize that jointly optimizing over both the grasp pose and the trajectory allows us to be more sample efficient w.r.t. decoupled approaches.

Given that the learned function is in SE(3) while the optimization is w.r.t. the robot's joint space, we redefine the cost as  $c(q_t,k)=E_{\theta}(\phi_{ee}(q_t),k)$ , with the forward kinematics  $\phi_{ee}:\mathbb{R}^{d_q}\to \text{SE}(3)$  mapping from robot configuration to the robot's end-effectors task space. To obtain minimum cost trajectories, we frame the motion generation problem as an inverse diffusion process. Using a planning-as-inference view [23], [37]–[39], we define a desired target distribution as  $q(\tau|k) \propto \exp(-\mathcal{J}(\tau,k))$ . This allows us to set an inverse Langevin diffusion process that evolves a set of random initial particles drawn from a distribution  $\tau_L \sim p_L(\tau)$  towards the target distribution  $q(\tau|k)$ 

$$\tau_{k-1} = \tau_k + 0.5 \ \alpha_k^2 \nabla_{\tau_k} \log q(\tau|k) + \alpha_k \epsilon, \ \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad (7)$$

with step dependent coefficient  $\alpha_k > 0$ , noise level moving from k = L to k = 1, and one particle corresponding to an entire trajectory. If we evolve the particles by this inverse diffusion process for sufficient steps, the particles at k = 1,  $\tau_1$  can be considered as particles sampled from  $q(\tau|k=1)$ . To obtain the optimal trajectory, we evaluate the samples on  $\mathcal{J}(\tau, 1)$  and pick the one with the lowest cost.

# IV. EXPERIMENTAL EVALUATION

The experimental section is divided in three parts. First, we evaluate our trained model for 6DoF grasp pose generation (Sec. IV-A). We train a SE(3)-DiF as a 6DoF grasp pose generative model using the Acronym dataset [21]. This

![](_page_4_Figure_0.jpeg)

**Fig. 4:** 6D grasp pose generation experiment. Left: Success rate evaluation. Right: Earth Mover Distance (EMD) evaluation metrics (lower is better).

simulation-based dataset contains successful 6DoF grasp poses for a variety of objects from ShapeNet [40]. We focus on the collection of successful grasp poses for 90 different mugs (approximately 90K 6DoF grasp poses). We provide a model trained in a larger dataset and conditioned on point cloud in the project page. We obtain the mugs' meshes from ShapeNet, and train the model as described in Algorithm 1. We generate a set of grasp poses from the learned models and evaluate on successful grasping and diversity. Second, we evaluate the quality of our trained model when used as an additional cost term for grasp and motion optimization (Sec. IV-B). We compare the performance of solving a grasp and motion optimization problem jointly (using the learned model as cost function), w.r.t. the stateof-the-art approaches that decouple the grasp selection and motion planning, or heuristically combine them. Finally, we validate the performance of our method in a set of real robot experiments (Sec. IV-C).

### A. Evaluation of 6DoF grasp pose generation

We evaluate grasp poses generated from our trained grasp SE(3)-DiF model in terms of the success rate, and the EMD between the generated grasps and the training data distribution. We consider 90 different mugs and evaluate 200 generated grasps per mug. We evaluate the grasp success on Nvidia Isaac Gym [41]. The EMD measures the divergence between two empirical probability distributions [42], providing a metric on how similar the generated samples are to the training dataset. To eliminate any other influence, we only consider the gripper and assume that we can set it to any arbitrary pose. We generate 6DoF grasp poses from SE(3)-DiF by an inverse diffusion process, following Eq. (6).

We compare against three baselines. First, based on [6], [43], we consider generating grasp poses by first sampling from a decoder of a trained VAE and subsequently running MCMC over a trained classifier for pose refinement (VAE+Refine). Second, we consider sampling from the VAE (without any further refinement). Third, we consider running MCMC over the classifier starting from random initial pose [44]. In this experiment, we assume the object's pose and id/shape to be known, and purely focus on evaluating the models' generative capabilities. For ensuring a fair comparison, all the baselines consider a shape code  $z_m$  to encode the object information as presented in Fig. 3. We add a pointcloud-conditioned experiment in the Appendix.

We present the results in Fig. 4. In terms of success rate, SE(3)-DiF outperforms VAE+Refine slightly (especially yielding lower variance), and VAE or classifier on their own significantly. The VAE alone generates noisy grasp poses that

![](_page_4_Figure_7.jpeg)

**Fig. 5:** Evaluation Pick in occlusion. We measure the success rate of 4 different methods based on different number of initializations.

are often in collision with the mug. In the case of classifier only, the success rate is low. We hypothesize that this might be related with the classifier's gradient, as specifically in regions far from good samples, the field has a large plateau with close to zero slopes [14]. This leads to not being able to improve the initial samples. Considering grasp diversity, i.e., EMD metric (lower is better), SE(3)-DiF outperforms all baselines significantly. A reason for the difference, might be that VAE+Refine overfits to specific overrepresented modes of the data distribution. In contrast, SE(3)-DiF's samples capture the data distribution more properly. We, therefore, conclude that SE(3)-DiF is indeed generating high-quality and diverse grasp poses. We add an extended presentation of the experiment in the Appendix in our project site.

### B. Performance on grasp and motion optimization

We evaluate the performance of our learned grasp SE(3)-DiF as a cost term into multi-objective grasp and motion optimization problems. We consider the task of picking amidst clutter (see Fig. 6) and measure the success rate on solving it. The success is measured based on the robot being able to grasp the object at the end of the execution. In the Appendix in our project site, we provide additional details on the chosen cost functions for the task. As introduced in Sec. III-C, we generate the trajectories by integrating our learned grasp SE(3)-DiF as an additional cost function to the motion optimization objective function. Then, given a set of initial trajectory samples, obtained from a Gaussian distribution with a block diagonal matrix as in [2], we apply gradient descent methods Eq. (7) to iteratively improve the trajectories on the objective function. We evaluate the success rate of the trajectory optimization given a different number of initial samples. As gradient-based trajectory optimization methods are inherently local optimization methods, multiple initializations might lead to better results. We consider three baselines (see Fig. 5). **Decoupl.**: we adopt the common routing to solve grasp and motion optimization problems in a decoupled way [11], [45], [46]. We first sample a set of 6DoF grasp poses from a generative model and then plan a trajectory that satisfies the selected grasp pose with CHOMP [1]. Second, we consider the **OMG-Planner** [47], that applies an online grasp selection and planning approach. Finally, **joint** (class.): we consider applying a joint optimization as in our approach, but using a 6DoF grasp classifier as cost function rather than a grasp SE(3)-DiF.

The results in Fig. 5 present a clear benefit from the joint optimization w.r.t. the decoupled approach and the OMG-Planner. In particular, our proposed joint optimization only

![](_page_5_Picture_0.jpeg)

![](_page_5_Picture_1.jpeg)

Fig. 6: Simulated and real robot environments for picking amidst clutter.

requires 25 particles to match the success rate of the decoupled approach with 800 particles. The reason for this significant gap in efficiency is that the decoupled approach generates SE(3) grasp poses that are not feasible given the environment constraints, such as clutter or joint limits. However, when optimizing jointly, we can find trajectories that satisfy all the costs by iteratively improving entire trajectories w.r.t. all objectives. We also observe the importance of using grasp SE(3)-DiF as cost term instead of a grasp classifier. The classifier model lacks proper gradient information to inform how to move the trajectories to grasp the object due to its lack of smoothness in the whole space. Thus, the motion optimization problem is unable to find solutions.

### C. Grasp and motion optimization on real robots

We conducted a thorough real-world evaluation of our joint grasp and motion optimization framework driven by our 6DoF grasp diffusion model, using it as an additional cost function, similarly to our simulated robot manipulation tasks. Fig. 1 depicts a sequence of a real-world pick-mug and place-on-shelf scenario. Overall, the experiments aim at assessing the method's capabilities in realistic conditions that include, i) non-perfect state information, as the mugs pose is retrieved from an external system (Optitrack) which induces small calibration errors, ii) variations in the mug's shape, as we use a mug that is slightly different from the one we specify for SE(3)-DiF, and iii) real-world trajectory execution. For optimization, we initialize 800 particles (trajectories), and only execute the one with lowest cost.

In the simplest testing scenario, where the robot has to pick up a mug from various poses in a scene without any clutter, we achieve 100% (20 successes / 20 trials) pickup-success. We also find that our method transfers well to the more difficult scenarios of picking up mugs that are initially placed upside down with 90% (18/20) success, picking in occluded scenes with 95% (19/20) success, and having to pick and place the mug in a desired pose inside the shelf of Fig. 1 with 100% (20/20) success. Our real-world results underline the effectiveness of our joint optimization approach. Videos of the experiments also showcase that our method still comes up with very versatile solutions<sup>1</sup>. Note that we attribute the increased real-world performance w.r.t. the simulated one to the simpler designed experimental scene, i.e., in simulation we considered flying obstacles that were not realizable in the real scene (Fig. 6). Nevertheless, our results confirm that

our proposed approach is highly performant in real settings, without suffering sim2real discrepancies.

Limitations In our experiments, we focused on evaluating our diffusion model's performance in grasp generation, besides full trajectory optimization, assuming full object state knowledge, without relying on complex perception systems. Potential sim2real gaps w.r.t. the real environment could potentially arise from imperfect perception, and hand-designed cost terms that may not capture well the relevant task description in more complex scenarios. Moreover, a limitation comes with increasing number of cost terms, as it becomes more difficult to weight them.

### V. RELATED WORK

**Diffusion models in Robotics** Diffusion models have appeared in Robotics in various tasks, from text-conditioned scene rearrangement [48], [49], decision-making [23], [50], [51] and controllable traffic generation [52]. We additionally highlight earlier works like [53], where a diffusion process is integrated into a motion planning problem.

**6D grasp generation.** 6D grasp pose generation is solved with a myriad of methods from classifiers to explicit samplers. [44], [54], [55] sample candidate grasps and score them with learned classifiers. [56] predicts grasping outcomes using a geometry-aware representation. Contrary to methods classifying grasps, generative models can be trained to generate grasp poses from data [6] but might require additional sample refinement. While the generator in [43] considers possible collisions in the scene, [57] proposes to learn a grasp distribution over the object's manifold. [33] uses scene representation learning to learn grasp qualities and explicitly predict 3D rotations. Recently, [58] proposed learning a 6 DoF SDF to represent grasp pose generation as a smooth cost function and optimize on top of it.

**Integrated grasp and motion planning.** Due to the interdependence of the selected grasp pose with the robot motion, multiple efforts have tried to integrate both variables into a single planning problem [47], [59]–[62]. In [59], [60], goal sets representing grasp poses are integrated as constraints in a motion optimization problem. In [61], [63], Rapidly-exploring Random Trees [64] is combined with a TCP attractor to bias the tree towards good grasps.

### VI. CONCLUSION

We proposed SE(3)-DiffusionFields (SE(3)-DiF) for learning task-space, data-driven cost functions to enable robotic motion generation through joint gradient-based optimization over a set of combined cost functions. At the core of SE(3)-DiFs is a diffusion model that provides informative gradients across the entire space and enables data generation through an inverse Langevin dynamics diffusion process. Besides having demonstrated that SE(3)-DiF generates diverse and high-quality 6DoF grasp poses, we also drew a connection between motion generation and inverse diffusion. Thus, we presented a joint gradient-based grasp and motion optimization framework, which outperforms traditional decoupled

optimization approaches. Our extensive experimental evaluations reveal the superior performance of the proposed method w.r.t. efficiency, adaptiveness, and success rates. In the future, we want to explore diffusion models for reactive motion control and the composition of multiple diffusion models to solve complex manipulation tasks in which multiple hard-to-model objectives might arise.

### REFERENCES

- [1] N. Ratliff, M. Zucker, J. A. Bagnell, and S. Srinivasa, "Chomp: Gradient optimization techniques for efficient motion planning," in *IEEE International Conference on Robotics and Automation*, 2009.
- [2] M. Kalakrishnan, S. Chitta, E. Theodorou, P. Pastor, and S. Schaal, "Stomp: Stochastic trajectory optimization for motion planning," in IEEE International Conference on Robotics and Automation, 2011.
- [3] J. Schulman, Y. Duan, J. Ho, A. Lee, I. Awwal, H. Bradlow, J. Pan, S. Patil, K. Goldberg, and P. Abbeel, "Motion planning with sequential convex optimization and convex collision checking," *The International Journal of Robotics Research*, 2014.
- [4] D. Rakita, B. Mutlu, and M. Gleicher, "RelaxedIK: Real-time synthesis of accurate and feasible robot arm motion." in *Robotics: Science and Systems*, 2018.
- [5] T. Osa, "Motion planning by learning the solution manifold in trajectory optimization," *The International Journal of Robotics Research*, 2022.
- [6] A. Mousavian, C. Eppner, and D. Fox, "6-DoF graspnet: Variational grasp generation for object manipulation," in *International Conference* on Computer Vision, 2019.
- [7] J. Urain, M. Ginesi, D. Tateo, and J. Peters, "Imitationflows: Learning deep stable stochastic dynamic systems by normalizing flows," in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2020.
- [8] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, and V. Sitzmann, "Neural descriptor fields: Se (3)equivariant object representations for manipulation," in *International Conference on Robotics and Automation*. IEEE, 2022.
- [9] D. Koert, G. Maeda, R. Lioutikov, G. Neumann, and J. Peters, "Demonstration based trajectory optimization for generalizable robot motions," in *IEEE-RAS International Conference on Humanoid Robots*, 2016.
- [10] A. Lambert, A. T. Le, J. Urain, G. Chalvatzaki, B. Boots, and J. Peters, "Learning implicit priors for motion optimization," *IEEE International Conference on Intelligent Robots and Systems*, 2022.
- [11] A. Murali, A. Mousavian, C. Eppner, C. Paxton, and D. Fox, "6-DoF grasping for target-driven object manipulation in clutter," in *IEEE International Conference on Robotics and Automation*, 2020.
- [12] Q. Lu, K. Chenna, B. Sundaralingam, and T. Hermans, "Planning multi-fingered grasps as probabilistic inference in a learned deep network," in *Robotics Research*, 2020.
- [13] C. Finn, S. Levine, and P. Abbeel, "Guided cost learning: Deep inverse optimal control via policy optimization," in *International Conference on Machine Learning*, 2016.
- [14] M. Arjovsky and L. Bottou, "Towards principled methods for training generative adversarial networks," arXiv preprint arXiv:1701.04862, 2017.
- [15] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in *International Conference* on *Learning Representations*, 2018.
- [16] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, "Score-based generative modeling through stochastic differential equations," in *International Conference on Learning Rep*resentations, 2020.
- [17] C. Luo, "Understanding diffusion models: A unified perspective," 2022. [Online]. Available: https://arxiv.org/abs/2208.11970
- [18] C.-W. Huang, M. Aghajohari, A. J. Bose, P. Panangaden, and A. Courville, "Riemannian diffusion models," arXiv preprint arXiv:2208.07949, 2022.
- [19] V. D. Bortoli, E. Mathieu, M. J. Hutchinson, J. Thornton, Y. W. Teh, and A. Doucet, "Riemannian score-based generative modelling," in *Advances in Neural Information Processing Systems*, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https://openreview.net/forum?id=oDRQGo817P
- [20] D. Gnaneshwar, B. Ramsundar, D. Gandhi, R. Kurchin, and V. Viswanathan, "Score-based generative models for molecule generation," arXiv preprint arXiv:2203.04698, 2022.
- [21] C. Eppner, A. Mousavian, and D. Fox, "Acronym: A large-scale grasp dataset based on simulation," in *IEEE International Conference on Robotics and Automation*, 2021.
- [22] Y. Song and D. P. Kingma, "How to train your energy-based models," arXiv preprint arXiv:2101.03288, 2021.
- [23] M. Janner, Y. Du, J. Tenenbaum, and S. Levine, "Planning with diffusion for flexible behavior synthesis," in *International Conference* on Machine Learning, 2022.

- [24] Y. Song and S. Ermon, "Generative modeling by estimating gradients of the data distribution," Advances in Neural Information Processing Systems, 2019.
- [25] P. Vincent, "A connection between score matching and denoising autoencoders," Neural computation, 2011.
- [26] S. Saremi, A. Mehrjou, B. Schölkopf, and A. Hyvärinen, "Deep energy estimator networks," arXiv preprint arXiv:1805.08306, 2018.
- [27] Y. Song and S. Ermon, "Improved techniques for training score-based generative models," Advances in Neural Information Processing Systems, 2020.
- [28] R. M. Neal et al., "Mcmc using hamiltonian dynamics," Handbook of markov chain monte carlo, 2011.
- [29] J. Sola, J. Deray, and D. Atchuthan, "A micro lie theory for state estimation in robotics," *arXiv preprint arXiv:1812.01537*, 2018.
- [30] G. Chirikjian and M. Kobilarov, "Gaussian approximation of nonlinear measurement models on lie groups," in *IEEE Conference on Decision and Control*, 2014.
- [31] L. Pineda, T. Fan, M. Monge, S. Venkataraman, P. Sodhi, R. Chen, J. Ortiz, D. DeTone, A. Wang, S. Anderson et al., "Theseus: A library for differentiable nonlinear optimization," arXiv preprint arXiv:2207.09442, 2022.
- [32] B. Wen, C. Mitash, B. Ren, and K. E. Bekris, "se (3)-tracknet: Data-driven 6d pose tracking by calibrating image residuals in synthetic domains," in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 10367–10373.
- [33] Z. Jiang, Y. Zhu, M. Svetlik, K. Fang, and Y. Zhu, "Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations," in *Robotics: Science and Systems*, 2021.
- [34] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, and V. Sitzmann, "Neural descriptor fields: Se(3)equivariant object representations for manipulation," in *International Conference on Robotics and Automation*, 2022.
- [35] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, "Deepsdf: Learning continuous signed distance functions for shape representation," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019.
- [36] F. Lagriffoul, D. Dimitrov, J. Bidot, A. Saffiotti, and L. Karlsson, "Efficiently combining task and motion planning using geometric constraints," *The International Journal of Robotics Research*, 2014.
- [37] M. Botvinick and M. Toussaint, "Planning as inference," Trends in cognitive sciences, 2012.
- [38] S. Levine, "Reinforcement learning and control as probabilistic inference: Tutorial and review," arXiv preprint arXiv:1805.00909, 2018.
- [39] J. Urain, P. Liu, A. Li, C. D'Eramo, and J. Peters, "Composable Energy Policies for Reactive Motion Generation and Reinforcement Learning," in *Robotics: Science and Systems*, 2021.
- [40] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su et al., "Shapenet: An informationrich 3d model repository," arXiv preprint arXiv:1512.03012, 2015.
- [41] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al., "Isaac gym: High performance gpu-based physics simulation for robot learning," arXiv preprint arXiv:2108.10470, 2021.
- [42] A. Tanaka, "Discriminator optimal transport," Advances in Neural Information Processing Systems, 2019.
- [43] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, "Contact-graspnet: Efficient 6-DoF grasp generation in cluttered scenes," in *IEEE International Conference on Robotics and Automation*, 2021.
- [44] A. ten Pas, M. Gualtieri, K. Saenko, and R. Platt, "Grasp pose detection in point clouds," *The International Journal of Robotics Research*, 2017.
- [45] K. Rahardja and A. Kosaka, "Vision-based bin-picking: Recognition and localization of multiple complex objects using simple visual cues," in *IEEE/RSJ International Conference on Intelligent Robots and Systems.*, 1996.
- [46] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg, "Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics," arXiv preprint arXiv:1703.09312, 2017.
- [47] L. Wang, Y. Xiang, and D. Fox, "Manipulation Trajectory Optimization with Online Grasp Synthesis and Selection," in *Robotics: Science* and Systems, 2020.
- [48] I. Kapelyukh, V. Vosylius, and E. Johns, "Dall-e-bot: Introducing web-scale diffusion models to robotics," arXiv preprint arXiv:2210.02438, 2022

- [49] W. Liu, T. Hermans, S. Chernova, and C. Paxton, "Structdiffusion: Object-centric diffusion for semantic rearrangement of novel objects," arXiv preprint arXiv:2211.04604, 2022.
- [50] A. Ajay, Y. Du, A. Gupta, J. Tenenbaum, T. Jaakkola, and P. Agrawal, "Is conditional generative modeling all you need for decision-making?" arXiv preprint arXiv:2211.15657, 2022.
- [51] Z. Wang, J. J. Hunt, and M. Zhou, "Diffusion policies as an expressive policy class for offline reinforcement learning," arXiv preprint arXiv:2208.06193, 2022.
- [52] Z. Zhong, D. Rempe, D. Xu, Y. Chen, S. Veer, T. Che, B. Ray, and M. Pavone, "Guided conditional diffusion for controllable traffic simulation," arXiv preprint arXiv:2210.17366, 2022.
- [53] W. Park, J. S. Kim, Y. Zhou, N. J. Cowan, A. M. Okamura, and G. S. Chirikjian, "Diffusion-based motion planning for a nonholonomic flexible needle model," in *Proceedings of the 2005 IEEE International Conference on Robotics and Automation*. IEEE, 2005, pp. 4600–4605.
- [54] X. Lou, Y. Yang, and C. Choi, "Collision-aware target-driven object grasping in constrained environments," in *IEEE International Confer*ence on Robotics and Automation, 2021.
- [55] H. Liang, X. Ma, S. Li, M. Görner, S. Tang, B. Fang, F. Sun, and J. Zhang, "Pointnetgpd: Detecting grasp configurations from point sets," in *International Conference on Robotics and Automation*, 2019.
- [56] X. Yan, J. Hsu, M. Khansari, Y. Bai, A. Pathak, A. Gupta, J. Davidson, and H. Lee, "Learning 6-DoF grasping interaction via deep geometry-aware 3d representations," in *IEEE International Conference on Robotics and Automation*, 2018.
- [57] J. Hager, R. Bauer, M. Toussaint, and J. Mainprice, "Graspme-grasp manifold estimator," in 2021 30th IEEE International Conference on Robot & Human Interactive Communication, 2021.
- [58] T. Weng, D. Held, F. Meier, and M. Mukadam, "Neural grasp distance fields for robot manipulation," arXiv preprint arXiv:2211.02647, 2022.
- [59] A. Dragan, G. J. Gordon, and S. Srinivasa, "Learning from experience in manipulation planning: Setting the right goals," *Robotics Research*, 2017.
- [60] D. Berenson, S. Srinivasa, and J. Kuffner, "Task space regions: A framework for pose-constrained manipulation planning," *The Interna*tional Journal of Robotics Research, 2011.
- [61] N. Vahrenkamp, M. Do, T. Asfour, and R. Dillmann, "Integrated grasp and motion planning," in *IEEE International Conference on Robotics* and Automation. IEEE, 2010.
- [62] N. Funk, C. Schaff, R. Madan, T. Yoneda, J. U. De Jesus, J. Watson, E. K. Gordon, F. Widmaier, S. Bauer, S. S. Srinivasa et al., "Benchmarking structured policies and policy optimization for real-world dexterous object manipulation," *IEEE Robotics and Automation Letters*, vol. 7, no. 1, pp. 478–485, 2021.
- [63] J. Fontanals, B.-A. Dang-Vu, O. Porges, J. Rosell, and M. A. Roa, "Integrated grasp and motion planning using independent contact regions," in *IEEE-RAS International Conference on Humanoid Robots*, 2014.
- [64] S. M. LaValle et al., "Rapidly-exploring random trees: A new tool for path planning," The annual research report, 1998.
- [65] D. F. Crouse, "On implementing 2d rectangular assignment algorithms," IEEE Transactions on Aerospace and Electronic Systems, 2016.
- [66] G. Sutanto, A. Wang, Y. Lin, M. Mukadam, G. Sukhatme, A. Rai, and F. Meier, "Encoding physical constraints in differentiable newton-euler algorithm," in *Machine Learning Research*, 2020.
- [67] M. Bhardwaj, B. Sundaralingam, A. Mousavian, N. D. Ratliff, D. Fox, F. Ramos, and B. Boots, "Storm: An integrated framework for fast jointspace model-predictive control for reactive manipulation," in *Confer*ence on Robot Learning, 2022.
- [68] Y. Xiang, T. Schmidt, V. Narayanan, and D. Fox, "Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes," in *Robotics: Science and Systems*, 2018.
- [69] C. Deng, O. Litany, Y. Duan, A. Poulenard, A. Tagliasacchi, and L. J. Guibas, "Vector neurons: A general framework for so (3)-equivariant networks," in *IEEE/CVF International Conference on Computer Vision*, 2021.
- [70] Y. Xie, T. Takikawa, S. Saito, O. Litany, S. Yan, N. Khan, F. Tombari, J. Tompkin, V. Sitzmann, and S. Sridhar, "Neural fields in visual computing and beyond," in *Computer Graphics Forum*, 2022.
- [71] D. Chetverikov, D. Svirko, D. Stepanov, and P. Krsek, "The trimmed iterative closest point algorithm," in *International Conference on Pattern Recognition*, 2002.

#### APPENDIX I

#### THEORY ON SE(3) LIE GROUP: DERIVATIVES AND DISTRIBUTIONS

The Lie group SE(3) is prevalent in robotics. A point  $H \in SE(3)$  represents the full pose (position and orientation) of an object or robot link

$$\boldsymbol{H} = \begin{bmatrix} \boldsymbol{R} & \boldsymbol{t} \\ \boldsymbol{0} & 1 \end{bmatrix} \in SE(3) \tag{8}$$

with  $R \in SO(3)$  the rotation matrix and  $t \in \mathbb{R}^3$  the 3D position. A Lie group is both a group and a differentiable manifold (See [29] for additional details on groups). Given SE(3) is a differentiable manifold, for any point  $H \in SE(3)$ , there exists a tangent space centered around H that is locally diffeomorphic to SE(3). The tangent space can be afterwards map to a Cartesian vector space  $\mathbb{R}^6$ . In particular, the tangent space at identity is known as Lie algebra and is noted by  $\mathfrak{se}(3)$ .

We can interact between the Lie group and the Lie algebra through the logmap and expmap functions. The logmap is a function that maps a point  $\mathbf{H} \in SE(3)$  to the Lie algebra  $\mathfrak{se}(3)$ , logmap :  $SE(3) \to \mathfrak{se}(3)$ . Inversely, the expmap moves the points  $\mathbf{h}^{\wedge} \in \mathfrak{se}(3)$  to the Lie group SE(3), expmap :  $\mathfrak{se}(3) \to SE(3)$ . Additionally, we can relate the elements in the Lie algebra  $\mathfrak{se}(3)$  with the Cartesian vector space  $\mathbb{R}^6$  through the hat and vee functions. The hat function  $(\cdot)^{\wedge} : \mathbb{R}^6 \to \mathfrak{se}(3)$  maps the points in the vector space  $\mathbf{h} \in \mathbb{R}^6$  to the Lie algebra  $\mathfrak{se}(3)$ . Inversely, the vee function  $(\cdot)^{\vee} : \mathfrak{se}(3) \to \mathbb{R}^6$ , moves the points in the Lie algebra  $\mathbf{h}^{\wedge} \in \mathfrak{se}(3)$  to the vector space  $\mathbb{R}^6$ . The vector space  $\mathbb{R}^6$  is isomorphic to  $\mathfrak{se}(3)$ . Then, we can move any point from  $\mathfrak{se}(3)$  to  $\mathbb{R}^6$  and back. Nevertheless,  $\mathbf{h} \in \mathbb{R}^6$  representation is more useful in our case as we can apply Linear algebra on them. Finally, we additionally call Logmap the map from SE(3) to  $\mathbb{R}^6$ , Logmap = logmap $(\cdot)^{\vee} : SE(3) \to \mathbb{R}^6$  and Expmap the map from  $\mathbb{R}^6$  to SE(3), Expmap = expmap $(\cdot)^{\wedge} : \mathbb{R}^6 \to SE(3)$ . Note that we use the upper case (Logmap, Expmap) to represent a mapping to the vector space and the lower case (logmap, expmap) to represent the mapping to the Lie algebra.

A vector field  $f: SE(3) \to \mathbb{R}^6$  is a function that outputs a vector in the Cartesian vector space  $\mathbb{R}^6$  for any point in SE(3). The vector's values are dependent on a particular tangent space centered at  $\mathbf{H} \in SE(3)$ . Given that there exist infinite tangent spaces (one per point in SE(3)), the value of the vectors might vary depending on the tangent space. We can transform the vectors related with one tangent space to another, with the adjoint matrix operator. The adjoint matrix operator is a linear map  $\dot{\mathbf{h}}_1 = \mathbf{A}_0^1 \dot{\mathbf{h}}_0$ , that transform a vector  $\dot{\mathbf{h}}_0 \in \mathbb{R}^6$  tied with the tangent space centered at  $\mathbf{H}_0 \in SE(3)$  to the vector  $\dot{\mathbf{h}}_1 \in \mathbb{R}^6$  tied with the tangent space centered at  $\mathbf{H}_1 \in SE(3)$ .

# A. Derivatives on Lie groups

To properly define derivatives on Lie groups, we are required to consider the geometry of the manifold. Given a function  $f(\cdot): \mathbb{R}^m \to \mathbb{R}^n$ , the Jacobian is defined as

$$J = \frac{\partial f(x)}{\partial x} \stackrel{\text{def}}{=} \lim_{\tau \to 0} \frac{f(x+\tau) - f(x)}{\tau} \in \mathbb{R}^{n \times m}, \tag{9}$$

with  $\tau \in \mathbb{R}^m$ . Nevertheless, if we aim to compute the Jacobian on the SE(3) Lie group, we are required to adapt the formulation, as we cannot directly sum x and  $\tau$ . Given a function  $f(\cdot) : \mathcal{M} \to \mathcal{N}$  from the manifold  $\mathcal{M}$  to the manifold.  $\mathcal{N}$ , the Jacobian is defined as

$$\frac{Df(\mathcal{X})}{D\mathcal{X}} \stackrel{\text{def}}{=} \lim_{\tau \to 0} \frac{f(\tau \oplus \mathcal{X}) \ominus f(\mathcal{X})}{\tau} = \lim_{\tau \to 0} \frac{\text{Logmap}(f(\mathcal{X})^{-1}f(\text{Expmap}(\tau)\mathcal{X}))}{\tau} \in \mathbb{R}^{m \times n}, \tag{10}$$

where m is the dimension of the manifold  $\mathcal{M}$  and n, the dimension of the manifold  $\mathcal{N}$ .  $\mathcal{X} \in \mathcal{M}$  is an element in  $\mathcal{M}$  and the output  $f(\mathcal{X}) \in \mathcal{N}$  an element in  $\mathcal{N}$ . The plus  $\oplus$  and minus  $\ominus$  operators must be selected appropriately:  $\oplus$  for the domain  $\mathcal{M}$  and  $\ominus$  for the codomain  $\mathcal{N}$  [29]. In our work, we derive assuming the left Jacobian (10); yet as presented in [29], it is also possible to compute the right Jacobian. For the case of SE(3), the Jacobian of the function f will transform a vector of dimension f to a vector in  $\mathbb{R}^6$ . Similarly, to functions mapping between Euclidean spaces, we can apply the chain rule given functions that map between manifolds. Given  $\mathcal{Y} = f(\mathcal{X})$  and  $\mathcal{Z} = g(\mathcal{Y})$ , the Jacobian of  $\mathcal{Z} = g(f(\mathcal{X}))$  is defined

$$J = \frac{DZ}{DX} = \frac{DZ}{DY} \frac{DY}{DX} = \frac{D(g(Y))}{DY} \frac{D(f(X))}{DX},$$
(11)

by the concatenation of the Jacobians of each function.

### B. Distributions on Lie groups

To apply the score matching loss, we first sample a datapoint from a Gaussian distribution  $q_{\sigma_k}(\hat{x}|x) = \mathcal{N}(\hat{x}|x, \sigma_k I)$  with the mean  $x \sim \rho_{\mathcal{D}}(x)$  sampled from the data distribution. A sample from  $q_{\sigma_k}(\hat{x}|x)$  can be easily obtain by perturbing a datapoint from the demonstrations with white noise  $\hat{x} = x + \epsilon$  with  $\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma_k I)$ . Nevertheless, given SE(3) is not an Euclidean space, we cannot directly sample from a Gaussian distribution as the generated sample might fall out of the manifold. In our work, we adapt the Gaussian distribution to Lie Groups. Similarly to [30], we model the sampling distribution in SE(3) as

$$q_{\sigma_k}(\hat{\boldsymbol{H}}|\boldsymbol{H}) \propto \exp\left(-\frac{1}{2} \left\| \text{Logmap}(\boldsymbol{H}^{-1}\hat{\boldsymbol{H}}) \right\|_{\Sigma^{-1}}^2\right),$$
 (12)

where  $\Sigma = \sigma_k^2 I$  the covariance matrix. Following the intuition from [30], as long as  $\sigma_k$  is small enough, the tails of the distribution decay to zero along every geodesic path leading away from identity. We can sample from (12),

$$\hat{\boldsymbol{H}} = \boldsymbol{H} \operatorname{Expmap}(\boldsymbol{\epsilon}), \ \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_k \boldsymbol{I}), \tag{13}$$

by first converting a white noise sample to a SE(3) element, and then, perturbing the mean of the distribution H with transformed white noise.

1) Score function in SE(3): The score matching loss encourages the gradient of the parameterized model  $DE_{\theta}(H)/DH$  to match the score of the perturbed distribution (12). We call  $\phi = \text{Logmap}(H^{-1}\hat{H})$  and  $M = H^{-1}\hat{H}$ . We compute the score function by the chain rule

$$\frac{D\log q_{\sigma_k}(\hat{\boldsymbol{H}}|\boldsymbol{H})}{D\hat{\boldsymbol{H}}} = \frac{D(-\frac{1}{2}\|\boldsymbol{\phi}\|_{\Sigma^{-1}}^2)}{D\boldsymbol{\phi}} \frac{D(\operatorname{Logmap}(\boldsymbol{M}))}{D\boldsymbol{M}} \frac{D(\boldsymbol{H}^{-1}\hat{\boldsymbol{H}})}{D\hat{\boldsymbol{H}}}.$$
(14)

The first part can be directly computed in the Euclidean space

$$\frac{D(-\frac{1}{2}\|\phi\|_{\Sigma^{-1}}^2)}{D\phi} = \frac{\partial(-\frac{1}{2}\|\phi\|_{\Sigma^{-1}}^2)}{\partial\phi} = -\frac{\phi}{\sigma_L^2}$$
(15)

that is the score function of a Gaussian distribution on Euclidean spaces. This is the score that is matched in [24]. The second term

$$\frac{D(\operatorname{Logmap}(\boldsymbol{M}))}{D\boldsymbol{M}} = \boldsymbol{J}_l^{-1}(\boldsymbol{\phi}) \tag{16}$$

is the inverse left-Jacobian on SE(3) (See [29]). The third term

$$\frac{D(\boldsymbol{H}^{-1}\hat{\boldsymbol{H}})}{D\hat{\boldsymbol{H}}} = \mathbf{Adj}_{\boldsymbol{H}^{-1}} \tag{17}$$

is the adjoint over  $H^{-1}$ .

# APPENDIX II ALGORITHMIC DETAILS

In this section, we provide the pseudocode for all 3 main algorithms (for training the diffusion models, for generating grasp poses, and for handling motion optimization with diffusion).

# A. Algorithmic implementation of the training procedure

Algorithm 2 summarizes the training procedure for obtaining our SE(3)-DiF diffusion models. This section thus complements Section III-B. Before starting to explain the training procedure, we want to point out that we are dealing with a combined objective (line 16). On the one hand, we refine the representation to learn the object's sdf. Learning the sdf should instill geometric reasoning into the proposed architecture. In the algorithm, all operations that are related with this part are commented with "SDF". On the other hand, our model should also be capable to match the score of the perturbed data distribution. Therefore, we use the denoising score matching loss as the second objective. All operations related to diffusion are marked with "DIF".

In every training iteration, we first sample a minibatch of b object ids. Next, we query the shape codes for all the selected objects. Please note that the shape codes are also learnable parameters, and actually updated during the training procedure. Afterwards follow the typical steps for learning the sdfs' of the selected object, i.e., sampling j 3D points per object and their groundtruth sdf values, before querying the predictions by the network and constructing the loss function. From line 7 on follow the steps for score matching. We first start by sampling i grasp poses per object from the Acronym dataset [21]. The dataset originally contains good (successful) and bad (unsuccessful) grasping poses. However, as we are only interested in learning the distribution of successful grasping poses, we do not consider the bad ones in this sampling step. Next follow

# **Algorithm 2:** Training procedure for SE(3)-DiF **Given:** $\theta_0$ : initial parameters of the function $E_{\theta}$ ;

```
S: Optimization steps;
   Dataset \mathcal{D}: \{\{\boldsymbol{H}_{k,i}\}_{i=0}^{I_k}, \{\boldsymbol{x}_{k,j}, \mathrm{sdf}_{k,j}\}_{j=0}^{J_k}, o_k\}_{k=1}^K: K objects, J 3D points \boldsymbol{x}_{k,j} \in \mathbb{R}^3 per object with the \mathrm{sdf}_{k,j} \in \mathbb{R} SDF value
    for each point. I H_{k,i} \in SE(3) good grasp poses per object, o_k \in \mathbb{R} object id.;
    H_w^o = I: object's pose set to identity for training.
1 for s \leftarrow 0 to S-1 do
          o_b \in \mathcal{D};
                                                                                                             // Sample a minibatch of b objects ids
2
          z_b = \text{shape codes}(o_b, \boldsymbol{\theta}_s);
3
                                                                                                              // get all shape codes (see Figure 3)
4
          \boldsymbol{x}_{b,j}, \mathrm{sdf}_{b,j} \in \mathcal{D};
                                                                // SDF: Sample a minibatch of j 3D points and sdf per object
          sdf_{b,j}, = F_{\boldsymbol{\theta}}(\boldsymbol{x}_{b,j}, \boldsymbol{z}_b, k);
                                                                                                          // SDF:get predicted sdf (see Figure 3)
          l_{\text{sdf}} = \mathcal{L}_{\text{mse}}(\hat{sdf}_{b,j}, sdf_{b,j});
                                                                                                                                         // SDF:compute sdf error
 6
          H_{b,i} \sim \mathcal{D};
                                                                            // DIF: Sample a minibatch of i grasp poses per object
7
         k, \sigma_k \leftarrow [0, \dots, L];
                                                                                                                                   // DIF:Sample a noise level
8
          \epsilon_{b,i} \sim \mathcal{N}(\mathbf{0}, \sigma_k \mathbf{I});
                                                                                                                                   // DIF:sample a white noise
          \hat{\boldsymbol{H}}_{b,i} = \boldsymbol{H}_{b,i} \operatorname{Expmap}(\boldsymbol{\epsilon}_{b,i});
                                                                                                                    // DIF:perturb grasp poses Eq. (4)
10
          \boldsymbol{x}_{b,i,n}^o = \boldsymbol{H}_w^o \hat{\boldsymbol{H}}_{b,i} \boldsymbol{x}_n;
                                                                                                  // DIF:Transform N 3d points (see Figure 3)
11
          s\hat{d}f_{b,i,n}, \boldsymbol{\phi}_{b,i,n} = F_{\boldsymbol{\theta}}(\boldsymbol{x}_{b,i,n}^{o}, \boldsymbol{z}_{b}, k);
                                                                                                      // DIF:get latent features (see Figure 3)
12
          \Phi_{b,i} = \text{Flatten}(\hat{sdf}_{b,i,n}, \phi_{b,i,n});
                                                                                                    // DIF:Flatten the features (see Figure 3)
13
          e_{b,i} = D_{\boldsymbol{\theta}}(\Phi_{b,i});
                                                                                                                // DIF:Compute energy (see Figure 3)
14
          l_{\text{dsm}} = \mathcal{L}_{\text{dsm}}(e_{b,i}, \mathbf{H}_{b,i}, \mathbf{H}_{b,i}, \sigma_k);
                                                                                                                // DIF:Compute dsm loss with Eq. (5)
15
          l = l_{\rm dsm} + l_{\rm sdf};
                                                                                                                                                                // Sum losses
16
         \boldsymbol{\theta}_{s+1} = \boldsymbol{\theta}_s - \alpha \nabla_{\boldsymbol{\theta}} l;
                                                                                                                                                // Update parameter \boldsymbol{\theta}
17
18 return \theta^*;
```

the steps of sampling noise and perturbing the previously selected grasping poses (lines 8-10). Following our explanations in Section III-B, we represent the SE(3) grasping poses through a collection of N 3D points that are sampled around the gripper's pose. We subsequently query our architecture to receive the features for each of these 3D points representing the grasping pose. Subsequently, we combine all of these points corresponding to a single grasping pose through flattening to obtain the predicted grasp quality (i.e., energy). We finally compute the DSM loss function, add the two objectives and perform gradient descent to update our network's parameters as well as the object's shape codes.

### B. Algorithmic implementation of 6D grasp generation using SE(3)-DiF

In Algorithm 3 we provide pseudocode for SE(3) grasp generation, closely following Section III-A. We nevertheless want to point out that in some experiments in which also table collisions have to be considered,  $E_{\theta}$  might consist of multiple terms and therefore not only represent the energies output by our learned diffusion model.

# Algorithm 3: SE(3) grasp generation pipeline

```
Given: \{\sigma_k\}_{k=1}^L: Noise levels;
   L: Diffusion steps;
   \epsilon: step rate;
   Initialize n_s initial samples \boldsymbol{H}_L^{n_s} \sim p_L(\boldsymbol{H})
1 for k \leftarrow L to 1 do
          e_{n_s} = E_{\boldsymbol{\theta}}(\boldsymbol{H}_k^{n_s}, k);
                                                                                                                                                          // Compute the energy per \boldsymbol{H}_k^{n_s}
2
           \alpha_k = \epsilon \cdot \sigma_k / \sigma_L;
                                                                                                                                                                         // Select step size \alpha_k
3
           \epsilon \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I});
                                                                                                                                 // Sample white noise vector of size \mathbb{R}^6
          \boldsymbol{H}_{k-1}^{n_s} = \operatorname{Expmap}\left(-\frac{\alpha_k^2}{2}\frac{De_{n_s}}{D\boldsymbol{H}_i^{n_s}} + \alpha_k\boldsymbol{\epsilon}\right)\boldsymbol{H}_k^{n_s},;
                                                                                                                                                                                         // Make a ld step
6 return H_0^{n_s};
```

### C. Algorithmic implementation of robot trajectory optimization using SE(3)-DiF

Algorithm 4 summarizes the procedure for trajectory optimization using inverse diffusion. The pseudocode follows Section III-C. Again, the total cost per trajectory is usually a combination of multiple cost terms. Finally, we only return the minimum cost trajectory and execute it in simulation / on the real robot.

### Algorithm 4: Trajectory optimization pipeline

```
Given: \{\sigma_k\}_{k=1}^L: Noise levels; L: Diffusion steps; T: trajectory lenght; Q: dimension of the configuration space; \epsilon: step rate; Initialize n_s initial samples \boldsymbol{\tau}_L^{n_s} \sim p_L(\boldsymbol{\tau}) of size Q \times T each 1 for k \leftarrow L to 1 do 2 c_{n_s} = \mathcal{J}(\boldsymbol{\tau}_k^{n_s}, k); // Compute the total cost per \boldsymbol{\tau}_k^{n_s} 3 \alpha_k = \epsilon \cdot \sigma_k/\sigma_L; // Select step size \alpha_k 4 \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}); // Sample white noise vector of size \mathbb{R}^{Q \times T} 5 \boldsymbol{\tau}_{k-1}^{n_s} = \boldsymbol{\tau}_k^{n_s} + \frac{1}{2}\alpha_k^2 \nabla_{\boldsymbol{\tau}_k} c_{n_s} + \alpha_k \epsilon; // Make a 1d step 6 return arg \min_{\boldsymbol{\tau}_0^{n_s}} \mathcal{J}(\boldsymbol{\tau}_0^{n_s}, 0);
```

![](_page_12_Picture_2.jpeg)

Fig. 7: A frame of the grasp success evaluation for the model SE(3)-DiF in Nvidia Isaac Sim.

# APPENDIX III EXTENDED EXPERIMENTAL EVALUATION

### A. Evaluation of SE(3)-DiffusionFields as 6D grasp pose generative models

In the following, we provide an extended presentation of the experiment in Section IV-A. We measure the success rate with the physics simulator Nvidia Isaac Sim. We present a visualization of the evaluation environment in Figure 7. To evaluate the success rate of each model, we first generate 200 SE(3) grasp poses with each model and for each object. As we can observe in Figure 7, the generated grasps are diverse and consider multiple grasping points. For our model, we get the initial SE(3) elements by sampling from a normal distribution on Lie groups

$$H_0 = \operatorname{Expmap}(\epsilon), \ \epsilon \sim \mathcal{N}(\mathbf{0}, \sigma \mathbf{I})$$
 (18)

with  $\sigma = \sigma_K$ , the biggest noise level during the training. Then, we evaluate the grasps quality in Nvidia Isaac Gym. We reset the Franka's end effector in the chosen grasp pose. We smoothly close the fingers until a tight grip is achieved and lift the gripper to a certain height. We consider the grasp to be successful if, after the lift, the mug remains close to the gripper. We also evaluate the divergence of the generated samples distribution w.r.t. data distribution. This divergence informs about how well the learned distribution matches the data distribution, covering all modes. We measure this divergence with the EMD [42]. We first sample N=1000 grasp poses from the data distribution and from the learned model, respectively, and build a table with the relative distance between all the SE(3) grasp poses as

$$d_{SO(3)+\mathbb{R}^3}(\boldsymbol{H}_i, \boldsymbol{H}_j) = \|\boldsymbol{t}_i - \boldsymbol{t}_j\| + \|\text{LogMap}(\boldsymbol{R}_i^{-1} \boldsymbol{R}_j)\|,$$
(19)

with  $t_i$  and  $t_j$  the 3D position and  $R_i$  and  $R_j$  the rotation matrix of  $H_i$  and  $H_j$  respectively. Then, we solve a Linear Sum Assignment optimization problem [65]. This problem solves an optimal transport problem that will search for the least-distance one-to-one assignment between the samples in the data distribution and the sampled grasp poses from the learned model. The smaller the distance, the closer the generated samples are from the data distribution.

We compare the performance of SE(3)-DiF w.r.t. three models that are inspired by 6dof-GraspNet [6] and present the results in Figure 4. We have trained a VAE to generate 6D grasp poses and a classifier to discriminate between good and bad grasp poses. The classifier network shares the same architecture of SE(3)-DiF, proposed in Figure 3. For the VAE, we have trained a conditioned VAE that receives as input the shape code of the object to grasp and the 6D pose. We jointly train a Deep Signed Distace Field (DeepSDF) [35] that shares the shape code with the conditioned VAE. We trained the classifier with a cross-entropy loss and added a gradient regularizer to encourage smoother gradients. Nevertheless, when

![](_page_13_Picture_0.jpeg)

![](_page_13_Picture_1.jpeg)

![](_page_13_Picture_2.jpeg)

![](_page_13_Picture_3.jpeg)

Fig. 8: Visualization of evaluation procedure for robot grasp pose generation. Note that the pictures illustrate the evaluation of the 5 lowest cost particles using our proposed joint optimization with SE(3)-DiF. Importantly, each particle is evaluated in its own environment (environment is identified by colored arm & mug) and there are no collisions between different environments. Left to right: 1) All environments start with the same initial mug & robot pose. 2) Setting the arms to the optimized robot grasp pose. 3 & 4) Attempting to lift the mugs. In this case, all the particles result in successes.

**TABLE I:** Comparing different approaches for robot grasp pose generation with  $n_s = 100$  initial samples.

|                                            | Objects upright |       | Objects flipped |       |
|--------------------------------------------|-----------------|-------|-----------------|-------|
| Method                                     | $s_{\Omega}$    | $s_1$ | $s_{\Omega}$    | $s_1$ |
| joint opt (classifier)                     | 0.03            | 0.77  | 0.03            | 0.68  |
| sample (SE(3)-DiF) + opt $(n_{sm} = 1)$    | 0.11            | 0.79  | 0.03            | 0.57  |
| sample (SE(3)-DiF) + opt ( $n_{sm} = 10$ ) | 0.46            | 0.80  | 0.12            | 0.76  |
| Ours                                       | 0.62            | 0.88  | 0.49            | 0.88  |

the grasp poses are too far from the data distribution, the classifier lacks informative gradients that would allow us to move the grasp poses to the high-probability regions (See Figure 4).

### B. Evaluation of SE(3)-DiffusionFields for robot grasp pose generation

This section complements the findings presented in Section IV-B. The results for the robot grasp pose generation have also been obtained in Nvidia Isaac Gym, using the procedure as shown in Figure 8.

For obtaining the results, the optimizations not only consider the grasp pose in SE(3), but also the robot joint configuration. In particular, for the two end-end approaches, i.e., classifier & joint opt, we aim to minimize the following objective function

$$\mathcal{J}(\mathbf{q}) = E_{\theta}(\phi_{ee}(\mathbf{q})) + c_{\text{table coll.}}(\mathbf{q})$$
(20)

with the learned grasp costs  $E_{\theta}$  and the table collision cost. The **table collision avoidance cost** is computed for all the collision spheres in the robot  $x_c = (x_c, y_c, z_c) \in \mathbb{R}^3$ . Given the radius for a particular collision body is  $r_c \in \mathbb{R}$ 

$$c_{\text{table coll.}}(\boldsymbol{q}) = \sum_{c=0}^{K} \text{ReLU}(-(z_c - z_{\text{table}} - r_c)). \tag{21}$$

For the separate optimization procedure, we first only optimize for the grasp poses  $H_{\text{grasp pose}}$  through

$$\mathcal{J}(\boldsymbol{H}_{\text{grasp pose}}) = E_{\boldsymbol{\theta}}(\boldsymbol{H}_{\text{grasp pose}}), \tag{22}$$

thereby not taking into account the current pose of the object, nor any other environmental constraint and thus have to subsequently optimize the following cost function in joint space (for fixed  $H_{grasp\ pose}$ )

$$\mathcal{J}(q) = c_{\text{des grasp dist}}(\phi_{ee}(q), H_{\text{grasp pose}}) + c_{\text{table coll.}}(q)$$
(23)

with current grasping pose H(q) and the cost on the distance to the previously optimized grasp pose

$$c_{\text{des grasp dist}}(\boldsymbol{H}(\boldsymbol{q}), \boldsymbol{H}_{\text{grasp pose}}) = 10 \|\boldsymbol{t}_{q} - \boldsymbol{t}_{\text{grasp pose}}\| + \|\text{LogMap}(\boldsymbol{R}_{q}^{-1}\boldsymbol{R}_{\text{grasp pose}})\|.$$
 (24)

Note the additional factor of 10 due to the different scales of position error in [m] and orientation error in [rad].

As we have shown in the main paper (cf. Section IV-B), and as it is also underlined by the accompanied videos (accesible in the linked website in the abstract), using the classifier, and the separate optimization performs substantially worse performance compared to our proposed joint optimization using SE(3)-DiF.

For the separate optimization procedure (sample + opt), we actually even ran two variants. The results of optimizing ten joint configuration samples per previously sampled grasp pose ( $n_{sm}$ =10) have been shown in the main paper, thus the second optimization phase even considers 1000 samples in total (100 grasping poses × 10 samples per grasp pose). When comparing these results to only optimizing one joint configuration per grasp in the second stage ( $n_{sm}$  = 1), we observe that optimizing for finding the single desired grasp pose while also avoiding table collisions is difficult. Allowing 10 joint configuration samples per grasp pose and only evaluating the best one ( $n_{sm}$ =10) performs substantially better, but still worse compared to our proposed joint optimization with SE(3)-DiF. Particularly, the experiments showcase a performance drop w.r.t. the flipped mug scenario. This underlines the major shortcoming of not being adaptive w.r.t. the current environment. The split optimization for grasp pose and joint configuration results in many proposed grasp poses which are simply infeasible.

Contrarily, for our proposed joint optimization the ratio of overall successful particles  $s_{\Omega}$  drops only slightly, and  $s_1$  even remains on the same high level of 0.88. We, thus, conclude that end-end gradient-based optimization with our SE(3)-DiF model results in highly performant, reliable, and adaptive robot grasp pose generation, despite the multi-objective scenario.

### Exact weighting of cost terms

In Table II, we additionally present the exact weighting of the cost terms that have been used for generating the results presented in Section IV-B & Section III-B.

|                           |                   | Weighting of cost terms             |                            |                                          |                                                                                            |
|---------------------------|-------------------|-------------------------------------|----------------------------|------------------------------------------|--------------------------------------------------------------------------------------------|
|                           |                   | Minimize distance                   |                            |                                          |                                                                                            |
|                           |                   | Grasp cost Table collision          |                            | to desired grasp                         |                                                                                            |
| Method                    | Phase             | $E_{\theta}(H_{\text{grasp pose}})$ | $E_{\theta}(\phi_{ee}(q))$ | $c_{\text{table coll.}}(\boldsymbol{q})$ | $c_{\text{des grasp dist}}(\phi_{ee}(\boldsymbol{q}), \boldsymbol{H}_{\text{grasp pose}})$ |
| joint opt (classifier)    |                   | -                                   | 2                          | 3                                        | =                                                                                          |
| our joint opt (SE(3)-DiF) |                   | -                                   | 2                          | 3                                        | -                                                                                          |
| sample (SE(3)-DiF)        | 1) grasp sampling | 1                                   | -                          | -                                        | -                                                                                          |
| + opt                     | 2) opt robot pose | -                                   | -                          | 0.5                                      | 1.5                                                                                        |

**TABLE II:** This table summarizes the weighting of the individual cost terms that have been used for generating robot grasp poses, as presented in Section IV-B & Section III-B. The table's first two rows describe the weighting of the cost terms when running one single joint optimization procedure in which grasp generation and avoiding table collisions are considered jointly. While for the method in the first row, we use the trained classifier as described in Section IV-B, all the other approaches use our proposed SE(3)-DiF diffusion model as the cost function for evaluating grasp poses. Moreover, table's last two rows detail the weighting of the cost terms for the split, two-stage optimization procedure. Note that the separate optimization thus requires running two optimizations.

### C. Evaluation of SE(3)-DiffusionFields for joint grasp and motion optimization

In the following, we provide an extended presentation of the experiments in Section IV-B. We evaluate the performance of SE(3)-DiF as cost function in a trajectory optimization problem. We consider three robot tasks in which both the selection of the grasping pose, and the trajectory planning are required. We explore if we can use SE(3)-DiF as a cost in a single trajectory optimization problem and jointly optimize both for the trajectory and the grasp pose at the last waypoint. The objective function

$$\mathcal{J}(\boldsymbol{\tau}, k) = E_{\boldsymbol{\theta}}(\phi_{ee}(\boldsymbol{q}_{t=T}), k) + \sum_{k} c_{k}(\boldsymbol{\tau})$$
(25)

is composed of both the learned SE(3)-DiF,  $E_{\theta}$  and a set of heuristics cost functions that represent different subtasks (trajectory smoothness, collision avoidance, ...). All trajectories are planned in the configuration space. Then, we frame the optimization problem as an inverse diffusion process and diffuse a set of initial trajectory samples as presented in Section III-C. We sample the initial trajectories as straight trajectories in the configuration space towards a randomly sampled configuration. After diffusing a set of trajectories, we pick the one with the lowest accumulated cost  $\mathcal{J}(\tau,1)$ . We evaluate this approach in three tasks that require the planning of both the trajectory and the grasping pose: picking an object with occlusions, picking and reorienting an object and placing on shelves.

1) Picking with occlusions: In the following, we provide an extended presentation of the experiment on picking an object with occlusions. The experimental evaluation is performed in three different scenarios, with the mug initialized both in normal pose or upside down. We illustrate the scenarios in Figure 9. We evaluate the success for 100 trajectories with

![](_page_14_Picture_10.jpeg)

![](_page_14_Picture_11.jpeg)

![](_page_14_Picture_12.jpeg)

Fig. 9: Scenarios for Picking with occlusions. The boxes and the table are obstacles and the robot must find a trajectory to grasp the mug. We consider the mug might be positioned both upright and upside-down.

different mugs positions for the three environments. The success is measured by following the generated trajectory. Once the robot is in the last position of the generated trajectory, we close the fingers. We consider a success case if the mug is in contact with both fingers once the fingers close.

The objective function for this problem is defined by the following cost functions: (a) Grasp Pose SE(3)-DiF over the final configuration  $q_T$ , (b) a trajectory smoothess cost, (c) a table avoidance cost, (d) box avoidance costs, (e) initial configuration fixing cost and, (f) a pregrasp cost. Given some of the costs are defined in the task space, we use a differentiable robot

![](_page_15_Figure_0.jpeg)

Fig. 10: Evaluation of the Success for picking with occlusions.

kinematic model, based on Facebook's kinematics model [66]. We define the collision body of our robot by a set of spheres similar to [67]. We set the **trajectory smoothness cost** 

$$c_{\text{smooth}}(\tau) = \sum_{t=1}^{T-1} \| \boldsymbol{q}_{t+1} - \boldsymbol{q}_t \|^2$$
 (26)

as the minimization of the relative distance between the neighbour points in the trajectory. This cost can be thought as a spring making all the point in the trajectory be attracted between each other. The **table collision avoidance cost** is computed for all the collision spheres in the robot  $x_{ct} = (x_{ct}, y_{ct}, z_{ct}) \in \mathbb{R}^3$ . Given the radius for a particular collision body is  $r_c \in \mathbb{R}$ 

$$c_{\text{table coll.}}(\tau) = \sum_{t=1}^{T} \sum_{c=0}^{K} \text{ReLU}(-(z_{ct} - z_{\text{table}} - r_c))$$
(27)

with  $z_{\text{table}}$  the height of the table and a Rectified Linear Unit (ReLU) to bound the cost. Given we have access to the SDF of the collision obstacles in the environment, we can set the **box collision cost** as

$$c_{\text{box coll.}}(\boldsymbol{\tau}) = \sum_{t=1}^{T} \sum_{c=0}^{K} \text{ReLU}(-(\text{SDF}(\boldsymbol{x}_{ct}) - r_c)). \tag{28}$$

In the trajectory optimization problems, we might want to fix the initial configuration to the current robot configuration. While the easiest approach is not updating  $q_0$  during optimization, we can alternatively set a **initial configuration fixing** cost

$$c_{fix}(\boldsymbol{\tau}) = \|\boldsymbol{q}_1 - \boldsymbol{q}_{\text{init}}\| \tag{29}$$

with  $q_{\text{init}}$  the initial configuration of the robot. Finally, we set also a **pregrasping cost**. It is common to approximate to the grasp in the cartesian space from a grasp a few centimeters over the grasp pose. We set a cost that encourages the optimized trajectory to approximate in this way

$$c_{\text{pregrasp}}(\boldsymbol{\tau}) = \sum_{t=T-n}^{T-1} d_{\text{SO(3)}+\mathbb{R}^3}(\boldsymbol{H}_{ee,t}, \boldsymbol{H}_{\text{pre,t}})$$
(30)

with  $H_{ee,t}$  the end effector pose in the instant t and  $H_{pre,t} = H_{ee,T}H_{z,t}$  a pose that is to a certain distance over the z axis from the final pose.

As baseline, we also evaluate the performance of solving the task in a hierarchical approach. In this case, we first sample a SE(3) grasp pose given our learned SE(3)-DiF and then, we solve the trajectory optimization problem, given the target grasp pose is fix with the cost  $d_{SO(3)+\mathbb{R}^3}$ . The complete evaluation can be found in Figure 10.

We evaluate the success rate of the model assuming a different set of initial particles. We observe that the performance in all the cases increases when considering more initial particles. Gradient based motion optimization is an inherently locally optimization method and therefore, its performance is highly influenced by the initialization. To enhance the performance, multiple initial particles, initialized in different states might explore better the optimization field and find more optimal solutions. We observe that the joint optimization approach outperforms the hierarchical approach in all the cases. This is expected solution. A hierarchical approach decouples the grasp selection from the trajectory optimization. Then, if the selected grasp is unfeasible for the robot, we will not be able to find a good trajectory. Instead a joint optimization problem iteratively updates the trajectory improving both the grasp cost and the rest of the costs. Therefore, we find that jointly optimizing is more sample efficient than a hierarchical approach.

### **Exact weighting of cost terms**

In Table III we present the weighting of the individual cost terms that we have used to obtain the trajectories for these scenarios of having to pick the mug under occlusions.

| Description                          | Cost                            | Weight |
|--------------------------------------|---------------------------------|--------|
| Grasp pose evaluation                | $E_{\theta}(\phi_{ee}(q_T), k)$ | .5     |
| Trajectory smoothness                | $c_{\mathrm{smooth}}(\tau)$     | 10.    |
| Table collision avoidance            | $c_{\text{table coll.}}(\tau)$  | 20.    |
| Box collision cost (other obstacles) | $c_{\text{box coll.}}(\tau)$    | 20.    |
| Initial configuration fixing cost    | $c_{fix}(\tau)$                 | 10.    |
| Pregrasping cost                     | $c_{\text{pregrasp}}(\tau)$     | 5.     |

**TABLE III:** This table summarizes the weighting of the individual cost terms that have been used for generating robot trajectories for the task of picking a mug under occlusions, as presented in Section IV-B & Section III-C.1.

2) Pick and reorient: In the following we provide an extended presentation of the experiment of picking and reorienting an object. This experiment aims to explore the performance of SE(3)-DiF in a complex manipulation task as the one of picking an object and reorient it. We highlight that the whole optimization problem on how to grasp the object, and how to move it to a target pose is solved in a single optimization loop. The problem is interesting as the optimized trajectory should not only consider that there is a collision free path to an affordable grasp pose, but also, that the chosen grasp pose allows us to put the object in a desired target pose. We evaluate the performance of our model 100 times in which the objects are initialized in an arbitrary random pose and have to be placed in an arbitrary placing pose. We consider a trial to be successful, if after executing the whole trajectory, the distance between the grasped object and target pose is smaller to a threshold.

The objective function for this problem maintains multiple costs from the pick on occluded problem (trajectory smoothness, pregrasp, initial target fix, table collision). Additionally, we consider the grasp SE(3)-DiF at the instant t=T/2 (we aim to grasp an object in the middle of the trajectory). Finally, we want to impose that the relative position of the object with respect to the gripper in the grasping moment should be the same as the relative position in the placing. We impose this by first computing the pose in the object's frame  $H^o_{ee,t} = (H^w_{o,t})^{-1}H^w_{ee,t}$ , with  $H^w_{ee,t}$  being the end effector pose in the world frame at the instant t and  $H^w_{o,t}$  the pose of the object in the world frame at the instant t. We define the grasp-place pose similarity cost as  $c_{\text{grasp place similarity}}(\tau) = d_{\text{SO(3)}+\mathbb{R}^3}(H^o_{ee,T/2}, H^o_{ee,T})$ , that encourages the end effector pose w.r.t. the object frame to be the same in both the grasping moment and the placing moment.

### **Exact weighting of cost terms**

In Table IV we present the weighting of the individual cost terms that we have used to obtain the trajectories for these scenarios of having to pickup a mug and reorienting it to fullfil a desired final pose.

| Description                       | Cost                                  | Weight |
|-----------------------------------|---------------------------------------|--------|
| Grasp pose evaluation             | $E_{\theta}(\phi_{ee}(q_{t=T/2}), k)$ | 2.     |
| Trajectory smoothness             | $c_{\mathrm{smooth}}(\tau)$           | 10.    |
| Table collision avoidance         | $c_{\text{table coll.}}(\tau)$        | 20.    |
| Initial configuration fixing cost | $c_{fix}(\tau)$                       | 1.     |
| Pregrasping cost                  | $c_{\text{pregrasp}}(\tau)$           | 1.     |
| Grasp-place pose similarity cost  | Carnen place similarity (T)           | 10.    |

**TABLE IV:** This table summarizes the weighting of the individual cost terms that have been used for generating robot trajectories for the task of picking a mug in the first half of the trajectory and reorienting it to a desired final pose in the second half of the trajectory, as presented in Section IV-B & Section III-C.2.

3) Pick and place on shelves: In the following, we provide an extended presentation of the experiment of picking and placing on shelves. Similarly to the pick and reorient task, this experiment was chosen to evaluate the performance of SE(3)-DiF solving complex manipulation tasks jointly. This task is of high interest as both the set of affordable grasping poses and placing poses is very small due to the possible collisions with the shelves and therefore, jointly optimizing the trajectory and the grasp pose might be highly beneficial. The task is visualized in Figure 1. We evaluate the task similarly to the pick and reorient task. We generate 100 trajectories, execute them, and measure how close the grasped object is to the target pose in the last instant.

We consider the same objective function as for the pick and reorient task, but we also add a collision-avoidance cost to take the shelves collisions into account.

### Exact weighting of cost terms

In Table V we present the weighting of the individual cost terms that we have used to obtain the trajectories for these scenarios of picking a mug located inside a shelf and placing it in another desired final pose.

### D. Pointcloud based SE(3)-DiffusionFields

The presented work is focused on evaluating the performance of diffusion models as both 6D grasp generative models and cost functions in trajectory optimization. Thus, to avoid perception related uncertainty, in this work, we assume the

| Description                       | Cost                                      | Weight |
|-----------------------------------|-------------------------------------------|--------|
| Grasp pose evaluation             | $E_{\theta}(\phi_{ee}(q_{t=T/2}), k)$     | 1.     |
| Trajectory smoothness             | $c_{\mathrm{smooth}}(\tau)$               | 10.    |
| Table collision avoidance         | $c_{\text{table coll.}}(\tau)$            | 10.    |
| Initial configuration fixing cost | $c_{fix}(\tau)$                           | 10.    |
| Pregrasping cost                  | $c_{\text{pregrasp}}(\tau)$               | 10.    |
| Grasp-place pose similarity cost  | $c_{\text{grasp place similarity}}(\tau)$ | 1.     |
| Box collision cost (shelf)        | $c_{\rm box\ coll.}(\boldsymbol{\tau})$   | 10.    |

**TABLE V:** This table summarizes the weighting of the individual cost terms that have been used for generating robot trajectories for the task of picking and placing a mug inside a shelf as presented in Section III-C.3.

![](_page_17_Picture_2.jpeg)

Fig. 11: An illustration of the PoiNt-SE(3)-DiF architecture for learning 6D grasp pose distributions. The architecture is similar to the autodecoder-based one (Fig. 3) with a few modifications. We substitute the shape code embeddings and the object's pose transformation for a single pointcloud encoder  $\mathcal{E}_{\theta}$  that transforms an input pointcloud P into a latent code z. We model  $\mathcal{E}_{\theta}$  with a Vector Neuron (VN)-PointNetI, that encodes SO(3) equivariant features. We train the model to jointly match the different object's SDF values (sdf) and to minimize the denoising loss. We jointly learn the parameters for the pointcloud encoder  $\mathcal{E}_{\theta}$ , the features encoder  $F_{\theta}$  and, the decoder  $D_{\theta}$ .

object shape and pose are known. We assume that we can rely on state-of-the-art object pose detection and segmentation to estimate the object class and pose [68]. We apply an autodecoder approach [35] and learn a set of latent codes z that represent the different shapes. Then, in practice, given we know the exact object, we can retrieve the shape code z given we know the index of the object.

For completeness, in this experimental section, we evaluate the performance of SE(3)-DiF with a Pointcloud encoder instead of an autodecoder. We modify the architecture in Figure 3 and add a pointcloud encoder  $\mathcal{E}_{\theta}$ . We refer to this model as PoiNt-SE(3)-DiF. We present the modified architecture in Figure 11. We model the pointcloud encoder  $\mathcal{E}_{\theta}$  with a VN-PointNet [69]. The network outputs SO(3)-equivariant features that allow us to easily encode the orientation of the different objects. A similar network has been previously applied in [70] to learn the features of a graspable object.

We aim to evaluate the performance difference between the autodecoder based SE(3)-DiF and the pointcloud encoder based SE(3)-DiF models. We consider three scenarios for the autodecoder-based model: (i) Both object shape and pose are known, (ii) Only the shape is known and (iii) We don't know neither the shape nor the pose of the object. The case (i), where both pose and shape of the object are known, is presented in Sec. IV-A. For the cases when either the object pose or both pose and shape code are unknown, we rely on pointclouds for inferring them. We follow a similar inference approach to the one proposed in [35] and extended it to infer also the pose of the object  $H_o^w$ . Given a pointcloud  $P: \{x_n\}_{n=0}^N$  and the learned SDF function  $F_\theta^{sdf}$ , we infer the  $H_o^w$  by

$$\boldsymbol{H}_{o}^{w*} = \operatorname*{arg\,min}_{\boldsymbol{H}_{o}^{w}} \frac{1}{N} \sum_{n=0}^{N} F_{\boldsymbol{\theta}}^{sdf}(\boldsymbol{H}_{o}^{w} \boldsymbol{x}_{n}, \boldsymbol{z})$$
(31)

given z the shape code of a known object. Intuitively, (31) searches for the object pose  $H_o^w$  that makes the pointcloud pose to match the one of the learned SDF function. We can think of this optimization problem as an Iterative Closest Point (ICP) algorithm [71], but instead of matching two sets of points, we match a set of points with the SDF function. For the case when neither pose nor shape of the object is known, we infer both z and  $H_o^w$ 

$$H_o^{w*}, z^* = \underset{H_o^w, z}{\arg \min} \frac{1}{N} \sum_{n=0}^{N} F_{\theta}^{sdf}(H_o^w x_n, z) + ||z||^2,$$
 (32)

and we extend the optimization for both the shape code z and the object's pose  $H_o^w$ . We additionally add a L2 regularizer over z as proposed by [35].

We evaluate the performance of the autodecoder-based approaches and the pointcloud encoder-based model w.r.t. their success rate in generating successful grasps and the EMD. We additionally add 6DoF-Graspnet [6] as baseline to compare all the methods. We follow the same evaluation procedure from Section IV-A. We present the results of the evaluation in

![](_page_18_Figure_0.jpeg)

Fig. 12: Evaluation of the Success for picking with occlusions. PoiNt-SE(3)-DiF refers to the model with a pointcloud encoder, SE(3)-DiF (Rot) to the model in which the pose is infer from the pointcloud, SE(3)-DiF the model in which both the pose and shape are known, SE(3)-DiF (Z+Rot) the model in which both the object pose and shape codes are inferred from the pointcloud, and 6DoF-Graspnet [6].

Figure 12. In Figure 12, we name SE(3)-DiF the case where both object shape and pose are known, SE(3)-DiF (Rot) the case where the object's shape is known, and the pose is inferred by pointclouds with (31), SE(3)-DiF (Z+Rot) the case in which both object pose and shape are inferred with (32) and PoiNt-SE(3)-DiF the Pointcloud conditioned model. We observe a high performance in terms of both success rate and EMD for SE(3)-DiF, SE(3)-DiF (Rot) and for the PoiNt-SE(3)-DiF. We also observe that the best success rate and EMD was achieved by SE(3)-DiF, followed by SE(3)-DiF (Rot) and PoiNt-SE(3)-DiF. We hypothesize that this might be related to the unknown variables on each case. PoiNt-SE(3)-DiF needs to infer both the shape and the pose, while SE(3)-DiF assumes this to be known. We observe that SE(3)-DiF (Z+Rot) was not able to achieve a high success rate. We were not able to properly infer both the shape code and the pose jointly by (32). Therefore, if both the shape and the pose are unknown, we propose using PoiNt-SE(3)-DiF, while if the shape and the pose are known, we rather propose using the autodecoder approach. We observe, that all diffusion-based methods, except the SE(3)-DiF (Z+Rot) outperformed 6DoF-GraspNet in terms of both success rate and Earth Mover Distance. While the success rate of 6DoF-GraspNet is is close to the one of the diffusion models, the EMD decays alot. This evaluation infers that the samples obtained by 6DoF-GraspNet are less diverse and the generation collapses to some modes in the dataset without covering it all.