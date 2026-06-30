# 6-DOF Grasping for Target-driven Object Manipulation in Clutter

Adithyavairavan Murali<sup>1,2</sup>, Arsalan Mousavian<sup>1</sup>, Clemens Eppner<sup>1</sup>, Chris Paxton<sup>1</sup>, Dieter Fox<sup>1,3</sup>

Abstract—Grasping in cluttered environments is a fundamental but challenging robotic skill. It requires both reasoning about unseen object parts and potential collisions with the manipulator. Most existing data-driven approaches avoid this problem by limiting themselves to top-down planar grasps which is insufficient for many real-world scenarios and greatly limits possible grasps. We present a method that plans 6-DOF grasps for any desired object in a cluttered scene from partial point cloud observations. Our method achieves a grasp success of 80.3%, outperforming baseline approaches by 17.6% and clearing 9 cluttered table scenes (which contain 23 unknown objects and 51 picks in total) on a real robotic platform. By using our learned collision checking module, we can even reason about effective grasp sequences to retrieve objects that are not immediately accessible. Supplementary video can be found here.

#### I. Introduction

Grasping is a fundamental robotic task, but is challenging in practice due to imperfections in perception and control. Most commonly, grasp planning involves generating gripper pose configurations (3D position and orientation) that maximize a grasp quality metric on a target object in order to find a stable grasp. There are several factors that affect grasp stability, including object geometry, material, gripper contacts, surface friction, mass distribution, amongst several others [1], [2]. Most traditional approaches to grasping assume a separate perception system that can perfectly [1], or with some uncertainty [3], infer object information such as pose and shape. This is followed by physics-based grasp analysis [1], [4] or nearest-neighbour lookup on a database of pre-computed grasps [5]. These methods are slow [6], prone to perception error and do not generalize to novel objects.

Grasp synthesis is much harder in clutter, such as the example in Fig 1. The target object has to be grasped without any unwanted collisions with surrounding objects or the environment. In a real world application, a personal robot might be commanded to grasp a specific beverage from a narrow kitchen cabinet packed with other items. Grasps sampled agnostic of the clutter could end up in collision with the environment. Even if the gripper pre-shape is not in collision, it may be challenging to plan a collision-free and kinematically feasible path for the manipulator to achieve the gripper configuration. One would have to generate a diverse set of grasps since not all the grasps will be kinematically feasible to execute in the environment. Most model-based approaches in the grasping and task and motion planning literature assume perfect object knowledge or use an

<span id="page-0-0"></span>Fig. 1: Given an unknown target object (*left*) our proposed method leads to robust grasping (*right*) despite challenging clutter and occlusions. This is enabled by explicitly reasoning about successful and colliding grasps (*center*).

occupancy-grid representation for collision checking, which may not be reliable or practical in real-world settings [1], [7]–[9].

A large part of the difficulty lies in perception. In clutter, large and important parts of object geometry are occluded by other objects. Traditional shape matching techniques will find it extremely challenging to operate in such conditions, even when object geometry is known. In addition, getting quality 3D information is challenging and previous methods resort to using high quality depth sensors [10] or using multiple views [11], which would require observation-gathering exploratory movements impossible in confined spaces. This limits the deployment of such systems outside of controlled environments.

Recent works have explored data-driven methods for grasping unknown objects [10]–[16]. However, they mainly focus on the limited setting of planar-grasping and bin-picking. Some recent methods tackle the more difficult problem of generating grasps in SE(3) from 2D (image) [17], 2.5D (depth, multi-view) [11], [18], [19] and 3D (point cloud) [20]– [22] data. These works primarily consider the problem from an object-centric perspective or in bin-picking settings. We consider the problem of 6-DOF grasp generation in structured clutter using a learning-based approach. Our method uses instance segmentation and point cloud observation from just a single view. We follow a cascaded approach to grasp generation in clutter, first reasoning about grasps at an object level and then checking the cluttered environment for collisions. We use a learned collision checker, which evaluates grasps for collisions from just raw partial point cloud observations and works under varying degrees of occlusion. Specifically, we present the following contributions:

 A learning-based approach for 6-DOF grasp synthesis for novel objects in structured clutter, which uses a learned collision checker conditioned on the gripper information

<sup>\*</sup> This work is done while the first author was an intern at NVIDIA.  $^1$ NVIDIA {amousavian, ceppner, cpaxton, dieterf}@nvidia.com,  $^2$ The Robotics Institute, CMU amurali@cs.cmu.edu,  $^3$ University of Washington

- and on the raw point cloud of the scene.
- Showing that our approach, trained only with synthetic data, achieves a grasp accuracy of 80.3% with 23 realworld test objects in clutter. It also outperforms a clutteragnostic baseline approach of 6-DOF GraspNet [20] with state-of-the-art instance segmentation [23] by 17.6%.
- Demonstrating an application of our approach in moving blocking objects away out of the way to grasp a target object that is initially occluded and impossible to grasp.

# II. RELATED WORK

Grasping is a widely studied field in robotics ([1], [2], [12]). In the following we will focus our comparison on existing approaches that are data-driven and the aspects in which they differ from the proposed method.

Grasping in clutter vs. isolated objects: Among learningbased methods for grasping a significant amount focuses on dealing with isolated objects on planar surfaces ([18], [20], [24]–[26]). Our approach specifically tackles the problem of grasping objects from a cluster of multiple objects. This problem is significantly harder since the accessible workspace around the target object is severely limited, occlusions are more likely to hamper perception and predicting outcomes might be more difficult due to contact interactions between objects. Although multiple learning-based approaches for dealing with grasping in clutter exist ([11], [13], [14], [27]) we will show in the following that they differ from our approach in multiple aspects.

Bin-picking vs. structured clutter: Most learning-based grasping approaches for clutter deal with rather small and light objects that are spread in a bin ([13], [14], [27], [28]). In contrast our approach focuses on *structured* clutter. We define structured clutter as packed configurations of mostly larger, heavier objects. Examples include kitchen cupboards or supermarket shelves. Compared to the bin-picking setup successful grasps are more sparsely distributed in structured clutter scenarios. Collisions and unintended contact is often more catastrophic since objects have fewer stable equilibria when they are not located on a pile. Since avoiding collision becomes more important, structured clutter is more prominent in evaluations of model-based task-and-motion-planning. Our approach explicitly predicts grasp configurations that are in collision and can do so despite occlusions.

Planar vs. spatial grasping in clutter: Many learningbased grasp algorithms for clutter are limited to planar grasps, representing them as oriented rectangles or pixels in the image ([13], [14], [29], [30]). As a result, grasps lack diversity and picking up an object might be impossible given additional task or arm constraints. This limitation is less problematic in bin-picking scenarios where objects are small and light. In structured clutter, spatial grasping is unavoidable, otherwise pre-grasp manipulations are needed [31]. Those learningbased approaches that plan full grasp poses are either based on hand-crafted features ([32]–[34]) or have non-learned components [11]. Our approach uses a learned grasp sampler that predicts the full 6D grasp pose and accounts for unseen parts due to occlusions.

Model-based vs. model-free: A lot of planning approaches exist that tackle scenarios of grasping in structured clutter ([7], [31], [35]–[37]). These approaches rely on full observability and prior object knowledge. In contrast, our method does not require any object models and poses; grasps are planned based on raw depth images. In that regard, it is similar to other data-driven methods for clutter ([11], [13], [14], [29], [30]) but differs from techniques that use hand-engineered features ([32]–[34], [38]).

Target-agnostic vs. target-driven: Few approaches focus on grasping specific objects in clutter ([39], [40]). Our method is target-driven as it uses instance segmentation [23] to match grasps with objects.

# III. 6-DOF GRASP SYNTHESIS FOR OBJECTS IN CLUTTER

We consider the problem of generating 6-DOF grasps for unknown objects in clutter. The input to our approach is the depth image of the scene and a binary mask indicating the target object. In particular, we aim to estimate the posterior grasp distribution P(G<sup>∗</sup> |X), where X is the point cloud observation and G<sup>∗</sup> is the space of successful grasps. We represent g ∈ G<sup>∗</sup> as the grasp pose (R, T) ∈ SE(3) of an opened parallel-yaw gripper that results in a robust grasp when closing its fingers.

The distribution of successful grasps is complex, multimodal and discontinuous. The number of modes for a new object is not known a-priori and is determined by the geometry, size, and physics of the object. Additionally, small perturbations of a robust grasp could lead to failure due to collision or slippage from poor contact. Finally, cluttered scenes limit the robot workspace significantly. Although a part of an object might be visible it could be impossible to grasp if the gripper itself is a large object (such as the Franka Panda robot hand we use in our experiments) that leads to collisions with surrounding objects.

## *A. Overview of Approach*

Our cascaded grasp synthesis approach factors the estimation of P(G<sup>∗</sup> |X) by separately learning the grasp distribution for a single, isolated object P(G<sup>∗</sup> |Xo) and a discriminative model P(C|X, g) which we call *CollisionNet* that captures collisions C between gripper at pose g and clutter observed as X. X is the cropped point cloud of the scene and X<sup>o</sup> = Mo(X) is the segmented object point cloud, where M<sup>o</sup> is the instance mask of the target object.

The advantage of this factorization is twofold. First, it allows us to build upon prior work [20] which successfully infers 6-DOF grasp poses for single, unknown objects. Second, by explicitly disentangling the reasons for grasp success, i.e., the geometry of the target object and a collision-free/reachable gripper pose, we can reason beyond simple pick operations. As shown in a qualitative experiment in Sec. [IV-C](#page-5-0) we can use our approach to infer which object to remove from a scene to maximize grasp success of the target object.

Fig. [2](#page-2-0) shows an overview of our approach. During inference, a target object can be selected based on a state-of-the-art segmentation algorithm [23]. Given this selection we infer

<span id="page-2-0"></span>Fig. 2: Overview of our cascaded grasping framework. A local point cloud centered on the target object is cropped from the scene point cloud using instance segmentation. 6-DOF grasps are then generated and ranked by collisions with the scene.

possible successful grasps for the object ignoring clutter, and combine it with the collision results provided by CollisionNet.

In the following two sections, we will present both of these models. Note that our particular design decisions are based on comparisons with alternative formulations. In Sec. IV-A we will show how our approach outperforms variants that do not distinguish between grasp failures due to collisions and target geometry, or use non-learned components.

# B. 6-DOF Grasp Synthesis for Isolated Objects

We first want to learn a generative model for the grasps given the point cloud observation of the cluttered scene. Though this generative model is learned from a reference set of positive grasps, it is not completely perfect due to several reasons. As a result, we follow the approach presented in [20] to have a second module to evaluate and further improve these generated grasps. Conditioned on the point cloud and grasp, the evaluator predicts a quality score for the grasp. This information could also be used to incrementally refine the grasp. We also explore the importance of object instance information in all stages of the 6-DOF grasping pipeline, from grasp generation to evaluation, in the ablation study.

Variational Grasp Sampling: The grasp sampler is a conditional Variational Autoencoder [41] and is a deterministic function that predicts the grasp g given a point cloud  $X_o$  and a latent variable z.  $P(z) = \mathcal{N}(0, I)$  is a known probability density function of the latent space. The likelihood of the grasps can be written as such:

$$P(G|X_o) = \int P(G|X_o, z)P(z)dz \tag{1}$$

<span id="page-2-1"></span>Optimizing Eqn 1 is intractable as we need to integrate over all the values of the latent space [41]. To make things tractable, an encoder  $Q(z|X_o,g)$  is used to map each pair of point cloud  $X_o$  and grasp g to the latent space z while the decoder reconstructs the grasp given the sampled z. The encoder and decoder are jointly trained to minimize the reconstruction loss  $\mathcal{L}(\hat{g},g)$  between the ground truth grasps  $g \in G^+$  and predicted grasps  $\hat{g}$ , with the KL-divergence penalty between the distribution Q and the normal distribution  $\mathcal{N}(0,I)$ :

$$\mathcal{L}_{VAE} = \sum_{z \sim Q, g \sim G^*} \mathcal{L}(\hat{g}, g) - \alpha \mathcal{D}_{KL}[Q(z|X_o, g), \mathcal{N}(0, I)]$$

Note that the input to the VAE is the point cloud of the target object segmented from the scene with instance mask.

To combine the orientation and translation loss, we define the reconstruction loss as  $\mathcal{L}(\hat{g},g) = \frac{1}{n}\sum ||\mathcal{T}(g;p) - \mathcal{T}(\hat{g};p)||$  where  $\mathcal{T}$  is the transformation of a set of predefined points p on the robot gripper. During inference, the encoder Q is discarded and latent values are sampled from  $\mathcal{N}(0,I)$ . Both the encoder and decoder are based on the PointNet++ architecture [42], where each point has a feature vector along with 3D coordinates. The features of each input point of the point cloud are concatenated to the grasp g and the latent variable z in the encoder and decoder respectively.

Though instance information can give a strong prior about the object, it is not perfect in practice. This is especially the case in cluttered scenarios where objects are occluded or very close to each other, resulting in noisy under and oversegmentation. When rendering the segmentation in simulation, we add random salt-and-pepper noise to the object boundaries and randomly merge partially occluded objects to neighboring ones in image space, to mimic the imperfections of instance segmentation methods on the real images.

**Grasp Evaluation:** Though the grasp sampler is trained with only positive grasps, it may still predict failed grasps which need to be identified and removed. We train an evaluator that predicts a grasp score  $P(S|X_o, g)$ , with the training data consisting of positive  $G_S^+ = G^+$  and negative  $G_S^- = G^$ grasps. The evaluator's input is  $X_o$ , the point cloud of the object segmented from the full scene. Since the space of all possible 6-DOF grasp poses is large, it is not possible to sample all the negative grasps for training the grasp evaluator  $P(S|X_o, g)$ . Therefore, during training we sample from true negatives but also sample hard negative grasps by perturbing positive grasps with a small translation and orientation and choosing those that are in collision with the object or are too far from the object to grasp the object. At test time on the robot, the grasps are ranked by their evaluator scores and only those above a threshold are selected.

**Grasp Refinement:** A significant proportion of the grasps rejected by the evaluator are actually in close proximity to robust grasps. This insight could be exploited to perform a local search in the region of g to iteratively improve the evaluator score. We concretely want to sample  $\Delta g$  to increase the probability of success, i.e.,  $P(S|\Delta g+g,X_o)>P(S|g,X_o)$ . The refinement was found using gradient descent

in [20]. In practice, computing gradients is not fast. Instead, we use Metropolis-Hastings sampling where a random  $\Delta g$  is sampled and with probability of  $\frac{P(S|g+\Delta g,X_o)}{P(S|g,x)}$  grasp  $g+\Delta g$  is accepted. We observe that this sampling scheme yields similar performance to the gradient-based one while it is computationally twice as fast.

# C. Collision Detection for Grasps in Clutter: CollisionNet

CollisionNet predicts a clutter-centric collision score P(C|X,g) given the full scene information X. The training data for CollisionNet is  $G_C^+ = \{g | g \in G_{ref}, \neg \Psi(g, \mathbf{x})\}$  and  $G_C^- = \{g | g \in G_{ref}, \Psi(g, \mathbf{x})\}$ . The ground truth labels are generated in simulation with a collision checker  $\Psi$  assuming full state information x. In each batch, we used balanced sampling of grasps within the subsets of the reference set  $G_{ref}$ , which consists of the positive and negative sets ( $G^+$ ,  $G^{-}$ ), hard-negatives generated by perturbing positive grasps  $(G_{hn}^{-})$  and grasps in free space  $G_{free}$ . We observed that balanced sampling improved the stability of training and generalization at test time over uniform sampling from  $G^+ \cup G^-$ . Similar to the grasp evaluator, the scene/object point cloud  $X/X_o$  and gripper point cloud  $X_g$  are combined into a single point cloud by using an extra indicator feature that denotes whether a point belongs to the object or to the gripper. The PointNet++ [42] architecture then uses the relative information between gripper pose g and point clouds for classifying the grasps. CollisionNet is optimized using cross entropy loss.

#### D. Implementation Details

Training data is generated online by arranging multiple objects randomly at their stable poses. Objects are added to the scene with rejection sampling poses to ensure they are not colliding with existing clutter. In order to generate grasps for the scenes, we combine the positive and negative grasps of each object from [20] which includes a total of 126 objects from several categories (boxes, cylinders, bowls, bottles, etc.). From each scene we take multiple 3D crops centered on the object (with some noise) along with grasps that are inside the crop. The cropped point cloud of the 3D box is down-sampled to 4096 points. All the samplers and VAEs are based on PointNet++ [42] architecture and the dimension of latent space is set to 2. During inference, object instances are segmented with [23]. The VAE sampler generates the grasps given the point cloud of the target object by sampling 200 latent values. Grasps are further refined with 20 iterations of Metropolis-Hastings. The whole inference takes 2.5s on a desktop with NVIDIA Titan XP GPU.

#### IV. EXPERIMENTAL EVALUATION

### <span id="page-3-0"></span>A. Ablation analysis and Discussion

**Evaluation Metrics:** Following [20], we used two metrics for evaluating the generated grasps: success rate and coverage. Success rate is the percentage of grasps that succeed grasping the object without colliding and coverage is the percentage of sampled ground truth grasps that are within 2cm of any of the generated grasps. The ablations were done in simulation

<span id="page-3-1"></span>Fig. 3: Comparing the VAE sampler and Surface Normal Sampler. The number next to the legend is the area under curve (AUC) and the VAE sampler has a higher AUC.

<span id="page-3-2"></span>Fig. 4: CollisionNet outperforms the Voxel-based approach in both success and coverage. The Voxel-based without Target Object ablation only considers collisions with the scene.

with a held-out test set of 15 unseen objects of the same categories arranged at random stable poses in 100 different scenes. Physical interactions are simulated using FleX [43]. Area under curve (AUC) of the success-coverage plot is used to compare different variation of the methods in the ablations.

Learned vs. Surface Normal Based Grasp Sampler: The first ablation study we consider is the effect of using a learned VAE to sample grasps in comparison with a geometric baseline. This baseline generates grasps by sampling random points on the object along surface normals, with random standoff, and random orientation along the surface normal. Fig. 3 shows that our learned VAE sampler yields more grasp coverage. It is worth noting that the surface-normal based sampler performed well for simpler shapes like boxes but failed to generate grasps for more complex geometry with rim, handles, etc.

CollisionNet vs Voxel-Based Collision Checking: We compared the effectiveness of CollisionNet with a voxel-based heuristic commonly used (such as in MoveIt! [44]) for obstacle avoidance in unknown 3D environments. In our case, from each object, 100 points are sampled using farthest point sampling. Each sampled point is represented by a voxel cube of size 2cm. Collision checking is done by checking the intersection of the gripper mesh with any of the voxels. As shown in Fig. 4, CollisionNet outperforms the voxel-based heuristic in terms of precision and coverage. Qualitatively, we observed that the voxel-based representation fails to capture

<span id="page-4-0"></span>Fig. 5: Examples where the voxel-based heuristic fails to predict collisions but CollisionNet succeeds. These false positives are due to missing points (region highlighted by dotted circle) from occlusion. These grasps will lead to critical collisions if executed.

collision when the gripper mesh intersects with occluded parts of objects, or if there is missing depth information (see Fig. 5). In cases where the voxel-based collision checking fails, CollisionNet has 89.7% accuracy in classifying the collisions correctly.

The voxel-based approach also has several false negatives by rejecting good grasps that are slightly penetrating voxels corresponding to points on the target object, as the voxels expand the spatial region for collision checking. Without considering the voxels on the target object for collisions, the coverage decreases marginally (blue curve in Fig. 4). The grasp success also decreases as grasps that are actually colliding with the target object are not pruned out. CollisionNet does not suffer from such biases and can reason about relative spatial information in the partial point clouds.

**Single-stage vs. Cascaded Evaluator:** Instead of a cascaded grasp generation approach, one could also use a *single-stage* sampler and evaluator with object instance information. Once the grasps are sampled, there is only a single evaluator that directly estimates  $P(S, \neg C|X, g)$ . The positive training set is  $G_{SC}^+ = \{g|g \in G^+, \neg \Psi(g, \mathbf{x})\}$  while the negative set is  $G_{SC}^- = \{g|g \in G^+, \Psi(g, \mathbf{x})\} \cup \{g|g \in G^-\}$ . As a result, some positive grasps  $g \in G^+$  will be in collision resulting in lower scores. An example of the input data to this baseline is shown in Fig. 7(b), where the indicator mask of the target object is passed as an additional feature to the PointNet++ architecture. We found that the cascaded model outperformed the single-stage model, as shown in Fig 6.

This improvement is due to two factors. First, the VAE is far more proficient in learning grasps from an object-centric observation than from scene-level information. Second, the cascaded architecture imposes an abstraction between having a grasp evaluator that is singly proficient in reasoning about grasp robustness and CollisionNet that is proficient in predicting collisions.

Role of Object Instance Segmentation: We compared our cascaded grasp sampling approach to an instance-agnostic baseline. Without instance information, the baseline is a single-stage grasp planner that uses the point cloud of the scene, since we cannot get a object-centric input. An example of the input data to this baseline is shown in Fig. 7(a). From the ablation shown in Fig. 6, we found that our cascaded grasp

<span id="page-4-2"></span>Fig. 6: Our cascaded approach demonstrates much higher success and coverage compared to a single-stage and instance-agnostic model.

<span id="page-4-1"></span>Fig. 7: Representative example of the data provided to the different grasping architectures a) single-stage model without object instance information b) single-stage model with object instance mask used as a feature vector along with the point cloud c) our cascaded model to sample with object-centric point cloud and evaluate for collisions with clutter-centric data. The target object is colored in blue.

sampler (using instance information and CollisionNet) had a AUC of 0.22 and outperformed the object instance-agnostic baseline in terms of both success and coverage, which had a AUC of 0.02. A common failure mode of the instance-agnostic model is that the variational sampler gets confused as to which object to grasp in the scene, with the latent space being potentially mapped to grasps for multiple objects and degrading the overall grasp quality for all the objects.

# B. Real Robot Experiments

In our robot experiments, we wanted to show that our cascaded grasp synthesis approach (1) transfers to the real world despite being trained only in simulation; (2) has competitive performance for target-driven grasping in real clutter scenes and (3) outperforms baseline methods using the clutter-agnostic 6-DOF GraspNet implementation [20] with instance segmentation and voxel-based collision checking. Our physical setup consists of a 7-DOF Franka Panda robot with a parallel-jaw gripper. We used a Intel Realsense D415 RGB-D camera mounted on the gripper wrist for perception. We execute the grasps in a open-loop fashion where the robot observes the scene once, generates the grasps and then executes solely based on the accurate kinematics of the robot. We found open-loop execution to work reasonably well in our setting. CollisionNet only considers collisions between the gripper and the clutter. We also use occupancy voxel-grid

<span id="page-5-3"></span>Fig. 8: Application of our approach in retrieving a partly occluded mug (highlighted in (a)). The blocking objects are ranked (colored in (b), red being most inhibiting) and removed from the scene. The target object is finally grasped in (f).

<span id="page-5-1"></span>Fig. 9: Scenes used for testing. See accompanying video for grasp performance.

collision checking on top of CollisionNet to make sure that the rest of the manipulator arm does not collide with the clutter and table during motion planning. We compared the performance of the method on 9 different scenes (see Fig [9\)](#page-5-1) with the fixed order (pre-computed randomly) of objects to be grasped. A grasp was considered a success if the robot could grasp the object within two consecutive attempts on the same scene. One could choose the order in which all the target objects are completely visible. To make the problem more challenging, half of the chosen target objects were occluded. To generate grasps, a batch size of 200 latents were sampled and the grasps that have scores lower than a threshold for each of the evaluator is filtered out. From the remainder of grasps, the one with maximum score is chosen to be executed.

TABLE I: Real Robot experiments

<span id="page-5-2"></span>

| Approach                                     | Trials | Performance (%) |
|----------------------------------------------|--------|-----------------|
| 6-DOF GraspNet [20] + Ins. Segmentation [23] | 32/51  | 62.7            |
| Object Instance                              | 31/51  | 60.7            |
| Object Instance + CollisionNet (Ours)        | 41/51  | 80.3            |

The results are summarized in Table [I.](#page-5-2) Our framework achieves a success rate of 80.3% and outperforms the baseline 6 DOF-GraspNet approach by 17.6%. Furthermore, without CollisionNet, our model performance degrades substantially. The two failure cases are the grasps that are colliding with the object but object centric evaluator predicts high score

for them. These grasps are filtered out by CollisionNet. The second failure mode pertains to the fact that the voxel-based representation cannot capture all collisions.

## <span id="page-5-0"></span>*C. Application: Removing Blocking Objects*

Consider scenarios such as that shown in Fig. [8,](#page-5-3) where the target object is being blocked by other objects and none of the sampled grasps are kinematically feasible. To accomplish this task, the model needs to generate potential grasps for the target object even though the target object is not physically reachable (detected by low scores from CollisionNet). Given the potential grasps, we can identify which objects are interfering with the generated grasps for the target object. The blocking object j is chosen to be the one with the largest increase in collision scores when removing the corresponding object points from the scene point cloud i.e. α<sup>j</sup> = P(C|Xˆ <sup>j</sup> , g) − P(C|X, g). The objects are colored by this ranking metric α<sup>j</sup> in Fig. [8\(](#page-5-3)b), with the red object being the most blocking. The modified point cloud Xˆ <sup>j</sup> , which hallucinates the scene without object j, is implemented by merging the object's instance mask with that of the table and projecting corresponding points to the table plane. Grasps are then generated for the blocking object and removed from the scene. Collision-free grasps can now be generated for the unoccluded target object for the robot to recover it. Target objects can be specified by any down-stream task but in this use case, it is specified by choosing the segmentation corresponding to the target object.

# V. CONCLUSION

We present a learning-based framework for 6-DOF grasp synthesis for novel objects in structured clutter settings. This problem is especially challenging due to occlusions and collision avoidance which is critical. Our approach achieves a grasp accuracy of 80.3% in grasping novel objects in clutter on a real robotic platform despite being only trained in simulation. A key failure mode of our approach is that it only considers gripper pre-shape collisions by design and hence motion planning could still fail on generated grasps. In future work, we hope to consider trajectory generation in grasp planning and explore the use of our approach in task planning applications. We also aim to apply this framework in grasping objects from challenging environments like kitchen cabinets and handle the case of retrieving stacked objects in structured clutter.

## REFERENCES

- [1] D. Prattichizzo and J. Trinkle, "Grasping," *Springer handbook of robotics*, pp. 671–700, 2008.
- [2] A. Bicchi and V. Kumar, "Robotic grasping and contact: a review," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2000.
- [3] J. Mahler, S. Patil, B. Kehoe, J. van den Berg, M. Ciocarlie, P. Abbeel, and K. Goldberg, "GP-GPIS-OPT: Grasp planning with shape uncertainty using gaussian process implicit surfaces and sequential convex programming," *IEEE International Conference on Robotics and Automation (ICRA)*, 2015.
- [4] F. Pokorny and D. Kragic, "Classical grasp quality evaluation: New algorithms and theory," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2013.
- [5] H. D. Corey Goldfeder, Matei Ciocarlie and P. K. Allen, "The columbia grasp database." in *IEEE International Conference on Robotics and Automation (ICRA)*, 2009.
- [6] C. Goldfeder and P. K. Allen, "Data-driven grasping," *Autonomous Robots*, vol. 31, no. 1, pp. 1–20, 2011.
- [7] N. Kitaev, I. Mordatch, S. Patil, and P. Abbeel, "Physics-based trajectory optimization for grasping in cluttered environments," in *IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2015, pp. 3102–3109.
- [8] M. Dogar and S. Srinivasa, "Push-grasping with dexterous hands: Mechanics and a method," *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2010.
- [9] D. Berenson, R. Diankov, K. Nishiwaki, S. Kagami, and J. Kuffner, "Grasp planning in complex scenes," *International Conference on Humanoid Robots*, 2007.
- [10] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg, "Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics," *RSS*, 2017.
- [11] A. ten Pas, M. Gualtieri, K. Saenko, and R. Platt, "Grasp pose detection in point clouds," *The International Journal of Robotics Research*, vol. 36, no. 13-14, pp. 1455–1473, 2017.
- [12] J. Bohg, A. Morales, T. Asfour, and D. Kragic, "Data-driven grasp synthesisa survey," *IEEE Transactions on Robotics*, 2014.
- [13] S. Levine, P. Pastor, A. Krizhevsky, and D. Quillen, "Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection," *International Symposium on Experimental Robotics (ISER)*, 2016.
- [14] L. Pinto and A. Gupta, "Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours," *IEEE International Conference on Robotics and Automation (ICRA)*, 2016.
- [15] A. Gupta, A. Murali, D. Gandhi, and L. Pinto, "Robot learning in homes: Improving generalization and reducing dataset bias," *Neural Information Processing Systems (NeurIPS)*, 2018.
- [16] D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke, and S. Levine, "Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation," *Conference on Robot Learning*, 2018.
- [17] A. Murali, L. Pinto, D. Gandhi, and A. Gupta, "CASSL: Curriculum accelerated self-supervised learning," *IEEE International Conference on Robotics and Automation*, 2018.
- [18] Q. Lu, K. Chenna, B. Sundaralingam, and T. Hermans, "Planning multifingered grasps as probabilistic inference in a learned deep network," *arXiv preprint arXiv:1804.03289*, 2018.
- [19] X. Yan, J. Hsu, M. Khansari, Y. Bai, A. Pathak, A. Gupta, J. Davidson, and H. Lee, "Learning 6-dof grasping interaction via deep 3d geometryaware representations," in *IEEE International Conference on Robotics and Automation (ICRA)*, May 2018.
- [20] A. Mousavian, C. Eppner, and D. Fox, "6-DOF GraspNet: Variational grasp generation for object manipulation," *International Conference on Computer Vision*, 2019.
- [21] C. Choi, W. Schwarting, J. DelPreto, and D. Rus, "Learning object grasping for soft robot hands," in *IEEE Robotics and Automation Letters*. IEEE, 2018, pp. 2370 – 2377.
- [22] H. Liang, X. Ma, S. Li, M. Gorner, S. Tang, B. Fang, F. Sun, and J. Zhang, "Pointnetgpd: Detecting grasp configurations from point sets," in *IEEE International Conference on Robotics and Automation*, 2019.
- [23] C. Xie, Y. Xiang, A. Mousavian, and D. Fox, "The best of both modes: Separately leveraging rgb and depth for unseen object instance segmentation," in *Conference on Robot Learning (CoRL)*, 2019.
- [24] I. Lenz, H. Lee, and A. Saxena, "Deep learning for detecting robotic grasps," *IJRR*, 2015.

- [25] D. Kappler, J. Bohg, and S. Schaal, "Leveraging big data for grasp planning," in *IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2015, pp. 4304–4311.
- [26] X. Yan, J. Hsu, M. Khansari, Y. Bai, A. Pathak, A. Gupta, J. Davidson, and H. Lee, "Learning 6-dof grasping interaction via deep geometryaware 3d representations," in *IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2018, pp. 1–9.
- [27] J. Mahler, F. T. Pokorny, B. Hou, M. Roderick, M. Laskey, M. Aubry, K. Kohlhoff, T. Krger, J. Kuffner, and K. Goldberg, "Dex-net 1.0: A cloud-based network of 3d objects for robust grasp planning using a multi-armed bandit model with correlated rewards," *IEEE International Conference on Robotics and Automation (ICRA)*, 2016.
- [28] M. Gualtieri, A. Ten Pas, K. Saenko, and R. Platt, "High precision grasp pose detection in dense clutter," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2016, pp. 598–605.
- [29] J. Mahler and K. Goldberg, "Learning deep policies for robot bin picking by simulating robust grasping sequences," *Conference on Robot Learning*, 2017.
- [30] A. Zeng, S. Song, K.-T. Yu, E. Donlon, F. R. Hogan, M. Bauza, D. Ma, O. Taylor, M. Liu, E. Romo *et al.*, "Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching," in *IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2018, pp. 1–8.
- [31] M. Dogar and S. Srinivasa, "A framework for push-grasping in clutter," *Robotics: Science and systems VII*, vol. 1, 2011.
- [32] E. Klingbeil, D. Rao, B. Carpenter, V. Ganapathi, A. Y. Ng, and O. Khatib, "Grasping with application to an autonomous checkout robot," in *2011 IEEE international conference on robotics and automation*. IEEE, 2011, pp. 2837–2844.
- [33] A. Herzog, P. Pastor, M. Kalakrishnan, L. Righetti, J. Bohg, T. Asfour, and S. Schaal, "Learning of grasp selection based on shape-templates," *Autonomous Robots*, vol. 36, no. 1-2, pp. 51–65, 2014.
- [34] A. Makhal, F. Thomas, and A. P. Gracia, "Grasping unknown objects in clutter by superquadric representation," in *2018 Second IEEE International Conference on Robotic Computing (IRC)*. IEEE, 2018, pp. 292–299.
- [35] A. Cosgun, T. Hermans, V. Emeli, and M. Stilman, "Push planning for object placement on cluttered table surfaces," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2011, pp. 4627–4632.
- [36] J. E. King, J. A. Haustein, S. S. Srinivasa, and T. Asfour, "Nonprehensile whole arm rearrangement planning on physics manifolds," in *IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2015, pp. 2508–2515.
- [37] W. C. Agboh and M. R. Dogar, "Real-time online re-planning for grasping under clutter and uncertainty," in *2018 IEEE-RAS 18th International Conference on Humanoid Robots (Humanoids)*. IEEE, 2018, pp. 1–8.
- [38] D. Fischinger, A. Weiss, and M. Vincze, "Learning grasps with topographic features," *The International Journal of Robotics Research*, vol. 34, no. 9, pp. 1167–1194, 2015.
- [39] E. Jang, S. Vijayanarasimhan, P. Pastor, J. Ibarz, and S. Levine, "Endto-end learning of semantic grasping," *arXiv preprint arXiv:1707.01932*, 2017.
- [40] M. Danielczuk, A. Kurenkov, A. Balakrishna, M. Matl, D. Wang, R. Martn-Martn, A. Garg, S. Savarese, and K. Goldberg, "Mechanical search: Multi-step retrieval of a target object occluded by clutter," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2019.
- [41] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," *International Conference on Learning Representations (ICLR)*, 2014.
- [42] H. S. Charles R Qi, Li Yi and L. J. Guibas, "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." *Neural Information Processing Systems (NeurIPS)*, 2017.
- [43] N. C. Miles Macklin, Matthias Muller and T.-Y. Kim, "Unified particle physics for real-time applications," *ACM Transactions on Graphics (TOG)*, 2014.
- [44] S. Chitta, I. Sucan, and S. Cousins, "Moveit![ros topics]," *IEEE Robotics & Automation Magazine*, vol. 19, no. 1, pp. 18–19, 2012.