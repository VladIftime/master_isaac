# Scooping Manipulation Via Motion Control With a Two-Fingered Gripper and Its Application to Bin Picking

Tierui He<sup>10</sup>, Shoaib Aslam<sup>10</sup>, Zhekai Tong<sup>10</sup>, Graduate Student Member, IEEE, and Jungwon Seo<sup>10</sup>, Member, IEEE

Abstract—This letter introduces a new method for scooping manipulation in constrained environments. The presented technique lends itself to relatively thin objects, with a large width-to-thickness ratio, which are commonly seen in many domestic and industrial tasks yet can cause considerable difficulty in handling. We present a novel approach to scooping based on a mobility analysis that leads to a new way to perform the operation via motion control with a two-fingered gripper. Our method thus expands the ways in which the manipulation of scooping can be performed, without necessitating underactuated control or a compliant mechanism. Our variable-length digit is suggested as a key facilitator for realizing the motion-controlled scooping maneuver. We present an extensive set of experiments showing the effectiveness of our approach in various scenarios featuring a constrained workspace, including bin picking from clutter.

Index Terms—Grasping, grippers and other end-effectors.

#### I. INTRODUCTION

HIN objects can be quite challenging to handle. For example, consider the situation depicted in Fig. 1, showing a robot trying to pick a plastic card lying on a flat support surface. We humans accomplish that task by taking advantage of our marvelous hand, with a high mobility and a sophisticated hybrid rigid-soft architecture, as well as advanced multi-modal sensing and robust control skills. These capabilities as a whole, which we almost take for granted, have no parallel in the robots at present. A careful arrangement is necessary to enable a common robot, such as the one with only two fingers shown in Fig. 1, to accomplish the picking task. Suction gripping, or any other means of exerting bilateral "sticking" forces, can be effective in some cases; however, these also have limitations, such as the necessity of a custom end-effector, the dependence on contact surface conditions, the difficulty with secondary manipulation after successful picking, among others.

Manuscript received February 24, 2021; accepted June 22, 2021. Date of publication July 1, 2021; date of current version July 15, 2021. This letter was recommended for publication by Associate Editor N. Wang and Editor H. Liu upon evaluation of the reviewers' comments. This work was supported by Hong Kong RGC ECS 26209319 and Hong Kong ITF ITS/240/17FX, ITS/018/17FP, and ITS/104/19FP. (Corresponding author: Tierui He.)

The authors are with the Hong Kong University of Science and Technology (HKUST), Clear Water Bay, Hong Kong (e-mail: theae@connect.ust.hk; sasla-maa@connect.ust.hk; ztong@connect.ust.hk; junseo@ust.hk).

This article has supplementary downloadable material available at https://doi.org/10.1109/LRA.2021.3093896, provided by the authors.

Digital Object Identifier 10.1109/LRA.2021.3093896

Fig. 1. (a) Picking through our scooping manipulation. (b) Ad hoc picking ending in failure.

This study presents a robotic scooping technique that is effective for the card picking task, for example. Fig. 1(a) shows how our scooping manipulation proceeds with a two-fingered gripper. While the blue finger presses down on the card, the yellow thumb gets through the space between the card and the support surface. A pinch grasp on the card's faces is finally obtained. This way of manipulation, scooping, has been a topic of interest. What makes our approach unique is a novel mobility analysis that enables motion-control-based scooping, which can be implemented with a common two-fingered gripper retrofitted with our variable-length digit (the length of the thumb in Fig. 1(a) is controllable). Underactuated control or a compliant mechanism is not needed.

Through a series of experiments performed autonomously, the advantages of our method will be clarified over ad hoc picking approaches such as the one shown in Fig. 1(b) where the robot was controlled to directly pinch-grasp the edge pair of the card. These experiments feature a range of practical yet challenging settings with a constrained workspace, from an object of interest lying on a support surface to bin picking tasks in which the object to pick is placed on an unordered cluster of other object instances.

#### II. BACKGROUND AND RELATED WORK

The challenges of handling objects with a thin profile have motivated many interesting robotic manipulation solutions to

2377-3766 © 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

develop. One notable recent approach is to take advantage of compliance and underactuation. Although these give rise to increased system complexity and unconstrained mobility, the resulting passive and natural dynamics has proved useful for improving the robustness of picking operations. The effectiveness of a soft fingertip in thin object picking was demonstrated in [1]. Underactuated hands with a compliant mechanism were used for the manipulation of scooping in [2], [3] by inducing suitable interaction between the fingers and the environment, and for a human-inspired process of the flip-and-pinch manipulation in [4] which was shown to be effective in picking various thin objects. In [5], the energy interaction modeling between a deformable thin object and a soft gripper facilitated picking a sheet of paper in a dynamic manner. In [6]–[8], grippers with movable contact surfaces that are capable of picking thin objects were presented. Beyond picking, dexterous insertion of thin objects through in-hand manipulation was investigated in [9], with applications to dry cell battery insertion, small part assembly, etc.

Our presented manipulation technique for object scooping is applicable not only to an isolated object lying on a support surface but also to a workspace cluttered with a multitude of object instances. Therefore, it can facilitate bin picking—singulating and picking items out of a cluttered bin. The importance of bin picking is proved by its ubiquity: the situation is very common in a wide variety of material/part handling scenarios. However, the complexity of bin picking has still to be fully addressed by the robot systems at present. One critical issue in bin picking is to choose the right gripper. The combination of a multi-fingered and a suction gripper, which can be applied to multi-affordance grasping [10], proved useful in a competition setting (for example, Amazon Picking Challenge). As for object detection and grasp planning for bin picking, learning-based, data-driven approaches have recently received considerable attention [11]–[15]. In the practice of bin picking, our recent work [16] showed that interacting with the clutter directly, in the form of a controlled quasistatic push, can facilitate singulating and securing each item.

# III. FRAMEWORK FOR MOTION-CONTROLLED SCOOPING

In [2], the manipulation of scooping is modeled using a planar manipulation setting illustrated in Fig. 2. Here a linear thumb is controlled to slide on a flat surface with an angle of attack ψ and to push a relatively thin object lying on the surface, against a wall formed by another fixture, shown as the linear finger. The thumb then penetrates under the object held by the finger. A pinch grasp is finally obtained as the thumb and the finger move closer. In this section, we study a more generalized case and perform a mobility analysis that enables our novel motion-control-based scooping.

## *A. Key Assumption*

Before presenting our generalized technique, we first make a key assumption. Successful scooping is contingent on the thumb being able to initially get through between the object and the support surface, that is, the transition from Fig. 2(a) to (b). This seems to happen quite reproducibly in practice with a thumb

Fig. 2. (a) Thin rectangular object lying on a flat surface. It is being pushed by the linear thumb against the finger placed vertically. These are all rigid bodies. (b) After the push, the thumb gets in the space between the object and the surface, and supports the object.

that is motion-controlled to simply push the object horizontally; see the video attachment. However, rigid body mechanics alone is inadequate to explain how it is conceptually possible for the motion-controlled thumb to take the object off from the surface vertically and then penetrate into the space between the object and the surface, without deliberate force control to exert forces with a nonzero vertical component on the object. Rigorously speaking, the workable initial penetration in practice may be accounted for by the dynamics of microscopic deformation at the point of contact between the thumb, the object, and the surface. A detailed examination of the phenomenon lies beyond the scope of this study. Instead, our framework for scooping will be based on rigid body mechanics and the following key assumption that the initial penetration is achievable:

*Assumption 1:* Consider a planar rigid manipulation setting, shown in Fig. 2(a), featuring a motion-controlled linear thumb pushing a flat-bottomed object on a flat, level support surface against another fixture that blocks the object. If the thumb's angle of attack ψ is sufficiently small, it can penetrate into the space between the object and the surface.

Similar assumptions have been made in the literature (for example, [2]), though implicitly.

# *B. Mobility Analysis for Scoopability*

In this work, we investigate a situation depicted in Fig. 3(a), generalized from Fig. 2(a) by allowing the finger to make contact on top of the object. The finger is thus represented simply by the contact point F in Fig. 3(a). This generalization is motivated by the observation that a contact placed on top of the object is also able to counterbalance the thumb's pushing force, as the finger wall does in Fig. 2, by means of both friction (contact force tangent to the object's surface) and geometry (contact normal with a nonzero horizontal component due to curved object shape). Consequently, Assumption 1 is also applicable to the transition from Fig. 3(a) to Fig. 3(b), during which the thumb tilts up the object after a push similarly to Fig. 2(b). This scooping situation with F on top of the object in Fig. 3 has also been studied in the literature, e.g. [3].

Contrary to that study, which presented a force analysis and then investigated the applicability of an underactuated, compliant mechanism, here we perform a mobility analysis for motioncontrol-based scooping manipulation. More specifically, we examine when it would be possible for the object to be caught and finally pinched without getting lost while F is kept fixed, which is promising for scooping using the minimalist gripper

Fig. 3. (a) The point finger *F* on the object's top face replaces the linear finger in Fig. 2. (b) First-order mobility analysis performed with the three contact normals at *F*, *G* (contact between the object and the surface), and *T* (contact between the object and the thumb). (c) Scoopable object with a semi-elliptical cross section. (d) The object is not scoopable at the configuration shown.

model—the system of the linear thumb and the point finger controlled by relative repositioning. The shaded triangular area in Fig. 3(b), with the minus sign "−," represents the collection of the feasible twists of the object resolved according to Reuleaux's method [17] (see [18], [19] for examples in different contexts). The triangle is delimited by the three contact normals at F, T, and G, and is interpreted as the region of the allowable centers of rotation for the object on the plane, to the first order due to the use of the contact normals. The "−" sign denotes the sense of the rotation, that is, into the page. Therefore, at the contact configuration of Fig. 3(b), the object can only rotate clockwise instantaneously about a point belonging to the triangular region without penetrating into the thumb, the finger, or the surface. In other words, if the gripper is held fixed, it is impossible for the object to escape from the gripper through the gap between F and G because the point coincident with G is not allowed to move toward the halfspace on the right of its current contact normal. Our definition of *scoopability* conceptualizes this observation:

*Definition 1:* At the contact configuration of the three contacts F, T, and G modeled in Fig. 3(b), the object is said to be *scoopable* if the motion of the point at G is forbidden toward the halfspace on the right of the normal at G.

If the object moves with a feasible twist at a scoopable configuration, in general the point coincident with G moves toward the thumb while the contacts T and F break free from the gripper. One way to induce the feasible motion is then to apply a pushing force at T or F by "closing" the gripper, for example, moving the thumb horizontally toward the fixed finger. The object can then be captured and pinch-grasped finally as it is forced to slide upward on the thumb. This accounts for what happened in Fig. 1(a).

Fig. 3(c) presents another example of a scoopable object with a semi-elliptical cross section. It can be seen that the convexity of the object's geometry at F helps to secure the

Fig. 4. With *F* as a frictional (or compliant) contact, the object is in forceclosure in (a), but no force-closure in (b). A test for force-closure is identical to Reuleaux's method shown in Fig. 3; here, *F* is represented by the two edges of its friction cone shaded red, instead of the sole normal in Fig. 3. The existence of the region shaded gray means no force-closure [17].

desired mobility: the shaded area could have been much smaller without the convex geometry. In contrast, Fig. 3(d) shows an object configuration that is not scoopable due to the concavity at F. In this case, it is not possible for G to move toward T using any of the allowed twist in the areas signed "−" and "+," and thus the gripper may lose hold of the object. The object will become scoopable by switching the thumb's location to the opposite, thick end of the object where G is currently located. We note that the mobility analysis in Fig. 3(b-c) shows (1) the benefit of a small attack angle ψ, which can enlarge the shaded area, and (2) the challenge of an object's flat bottom, which will locate G as far as possible to the right, and will thus minimize the area if other conditions are the same.

## *C. Effect of Compliance and Friction*

Compliance and friction at contact, which are unavoidable in practice, are ignored in the mobility analysis above. Will they facilitate scooping? We answer this affirmatively based on empirical evidence. We performed scooping experiments in an end-to-end manner all the way from the fulfillment of Assumption 1. A test object (see Table I) was initially placed on a flat surface, and then a scoop was attempted with a two-fingered gripper, similarly to Fig. 1(a) (Section. V has full details of the experiment protocol). We then compared the performance of three fingertip materials to make the contact F while other conditions were the same. When a soft, high-friction material (silicone rubber) was used, no failures happened out of 150 scooping attempts. With a hard, low-friction (hard plastic) and a hard, high-friction material (sandpaper), respectively, the success rates were 52/150 and 146/150. Almost all the failed cases were caused by unsuccessful initial penetration. In other words, once Assumption 1 was fulfilled, successful scooping happened highly reliably. The results verify the augmented robustness brought by compliance and friction in fulfilling the key assumption, as well as the viability of the mobility analysis. We reconfirmed these points through another set of experiments performed with again the three types of materials for the support surface at G. The witnessed benefit of the softer, higher-friction materials actually comes as no surprise, since it has been discovered repeatedly in the literature.

Despite the advantage of robustness, the object may get stuck because of friction and compliance. That is, *wedging* [17] can

Fig. 5. Our two-fingered gripper for scooping. The length of the yellow finger is controllable using a rack and pinion actuated by a servo. A silicone rubber pad is attached to the tip of the blue finger, to make the contact *F*.

happen due to force-closure [17]. See Fig. 4 where F is modeled as a frictional contact with a friction cone, which can also model compliance.<sup>1</sup> The object is in force-closure in Fig. 4(a) even with frictionless T and G. Solutions to avoiding force-closure include reducing the angle of attack: no force-closure in Fig. 4(b) with decreased ψ.

# IV. PRACTICE OF MOTION-CONTROLLED SCOOPING WITH A PARALLEL-FINGER GRIPPER

This section presents the logistics of our scooping process by elaborating gripper design, planning, and control.

# *A. Design*

Our mobility analysis in Section. III necessitates a twofingered gripper, composed of a finger and a thumb, with two degrees-of-freedom (DOF) such that the fingertip F can be placed freely with respect to the thumb. Fig. 5 shows our customized gripper to be used in our experiments. It is essentially a parallel-finger gripper retrofitted with a variable-length digit. Therefore, it is possible to place one fingertip freely relative to the other finger by opening/closing the gripper and by extending/retracting the variable-length digit. According to the lesson from Section. III-C, a soft material is applied to one fingertip.

# *B. Pre-Scoop Planning*

Procedure 1 below describes the steps of pre-scoop planning for testing scoopability and for determining how to initially configure the gripper by fixing the location of F and the thumb's angle of attack ψ. It takes as input the 3D model of an object and returns feasible pre-scoop configurations.

The planner starts with FindAntipodalPairs. The given object model, with a large width-to-thickness ratio, is first approximated as an n-sided convex polygon by disregarding its thickness, and then its antipodal pairs, which can be a vertex-vertex, vertex-edge, or edge-edge pair, are found. This is a classic computational geometry problem with a range of solution approaches.

For each antipodal pair, the following is repeated. First, in GetCrossSection method, the 2D cross section of the object model on the plane normal to that of the approximated

# **Procedure 1:** Pre-Scoop Planner.

**Data**: 3D Object model

**Result**: Feasible pre-scoop configurations

- 1: FindAntipodalPairs()
- 2: **Repeat** for each antipodal pair
- 3: GetCrossSection()
- 4: TestScoopability()

Fig. 6. Pre-scoop planning with our test object models (Table I). Based on the chosen antipodal pair in the middle row, the blue arc in each panel of the bottom row marks the acceptable locations of *F* for successful scooping. Also shown is the gray dot that represents the leftmost (closest to the thumb) allowable location of *F* obtained through experiments (Table I).

polygon and aligned with the antipodal pair is obtained. A computational geometry package that can handle polygonal meshes can be applied here, for example, 'trimesh' in Python. Then in TestScoopability method, the mobility analysis in Section. III-B is performed with the cross section by solving the following linear programs that execute Reuleaux's method at the configuration whereby Assumption 1 is applied with a small nonzero φ and a chosen angle of attack ψ (Fig. 3(b)):

$$find x (1)$$

subject to 
$$a_i^T \mathbf{x} < \mathbf{a}_i^T \mathbf{x}_i, i \in \{T, G, F\}$$
 (2)

$$find x (3)$$

subject to 
$$a_i^T \mathbf{x} > \mathbf{a}_i^T \mathbf{x}_i, i \in \{T, G, F\}$$
 (4)

The programs (1-2) and (3-4) are for finding the areas signed "−" and "+," respectively. These collectively represent the set of the allowable centers of rotation for the object. **a***<sup>i</sup>* denotes a vector orthogonal (rotated 90◦ counterclockwise) to the contact i's normal. **x***<sup>i</sup>* denotes the initial point of **a***i*, on the line of the contact normal at i. See Fig. 3(b) showing **a***<sup>T</sup>* and **x***<sup>T</sup>* for example. Subsequently, scoopability (Definition 1) is verified by checking the direction of the allowed velocity at G. TestScoopability itself can be repeated to obtain the collection of the acceptable locations of F given ψ, which can

<sup>1</sup>According to [20], "For planar problems, a point contact with friction and a soft contact are identical."

Fig. 7. Scooping operations performed on a flat surface: (a) Domino block (widthwise), (b) Domino block (lengthwise), (c) Go stone, (d) Triangular prism. See also the video attachment.

testify the suitability of scooping with the object. Fig. 6 shows examples obtained with our software.<sup>2</sup>

## *C. Scooping Via Motion Control*

Based on the resulting pre-scoop plan, a scoop is executed through motion control using the gripper, as can be previewed in Fig. 7. First, the gripper is aligned with the selected antipodal pair and the finger contact F is made on top of the object by the fixed-length digit, according to the mobility analysis (Fig. 6). The thumb as a variable-length digit is then controlled to slide on the support surface toward F. This maneuver leads to successful scooping while the object slides upward on the thumb's face. Our software<sup>2</sup> keeps the thumb attack angle ψ constant, the fingertip at F fixed, and the thumb's tip in contact with the surface as the gripper closes, by coordinating the motions of our entire manipulator system—the base gripper, the variablelength finger, and a six-DOF arm. The necessity of the wholearm motion coordination is accounted for by the fact that the mechanism of the base gripper (by Robotiq Inc.) is arranged such that the fingertips move distally as the fingers close. The software also takes advantage of wrist force-torque sensing to detect initial contacts on the tips of the digits, which will trigger the scooping motion.

Alternatively, it is conceivable to control the gripper such that the thumb remains fixed and the fingertip F moves toward the thumb. In this case, the object will necessarily slide at G all the way through for successful scooping, against the maximum possible friction force at G in the direction away from T. We therefore expect this finger-driven scooping to be outperformed by the thumb-driven scooping explained above, whereby G does not need to slide constantly. Our experiments in Section. V will confirm this. Another issue is whether to use the variable-length digit as the thumb or the finger. Our experiments will provide the empirical evidence that the preferred option is to use the variable-length digit as the thumb.

# V. PICKING BY SCOOPING: IMPLEMENTATION, EXPERIMENTS, AND DISCUSSION

This section presents the results of a series of experiments performed autonomously in a range of scenarios featuring a constrained workspace: scooping from a flat surface and bin picking via scooping.

## *A. Scooping From a Flat Support Surface*

We first examined the performance of our scooping manipulation in the task of picking objects lying on a flat surface. The objects that we tested are a plastic card, a domino block, a Go stone, and a flat triangular prism, all with a large widthto-thickness ratio, as can be seen in Table I. These all feature a flat bottom face that can pose a challenge, as discussed in Section. III-B. For example, the Go stone has a semi-elliptical cross section; if it were fully-elliptical, it would be easier to scoop. According to the pre-scoop planning (Fig. 6), a parallel edge pair, an antipodal vertex pair, and a vertex-edge pair were selected to accommodate feasible pre-scoops for the rectangular objects (the plastic card and the domino block), the circular Go stone, and the triangular prism, respectively. Following initial localization by a fiducial marker, a scoop was then performed as instructed in Section. IV-C. See Fig. 7.

The lessons from the results reported in Table I, featuring the success rates (a success is defined as a successful final pinch grasp) of scooping under various conditions, are:

- - A small angle of attack ψ can facilitate scooping, as discussed in Section. III-B and III-C. The values of ψ in the experiments were determined such that the somewhat bulky motor box (Fig. 5) does not interfere with the support surface.
- - When the thumb was controlled to approach to the fixed finger (the columns labeled "thumb-driven" in the table), the success rates were higher. This confirms our discussion in Section. IV-C.
- - When the variable-length digit was used as the thumb, the success rates were higher, as stated in Section. IV-C.

The table also presents the minimum values of F's dimensionless position that can guarantee the reproducibility of the results, i.e. all the way from the fulfillment of Assumption 1 to successful picking, with the soft material (silicone rubber) fingertip. The value ranges from 0 to 1. When it is closer to 1, F is closer to G. These values confirm the conservativeness of the mobility analysis, as compared in Fig. 6.

By comparison, an ad hoc approach to picking a thin object lying on a flat surface proved ineffective, as shown back in Fig. 1(b). Here, the fingers were controlled to slide on the surface toward each other in an effort to obtain a pinch grasp on an edge

<sup>2</sup>Available at<https://github.com/HKUST-RML/Scooping>

TABLE I EXPERIMENT RESULTS: SCOOPING OBJECTS LYING ON A FLAT SURFACE

TABLE II THE EFFECT OF CLUTTER SIZE ON BIN PICKING EFFICIENCY

| Object              | Max. clutter size  | Success rate | PPH |
|---------------------|--------------------|--------------|-----|
| Domino<br>block     | 6                  | 39/40        | 150 |
|                     | 18 (half-full bin) | 36/40        | 129 |
|                     | 36 (full bin)      | 34/40        | 119 |
| Go stone            | 8                  | 40/40        | 144 |
|                     | 55 (half-full bin) | 35/40        | 115 |
|                     | 115 (full bin)     | 32/40        | 101 |
| Equilateral         | 8                  | 37/40        | 123 |
| triangular<br>prism | 50 (half-full bin) | 35/40        | 110 |
|                     | 100 (full bin)     | 33/40        | 105 |
|                     |                    |              |     |

pair of the object, a plastic card. Out of 30 picking attempts, none were successful.

## *B. Bin Picking*

The next set of experiments are concerned with picking items out of a cluttered bin. The lessons confirmed in Section. V-A are exploited by adopting the variable-length digit as the thumb, moving the thumb toward the fixed finger, positioning F in accordance with the minimum threshold, and finally running the same control software.

Given a 2D image of an object cluster seen from above, we identify each instance through instance segmentation and then its pose is estimated using standard vision techniques. The instance with the largest detected footprint and the most distant from the bin's wall is heuristically selected as the one to pick at the time. A scoop is then executed after the robot finds a feasible pre-scoop configuration with no collision with the wall of the bin. Fig. 8(a-c) illustrates the process.

We evaluated the bin picking performance while varying the size of the clutter: we set a maximally allowed number of items and topped the bin up occasionally. Table II provides two evaluation metrics, success rates and PPH (picks per hour) values. PPH is obtained by multiplying the success rate and the rate of picking, which is inclusive both perception and manipulation time. Since our implementation is not based on a highly-optimized hardware/software setting, the metrics are meant for comparing relative performance. It can be seen that the best performance is obtained when there are just a few items around, which may be accounted for by the decreased pose uncertainty when an item is supported by a flat, hard bottom surface, rather than a cluster of other items.

# *C. Complete Bin Picking*

Finally, the robot is tasked with *complete* bin picking: given a bin filled with a homogeneous group of the test objects, pick *all* of them one after the other out of the bin.

- *1) Autonomous Complete Bin Picking Via Scooping:* First, Fig. 8(d) presents some snapshots during the complete bin picking of plastic cards, advanced from our recent work [21] that necessitated dual-arm manipulation. This was performed autonomously using the same vision and control procedure explained in Section. V-B. By repeating multiple sets of the experiments, we were able to achieve 73 PPH. Second, the complete bin picking of 25 domino blocks was also executed autonomously, in a collision-aware manner such that the robot avoids collision with the wall of the bin. In one trial, it took 829 seconds at 108 PPH to pick all the blocks one by one. Here, the robot was also programmed opportunistically (1) to get a direct pinch grasp in case both the two largest opposite faces of a block are reachable and (2) to relocate a block by simply pushing toward the center of the bin if a collision-free path can be found in case a small number of blocks are around, which can facilitate initial positioning of the gripper on the object. The video attachment provides the coverage of these capabilities too.
- *2) Complete Bin Picking Via Scooping and Dig-Grasping:* Our recent study [16] presents the method of *dig-grasping*, which is also applicable to bin picking. A dig-grasp is executed

Fig. 8. (a-d) Scooping applied to the bin picking of dominoes, Go stones, triangular prisms, and plastic cards, respectively. The object instance being picked is identified through vision, as shown on the leftmost panels. See also the video attachment.

by "digging" the clutter in the form of a controlled push maneuver. Although this is distinct from the way an item is scooped using the presented technique here, both dig-grasping and scooping are implemented using the gripper with adjustable finger lengths (Section. IV-A). We have discovered that dig-grasping generally outperforms scooping in terms of PPH, provided other conditions (e.g. vision) are the same. However, it is not possible to complete bin picking using only dig-grasping because a nonzero positive digging depth is needed; for example, it cannot be applied to the tasks in Fig. 7, which can be accomplished easily with scooping, because the finger cannot dig the hard support surface. We thus tried to integrate the two, in order to accelerate the pace of bin picking. Our strategy is to start off with dig-grasping (Fig. 9(a)) and to switch to scooping (Fig. 9(c)) as the remaining amount of items becomes insufficient to dig-grasp. Note, dig-grasping is facilitated by a rigid fingertip, unlike scooping. This is because a softer fingertip is more likely to give rise to *stable pushing* [22], according to the mechanics of pushing, instead of a successful dig-grasp—quickly rotating the object and funneling it into the gripper's workspace. Therefore the mechanism in Fig. 9(b) proved useful. We applied the strategy to the complete bin picking of bowl-full, 115 Go stones. Out of 128 picking attempts in total, reported in our multimedia material submitted,<sup>3</sup> there were 101 successful singulation and subsequent picking events, 7 attempts in which two stones were

Fig. 9. (a) Dig-grasping for an almost full bin. (b) The fingertip material can be exchanged between a soft (for scooping) and a rigid one (for dig-grasping), using a rack and pinion. (c) Scooping for an almost empty bin.

<sup>3</sup>Full video available at<https://youtu.be/A1oetxHKOyY>

simultaneously captured, and 20 picking failures. Dig-grasping was executed 37 times at 158 PPH. This helped achieve 138 PPH throughout the whole process, close to the best performance that our scooping can accomplish, according to Table II.

## VI. CONCLUSION

In this study, we presented an approach to motion-controlbased scooping that is suitable for picking relatively thin objects in a constrained workspace. Our mobility analysis justifies the novel use of a motion control scheme for scooping, and our system is arranged to implement the required motion, with demonstrated effectiveness and robustness.

## *A. Advantages and Limitations*

Our motion-control-based scooping offers advantages. It lends itself to a conventional motion-controlled gripper, without underactuated control or a compliant mechanism. In terms of gripper positioning, a contact on the top face of an object, as featured in this study, can be easier to target and reach than the side edges, as can be seen in Fig. 1. Taking advantage of a contact on the face also makes it possible to pick objects larger than the gripper's opening range.

Our method does have limitations. If Assumption 1 is violated, although its viability has been demonstrated, a successful scoop will not happen. The mobility analysis requires accurate information about the object's geometry.

## *B. Generalization*

Possible directions for generalization include the following. The adoption of the simple gripper model, in which a point finger is motion-controlled with respect to the face of a thumb, makes it possible to apply a wide range of grippers other than the one shown in this work, for example, a gripper with anthropomorphic, multi-phalanx fingers. Reactive and spontaneous scooping, rather than the planned approach shown in this work, is conceivable using fingertip tactile feedback for detecting contact geometry.

# REFERENCES

- [1] T. Matsuno, K. Kanada, F. Arai, H. Matsuura, and T. Fukuda, "Strategy of picking up thin plate by robot hand using deformation of soft fingertip," in *Proc. IEEE Int. Conf. Robot. Automat.*, pp. 2326–2331, 2005.
- [2] V. Babin and C. Gosselin, "Picking, grasping, or scooping small objects lying on flat surfaces: A design approach," *Int. J. Robot. Res.*, vol. 37, no. 12, pp. 1484–1499, 2018.

- [3] V. Babin, D. St-Onge, and C. Gosselin, "Stable and repeatable grasping of flat objects on hard surfaces using passive and epicyclic mechanisms," *Robot. Comput.-Integr. Manuf.*, vol. 55, pp. 1–10, 2019.
- [4] L. U. Odhner, R. R. Ma, and A. M. Dollar, "Open-loop precision grasping with underactuated hands inspired by a human manipulation strategy," *IEEE Trans. Automat. Sci. Eng.*, vol. 10, no. 3, pp. 625–633, Jul. 2013.
- [5] C. Jiang, A. Nazir, G. Abbasnejad, and J. Seo, "Dynamic flex-and-flip manipulation of deformable linear objects," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.*, pp. 3158–3163, 2019.
- [6] K. Morino, S. Kikuchi, S. Chikagawa, M. Izumi, and T. Watanabe, "Sheetbased gripper featuring passive pull-in functionality for bin picking and for picking up thin flexible objects," *IEEE Robot. Automat. Lett.*, vol. 5, no. 2, pp. 2007–2014, Apr. 2020.
- [7] S. Yuan, A. D. Epps, J. B. Nowak, and J. K. Salisbury, "Design of a rollerbased dexterous hand for object grasping and within-hand manipulation," in *Proc. IEEE Int. Conf. Robot. Automat.*, pp. 8870–8876, 2020.
- [8] T. Ko, "A tendon-driven robot gripper with passively switchable underactuated surface and its physics simulation based parameter optimization," *IEEE Robot. Automat. Lett.*, vol. 5, no. 4, pp. 5002–5009, Oct. 2020.
- [9] C. H. Kim and J. Seo, "Shallow-depth insertion: Peg in shallow hole through robotic in-hand manipulation," *IEEE Robot. Automat. Lett.*, vol. 4, no. 2, pp. 383–390, Apr. 2019.
- [10] A. Zeng *et al.*, "Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching," in *Proc. IEEE Int. Conf. Robot. Automat.*, pp. 3750–3757, 2018.
- [11] Y. Domae, H. Okuda, Y. Taguchi, K. Sumi, and T. Hirai, "Fast graspability evaluation on single depth maps for bin picking with general grippers," in *Proc. IEEE Int. Conf. Robot. Automat.*, pp. 1997–2004, 2014.
- [12] I. Lenz, H. Lee, and A. Saxena, "Deep learning for detecting robotic grasps," *Int. J. Robot. Res.*, vol. 34, no. 4-5, pp. 705–724, 2015.
- [13] L. Pinto and A. Gupta, "Supersizing self-supervision: Learning to grasp from 50 k tries and 700 robot hours," in *Proc. IEEE Int. Conf. Robot. Automat.*, pp. 3406–3413, 2016.
- [14] A. ten Pas, M. Gualtieri, K. Saenko, and R. Platt, "Grasp pose detection in point clouds," *Int. J. Robot. Res.*, vol. 36, no. 13-14, pp. 1455–1473, 2017.
- [15] S. Levine, P. Pastor, A. Krizhevsky, J. Ibarz, and D. Quillen, "Learning hand-eye coordination for robotic grasping with deep learning and largescale data collection," *Int. J. Robot. Res.*, vol. 37, no. 4-5, pp. 421–436, 2018.
- [16] Z. Tong, Y. H. Ng, C. H. Kim, T. He, and J. Seo, "Dig-grasping via direct quasistatic interaction using asymmetric fingers: An approach to effective bin picking," *IEEE Robot. Automat. Lett.*, vol. 6, no. 2, pp. 3033–3040, Apr. 2021.
- [17] M. T. Mason, *Mechanics of Robotic Manipulation*. Cambridge, MA, USA: MIT press, 2001.
- [18] C. Mucchiani, M. Kennedy, M. Yim, and J. Seo, "Object picking through in-hand manipulation using passive end-effectors with zero mobility," *IEEE Robot. Automat. Lett.*, vol. 3, no. 2, pp. 1096–1103, Apr. 2018.
- [19] A. Specian, C. Mucchiani, M. Yim, and J. Seo, "Robotic edge-rolling manipulation: A grasp planning approach," *IEEE Robot. Automat. Lett.*, vol. 3, no. 4, pp. 3137–3144, Oct. 2018.
- [20] K. M. Lynch and F. C. Park, *Modern Robotics*. Cambridge, U.K.: Cambridge Univ. Press, 2017.
- [21] Z. Tong, T. He, C. H. Kim, Y. H. Ng, Q. Xu, and J. Seo, "Picking thin objects by tilt-and-pivot manipulation and its application to bin picking," in *Proc. IEEE Int. Conf. Robot. Automat.*, pp. 9932–9938, 2020.
- [22] K.M. Lynch andM. T.Mason, "Stable pushing:Mechanics, controllability, and planning," *Int. J. Robot. Res.*, vol. 15, no. 6, pp. 533–556, 1996.