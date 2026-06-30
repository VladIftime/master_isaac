# More than a Million Ways to Be Pushed. A High-Fidelity Experimental Dataset of Planar Pushing

Kuan-Ting Yu<sup>1</sup> , Maria Bauza<sup>2</sup> , Nima Fazeli<sup>2</sup> , Alberto Rodriguez<sup>2</sup> <sup>1</sup> Computer Science and Artificial Intelligence Laboratory — Massachusetts Institute of Technology <sup>2</sup> Mechanical Engineering Department — Massachusetts Institute of Technology peterkty@csail.mit.edu, <bauza,nfazeli,albertor>@mit.edu

*Abstract*— Pushing is a motion primitive useful to handle objects that are too large, too heavy, or too cluttered to be grasped. It is at the core of much of robotic manipulation, in particular when physical interaction is involved. It seems reasonable then to wish for robots to understand how pushed objects move.

In reality, however, robots often rely on approximations which yield models that are computable, but also restricted and inaccurate. Just how close are those models? How reasonable are the assumptions they are based on? To help answer these questions, and to get a better experimental understanding of pushing, we present a comprehensive and high-fidelity dataset of planar pushing experiments. The dataset contains timestamped poses of a circular pusher and a pushed object, as well as forces at the interaction. We vary the push interaction in 6 dimensions: surface material, shape of the pushed object, contact position, pushing direction, pushing speed, and pushing acceleration. An industrial robot automates the data capturing along precisely controlled position-velocity-acceleration trajectories of the pusher, which give dense samples of positions and forces of uniform quality.

We finish the paper by characterizing the variability of friction, and evaluating the most common assumptions and simplifications made by models of frictional pushing in robotics.

## I. INTRODUCTION

Pushing is a widely used motion primitive for robotic manipulation. It can aid in the positioning and reorientation of parts [\[1,](#page-7-0) [2,](#page-7-1) [3,](#page-7-2) [4\]](#page-7-3); facilitate grasping under pose uncertainty [\[5\]](#page-7-4) or clutter [\[6\]](#page-7-5); or help in the transportation of large or heavy objects [\[7,](#page-7-6) [8\]](#page-7-7). The mechanics of pushing have also been used to aid perception, for example to track the pose of a pushed object [\[9,](#page-7-8) [10,](#page-7-9) [11\]](#page-7-10), to estimate its shape [\[11\]](#page-7-10), and to identify inertial parameters such as mass, moment of inertia or coefficient of friction [\[12,](#page-7-11) [13\]](#page-7-12). All these applications rely on a good understanding of the mechanics of pushing, which let us predict how an object moves under a certain push.

At an analytical level, pushing is a well understood problem. For decades, the mechanics and robotics com-

This work was supported by NSF awards [NSF-IIS-1427050] and [NSF-IIS-1551535] through the National Robotics Initiative.

munities have developed models to explain the interaction at the interface between a pushed object and a support surface [\[1,](#page-7-0) [14,](#page-7-13) [10,](#page-7-9) [15,](#page-7-14) [8\]](#page-7-7). These are usually based on Coulomb's friction law, often rewritten as the maximumpower inequality [\[16\]](#page-7-15). In Section [II,](#page-1-0) we summarize these, and see how they have led to compact and deterministic models that, under sufficient assumptions, can be used to explain and control the motion of a pushed object.

The reality, however, is bitter. Predicting the motion of a pushed object is not trivial. In practice, the sensitivity of the task to small changes in contact geometry, along with the variability of friction, hinders accurate predictions.

More recently, data-driven models [\[17,](#page-7-16) [7,](#page-7-6) [18,](#page-7-17) [19,](#page-7-18) [20\]](#page-7-19) have been proposed as an alternative approach to analysis. Studies are still incipient and either do not offer sufficient generality, or do not address variability. A lack of common datasets or benchmarks may explain why learning has not yet had the same effect that it has in other disciplines, such as computer vision. Datasets could facilitate research in model development by enabling evaluation and comparison of solutions. Although robots excel at accuracy and repetition, capturing large amounts of real data requires setting and resetting of experiment conditions, which is tedious with human intervention, and difficult without it. This is in stark contrast, for example, to collecting digitized daily images for computer vision research [\[21\]](#page-7-20).

In this study, we have automated the setting and execution of controlled pushing experiments, and captured a large highfidelity dataset of pushing interactions. The dataset, detailed in Section [III,](#page-1-1) includes a wide variety of controlled pushes recorded at a high sample rate with both a force/torque sensor, and a Vicon tracking system.

The dataset covers variations in surface material, object shape, contact position, pushing direction, pushing speed, and pushing acceleration. To our knowledge, this is the first dataset of this caliber. A significant novelty of the dataset is that it contains dynamic pushing where the inertial components have an appreciable effect over frictional forces,

Fig. 1. Data capturing hardware: A steel pusher is attached to an ATI force/torque sensor, and driven by an ABB IRB 120 robot arm. The object (stainless steel block) is instrumented with reflective markers and tracked with a Vicon motion tracking system. The block slides on top of an interchangeable support surface.

<span id="page-1-3"></span>for which there is little work in the literature. We hope it will provide a tool for an experimental study of pushing.

In summary, our contributions are:

- A dataset of planar pushing containing time series of high-fidelity poses of both the pusher and the pushed object, and forces experienced by the pusher. This yields more than a million timestamped data points of poses and forces for each combination of shape and material.
- An evaluation of common assumptions made by analytical models of pushing in robotics.

#### II. RELATED WORK

<span id="page-1-0"></span>One of the most common assumptions in robotic pushing, and possibly in robotic manipulation, is quasistatic interaction. In the context of pushing, quasistatic interaction means that the velocity of the involved objects is small enough that inertia is negligible. Instantaneous motion is then a consequence of the balance between contact forces, frictional forces, and gravity. The quasistatic assumption makes the problem more tractable, yielding simpler models, and is a reasonable assumption for the scales and speeds in much of robotic manipulation [\[22\]](#page-7-21).

Mason [\[1\]](#page-7-0) starts the line of work on pushing by proposing a voting theorem to determine the rotation direction of a pushed object. Goyal et al. [\[16\]](#page-7-15) propose a limit surface representation to map motions and frictional forces of a sliding object. These serve as the foundation of much subsequent work on pushing. Lee and Cutkosky [\[23\]](#page-7-22) propose to approximate the limit surface as an ellipsoid to improve computational time. Lynch et al. [\[24\]](#page-7-23) apply the ellipsoidal approximation to derive a closed-form analytical

<span id="page-1-2"></span>TABLE I ASSUMPTIONS AND APPROXIMATIONS MADE IN PRIOR WORK

| Condition                               | Work examples         |
|-----------------------------------------|-----------------------|
| Uniform friction*                       | [all]                 |
| Known pressure distribution             | [16, 24, 23, 6]       |
| Known center of friction                | [1]                   |
| (the centroid of pressure distribution) |                       |
| Coulomb friction*                       | [1, 24, 10, 3, 26, 6] |
| Maximum power inequality*               | [all]                 |
| (generalized Coulomb friction)          |                       |
| Quasistatic interaction                 | [24, 1, 3, 6]         |
| Ellipsoidal limit surface*              | [23, 24]              |
| Sliding (frictionless) pushing          | [10]                  |
| Sticking (infinite friction) pushing    | [8]                   |

<sup>\*</sup> Conditions that we explicitly validate in this paper.

solution for quasistatic pushing, including both sticking and sliding behaviors. Howe and Cutkosky [\[25\]](#page-7-25) explore more approximation methods of limit surfaces, and provide a guide for choosing between them based on the pressure distribution, computation cost and accuracy. These models provide functional representations of Coulomb's friction law, and the maximum power inequality [\[16\]](#page-7-15). We find practical use cases, for example, for stable pushing of a planar object with a fence-shaped finger [\[3\]](#page-7-2), for planning robust pushgrasps for objects in clutter [\[6\]](#page-7-5), and for planning in-hand manipulation with patch contacts [\[26\]](#page-7-24).

Peshkin and Sanderson [\[14\]](#page-7-13) address the uncertainty in pushing by sampling all possible pressure distributions and predicting a range of possible object motions. Jia and Erdmann [\[10\]](#page-7-9) investigate dynamic pushing but assume frictionless interaction between pusher and object. Behrens [\[8\]](#page-7-7) instead studies dynamic pushing assuming infinite friction between pusher and object. Table [I](#page-1-2) summarizes these works along with their assumptions and approximations. In this paper we explicitly validate the assumptions marked with an asterisk (\*).

#### III. THE PUSHING DATASET

<span id="page-1-1"></span>This dataset records the poses of a pusher and a pushed object together with the interaction forces for a variety of pushing experiments. Variations in experiments cover 6 dimensions: object shape, surface material, pusher direction, pusher speed, pusher acceleration, and initial contact position. This section describes these dimensions, which are also summarized in Table [II.](#page-2-0) For extended details, refer to the dataset website [\[27\]](#page-7-26):

- · Shape. Different shapes can give us insights into phenomena such as the dependence of friction with variations in the support pressure distribution. We use 3 rectangles with different aspect ratios (rect1-3), 3 right triangles with different skews (tri1-3), 3 ellipses with different eccentricities (ellip1-3), 1 hexagon (hex), and 1 butterfly shape (butter). See Table [III](#page-3-0) for dimensions and other physical properties. All objects are fabricated in stainless steel, and bead blasted to give a rough finish free of burrs.
- · Surface material. The support surface where the object slides is of great importance as it dictates the frictional

TABLE II
SUMMARY OF DIMENSIONS EXPLORED IN THE DATASET.

<span id="page-2-0"></span>

| Shape                            | rect1, rect2, rect3, hex, ellip1, ellip2, ellip3, butter, tri1, tri2, tri3   |  |  |  |
|----------------------------------|------------------------------------------------------------------------------|--|--|--|
| Surface                          | abs, derlin, polywood, pu                                                    |  |  |  |
| Speed (mm/s)                     | 10, 20, 50, 75, 100, 150, 200, 300, 400, 500                                 |  |  |  |
| Acceleration (ms <sup>-2</sup> ) | 0, 0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 2.5                                       |  |  |  |
| Initial contact                  | 33 points for tri1-3 and hex, 40 for ellip1-3 and butter, and 44 for rect1-3 |  |  |  |
| Initial push direction           | 0°, 20°, 40°, 60°, 80°, -20°, -40°, -60°, -80°                               |  |  |  |

interaction with the object. We experiment with four surfaces: i) ABS, ii) Delrin, iii) plywood, and iv) polyurethane (hardness 80A durometer). The first two are widely used hard plastics. The third is a softer material and the fourth has a rubber-like texture. We will refer to the materials as abs, delrin, plywood, and pu respectively throughout the paper. In section V and VI, we characterize relevant frictional properties of these surfaces.

- **Speed.** Speed dictates the regime in which the object moves: quasistatically (negligible inertia) or dynamically (meaningful inertia). We explore pusher trajectories with constant speeds: 10, 20, 50, 75, 100, 150, 200, 300, 400, and 500 mm/sec.
- Acceleration. The acceleration of the pusher is a relatively unexplored dimension. We capture interactions both with constant speed (zero acceleration) and with constant accelerations 0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, and 2.5 ms<sup>-2</sup> starting from rest.
- Contact position. Each object is pushed at a number (between 33 and 44) of evenly-spaced contact locations.
- **Push direction.** For each contact location we vary the direction of the push between -80° to 80° around the contact normal with increments of 20°, for a total of 9 directions.

Each experiment executes an open-loop pusher trajectory in the reference frame of the initial pose of the object, leading to evolving contact geometry between object and pusher. The trajectory is executed and recorded at 250 Hz, which allows us to explore different contact interactions efficiently, including transitions between sticking and sliding.

The experiments are position controlled in preference to force controlled for three reasons: First, the position, speed and acceleration of the pusher can be controlled very accurately with an industrial robot, whereas force sensors typically have much lower signal-to-noise ratio; Second, controlling the force between pusher and object is challenging because it is constrained by friction, and errors in the friction coefficient can lead to unexpected trajectories; Third, although we do not control force directly, we record the force and pose of the object at a high frame rate. The data is still useful to study the relation between forces and motions.

#### IV. DATA COLLECTION SPECIFICATIONS

In this section we detail the data collection system and the automated process designed to record the pushes. Figure 1 shows the setup: a 6 DOF industrial robotic manipulator equipped with a stiff cylindrical rod acting as a pusher.

#### A. Hardware

**Robot.** The system uses an ABB IRB 120 industrial robotic arm with 6 DOF to control precisely the position, velocity and acceleration of its tool center point (TCP). The robot has a horizontal reach of 580 mm and a payload of 3 kg, which is sufficient since the pushed objects have a mass in the order of 1 kg.

**Force sensing.** We use an ATI Gamma force-torque sensor rigidly attached to the 6th link of the robot to measure the reaction force from the object on the pusher. The sensor has high sensitivity with force resolution of 1/160 N in the pushing plane, and torque resolution of 1/2000 N⋅m perpendicular to the pushing plane.

**Motion sensing.** We track the pose of the object with a Vicon motion tracking system, composed of 5 Bonita cameras with a wide field of view. Each object is fitted with 4 reflective markers. Although 3 are in theory sufficient, in practice 4 asymmetric markers give more stable readings.

The noise in the recording system is quite small. The accuracy of the object position depends on the accuracy of the Vicon system, which is below 0.5 mm for translation and  $0.5^{\circ}$  for rotation. The pose of the pusher is directly given by the robot, with an accuracy of 0.1 mm.

**Pusher.** The robot is equipped with a stiff cylindrical steel pusher, mounted on and perpendicular to the measurement plate of the force-torque sensor. The pusher has length 156 mm and diameter 9.5 mm, which we found to be a good trade-off to minimize occlusions and provide rigidity.

**Objects.** We use a total of 11 objects, all water-jet cut in stainless steel for durability. They are bead blasted to remove burrs, retaining a more realistic "rough" surface. Object mass ranges between 0.75 and 1.4 kg depending on the shape. All objects are 13 mm thick. The friction coefficient between the pusher and the object is approximately 0.25, which was determined using a traditional variable slope experiment.

TABLE III SET OF OBJECTS IN THE DATASET. PHYSICAL PROPERTIES.

<span id="page-3-0"></span>

| Object | Mass (g) | Dimension (mm)           | Moment of          |
|--------|----------|--------------------------|--------------------|
|        |          |                          | inertia (g·m2<br>) |
| rect1  | 837      | w:90, h:90               | 1.13               |
| rect2  | 1045     | w:90, h:112.5            | 1.81               |
| rect3  | 1251     | w:90, h:135              | 2.74               |
| hex    | 983      | circumradius: 60.5       | 1.50               |
| ellip1 | 894      | w:105, h:105             | 1.23               |
| ellip2 | 1110     | w:105, h:130.9           | 1.95               |
| ellip3 | 1334     | w:105, h:157             | 2.97               |
| butter | 1197     | w1:95.3, w2:54.7, h: 156 | 2.95               |
| tri1   | 803      | leg1: 125.9, leg2: 125.9 | 1.41               |
| tri2   | 983      | leg1: 125.9, leg2: 151.0 | 2.11               |
| tri3   | 1133     | leg1: 125.6, leg2: 176.5 | 2.96               |

### *B. Software*

To facilitate integration of various components such as robot control, force-torque sensor, and motion tracker, we use the Robot Operating System (ROS) framework. Data streams (robot pose, object pose and force-torque) are published as ROS topics and recorded at 250 Hz. The experiments are logged as ROS bag files and parsed into HDF5 and JSON format. Refer to [\[27\]](#page-7-26) for more format details.

#### *C. Data Collection Process*

The pushing experiments follow these steps:

- 1. The tracker locates the object.
- 2. The robot executes an open-loop straight push along a predefined position-velocity-acceleration trajectory, in the initial reference frame of the object. The Vicon tracker and force-torque sensor record the interaction.
- 3. If needed, the robot resets the location of the object by dragging it to approximately the center of the plate.
- 4. Iterate.

The reset mechanism, key for capturing a very large number of experiments, is implemented by a thick tapered washer on the top of the object, that allows the pusher to easily drag the object in the plane.

The pusher starts in contact with the object and follows a straight line of 5 cm. The figure under the paper title shows 6 examples of straight-line pushes.

The data collection results in an approximate total of 6,000 pushes per object and surface. Each push produces an average of 200 timestamped interactions. In total, the experiments yield more than a million triples of pusher motion, object motion and pushing force.

### <span id="page-3-1"></span>V. VARIABILITY OF DYNAMIC SURFACE FRICTION

To support the previous dataset, we have conducted a series of experiments to characterize surface friction. In particular we are interested in studying the variability of the effective coefficient of friction with respect to these factors of a sliding motion:

- 1) location,
- 2) repetition,
- 3) speed, and
- 4) direction.

Fig. 2. Cage for experiments of variability of surface friction. When engaged (right) the object fits loosely in the cage. The force-torque sensor measures both the force and moment of friction in the plane.

<span id="page-3-2"></span>To study these effects we design the cage in Figure [2](#page-3-2) to push the rectangle rect1 in a controlled manner. The cage does not clamp the object but traps it, with a small gap (< 1 mm) between object and cage. This gives the robot full control over the object sliding motion. We are interested in characterizing the dynamic friction force between the object and the surface, measured as the reaction force in the horizontal plane on the force/torque sensor at the robot wrist. We define then the theoretical dynamic coefficient of friction (DCoF) as the ratio between the measured reaction force f<sup>f</sup> and the supporting normal force fn, DCoF = ff f<sup>n</sup> .

In the experiments, the robot performs line scanning motions tailored to exploring specific dimensions: 1) location, 2) repetition, 3) speed and 4) direction. For dimensions 1-3) in each pass of the scan, the robot pushes the block from left to right and back to the starting point. It then moves down to transit to the next scanline. Neighboring scanlines are separated by 5 mm for 1) for high spatial resolution, and 100 mm for 2) and 3) for non-overlapping scans. All scanlines for 4) follow the diameters of a circle at the surface center. Data for 1), 2), and 4) are captured at a speed of 20 mm/s; for 3), we conduct 10 scans for each of the 10 speeds described in Section [III.](#page-1-1)

1) Spatial variability. Surface imperfections yield variation in the DCoF. Figure [3a](#page-4-0) shows the spatial distributions recovered for all four surfaces. The areas mapped are approximately 20 cm by 40 cm. It is interesting to note that even seemingly smooth and uniform materials such as delrin have distinguishable differences on the surface. Figure [3b](#page-4-0) shows the histogram of the measured DCoF for each surface material. Sorting their standard deviation from low to high, we have delrin: 0.016, abs: 0.017, plywood: 0.024, and much larger pu: 0.064. Interestingly, the histograms resemble Gaussian distributions which could be considered as a basic model for frictional sliding.

<span id="page-4-0"></span>Fig. 3. a) Spatial distribution of the coefficient of friction (DCoF) for four materials. The darker the color, the higher the coefficient. b) Histogram of the same distributions.

0 100 200 300 400 500 Speed (mm/s) 0.0 0.5 1.0 1.5 Coefficient of friction abs delrin plywood pu

<span id="page-4-1"></span>Fig. 4. Evolution of the coefficient of friction (DCoF) over 100 scans, for four different materials. Note that abs and delrin have a relatively short break-in phase, plywood does not stop degrading, and pu is more resistant to abrasion.

<span id="page-4-2"></span>Fig. 5. Change of the coefficient of friction (DCoF) with sliding speed of the object.

- 2) Temporal variability. A surface generally becomes smoother after being repeatedly rubbed in a polishing process. Similarly, sliding objects polish the surface they slide on and therefore change its effective DCoF. Here we quantify the polishing effect on newly purchased surfaces. Figure [4](#page-4-1) shows a decreasing trend of the effective DCoF for all materials. This effect is sometimes called break-in. After 100 scans, their respective DCoFs change like:
  - abs: 0.15 to 0.13 (-13.6%);
  - delrin: 0.16 to 0.12 (-22.2%);
  - plywood: 0.28 to 0.24 (-11.3%);
  - pu: 0.29 to 0.28 (-2.3%).

We observe that delrin and abs have an appreciable break-in period after which the DCoF converges to an almost constant value. For plywood, the break-in period is much longer. For pu, the break-in period is almost non-existent, hinting that for the range of forces we consider, there is almost no degradation of the material over time. Indeed, the type of polyurethane we used is abrasion-resistant.

3) Speed variability. Coulomb friction states that the magnitude of the friction force should not depend on the object sliding speed. Figure [5](#page-4-2) shows the results for experiments conducted with different speeds. Indeed, delrin, abs and plywood present little variability of DCoF with speed. The DCoF of pu however, increases up to 1.0 for high speeds. The phenomenon is already observed in [\[28\]](#page-7-27) for rubbers, and [\[29\]](#page-7-28) states that pu possess this characteristic. Coulomb friction then would not be a good approximation when the speed of experiments spans a wide range.

4) Direction variability. When a material presents friction independent of the sliding direction, we say it is isotropic; otherwise, anisotropic. To test it, we perform successive scans where we force the object to slide through the center of the plate in different directions. Figure [6](#page-5-1) shows the set of friction forces collected. An isotropic material would show a circular force profile. The figure shows that abs and delrin are close to isotropic, plywood slightly less, and pu the least. For pu, the ratio between the largest friction and the smallest is around 3/2, which is a significant difference. This could explain, in part, the large standard deviation of the DCoF observed in Figure [3b](#page-4-0) since scans are run forward and backward.

<span id="page-5-1"></span>Fig. 6. Directionality of friction force. Experiments show that abs, delrin, and plywood are mostly isotropic, i.e., friction is not direction dependent.

# <span id="page-5-0"></span>VI. EVALUATION OF MODELS OF FRICTIONAL SLIDING

In this section we study whether:

- 1) frictional sliding follows the maximum-power inequality, a.k.a. maximum dissipation principle;
- 2) the limit surface of a particular material can be well approximated by an ellipsoid.

We conduct experiments with the same setup as in Section V, with all scans passing through the center of the plate. To analyze the behavior of the frictional sliding wrench (force and torque), we conduct experiments controlling the instantaneous sliding twist of the object (linear and angular velocity) as it passes through the center of the plate.

To achieve this we generate trajectories with different ratios of translation and rotation velocity. We perform linear scans that approach the center of the plate at different angles in increments of  $5^{\circ}$  and from a starting distance from the center  $\in \{50, 25, 12.5, 0\}$  mm. For each scan, the object rotates between angles  $\theta$  and  $-\theta$  where we vary  $\theta \in \{-88^{\circ} \dots 88^{\circ}\}$  with increments of  $4^{\circ}$ .

1) Principle of maximum-power inequality. The curves in Figure 6 are known as limit curves (LC), i.e., the set of all possible frictional forces between object and surface material in pure translational sliding. The principle of maximum-power inequality [16] states that the resolution of frictional force and sliding motion is such that dissipation of power will be maximized. We can state the principle as:

$$\forall f^* \in LC, (f - f^*) \cdot v \ge 0,$$

where f and v are the friction force and sliding velocity at contact, and  $f^*$  is any other friction force in the LC.

In a general contact/friction problem, this principle is difficult to resolve, since it is a constraint that involves both forces and motions [26]. In our experiments, however, we force a particular velocity on the object. Then it is straightforward to verify if  $\Delta P = f \cdot v - \max_i (f_i \cdot v) \ge 0$ ,

<span id="page-5-2"></span>Fig. 7. Difference between power dissipated and maximum dissipable power  $(\Delta P)$  for the squared object rect1 sliding along different directions. The maximum power inequality expresses that  $\Delta P$  should be zero.

where *j* spans all points in the LC. To avoid issues with the different types of frictional variability discussed Section V, we only use data for the object passing through a particular point of interest.

Figure 7 shows  $\Delta P$  for experiments with different direction when passing through the center of the plate. All materials except for pu, yield  $\Delta P$  very close to 0. For pu, there are 2 regions where  $\Delta P$  is significantly less than 0. They correspond to the abrupt transitions at the top and bottom of its limit curve in Figure 6.

2) Limit surface and ellipsoidal approximation. A sliding planar object can both translate and rotate. The corresponding frictional wrench will have then both force and torque components. The limit curve (LC) discussed above generalizes into a limit surface (LS), the 2D set of frictional wrenches in the 3D wrench space that a surface can exert on a sliding object. Figure 8a shows a simple visualization of that limit surface. Similar to the LC, the LS works as follows: If the object is sliding, the friction wrench lies on the LS; otherwise, it lies strictly inside the LS. The maximum-power inequality dictates then that the motion corresponding to a particular frictional force must be orthogonal to the LS at that point [16].

For computational reasons, the LS is occasionally approximated as an ellipsoid [25]. Here we verify that approximation by constructing the real limit surface from real measurements. Figure 8b shows the recovered LS for four materials. We fit an ellipsoid to that data, by assuming it is centered at the origin, and estimating the moment magnitude from pure rotational motion and force magnitude from pure translational motion. The shade region shows the  $2\sigma$  uncertainty region.

Observe that the real limit surface is closer to thicker noisier ring. We can also see that the underlying curve of the data resembles an ellipse but not exactly. Finally, we observe that delrin has the most symmetric LS, abs and plywood are slightly biased toward the left side, possibly due to slight anisotropy, and pu resembles very little to an actual LS.

<span id="page-6-0"></span>Fig. 8. (a) Conceptual limit surface. The set of all possible frictional wrenches (force/torques) that the sliding object can receive from the surface (b) Experimental set of frictional force/torques exerted on the sliding object rect1 by different materials, and best ellipsoidal fitting, in the  $f_x$ -m plane.

### VII. STOCHASTICITY OF PUSHING MOTION

Experiments in Section V show that friction expresses a degree of variability in several dimensions, including space, time, speed and direction. If we want to use simple models, that will not be overconfident about frictional behavior, one could treat friction as a stochastic process. Here, we describe how the uncertainty looks like and motivate further in-depth studies to gain insight about the following questions:

- Does the object motion distribution appear Gaussian?
- How wide is it?
- Are the predictions from models in use close to the experimental behavior?
- Is the distribution dependent on the surface material?

To do so, we repeat a particular straight-line push experiment 2,000 times. The particular settings are:

- shape: rect1;
- contact location: half way in between the center and edge of the block side;
- contact angle: normal direction;
- speed: 20 mm/s (quasistatic speed);
- acceleration: 0 mm/s<sup>2</sup>;
- surface: all 4 materials;
- pusher displacement: 15 cm.

For parsing the results, we denote the object trajectory as starting from  $(x, y, \theta) = (0, 0, 0)$ , and ending at

<span id="page-6-1"></span>TABLE IV
DISTRIBUTION OF OBJECT DISPLACEMENTS AFTER REPEATED PUSHES

| Surface        | Mean                | Trans. std   | Rot. std   |
|----------------|---------------------|--------------|------------|
|                | (mm, mm, deg)       | (mm)         | (deg)      |
| abs            | (40.1, -67.6, 74.7) | 5.5 (7.1%)   | 3.2 (4.3%) |
| delrin         | (38.8, -50.7, 78.5) | 3.4 (5.2%)   | 1.3 (1.6%) |
| plywood        | (36.4, -93.6, 70.2) | 8.1 (8.0%)   | 4.2 (6.0%) |
| pu             | (40.2, -85.0, 69.3) | 11.7 (12.5%) | 4.5 (6.5%) |
| simulator [24] | (41.0, -98.1, 66.3) | N/A          | N/A        |

Fig. 10. Histogram of displacements  $\Delta x,\,\Delta y,\,$  and  $\Delta \theta$  produced by the 2000 pushes in Figure 9.

 $(\Delta x, \Delta y, \Delta \theta)$ . Figure 9 shows the resulting trajectories. The distribution of final poses seems to have at least three modes, and its shape is clearly not Gaussian. Table IV shows the standard deviation (std) of ending poses, which depends on the surface type. We normalize the error by the mean displacement to get an error rate (%). Qualitatively the standard deviation is related to the characterization of friction variability in Section V. It would be interesting to further investigate what is the nature of that relationship.

Figure 9 also shows a comparison of the mean experimental trajectory and the prediction by a model driven simulator [24]. They look quite different. Thus, another interesting future direction is to better evaluate those differences, in particular, under what conditions the predictions of a simple deterministic model are reasonable.

Experiments show that even when trying to replicate the same initial conditions with an accurate vision system and an accurate robot, a determined pushing interaction yields appreciable and structured uncertainty at the outcome. This motivates further investigation of effective ways to take into account uncertainty or variability in friction.

#### VIII. CONCLUSION

This paper presents a large and high-fidelity experimental dataset of planar pushing interactions. The data spans six different dimensions of the pushing problem: the shape of the pushed object, the material of the surface where it slides, the location of contact between pusher and slider, and the direction, velocity, and acceleration along which the pusher moves. Overall, these generate more than a million timestamped samples of positions of pusher and slider, as well as interaction forces.

We also describe and evaluate the most common assumptions and approximations used in models of planar pushing. The results say that while some assumptions such as the

<span id="page-7-29"></span>Fig. 9. Example of 2000 pushes of object rect1 on surface material abs. (a) Mean object trajectory. The thick solid line traces the center of mass of the object. (b) Comparison with simulated trajectory (dashed line) with the model in [\[24\]](#page-7-23). The experiment yields a significant difference. (c) Distribution of the final locations of the center of mass of the object. Note the multi-modality.

maximum power inequality are generally good representations of the relationship between the directions of friction and motion, other assumptions are not equally respected. In particular the ratio between the magnitudes of normal force and friction force at contact (i.e. the coefficient of friction) is not necessarily constant, and the ratio changes in space, with orientation, with velocity, and with time. As expected, materials that are harder yield slightly better approximations.

Our current and future work include leveraging this dataset to develop more accurate semi-parametric and stochastic models of frictional pushing; investigating its use in the context of simulation, planning, and control; as well as continued efforts in collecting experimental data for prehensile [\[30\]](#page-7-30) and non-prehensile [\[12\]](#page-7-11) contact interactions. Of particular interest are out-of-plane motions that involve different manipulation actions such as rolling or toppling.

Our long term goal is to steer away from a manipulation paradigm that relies heavily on open loop executions of motions that are planned with simple deterministic models of frictional interaction.

#### REFERENCES

- <span id="page-7-0"></span>[1] M. T. Mason, "Mechanics and planning of manipulator pushing operations," *IJRR*, vol. 5, no. 3, 1986.
- <span id="page-7-1"></span>[2] S. Akella and M. T. Mason, "Posing polygonal objects in the plane by pushing," *IJRR*, vol. 17, no. 1, 1998.
- <span id="page-7-2"></span>[3] K. M. Lynch and M. T. Mason, "Stable pushing: Mechanics, controllability, and planning," *IJRR*, vol. 15, no. 6, 1996.
- <span id="page-7-3"></span>[4] K. Y. Goldberg, "Orienting Polygonal Parts without Sensors," *Algorithmica*, vol. 10, no. 204, 1993.
- <span id="page-7-4"></span>[5] Randy C. Brost, "Automatic Grasp Planning in the Presence of Uncertainty," *IJRR*, vol. 7, no. 1, feb 1988.
- <span id="page-7-5"></span>[6] M. Dogar and S. Srinivasa, "A framework for push-grasping in clutter," *Robotics: Science and systems VII*, 2011.
- <span id="page-7-6"></span>[7] T. Meric¸li, M. Veloso, and H. L. Akın, "Push-manipulation of complex passive mobile objects using experimentally acquired motion models," *Autonomous Robots*, vol. 38, no. 3, 2015.
- <span id="page-7-7"></span>[8] M. J. Behrens, "Robotic manipulation by pushing at a single point with constant velocity: Modeling and techniques," Ph.D. dissertation, University of Technology, Sydney, 2013.
- <span id="page-7-8"></span>[9] M. Koval, N. Pollard, and S. Srinivasa, "Pose estimation for planar contact manipulation with manifold particle filters," *IJRR*, vol. 34, no. 7, June 2015.

- <span id="page-7-9"></span>[10] Y.-B. Jia and M. Erdmann, "Pose and motion from contact," *IJRR*, vol. 18, no. 5, 1999.
- <span id="page-7-10"></span>[11] K.-T. Yu, J. Leonard, and A. Rodriguez, "Shape and Pose Recovery from Planar Pushing," in *IROS*, 2015.
- <span id="page-7-11"></span>[12] N. Fazeli, R. Tedrake, and A. Rodriguez, "Identifiability analysis of planar rigid-body frictional contact," in *ISRR*, 2015.
- <span id="page-7-12"></span>[13] K. M. Lynch, "Estimating the friction parameters of pushed objects." in *IROS*, vol. 93. Citeseer, 1993.
- <span id="page-7-13"></span>[14] M. Peshkin and A. C. Sanderson, "The motion of a pushed, sliding workpiece," *IEEE Journal of Robotics and Automation*, 1988.
- <span id="page-7-14"></span>[15] H. Liu, "Pushing with a physics-based model," Master's thesis, Massachusetts Institute of Technology, 2011.
- <span id="page-7-15"></span>[16] S. Goyal, A. Ruina, and J. Papadopoulos, "Planar Sliding with Dry Friction Part 1. Limit Surface and Moment Function," *Wear*, 1991.
- <span id="page-7-16"></span>[17] M. Salganicoff, G. Metta, A. Oddera, and G. Sandini, *A vision-based learning method for pushing manipulation*. U. of Pennsylvania, Department of Computer and Information Science, 1993.
- <span id="page-7-17"></span>[18] M. Lau, J. Mitani, and T. Igarashi, "Automatic learning of pushing strategy for delivery of irregular-shaped objects," in *IEEE ICRA*, 2011.
- <span id="page-7-18"></span>[19] S. Walker and J. K. Salisbury, "Pushing using learned manipulation maps," in *IEEE ICRA*, 2008.
- <span id="page-7-19"></span>[20] J. Zhou, R. Paolini, J. A. Bagnell, and M. T. Mason, "A convex polynomial force-motion model for planar sliding: Identification and application," in *ICRA*, 2016.
- <span id="page-7-20"></span>[21] A. Torralba, R. Fergus, and W. T. Freeman, "80 million tiny images: A large data set for nonparametric object and scene recognition," *IEEE PAMI*, 2008.
- <span id="page-7-21"></span>[22] M. T. Mason, *Mechanics of robotic manipulation*. MIT press, 2001.
- <span id="page-7-22"></span>[23] S. H. Lee and M. Cutkosky, "Fixture planning with friction," *Journal of Manufacturing Science and Engineering*, vol. 113, no. 3, 1991.
- <span id="page-7-23"></span>[24] K. M. Lynch, H. Maekawa, and K. Tanie, "Manipulation and active sensing by pushing using tactile feedback." in *IROS*, 1992.
- <span id="page-7-25"></span>[25] R. D. Howe and M. R. Cutkosky, "Practical force-motion models for sliding manipulation," *IJRR*, vol. 15, no. 6, 1996.
- <span id="page-7-24"></span>[26] N. Chavan-Dafle and A. Rodriguez, "Prehensile Pushing: In-hand Manipulation with Push-Primitives," in *IROS*, 2015.
- <span id="page-7-26"></span>[27] Website for push dataset. [Online]. Available: [http://mcube.mit.edu/](http://mcube.mit.edu/push-dataset) [push-dataset](http://mcube.mit.edu/push-dataset)
- <span id="page-7-27"></span>[28] F. L. Roth, R. L. Driscoll, and W. L. Holt, "Frictional properties of rubber," *Rubber Chemistry and Technology*, vol. 16, no. 1, 1943.
- <span id="page-7-28"></span>[29] I. Clemitson, *Castable polyurethane elastomers*. CRC Press, 2015.
- <span id="page-7-30"></span>[30] R. Kolbert, N. Chavan-Dafle, and A. Rodriguez, "Experimental Validation of Contact Dynamics for In-Hand Manipulation," in *ISER*, 2016.