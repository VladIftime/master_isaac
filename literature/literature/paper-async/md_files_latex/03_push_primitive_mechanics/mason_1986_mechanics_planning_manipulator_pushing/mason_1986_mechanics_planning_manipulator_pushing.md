# Matthew T. Mason

Computer Science Department and Robotics Institute Carnegie-Mellon University Pittsburgh, Pennsylvania 15213

# Mechanics and Planning of Manipulator Pushing Operations

#### Abstract

Pushing is an essential component of many manipulator operations. This paper presents a theoretical exploration of the mechanics of pushing and demonstrates application of the theory to analysis and synthesis of robotic manipulator operations.

#### 1. Introduction

Effective robotic manipulation requires an understanding of the underlying physical processes. We need to know how the different elements of a manipulator operation combine to give a desired effect, under what circumstances an operation is appropriate, and how to select and tailor an operation to fit a given task.

Pushing is a key element of many manipulator operations. Pushing is a good way to reduce uncertainty in the locations of objects, to move many objects at once, and to move objects that are hard to grasp. Pushing, whether intentional or accidental, can readily be observed in almost every mechanical assembly task.

The primary result of this paper is a derivation of the fundamental mechanics of pushing. Pushing is a very complex mechanical process—frictional forces play the dominant role, and the applied forces are usually unpredictable. As a result, the motion of an object being pushed is partially indeterminate in many

The bulk of this research was performed at the MIT AI Lab, supported by DARPA contract N00014-80-C-0505 and ONR contract N00014-77-C-0389. The research at CMU was supported by a grant from the System Development Foundation.

The International Journal of Robotics Research, Vol. 5, No. 3, Fall 1986, © 1986 Massachusetts Institute of Technology.

practical situations. Nonetheless, the theory presented in this paper provides a foundation sufficient for analyzing and planning effective pushing operations. Some example applications are described at the end of the paper.

The use of pushing in manipulation is aptly demonstrated by the hinge-plate grasping operation illustrated in Fig. 1. The operation was devised by Pingle, Paul, and Bolles (1974) to grasp a hinge-plate prior to assembling a hinge. Besides grasping the hinge-plate, this operation eliminates initial variations in the location of the hinge-plate without sensory feedback. The forward sweeping of the hand eliminates rotational variations and some translational variations. The remaining translational variations are eliminated by the squeezing of the fingers.

The hinge-plate grasping operation progresses in stages, as shown in Fig. 2. Beginning with no contact, the fingers proceed in the direction shown. When one of the fingers strikes the hinge-plate, the hinge-plate will begin to move on the table, rotating with respect to the fingers (stage 1). Eventually the hinge-plate rotates to contact the second finger, which initiates stage 2. In stage 2, the hinge-plate translates without rotation, while one or both of the fingers slides towards the sleeves. Stage 3 occurs with one sleeve in contact with a finger, and the operation is completed when the second finger contacts its sleeve.

To understand the action of these stages, we have to relate the motion of the hinge-plate to the motion of the fingers and the all-important frictional forces. We will analyze stage 1—see Mason (1982) for an analysis of the other stages.

When stage 1 commences, one finger is in contact with the appropriate edge of the hinge-plate. We wish to ensure that the hinge-plate will rotate in the proper direction until contact with the second finger occurs. Without loss of generality, assume the initial contact is

Fig. I. Hinge grasping. Each finger performs a uniform translation without sensory feedback. Ultimately the hinge-plate will be aligned with the leading edges of the

fingers and centered between the fingers, eliminating all uncertainty in the position of the hinge-plate with respect to the fangers.

Fig. 2. The hinge-plate grasping operation progresses in stages.

From this contact, construct the normal to the hingeplate edge (Fig. 3). Construct two rays, making an angle a with the normal, where a is the arc-tangent of the coefficient of friction. The area between these two rays is the friction cone. We also indicate the center of gravity of the hinge. In Section 2 we will show that the hinge-plate will rotate counterclockwise if the friction cone passes entirely to the right of the center of gravity. With the center of gravity as shown, the proper rotation will occur no matter where the finger touches the edge, regardless of the initial orientation of the hinge-plate.

To guarantee that stage 1 terminates successfully, we would have to address two other problems. It is easy to check that the finger cannot slip off the edge. To check that the rotation is completed before the fingers close is more difficult (Mason 1982).

Implicit in the above analysis are two important assumptions. First, it is assumed that the frictional forces obey Coulomb's law (Coulomb 1 81). Second, it is assumed that frictional forces dominate the inertial forces arising from acceleration of the hinge-plate. For the hinge-plates on my office door, the results should be accurate at a finger speed of 20 cm/s or less (Mason 1982). Perhaps the most important assumption is one that is not required: there is no assumption on the distribution of support forces between the tabletop and the hinge-plate. Except for the fact that the support forces must balance the other applied forces, these forces are completely indeterminate.

The verification of hinge-plate grasping demonstrates a deep understanding of the operation. We know why it works and can find conditions that guarantee successful completion of the operation. Further-

Fig. 3. Analysis of stage 1. Counterclockwise rotation occurs, regardless of the initial orientation, for any contact between right finger

and corresponding edge of hinge-plate. a is the arc-tangent of the coefficient of friction.

Fig. 4. Generalizing the hinge-plate grasping operation to other shapes. A. Suggests a special gripper shape for gripping the box. . B. Suggests an approach ' using a multifingered hand. '

more, the scope of the operation is surprisingly large. The very same motions can be applied with identical effect to a wide range of objects, provided that a multifingered hand is available. For example, Fig. 4 shows how the fingers of a dexterous hand could be configured to apply the hinge-plate grasping operation to a simple block.

#### 1.1. OVERVIEW

So far, we have briefly considered the role of pushing in manipulation and looked at a detailed example of a grasping operation. This section surveys the important results of the paper. Section 1.2 discusses the implica tions of this work for manipulation. Sections 1.3 and 1.4 describe previous work. The body of the paper is divided into two parts: theory and application, treated in Sections 2 and 3, respectively. Each part is outlined immediately below.

## l. ~'. l. Theory

Section 2 develops a few key results in the mechanics of pushing, providing a foundation for the analysis

and synthesis of pushing operations. The basic problem is formulated as follows:

- 1. A rigid object is in planar motion on a horizontal support. The object has a single point contact with a pusher. The pusher velocity is given.
- 2. The support forces and the pushing force obey Coulomb's law. The support forces are distributed over a finite area, giving finite pressure. The only other force applied to the object is gravity. The object's center of mass is given.
- 3. Inertial forces are assumed to be negligible.
- 4. The locations and magnitudes of the support forces are indeterminate.

Point 4 is an important one: the results would have little practical value if they required detailed knowledge of the support forces. In many practical situations the support forces are very sensitive to small variations in the shape of the object and cannot be predicted.

Two problems will be considered:

- 1. What is the motion of the object?
- 2. Does the object rotate? If so, is the rotation clockwise or counterclockwise?

The answer to problem 1 is indeterminate. The object's motion depends on the distribution of support forces, which we assume is indeterminate. If we assume the support forces are given, deleting assumption 4, then a numerical procedure, illustrated in Fig. 5, can readily determine the object's instantaneous rotation center.

The answer to problem 2 is determined and, in fact, is quite easily found. We use the same three rays as in Fig. 5: the two edges of the friction cone and a ray

Fig. 5. Finding the motion of the object, given the distribution of support farces. First a plot of rotation centers is constructed by numerical means, parameterized by the angle of force y. We also construct three rays, comprising the velocity of the pusher R, and the edges of the friction cone RR and R,, making angles yR and y,.

The feasible rotation centers must lie in the segment of the plot of rotation centers delimited by XlYR) and xlYL). A line perpendicular to the ray of pushing is constructed. If this line strikes the plot within the delimited region, the intersection is the rotation center. Otherwise the nearest of XiYR)' x¡'(YL) is the rotation center.

parallel to the pusher velocity. The rays vote on the direction of rotation, with ties resulting in a pure translation (see Fig. 6). For instance, if two rays pass to the left of the center of mass, then the object will rotate clockwise.

## 1.1.2. Applications

We have already seen an application of the theory to verify a grasping operation. Section 3 includes two other applications: (1) the design of automatic orienting equipment and (2) automatic planning of grasping operations. For each problem we will demonstrate application of the theoretical results to manipulation problems.

Pushing is already used in parts-orienting equipment. The simplest application is to suspend a fence above a conveyor belt or a roller conveyor. With the proper choice of fence angle and coefficient of friction, objects arriving with arbitrary orientations will tumble along the fence until reaching the desired orientation. The theoretical results outlined above can easily be used to determine the correct parameters.

The design of robot grasping operations is quite similar, except that smaller orientation uncertainties can be tolerated due to constraints on the length of the push. The problem used for demonstration involves orientation tolerances of 30 degrees. An automatic planning procedure is demonstrated that finds the approach angle, the approach velocity, and the length of push required.

#### 1.2. DiscusSION

The work reported in this paper addresses a number of issues in manipulation research. The most obvious contribution is the focus on pushing as an important phenomenon in manipulation. We have already seen how pushing can eliminate uncertainty during a grasping operation. Some other uses of pushing are:

Placing. As with grasping, some compliance is required when placing an object on a support surface. One possible source of this compliance is that the object may slide between the fingers.

Alignment. To straighten a deck of cards, we do not place the cards on the deck one at a time. Rather, we squeeze the deck alternately at the sides and at the top and bottom. Not only is this method much faster, but the alignment of the cards is much more accurate. The same technique has been used by a robot to align a lid on a box (Albus and Evans 1976). Of course, the first stage of the hinge-grasping operation worked by aligning the hinge-plate with the leading edge of the fingers.

Handling multiple objects. Straightening a deck of cards is a good example of handling many objects simultaneously. Another example is provided by pharmacists, who use a small knife and tray to count out groups of pills. Brooms and roulette croupiers work on the same principle.

Positioning. Usually, we expect a robotic manipulator to use a pick-and-place approach to position an object. However, if the initial position and the goal position share a common support surface, pushing is sometimes preferable. Pushing an object to a particular position is complicated by the

Fig. c5. Finding the direction of rotation. a = tax-1 11. The three rays vote on the direction of the object's rotation. Ties result in pure translation. Each vote is determined

by the relation of the ray to the object's center of mass. In this case, the vote is unanimous for clockwise rotation.

indeterminacy that arises in pushing, but there are a number of ways to address this problem. Rearranging furniture is a good example of pushing to position an object.

Parts orienting and feeding. In manufacturing applications, robotic manipulators are often surrounded by parts feeders. Each part must be presented at a particular location with a particular orientation. Pushing is often a very simple way to orient parts.

Pushing operations use the mechanics of the task environment instead of stiff position control. Other operations, such as throwing, dropping, and striking, have a similar flavor (Mason 1985). These operations appear to be important in human manipulation and avoid some of the problems of pick-and-place operations. The pick-and-place approach is severely limited: a gripper must be designed for the task objects, the arm must be able to lift the object, the trajectory must fall within the workspace of the robot, and accuracy is required. Furthermore, placing an object requires a great deal of time, since the entire manipulator must be decelerated to zero velocity at the target point. These limitations are not inherent in the task; they arise in the solution that we have assumed. Humans habitually transcend these limitations by using gravity, inertia, friction, impact, and geometric constraint to get objects where they are needed.

# 1.3. PREVIOUS WORK: ROBOTIC MANIPULATION

Although there is no previous work on manipulator pushing operations, there is a great deal of work on the mechanics of other operations. A number of papers share a common approach: to study the fundamental mechanics of an operation, focusing on the interactions between the robot and the objects in the task domain. Besides the present work on pushing, this approach has been applied to peg insertion, gross and fine motions, compliant motion, and a variety of grasping operations.

The simplest manipulator operation is programmed positioning of the manipulator, which involves manipulator kinematics and dynamics, control, and trajectory planning. If we focus on the interactions (or avoidance of interactions) between the robot and the environment, then the main problem is to plan a collision-free path for the robot and any objects that might be in the robot's grasp. There is a wealth of recent research on this problem, but a good starting point is Lozano-Perez ( 1981 ). The main contribution of this paper, based on the earlier work in Lozano-Perez (1976), Udupa (1977), Lozano-Perez and Wesley ( 1979), is its use of configuration-space to pick grasp points and plan collision-free paths.

After programmed positioning, the most-studied manipulator operation is the insertion of a peg in a hole. Two seminal papers in this area are Simunovic (1975) and McCallion and Wong (1975). Some representative recent works are Ohwovoriole and Roth ( 1981 ) for the kinematics of insertions, and Whitney (1982) for the dynamics. The results of this research have found wide application in industry in the form of high-performance, peg-insertion devices.

Another important operation is known as compliant motion: i.e., positioning an object that is constrained by contact with the environment. This is equivalent to controlling the position of the robot on a surface in configuration-space. Two different approaches to compliant motion have been widely investigated. The first approach, hybrid force/position control, is to use position control along tangents of the constraint surface and force control along normals to the constraint surface. The rationale for hybrid force/position control is provided by Takase et al. (1974) and Mason (1981) and an implementation is described by Raibert and

Craig ( 1981 ). The idea goes back as far as Inoue ( 1 71) and Paul (1972), however. The second approach to compliant motion is to make the manipulator behave like a spring or a damper. Recent work on this approach includes Salisbury (1980) and Hogan (1985), but the idea goes back at least as far as Nevins and Whitney ( 1974).

Lozano-Perez, Mason, and Taylor (1984) explore automatic planning of fine motions, using well-defined models of compliant control, contact forces, and uncertainty. Mason (1984) uses the same models to explore a planner and shows that the planner is complete, i.e. if a given manipulation problem has a solution, the planner will find it. Erdmann (1984) shows that Mason's planning procedure is not computable for very complex environments, and he also implements a variant planner.

Finally, there are a variety of grasping operations that have inspired mechanical analysis. Some of the spatial-planning issues are treated by Lozano-Perez (1976; 1981 ). Hanafusa and Asada (1977) define a potential field to find stable grasping configurations for springy, frictionless fingers. Salisbury (1982) used kinematic analysis to perform multifinger repositioning of an object in a hand. Cutkosky (1985) explores a number of issues in stable grasping and compliance, most notably the consequences for stable grasping of nonpoint contact between fingers and the object. z

#### 1.4. PREVIOUS WORK: MECHANICS OF PUSHING

This paper closely follows Mason (1982). Errors in some of the proofs have been eliminated, some conjectures have now been proven, and the statement of primary results is generally improved. Prior to Mason (1982) there is very little work on the mechanics of pushing. A short survey of the relevant history follows.

#### 1.4.1. Coulomb's Law

Coulomb conducted the first thorough investigation of sliding friction (Coulomb 1781 ). He searched for dependencies of the frictional force on every conceivable parameter, including the time of repose before sliding commences, elapsed time of sliding, speed of sliding, types of surfaces, cleanliness of surfaces, and, of

course, the magnitude of the normal force. Although all of these factors affect the frictional force, Coulomb found that over an enormous range of materials and normal forces the frictional force is virtually independent of every factor except the materials and the magnitude of the normal force. The result is now commonly identified as Coulomb's Law: the tangential force of friction during sliding is directed opposite to the direction of motion, with magnitude proportional to the normal force. The constant of proportionality is known as the dynamic coefficient of friction, and depends on the contacting materials, but not on the speed of motion. If no motion is in progress, then the tangential force is not completely determined-the magnitude of the tangential force is bounded and the direction is unconstrained. The bound on magnitude is determined by the static coefficient of friction, which is often higher than the dynamic coefficients of friction.

This law is not original with Coulomb; Amontons (1699) had previously asserted the law and had published engineering tables based on it. But the earliest known statement of the law appears in the notes of Leonardo da Vinci (Truesdell 1968, p. 9):

Every body resists in its friction with a power equal to the fourth of its heaviness if the motion is plane [or slow?] and the surfaces dense and polished [or clean?].

Leonardo thought that there was a universal coefficient of sliding friction - one fourth. This value is nearly correct for many combinations of common materials. In spite of the prior claims of Leonardo and Amontons, there is ample justification for naming the law for Coulomb. It is doubtful that Leonardo engaged in any substantial experimental verification of the law (Truesdell 1968, p. 9), and it is clear from the tables published by Amontons that most of the data is extrapolation from a few experiments of limited scope (Gillmor 1971, p. 123).

# 1.4.2. Friction Cone and Friction Angle

There is a useful geometric interpretation of Coulomb's law, apparently first constructed by Moseley ( 1835). Consider a point moving on a surface (Fig. 7). We construct the vector f representing total contact force acting on the point, comprising the normal force

Fig. 7. Friction cone. Coulomb's law relating the magnitude of normal and tangential components of force has a geometric interpretation depicted above: the total

force must make an angle tan-Ill with the surface normal. The set of all vectors satisfying this geometric condition form the cone.

fn due to the surface stiffness and the tangential force f due to friction. Coulomb's law states that these are related by f =,uf,~. If we construct the normal to the surface, then Coulomb's law is equivalent to the statement that the total contact force f will make an angle a = tan-1 it/In with the normal. The set of vectors making the proper angle with the surface normal form a cone, called the friction cone. The angle a is called the friction angle.

## 1. 4.3. General Planar Motion with Friction

I have located only three works dealing with sliding friction during general planar motion (Jellett 1872; Prescott 1923; MacMillan 1936). Jellett avoids most of the complications of general planar motion, but he does show that if all the applied forces (excluding the support contact forces) reduce to a single force, then the problem of planar motion of a body reduces to the problem of the motion of a point in the plane.

Prescott provides the first substantial discussion of friction in the context of general planar motion of a solid body. He developed expressions for the moment of frictional forces about the instantaneous rotation center, under the assumption that the weight of the

object is uniformly distributed on its base. He also considers the problem of the motion of a sliding object subject to an additional force applied at a single point. For objects with known support forces at a finite number of supporting points, he develops the conditions under which the object will rotate about a given point. For a special case (two points of support equidistant from the point of application of the external force), he derives the set of all possible rotation centers. Some of the methods we will explore in this paper rely upon construction of similar plots of rotation centers. I have implemented a numerical algorithm to construct such plots for any given distribution of support forces.

MacMillan considered a more general problem. Rather than assuming uniform pressure, or known pressure with a finite number of points of support, MacMillan assumes a linear pressure distribution:

$$p(x, y) = ax + by + c$$

where (x, y) represents a point in the contact region. The three parameters a, b, and c can be determined if the external applied forces are known. He then derives expressions for the force and moment, given a planar motion of the object. Unfortunately these expressions are complicated, requiring evaluation of seven different line integrals taken around the boundary of the contact region. More useful is MacMillan's observation that during a pure translation, the system of frictional forces arising in the contact area may be reduced to a single force acting through a fixed point-the center offriction.

The results reported here generalize MacMillan's center of friction to arbitrary pressure distributions. We will also develop expressions for force and moment of friction during general planar motion. Without assuming any particular form for the pressure distribution, the location of the center of friction can be calculated from knowledge of the external applied forces. The generality of this result is very important, since in practice the pressure distribution is usually complicated and unpredictable.

MacMillan used the center of friction to find the frictional force and moment during translation, but we have seen that its utility goes well beyond this-its position determines whether an object will rotate when it is pushed, and, if so, whether clockwise rotation or

counterclockwise rotation occurs. This characterization of an object's motion is incomplete, but it is sufficient for analyzing and planning manipulator pushing and grasping operations.

2. Mechanics of Pushing

of addressing one problem at a time.

#### 1.4.4. Modern Work

The problem of general planar motion in the presence of friction has not attracted much attention recently. Frictional force is, of course, nonconservative, rendering many of the techniques of theoretical mechanics inapplicable. Most of the discussion of the problem in the mechanics literature is limited to systems with a single degree of freedom. There has been some modern applied mechanics work on sliding friction in oscillatory systems (Den Hartog 1956), but it does not extend to general planar motion.

Modern texts on sliding friction treat single degreeof-freedom problems, or problems that clearly reduce to a single degree of freedom or to a point on a plane. It is interesting to note that textbooks never discuss the circumstances under which the crucial reductions may be applied but still expect students to apply the reductions. It is also interesting that this omission is not noticed by the students.

The field of tribology (Bowden and Tabor 1950) focuses on the description and mechanism of frictional forces, not on the motion of objects subject to friction. One result of this work is the development of descriptions that are more accurate than Coulomb's law. This is certainly relevant to the present work, since determination of the frictional forces is an important part of the theory. Nonetheless, we will assume Coulomb friction throughout the paper and will assume that the static and dynamic coefficients of friction are equal. To quote Prescott (1923, p. 106):

But all these small inaccuracies we shall disregard for two very good reasons: firstly, because the laws are sufficiently accurate for most practical purposes; and secondly, because exact laws are not known.

To this I should add that the focus of this paper is to characterize the motion of objects being pushed. The results reported here go considerably further than the previous works cited, despite relaxing assumptions on the form of the pressure distribution. Further generalThis section presents a theoretical development of the mechanics of pushing, providing the foundation for analysis and synthesis of manipulator pushing operations. We will focus on the two problems described previously: (1) which way does an object being pushed rotate and (2) where is the object's instantaneous rotation center? First, though, we will develop some useful expressions for the total frictional force and the moment of frictional force arising from general planar motion.

ization would certainly be desirable, but it is a matter

#### 2.1. Force and Moment of Friction

Any planar motion is either a simple translation or a rotation about some instantaneously motionless point. Translation and rotation are handled separately. For translation, we find that the system of frictional forces reduces to a single force through a point whose position is independent of the direction of translation. This point is the *center of friction* (MacMillan 1936). No analogous reduction occurs for rotation.

To find useful expressions for the force and moment of sliding friction some nomenclature is required:

- R region of contact between object and support surface
- dA differential element of area of R
- x position of dA
- $p(\mathbf{x})$  pressure at  $\mathbf{x}$ 
  - v<sub>y</sub> velocity of object relative to support at x
  - f<sub>c</sub> total frictional force
  - $m_t$  signed magnitude of the frictional moment

We will assume Coulomb friction with a coefficient of friction  $\mu$ . The normal force at x is given by

 $p(\mathbf{x})dA$ ,

so the application of Coulomb's law gives for the tan-

gential force at x

$$-\mu \frac{\mathbf{v}_x}{|\mathbf{v}_x|} p(\mathbf{x}) dA.$$

The total frictional force  $f_f$  is obtained by integrating over the support contact region R:

$$\mathbf{f}_f = \int_{R} -\mu \frac{\mathbf{v}_x}{|\mathbf{v}_x|} p(\mathbf{x}) dA. \tag{1}$$

The total frictional moment is obtained by similar means:

$$m_f \hat{\mathbf{k}} = \int_R \mathbf{x} \otimes -\mu \frac{\mathbf{v}_x}{|\mathbf{v}_x|} p(\mathbf{x}) dA,$$
 (2)

where  $\hat{\mathbf{k}}$  is the unit vector normal to the x - y plane.

#### 2.1.1. Translation

During a pure translation, all points in the object move in the same direction  $\mathbf{v}_x/|\mathbf{v}_x|$ . Hence this term may be factored out of the integral along with the coefficient of friction:

$$\mathbf{f}_f = -\mu \frac{\mathbf{v}_x}{|\mathbf{v}_x|} \int_{R} p(\mathbf{x}) dA, \tag{3}$$

$$m_f \hat{\mathbf{k}} = -\mu \int_{R} \mathbf{x} p(\mathbf{x}) dA \otimes \frac{\mathbf{v}_x}{|\mathbf{v}_x|}.$$
 (4)

Let  $f_0$  be the total normal contact force, and let  $x_0$  be the centroid of the pressure distribution p(x):

$$f_0 = \int_R p(\mathbf{x}) dA,$$

$$\mathbf{x}_0 = \frac{1}{f_0} \int_{\mathbb{R}} \mathbf{x} p(\mathbf{x}) dA.$$

Substituting into Eqs. (3) and (4) we obtain

$$\mathbf{f}_f = -\mu \, \frac{\mathbf{v}_x}{|\mathbf{v}|} f_0,$$

$$m_f \hat{\mathbf{k}} = \mathbf{x}_0 \otimes \mathbf{f}_f$$
.

Inspection of these equations shows that the system of frictional forces reduces to a single force applied at  $x_0$ . This result is the basis upon which all other results rest.

Theorem 1 The system of frictional forces of a translating object reduces to a single force, applied at the centroid of the pressure distribution, whose direction is opposite to the direction of translation. Proof: given above.

When the system of frictional forces reduces to a single force applied at a particular point, whose location is independent of the direction and velocity of motion and whose direction is opposite to the direction of motion, we say that that point is the *center of friction*. Theorem 1 says that during translation a center of friction exists, and it is the centroid of the pressure distribution. We also know that the magnitude of the frictional force is the product of the applied normal force with the coefficient of friction, and hence that the problem reduces to the problem of a point in a plane with the same coefficient of friction, although that is less important for our purposes.

This result may appear to be useless in those cases where the pressure distribution is indeterminate. In many practical situations, the pressure distribution is not known. Microscopic variations in the contact surfaces may drastically alter the pressure distribution. Fortunately, the contact pressure and moment must balance certain components of the applied forces, fixing the position of the centroid in the process.

First, if gravity alone acts on the object, the centroid of the pressure distribution lies directly beneath the center of mass. For the case of an arbitrary system of applied forces, let  $f_{a,z}$  be the normal component of the total applied force, let  $m_{a,x}$  and  $m_{a,y}$  be the total moments of applied force about the x-axis and y-axis, respectively. Then the magnitude of the total force due to the contact pressure is

$$f_0 = f_{az}$$

and the center of friction is

$$\mathbf{x_0} = -\frac{1}{f_{a,x}} \begin{bmatrix} m_{a,y} \\ m_{a,x} \end{bmatrix}.$$

With this result one can find the center of friction given the applied forces.

The final complication is that we must consider the pushing force as one of the applied forces. If the support forces are not known, then the pushing force is generally indeterminate. If the pushing contact is above or below the support plane, the location of the center of friction can vary in an indeterminate fashion. Fortunately, the method for predicting the direction of rotation is not affected by these variations (Mason 1982). Hence, we can ignore the pushing force when constructing the center of friction.

#### 2.1.2. Rotation

Let  $x_r$  be the instantaneous center of rotation, and let  $\omega$  be the angular velocity vector. The velocity at x is given by

$$\mathbf{v}_{x} = \boldsymbol{\omega} \times (\mathbf{x} - \mathbf{x}_{r}).$$

Since  $\omega = \dot{\theta} \hat{\mathbf{k}}$ , we have

$$\mathbf{v}_{x} = \dot{\theta}(\mathbf{\hat{k}} \times (\mathbf{x} - \mathbf{x}_{r})).$$

The direction of the motion at x is given by

$$\frac{\mathbf{v}_x}{|\mathbf{v}_x|} = \operatorname{sgn}(\dot{\theta})\hat{\mathbf{k}} \times \frac{\mathbf{x} - \mathbf{x}_r}{|\mathbf{x} - \mathbf{x}_r|}.$$

Substituting into Eqs. (1) and (2), we obtain after simplification:

$$\mathbf{f}_f = -\mu \operatorname{sgn}(\dot{\theta})\hat{\mathbf{k}} \times \int_R \frac{\mathbf{x} - \mathbf{x}_r}{|\mathbf{x} - \mathbf{x}_r|} p(\mathbf{x}) dA; \qquad (5)$$

$$m_f \hat{\mathbf{k}} = -\mu \operatorname{sgn}(\hat{\theta}) \hat{\mathbf{k}} \int_{\mathcal{R}} \mathbf{x} \cdot \frac{\mathbf{x} - \mathbf{x}_r}{|\mathbf{x} - \mathbf{x}_r|} p(\mathbf{x}) dA.$$
 (6)

Comparing with Eqs. (3) and (4), we see that the simplification obtained for the case of translation does not apply to rotation. The center of friction plays no apparent role during rotation of an object.

It is sometimes useful to consider a translation to be a rotation about an infinitely distant rotation center. Henceforth we will allow the range of x, to include

Fig. 8. Rays. The rays of motion and of pushing,  $R_{\rm M}$  and  $R_{\rm P}$  respectively, are parallel to the velocities of the two points in contact. The ray of motion refers to the motion of the contact point in the object; the ray of pushing refers to the motion of the contact point in the pusher. The rays  $R_{\rm R}$  and  $R_{\rm L}$ 

delimit the friction cone. The ray of force  $R_F$  is parallel to the force applied to the object at the pushing contact. The three cases illustrated exhaust the possibilities: either  $R_F = R_R$  and  $R_M$  is to the left of  $R_P$ ;  $R_F = R_L$  and  $R_M$  is to the right of  $R_P$ ; or  $R_F$  lies in the friction cone and  $R_M = R_P$ .

points at infinity, so that Eqs. (5) and (6) may be used for both rotation and translation.

#### 2.2. WHICH WAY DOES IT TURN?

In this section we derive the method for determining the sense of rotation of an object being pushed. The arguments often depend on comparing various rays with the center of friction. When we say that a ray dictates the sense of rotation, we mean that if the center of friction lies to the left of the ray, counterclockwise rotation occurs, and if the center of friction lies to the right of the ray, clockwise rotation occurs. If the center of friction lies on the ray, translation occurs.

The five rays of interest and their interrelationships are shown in Fig. 8. Both the ray of force and the ray of motion dictate the sense of rotation, but these rays are often indeterminate. The primary result, stated in theorem 4, says that the rays  $R_R$ ,  $R_L$ , and  $R_P$  vote on the sense of rotation. The argument is indirect, however. First we will prove that the ray of motion  $R_M$  dictates the sense of rotation (theorem 2), and that the ray of force  $R_F$  dictates the sense of rotation (theorem 3).

Theorem 2 The ray of motion dictates the sense of rotation. Proof: First, we construct an x-y coordinate system with the origin at the contact point and with the y-axis parallel to the ray of motion (see Fig. 9). The instantaneous rotation center must lie on a line perpendicular to the velocity at the contact point, i.e., it must lie on the x-axis. Let  $x_r = (x_r, 0)^T$  be the rotation center.

Define  $m_f(x_r)$  giving the total scalar frictional moment as a function of the directed distance to the rota-

Fig. 9. Coordinate conventions for the proof of theorem 2. The origin coincides with the contact point, and the

y-axis coincides with the ray of motion. Consequently, the instantaneous rotation center lies on the x-axis.

Fig. 10. A typical plot of  $m_f(x_r)$ . Analysis of Eq. (6) shows that  $m_f(x_r)$ , where  $x_r = (x_r, 0)^T$ , is continuous

and strictly decreasing.
There is a unique root,
whose sign is dictated by the
ray of motion.

tion center. The domain of this function includes infinity, but not zero. We consider the high, positive, x-axis to be connected to the low, negative, x-axis at infinity. Since the pushing force is applied at the contact point, it exhibits no moment about the origin. Hence a rotation center giving sliding equilibrium would be a root of  $m_f(x_r)$ .

Appendix I shows that  $m_f(x_r)$  is continuous and nonincreasing. As  $x_r$  approaches zero from above,  $m_f(x_r)$  approaches  $\mu \int |\mathbf{x}| p(\mathbf{x}) dA$ , which is positive. As  $x_r$  approaches zero from below,  $m_f(x_r)$  approaches  $-\mu \int |\mathbf{x}| p(\mathbf{x}) dA$ , which is negative. Hence as  $x_r$  increases from small positive values, through infinity, and on up to low negative values,  $m_f$  passes continuously from positive values to negative values. A typical plot is shown in Fig. 10.

The result is obtained by considering the moment during translation  $m_f(\infty)$ . During translation, the moment is easily found by theorem 1. For instance, assume that the center of friction lies in the right halfplane. We know that the tangential support forces reduce to a single force acting through the center of friction. This will give a negative moment, i.e.  $m_f(\infty) < 0$ . We conclude that the root of  $m_f(x_r)$  lies in the positive x-axis and that clockwise rotation occurs. We can likewise show that if the center of friction is in the left half-plane, counterclockwise rotation occurs. If the center of friction is on the y-axis, the root is at infinity, so translation occurs.

If we could determine the ray of motion, we could

determine the sense of rotation. However, the ray of motion is often indeterminate, so theorem 2 doesn't solve the problem by itself. The next step is to prove a similar result for the ray of force  $R_F$ .

Theorem 3 The ray of force dictates the sense of rotation. Proof: Let  $\gamma$  be the angle of the ray of force relative to  $\overline{\mathbf{x}_c \mathbf{x}_0}$  and let  $\mathbf{x}_r(\gamma)$  be the rotation center as a function of the force angle. Appendix II shows that  $\mathbf{x}_r(\gamma)$  exists and is continuous. Theorem 1 says that if  $\gamma = 0$ ,  $\pi$ , then  $|\mathbf{x}_r| = \infty$ , and that for all other  $\gamma$ ,  $|\mathbf{x}_r|$  is finite.

Consider the interval  $\Gamma_{CW} = (0, \pi)$ . We will prove that all force angles in this interval lead to clockwise rotation. Since  $\mathbf{x}_r(\gamma)$  is continuous, it maps  $\Gamma_{CW}$  into a connected set, which does not intersect the line at infinity. Nor does it intersect  $L_\perp$  (see Fig. 11A); such a point would give a ray of motion passing through  $\mathbf{x}_0$ , which would imply  $|\mathbf{x}_r| = \infty$ .

Thus all of  $\mathbf{x}_r(\Gamma_{CW})$  is confined either to the right of  $L_\perp$  or to the left of  $L_\perp$ . We can determine which of these two alternatives holds by considering the case shown in Fig. 11B, where  $\gamma=\pi/2$ . This case corresponds to pulling the object by a rope running upward in the figure. The ray of motion has a positive vertical component—the rope is performing work to overcome the frictional forces. Hence the ray of motion passes  $\mathbf{x}_0$  on the left, dictating a clockwise rotation. We conclude that  $\mathbf{x}_r(\Gamma_{CW})$  lies entirely to the right of  $L_\perp$ , as shown in Fig. 11A, giving a clockwise rotation in every case. The proof for  $\gamma \in \Gamma_{CCW}$  is similar.

When an object is being pulled by a rope, the ray of force coincides with the rope. In this case, theorem 3

Fig. 11. Constructions for theorem 3. A. Shows a plot of  $x_r(\gamma)$ , for  $\gamma \in (0, \pi)$ . The plot lies entirely to one side of  $L_{\perp}$ . B. Shows the case of

 $\gamma = \pi/2$ . This case shows that the plot in A must lie to the right of  $L_{\perp}$ , which implies clockwise rotation.

Fig. 12. An example illustrating indeterminacy in the sense of rotation. Two equal support forces are applied at  $\mathbf{x}_1$  and  $\mathbf{x}_2$ . The contact point coincides with the center of

friction. A vertical ray of force (motion) can arise from a rotation center anywhere on the line through  $\mathbf{x}_1$  and  $\mathbf{x}_2$ , excluding the segment between the two points.

can be applied directly, but usually the ray of force is indeterminate. We now have the ammunition to prove the primary result, which gives the sense of rotation using easily observed data.

One of the assumptions stated in Section 1 is that the support pressure must be finite. When this assumption is relaxed, the statements of theorems 2 and 3 must be softened slightly. In Fig. 12 all of the support force is applied at two points, giving infinite pressure at those points. In this case, the theory correctly predicts the sense of rotation if the ray of motion (or force) passes the center of friction on either side. But if the ray of motion (or force) passes through the center of friction, the sense of rotation is undetermined translation might occur, but a rotation in either direction is also possible. This is clear in the figure, where the contact point coincides with the center of friction. A rotation center anywhere to the left or right of the support points, on the line through the support points, gives the same instantaneous directions of motion for the support points and gives the same frictional forces. A case such as this is important from a theoretical standpoint, but this instance of indeterminacy does not present any practical difficulties. In practice, the question of whether an object could translate or must translate is settled not by the instantaneous analysis being presented here but by a stability analysis introduced in Section 3.

Theorem 4 The rays  $R_L$ ,  $R_R$ , and  $R_P$  vote to determine the sense of rotation. Proof: The vote can be represented by the tree shown in Fig. 13. First, consider the situation if the votes of  $R_L$  and  $R_R$  agree. The ray of force  $R_F$  must lie in the friction cone, so we can invoke theorem 3 to show that the vote is correct. That takes care of four of the eight possible outcomes of the poll, as indicated in the figure. Now consider the case indicated by the box, where the votes cast by  $R_P$ ,  $R_L$ , and  $R_R$  are CW, CW, CCW, respectively. As in Fig. 8, there are three cases to consider:

- 1.  $R_F = R_L$ . By theorem 3, this implies clockwise rotation.
- 2.  $R_M = R_P$ . By theorem 2, this implies clockwise rotation.
- 3.  $R_F = R_R$ . By theorem 3, this implies counter-clockwise rotation.

For cases 1 and 2, the outcome of the vote is correct. We will complete the proof by proving that case 3 cannot happen. Figure 14 shows the situation arising from case 3. The votes of  $R_L$  and  $R_R$  imply that the center of friction is in the friction cone. A typical choice is shown in the figure. Now  $R_P$  voted CW, so the center of friction is on its right. Counterclockwise rotation implies that the center of friction is to the left of the ray of motion  $R_M$ . The possible directions of  $R_P$ and  $R_M$  are shown by arcs in the figure. Note that both  $R_P$  and  $R_M$  must be above the pushing constraint — otherwise the pusher would not be pushing. Now it is obvious from the figure that the ray of motion must lie to the right of the ray of pushing, which according to Coulomb's law would give us  $R_F = R_L$ , a contradiction. We conclude that either case 1 or case 2 must apply, and hence that the outcome of the vote is correct. That concludes the proof for the case indicated by the box. The remaining three cases are similar.

Figure 15 shows a few different pushing configurations, each resulting in clockwise rotation. The differences between these cases are particularly interesting in

Fig. 13. Construction for theorem 4. A tree showing the possible results of polling Rp, R,, and RR' The four cases where RL and R, agree are easily verified. The remaining four cases are similar to the case marked by the square, which is analyzed in Fig. 14.

Fig. 14. For the case marked with a square in Fig. 13,

assume the outcome of the vote to be incorrect. The center of friction Xo is constrained to lie in the friction cone above the pushing constraint, and the rays of pushing and of motion are constrained as shown. But a ray of pushing to the left of the ray of motion would imply that R, = R, , contrary to assumption.

the context of manipulation. If we are pushing an object with a finger, then for Fig. 15A the object's rotation depends on the finger motion and on the finger orientation. In Fig. 15B the object's rotation depends only on the finger's motion. In Figs. 15C and 15D the object's rotation is independent of the finger's orientation and the finger's motion!

Fig. 15. Typical pushing configurations. The theory predicts clockwise rotation in each case. In A the friction cone varies with the angle of the pusher, whereas in B, C, and D it is independent of the angle of the pusher. In A,

C, and D, the pusher velocity does not affect the sense of rotation. The prediction for situation D may seem counter-intuitive, but it may readily be confirmed with a glass ashtray and a pencil point.

### 2.3. WHERE IS THE INSTANTANEOUS ROTATION CENTER? .

In this section we will investigate the circumstances under which the trajectory of the pushed object may be completely determined. The simplest characterization of a planar motion is in terms of instantaneous rotation centers. For pushing, this is especially convenient, since, once the rotation center is determined, the remaining variable (the speed of rotation) is a simple function of the speed of the pusher.

The motion of an object being pushed generally depends on the distribution of support forces. The basic approach of this section is the use of the plot of rotation centers, which is compiled from the support forces and the pushing contact location. Once the plot is constructed, the instantaneous rotation center may readily be obtained for a variety of problems. If the ray of force or the ray of motion is given, the instantaneous rotation center is determined directly. If the friction cone and ray of pushing are given, the procedure illustrated in Fig. 5 is applicable. The plot of rotation centers also provides some useful insights into pushing.

Fig. 16. Construction of the motion cone. The friction cone constrains the force angle to lie in the interval [YR' Yd, so the rotation center is confined to the

segment x¡'([YR, Yd). This in turn constrains the angle of the ray of motion to an interval, which defines the motion cone.

Theorem 5 The procedure described in Fig. 5 for finding the instantaneous rotation center is correct. Proof Refer to Fig. 16. The first step is to define and construct a motion cone comprising all feasible rays of motion. The motion cone plays the same role for the ray of motion that the friction cone plays for the contact force. The construction is simple: since the contact force must lie in the friction cone, the angle of force y must lie in the interval [yR, yL]. Define yr(y) to be the angle of Il with respect to ~ . Since x,(Y) is continuous, with x, =1= XC, Ij/(Y) is continuous. Appendix II shows that V/(y) is strictly increasing. Hence the feasible rotation centers form a segment of the plot, delimited by x,(yR) and x,(yL). A feasible ray of motion must be perpendicular to a ray through a feasible x&dquo; so the direction of motion must lie in the interval [If/(YR) + 0/2, V(7L) + 0/2]. This interval of directions defines a cone, the motion cone, within which the ray of motion must lie.

The rest of the proof is simple. Suppose the ray of pushing lies to the left of the motion cone. Then the object must slide rightward along the pusher, and the ray of force is equal to RL. The rotation center is x,(yL). Similarly, if the ray of pushing lies within the motion cone, the rotation center is x,(yp), and if the ray of pushing lies to the right of the motion cone, the rotation center is x,(yR).

The plot of rotation centers has a value that goes

beyond its practical utility in predicting motions. It is a valuable tool for thinking about pushing, as illustrated in construction of the motion cone. There are some conditions that the plot of rotation centers must always satisfy. Since a ray of force through the center of friction implies translation, the plot of rotation centers passes through infinity twice as the force angle passes from 0 to 2n. In fact, the rotation center retraces the same curve twice, once for clockwise rotations, and once for counterclockwise rotations. Another useful observation is that the plot of rotation centers tends to dwell at points of support. This means that you can &dquo;walk&dquo; your refrigerator using subtle shifts of weight, rather than actually tipping it from one leg to another.

# 3. Application to Robotic Manipulation

This section focuses on two example applications of the mechanics of pushing to practical manipulation problems. The goal is to demonstrate the value of the theory and also to illustrate the application process.

#### 3.1. DESIGN OF AUXILIARY ORIENTING MACHINERY

In many manufacturing applications, parts to be acquired by a manipulator must be presented accurately at a specified position and orientation. There is a resulting need for machines that can accept a part with variable orientation and reliably move the part to a desired orientation. Usually this is accomplished by special-purpose machines, which are designed for a specific part shape. The main problem with this approach is that new machines have to be designed and fabricated for every new task. This compromises the ' advantages of robots-although the robot can be reprogrammed for a new task quickly, the supporting auxiliary machinery must still go through a timeconsuming and expensive development process.

In this section we show a simple method for orienting a wide variety of parts. This method greatly simplifies the problem of designing the auxiliary orienting machine; it even allows the orienting to be performed by the robot, eliminating the auxiliary machine altogether. The method employs a fence that rotates the

Fig. 17. This system, installed in a DuPont product distribution center, automatically orients boxes. The rollers are skewed to push the boxes into the fence. The boxes roll and slide until aligned with the fence.

object to a known orientation just by pushing in a straight line. The motion and orientation of the fence are determined by the shape of the object and the location of its center of friction. The approach may be implemented by suspending a fence across a conveyor or by adding a simple auxiliary fence to the robot. Although the method has been in use for some time (see Fig. 17, for instance) the principles had not been elucidated before.

The instantaneous analysis of an object being pushed by a fence was developed in the previous section. We can only go so far as to say which direction the object will rotate, depending on the location of the center of friction with respect to the ray of pushing and the two edges of the friction cone.

For our present purposes, an instantaneous analysis is not enough—we need a global analysis to determine whether the orientation of the object will converge. This global analysis is obtained by a simple construction (see Fig. 18). We roll the object along the fence, noting the locus of the center of friction. We orient the construction so that the intermediate ray from  $\{R_L, R_R, R_P\}$  is vertical. Where the locus of the center of friction is horizontal, an equilibrium occurs. Where the locus has a local minimum, a stable equilibrium occurs. Thus the locus of the center of friction has much of the flavor of a potential curve, although the analogy goes no further than identifying equilibria.

Fig. 18. The motion of a box tumbling along a fence can be predicted by examining the path of the center of friction. The diagram is drawn with the intermediate

ray from  $\{R_L, R_R, R_P\}$ , that is, the ray with the deciding vote, pointing straight up. Local minima in the path correspond to stable orientations for the box.

Once this construction is discovered, analysis and synthesis of fence-pushing operations is very simple. After constructing the locus of the center of friction at some arbitrary angle, the construction is rotated to an angle that eliminates all undesirable minima. Then we must arrange for the intermediate ray to be vertical. Presumably the most reliable way to get an intermediate ray is to choose a coefficient of friction large enough to ensure that the ray of pushing lies inside the friction cone and then to use a vertical ray of pushing. This method was applied to orient a wooden block using a fence suspended over an x-y table (Fig. 19). The fence was constructed from aluminum and covered with rubber to obtain a large coefficient of friction. The resulting procedure reliably oriented and positioned the block, starting from a completely unconstrained initial orientation, with variations in initial position of up to 2 cm. The limiting factor was the workspace of the x-y table, which limited the pushing distance.

#### 3.2. Automatic Planning of a Push-Grasp

The method of orienting by pushing, described in the previous section, can be incorporated into a grasping motion to eliminate rotational uncertainty. The object is pushed by one finger for a time, and then the second finger squeezes the object to complete the grasp. We will refer to this operation as the *push-grasp*. There

Fig. 19. Orienting parts by pushing was demonstrated in the laboratory, using a fence suspended over an x-y table. Two stages of pushing were used to eliminate all uncertainty in the block's position and orientation.

Fig. 20. The parameters for grasping this spring-nut were planned automatically. The finger angle 0 and pushing direction /If were chosen to

give the greatest latitude in initial orientation. The operation tolerated errors in excess oaf 15 degrees.

is a similar operation, which we might call the squeeze-grasp, in which the fingers squeeze the object without the initial pushing stage. The squeeze-grasp is quicker than the push-grasp, but the push-grasp can tolerate larger angular uncertainties.

Figure 20 shows an example of the push-grasp operation that was planned automatically. The object is a large steel nut with a spring attached. Working from a model of the object's shape, including the location of the center of gravity, a computer program selected the best edge to align with the finger and then calculated a finger angle and pushing direction that would cause the object to rotate to the chosen edge. The given angular uncertainty was ± 15 degrees (in tests the operation could tolerate much greater variations).

Besides the finger orientation and the pushing direction, the pushing distance d must be determined. Since the actual rate of rotation is indeterminate, we must think in terms of the worst case. The procedure searches a few plausible pressure distributions, calculates a lower bound on the rotation rate, and also throws in a fudge factor for good measure. Further work is required, both theoretical and applied, to improve this estimation procedure.

# Acknowledgments

This research is largely the result of my long association with Tomas Lozano-Perez, Marc Raibert, and John Hollerbach. I would also like to thank Berthold Horn and Patrick Winston for their support at MIT. Mary Mason provided some of the figures, as well as other, less tangible, forms of support.

# Appendix I

We must show that m¡(xr) is continuous and nonincreasing. We assume that the contact point is at the origin and the ray of pushing lies on the y-axis (see Fig. 9). The center of rotation is on the x-axis: thus we can write x,. = (xn 0). The expression for the moment of the frictional forces, previously derived, is

$$m_f = -\mu \operatorname{sgn}(\dot{\theta}) \int_R \mathbf{x} \cdot \frac{\mathbf{x} - \mathbf{x}_r}{|\mathbf{x} - \mathbf{x}_r|} p(\mathbf{x}) dA.$$
 (A1)

The most direct way to show that m f(x,) is continuous and nonincreasing is to differentiate Eq. (A 1 ) with respect to Xn but this is complicated by the fact that x, can be infinite. This problem is solved by projecting the extended plane of rotation centers onto a sphere. We will rewrite Eq. (A 1 ) to get the moment as a function of a spherical coordinate (1, which is always finite. Then differentiation is straightforward.

The projection is illustrated in Fig. 21. Expressions for the transformation are easily derived using similar triangles:

$$\begin{bmatrix} x \\ y \end{bmatrix} = -\frac{1}{z'} \begin{bmatrix} x' \\ y' \end{bmatrix},$$

Fig. 21. The topology of the space of rotation centers is the topology of the sphere. This becomes apparent using central projection of the sphere. The upper hemi-

sphere gives counterclockwise rotations, the lower hemisphere gives clockwise rotations, and the equator maps into a line at infinity, giving translations.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = -z' \begin{bmatrix} x \\ y \end{bmatrix},$$
$$z' = \frac{\operatorname{sgn} \dot{\theta}}{\sqrt{x^2 + y^2 + 1}}.$$

We are interested in rotation centers giving a ray of motion along the positive y-axis. When projected onto the sphere, these rotation centers define a great semicircle defined by  $x_r' > 0$ ,  $y_r' = 0$ . With each such point we can identify a unique angle  $\sigma \in (-\pi/2, \pi/2)$ , defined by

$$\sigma = \tan^{-1}\left(z_r'/x_r'\right).$$

The coordinates of the points on the semi-circle are given by

$$\begin{bmatrix} x_r' \\ y_r' \\ z_r' \end{bmatrix} = \begin{bmatrix} \cos \sigma \\ 0 \\ \sin \sigma \end{bmatrix}.$$

Let u be the vector

$$\begin{bmatrix} \cos \sigma + x \sin \sigma \\ y \sin \sigma \end{bmatrix}$$

It is easily verified that for finite  $x_r$ 

$$\frac{\mathbf{u}}{|\mathbf{u}|} = \begin{cases} \operatorname{sgn} \dot{\theta} \frac{\mathbf{x} - \mathbf{x}_r}{|\mathbf{x} - \mathbf{x}_r|}, & \text{for } |\mathbf{x}_r| < \infty; \\ \begin{bmatrix} 1 \\ 0 \end{bmatrix} & \text{for } |\mathbf{x}_r| = \infty. \end{cases}$$

Thus we can write

$$m_f = -\mu \int_R \mathbf{x} \cdot \frac{\mathbf{u}}{|\mathbf{u}|} p(\mathbf{x}) dA,$$

which is equally valid for finite and infinite rotation centers. We differentiate the above expression, and after simplification we obtain

$$\frac{dm_f}{d\sigma} = -\mu \int_R \frac{y^2 \cos \sigma}{|\mathbf{u}|^3} p(\mathbf{x}) dA.$$

First, we note that a singularity occurs in the integrand when  $|\mathbf{u}| = 0$ , which occurs when  $\mathbf{x} = \mathbf{x}_r$ . This is an isolated singularity and cannot affect the value of the integral as long as  $p(\mathbf{x})$  is finite. Now, for the range of interest of  $\sigma$ , the integrand is always positive. If  $\int p(\mathbf{x})dA > 0$ , that is, if there is any normal support force, the integral is positive, which implies that  $m_f$  is strictly decreasing.

## Appendix II

We want to show that the function  $\mathbf{x}_r(\gamma)$  exists, and that  $\psi$ , the angle of  $\mathbf{x}_r$ , is strictly increasing in  $\gamma$ . Figure 22 shows the notational definitions. Note that the angle  $\psi$  is defined to be the angle of the ray of motion, less  $\pi/2$ . A clockwise rotation can be distinguished from a counterclockwise rotation about the same center by examining the angle  $\psi$ .

The result is obtained by constructing a function  $\gamma(\psi)$  and showing that it is never stationary. It is then a simple matter to invert this function to obtain  $\psi(\gamma)$ , and then compose with the function  $\mathbf{x}_r(\psi)$  to obtain  $\mathbf{x}_r(\gamma)$ .

First we will construct  $x_r(\psi)$ . Given a  $\psi$ , we can construct the ray of motion  $R_M$  and use the procedure in the proof of theorem 2 to obtain a magnitude  $x_r(\psi)$ .

Fig. 22. Notational conventions for Appendix II.

The rotation center is defined by its polar coordinates  $\mathbf{x}_r(\psi) = (x_r(\psi), \psi)$ . To show that  $\mathbf{x}_r(\psi)$  is continuous is a fairly straightforward application of the implicit function theorem.

Now we will construct  $\gamma(\psi)$ . Equation (5) gives the frictional force  $\mathbf{f}_c$  as a continuous function of the rotation center  $\mathbf{x}_r$ . Since the contact force cannot be zero, the vector angle  $\gamma$  is also continuous. By composition we obtain a continuous function  $\gamma(\psi)$ . This is the function we wish to invert, but first we have to show that this function is one-to-one.

To begin we will show that  $\gamma(\psi)$  is never stationary. Construct  $L_T$  tangent to  $\mathbf{x}_r(\psi)$  (Fig. 23). There are two cases:

- 1.  $R_F$  intersects  $L_T$ . Let A be the intersection of  $L_T$  and  $R_F$ . We can adapt the analysis of Appendix I to show that the moment at A is not stationary with respect to  $\psi$ . Now, the ray of force passes through A, so the moment at A is zero, and will be stationary at zero if the ray of force is stationary. We conclude that the angle of force  $\gamma$  cannot be stationary.
- 2.  $R_F \parallel L_T$ . Let A represent the point of tangency, and let B be any other point on  $L_T$ . Let s give the distance between  $R_F$  and  $L_T$ . The moments  $m_A$  and  $m_B$  will be the same  $-s \times |\mathbf{f}_c|$ . As the rotation center passes through A, the moment  $m_A$  is instantaneously stationary. The analysis of Appendix I shows that  $m_B$  is not

Fig. 23.  $\gamma(\psi)$  is never stationary. Case (i) is for a ray of force intersecting the tangent line. If the angle of force  $\gamma$  were stationary, then the moment at point A would also be stationary. The analysis of Appendix I can be applied to show that this moment is not stationary. Case (ii) is for a ray of force parallel to the tangent line. Two

points A and B are constructed on the tangent line, with A at the point of tangency. If the angle of force y were stationary, then the moments at A and B would have identical rates of change. But the moment at A is stationary, whereas the moment at B must be changing, by Appendix I.

stationary. A stationary angle of force would imply that  $dm_A = dm_B$ , which is not possible.

We conclude that  $\gamma(\psi)$  is not stationary.

If  $\gamma$  and  $\psi$  ranged over real intervals, we could immediately conclude that  $\gamma(\psi)$  is one-to-one and form the inverse function  $\psi(\gamma)$ . But  $\gamma$  and  $\psi$  are angles, and their ranges have the topology of a circle. We have to be certain that  $\gamma$  does not "lap"  $\psi$ —that is, as  $\psi$  traces the circle once, we must be sure that  $\gamma$  traces the circle only once.

This is demonstrated by an energy argument. Frictional sliding requires work, which is supplied by the pushing contact. This implies that  $\mathbf{f}_c \cdot \mathbf{v}_c > 0$ , which means that  $\gamma \in (\psi, \psi + \pi)$  at all times. The rest of the proof is relatively straightforward.  $\gamma(\psi)$  is, in fact, one-to-one and monotonically increasing. By inverting, we obtain the monotonically increasing function  $\psi(\gamma)$ , and composition with the function  $\mathbf{x}_r(\psi)$  gives us the plot of rotation centers  $\mathbf{x}_r(\gamma)$ .

#### REFERENCES

Albus, J. S., and Evans, J. M. 1976. Robot systems. Scientific American 234(2):76–87.

Amontons, G. 1699. De la résistance causée dans les machines. Paris: *Mémoires de l'Académie royale des Sciences*. pp. 206-227.

- Bowden, F. P., and Tabor, D. 1950. Friction and lubrication of solids. Oxford, England: Clarendon Press.
- Coulomb, C. A. 1781. Théorie des machines simples en ayant égard au frottement de leurs parties et à la roideur des cordages. Paris: Mémoires de mathématique et de physique présentés à l'Académie royale des Sciences.
- Cutkosky, M. R. 1985. Grasping and fine manipulation for automated manufacturing. Ph.D thesis, Carnegie-Mellon University.
- Den Hartog, J. P. 1956. Mechanical vibrations. New York: McGraw Hill.
- Erdmann, M. A. 1984. On motion planning with uncertainty. Number 810. Massachusetts Institute of Technology Artificial Intelligence Laboratory.
- Gillmor, C. S. 1971. Coulomb and the evolution of physics and engineering in eighteenth-century France. Princeton, N.J.: Princeton University Press.
- Hanafusa, H., and Asada, H. 1977. Stable prehension by a robot hand with elastic fingers. *Trans. Soc. Instrumentation Contr. Engineers* 13(4):361-368.
- Hogan, N. 1985. Impedance control: an approach to manipulation. ASME J. Dyn. Sys. Meas. Contrl 107(1):1-24.
- Inoue, H. 1971. Computer controlled bilateral manipulator. *Bull. JSME* 14(69):199–207.
- Jellett, J. H. 1872. A treatise on the theory of friction. London: MacMillan. 681–689.
- Lozano-Perez, T. 1976. *The design of a mechanical assembly system*. Number 397. Massachusetts Institute of Technology Artificial Intelligence Laboratory.
- Lozano-Perez, T. 1981. Automatic planning of manipulator transfer movements. *IEEE Trans. Sys., Man, Cyber.* SMC-11(10):681-689.
- Lozano-Perez, T., Mason, M. T., and Taylor, R. H. 1984. Automatic synthesis of fine-motion strategies for robots. *Int. J. Robotics Res.* 3(1):3-24.
- Lozano-Perez, T., and Wesley, M. A. 1979. An algorithm for planning collision-free paths among polyhedral obstacles. *Communications of the ACM* 22(10):560-570.
- MacMillan, W. D. 1936. *Dynamics of rigid bodies*, New York: McGraw-Hill.
- Mason, M. T. 1981. Compliance and force control for computer controlled manipulators. *IEEE Trans. Sys., Man, Cyber.* SMC-11(6):418-432.
- Mason, M. T. 1982. Manipulator grasping and pushing operations. Number 690. Massachusetts Institute of Technology Artificial Intelligence Laboratory.
- Mason, M. T. 1984. Automatic planning of fine motions:

- Correctness and completeness. *Proc. Int. Conf. Robotics*. Atlanta, Ga.: IEEE Computer Society.
- Mason, M. T. 1985 (St. Louis). The mechanics of manipulation. Proc. IEEE 1985 Int. Conf. Robotics and Automation.
- McCallion, H., and Wong, P. C. 1975. Some thoughts on the automatic assembly of a peg and a hole. *Industr. Robot* 2(4):141–146.
- Moseley, H. 1835. On the equilibrium of the arch. Trans. Cambridge Philosophical Soc. V:293-314.
- Nevins, J. L., and Whitney, D. E. 1974. The force vector assembler concept. *1st IFToMM Symp. Theory and Practice of Robots and Manipulators*.
- Ohwovoriole, M. S., and Roth, B. 1981. A theory of parts mating for assembly automation. *Proc. Ro.Man.Sy-81*.
- Paul, R. P. 1972. Modelling, trajectory calculation and servoing of a computer controlled arm. Tech. Report AIM-177. Stanford, Calif.: Stanford Artificial Intelligence Laboratory.
- Paul, R. P., and Shimano, B. 1976. Compliance and control. Proc. Joint Automatic Contr. Conf.
- Pingle, K., Paul, R., and Bolles, R. 1974. *Programmable assembly, three short examples* (film).
- Prescott, J. 1923. Mechanics of particles and rigid bodies. London: Longmans, Green, and Co.
- Raibert, M. H., and Craig, J. J. 1981. Hybrid position/force control of manipulators. J. Dyn. Sys., Meas., and Contrl. 102:126-133.
- Salisbury, J. K. 1980. Albuquerque. Active stiffness control of a manipulator in cartesian coordinates. *IEEE Decision* and Control Conf.
- Salisbury, J. K. 1982. Kinematic and force analysis of articulated hands. Ph.D thesis, Stanford University, Department of Mechanical Engineering.
- Simunovic, S. N. 1975. Force information in assembly processes. *Proc. 5th Int. Symp. Industr. Robots*.
- Takase, K., et al. 1974. The design of an articulated manipulator with torque control ability. *Proc. 4th Int. Symp. Industr. Robots.*
- Truesdell, C. 1968. Essays in the history of mechanics. New York: Springer-Verlag.
- Udupa, S. 1977. Collision detection and avoidance in computer controlled manipulators. Ph.D thesis. Pasadena, Calif.: California Institute of Technology Department of Electrical Engineering.
- Whitney, D. E. 1982. Quasi-static assembly of compliantly supported rigid parts. J. Dyn. Sys. Meas. Contrl. 104:65–77.