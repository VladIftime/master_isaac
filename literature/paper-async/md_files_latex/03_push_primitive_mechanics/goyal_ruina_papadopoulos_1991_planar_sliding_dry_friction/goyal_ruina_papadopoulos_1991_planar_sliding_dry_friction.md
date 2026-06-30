# Planar sliding with dry friction Part 1. Limit surface and moment function

# **Suresh GoyaI\*"** and **Andy Ruinab**

*"Mechanical Engineering and Computer Science, bTheoretical and Applied Mechanics, Cornell University, Ithaca, NY 14853 (U.S.A.)* 

#### **Jim Papadopoulos**

*Mechanical Engineering, Northern Illinois University, DeKalb, IL 60115 (U.S.A.)* 

(Received March 27, 1990; accepted August 15, 1990)

#### **Abstract**

We present two geometric descriptions of the net frictional force and moment between a rigid body and a planar surface on which it slides. The limit surface, from classical plasticity theory, is the surface in load space which bounds the set of all possible frictional forces and moments that can be sustained by the frictional interface. Zhukovskii's moment function (N. E. Zhukovskii, *Collected Works,* Vol. 1, Gostekhizdat, Moscow, 1948, pp. 339-354) is the net frictional moment about the body's instantaneous center of rotation as a function of its location. Both of these descriptions implicitly contain the full relation between slip motion and frictional load. While Zhukovskii's moment function applies only to ordinary isotropic Coulomb friction, the limit surface applies to a wider class of friction laws that includes, for example, contact mediated by massless rigid wheels. Both the limit surface and the moment function can be used to deduce results concerning the motion of sliding rigid bodies.

#### **1. Introduction**

In order to plan effectively and control the motion of an object in contact with the ground and/or another object, it is helpful to understand the nature of the frictional- contact forces between them. However, much about the mechanics of frictional contact is still poorly understood. The micro-mechanics responsible for the energy dissipation in slip include a variety of mechanisms (e.g. adhesion, plastic deformation, fracture) that conspire in a complex way to cause what we describe macroscopically as friction, e.g. in ref. 1. Even if one is not interested in the micro-mechanical causes, appropriate empirical macroscopic descriptions for friction are not settled. Descriptions with dependences of friction on normal force or stress, on normal separation distance, on slip displacement, on slip velocity, on time of stationary contact, on slip history, and on self-induced vibrations are reviewed by Oden and

<sup>\*</sup>Present address: AT&T Bell Laboratories, Murray Hill, NJ 07974, U.S.A.

Martins [ 2 I and somewhat less extensively by Ruina [ 31. Finally, after a possibly appropriate friction law has been chosen, there still remain issues pertaining to the behavior of sliding objects.

A restricted set of mechanics problems relates to the slip of a single rigid body with Coulomb-Amontons-DaVinci friction, the most commonly applied dry-friction law. In the context of robotic motion planning, rigid body motion with such simple friction has been studied by Erdmann [4], Mason and Salisbury [ 51, and Peshkin and Sanderson [ 6-81. Erdmann's work [4] emphasizes cases where normal contact forces are coupled to the motion. The others have looked at the quasi-static motion of an object being pushed by a controlled finger or edge on a surface with possibly indeterminate normal contact. That these problems are mechanically interesting follows from the rigid body being finite in spatial extent. The various (or continuum of) load bearing points of contact, each of which contributes to the net frictional force and moment, can at any instant have different slip velocities and corresponding friction forces.

Towards the eventual goal of better understanding the motion of a rigid body in frictional contact with its supporting plane, we study here the nature of the resultant of the frictional contact forces when the support forces are given. Specifically we consider the sum of the frictional forces and moments for the following problem:

- (1) A rigid body slides on a planar surface. Only the interactions of this body with the surface are studied and not the interactions with any other pushing or restraining objects.
- (2) The contact normal force (or pressure) distribution is assumed to be known.
- (3) At each point of contact, the frictional force depends only on the (known) normal contact force and on the slider's orientation and direction of slipping (relative to its support plane), not on the magnitude of the slipping rate, the net slip displacement or the slip history.
- (4) At each point of contact the dependence of the friction force on direction is consistent with a maximum-power inequality (to be discussed). The maximum-power inequality generalizes isotropic Coulomb friction to include, for example, anisotropic contact mediated by a massless wheel, a wheel with bearing friction, a wheel with a ratchet, etc.
  - (5) Friction dissipation is positive.

Assuming (l)-(5) above, the force and moment required to overcome friction for a given motion can usually be found by simply adding the frictional forces at each of the contact points. The inverse problem of finding the motion associated with a given total force and moment is somewhat more difficult. But this problem can be solved geometrically by use of a limit surface exactly of the type used in classical plasticity [ 91. The limit surface fully characterizes the friction of a slider of the type we consider. With the additional restriction of isotropic friction at every point, friction for a rigid body is also fully characterized by the moment function of Zhukovskii [ 101. Our central goal here is to introduce these geometric descriptions of the

net frictional force and torque on a rigid body. A second paper [ 1 l] deals with the dynamics of free slip.

The plan of the remainder of this paper is first to review briefly the relations between friction and plasticity. We then describe the motion of, and the net frictional load on, a planar rigid body. Next, the nature of friction at a single point is discussed. The maximum-power inequality is introduced with which a convex limit curve describes the forces arising during slip of a point of contact. The limit curves for Coulomb friction (a circle), for an ideal wheel (a straight line), and for some less ideal wheels are given as examples. The load-motion inequality for the overall body is then derived and the resulting concept of a limit surface is introduced and illustrated with two somewhat artificial examples (a body with two points of Coulombic support, and a ring of ratcheted wheels). The moment function is then presented. We conclude with a discussion of some facts and results related to limit surfaces and to the moment function.

A condensed form of this paper was presented in ref. 12. More details and references are presented in ref. 13.

#### 2. **Friction and plasticity**

Rate-independent friction is related to plasticity in several ways. Friction is variously used as (a) an analogy for, (b) an example of, (c) a counterexample to, and (d) something to be explained by, plasticity; see ref.14. (a) Frictional slip is analogous to plastic deformation in that both involve (more or less) rate-independent forces. The analogy can be made more precise when particular friction and plasticity laws are considered. (b) If one thinks of slip as an extremely localized version of continuum deformation, then "frictional slip" is an example of inelastic (plastic) deformation. For the case of isotropic friction, and a finite number of support points between the slider and the supporting surface, our problem is precisely equivalent to the failure of a frictionless (or loose) lap joint between plates held by rigidplastic rivets or bolts, a problem discussed previously, e.g. by McGuire [ 15 1. Similarly (as pointed out by an anonymous reviewer) there is a correspondence between the problems analyzed here (those that use isotropic friction) with torsion and shear of a very short prismatic bar (or bars) made of ideally plastic material. Further, the normality principles we use throughout are directly quoted from classical plasticity theory, so for our purposes friction is an example of plasticity. (c) On the other hand, if the normal load is included as a variable, common friction laws violate the maximum plastic work postulate which is often assumed in plasticity, so that friction is a counter-example to most descriptions of plasticity in this regard. However, by allowing a "non-associated" flow rule, Curnier [ 161 is able to describe friction, with the normal degree of freedom included, in a formalism like that used for classical plasticity (our paper seems to depend fully on the

maximum-work postulate and thus cannot use Curnier's non-associated flow rules). (d) It is commonly held (e.g. ref. 1) that plastic deformation of the adjoining solids during slip is part of the micro-mechanics of frictional dissipation.

#### 3. **Motion of a rigid slider**

Since we shall be using rate-independent friction laws and always assume that the normal forces are known a priori, we can determine the friction forces at contact points from the directions of the velocity field u at these points. These are determined by the instantaneous motion for the rigid planar slider of Fig. 1, as described by one of the following.

- (1) The unit motion vector q, or "versor" in the language of Zmitrowicz [ 171, parallel to the motion vector Q. (Note: we use the word vector to describe any list of two or three scalars. We use the geometry of the Cartesian space they define when performing vector operations.) That is q = [qz, *qv,*  q,l = *Q/JQI, w h* ere the motion vector Q = *[V,, Vu, w]* has components which are the translation velocity of a reference point 0 on the slider, and the angular velocity of the slider. To simplify notation, we assume that the components of Q are non-dimensionalized by a characteristic length (say the radius of gyration of the rigid body) and some characteristic time. For definiteness the x and y axis at 0 may be considered fixed to the slider (both here and below).
- (2) The instantaneous motion of the slider may be defined by the location r, of the center of rotation C about which the body's motion is instantaneously a pure rotation. The motion is then described by the location of a point on one of two planes, one for clockwise rotation, and one for counterclockwise rotation, or by a point on the circle at infinity for pure translations. For

**Fig. 1. Rigid slider on a planar surface. The instantaneous center of rotation C is marked, as are the reference point 0 and the coordinate system used for describing motion and loads (both, for accuracy, considered fixed to the slider).** 

consistency, coordinate values for rc should be expressed in terms of the same characteristic length as used for the non-dimensionalization of 9.

The above two definitions are simply related since the location of the center of rotation is given by rc = WX *V/l'/lwj2* with coordinates (z;,, y,] = [ -ql,/q,, qz/q,] (W should be interpreted either as a scalar or as a vector orthogonal to the x-g plane, as appropriate). One may note, as is verified by use of similar triangles, that q and C are related by the projection shown in Fig. 2. The unit motion vector q is on the unit sphere, and the center of rotation is the intersection of the extension of q with a plane tangent to the top or bottom of the sphere (with its origin 0 at the point of tangency).

# 4. The **net frictional load**

The net frictional load is defined as P= [F,, *F,, M]* where *F,* and *F, are* the net forces that the planar slider exerts on the support surface and M is the net moment about a z axis passing through 0 (P is the negative of that which would represent the frictional load on a free-body diagram of the slider). When inertia is negligible, P is the total external load which must be applied to the slider to overcome frictional resistance. The components of P are assumed to have been non-dimensionalized by the same length used to non-dimensionalize the velocity, and by any convenient force, say the weight of the slider.

P can be expressed by integrals over the entire contact region *A,* of the frictional traction (stress)f= [f,,f,] that the slider causes on the support

Fig. 2. The relation between the unit motion vector q and the center of rotation C is described by the projection shown from the unit motion sphere to the tangent planes at top (for clockwise rotation) or bottom (for counterclockwise rotation). Note the orientations of the axes.

plane at each contact point a with position coordinates [x~, ~~1 relative to 0:

$$F_x = \int_A f_{ax} dA \qquad F_y = \int_A f_{ay} dA \qquad M = \int_A (x_a f_{ay} - y_a f_{ax}) dA \qquad (1)$$

We will also make use of the moment M, of the frictional forces about a vertical axis passing through the center of rotation C at *rc = [x,, y,],* given by:

$$M_{c} = \int_{A} \{ (x_{a} - x_{c}) f_{ay} - (y_{a} - y_{c}) f_{ax} \} dA = M + x_{c} F_{y} - y_{c} F_{x}$$
 (2)

where it should be recognized that force terms in eqns. (1) and (2) depend on the motion. If point supports are involved, the integrands in eqns. (1) and (2) above contain delta functions, or equivalently, are replaced by discrete sums. The sums (1) and (2) and their relation to the motion q (or C) are the central subject of this presentation. We use discrete sums or integrals as is convenient or appropriate.

### **5. Isotropic friction**

The simplest friction law to which this paper applies, isotropic friction, is also the most often used (by far). During slip the friction force (or traction) at a point is in the direction of motion, and its magnitude If] is a constant independent of the direction of motion. During stick the magnitude of the friction force is less than or equal to this constant. Our central interest is in this product of the friction coefficient and the normal force rather than either the normal force or the friction coefficient separately.

An equivalent, and at first sight somewhat awkward, description of this same friction law is the following pair of statements:

- (1) There is a circle centered at the origin of *[f,, f,]* space. We call this circle the limit curve (LC) for isotropic friction at the given point of contact.
- **(2)** The maximum-power inequality, equivalent to the classical principle of maximum plastic work, is always satisfied:

$$(f - f^*)v \geqslant 0 \tag{3}$$

where f and u are the actual friction force and relative slip velocity at the point of contact,f andf\* are on or inside the limit curve, andf\* is otherwise arbitrary. The situation is shown in Fig. 3 for contact at a point a on the sliding body. The usefulness of the maximum-power inequality (as well as its name) comes from its validity for **all f\*** on or inside the limit curve. For a given u (possibly 0) the co~espond~g f is that f, on or inside the limit curve, which maximizes the power over all possible f\*, since fu &J\*u for all *f \*.* Note that *f* is not unique if u =0, and that applying inequality (3) to

**Fig. 3. (a) Coulomb friction at a point with position r,. The friction force isf, and the slip velocity** *u,f,\* is any* **friction force on or inside the LC, which is a circle (b) for isotropic Coulomb friction.** 

a circular LC with a variety of possible *f \*s* restricts *f* to be parallel to u (when u z 0).

The maximum-power inequality is a description of material behavior. The reader is cautioned against trying to relate it to common notions of minimum potential energy or positive entropy production. For another interpretation see ref. 18.

#### 6. **Anisotropic friction satisfying normality**

The tool which we use in establishing the properties of the frictional limit surface (to be defined below) is the maximum-power inequality (3) for each sliding contact point; the isotropy of friction is not essential. So it is safe, and perhaps useful for some applications, to allow any friction law that is described by the statements:

- (1) A closed curve in force space is specified.
- (2) The maximum-power inequality (3) is satisfied by all *f* and u. (As discussed further below, one should not blindly assume that all rate-independent friction laws actually satisfy inequality (3).)

The reasoning we present based on statements (1) and (2) above is borrowed directly from classical plasticity theory, e.g. ref. 9, and is partially repeated here for readers not familiar with that subject. Statement (2) above requires that the specified closed curve of (1) be convex, and positive dissipation requires that it enclose the origin. A possible (if artificial-looking) limit curve is shown in Fig. 4 where the existence of a flat region and a vertex should be noted.

Consequences of the maximum-power inequality and the limit curve description are: (a) if *f* is inside the limit curve then u = 0; (b) u is perpendicular to the limit curve in places where the curve has a well-defined normal, thus

Fig. 4. Generic LC. Friction forces that can occur during sliding for an imagined anisotropic material. The curve is convex and encloses the origin. Where it is smooth, velocity is normal to the curve. Where it is flat, a range of friction forces is possible for the given velocity. Where it is kinked (at the vertex), a range of velocity directions is possible with the given friction force.

the friction law is said to "satisfy normality"; (c) the slip direction vllvl is non-unique if a given friction force is at a vertex on the limit curve (all normals to an imagined rounded vertex are possible); and (d) the friction force is non-unique if a given slip velocity v is normal to a flat region on the limit curve (all forces on the flat region are possible). These facts follow from simple geometric reasoning and use of inequality (3). For example, (b) above follows from demanding the truth of inequality (3) for two values off *\** on the limit curve, arbitrarily close to *f,* one on each side off. All of the properties (a)-(d) are worth noting: analogous corners and flat regions may appear on the limit surface for a slider even when the LC at every point of contact is a circle.

# *6.1. Wheels as anisotropic friction*

Contact mediated by a wheel provides a useful and consistent example of anisotropic friction satisfying statements (1) and (2) [ 19-21 I. One may want to model an object, say a car or a cart, using anisotropic contact friction, rather than adding the motion of the wheel as an additional degree of freedom.

# *6.2. Ideal wheel or skate*

Figure 5(a) shows a microscopic, massless, rigid wheel attached by a frictionless bearing to the slider and with ordinary isotropic friction at its contact with the ground. We consider the wheel as a micromechanism for the transmission of force from the rigid slider to the support plane. A simple

Fii. 5. (a) Microscopic wheel embedded in the slider. It is massless and makes ordinary frictional contact with the ground. The set of forces it can transmit to the ground, along with the motion of the slider to which they correspond is shown by (b) the LC for contact mediated by an ideal wheel with no bearing friction. (c) The LC for a wheel with bearing friction. (d) The LC for a wheel with a ratcheted axle.

model of an ice skate has the same idealization. The LC for the set of possible transmitted forces is highly degenerate, as seen in Fig. 5(b). It encloses no area and has nothing but flat regions and vertices. During rolling

the side force is indeterminate (anything on the limit curve), and during side slip (where the force is unique) a whole range of velocity directions is possible.

If bearing friction at the axle of the wheel exerts less torque about the axle than can be overcome by the frictional strength at the base of the wheel, then the LC is given by the convex curve shown in Fig. 5(c) [ 19, 2 11. The straight segments on the LC correspond to pure forward and backward rolling of the wheel and the circular arcs correspond to skidding. If the bearing friction is sufficiently large, the wheel locks and the limit curve reduces to a circle.

# 6.4. Wheel *with ratchet*

If the axle has a ratchet so that the wheel is allowed to roll freely in only one direction and it locks in the other direction, then the LC is the closed semicircle of Fig. 5(d). The straight segment passing through the origin corresponds to rolling of the wheel in the preferred direction, and the semicircle corresponds to skidding when the wheel is locked by the ratchet.

#### 6.5. **Wheel** *of wheels, etc.*

*An* ideal wheel with contact at its rim mediated by a continuum of orthogonal ideal wheels, equivalent to a frictionless ball bearing, does not resist motion in any direction and hence has the most degenerate possible limit curve, a point. If either the large wheel or all of the small wheels at the rim are supplied with ratchets the limit curve is half of the line segment of Fig. 5(b). Other combinations of wheels of wheels with and without ratchets and bearing friction lead to a variety of limit curves. Ilon's [22] clever "Ilonator" wheel, which has been used as a drive wheel in some robotics experiments, is a wheel of wheels.

#### 6.6. *Other normal friction laws*

Ziemba [ 231 postulates a law of anisotropic friction that obeys the maximum-power inequality and has an elliptical LC, Moszynski [19] also presents some other anisotropic friction laws that obey the maximum-power inequality (3).

# 7. **Non-normal friction laws**

Our fo~alism requires the strong restriction that friction laws sat&s& the maximum-power inequality (3) (normality). Since it is easy to imagine friction laws which violate inequality (3), we pause to consider these laws

for which the remaining portions of this paper do not apply. Anaive construction of anisotropic friction has the friction force always parallel to the slip direction but with magnitude dependent on direction. Figure 6(a) shows the limit curve and corresponding slip velocities. It is apparent that the velocities are not normal to the limit curve so this friction law violates inequality (3). We do not know of any experiments which would support such a law.

Ziegler [24 ] seems to argue that violation of normality (as we use the term) is thermodynamically impermissible. In support of Ziegler, the rateindependent friction laws which we have been able to generate are all consistent, at some level, with normality. But the following examples, which use explicitly described micromechanisms, should demonstrate that the situation is somewhat subtle. In particular, when there is relaxation in a micromechanism, normality may not hold for the "relaxed limit curve" associated with the set of the micromechanism's relaxed states. This situation is reminiscent of that seen in the critical state theory of soil mechanics, e.g. ref. 25.

Fig. 6. (a) The friction force for an imagined friction law having friction force parallel to slip direction but with magnitude dependent on direction. This violates the maximum-power inequality (3). (b) The LC for an imagined isotropic friction law that violates inequality (3). (c) A crooked castor with bearing friction that causes the non-normal limit curve in Fig. 6(b) when only its relaxed states are considered. (d) The instantaneous limit curve for the wheel in the castor above superimposed on the castor linkage. The particular velocity shown corresponds to the relaxed state where the wheel is aligned with, and moves in the direction of, IJ.

### *7.1. Crooked castor with frictional axle*

At some level of macroscopic phenomenology, the mechanism described below generates a non-normal version of Coulomb friction where the "relaxed" friction force has a magnitude independent of slip direction, but is not parallel to the slip. Figure 6(b) shows the resulting relaxed circular LC and nonnormal slip velocity.

Contact between a slider and the supporting surface is mediated by the small (massless and perfectly rigid) castor wheel shown in Fig. 6(c). The wheel is connected by means of a massless rigid structure (in a plane parallel to the slider), "L" shaped in our illustration. One end of the "L" pivots freely at its attachment to the slider, and the wheel is mounted at the other end. The wheel, which rolls and slides on the support plane, has some bearing friction. The instantaneous force-velocity relation for the castor wheel itself obeys convexity and normality (Figs. 5(c), 6(d)). In plan view, the "L" is a two-force member so the transmitted force is always along the line connecting the wheel center to the pivoting attachment. Thus, the instantaneous force-velocity relation for the whole mechanism (as sensed at the attachment to the slider) is similar to that of an ideal wheel (Fig. 5(b)) whose axle is parallel to the line connecting the pivot to the wheel's center.

The instantaneous force-velocity relation for the mechanism satisfies the maximum-power inequality (31, but for any given slip velocity of the castor pivot point (i.e. slider slip velocity), the castor swings in, and after a few castor lengths, relaxes to a state where the wheel is aligned with u. The castor wheel drags on the relatively moving support plane much as a light kite flies in the wind, with the string force and relative wind velocity not parallel. One may now think of the castor arm as being negligibly small (in much the same way as one may agree to neglect the mass of a wheel or the transients associated with the deformation of a rubber wheel's contact patch, etc.). So the castor's reorientation may be regarded as ~st~taneous and one can consider the castor as always equihbrated to its steady state orientation, shown in Fig. 6(d). Considering only these relaxed steady state conditions, the force-velocity relation does not obey normality. The relaxed limit curve and related velocities are shown in Fig. 6(b).

Another non-normal (in the set of relaxed confi~rations) friction relation comes from an ordinary in-line castor (axle of the wheel is perpendicular to the castor arm), weakly supported so the top of the wheel rubs on the slider while the bottom rolls on the support plane. Assume the friction between the wheel and the support plane is high and that between the wheel and slider is low but dependent on the point on the slider where the wheel rubs. Instantaneously this device mediates a friction law which is equivalent to a point of isotropic frictional contact where the wheel rubs the slider and hence satisfies normality. But if relaxation transients are again neglected, its behavior leads to a law of the type described above as na'ive anisotropy (Fig. 6(a)).

# *7.3. Other non-normal friction laws*

The microscopic friction model of Michalowski and Mroz [ 141, with anisotropy due to wedge asperities, also does not satisfy the maximum-power inequality, nor does the orthotropic friction tensor of Zmitrowicz [ 17 1. Since our results explicitly depend on the maximum-power inequality (3) they do not apply to these non-normal friction laws, whether or not they are usefully accurate descriptions.

#### **8. Maximum power and the limit surface**

The formalism used to describe friction at a point carries over directly to the rigid body as a whole. The three-dimensional friction load P and the motion versor g also satisfy a maximum-power inequality with respect to a closed, convex, limit surface. We term this inequality the "load-motion" inequality to distinguish it from the maximum-power inequality for friction at a point.

The formal construction of the overall limit surface in load space from the limit curves follows from inequality (3) applied to every point with the principle of virtual power. The principle of virtual power (often called the principle of virtual work) for our rigid body may be expressed as:

$$PQ = \sum fv \tag{4}$$

where P = [F,, *F,, M]* andf= *[f,, f,] are* any load vector and force distribution related by the 'equilibrium' sums (1) (whether or not the fs are properly associated with any friction laws). Q= *[V,, V,, 01* and u (slip velocity at position r) are any motion vector and velocity distribution related by the rigid body "compatibility" equation:

$$v = [v_x, v_y] = V - \omega \times r = [V_x - \omega r_y, V_y + \omega r_x]$$

Assume thatf andf\* are two frictional force distributions that do not violate the slip condition anywhere (i.e. f and f\* are on or inside their LCs) and they correspond through eqn. (1) respectively to P and P\*. Further assume thatf at every point is also consistent with the friction law and the velocity arising from the motion Q. Applying the principle of virtual power (4) to each of the two cases (PQ =Z\_fu and I'\*& = I&+"%), subtracting, and then applying the inequality (3) to each term, gives the load-motion inequality:

$$(\mathbf{P} - \mathbf{P}^*) \mathbf{q} \geqslant 0. \tag{5}$$

Here, P is the load vector at 0 during slip associated with a motion vector 4 = &/lQj, and P\* is any other load vector at 0 that results from summing possible **f\*s** (where "possible" refers to that which is on or inside the limit curves for each contact point, with no attention paid to the velocities at those points).

The set of allP\* is the set of all possible sums of the possible contributions from each of the contact points. The boundary of this set is the limit surface. It is a closed convex surface in load space that encloses the origin. The limit surface for the object, with inequality (5), fully describes the relation between friction loads *P* and motions g. So the net friction load exerted by a rigid planar slider, subject to the assumptions named previously, is fully characterized by (i) a closed convex surface in load space that encloses the origin (which may be regarded either as a purely macroscopic description of the slider, or as a description constructed from the microscopic contact distribution); and (ii) the load-motion inequality (5), *(P-P\*)q a 0. As* with the limit curves: (a) if *P* is not on the limit surface then Q =O, (b) 4 is normal to the limit surface where it is smooth, (c) there is a non-uniqueness in the slip versor 4 for a given friction load *P* if it is at a vertex (both point and edge vertices) on the limit surface (all normals to an imagined rounded vertex are possible), and (d) there is a non-uniqueness in the friction load *P* for given motion 4 if g is normal to a flat region on the limit surface (all loads on the flat region are possible). Note that a convex developable surface, like the side of a cylinder, is flat in one direction.

# *8.1. Limit surface for a single contact point*

A slider's limit surface is easily constructed if the two-dimensional forcespace limit curves for each point of contact are replaced by appropriate limit surfaces in three-dimensional load space.

The limit surface for a single point of contact (labeled as contact i) at position *r* (relative to 0) bounds the set of all possible frictional loads generated by that point. The force fi on or inside the limit curve is associated with load *Pi* by eqns. (1) (no integration or summation necessary). So the set of all *fi* on or inside the LC corresponds to a set *Pi,* the point's limit surface.

These formulae may be interpreted geometrically as follows. Add a third axis perpendicular to the *fz-fu* plane to represent moments. The limit surface is found by projecting the limit curve (with origin 0) vertically onto a plane tilted about *r* with slope *Irl.* Standing at the origin one is looking up the tilt if *r* is directly to one's right. The projected curve and its in-the-tiltedplane interior make up the point's limit surface. It is singular in that it encloses no volume, and is everywhere either a flat region or a vertex.

For example a point of isotropic friction located at 0 yields a pennyshaped limit surface in the *F,-F,* plane centered at the origin: its friction force generates no moment. A point of isotropic friction at *r = [0,* l] gives rise to an elliptical disk, centered at the origin in *P* space with its major axis direction being [ 1, *0, -* 1 ] and minor axis direction [0, 1, *01.* This is because the z component of load, i.e. the moment of the frictional force, is zero when sliding in the y direction and maximum when sliding in the z direction. The projection of this limit surface onto the *F,-F,* plane is just the circular disk of the first example.

CSI *Once the* ~di~du~ con~butions are expressed in P space the sum (1) be written as P=I;p, and the set of a11 possible P is the set of all possible sums XP+ The limit surface, then, is the boundary of the Minkowsky sum of the individual limit surfaces.

Another way of generating the limit surface is by the convolution of the individual limit surfaces. An individual limit surface is chosen and, keeping it fixed, the origin of another individual limit surface (while keeping its o~en~tion fixed) is slid on the boundary of the fixed limit surface, and the envelope of the voIume thus swept out, obtained. Then successively the origins of the remaining individual limit surfaces are slid with fixed orientations over the envelope obtained from the previous convolution. The overall limit surface is then the outer boundary of the region swept out after all the convolutions.

# **9. Example limit surfaces**

## 9.1. *Sliding dumbell*

*Figure 7* shows a symmetric bar supported at its ends by two identical points of isotropic friction with strength d2. It has a dimensionless weight of unity and a half-length of unity. The individual limit surfaces for the two support points are elliptical Iaminae centered on the origin and tilted oppositely

Fig. 7. Sample rigid object: a bar supported at its ends. The net frictional load is shown in (a), the contact frictional forces in (b).

about the y axis. They both project verticahy to a circle of radius ,u/Z on the x-y plane. The limit surface for this object (Fig. 8(a)) is the boundary of the volume swept by the convolution of the two ellipses (the solid swept when the center of one ellipse is dragged over all points on and inside the other ellipse, their orientations remaining fixed). The intersection of the limit surface with the [F,?, F,] plane is a circle of radius p. The intersection of

**Fig. 8. (a) The limit surface for the bar of Fig. 7. The set of all possibie frictional loads during slip. (b) The section of the limit surface of (a) on the [F,, M] plane. Various non-uniquenesses are illustrated with respect to the flat sides and the sharp comers. An "eigen-direction" where**  *P is* **parallel to q is shown.** 

the limit surface with the [M, FU] plane is also a circle, and the intersection with the [M, F,] plane is a square as shown in Fig. 8(b). An eigen-direction, important for dynamics considerations [ 11, 131, where P is parallel to q, is indicated.

The limit surface has four point vertices where it intersects the F, and the M axes. It has four flat regions showing as elliptical facets whose normals are in the [M, F.] plane. These represent extreme normal displacements of the ellipse being dragged (during convolution), and they blend smoothly with the non-flat parts of the limit surface. Excepting the four point vertices, the limit surface is everywhere smooth, with a well-defined normal g.

The vertices on the limit surface correspond to non-uniqueness of the slip motion, in particular when either a pure force in the x direction or a pure moment is applied. If a force is applied in the x direction, for example, the object may translate or translate with a small amount of rotation. The flat elliptical facets correspond to a whole set of friction loads all of which have the same normal (unit motion vector q). In particular, they correspond to rotation about one of the support points. When the bar rotates about a support point, that point is not sliding and its friction load can be anything on or inside its individual limit surface, hence the non-uniqueness in the net frictional load P.

Excepting the flat facets, the limit surface of Fig. 8 can be parameterized by representing the frictional loads (F,, F,, M] as functions of the instantaneous center of rotation location [x0 y,], or the motion vector [q,, q,, q,] (with Qz2 + QV2 + 4," = 1). Specifically [ 13)

$$F_{x} = \frac{\mu(y_{c}+1)}{2\{x_{c}^{2}+(y_{c}+1)^{2}\}^{1/2}} + \frac{\mu(y_{c}-1)}{2\{x_{c}^{2}+(y_{c}-1)^{2}\}^{1/2}}$$

$$= \frac{\mu(q_{x}+q_{\omega})}{2\{1+2q_{x}q_{\omega}\}^{1/2}} + \frac{\mu(q_{x}-q_{\omega})}{2\{1-2q_{x}q_{\omega}\}^{1/2}}$$

$$F_{y} = -\frac{\mu x_{c}}{2\{x_{c}^{2}+(y_{c}+1)^{2}\}^{1/2}} - \frac{\mu x_{c}}{2\{x_{c}^{2}+(y_{c}-1)^{2}\}^{1/2}}$$

$$= \frac{\mu q_{y}}{2\{1+2q_{x}q_{\omega}\}^{1/2}} + \frac{\mu q_{y}}{2\{1-2q_{x}q_{\omega}\}^{1/2}}$$

$$M = \frac{\mu(y_{c}+1)}{2\{x_{c}^{2}+(y_{c}+1)^{2}\}^{1/2}} - \frac{\mu(y_{c}-1)}{2\{x_{c}^{2}+(y_{c}-1)^{2}\}^{1/2}}$$

$$= \frac{\mu(q_{x}+q_{\omega})}{2\{1+2q_{x}q_{\omega}\}^{1/2}} - \frac{\mu(q_{x}-q_{\omega})}{2\{1-2q_{x}q_{\omega}\}^{1/2}}$$
(6)

#### 9.2. Circle *of tangential ratcheted wheels*

We now consider the intentionally contrived slider formed of a rigid ring with a continuous distribution of wheels at its perimeter. Each of these wheels is tangential to the ring (their axles are radial) and is ratcheted so that the ring can rotate clockwise freely. The ring, with the LCs for some

Fig. 9. (a) A ring sliding via a continuum of ratcheted wheels which freely allow clockwise rotation of the ring. A counterclockwise rotation about C is shown. The wheels both roll and slide inside -+# and are locked (slide only) outside ~c#s. (b) The axisymmetric limit surface for the ring of 9(a) is shown by its generating curve. Note the flat bottom and the vertex at the lower right corner.

of its wheels, is shown in Fig. 9(a). Small arrows show the friction force for counter clockwise rotation of this object about point C. Wheels between \$J and - 4 both roll and skid, and hence develop a pure skidding force. Others are locked by their ratchets, and develop a force in the direction of slip. The limit surface for this ring is axisymmetric, since the object is axisymmetric, and can be represented by a cross-section, shown in Fig. 9(b). The details of the integration used to obtain this figure are given in ref. 13.

The flat bottom facet of this limit surface corresponds to zero net moment and the indeterminacy of the sideways reactions of the wheels when the ring is rotating clockwise about its center. The edge (vertex) at the bottom outside allows clockwise rotations about centers of rotation on a given radial line segment inside the ring, since they all lead to the same frictional load (a pure force perpendicular to the line segment). Pure translation engenders a moment about the center (because only some wheels are rolling).

# **10. Construction of limit surfaces**

Given the normal pressure distribution and the friction law everywhere on the contacting surfaces, different approaches may be used to obtain a slider's limit surface.

- (1) Direct summation (or integration) of the friction forces and their moments for each possible motion of the slider. As demonstrated earlier, this is equivalent to finding the outer boundary of the set of all possible sums of forces from the individual contact points without enforcing compatibility of the motions of the individual contact points.
- (2) More geometrically, one may generate the slider's limit surface as the outer envelope of the convolution of the limit surfaces of the individual points of contact. This method is particularly suitable for sliders with point supports and, as pointed out earlier, was used to generate the picture of the limit surface for the sliding dumbell shown in Pig. 8(a). The convolution is simplified by noting that for a given point P on the limit surface with normal 4 all of the contributing *Pis* on the individual limit surfaces are those with normal Q (i.e. those that are greatest in the q direction).
- (3) For isotropic friction, the limit surface can be found from evaluation and differentiation of the moment function as prescribed in eqn. (7) below. The eqns. (6) parameterizing the limit surface for the sliding dumbell were obtained in this manner.

# *10.1. Contact distribution from a given limit surface*

We think that the inverse problem of finding a slider which generates a given convex limit surface is not solvable, in general. That is, if one restricts the friction law to any of the laws we have explicitly considered (e.g. isotropic friction, wheels), there exist convex, origin-enclosing surfaces that are not associated with any pressure distribution. In those cases where a solution does exist, we do not know of a simple algorithm for finding it.

On the other hand, we suspect that limit surfaces whose shape (though not position) is unchanged by reflection through the origin, i.e. P( -a) =P(q) + const. can be generated from a distribution of ratcheted wheels of wheels. For example, Goyal [ 131 generates a spherical limit surface with an axisymmetric pressure distribution of ideal wheels, each of which could be replaced by a pair of ratcheted wheels of wheels.

#### **11. Moment function for isotropic friction**

We have rediscovered another simple geometric representation of the friction load for the special case of isotropic friction, presented by Zhukovskii [lo]. For rotation about a point C with position [12;,, yJ the friction moment M, can be calculated by the sum in eqn. (2). The value of M, as a function of [LX,, yc] is the moment function. It is single-valued and well defined, since the only non-uniqueness in the sums (2) is due to the force of the nonsliding center of rotation which does not contribute to M,.

The moment function can be visualized as follows. For every point of support the moment function is a round cone that opens upwards, with its vertex at the point of supp.ort on the [x,, yJ plane (the plane of instantaneous centers of rotation), and its slope proportional to the friction force at that point (since M, is the friction force times distance from the point). The moment function is the vertical sum of these cones over all points of contact.

A straightforward calculation shows that the moment function contains all the information about the slider's contact. In particular, differentiation of eqns. (1) (using isotropic friction when evaluating f(q)) shows that

$$F_x = \frac{\partial M_c}{\partial y_c} \qquad F_y = -\frac{\partial M_c}{\partial x_c} \qquad M = M_c - r_c \times F$$
 (7)

where *rc = [x,, yc]* is the position of the center of rotation relative to the reference point 0 (say the center of mass or pressure) and the cross product is scalar-valued. The moment function for the bar with two isotropic points of support is shown in Fig. 10. Its gradient is not well defined for the two corners at the bottom, so the differentiation above is not sensible and the friction load is not well defined. This corresponds to the center of rotation coinciding with a support point. This non-uniqueness in *[F,, F,]* at vertex points on the moment function is the same as that represented by flat facets on the limit surface.

The moment function has a constant value and zero gradient for all points on the y, axis between the support points, so those centers of rotation correspond to motions with zero force and constant moment. Similarly, on the yc axis outside of the bar ikl, = *r, x F, M = 0* and again a set of centers of rotation is found with a fixed load. These straight lines on the moment function correspond to vertices on the corresponding limit surface.

**Fig. 10. The moment function for the bar of Fig. '7. The moment function is the vertical sum of two identical cones each centered at a contact point. The horizontal bottom is rounded, but the two corners (corresponding to cone apices) are sharp.** 

#### **12. Discussion**

**One may** identify certain points of possible interest on any sliding object. The center of mass (CM) of the planar object is the centroid of its mass distribution. The center of pressure (CP) is the centroid of the contact pressure distribution between the sliding object and the supporting surface. With all externally applied loads being in the plane of the slider (except gravity), and with either negligible acceleration or CM on the plane, then CP is the same as CM.

The center of twist (CT) is that center of rotation for which the generalized load is a pure moment. Zhukovskii [lo] calls the CT the pole of friction. The CT is at the minimum of the moment function (where its gradient and thus the net force is zero) and is not necessarily unique when there are points of finite support and all support is on a line. On the dumbell of Fig. 7, for example, all points on the bar are CTs. For isotropic friction, the CT lies within the convex hull of the support distribution [lo]. But this is not necessarily true for anisotropic friction. For example, if the symmetric bar of Fig. 7 is supported s~metri~ally at its ends with ideal wheels which roll along the bar direction, then all points in the strip-shaped region contained between the two parallel lines drawn at the ends of the bar (and perpendicular to it), are CTs.

The centroid of the frictional forces during pure translation is called the center of friction (CF) [5, 261. Assuming isotropic but possibly nonuniform CouIomb friction, the CF is unique, and it lies within the convex hull of the support dist~bution. No unique CF exists for the ring of ratcheted wheels of Fig. 9 - there is a different point for every translation direction.

For uniform isotropic friction and in-plane loads (for example all the pressure distributions considered by Zhukovskii [lo]), CP equals CF. Additionally, if the pressure distribution has certain symmetries, then all the above points (CM, CP, CT, CF) coincide.

# 12.2. *Properties of the limit surface*

Some general features of limit surfaces may be noted, mostly for sliders with isotropic friction; for further explanation see ref. 13.

- (1) With isotropic friction, flat regions appear on the limit surface if and only if it has points of finite support (infinite stress). Each point of isotropic support on the slider leads to two parallel elliptical facets on its limit surface as in, for example, Fig. 8. Since flat facets can make up a reasonable fraction of the possible frictional loads, the propensity of objects to rotate about points of finite support is in some sense partially explained (further explanation related to the dynamics of slip is discussed in ref. 11). With anisotropic friction, where the LCs may have vertices and flat segments, the limit surface may have facets even when contact is distributed, as with the ring of wheels in Fig. 9.
- (2) With isotropic friction, vertices appear on the limit surface if and only if all support points lie on one straight line segment. The consequent non-uniqueness in the motion corresponds to the friction load being the same for all rotations about an extension of one side of the segment, as depicted in Figs. 7 and 8. With anisotropic friction, vertices may appear even with non-collinear contact, as in Fig. 9.
- (3) Choosing a different reference point 0 distorts the limit surface by a simple vertical shear. By placing 0 at the CT, the limit surface is horizontal (has vertical normal) at the z (moment) axis.
- (4) For isotropic friction, the intersection of the limit surface with the x-y plane is a circle. If the reference point 0 is chosen at the CF the normal to the limit surface on this circle is horizontal, corresponding to translation (without rotation) in the direction of the friction force.
- (5) The limit surface of a slider shares the symmetry of its distribution of friction laws, assuming that the support plane is homogeneous and isotropic. An axisymmetric object has an axisymmetric limit surface. Shvedenko's claim [27] that all objects with isotropic friction have isotropic moment functions is incorrect (eqn. 1.6 in the proof of theorem 1 therein is wrong) as Fig. 10 here illustrates.

### 12.3. Maximum *load in a given direction*

Using the moment function Zhukovskii [ lo] proved the following theorems for isotropic friction: (1) the frictional force is maximized over all possible motions by pure translation, and (2) the net moment measured relative to the CT is maximized over all possible centers of rotation by rotation about the CT.

In fact, the load-motion inequality (5), with proper verbal ornamentation, is almost a direct statement of a more general result, valid for all the friction laws we consider. It follows directly from the load-motion inequality (5) that the component of generalized load in any direction *n* is maximized by motion in that direction. That is for given *n, Pn* is maximized over all *P*  on or inside the limit surface by setting g equal to *n.* For example, the moment relative to any point is maximized by rotation .about that point, since the maximum moment on the limit surface (drawn with this point as the origin) occurs where the limit surface normal is along the M axis, which is a direct consequence of the convexity and normality of the limit surface. Similarly, the component of force in any given direction is maximized by translation in that direction.

# 13. Conclusions

The limit surface from classical plasticity (and, for isotropic friction, Zhukovskii's moment function) provides compact geometric descriptions of the overall relation between frictional load and motion of a sliding rigid body whose slip is governed by a "normal" friction law. These surfaces, with their associated properties, provide a direct geometric answer to the question: what is the motion associated with a given frictional load? They can be used to visualize or explain many features of rigid body slip, for example when frictional load is or is not a single-valued function of the motion, and vice versa. Dynamics results using the limit surface are presented in Part 2 of this paper [ll].

### Acknowledgments

We thank Frank Moon for bringing us (S.G. and A.R.) together, John Hopcroft for encouragement, Michael Peshkin, Matt Mason and Bruce Donald for initial inspiration, A. Zmitrowicz for controversial comments on normality, Olga Peschansky for translating the Zhukovskii reference, J. J. Moreau for supportive comments, Laurie Miller, Win Nelson, Ross Levinsky and two anonymous reviewers for editorial comments and Eric Wagner for help with figures. One of us (S.G.) was supported by COMEPP, Emerson Electric, ONR grants No. NOOO14-86K-0281, N00014-88K-0591, NSF grant No. DMC-86- 17355, Sandia National Labs-75-6562, and also by A.R.'s NSF PYI award which also supported other aspects of this research.

#### References

- 1 D. Tabor, Friction-the present state of our understanding, J. *Lubr. Technd., 103* (1981) 169-179.
- 2 J. T. Oden and J. A. C. Martins, Models and computational methods for dynamic friction phenomena, FENOMECH III, Stuttgart, F.R.G., 1984.
- 3 A. L. Ruina, Constitutive relations for frictional slip. In Z. B&rant (ed.), *Mechanics of Geomaterials,* Chapter 9, Wiley, New York, 1985, pp. 169-188.
- 4 M. A. Erdmann, On *Motion Planning with Uncertainty,* Tech. Rep. 810 (Massachusetts Institute of Technology AI Laboratory, Cambridge) 1984.

- 5 M. T. Mason and J. K. Salisbury, *Robot Hands and The Mechanics of Man@.&timL,*  M~sachuset~ Institute of Technology Press, Cambridge, MA, 1985.
- 6 M. A. Peshkm and A. C. Sanderson, The motion of a pushed, sliding workpiece, IEEE *J. Robotics & Automation,* 4 (6) (1988) 596-598.
- 7 M. A. Peshkii and A. C. Sanderson, Planning robotic manipulation strategies for workpieces that slide, *IEEE J. Robotics & Automation,* 4 (5) (1988) 524-531.
- 8 M. A. Peshkin and A. C. Sanderson, Minimization of energy ln quasistatic manipulation, *IEEE J. Robotics & Automation, 5 (1) (1989) 53-60.*
- *9* W. Pragcr, *In&-oduction to Plasticity,* Addison-Wesley, 1959.
- 10 N. E. Zhukovskii, Equilibrium condition for a rigid body resting on a fixed plane with some area of contact, and capable of moving along the plane with friction. In *Co&x&d Works, Vo'ol. 1: General Mechanics,* Gostekhizdat, Moscow-Len~grad, 1948, pp. 339-354. (In Russian.)
- II S. Goyal, A. Ruina and J. Papadopoulos, Planar sliding with dry friction. Part 2. Dynamics of motion, *Wea.?,* 143 (1991) 331-352.
- 12 S. Goyal, A. Ruina and J. Papadopoulos, Limit surface and moment function descriptions of planar sliding, Proc. *IEEE Int. Conf. Robotics and Automation, May 1989,* Vol. 2, IEEE Computer Society Press, Washington, D.C., pp. 794-799.
- 13 S. Goyal, Planar sliding of a rigid body with dry friction: limit surfaces and dynamics of motion, *Ph.D. Thesis,* Cornell University, Ithaca, 1989.
- 14 R. Mieh~owski and Z. Mroz, Associated and non-associated sliding rules in contact friction problems, *Arch. Mech.,* 30 (3) (1978) 259-276.
- 15 W. McGuire, *Steel Structures,* Prentice-Hall, Englewood Cliffs, NJ, 1968, Section 6.5, pp. 812-820.
- 16 A. Curnier, A theory of friction, *Znt. J. Solid Struct.,* 20 (7) (1984) 637-647.
- 17 A. Zmitrowicz, A theoretical model of anlsotropic dry friction, Wear, 73 (1981) 9-39.
- 18 D. G. Drucker, A more fundamental approach to plastic stress-strain relationships, ln Proc. *1st National Congress of Applied Mechanics,* ASME, New York, 1951, pp. 487-491.
- 19 W. Moszynski, Frottement des corps solides dans le cas de l'anlsotropie naturelle et artificielle, Bull. *Acad. Pal. Sci. L&t., Ser. 7,* Suppl. 4 (1951) pp. 447-486.
- 20 J. J. Moreau, On unilateral constraints, friction and plasticity. In G, Capriz and G. Stampacchia (eds.), Nero *Variational Techniques* in *Mathematical Physics,* CIME, II ciclo 1973, Edlzioni Cremonese, Roma, 1974, pp. 175-322.
- 21 J. J. Moreau, Application of convex analysis to some problems of dry friction. In H. Zorski (ed.), *T?-cnds in Application of Pure Mathematics to Mechanics,* Vol. 2, Pitman, London, 1979, pp. 263-280.
- 22 B. E. Ron, Wheels for a course stable self-propelling vehicle movable in any desired direction on the ground or some other base, U.S. *Patent* 3 876 255 (1975).
- 23 S. Ziemba, On certain cases of anisotropic friction, *Arch.* Mech., 4 (1952) 105-121 (m Polish).
- 24 H. Ziegler, Discussion of some objections to thermomechanical orthogonahty, In9. Arch., 50 (1981) 149-164.
- 25 A. Schofield and P. Wroth, *Critical State Soil Mechanics,* McGraw-Hill, New York, 1968.
- 26 W. D. MacMillan, mnamics *of Rigid Bodies,* Dover Publications, New York, 1936.
- 27 V. N. Shvedenko, Equilibrium of a plane system of friction forces, Izv. *Akad. Nauk. SSSR Mekh. Tuerd. Tela, 21* (1) (1986) 37-42.