# Robert D. Howe

Division of Engineering and Applied Sciences Harvard University Cambridge, Massachusetts 02138

# Mark R. Cutkosky

Department of Mechanical Engineering Stanford University Stanford, California 94305

# Practical Force-Motion Models for Sliding Manipulation

## Abstract

The goal of this article is to develop practical descriptions of the relationship between forces and motions in sliding manipulation. We begin by reviewing the limit surface, a concept from the mechanics of sliding bodies that uses kinematic analysis to find the force and moment required to produce any given sliding motion. Next we provide experimental results showing that the limit surface only approximates the actual force-motion relationship. Then we look at other approximations that can be used to provide a simplified model useful in control, planning, and simulation of manipulation. These approximations include square pyramids, cones, ellipsoids, and ellipsoids with facets removed. Different approximations may be most appropriate, depending on the required computational speed and accuracy and the need to produce conservative results.

#### 1. Introduction

Consider picking up one of the tools on your desk (e.g., a pencil, pair of scissors, or telephone receiver) and moving it between your fingers to shift to a useful grasp. Such tasks inevitably require manipulation with controlled sliding. Sliding increases the mobility of the grasped object and the dexterity of the hand, provided that the force required to initiate sliding and the subsequent direction of motion are predictable. At other times, as when holding a heavy or delicate object, the main concern is to prevent unwanted slips. Again, it is important to know the forces at which sliding will commence. Knowledge of the probable direction of sliding can also be useful for readjusting the grasp.

The study of sliding with friction has become an increasingly important research topic in robotics; for recent comprehensive reviews, see Armstrong-Helouvry et al.

(1994) and Erdmann (1994). In some of this work the emphasis is on constitutive laws to accurately predict friction forces as a function of materials properties, pressures, velocities, etc. (Cutkosky and Wright 1986). This work draws on an extensive literature on tribology (Schallamach 1957, Armstrong-Helouvry 1991). Other work has been concerned with ambiguities and inconsistencies in quasistatic mechanical analyses that assume both rigid body and Coulomb friction models; examples include Lbtstedt (1981), Howard and Kumar (1993), Dupont (1993), and Lynch and Mason (1995).

A number of problems that involve sliding with friction have received attention in robotic manipulation research. One topic is the manipulation of objects by pushing them along a surface (Mason 1986; Peshkin 1986; Brost 1991; Lynch 1992; Rao and Goldberg 1993; Alexander and Maddocks 1993). Another is the assembly of close-fitting parts that slide against each other as part of the mating process (Whitney 1982; Erdmann 1994). In some work in the area of manipulation by multifingered hands, the goal is the prevention of sliding at the fingertips (Kerr and Roth 1986; Jameson 1985). Similar concerns motivate work toward configuring fixtures for holding workpieces to minimize unwanted slips (Sakurai 1990; Lee and Cutkosky 1991).

In other work with robot hands, the emphasis is on the kinematics of manipulation with sliding; one example is determining the admissible motions and trajectories that allow a sliding body to maintain contact with a set of fingertips (Montana 1988; Cai 1990; Rus 1992; Trinkle and Zeng 1994). In other work the goal is the use of controlled sliding to reorient the object in the hand (Brock 1988; Cole et al. 1992; Kao and Cutkosky 1992; Sun and Shi 1995). The methods used in a number of these studies are reviewed and contrasted in Section 4.1 below.

In this article the emphasis is on modeling the relationship between applied forces and moments and the corresponding directions of motion for distributed areas

of contact. We are particularly interested in finding simple models that are easily computed for use in real-time planning and control of manipulation. We begin with a review of the theoretical developments concerning friction with finite contact areas. These kinematic arguments use a description of the motion of a body sliding over a planar surface to derive frictional forces; the results are then inverted to find the motion resulting from an applied force and moment. We next compare the theoretical predictions with analytic approximations and the results of experi ments that measure the force-motion relationships. The conclusion is that there are several approximations that may be used in particular situations to provide the right balance of accuracy and computational efficiency.

# 2. Models of the Force-Motion Relationship

This section examines methods for calculating and describing the relationship between sliding motions and applied forces and moments. We also examine how the motion-force relationship is affected by various contact conditions. The analytical approach is closely related to work in classic plasticity theory concerning the relationship between stresses and plastic strain rates (Drucker 1954). In much of the work, the emphasis has been on the forward problem: given a known sliding motion, find the resulting friction force and moment. Liu and Paul (1989) provide a summary of integration methods for computing the force and moment for various pressure distributions over the contact area and provide plots of force and moment as a function of the instantaneous ratio of translational to rotational sliding. Goyal (1989) and Goyal et al. (1991) extend the analysis and provide useful tools for characterizing the motion-force relationship. Jameson (1985) considered the inverse problem for axisymmetric contacts in the context of grasp planning, with a goal of preventing unwanted slips.

### 2.1. Kinematics of Sliding

In the following description we will draw primarily on the concept of the limit surface, as developed in Goyal (1989) and Goyal et al. (1991). Construction of the limit surface requires a detailed analysis involving the pressure distribution at each point across the contact and the contribution of each point to the total frictional force and moment. Although the limit surface is constructed by solving the forward problem (computing forces and moments for each possible translational and rotational motion), it provides a mapping between applied forces and resulting motions that may be used to solve the inverse problem as well.

The basic assumptions in this analysis are (1) a body undergoes fully developed sliding on a locally planar

surface~; (2) the distribution of normal force (or pressure) across the contact is known; and (3) the friction force depends only on the local normal force and direction of slip, and not on the magnitude of the slip velocity or the slip history. The fully developed sliding criteria require that the relative velocity field across the contact area corresponds to a unique center of rotation, as explained below. This is always true for a rigid body and applies to deformable bodies such as soft fingertips to the extent that deformations of the contact area are slow compared to the sliding speed. Further restrictions on the form of the friction force are also required; see Goyal (1989) and Goyal et al. (1991) for details. Coulomb friction satisfies all of the specified restrictions, and variation of friction coefficient with normal force or with location across the contact area is acceptable.

The Coulomb friction model (also called Amonton, da Vinci, or dry friction) specifies that for a body in sliding motion, the tangential frictional force is proportional to the normal force and opposed to the direction of sliding. The one-dimensional force constraint is usually written as

$$f_t \le -\mu f_n sgn(v), \tag{1}$$

where ft is the translational force (tangent to the surface), f n is the local normal force (perpendicular to the surface), sgn(v) is the sign of the sliding velocity at the contact, and p is the coefficient of friction, assumed to have no dependence on velocity or normal force. The inequality applies before sliding initiates, and the equality applies during sliding.

In two dimensions the situation is more complicated, as the direction of motion at each point on the slider's surface must be determined to find the direction of the friction force. Kinematically, the instantaneous motion of a rigid body in the plane can always be described as a pure rotation (consistently specified as clockwise or counterclockwise) about some point, referred to as the center of rotation (COR). The relationship between sliding motion and total frictional force and moment for the sliding body may be calculated by assuming a known COR location and then summing up the contribution of the frictional force at each point across the contact.

Figure 1 shows the situation in the sliding plane. We assume that a rectilinear coordinate system is fixed in the plane, with the origin located at the friction-weighted center of pressure, (x,, Y,):

$$x_c = \frac{\int x \, \mu(x,y) \, p(x,y) \, dA}{\int \mu(x,y) \, p(x,y) \, dA}$$

<sup>1.</sup> Extensions to three-dimensional problems are addressed in Section 4.4.

Fig. 1. A contact in the sliding plane. The contact is rotating clockwise about the instantaneous center of rotation (COR), with resulting velocity  $\mathbf{v}$  at the infinitesimal contact area  $d\mathbf{A}$ .

$$y_c = \frac{\int y \,\mu(x,y) \,p(x,y) \,dA}{\int \int \mu(x,y) \,p(x,y) \,dA}.$$
 (2)

The vector from the origin to an element of the contact area is  $\mathbf{r} = [x \ y]^T$ , the vector from the origin to the instantaneous COR is  $\mathbf{r}_{\text{COR}}$ , and the vector from the COR to the point of interest is  $\mathbf{d}(x,y) = [d_x \ d_y]^T$ , so  $\mathbf{d} = \mathbf{r} - \mathbf{r}_{COR}$ . We have assumed that friction is independent of sliding speed, so the velocity can be represented by the unit vector  $\hat{\mathbf{v}}(x,y) = \mathbf{v}(x,y)/|\mathbf{v}(x,y)|$ . Since the contact is instantaneously rotating about the COR, this velocity vector is perpendicular to  $\mathbf{d}$ , and

$$\hat{\mathbf{v}}(x,y) = \frac{[-d_y \quad d_x]^T}{|\mathbf{d}(x,y)|}.$$
 (3)

At any point in the contact area the local normal force is given by  $df_n = p \, dA$ , where p(x,y) is the local value of the pressure distribution, and dA is the infinitesimal area at (x,y). The magnitude of the tangential frictional force vector at this point is  $df_t = \mu \, p \, dA$ , where  $\mu(x,y)$  is the local coefficient of friction. The direction of this force will be opposed to the velocity at the point, so the local frictional force vector is  $d\mathbf{f}_t = -\mu \, p \, \hat{\mathbf{v}} dA$ . The total frictional force is then found by integrating over the contact

$$\mathbf{f}_{t} = \begin{bmatrix} f_{x} \\ f_{y} \end{bmatrix} = -\int_{CONTACT} \mu \, p \, \hat{\mathbf{v}}(x, y) dA. \tag{4}$$

The contribution to the frictional moment (resolved to the origin) is given by the cross-product of the vector  $\mathbf{r}$  and

the local frictional force. Since the location and velocity vectors are coplanar, the moment is always perpendicular to the plane, and we can treat it as a scalar. The total moment m is the integral of this quantity

$$m = -\int [\mathbf{r} \times \hat{\mathbf{v}}] \mu \, p \, dA. \tag{5}$$

With these equations we can find the total frictional force and moment corresponding to any motion (described by the COR location  $\mathbf{r}_{\text{COR}}$ ) for a given pressure distribution p(x,y) and coefficient of friction  $\mu(x,y)$ . By performing this calculation for a number of COR locations, we can build up a picture of the relationship between sliding motion and force-moment. Unfortunately, the integrals depend on the details of the pressure distribution, and except for a few special cases, analytic solutions do not exist. To understand the nature of the force-motion relationship, we next consider the case of axisymmetric pressure distributions.

#### 2.2. Axisymmetric Contacts and Limit Curves

Axisymmetric pressure distributions are of practical value for approximating the contact pressure between a rounded, elastic fingertip and a hard object. In addition, as we will see below, the limit surface for other pressure distributions is often close enough to the axisymmetric case that the same limit curves can be used.

For axisymmetric contacts, we can without loss of generality orient the x-axis so that it passes through the COR, with the origin located at the centroid of the contact region. We can then write the vector between the origin and the COR as  $\mathbf{r}_{\text{COR}} = [c \ 0]^T$ , where c is the distance from the origin to the COR. For this choice of coordinates, the frictional force component in the x direction,  $f_x$ , is always zero, since the integral of this component over the upper half-plane cancels the integral over the lower half. For this case, equations (4) and (5), written in terms of polar coordinates  $(r, \theta)$ , become

$$f_{y} = -\int_{0}^{2\pi} \int_{0}^{R} \mu(r)p(r) \frac{(r\cos\theta - c)r}{\sqrt{r^{2} + c^{2} - 2cr\cos\theta}} dr d\theta$$
(6)  
$$m = -\int_{0}^{2\pi} \int_{0}^{R} \mu(r)p(r) \frac{(r - c\cos\theta)r^{2}}{\sqrt{r^{2} + c^{2} - 2cr\cos\theta}} dr d\theta.$$
(7)

Here R is the radius of the contact area. These are elliptic integrals; analytic solutions exist for special cases of the pressure distribution and COR location.

It is evident from equations (6) and (7) that  $f_y$  and m vary as a function of the COR distance, c. For example, Figure 2b plots m as a function of  $f_y$  for the case of a circular contact with uniform pressure. The result is an approximately elliptical shape called the *limit curve* for

Fig. 2. Relationship between sliding motion and frictional force and moment for a circular contact area with uniform pressure distribution. (A), Locations of the COR in sliding plane. (B), Corresponding force and moment combinations on the limit curve in the force-moment plane.

the contact. The limit curve explicitly demonstrates the coupling of forces and moments in sliding: the magnitude of the force required for sliding decreases as the applied moment increases, and vice versa. Note that the figure shows only one quadrant of the limit curve; reversing the sign of the moment and/or the force reflects the curves across the appropriate axes. This corresponds to a change in the direction of rotation, or changing the COR from the positive to negative x-axis. Also note that if u is assumed constant, it can be taken outside the integral and the entire curve simply scales with changes in the coefficient of friction.

The foregoing kinematic arguments and equations (6) and (7) provide a means for finding the forces that correspond to a given motion. This is useful for planning and control of manipulation, where we need to find the forces to be applied to produce some desired motion. However, the limit curve is useful in several other contexts as well. One is in tasks where it is important to avoid slips. Here the limit curve provides a means to determine whether sliding will result from a given combination of applied force and moment. This is accomplished by locating the point corresponding to the given force and moment on the limit curve plot: if the point is inside the curve (closer to the origin), no sliding will occur; if the point is on the curve, steady sliding will result; and if the point is outside the curve, then the contact will slide and accelerate, since the applied force and moment exceed the frictional forces for steady-state sliding. It can be shown (Goyal 1989) that friction limit curves and surfaces are always

convex; therefore, it is easy to determine whether a forcemoment combination lies inside the curve.

The limit curve can also be used to reverse the calculation from which it was constructed: to find the motion, both translational and rotational, tha~ results from a given applied force and moment. This is important for dynamic simulation of manipulation tasks and for planning and controlling contact tasks where forces may be generated by the task. For any combination of force and moment on the curve, we can (with appropriate interpolation) read off the value of c, which tells us the location of the COR and thus specifies the instantaneous motion of the contact. We will further discuss uses of the limit curve in Section 4.3.

### 2.3. Effect of Pressure Distributiort Variation

We may gain insight into the role of pressure distribution in determining sliding behavior by evaluating equations (6) and (7) for three cases that span the range of pressure distributions expected with soft fingertips on a robotic or human hand. These pressure distributions vary from a Hertzian distribution concentrated at the center (John son 1985) through a uniform distribution to a thin ring concentrated at the periphery (as in grasping the end of a tube). To isolate the effects of changing the pressure distribution, we will use the same radius for each contact and the same normal force In = f p(~, y) dA. We coNTAc' also assume that the coefficient of friction J1 is constant and identical in all three cases. Consequently, the maximum tangential force is the same in all cases, although

Table 1. Summary of Pressure Distributions and Maximum Moments for Three Axisymmetric Pressure Distributions.

| Contact Type | Pressure Distribution $p(r)$                                                                                                                              | Maximum Moment $(m_{\text{max}} \text{ in eqn.}(7) \text{ with } c = 0).$ |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Hertzian     | $p_H(r) = \begin{cases} p_0 \sqrt{1 - \left(\frac{r}{R}\right)^2} & 0 \leq r \leq R \ 0 & R < r \ p_0 = 3f_n/2\pi R^2 = \text{max. pressure} \end{cases}$ | $\frac{3\pi}{16}\mu f_n R$ $(\approx 0.589\mu f_n R)$                     |
| Uniform      | $p_U(r) = \left\{ \begin{array}{ll} f_n/\pi R^2 & 0 \leq r \leq R \ 0 & R < r \end{array} \right.$                                                        | $\frac{2}{3}\mu f_n R$ $(\approx 0.667\mu f_n R)$                         |
| Annular      | $p_{\scriptscriptstyle A}(r) = \left\{ \begin{array}{ll} 0 & 0 \leq r < sR \ p_s & sR \leq r \leq R \ 0 & R < r \end{array} \right.$                      | $\frac{2(1-s^3)}{3(1-s^2)}\mu f_n R$                                      |
|              | $p_s = f_n/\pi R^2 (1-s^2), 0 < s < 1$<br>sR = inner radius of annulus                                                                                    | $(\approx 0.951 \mu f_n R \text{ if } s = 0.9)$                           |

the maximum moment varies significantly, as shown in Table 1.

However, the shape of the limit curve is little changed by the variation in maximum moment. Figure 3 shows the limit curves for the three cases with the moment values normalized by dividing by the maximum moment for each case. A quarter ellipse is also drawn in the figure, scaled with the principal axes coincident with the maximum force and moment; this analytic curve lies near the limit curves for all three pressure distributions. This suggests that it may be possible to construct a single approximate limit curve, which can be scaled as appropriate for the details of a particular pressure distribution. We will return to this idea below.

# 2.4. Noncircularly Symmetric Contacts: Limit Surfaces

For applicability to a broader range of contacts, we must relax the assumption of circular symmetry so the pressure can vary as a function of both planar coordinates, p(x, y) or p(r, 0). Equations (4) and (5) remain the principal tools, but we are no longer free to orient the x-axis in an arbitrary direction. We must then consider COR locations throughout the plane. A further implication is that the frictional force in the x direction, fx, may be nonzero, so we must consider the direction of the frictional force as well as its magnitude. For each COR location, equations (4) and (5) are evaluated to find the corresponding values for the friction force components fx and fy and moment m. Thus, we can construct a three-dimensional limit Surface (LS) in a space parameterized by (fx, fy,m). Figure 4 shows an example of a limit surface, with the force components forming the horizontal plane and the

moment the vertical axis. In analogy with the circular contact case above, we will see that in many instances an ellipsoid is a reasonable approximation to the exact limit surface.

One important feature of the LS is that the frictional force always lies within a circular region on the (ix. fy) plane. In pure translation, all points of the contact are moving in the same direction (perpendicular to the direction to the COR, which is located at an infinite distance from the contact). The local unit velocity vector -v can then be taken out of the integral in equation (4)

$$\mathbf{f}_{t} = \begin{bmatrix} f_{x} \\ f_{y} \end{bmatrix} = -\hat{\mathbf{v}} \int \mu \, p(x, y) \, dA. \tag{8}$$

In this case it is clear that the direction of the friction force is given by -v and the magnitude is given by the scalar quantity inside the integral, which does not depend on the direction of motion. Thus, for any pressure distribution in pure translational sliding, the locus of the frictional force in the (fix, fy) plane is a circle of radius ft = f ~ + f ~ . If the coefficient of friction is constant, then the radius is simply ft = pfn.

Now we examine the frictional moment m. If the origin of the coordinate system is at the center of pressure, then m = 0 for pure translation. This is because the moment generated by the frictional force on the side of the contact closest to the COR is balanced by the contribution from the other side. As in the circularly symmetric case, the maximum moment occurs when the COR is at the origin (Goyal 1989); here all local velocity vectors are perpendicular to lines radiating from the origin and thus contribute fully to the moment. For many pressure

Fig. 3. Limit curves for the three circular pressure distributions. Each curve has been normalized by dividing by the maximum shear force and maximum moment for that pressure distribution. A true ellipse is plotted for comparison. For the annular pressure distribution, s = 0.9, (i.e., the inner radius is 0.9 of the outer radius).

distributions, this COR location will also result in zero tangential force so that the LS peaks at  $f_x = f_y = 0$ .

In general, the LS is most symmetric if the origin of coordinate system is located at the center of pressure (equation (2)). Under this condition, the peak value of the moment will be associated with zero tangential force if the pressure distribution displays periodic rotational symmetry (i.e., if the pressure distribution repeats itself in a revolution about the center of pressure). This obviously includes circularly symmetric distributions, as well as rectangles, equilateral triangles, etc.

Figure 4 also reveals a characteristic flattening of the LS for certain pressure distributions. Like the contact itself, the LS has two axes of symmetry. The extreme cross sections are plotted in Figure 5. We observe that COR locations on the y-axis (the minor axis of the pressure distribution) correspond to a cross section through the  $(f_x, m)$  plane of the LS. This produces a limit curve that is more convex than an approximating ellipse. Conversely, the cross section in the  $(f_u, m)$  plane produces a flattened cross section within the ellipse, which begins to approach a straight line for COR locations greater than about one unit from the origin. The case illustrated in Figures 4 and 5 is for a rectangle with a 10:1 ratio of side lengths; further increases in the aspect ratio have little effect. In general, as a contact patch becomes narrow in any direction, there is a corresponding flattening of the limit surface in the orthogonal direction.

As with the limit curves for circularly symmetric contacts, the sliding direction corresponding to any forcemoment combination on the LS is given by the vector  $\hat{\mathbf{n}} = [n_{f_x} n_{f_y} n_m]^T$  normal to the surface at that point (Goyal 1989). In Figure 5, normal vectors are drawn for several COR locations. The ratio of the translational to rotational velocities is given by the ratio of the component in the moment direction,  $n_m$ , to the component in the force direction,  $n_f = [n_{f_x}^2 + n_{f_y}^2]^{1/2}$ . On an LS with normalized axes, the slopes of the normal vectors are scaled by the same ratio,  $m_{\text{max}}/f_{t\,\text{max}}$ , used to normalize the rest of the plot, so that they remain perpendicular to the curves. Therefore, the ratio of angular to linear velocity is given by

$$\frac{|\mathbf{v}|}{\omega} = \left(\frac{n_f}{n_m}\right) \left(\frac{m_{\text{max}}}{f_{t\,\text{max}}}\right). \tag{9}$$

This ratio is equal to the magnitude of  $\mathbf{r}_{COR}$ . The direction of the translational velocity  $\hat{\mathbf{v}}$  is antiparallel to the force components of the normal vector, so

$$\hat{\mathbf{v}} = \frac{-[n_{f_x} - n_{f_y}]^T}{n_f}.$$
 (10)

#### 2.5. Point Contacts and Limit Surface Facets

The final feature of the LS we must consider is the appearance of facets when the pressure distribution is

Fig. 5. Cross sections in the  $(f_x, m)$  and  $(f_y, m)$  planes through the limit surface for a rectangular contact with a 10:1 ratio of side lengths. An ellipse is also drawn for comparison. Axes have been normalized by dividing by the maximum values of moment or tangential force.

composed of discrete points of contact. In manipulation, this can be a useful approximation when a relatively large object is grasped by several fingertips, where the diameter of each contact is substantially smaller than the distance between the contacts. Another common application is holding a part in fixtures and strap clamps with relatively small areas of contact (Lee and Cutkosky 1991).

Figure 6 shows the LS for three equidistant contact points. Each facet appears when the COR is located beneath one of the points of support. When the COR is immediately beneath a contact point, the point is not sliding, and the friction force is not completely defined—we know only that  $0 \le f_{t_i} \le \mu f_{ni}$ , where  $f_{t_i}$  and  $f_{ni}$  are the tangential and normal forces at the point i. Consequently, there is a range of possible values for the total frictional force and moment, obtained by summing the contributions of all contact points. In terms of the LS, this means that there is a range of possible forces and moments corresponding to a single COR lo-

cation (i.e., a single orientation of the LS normal). The result is a flat region corresponding to each discrete point of support. This indeterminacy occurs only if the COR is immediately beneath a point of support; if the COR moves even infinitesimally away, the point begins to slide, and the friction force is immediately given as  $\mu f_{ni}$ .

For larger numbers of contact points, the contribution of each point to the total force-moment balance becomes smaller, and the facet associated with each point shrinks. If the contact points are replaced by continuous pressure distributions, the facets vanish. However, the transition from faceted to continuous limit surfaces is gradual. For situations involving several small but extended regions of support, the LS develops flattened patches that, in the limit, would produce facets. For example, the LS in Figure 4 has flattened areas corresponding to locating the COR along the major axis of the thin rectangular contact. If the pressure distribution were approximated by a row of points, facets would appear on the LS.

Fig. 6. Top, pressure distribution consisting of three isolated contact points. Bottom, Corresponding limit surface showittg fcacets. (From Lee j1991 ]).

The gradual transition from flattened to faceted limit surfaces suggests an approximation technique. Because limit surfaces are always convex (Drucker 1954; Goyal 1989), it can be proven (Lee 1991) that approximating a set of small regions of pressure with points of support that satisfy the same equilibrium conditions leads to a conservative approximating LS. Moreover, if the distances between the regions of support are greater than the average region diameter, the resulting approximation will be close to the exact result. A procedure for rapidly creating an approximating limit surface and then removing facets is provided in Section 4.2.

# 3. Experiments

The above analysis establishes a theoretical force-motion relation for sliding. However, it is based on problematic assumptions (such as velocity independence of the coefficient of friction and rigid-body motion at the contact), and requires knowledge of the pressure distribution and coefficient of friction across the contact. Even for hard materials, for which the Coulomb friction model is frequently accurate, there may be local variations in the pressure distribution and, if dirt and moisture are present, in the friction coefficient. Given these complicating factors, it is important to examine how well the limit surface predicts the onset of sliding in realistic circumstances. In this section we describe two sets of experiments on the onset of sliding.

The first experiments involve metallic contacts, as when a metal workpiece is held by fixturing elements or the fingers of an industrial gripper. Figure 7 shows a typical result for a rectangular part clamped in a vise. In such clamping arrangements, the details of the pressure distribution are not generally known; uniform pressure was assumed. The total normal force and the applied tangential force and moment were measured with load cells while the onset of sliding was recorded.

The plot in Figure 7 is a cross section of the limit surface in the ( f y, m) plane, corresponding to COR locations along the major axis of the contact. The &dquo;approximation&dquo; curve is an ellipse fit to the values of maximum moment and maximum tangential force. For the most part, the values agree with the theoretical limit curve, especially considering the unknown pressure distribution. Results for other fixturing arrangements can be found in Lee (1991). In general they agree with theoretical predictions, the closeness of the match depending on how well the pressure distribution can be estimated.

In another set of experiments, rubber &dquo;fingertips&dquo; of different shapes were pressed against a smooth glass surface ; then a torsion load was applied, and the shear force was increased until the contact began to slip. The upper set of points in Figure 8 presents the measured slip

Fig. 7. Experimentally measured friction limits for a rectangular metallic contact. An ellipse fitted to the maximum moment artd force is also slzown. Error boxes surrounding empirical points reflect uncertainty in the load cell measurements and in determining when the workpiece first slipped. (From Lee [19911).

values for a flat circular fingertip made of a relatively hard silicone rubber. The dashed line is the theoretical limit curve for a uniform pressure distribution, calculated from equations (6) and (7). The value for the coefficient of friction was inferred from the pure translation data. The data show general agreement with the calculated slipping points, particularly near the pure rotation extreme. At intermediate values, however, slipping commenced at lighter loads than predicted. This will obviously lead to difficulties if the theoretical curves are used to plan manipulation tasks where slip is to be avoided.

The lower set of points in Figure 8 is for a hemispherical fingertip made of a soft natural rubber, while the dotted line represents the calculated limit curve for a Hertzian pressure distribution fitted to the maximum shear force data. In general, the measured slip points trace

out a flattened elliptical curve similar to the flat fingertip. These measured points fall on the calculated limit curve for small moments, but measured points are well above the curve for large moments. This deviation can be explained by the frictional properties of elastomers, which can vary greatly with sliding speed and normal force (Schallamach 1957), contrary to the assumptions of the theoretical development. In particular, for many rubbers the coefficient of friction decreases at low sliding speeds, which can cause &dquo;creep&dquo; even when the applied shear load is less than the apparent friction limit. In addition, the effective coefficient of friction increases as local pressure decreases for many rubbers. Rubber fingertips can also deform across the contact area, violating the assumption of fully developed sliding at the contact. All of these effects are, in general, more pronounced for softer elastomer compounds.

Fig. 8. Experimentally measured friction limits for two rubber fingertips. Calculated limit curves are also shown. Error bars near 2N data points indicate typical moment measurement uncertainty.

From our rubber fingertip experiments, we observe that the shape of the measured limit curve is often a flattened ellipse, significantly below the theoretical limit curve for the middle ranges of loading. Also, the hemispherical fingertip results suggest that it can be difficult to estimate the size of the limit surface from a single slip measurement. However, the reasonable agreement with theory for the flat contact measurements shows that these difficulties can be minimized if an appropriate elastomer is selected (Cutkosky and Wright 1986).

# 4. Practical Methods of Computing the Force-Motion Relationship

The basic method of constructing a limit surface described above consists of selecting a reasonably dense set of COR locations, applying equations (4) and (5) at each point to find the corresponding force-moment combinations, and then plotting the results in force-moment space-a potentially time-consuming operation requiring detailed knowledge of the pressure distribution. A number of other methods have been devised for finding the force-velocity relationship; none, however, summarizes the entire relationship for the contact as directly and succinctly as the limit surface. In this section we review other methods and then propose approximations to the limit surface that are particularly fast to compute. We also discuss applications of these approximations in grasp

and motion planning and extensions to three-dimensional problems.

### 4.1. ~'omparison of Methods

In an approach related to the construction of limit surfaces, Mason (1986) examines the motion of an object pushed across a flat surface by a fence or &dquo;pusher.&dquo; The pressure distribution between the object and the surface is assumed unknown, but the contact between the pusher and the object provides an applied force of known magnitude, due to the friction between them. The resulting equilibrium equations contain the location of the COR implicitly. Mason examines all possible directions for the velocity of the contact point that satisfy geometric and equilibrium constraints and are consistent with the pusher velocity. In other words, he first plots a COR locus and then finds the correct location along the locus by requiring the contact forces to match the actual value. Peshkin ( 1986), Brost (1991), and Alexander and Maddocks (1993) provide extensions to the sliding analysis, all of which maintain the assumption that the contact pressure distribution is not known.

Other approaches directly exploit the convexity of the limit surface, which derives from the &dquo;maximum work inequality&dquo; of Coulomb friction (Drucker 1954). This condition prescribes that, given a direction of sliding motion, the actual friction force-moment will maximize the work done in sliding (i.e., the dot product of force

and velocity) while also satisfying equation (1) for each element of the contact. For the trivial case of a single point of contact, the maximum dot product clearly obtains if the friction force is parallel to the direction of sliding. A consequence of the convexity of limit surfaces is that the limit surface for a collection of points or pressure distributions can be obtained by Minkowski summation of the individual surfaces for each element. Typically, however, this approach is more time consuming than computing the limit surface directly using equations (4) and (5).

For fixturing applications involving small regions or discrete points of support as in Section 2.5, Sakurai (1990) uses a &dquo;maximum force method&dquo; to compute the COR location for an applied force and moment by maximizing the inner product of a vector of the applied forces and moments with a vector of the friction forces. The maximum is found subject to equilibrium equations, to the known magnitudes of the friction forces at each contact point, and to the known direction of the externally applied tangential force. Because the direction of the applied force is known, a limit surface is not needed. In terms of limit surfaces, this is equivalent to solving for the intersection of a line representing fixed ratios of /a-/m and fy/m with the limit surface. Other optimization methods have been applied to three-dimensional problems involving point contacts with friction by Trinkle and Zeng (1994). Such methods are faster than constructing a limit surface. However, the main time saving is due to the assumption that the direction of the loading force is known.

Another solution method is possible if the applied tangential force and the pressure distribution over the contact are known. Bicchi et al. (1988) consider the case of a block squeezed between two flat fingers of a paralleljaw gripper. The resultant friction forces fx and fy are assumed known (perhaps measured by strain gauges in the gripper jaws), along with the pressure distribution over the contact area (perhaps measured by a tactile array sensor). They obtain the location of the COR by solving two simultaneous nonlinear equations of equilibrium in the x and y directions:

$$f_x - \sum_{i=1}^n \frac{-\mu p_i (y_i - y_{COR})}{\left[ (x_i - x_{COR})^2 + (y_i - y_{COR})^2 \right]^{1/2}} = 0,$$

$$f_y - \sum_{i=1}^n \frac{-\mu p_i (x_i - x_{COR})}{\left[ (x_i - x_{COR})^2 + (y_i - y_{COR})^2 \right]^{1/2}} = 0,$$

where n is the number of pressure sensing array elements and pi is the measured normal force at element location (xi, y2)~ Since fx and fy are also measured directly, there are just two unknowns, XCOR and YCOR. This method is also applicable to situations such as those examined by

Sakurai involving discrete points of support, but requires more measurements than the other methods.

# 4.2. Practical Approximate Solutions

In this section we consider practical approximations to the limit surface that can be used for grasp and motion planning. As seen in Section 3, the theoretical limit surface is in reasonable agreement with the experimental results to the extent that the coefficient of friction and the pressure distribution are accurately known. Consequently, there arises a tradeoff between desired limit surface accuracy and computational effort. For practical purposes, we seek an approximation whose deviation from the exact limit surface is no worse than typical errors caused by uncertainties in the coefficient of friction and pressure distribution.

As Figures 2 through 8 reveal, the obvious approximation for most limit surfaces is an ellipsoid. For circularly symmetric pressure distributions, the limit surface is axisymmetric, and for noncircular distributions, a typical limit surface will be considerably more axisymmetric than the pressure distribution itself (e.g., Figure 4). The reasons for this effect are: first, that all pressure distributions result in a circular cross section in the (fx, fy) plane due to the assumed isotropy of friction; and second, that the forces and moment in equations (4) and (5) are computed as integrals over the pressure distribution so that they tend to smooth local variations in the pressure and friction coefficient.

Consequently, an ellipse fit to the maximum moment and maximum tangential force provides a good approximation in most cases. For asymmetric pressure distributions (where fx and fy are not zero when m is maximum), an ellipsoid with a tilted major axis can be constructed, and for a pressure distribution with discrete points of support, facets can be removed; this procedure is outlined in Table 2. In each case the ellipse or ellipsoid is easy to construct. Because an analytic surface is obtained, it is also easy to find the normal vector for any point on the surface and to determine whether any combination of force and moment lies inside the surface.

One shortcoming of this approximation is the potential for a relatively large error in the surface normal direction. Figure 3 shows that the ellipse lies reasonably close to the actual normalized limit curve despite variation in the pressure distribution. Close examination, however, reveals significant differences in the surface normal direction at neighboring points on the various curves, particularly at force-moment combinations near the center of the range. This implies that the ellipse approximation is relatively accurate in determining whether a given combination of force and moment will slip, and less accurate in relating force-moment combinations to the sliding velocity.

### Table 2. Summary of Procedures for Constructing Approximating Ellipses and Ellipsoids for Limit Surfaces.

#### 1. Characterize contact.

- Determine the pressure distribution: estimate (e.g., assume Hertzian contact for round fingertip) or measure (e.g., with tactile array sensor).
- Place the origin of the sliding coordinate frame at the center of pressure using equation (2).
- Determine the coefficient of friction: estimate or measure (e.g., take ratio of actual forces while sliding fingertip (Bicchi et al. 1991) or use an incipient slip sensor (Tremblay and Cutkosky 1993; Son et al. 1994).

#### 2. Find approximating ellipse.

- Find the maximum frictional force  $f_{t max}$ : calculate (equation (1)) or measure (in pure translational sliding).
- Find the maximum frictional moment and corresponding forces  $(f_{x0} f_{y0} m_{\text{max}})$ : calculate (equation (5) with COR at origin) or measure (while rotating about center of pressure).
- •This force-moment combination defines the end point and tilt angles of the ellipsoid major axis:
- —If  $f_{x0}$  and  $f_{y0}$  are small, assume the major axis is vertical. Then the ellipsoid is circularly symmetric, and the problem reduces to fitting an ellipse to  $m_{\text{max}}$  and  $f_{\text{t max}}$ .
- —If the ellipsoid is tilted due to a highly asymmetric pressure distribution, it may be necessary to define a new coordinate frame  $(\tilde{f}_x, \tilde{f}_y, \tilde{m})$  aligned with the major axis of the ellipsoid and to transform the points of maximum tangential force into this new coordinate frame. For details see Lee (1991) and Lee and Cutkosky (1991).

#### 3. Normalize ellipse.

• To normalize the ellipse, scale each point (f, m) such that  $(f', m') = (f/f_{t max}, m/m_{max})$ . In this case the ellipse is simply a circle. Note that this circular limit curve scales with increasing normal force (which just proportionately increases  $m_{max}$  and  $f_{t max}$ ).

#### 4. Add facets/flattening.

•If the pressure distribution is dominated by a few discrete and widely separated points or regions of contact, it may be desirable to remove the facets that correspond to them. For each such point  $\mathbf{r}_i = [r_x, r_y]^T$  of contact, a plane can be constructed corresponding to locating the COR at that point. This defines a normal,  $\mathbf{n}_i = [-r_x, r_y, 1]^T$ , in the force-moment space of the limit surface. The distance from the origin to the plane is then defined by picking any point on the facet. Recall that the friction force at a discrete point of support is not uniquely defined. One valid solution is zero force, which defines a point on the plane and thereby uniquely defines the plane in  $(f_x, f_y, m)$  space.

#### 5. Relate sliding motion to force-moment combination.

- If the limit surface is approximated with an ellipse, the translational sliding direction is parallel to the tangential force.
- The ratio of translational to rotational velocities is given by the horizontal and vertical components,  $n_f$  and  $n_m$  of the normal to the limit curve at (f, m). For an ellipse it is straightforward to show that

$$\frac{n_f}{n_m} = \lambda^2 \frac{f}{m}$$

where

$$\lambda = \frac{m_{\textsf{max}}}{\textsf{f}_{\textsf{t}\,\textsf{max}}}.$$

If the ellipse is normalized as in step 3, the relationship becomes

$$\frac{|v|}{\omega} = \frac{n'_f}{n'_m} = \lambda \frac{f'}{m'}.$$

In general, the best method of accurately relating forcemoment to velocity is to find a number of points on the limit surface using equations (4) and (5). This requires choosing an appropriate set of sample COR locations and calculating the corresponding forces and moments. Since the COR uniquely specifies the velocity, these values may then form a look-up table.

For certain applications such as real-time control, a faster approximation may be needed, especially if there are discrete points of support for which it is necessary to first construct an ellipsoid and then remove facets. Also, as seen in Figure 5, the ellipsoidal approximation is not conservative; it can overestimate the friction limit for contacts that are not circularly symmetric. Thus, a fast and conservative approximation is a cone with its apex at mmax and a base circle of radius it max. Because limit surfaces are convex, the cone is strictly conservative. In analogy with a circularly symmetric ellipsoid, the cone can be reduced to a straight line for many cases. Figure 8 shows that a straight line may be a better approximation than an ellipse for soft rubber fingertips. One drawback to this approximation is that the correspondence between the normal vectors and the ratio of rotational to translational sliding is lost.

Finally, for some analyses (e.g., grasp planning or optimal control via linear programming) it is useful to have a linear constraint function for friction. For example, the limit surface may be approximated as a set of inequalities representing a pyramid (Kerr and Roth 1986)

$$|f_x| \le \frac{1}{\sqrt{2}} \left( \mu f_n - \frac{m}{\lambda} \right), \quad |f_y| \le \frac{1}{\sqrt{2}} \left( \mu f_n - \frac{m}{\lambda} \right),$$

$$|m| \le m_{\text{max}}. \tag{11}$$

However, as with the conical limit surface approximation, the relationship between the normal to the limit surface and the direction of sliding motion is lost.

#### 4.3. Uses: Planning, Simulation, and Control

As we have seen, the LS provides essential information for sliding manipulation. To determine whether a contact will slip, the load (ix, fy, m) is plotted in force-moment space and compared to the LS. If the vector lies on the surface, steady sliding will result; if it lies outside, the applied force exceeds the friction limit, and the contact will probably accelerate; if it lies inside, the distance from the tip of the load vector to the surface provides a safety margin against sliding.

As discussed earlier, the LS also relates applied forces and moments to the direction of motion. One example of using the LS for sliding manipulation starts with the desired velocity vector. Then the point on the LS normal to this vector is located. The values of the force and

moment at that point are the loads that should be applied to produce the desired motion. Alternatively, given a history or sequence of task-related forces and moments, the instantaneous motion directions obtained from the LS can be integrated to obtain sliding trajectories (Kao and Cutkosky 1993; Sun and Shi 1995). Even for static applications, knowing the direction of sliding is useful for reconfiguring a grasp to minimize the likelihood of slips.

When compared with other methods for determining when sliding will occur, the LS has the advantage of being computed just once for a contact arrangement. This is particularly useful for grasp and fixture planning where the LS is constructed once and then tested against any combination or sequence of applied forces and moments. Moreover, the surface scales linearly with changes in the coefficient of friction and, for cases of uniform pressure, with changes in the normal force.

One difficulty encountered during manipulation is that the apparent coefficient of friction varies in time with a number of factors (e.g., surface cleanliness, temperature, etc.). Thus, it is desirable to perform active slip detection and to update the LS. One approach is to measure the onset of slip using dynamic tactile sensors (Tremblay and Cutkosky 1993; Son et al. 1994) and to measure the contact forces and moment using fingertip force sensors whenever the onset of sliding is detected. With these measurements it is possible to adaptively update the estimate of the friction coefficient and maximum force and moment. It is then easy to scale the LS as these values change. There is evidence that people unconsciously use a similar strategy, detecting the onset of sliding with cutaneous mechanoreceptors to maintain a grasp force sufficient to avoid dropping objects while minimizing effort (Johansson and Westling 1987).

# 4.4. Three-Dimensional Applications

Thus far we have only considered sliding on a (locally) planar surface. However, the basic approach embodied in limit surfaces can be extended to applications in which a set of contacts is rigidly connected but not confined to a plane-for example, a part clamped in fixtures with contacts on orthogonal faces. A detailed treatment of such applications is beyond the scope of this article, but the basic approach is as follows (Lee and Cutkosky 1991 ):

- 1. For each contact, construct a LS in the local coordinate system for feasible combinations of translation and rotation (analogous to COR locations).
- 2. Express the limit surfaces of all contacts in a common force-moment coordinate system and add them by Minkowski summation. During this step, a limit surface with up to six dimensions (three moment

and three force components) may result; however, only the projections on subspaces for which sliding motion is kinematically possible need be considered. Although this step is potentially time consuming and could lead to limit surfaces of more than three dimensions, in practice this is unlikely because a good grasp or fixturing arrangement will only have two or three degrees of freedom for which sliding is possible.

This approach works for cases such as a part clamped in fixtures, for which the contacts are all rigidly connected. If the contacts are connected by compliant members, such as the fingers of a robot hand, it is necessary to consider each contact separately to see if it will slide and, if so, in which direction. However, by modeling the grasp compliance, it is possible to compute the normal and tangential forces at each fingertip as a function of external loading on the grasped object and subsequently to compute sliding motions of a grasped object (Kao and Cutkosky 1992; Sun and Shi 1995).

#### 5. Conclusions

Sliding manipulation requires appropriate models of the relationship between frictional forces and sliding motion. When both rotational and translational sliding are involved, the familiar Coulomb friction law is replaced with a limit surface that represents a mapping between applied forces and moments and resulting motions. The complete mapping can be time consuming to construct, as it involves numerical integration for each point on the surface. However, approximate solutions can be used that provide errors no worse than those caused by typical uncertainties in contact pressure distributions or friction coefficients. Depending on the geometry of the contact and the desired accuracy, the most appropriate approximation may be an ellipsoid (perhaps with facets), a cone or a pyramid in three-dimensional force-moment space. In many cases the surface is nearly axisymmetric, in which case the ellipsoid and cone can be replaced by an ellipse and a straight line, respectively, in a two-dimensional plot of moment versus tangential force.

Once constructed, the limit surface is useful for a variety of grasp and motion planning applications. Any series of anticipated forces and moments can be tested against the surface-those that fall inside are &dquo;safe,&dquo; and those that lie on or outside the surface will produce sliding. The surface also relates forces and moments to the direction of sliding. For each point on the surface, the direction cosines of the normal vector at that point are proportional to the relative magnitudes of rotational and translational motion. We note that friction coefficient and pressure distributions are likely to change over time as functions of

loading and surface conditions, and the detection of the onset of sliding permits the limit surface to be updated dynamically.

# Acknowledgments

This work was supported by the Office of Naval Research under grant no. N00014-92-J-1887 and by the National Science Foundation under grant no. IRI-9357768.

#### References

- Alexander, J. C., and Maddocks, J. H. 1993. Bounds on the friction-dominated motion of a pushed object. Int. J. Robot. Res. 12(3):231-248.
- Armstrong-Helouvry, B. 1991. Control of Machines with Friction. Norwell, MA: Kluwer Academic Publishers.
- Armstrong-Helouvry, B., Dupont, P., and Canudas De Wit, C. 1994. A survey of models, analysis tools and compensation methods for the control of machines with friction. Automatica 30(7):1083-1138.
- Bicchi, A., Bergamasco, M., Dario, P., and Fiorillo, A. 1988 (Zürich, February 2-4). Integrated tactile sensing for gripper fingers. Proc. 7th Int. Canf. on Robot Vision and Sensory Controls. Bedford, UK: IFS Publications and Berlin: Springer-Verlag.
- Bicchi, A., Salisbury, J. K., and Brock, D. L. 1991 (Toulouse, France, June 25-27). Experimental evaluation of friction characteristics with an articulated robotic hand. In Chatila, R., and Hirzinger, G. (eds.): Experimental Robotics II: The Second International Symposium. New York: Springer-Verlag, pp. 153- 167.
- Brock, D. L. 1988 (Philadelphia, April 24-29). Enhancing the dexterity of a robot hand using controlled slip. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 249-251.
- Brost, R. C. 1991. Analysis and planning of planar manipulation tasks. Ph.D. thesis. School of Computer Science, Carnegie Mellon University,
- Cai, C. 1990 (Cincinnati, May 13-19). Control of a manipulator under multiple contacts between rigid curved surfaces. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 1356-1361.
- Cole, A. A., Hsu, P., and Sastry, S. S. 1992. Dynamic control of sliding by robot hands for regrasping. IEEE Trans. Robot. Automation 8(1): 42-52.
- Cutkosky, M. R., and Wright, P. 1986. Friction, stability and the design of robotic fingers. Int. J. Robot. Res. 5(4):20-37.
- Drucker, D. C. 1954. Coulomb friction, plasticity, and limit loads. J. Applied Mechanics 21(1):71-74.

- Dupont, P. E. 1993. The effect of friction on the forward dynamics problem. Int. J. Robot. Res. 12(2):164-179.
- Erdmann, M. 1994. On a representation of friction in configuration space. Int. J. Robot. Res. 13(3):240-271.
- Goyal, S. 1989. Planar sliding of a rigid body with dry friction: Limit surfaces and dynamics of motion. Ph.D. thesis. Department of Mechanical Engineering, Cornell University.
- Goyal, S., Ruina, A., and Papadopoulos, J. 1991. Planar sliding with dry friction, part 1. Limit surface and moment function. Wear 143:307-330.
- Howard, W. S., and Kumar, V. 1993 (Atlanta, May 2- 6). A minimum principle for the dynamic analysis of systems with frictional contacts. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 437-442.
- Jameson, J. W. 1985. Analytic techniques for automated grasp. Ph.D. thesis. Department of Mechanical Engineering, Stanford University.
- Johansson, R. S., and Westling, G. 1987. Signals in tactile afferents from the fingers eliciting adaptive motor responses during precision grip. Exp. Brain Res. 66:141-154.
- Johnson, K. L. 1985. Contact Mechanics. Cambridge, UK: Cambridge University Press.
- Kao, I., and Cutkosky, M. R. 1992. Quasistatic manipulation with compliance and sliding. Int. J. Robot. Res. 11(1):20-40.
- Kao, I., and Cutkosky, M. R. 1993. Comparison of theoretical and experimental force/motion trajectories for dextrous manipulation with sliding. Int. J. Robot. Res. 12(6):529-534.
- Kerr, J., and Roth, B. 1986. Analysis of multifingered hands. Int. J. Robot. Res. 4(4):3-17.
- Lee, S.-H. 1991. Concurrent fixture planning and analysis for machined parts. Ph.D. thesis. Department of Mechanical Engineering, Stanford University.
- Lee, S.-H., and Cutkosky, M. R. 1991. Fixture planning with friction. ASME J. Engineering Industry 113(3):320-327.
- Liu, C., and Paul, B. 1989. Fully developed sliding of rough surfaces. J. Tribology 111:445-451.
- Lötstedt, P. 1981. Coulomb friction in two-dimensional rigid-body systems. Zeitschrift für Angewandte Mathematik und Mechanik 61:605-615.
- Lynch, K. M. 1992 (Nice, France, May 12-14). The mechanics of fine manipulation by pushing. Proc. 1992

- IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 2269-2276.
- Lynch, K. M., and Mason, M. T. 1995. Pulling by pushing, slip with infinite friction, and perfectly rough surfaces. Int. J. Robot. Res. 14(2):174-183.
- Mason, M. T. 1986. Mechanics and planning of manipula tor pushing operations. Int. J. Robot. Res. 5(3):53-71.
- Montana, D. J. 1988. The kinematics of contact and grasp. Int. J. Robot. Res. 7(3):17-32.
- Peshkin, M. A. 1986. Planning robotic manipulation strategies for sliding objects. Ph.D. thesis. Depart ment of Mechanical Engineering. Carnegie Mellon University.
- Rao, A. S., and Goldberg, K. Y. 1993 (Atlanta, May 2- 6). On the relation between friction and part shape in parallel-jaw grasping. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 461- 466.
- Rus, D. 1992 (Nice, France, May 12-14). Dexterous rotations of polyhedra. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 2758-2763.
- Sakurai, H. 1990. Automatic setup planning and fixture design for machining. Ph.D. thesis. Department of Mechanical Engineering, Massachusetts Institute of Technology.
- Schallamach, A. 1957. Friction and abrasion of rubber. Wear 1: 384-417.
- Son, J. S., Monteverde, E. A., and Howe, R. D. 1994 (San Diego, May 8-13). A tactile sensor for localizing transient events in manipulation. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 471-476.
- Sun, D., and Shi, X. 1995 (Nagoya, Japan, May 21- 27). Coordination of two robots manipulating a flat object with sliding constraints. Proc. IEEE Int. Conf. on Robotics and Automation. New York: IEEE, pp. 1814-1819.
- Tremblay, M. R., and Cutkosky, M. R. 1993 (Atlanta, May 2-6). Estimating friction using incipient slip sensing during a manipulation task. Proc. IEEE Int. Conf. on Robotics and Manipulation. New York: IEEE, pp. 363-368.
- Trinkle, J. C., and Zeng, D. C. 1994. Prediction of the quasistatic planar motion of a contacted rigid body. IEEE Trans. Robot. Automation 11(2):229-246.
- Whitney, D.E. 1982. Quasi-static assembly of compliantly supported parts. ASME J. Dynam. Sys. Meas. Control 104(1):65-77.