### F. C. Park

Mechanical Design and Production Engineering Seoul National University Seoul, Korea

# J. E. Bobrow S. R. Ploen

Department of Mechanical and Aerospace Engineering University of California, Irvine Irvine, California 92717

# A Lie Group Formulation of Robot Dynamics

## **Abstract**

In this article we present a unified geometric treatment of robot dynamics. Using standard ideas from Lie groups and Riemannian geometry, we formulate the equations of motion for an open chain manipulator both recursively and in closed form. The recursive formulation leads to an O(n) algorithm that expresses the dynamics entirely in terms of coordinate-free Lie algebraic operations. The Lagrangian formulation also expresses the dynamics in terms of these Lie algebraic operations and leads to a particularly simple set of closed-form equations, in which the kinematic and inertial parameters appear explicitly and independently of each other. The geometric approach permits a high-level, coordinate-free view of robot dynamics that shows explicitly some of the connections with the larger body of work in mathematics and physics. At the same time the resulting equations are shown to be computationally effective and easily differentiated and factored with respect to any of the robot parameters. This latter feature makes the geometric formulation attractive for applications such as robot design and calibration, motion optimization, and optimal control, where analytic gradients involving the dynamics are reauired.

### 1. Introduction

From a certain point of view the problem of generating the equations of motion for a robot manipulator presents no difficulty; by regarding the robot as a system of coupled rigid bodies, the equations can be derived straightforwardly from either Newton's Laws or Lagrange's equations of motion. Yet even for the simplest open-chain robots these equations can become extremely complex; numerous formulations have therefore been proposed that attempt to reduce their symbolic and numerical

complexity. Recent advances in computing technology have diminished somewhat the importance of finding ever more efficient algorithms. However, the increasing complexity of methods for the design, control, and motion planning of robot manipulators, combined with the increasing complexity of the robots themselves, has now shifted the emphasis to finding more systematic and elegant dynamics formulations. For the applications mentioned earlier, a dynamics formulation in which the robot parameters appear in an explicit fashion, so that gradients with respect to these parameters are easily obtained, is clearly desirable.

In this article we use standard ideas from Lie groups and Riemannian geometry to formulate the dynamic equations for an open chain manipulator. The cornerstone of our approach is to regard SE(3), the Euclidean group of rigid-body motions, as a Lie group. By adopting this geometric framework, many of the ad hoc definitions and notational conventions found in existing dynamics algorithms can be avoided. At the same time, explicit contact can be made with the significant and growing body of work in differential geometry; several researchers have already shown connections between some of these geometric ideas and various aspects of robotics (e.g., Loncaric 1985; Brockett 1990; Spong 1992). Here our specific objectives are to suitably "geometrize" both the Newton-Euler and Lagrangian formulations of robot dynamics, extending the work previously initiated by Li (1989). We first derive a recursive O(n)dynamics algorithm in which the link velocities and accelerations are expressed in terms of standard operations on the Lie algebra of SE(3). A comparison with Featherstone's (1987) recursive algorithm reveals some interesting connections between his six-dimensional spa-

The International Journal of Robotics Research, Vol. 14, No. 6, December 1995, pp. 609–618, © 1995 Massachusetts Institute of Technology.

<sup>1.</sup> Balafoutis and Patel (1991) discuss computational aspects of several recursive dynamics algorithms in the literature.

tial vector notation and definitions to these general Lie algebraic operations.

We also present a Lagrangian dynamics formulation that leads to a particularly simple set of closed-form equations of motion. One of the strengths of this formulation is that the robot parameters now appear explicitly: they can be easily manipulated and factored from these equations, without the complex rules or iterations typical of most recursive schemes. Applications of such a set of closed-form dynamic equations are numerous. For example, in robot adaptive control, the exact linear relationship between the inertial parameters and the applied forces and torques must be known a priori, and in many robust control schemes the equations must be linearized about some trajectory. Also, the performance of many optimal robot design and motion optimization algorithms depends crucially on knowledge of the gradient of the objective function; often this requires differentiating the dynamics equations with respect to various robot parameters, and with our closed-form equations this can now be done quite easily. We also compare our formulation with the spatial operator algebra formulation of Rodriguez et al. (1991), who obtain similar results by establishing a rather interesting analogy between robot dynamics and the equations for linear state-space estimation.

We begin with the necessary geometric fundamentals of the Euclidean group SE(3), focusing in particular on what the correct Lie algebraic representation for generalized forces should be, and the different Jacobian formulations that are possible from the product of exponentials formula. (See Murray et al. [1993] for a thorough introduction to these topics.) The geometric formulation of the recursive Newton-Euler dynamics is then presented, followed by the closed-form Lagrangian formulation.<sup>2</sup> We conclude with a discussion on how these equations can be advantageous for certain robotics applications.

# 2. Geometric Background

## 2.1. SE(3) and se(3)

For our purposes it is sufficient to think of SE(3), the *Special Euclidean Group* of rigid-body motions, as consisting of matrices of the form

$$\left[\begin{array}{cc} \mathbf{\Theta} & \mathbf{b} \\ 0 & 1 \end{array}\right],$$

where  $\Theta \in SO(3)$  and  $\mathbf{b} \in \Re^3$ . Here SO(3) denotes the group of  $3 \times 3$  rotation matrices. Elements of SE(3) will alternatively be denoted by the ordered pair  $(\Theta, \mathbf{b})$ , with group multiplication understood to be  $(\Theta_1, \mathbf{b}_1) \cdot (\Theta_2, \mathbf{b}_2) =$ 

 $(\Theta_1\Theta_2, \Theta_1\mathbf{b}_2 + \mathbf{b}_1)$ . The *Lie algebra* of SE(3), denoted se(3), consists of matrices of the form

$$\begin{bmatrix} [\boldsymbol{\omega}] & \mathbf{v} \\ 0 & 0 \end{bmatrix},$$

where

$$[\boldsymbol{\omega}] = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}.$$

The set of  $3 \times 3$  real skew-symmetric matrices forms the Lie algebra of SO(3), denoted so(3). Note that an element  $[\omega] \in \text{so}(3)$  can also be regarded as a vector  $\omega \in \Re^3$ ; since in most cases it will be clear from the context which representation is implied, an element of so(3) will also be simply denoted by  $\omega$ . Elements of se(3) will also be represented as  $(\omega, \mathbf{v}) \in \Re^6$ .

On matrix Lie algebras the *Lie bracket* is given by the matrix commutator: if **A** and **B** are elements of a matrix Lie algebra, then  $[\mathbf{A}, \mathbf{B}] = \mathbf{A}\mathbf{B} - \mathbf{B}\mathbf{A}$ . In particular, on so(3) the Lie bracket of two elements corresponds to their vector product:  $[\omega_1, \omega_2] = \omega_1 \times \omega_2$ . On se(3) the Lie bracket of two elements  $(\omega_1, \mathbf{v}_1)$  and  $(\omega_2, \mathbf{v}_2)$  is given by

$$[(\omega_1, \mathbf{v}_1), (\omega_2, \mathbf{v}_2)] = (\omega_1 \times \omega_2, \omega_1 \times \mathbf{v}_2 - \omega_2 \times \mathbf{v}_1). \quad (1)$$

An element of a Lie group can also be identified with a linear mapping between its Lie algebra via the *adjoint* representation. Suppose G is a matrix Lie group with Lie algebra g. For every  $X \in G$  the adjoint map  $Ad_X : g \to g$  is defined by  $Ad_X(x) = XxX^{-1}$ . If  $X = (\Theta, b)$  is an element of SE(3), then its adjoint map acting on an element  $x = (\omega, v)$  of SE(3) is given by

$$Ad_{\mathbf{X}}(\mathbf{x}) = (\Theta\omega, \mathbf{b} \times \Theta\omega + \Theta\mathbf{v}), \tag{2}$$

which also admits the  $6 \times 6$  matrix representation

$$Ad_{\mathbf{X}}(\mathbf{x}) = \begin{bmatrix} \mathbf{\Theta} & 0 \\ [\mathbf{b}] \mathbf{\Theta} & \mathbf{\Theta} \end{bmatrix} \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}. \tag{3}$$

It is easily verified that  $Ad_{\mathbf{X}}^{-1} = Ad_{\mathbf{X}^{-1}}$  and  $Ad_{\mathbf{X}}Ad_{\mathbf{Y}} = Ad_{\mathbf{X}\mathbf{Y}}$  for any  $\mathbf{X}, \mathbf{Y} \in SE(3)$ . The dual operator  $Ad_{\mathbf{X}}^*$ :  $se(3)^* \to se(3)^*$  also has a matrix representation (with respect to the standard dual basis on se(3)) given by the transpose of  $Ad_{\mathbf{X}}$ : if  $\mathbf{z} = (\mathbf{m}, \mathbf{f})$  is an element of  $se(3)^*$ , then

$$Ad_{\mathbf{X}}^{*}(\mathbf{z}) = \begin{bmatrix} \mathbf{\Theta}^{T} & \mathbf{\Theta}^{T} [\mathbf{b}]^{T} \\ 0 & \mathbf{\Theta}^{T} \end{bmatrix} \begin{bmatrix} \mathbf{m} \\ \mathbf{f} \end{bmatrix}. \tag{4}$$

Elements of a Lie algebra can also be identified with a linear mapping between its Lie algebra via the Lie bracket. Given an element  $\mathbf{x} \in \mathbf{g}$ , its adjoint representation is the linear map  $\mathrm{ad}_{\mathbf{x}}: \mathbf{g} \to \mathbf{g}$  defined by  $\mathrm{ad}_{\mathbf{x}}(\mathbf{y}) = [\mathbf{x},\mathbf{y}]$ . If  $\mathbf{x} = (\omega_1,\mathbf{v}_1)$  and  $\mathbf{y} = (\omega_2,\mathbf{v}_2)$  are elements of se(3), then

$$ad_{\mathbf{x}}\mathbf{y} = (\boldsymbol{\omega}_1 \times \boldsymbol{\omega}_2, \boldsymbol{\omega}_1 \times \mathbf{v}_2 - \boldsymbol{\omega}_2 \times \mathbf{v}_1), \tag{5}$$

<sup>2.</sup> Portions of the Lagrangian dynamics formulation have been previously reported in Brockett et al. (1993).

which also admits the matrix representation

$$ad_{\mathbf{x}}(\mathbf{y}) = \begin{bmatrix} \boldsymbol{\omega}_1 \\ \mathbf{v}_1 \end{bmatrix} \begin{bmatrix} \boldsymbol{\omega}_1 \\ \mathbf{v}_2 \end{bmatrix}.$$
 (6)

Similarly, the matrix representation of the dual operator  $ad_x^* : se(3)^* \rightarrow se(3)^*$  is given by its transpose:

$$ad_{\mathbf{x}}^{*}(\mathbf{z}) = \begin{bmatrix} -[\omega_{1}] & -[\mathbf{v}_{1}] \\ 0 & -[\omega_{1}] \end{bmatrix} \begin{bmatrix} \mathbf{m} \\ \mathbf{f} \end{bmatrix}. \tag{7}$$

### 2.2. Generalized Velocities and Generalized Forces

Recall that there exist two natural ways in which the tangent vector  $\dot{\mathbf{X}}(t)$  of a curve  $\mathbf{X}(t) = (\boldsymbol{\Theta}(t), \mathbf{b}(t))$  in SE(3) can be identified with an element of se(3): if X(t) describes the motion of a rigid body relative to an inertial reference frame, then  $\dot{\mathbf{X}}\mathbf{X}^{-1} = (\dot{\mathbf{\Theta}}\mathbf{\Theta}^{-1}, \dot{\mathbf{b}} - \dot{\mathbf{\Theta}}\mathbf{\Theta}^{-1}\mathbf{b}),$ and  $\mathbf{X}^{-1}\dot{\mathbf{X}} = (\boldsymbol{\Theta}^{-1}\dot{\boldsymbol{\Theta}}, \boldsymbol{\Theta}^{-1}\dot{\mathbf{b}})$  are both elements of se(3). The latter is referred to as the body-fixed velocity representation of  $\dot{\mathbf{X}}$ , since  $\boldsymbol{\Theta}^{-1}\dot{\boldsymbol{\Theta}}$  and  $\boldsymbol{\Theta}^{-1}\dot{\mathbf{b}}$  are the angular and translational velocities of the rigid body relative to its body-fixed frame, respectively. By a similar argument we call  $\dot{\mathbf{X}}\mathbf{X}^{-1}$  the inertial velocity representation of  $\dot{\mathbf{X}}$ . One subtle and important difference in the interpretation of the inertial velocity representation is that, while  $\dot{\Theta}\Theta^{-1}$ is indeed the angular velocity of the rigid body relative to the inertial frame, the translational velocity relative to the inertial frame is not  $\dot{\mathbf{b}} - \dot{\mathbf{\Theta}} \mathbf{\Theta}^{-1} \mathbf{b}$ , but simply  $\dot{\mathbf{b}}$ . Also, observe that if X(t) undergoes a coordinate transformation of the form  $\mathbf{X}(t) \mapsto \mathbf{X}(t)\mathbf{T}$ , where  $\mathbf{T} \in SE(3)$  is constant, then its new body-fixed velocity representation is  $T^{-1}X^{-1}XT=\text{Ad}_{T^{-1}}(X^{-1}X).$ 

While generalized velocities can be regarded as elements of se(3), in contrast forces and moments are normally thought of as inhabiting its dual se(3)\*. This identification can be traced to the fundamental fact that forces (which behave as gradient-like quantities) transform differently under coordinate changes than velocities (which behave as tangent vectors). Another possible explanation involves virtual work; because work is the time-integral of force times velocity, or moment times angular velocity, one can associate forces with velocities and moments with angular velocities. Forces and moments can therefore be naturally thought of as belonging to the dual space of velocities and angular velocities, respectively. With this point of view, a moment-force pair  $(\mathbf{m}, \mathbf{f}) \in se(3)^*$  can be regarded as consisting of a skewsymmetric matrix and a three-vector, respectively; i.e.,

$$[\mathbf{m}] = \begin{bmatrix} 0 & -m_3 & m_2 \\ m_3 & 0 & -m_1 \\ -m_2 & m_1 & 0 \end{bmatrix}, \qquad \mathbf{f} = \begin{bmatrix} f_1 \\ f_2 \\ f_3 \end{bmatrix}. \tag{8}$$

Somewhat paradoxically, under a change of reference frame, forces transform as angular velocities, and moments as velocities. To illustrate, let  $(\mathbf{m}, \mathbf{f})$  be the moment-force pair applied to a rigid body about the origin of its body-fixed frame, expressed in body-fixed coordinates. If this frame is now relocated to another point on the rigid body, so that with respect to the inertial frame it undergoes a right translation by  $\mathbf{T} = (\boldsymbol{\Theta}, \mathbf{b})$ , then the moment-force pair applied about the origin of this new frame is given in the new body-fixed coordinates by

$$\begin{bmatrix} \mathbf{f'} \\ \mathbf{m'} \end{bmatrix} = \begin{bmatrix} \mathbf{\Theta} & 0 \\ [\mathbf{b}] \mathbf{\Theta} & \mathbf{\Theta} \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{f} \\ \mathbf{m} \end{bmatrix}, \tag{9}$$

which can also be written

$$\begin{bmatrix} \begin{bmatrix} \mathbf{f'} \end{bmatrix} & \mathbf{m'} \\ 0 & 0 \end{bmatrix} = Ad_{\mathbf{T}}^{-1} \begin{pmatrix} \begin{bmatrix} \mathbf{f} \end{bmatrix} & \mathbf{m} \\ 0 & 0 \end{bmatrix} . \tag{10}$$

Under a change of reference frame, therefore, forces transform as angular velocities, and moments as velocities. For this reason one will find, in the kinematics literature, forces expressed as skew-symmetric matrices and moments as vectors. Strictly speaking, however, the operator  $Ad_T$  is a mapping from se(3) to se(3) and should not be applied to forces and moments. A more mathematically consistent approach is to express  $\mathbf{m}'$  and  $\mathbf{f}'$  in terms of the dual adjoint operator  $Ad_T^*$ :

$$\begin{bmatrix} \mathbf{m}' \\ \mathbf{f}' \end{bmatrix} = \begin{bmatrix} \mathbf{\Theta}^T \mathbf{m} - \mathbf{\Theta}^T (\mathbf{b} \times \mathbf{f}) \\ \mathbf{\Theta}^T \mathbf{f} \end{bmatrix}$$

$$= \begin{bmatrix} \mathbf{\Theta}^T & \mathbf{\Theta}^T [\mathbf{b}]^T \\ 0 & \mathbf{\Theta}^T \end{bmatrix} \begin{bmatrix} \mathbf{m} \\ \mathbf{f} \end{bmatrix},$$
(11)

or  $(\mathbf{m}',\mathbf{f}')=\mathrm{Ad}_{\mathbf{T}}^*(\mathbf{m},\mathbf{f})$ . Thus, under a change of coordinates by some right translation  $\mathbf{T}\in SE(3)$  the generalized force  $(\mathbf{m},\mathbf{f})$  transforms according to  $\mathrm{Ad}_{\mathbf{T}}^*$ . Both physical and mathematical considerations therefore suggest that moments be arranged as skew-symmetric matrices, and forces as vectors, contrary to the usual convention in the kinematics literature.

### 2.3. Kinetic Energy as a Quadratic Form on se(3)

Given an inertial reference frame, the kinetic energy of a rigid body can be computed in terms of the angular velocity in the body-fixed frame, and the velocity of its center of mass with respect to the inertial frame. Let  $\mathbf{x}(t)$  denote the trajectory (with respect to the inertial frame) of a body-fixed frame attached to the rigid body's center of mass. Let  $\bar{\mathbf{I}}$  denote the inertia matrix of the rigid body with respect to its body-fixed frame, and m the mass. The angular and translational velocities of the moving body in terms of its body-fixed frame are then  $\mathbf{X}^{-1}\dot{\mathbf{X}} = (\boldsymbol{\omega}_b, \mathbf{v}_b)$ , and the kinetic energy T is  $\frac{1}{2}\boldsymbol{\omega}_b^T\bar{\mathbf{I}}\boldsymbol{\omega}_b + \frac{1}{2}m\mathbf{v}_b^T\mathbf{v}_b$ . The inertia matrix  $\bar{\mathbf{I}}$  and mass m define an inner product on

se(3), given by the quadratic form

$$\left[\begin{array}{cc} \ddot{\mathbf{I}} & 0 \\ 0 & m & \mathbf{1} \end{array}\right],$$

where 1 is the  $3 \times 3$  identity matrix.

## 2.4. The Product of Exponentials Formula

We now review the product of exponentials (POE) formula (Brockett 1983) for open kinematic chains. If a right-handed reference frame is fixed at the tip of each link of the chain, then the Euclidean transformation that describes the position and orientation of the ith frame in terms of the (i-1)st frame is  $f_{i-1,i} = e^{\mathbf{P}_i x_i} \mathbf{M}_i$ , where  $\mathbf{M}_i \in SE(3)$ ,  $\mathbf{P}_i \in se(3)$ , and  $x_i \in \Re$  is the joint variable, i = 1, 2, ..., n. The frame fixed at the tip is then related to that at the base by the product

$$f(x_1,\ldots,x_n) = e^{\mathbf{P}_1 x_1} \mathbf{M}_1 e^{\mathbf{P}_2 x_2} \mathbf{M}_2 \cdots e^{\mathbf{P}_n x_n} \mathbf{M}_n. \quad (12)$$

By repeatedly applying the identity  $\mathbf{M}^{-1}e^{\mathbf{P}}\mathbf{M}$  $e^{\mathbf{M}^{-1}\mathbf{P}\mathbf{M}}$ , f can be written

$$f(x_1, x_2, \dots, x_n) = e^{\mathbf{A}_1 x_1} e^{\mathbf{A}_2 x_2} \cdots e^{\mathbf{A}_n x_n} \mathbf{M},$$
 (13)

where  $A_1 = P_1, A_2 = M_1 P_2 M_1^{-1}, A_3 =$  $(\mathbf{M}_1\mathbf{M}_2)\mathbf{P}_3(\mathbf{M}_1\mathbf{M}_2)^{-1}$ , etc. Alternatively, f can also be rewritten as

$$f(x_1, x_2, \dots, x_n) = \mathbf{M}e^{\mathbf{B}_1 x_1}e^{\mathbf{B}_2 x_2} \cdots e^{\mathbf{B}_n x_n},$$
 (14)

where  $\mathbf{B}_i = \mathbf{M}^{-1} \mathbf{A}_i \mathbf{M}, i = 1, ..., n$ . The matrix exponentials in this formula can be easily computed from the following: let  $[\omega] \in so(3)$  such that  $\omega_1^2 + \omega_2^2 + \omega_3^2 = 1$ , and  $\mathbf{v} \in \mathbb{R}^3$ . Then for any  $\phi \in \mathbb{R}$ ,

$$\exp\left(\begin{bmatrix} \begin{bmatrix} \boldsymbol{\omega} \end{bmatrix} & \mathbf{v} \\ 0 & 0 \end{bmatrix} \phi\right) = \begin{bmatrix} \exp([\boldsymbol{\omega}]\phi) & \mathbf{b} \\ 0 & 1 \end{bmatrix}$$
 (15)

is an element of SE(3), where

$$\exp([\boldsymbol{\omega}]\phi) = \mathbf{I} + \sin\phi [\boldsymbol{\omega}] + (1 - \cos\phi) [\boldsymbol{\omega}]^2, \tag{16}$$

$$\mathbf{b} = (\phi \mathbf{I} + (1 - \cos \phi) [\boldsymbol{\omega}] + (\phi - \sin \phi) [\boldsymbol{\omega}]^2) \mathbf{v}.$$
(17)

See Murray et al. (1993) for a derivation and discussion of these formulas.

One of the attractive features of the POE formula is the compact expression for the Jacobian. If f describes the tip frame relative to the inertial frame, then recall that  $f^{-1}\dot{f}$  is an element of se(3) corresponding to the generalized velocity of the tip frame relative to itself.

Using equation (14) and the fact that  $(e^{\mathbf{A}x})^{-1} = e^{-\mathbf{A}x}$ , a direct calculation shows that

$$f^{-1}\dot{f} = \mathbf{B}_{n}\dot{x}_{n} + e^{-\mathbf{B}_{n}x_{n}}\mathbf{B}_{n-1}e^{\mathbf{B}_{n}x_{n}}\dot{x}_{n-1} + \dots$$
(18)  
+  $e^{-\mathbf{B}_{n}x_{n}}\cdots e^{-\mathbf{B}_{2}x_{2}}\mathbf{B}_{1}e^{\mathbf{B}_{2}x_{2}}\cdots e^{\mathbf{B}_{n}x_{n}}\dot{x}_{1},$ 

which can also be written

$$f^{-1}\dot{f} = \mathbf{M}^{-1} \left( \mathbf{A}_{n} \dot{x}_{n} + e^{-\mathbf{A}_{n} x_{n}} \mathbf{A}_{n-1} e^{\mathbf{A}_{n} x_{n}} \dot{x}_{n-1} + \ldots \right) \mathbf{M}$$

$$= \mathbf{Ad}_{\mathbf{M}^{-1}} (\mathbf{A}_{n} \dot{x}_{n} + e^{-\mathbf{A}_{n} x_{n}} \mathbf{A}_{n-1} e^{\mathbf{A}_{n} x_{n}} \dot{x}_{n-1} + \ldots).$$
(20)

These formulas turn out to be especially useful in our Lagrangian dynamics formulation.

# 3. A Recursive Formulation of Robot **Dynamics**

We now develop a recursive formulation of robot dynamics using the geometric framework constructed above. For purposes of computational efficiency, we find it advantageous to attach reference frames to each link at the joints and to express the various quantities associated with each link in terms of these local frames.<sup>3</sup> The general idea behind the recursive formulation is a two-step iteration process. In the first iteration the velocities and accelerations of each link are propagated from the base to the tip, each expressed in terms of local link frame coordinates. In the second iteration the forces and torques are propagated backward from the tip to the base, also expressed in terms of local frame coordinates. We adopt the following notation: for i = 1, 2, ..., n, let

- $f_{i-1,i} = \mathbf{M}_i e^{\mathbf{S}_i x_i}$  = the location of the link *i* frame relative to the link i-1 frame, where  $M_i \in SE(3)$ ,  $\mathbf{S}_i \in \text{se}(3)$ , and  $x_i \in \Re$ .
- $V_i$  = the six-dimensional generalized velocity of the link i frame, expressed in link i frame coordinates. If  $f_i = f_{0,1} f_{1,2} \cdots f_{i-1,i}$  describes the location of the link i frame relative to the inertial reference frame, then  $\mathbf{V}_i = f_i^{-1} \dot{f}_i$ .  $\mathbf{J}_i \in \Re^{6 \times 6}$  is a symmetric positive-definite matrix
- defined as

$$\mathbf{J}_{i} = \begin{bmatrix} \mathbf{I}_{i} - m_{i}[\mathbf{r}_{i}]^{2} & m_{i}[\mathbf{r}_{i}] \\ -m_{i}[\mathbf{r}_{i}] & m_{i} \cdot 1 \end{bmatrix}, \tag{21}$$

where (1)  $m_i$  is the mass of link i; (2)  $\mathbf{r}_i$  is the vector from the origin of the link i frame to the

<sup>3.</sup> Here we follow the convention of Featherstone (1987), which is computationally more efficient than Luh et al.'s (1980) convention of attaching reference frames to the center of mass of each link.

center of mass of link i, expressed in link i frame coordinates; (3)  $\mathbf{I}_i$  is the inertia matrix of link i about the center of mass, relative to a frame at the center of mass that is parallel to the link i frame. The inertia matrix about the link i frame is then  $\mathbf{I}_i - m_i[\mathbf{r}_i]^2$  (see, e.g., Greenwood [1965]).

- F<sub>i</sub> = the total generalized force transmitted from link i 1 to link i through joint i; the first three components of F<sub>i</sub> correspond to the moment vector.
- $\tau_i = \text{joint } i \text{ actuator torque.}$

The recursive formulation can now be written using our geometric definitions and notation as follows (see the Appendix for the derivation):

Initialization

$$\mathbf{V}_0 = \dot{\mathbf{V}}_0 = \mathbf{F}_{n+1} = 0 \tag{22}$$

• Forward recursion: for i = 1 to n do

$$f_{i-1,i} = \mathbf{M}_i e^{\mathbf{S}_i x_i} \tag{23}$$

$$\mathbf{V}_i = \mathrm{Ad}_{f_i^{-1}}(\mathbf{V}_{i-1}) + \mathbf{S}_i \dot{x}_i \tag{24}$$

$$\dot{\mathbf{V}}_{i} = \mathbf{S}_{i}\ddot{x}_{i} + \operatorname{Ad}_{f_{i-1,i}^{-1}}(\dot{\mathbf{V}}_{i-1}) 
+ \operatorname{ad}_{\operatorname{Ad}_{f_{i-1,i}^{-1}}(\mathbf{V}_{i-1})}(\mathbf{S}_{i}\dot{x}_{i})$$
(25)

• Backward recursion: for i = n to 1 do

$$\mathbf{F}_i = \operatorname{Ad}_{f_{i,i+1}^{-1}}^*(\mathbf{F}_{i+1}) + \mathbf{J}_i \dot{\mathbf{V}}_i - \operatorname{ad}_{\mathbf{V}_i}^*(\mathbf{J}_i \mathbf{V}_i)$$
 (26)

$$\tau_i = \mathbf{S}_i^T \mathbf{F}_i \tag{27}$$

Note that the forward recursion for velocities and accelerations is given solely in terms of the adjoint maps, whereas the backward recursion of forces and moments depends only on the dual adjoints. The algorithm bears a close resemblance to that of Featherstone (1987). In particular, his spatial vectors representing the joint axes are immediately identifiable with the  $S_i$ s in our expression for  $f_{i-1,i}$ , and his spatial cross-product is simply the Lie bracket (or adjoint) on se(3). A few differences are also apparent: Featherstone's spatial inertia matrices are arranged in the somewhat unorthodox form

$$\begin{bmatrix} -m_i[\mathbf{r}_i] & m_i \cdot 1 \\ \mathbf{I}_i - m_i[\mathbf{r}_i]^2 & m_i[\mathbf{r}_i] \end{bmatrix}, \tag{28}$$

and his spatial inner product  $\hat{\cdot}$ , defined as  $(\mathbf{a}, \mathbf{b}) \hat{\cdot} (\mathbf{c}, \mathbf{d}) = \mathbf{a}^T \mathbf{d} + \mathbf{b}^T \mathbf{c}$ , is replaced by the ordinary inner product in our formulation. Featherstone estimates that his algorithm requires, for a general n-link manipulator, 130n - 68 scalar multiplications and 101n - 56 scalar additions. Since our algorithm is essentially a geometric version of Featherstone's, the computational requirements should

be similar. Li's formulation does not introduce the ad\* operator; consequently the generalized velocity  $(\omega_i, \mathbf{v}_i)$  is decomposed into its angular and translational components, which complicates his backward recursion.

# 4. A Lagrangian Formulation of Robot Dynamics

In the Lagrangian formulation the dynamic equations can be written as

$$\tau_k = \sum_{j=1}^n m_{kj}(x) \dot{x}_j + \sum_{i=1}^n \sum_{j=1}^n \Gamma_{ijk}(x) \dot{x}_i \dot{x}_j + \phi_k(x),$$

$$k = 1, \dots, n \qquad (29)$$

where  $\tau_k$  is the applied force/torque at joint k,  $m_{kj}$  are elements of the inertia matrix,  $\Gamma_{ijk}$  are the Christoffel symbols of the first kind relative to the inertia matrix representing the centrifugal and Coriolis terms, and  $\phi_k$  are the forces due to gravity (see, e.g., Spong and Vidyasagar [1989]). We now derive each of these terms using our geometric formulation.

First, let the forward kinematics of a general n-link open chain containing either revolute or prismatic joints be given as in (13):

$$f(x_1,\ldots,x_n) = e^{\mathbf{A}_1 x_1} e^{\mathbf{A}_2 x_2} \cdots e^{\mathbf{A}_n x_n} \mathbf{M},$$
 (30)

where  $\mathbf{A}_1,\ldots,\mathbf{A}_n\in\operatorname{se}(3),\,x_1,\ldots,x_n$  are the joint variables, and  $\mathbf{M}\in\operatorname{SE}(3)$ . We make the following definitions. With the joints in their zero position, let  $\mathbf{M}_i=(\Theta_i,\mathbf{b}_i)$  be an element of  $\operatorname{SE}(3)$  denoting the body-fixed frame attached to the center of mass of link i, expressed relative to the inertial frame. Let  $m_i$  denote the mass of link i,  $\bar{\mathbf{I}}_i$  the  $3\times 3$  inertia matrix of link i relative to its body-fixed frame, and  $\mathbf{I}_i$  the inertia matrix of link i with respect to the inertial frame; observe that  $\mathbf{I}_i$  is defined differently from the previous section. Note further that  $\mathbf{I}_i$  and  $\bar{\mathbf{I}}_i$  are related by  $\mathbf{I}_i = \Theta_i \bar{\mathbf{I}}_i \Theta_i^T - m_i [\mathbf{b}_i]^2$ . The map  $f_i$  defined by

$$f_i(x_1, \dots, x_i) = e^{\mathbf{A}_1 x_1} e^{\mathbf{A}_2 x_2} \dots e^{\mathbf{A}_i x_i}, \mathbf{M}_i$$
 (31)

i = 1, 2, ..., n, then describes the center-of-mass frame of link i relative to the inertial frame, as a function of the joint variables  $x_1, ..., x_i$ .

### 4.1. The Inertia Matrix

The first step in the Lagrangian formulation is to derive the total kinetic energy T of the system. In terms of the inertia matrix T can be expressed as

$$T = \sum_{i,j} m_{ij}(x)\dot{x}_i\dot{x}_j. \tag{32}$$

If  $T_i$  denotes the kinetic energy of link i, then  $T = \sum T_i$ . From the prior definitions the body-fixed generalized velocity of link i's center of mass is  $f_i^{-1}\dot{f}_i=(\bar{\omega}_i,\bar{\mathbf{v}}_i)$ , and  $T_i=\frac{1}{2}\bar{\omega}_i^T\bar{\mathbf{l}}_i\bar{\omega}_i+\frac{1}{2}m_i\bar{\mathbf{v}}_i^T\bar{\mathbf{v}}_i$ . In terms of the joint rates  $f_i^{-1}\dot{f}_i$  can be expressed as

$$f_{i}^{-1}\dot{f}_{i} = \operatorname{Ad}_{\mathbf{M}_{i}^{-1}} \left( \mathbf{A}_{i}\dot{x}_{i} + \operatorname{Ad}_{e-\mathbf{A}_{i}x_{i}} (\mathbf{A}_{i-1}\dot{x}_{i-1}) + \dots \right.$$

$$+ \operatorname{Ad}_{e-\mathbf{A}_{i}x_{i}\dots e^{-\mathbf{A}_{2}x_{2}}} (\mathbf{A}_{1}\dot{x}_{1}) \right).$$
(33)

If the quantity in parentheses,  $\mathbf{A}_i \dot{x}_i + \mathrm{Ad}_{e^-\mathbf{A}_i x_i} (\mathbf{A}_{i-1} \dot{x}_{i-1}) + \ldots$ , is denoted by  $(\boldsymbol{\omega}_i, \mathbf{v}_i)$ , then  $\mathbf{f}_i^{-1} \dot{\mathbf{f}}_i = \mathrm{Ad}_{\mathbf{M}_i^{-1}} (\boldsymbol{\omega}_i, bv_i)$ . In vector notation  $f_i^{-1} \dot{f}_i = (\hat{\boldsymbol{\omega}}_i, \hat{\mathbf{v}}_i)$  can now be expressed as

$$\begin{bmatrix} \bar{\boldsymbol{\omega}}_i \\ \bar{\mathbf{v}}_i \end{bmatrix} = \begin{bmatrix} \boldsymbol{\Theta}_i^T & \mathbf{0} \\ -[\boldsymbol{\Theta}_i^T \mathbf{b}_i] \boldsymbol{\Theta}_i^T & \boldsymbol{\Theta}_i^T \end{bmatrix} \begin{bmatrix} \boldsymbol{\omega}_i \\ \mathbf{v}_i \end{bmatrix}. \tag{34}$$

By applying the identity  $\Theta[\mathbf{b}]\Theta^T = [\Theta \mathbf{b}]$  for  $\Theta \in SO(3)$  and  $[\mathbf{b}] \in so(3)$ , the kinetic energy  $T_i$  of link i can be expressed in terms of  $(\omega_i, \mathbf{v}_i)$  as

$$T_{i} = \frac{1}{2} \begin{bmatrix} \boldsymbol{\omega}_{i}^{T} & \mathbf{v}_{i}^{T} \end{bmatrix} \begin{bmatrix} \mathbf{I}_{i} & m_{i}[\mathbf{b}_{i}] \\ m_{i}[\mathbf{b}_{i}]^{T} & m_{i} \cdot 1 \end{bmatrix} \begin{bmatrix} \boldsymbol{\omega}_{i} \\ \mathbf{v}_{i} \end{bmatrix}, \quad (35)$$

where 1 is the  $3 \times 3$  identity matrix and  $I_i$  is the inertia matrix of link i relative to the inertial frame. This quadratic form is used to define an inner product on se(3):

DEFINITION 1. Let  $\langle \cdot, \cdot \rangle_i$  be an inner product on se(3) defined by the quadratic form

$$\begin{bmatrix} \mathbf{I}_i & m_i[\mathbf{b}_i] \\ m_i[\mathbf{b}_i]^T & m_i \cdot \mathbf{1} \end{bmatrix}. \tag{36}$$

 $T_i$  can therefore be written as  $\langle (\omega_i, \mathbf{v}_i), (\omega_i, \mathbf{v}_i) \rangle_i$ . We also adopt the following notation:

DEFINITION 2. Given  $\mathbf{A}_1,\ldots,\mathbf{A}_n\in se(3)$  and  $x_1,\ldots,x_n\in\Re$ , define the map  $\mathrm{Ad}_j^i:se(3)\to se(3)$  by

$$Ad_{j}^{i}(\mathbf{H}) = \begin{cases} e^{-\mathbf{A}_{j}x_{j}} \cdots e^{-\mathbf{A}_{i}x_{i}} \mathbf{H} e^{\mathbf{A}_{i}x_{i}} \cdots e^{\mathbf{A}_{j}x_{j}}, & i \leq j \\ \mathbf{H}, & i > j \end{cases}$$
(37)

Note that by this definition  $Ad_j^j(\mathbf{A}_j) = \mathbf{A}_j$ . Using this notation  $(\boldsymbol{\omega}_i, \mathbf{v}_i)$  from above can be written in terms of the kinematic parameters as

$$(\boldsymbol{\omega}_i, \mathbf{v}_i) = \sum_{j=1}^i \mathrm{Ad}_i^{j+1}(\mathbf{A}_j). \tag{38}$$

The kinetic energy of link i is then

$$T_{i} = \langle \sum_{j=1}^{i} Ad_{i}^{j+1}(\mathbf{A}_{j}\dot{x}_{j}), \sum_{j=1}^{i} Ad_{i}^{j+1}(\mathbf{A}_{j}\dot{x}_{j}) \rangle_{i}$$
 (39)

and  $T = \sum T_i$ . Equating T with  $\sum \sum m_{ij}(x)\dot{x}_i\dot{x}_j$  and matching terms, the components of the inertia matrix are given by

$$m_{ij} = \sum_{k=i}^{n} \langle \operatorname{Ad}_{k}^{i+1}(\mathbf{A}_{i}), \operatorname{Ad}_{k}^{j+1}(\mathbf{A}_{j}) \rangle_{k}, \tag{40}$$

for  $j \geq i$ , and  $m_{ii} = m_{ij}$ .

We now express the inertia matrix in block-matrix form. Let  $\mathcal{A}\in\Re^{6n\times n}$  be a block-diagonal matrix of the form

$$\mathcal{A} = \begin{bmatrix} \mathcal{A}_1 & 0 & \dots & 0 \\ 0 & \mathcal{A}_2 & \dots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \dots & \dots & \mathcal{A}_n \end{bmatrix}$$
(41)

where  $A_i \in \Re^6$  is the six-dimensional column vector representation of  $\mathbf{A}_i \in \operatorname{se}(3)$ . Also, let  $(\operatorname{Ad}_j^i)$  denote the  $6 \times 6$  matrix representation of the map  $\operatorname{Ad}_j^i : \operatorname{se}(3) \to \operatorname{se}(3)$ ; that is,

$$(\mathbf{Ad}_{j}^{i}) = \begin{bmatrix} \mathbf{\Theta} & 0 \\ [\mathbf{b}] \mathbf{\Theta} & \mathbf{\Theta} \end{bmatrix},$$

$$\begin{bmatrix} \mathbf{\Theta} & \mathbf{b} \\ 0 & 1 \end{bmatrix} = e^{-\mathbf{A}_{j}x_{j}}e^{-\mathbf{A}_{j-1}x_{j-1}} \cdots e^{-\mathbf{A}_{i}x_{i}}.$$

$$(42)$$

Then define  $\mathcal{L} \in \Re^{6n \times 6n}$  to be the lower block-triangular matrix

$$\mathcal{L} = \begin{bmatrix} \mathbf{1} & 0 & 0 & \dots & 0 \\ (Ad_2^2) & \mathbf{1} & 0 & \dots & 0 \\ (Ad_3^2) & (Ad_3^3) & \mathbf{1} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ (Ad_n^2) & (Ad_n^3) & (Ad_n^4) & \dots & \mathbf{1} \end{bmatrix}$$
(43)

and  $\mathcal{D}_i \in \Re^{6 \times 6}$  to be the kinetic energy quadratic form associated with link i:

$$\mathcal{D}_i = \begin{bmatrix} \mathbf{I}_i & m_i[\mathbf{b}_i] \\ m_i[\mathbf{b}_i]^T & m_i \cdot \mathbf{1} \end{bmatrix}.$$
 (44)

Observe that  $\mathcal{D}_i$  is a constant matrix with exactly 10 independent parameters. Let  $\mathcal{D} \in \Re^{6n \times 6n}$  be the block-diagonal matrix

$$\mathcal{D} = \begin{bmatrix} \mathcal{D}_1 & 0 & \dots & 0 \\ 0 & \mathcal{D}_2 & \dots & 0 \\ \vdots & & \ddots & \vdots \\ \mathbf{0} & \dots & \dots & \mathcal{D}_n \end{bmatrix}. \tag{45}$$

The inertia matrix  $\mathbf{M}(x) \in \Re^{n \times n}$  is then given by

$$\mathbf{M} = \mathcal{A}^T \mathcal{L}^T \mathcal{D} \mathcal{L} \mathcal{A}. \tag{46}$$

Note that all the inertial parameters are contained in the constant block-diagonal matrix  $\mathcal{D}$ , whereas  $\mathcal{A}$  is a constant matrix containing only the kinematic parameters, and  $\mathcal{L}$  is the only matrix depending on the joint values. This inertia matrix factorization clearly separates the kinematic and dynamic parameters and is similar to the factorization obtained by Rodriguez and Kreutz-Delgado (1992). However, the kinematic parameters appear explicitly as matrix exponentials in the  $\mathcal{L}$  matrix above; in the factorization of Rodriguez the ijth block entry of the  $\mathcal{L}$  matrix is of the form

$$\begin{bmatrix} \mathbf{1} & 0 \\ [\mathbf{p}_{ij}] & \mathbf{1} \end{bmatrix},$$

where  $\mathbf{p}_{ij}$  is the vector from the origin of link frame i to the origin of link frame j. Li's (1989) formulation does not provide a factorization, and the inertia terms are more complex, because the forward kinematics are not expressed as a strict product of exponentials.

The factorization of M in (46) is related to the vector and matrix quantities appearing in the recursive Newton-Euler formulation by the following formulas: define

$$S = \text{Diag}[\mathbf{S}_1, \mathbf{S}_2, \dots \mathbf{S}_n] \in \Re^{6n \times n} \tag{47}$$

$$\mathcal{J} = \text{Diag}[\mathbf{J}_1, \mathbf{J}_2, \dots \mathbf{J}_n] \in \Re^{6n \times 6n}$$
 (48)

$$\mathcal{G} = \begin{bmatrix} \mathbf{1} & 0 \cdots & 0 \\ Ad_{f_{1,2}^{-1}} & \mathbf{1} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ Ad_{f_{1,n}^{-1}} & Ad_{f_{2,n}^{-1}} & \cdots & \mathbf{1} \end{bmatrix}$$
(49)

$$Q = \text{Diag}[Ad_{\mathbf{M}_1}, Ad_{\mathbf{M}_2}, \dots, Ad_{\mathbf{M}_1 \dots \mathbf{M}_n}].$$
 (50)

Then M can be factored as

$$\mathbf{M} = \mathcal{S}^T \mathcal{G}^T \mathcal{J} \mathcal{G} \mathcal{S} \tag{51}$$

where A = QS,  $L = QQQ^{-1}$ , and  $D = Q^{-T}JQ^{-1}$ .

#### 4.2. Coriolis Terms

The Coriolis terms of the dynamic equations involve computation of  $\Gamma_{ijk}$ , the Christoffel symbols of the first kind relative to the metric defined by the inertia matrix:

$$\Gamma_{ijk} = \frac{1}{2} \left( \frac{\partial m_{kj}}{\partial x_i} + \frac{\partial m_{ki}}{\partial x_j} - \frac{\partial m_{ij}}{\partial x_k} \right)$$
 (52)

and  $\Gamma_{ijk} = \Gamma_{jik}$ . The following result is useful for computing the Christoffel symbols:

PROPOSITION 1. Given the map  $\operatorname{Ad}_{j}^{i}: se(3) \to se(3)$  as in Definition 2,

$$\frac{\partial}{\partial x_k} Ad_j^i(\mathbf{B}) = \begin{cases} Ad_j^k([Ad_{k-1}^i(\mathbf{B}), \mathbf{A}_k]), & i \le k \le j \\ 0 & \text{otherwise} \end{cases}$$
(53)

Here  $[\cdot, \cdot]$  denotes the Lie bracket on se(3). The following result, while not directly used in computing the Coriolis terms, is useful for, e.g., linearizing the dynamics or computing the curvature tensor:

PROPOSITION 2. Given the map  $\mathrm{Ad}_j^i: se(3) \to se(3)$  as in Definition 2, suppose  $i \leq j$ . Then

$$\begin{split} \frac{\partial}{\partial x_{l}} \frac{\partial}{\partial x_{k}} \mathrm{Ad}_{j}^{i}(\mathbf{B}) &= \\ &\left\{ \begin{array}{l} \mathrm{Ad}_{j}^{l} \left( [\mathrm{Ad}_{l-1}^{k}([\mathrm{Ad}_{k-1}^{i}(\mathbf{B}), \mathbf{A}_{k}]), \mathbf{A}_{l}] \right), & i \leq l \leq k \leq j \\ \mathrm{Ad}_{j}^{k} \left( [\mathrm{Ad}_{k-1}^{l}([\mathrm{Ad}_{l-1}^{i}(\mathbf{B}), \mathbf{A}_{l}]), \mathbf{A}_{k}] \right), & i \leq k \leq l \leq j \\ 0 & \text{otherwise} \\ \end{array} \right. \end{split}$$

Using these formulas, the Christoffel symbols are calculated to be the following:

• Case 1:  $k \le i \le j$ :

$$\Gamma_{ijk} = \frac{1}{2} \sum_{l=j}^{n} \langle \operatorname{Ad}_{l}^{i}([\operatorname{Ad}_{i-1}^{k+1}(\mathbf{A}_{k}), \mathbf{A}_{i}]), \operatorname{Ad}_{l}^{j+1}(\mathbf{A}_{j}) \rangle_{l}$$

$$+ \langle \operatorname{Ad}_{l}^{j}([\operatorname{Ad}_{j-1}^{k+1}(\mathbf{A}_{k}), \mathbf{A}_{j}]), \operatorname{Ad}_{l}^{i+1}(\mathbf{A}_{i}) \rangle_{l}$$

$$+ \langle \operatorname{Ad}_{l}^{j}([\operatorname{Ad}_{j-1}^{i+1}(\mathbf{A}_{i}), \mathbf{A}_{j}]), \operatorname{Ad}_{l}^{k+1}(\mathbf{A}_{k}) \rangle_{l}$$
(55)

• Case 2:  $i < k \le j$ :

$$\Gamma_{ijk} = \frac{1}{2} \sum_{l=j}^{n} \langle \operatorname{Ad}_{l}^{j}([\operatorname{Ad}_{j-1}^{k+1}(\mathbf{A}_{k}), \mathbf{A}_{j}]), \operatorname{Ad}_{l}^{i+1}(\mathbf{A}_{i}) \rangle_{l}$$

$$+ \langle \operatorname{Ad}_{l}^{j}([\operatorname{Ad}_{j-1}^{i+1}(\mathbf{A}_{i}), \mathbf{A}_{j}]), \operatorname{Ad}_{l}^{k+1}(\mathbf{A}_{k}) \rangle_{l}$$

$$- \langle \operatorname{Ad}_{l}^{k}([\operatorname{Ad}_{k-1}^{i+1}(\mathbf{A}_{i}), \mathbf{A}_{k}]), \operatorname{Ad}_{l}^{j+1}(\mathbf{A}_{j}) \rangle_{l}$$
(56)

• Case 3:  $i \le j < k$ :

$$\Gamma_{ijk} = \frac{1}{2} \sum_{l=k}^{n} \langle \operatorname{Ad}_{l}^{j}([\operatorname{Ad}_{j-1}^{i+1}(\mathbf{A}_{i}), \mathbf{A}_{j}]), \operatorname{Ad}_{l}^{k+1}(\mathbf{A}_{k}) \rangle_{l}$$

$$- \langle \operatorname{Ad}_{l}^{k}([\operatorname{Ad}_{k-1}^{i+1}(\mathbf{A}_{i}), \mathbf{A}_{k}]), \operatorname{Ad}_{l}^{j+1}(\mathbf{A}_{j}) \rangle_{l}$$

$$- \langle \operatorname{Ad}_{l}^{k}([\operatorname{Ad}_{k-1}^{j+1}(\mathbf{A}_{j}), \mathbf{A}_{k}]), \operatorname{Ad}_{l}^{i+1}(\mathbf{A}_{i}) \rangle_{l}$$
(57)

#### 4.3. Potential Terms

In what follows we assume the only source of potential energy is gravity. Let  $\mathbf{g}$  denote the gravity vector expressed in the inertial reference frame. Recall that  $f_i = e^{\mathbf{A}_i x_1} \cdots e^{\mathbf{A}_i x_i} \mathbf{M}_i$  describes the center-of-mass frame of link i relative to the inertial frame. Define the constant vector  $\mathbf{r} = [0\ 0\ 0\ 1\ ]^T$ . The Cartesian position of the center of mass of link i in inertial frame coordinates is then  $f_i \mathbf{r}$ . The potential energy of link i, denoted  $U_i$ , is  $\langle m_i \mathbf{g}, f_i \mathbf{r} \rangle$ , where  $\langle \cdot, \cdot \rangle$  is the standard Euclidean norm. The total potential energy of the manipulator is then

$$U = \sum_{i=1}^{n} \langle m_i \mathbf{g}, f_i \mathbf{r} \rangle. \tag{58}$$

The potential term in the equations of motion (29) is therefore given by

$$\phi_k = \frac{\partial}{\partial x_k} U.$$

After some manipulation, the potential term can be expressed as

$$\phi_k = \sum_{i=k}^{n} \langle m_i \mathbf{g}, f_i \cdot \mathbf{M}_i^{-1} \mathbf{A} \mathbf{d}_i^{k+1} (\mathbf{A}_k) \mathbf{M}_i \mathbf{r} \rangle$$
 (59)

for  $1 \le k \le i$ .

### 4.3.1. Example

The dynamic equations for a general open chain manipulator, ignoring gravity, can be written as  $\tau = \mathbf{M}(x)\ddot{x} + \mathbf{C}(x,\dot{x})\dot{x}$ . For a general 2R manipulator,

$$\mathbf{M}(x) = \begin{cases} \langle \mathbf{A}_1, \mathbf{A}_1 \rangle_1 + \langle \mathbf{A} \mathbf{d}_2^2(\mathbf{A}_1), \mathbf{A} \mathbf{d}_2^2(\mathbf{A}_1) \rangle_2 & \langle \mathbf{A} \mathbf{d}_2^2(\mathbf{A}_1), \mathbf{A}_2 \rangle_2 \\ \langle \mathbf{A} \mathbf{d}_2^2(\mathbf{A}_1), \mathbf{A}_2 \rangle_2 & \langle \mathbf{A}_2, \mathbf{A}_2 \rangle_2 \end{cases}$$

the  $C(x, \dot{x})$  matrix is as follows:

$$c_{11} = \langle \mathrm{Ad}_2^2([\mathbf{A}_1, \mathbf{A}_2]), \mathrm{Ad}_2^2(\mathbf{A}_1) \rangle_2 \, \dot{x}_1 \dot{x}_2$$
 (61)

$$c_{12} = \langle Ad_2^2([\mathbf{A}_1, \mathbf{A}_2]), Ad_2^2(\mathbf{A}_1) \rangle_2 \, \dot{x}_1 \dot{x}_2$$

$$+ \langle Ad_2^2([\mathbf{A}_1, \mathbf{A}_2]), \mathbf{A}_2 \rangle_2 \, \dot{x}_2^2$$
(62)

$$c_{21} = \langle Ad_2^2([\mathbf{A}_1, \mathbf{A}_2]), Ad_2^2(\mathbf{A}_1) \rangle_2 \dot{x}_1^2$$
 (63)

$$c_{22} = 0$$
 (64)

# 4.4. A Coordinate-Free Interpretation of the Inertia Matrix

One of the additional insights acquired from the geometric Lagrangian formulation is that the inertia matrix

can be interpreted as the pullback of a certain Riemannian metric. We illustrate this for an all revolute joint manipulator. Let  $f_i: T^i \to SE(3)$  be the kinematic map to the center of mass of link i as in (31), where  $T^i$  is the i-dimensional torus, and define a map  $f: T^n \to SE(3) \times SE(3) \times \cdots \times SE(3)$  (n copies) by

$$f(x) = (f_1(x), f_2(x), \dots, f_n(x)).$$
 (65)

Define a Riemannian metric on the range by  $g = g_1 \otimes g_2 \otimes \cdots \otimes g_n$ , where each  $g_i$  is the metric defined by the generalized inertia tensor of link i. The inertia matrix of the robot is then the pullback metric  $f^*g$ , which admits the coordinate representation  $\mathbf{J}^T\mathbf{G}\mathbf{J}$ , where  $\mathbf{J}$  is the Jacobian of the map f. In our factorization  $\mathbf{G}$  can be taken to be the block-diagonal matrix  $\mathcal{D}$ , and from the product-of-exponentials formula,  $\mathbf{J}$  can be factored as  $\mathcal{L}(x)\mathcal{A}$ .

### 5. Conclusions

Using methods of Lie groups and Riemannian geometry, two new formulations of the Newton-Euler and Lagrangian dynamics equations for robots have been presented. A principal advantage of this approach is that a high-level description of robot dynamics can now be obtained: explicit connections between the equations of motion and standard concepts from Lie theory are established, avoiding the need for ad hoc definitions and choices of notation.

The recursive Newton-Euler formulation provides a computationally efficient O(n) algorithm for computing the inverse dynamics. The Lagrangian formulation provides a simple closed-form set of equations of motion that are particularly useful for applications in robot design and control. Spong (1992) shows that if a manipulator's inertia matrix has vanishing Riemannian curvature, then there exist a set of coordinates in which the dynamic equations assume a particularly simple form. The curvature of the inertia matrix can in turn be regarded as a measure of the dynamic response of the robot, providing information about the sensitivity of the dynamics equations to certain robot parameters. A reasonable argument can be made that minimum curvature is a useful measure of robot performance. The explicit formula for the curvature tensor is given by

$$R_{ijkl} = \frac{\partial \Gamma_{jli}}{\partial x_k} - \frac{\partial \Gamma_{jki}}{\partial x_l} + \sum_{r=1}^{n} (\Gamma_{ilr} \Gamma_{jk}^r - \Gamma_{ikr} \Gamma_{jl}^r), \quad (66)$$

where  $\Gamma_{ij}^k = \sum_{h=1}^n m^{kh} \Gamma_{ijh}$  and  $m^{ij}$  are the components of the inverse of the inertia matrix. With our formulas for the derivatives of the Christoffel symbols, the curvature can now be computed directly. Motion

optimization applications can also benefit: minimum energy paths, for example, have the objective function  $J(x) = \int \sum \sum m_{ij}(x)\dot{x}_i\dot{x}_j\ dt$ , and solutions can be found much more robustly and efficiently if the gradient of J(x) (which involves derivatives of  $m_{ij}(x)$ ) is available.

The derivative formulas are also useful for linearizing the equations of motion about a nominal trajectory—many feedback controllers are based on such a set of linearized equations, for example. Most adaptive controllers also update the inertial parameters of a robot, and in many cases identifying a minimal parameter set can significantly improve their efficiency. In our Lagrangian formulation the inertial parameters appear linearly in an explicit way, so that they can be easily factored from the general equations of motion. Finally, it should be possible to extend the geometric formulation to treat general multibody systems with closed chains and flexible links, similar to the ideas explored in the spatial operator algebra formulation of Rodriguez et al. (1991).

## **Appendix**

We derive here the recursive Newton-Euler dynamics algorithm presented in Section 3. We begin with the forward iteration of velocities and accelerations. Let the transformation from frame i-1 to frame i be given by  $f_{i-1,i} = \mathbf{M}_i e^{\mathbf{S}_i x_i}$ , where  $\mathbf{M}_i \in \text{SE}(3), \mathbf{S}_i \in \text{se}(3)$ , and  $x_i$  is the scalar joint variable. It is trivial to verify that  $f_{i-1,i}^{-1} \dot{f}_{i-1,i} = \mathbf{S}_i \dot{x}_i$ . Now  $f_i$ , which is the link i reference frame relative to the inertial frame, is

$$f_i = f_{0,1} f_{1,2} \cdots f_{i-1,i}.$$
 (67)

Let  $V_i = f_i^{-1} \dot{f_i}$  be the six-dimensional generalized velocity of the link i frame, expressed in link i frame coordinates. Then

$$\mathbf{V}_{i} = (f_{i-1}f_{i-1,i})^{-1} \frac{d}{dt} (f_{i-1}f_{i-1,i})$$
(68)

$$= f_{i-1,i}^{-1}(f_{i-1}^{-1}\dot{f}_{i-1})f_{i-1,i} + f_{i-1,i}^{-1}\dot{f}_{i-1,i}$$
 (69)

$$= \mathrm{Ad}_{f_{i-1,i}^{-1}}(\mathbf{V}_{i-1}) + \mathbf{S}_i \dot{x}_i \tag{70}$$

as claimed. To find the generalized acceleration  $\dot{\mathbf{V}}_i$ , observe first that

$$\frac{d}{dt} \operatorname{Ad}_{f_{i-1,i}^{-1}}(\mathbf{V}_{i-1}) = \frac{d}{dt} (f_{i-1,i}^{-1} \mathbf{V}_{i-1} f_{i-1,i})$$
(71)

$$= \dot{f}_{i-1,i}^{-1} \mathbf{V}_{i-1} f_{i-1,i} \tag{72}$$

$$+ f_{i-1,i}^{-1} \dot{\mathbf{V}}_{i-1} f_{i-1,i} + f_{i-1,i}^{-1} \mathbf{V}_{i-1} \dot{f}_{i-1,i}.$$

Applying the identity  $\dot{f}_{i-1,i}^{-1}=-f_{i-1,i}^{-1}\dot{f}_{i-1,i}f_{i-1,i}^{-1}$ ,  $\dot{\mathbf{V}}_i$  can now be simplified to

$$\dot{\mathbf{V}}_{i} = \mathrm{Ad}_{f_{i-1,i}^{-1}}(\dot{\mathbf{V}}_{i-1}) + [\mathrm{Ad}_{f_{i-1,i}^{-1}}(\mathbf{V}_{i}), \mathbf{S}_{i}\dot{x}_{i}] + \mathbf{S}_{i}\ddot{x}_{i}, \quad (73)$$

which agrees with the recursive formula for the acceleration.

We now derive the backward recursive algorithm for the forces and torques. The following three-dimensional vectors are all expressed in terms of local frame coordinates:  $\mathbf{v}_i$  = velocity of the link i frame,  $\omega_i$  = angular velocity of the link i frame,  $\mathbf{a}_i$  = acceleration of the link i frame,  $\bar{\mathbf{a}}_i$  = acceleration of link i's center of mass,  $\mathbf{r}_i$  = vector from origin of link i frame to center of mass of link i,  $\mathbf{f}_i$  = resultant force applied to link i,  $\mathbf{m}_i$  = resultant moment about the origin of the link i frame. In addition, let  $m_i = \text{mass of link } i$ , and  $\mathbf{I}_i = \text{inertia ma}$ trix of link i about the center of mass, relative to a frame at the center of mass that is parallel to the link i frame. We now show that the standard equations of motion for a rigid body (as derived in, say, Greenwood [1965]) are equivalent to our recursive formulation. The equations of motion for link i are

$$\mathbf{f}_i = m_i \tilde{\mathbf{a}}_i \tag{74}$$

$$\mathbf{m}_i = \mathbf{I}_i \dot{\boldsymbol{\omega}}_i + \boldsymbol{\omega}_i \times \mathbf{I}_i \boldsymbol{\omega}_i + \mathbf{r}_i \times m_i \tilde{\mathbf{a}}_i, \tag{75}$$

where

$$\mathbf{a}_i = \dot{\mathbf{v}}_i + \boldsymbol{\omega}_i \times \mathbf{v}_i \tag{76}$$

$$\bar{\mathbf{a}}_i = \mathbf{a}_i + \dot{\boldsymbol{\omega}}_i \times \mathbf{r}_i + \boldsymbol{\omega}_i \times (\boldsymbol{\omega}_i \times \mathbf{r}_i). \tag{77}$$

Note that the moments are summed about the origin of the link i frame rather than the center of mass of link i. Writing the cross-product of two vectors  $\mathbf{u} \times \mathbf{v}$  as the matrix-vector product  $[\mathbf{u}]\mathbf{v}$ , the equations of motion become

$$\mathbf{f}_i = m_i \left( \dot{\mathbf{v}}_i + [\omega_i] \mathbf{v}_i + [\dot{\omega}_i] \mathbf{r}_i + [\omega_i]^2 \mathbf{r}_i \right)$$
(78)

$$\mathbf{m}_{i} = \mathbf{I}_{i}\dot{\boldsymbol{\omega}}_{i} + \boldsymbol{\omega}_{i} \times \mathbf{I}_{i}\boldsymbol{\omega}_{i} + m_{i}[\mathbf{r}_{i}]\dot{\mathbf{v}}_{i} + m_{i}[\mathbf{r}_{i}][\boldsymbol{\omega}_{i}]\mathbf{v}_{i} + m_{i}[\mathbf{r}_{i}][\dot{\boldsymbol{\omega}}_{i}]\mathbf{r}_{i} + m_{i}[\mathbf{r}_{i}][\boldsymbol{\omega}_{i}]^{2}\mathbf{r}_{i}.$$
(79)

We now write out the resulting translational and rotational equations of motion from our geometric formulation, and show that they are equivalent to equations (78) and (79). The six-dimensional generalized force vector  $\mathbf{F}_i = (\mathbf{m}_i, \mathbf{f}_i)$ . By definition  $\mathbf{F}_{i+1}$  is the force/moment exerted by link i on link i+1, expressed in link i+1 frame coordinates. Then  $\mathrm{Ad}_{f_{i,i+1}}^*(\mathbf{F}_{i+1})$  is  $\mathbf{F}_{i+1}$  transformed to link i coordinates, and the sum of the generalized forces acting on link i are, again in terms of link i frame coordinates,

$$(\mathbf{m}_i, \mathbf{f}_i) = \mathbf{F}_i - \mathrm{Ad}_{f_{i+1,i}^{-1}}(\mathbf{F}_{i+1}).$$
 (80)

The minus sign is a consequence of our definition of  $\mathbf{F}_{i+1}$ . With these identifications the force equations are

$$\mathbf{f}_i = m_i (\dot{\mathbf{v}}_i + [\boldsymbol{\omega}_i] \mathbf{v}_i + [\dot{\boldsymbol{\omega}}_i] \mathbf{r}_i + [\boldsymbol{\omega}_i]^2 \mathbf{r}_i), \tag{81}$$

which is identical to equation (78). Also, the moment equation is

$$\mathbf{m}_{i} = \mathbf{I}_{i}\dot{\boldsymbol{\omega}}_{i} - m_{i}[\mathbf{r}_{i}]^{2}\dot{\boldsymbol{\omega}}_{i} + m_{i}[\mathbf{r}_{i}]\dot{\mathbf{v}}_{i} + [\boldsymbol{\omega}_{i}]\mathbf{I}_{i}\boldsymbol{\omega}_{i}$$
(82)  
+  $m_{i}[\boldsymbol{\omega}_{i}][\mathbf{r}_{i}]^{2}\boldsymbol{\omega}_{i} + m_{i}[\boldsymbol{\omega}_{i}][\mathbf{r}_{i}]\mathbf{v}_{i}$ (83)  
+  $m_{i}[\mathbf{v}_{i}][\boldsymbol{\omega}_{i}]\mathbf{r}_{i}.$ 

Using the Jacobi identity  $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) + \mathbf{c} \times (\mathbf{a} \times \mathbf{b}) + \mathbf{b} \times (\mathbf{c} \times \mathbf{a}) = 0$ , the second term  $-m_i |\mathbf{r}_i|^2 \dot{\omega}_i$  is equal to  $m_i [\mathbf{r}_i] [\boldsymbol{\omega}_i]^2 \mathbf{r}_i$ , and the last two terms simplify to  $m_i [\mathbf{r}_i] [\boldsymbol{\omega}_i] \mathbf{v}_i$ , which verifies the moment equations.

# Acknowledgments

We would like to thank R. Featherstone for his comments on generalized velocities and generalized forces, A. Kecskeméthy for drawing our attention to the pullback interpretation of the inertia matrix, and G. Rodriguez for useful discussions and constructive comments. This research was supported by the National Science Foundation under award MSS-9403019. F. C. Park was also supported by the KOSEF Engineering Research Center for Advanced Instrumentation and Control at Seoul National University.

### References

- Balafoutis, C. A., and Patel, R. V. 1991. Dynamic Analysis of Robot Manipulators: A Cartesian Tensor Approach. Boston: Kluwer.
- Brockett, R. W. 1983. Robotic manipulators and the product of exponentials formula. *Proc. Int. Symp. Math. Theory of Networks and Systems*, Beersheba, Israel.
- Brockett, R. W. 1990. Some mathematical aspects of robotics. In *Robotics*. AMS Short Course Lecture

- Notes, Vol. 41. Providence: Amererican Mathematical Society.
- Brockett, R., Stokes, A., and Park, F. 1993. A geometrical formulation of the dynamic equations describing kinematic chains. In *Proc. IEEE Int. Conf. Robotics and Automation*. Atlanta: IEEE, pp. 637–641.
- Featherstone, R. 1987. *Robot Dynamics Algorithms*. Boston: Kluwer.
- Greenwood, D. T. 1965. *Principles of Dynamics*. Englewood Cliffs, NJ: Prentice Hall.
- Li, Z. 1989. Kinematics, Planning, and Control of Dexterous Robot Hands. Ph.D. thesis. Dept. of Electrical Engineering and Computer Science, University of California, Berkeley.
- Loncaric, J. 1985. Geometric Analysis of Compliant Mechanisms in Robotics. Ph.D. thesis. Harvard University.
- Luh, J. Y. S., Walker, M. H., and Paul, R. P. 1980. Online computational scheme for mechanical manipulators. ASME J. Dyn. Sys. Meas. Control 102:69–76.
- Murray, R. M., Li, Z., and Sastry, S. S. 1993. *A Mathematical Introduction to Robotic Manipulation*. Boca Raton, FL: CRC Press.
- Rodriguez, G., Jain, A., and Kreutz-Delgado, K. 1991. A spatial operator algebra for manipulator modeling and control. *Int. J. Robot. Res.* 10:371–381.
- Rodriguez, G., and Kreutz-Delgado, K. 1992. Spatial operator factorization and inversion of the manipulator mass matrix. *IEEE Trans. Robot. Automat.* 8:65–76.
- Spong, M. W. 1992. Remarks on robot dynamics: Canonical transformations and Riemannian geometry. *Proc. IEEE Int. Conf. Robotics Automat.* Nice, Italy: IEEE.
- Spong, M. W., and Vidyasagar, M. 1989. Robot Dynamics and Control. New York: Wiley.