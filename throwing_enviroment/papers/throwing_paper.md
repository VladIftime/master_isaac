2023 IEEE International Conference on Robotics and Automation (ICRA 2023)
May 29 - June 2, 2023. London, UK
Throwing Objects into A Moving Basket While Avoiding Obstacles
Hamidreza Kasaei1 and Mohammadreza Kasaei2
Abstract—Thecapabilitiesofarobotwillbeincreasedsignif-
icantlybyexploitingthrowingbehavior.Inparticular,throwing
will enable robots to rapidly place the object into the target
basket, located outside its feasible kinematic space, without
traveling to the desired location. In previous approaches,
the robot often learned a parameterized throwing kernel
through analytical approaches, imitation learning, or hand-
coding. There are many situations in which such approaches
do not work/generalize well due to various object shapes,
heterogeneous mass distribution, and also obstacles that might
be presented in the environment. It is obvious that a method
is needed to modulate the throwing kernel through its meta-
parameters. In this paper, we tackle object throwing problem
through a deep reinforcement learning approach that enables
robots to precisely throw objects into a moving basket while
there is an obstacle obstructing the path. To the best of our
knowledge, we are the first group that addresses throwing
objectswithobstacleavoidance.Suchathrowingskillnotonly
increases the physical reachability of a robotic arm but also Fig. 1: An example scenario of throwing an object into a
improves the execution time. In particular, the robot detects
movingbasketlocatedoutsideoftherobot’sworkspacewhile
the pose of the target object, basket, and obstacle at each time
anobstacleobstructingthepath.Toaccomplishthistasksuc-
step, predicts the proper grasp configuration for the target
object, and then infers appropriate parameters to throw the cessfully, the robot should perceive the environment through
object into the basket. Due to safety constraints, we develop its RGB-D camera, and then infer the proper parameters to
a simulation environment in Gazebo to train the robot and throw the object into the basket.
then use the learned policy in real-robot directly. To assess the
performers of the proposed approach, we perform extensive
sets of experiments in both simulation and real-robot in three gripper) to the physical properties of the object (e.g. shape,
scenarios. Experimental results showed that the robot could
size, softness, mass, material, etc.). Many of these elements
precisely throw a target object into the basket outside its
are challenging to describe or measure analytically, hence,
kinematic range and generalize well to new locations and
objects without colliding with obstacles. The video of our earlier research has frequently been limited to assuming
experiments can be found at https://youtu.be/VmIFF c 84 predefined objects (e.g., ball) and initial conditions (e.g.,
manually placing objects in a per-defined location). When
I. INTRODUCTION
obstaclesarepresentintheenvironmentandthetargetbasket
Thehumanabilitytothrowobjectsisawidelyrecognized
ismoving,throwingbecomesevenmoredifficult.Tothebest
skill,acquiredthroughparticipationinvariousactivitiessuch
of our knowledge, we are the first group to address such a
as ball games (e.g., basketball) or tossing objects into a
challenging object throwing problem.
container (e.g., laundry basket). We throw objects either
Toaccomplishthethrowingtasksuccessfully,arobotmust
to speed up tasks by reducing the time of pick-and-place
process the visual information to realize which objects exist
or to place them in an unreachable place [1]. Given the
in the scene (i.e., target object, basket, and obstacles), what
benefits of throwing, integrating this motion into a robotic
arethestateoftheobjects(i.e.,pose,speed,etc.),andhowto
manipulator could significantly improve its functionality. In
graspthetargetobject(graspsynthesis).Therobotthenfinds
particular,throwingobjectisagreatwaytoexploitdynamics
an obstacle-free trajectory to grasp the object. Afterward,
and increase the power of a robot by enabling it to quickly
given the obstacles that exist in the scene and the state of
place objects into the target locations outside of the robot’s
the target basket, it needs to predict throwing parameters to
kinematic range. However, the act of precisely throwing is
throw the object to the desired location precisely (e.g., the
actually far more complex than it appears and requires a lot
velocity of executing the throw trajectory, time of release,
of practice since it depends on many factors, ranging from
etc.). Lastly, the robot executes the throwing motion using
pre-throwconditions(e.g.initialposeoftheobjectinsidethe
those parameters.
1Hamidreza Kasaei is with the Department of Artificial Intelligence, In this paper, we formulate object throwing as an RL
Bernoulli Institute, Faculty of Science and Engineering, University of problem to enable the robot to generalize well across a
Groningen,TheNetherlands.Email:hamidreza.kasaei@rug.nl
varietyofobjectsandreactquicklytodynamicenvironments
2 Mohammadreza Kasaei is with the School of Informatics, University
ofEdinburgh,UK.Email:m.kasaei@ed.ac.uk (i.e., a moving basket). For RL, the exploration phase is
979-8-3503-2365-8/23/$31.00 ©2023 IEEE 3051
51206101.3202.19884ARCI/9011.01
:IOD
|
EEEI
3202©
00.13$/32/8-5632-3053-8-979
|
)ARCI(
noitamotuA
dna
scitoboR
no
ecnerefnoC
lanoitanretnI
EEEI
3202
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore. Restrictions apply.

often unsafe in the real-world. It takes a while to build up B. Learning Approaches
enoughexperiencetotrainthepolicytofunctionsuccessfully
|     |     |     |     |     | Unlike analytical |     | approaches |     | for throwing, | learning-based |     |
| --- | --- | --- | --- | --- | ----------------- | --- | ---------- | --- | ------------- | -------------- | --- |
inadynamicenvironmentwithmovingtargetsandobstacles. methods enable robots to learn/optimize the main task di-
| Therefore, | we develop a simulation | in Gazebo, | very | similar |     |     |     |     |     |     |     |
| ---------- | ----------------------- | ---------- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
rectlythroughsuccessorfailuresignals.Ingeneral,learning-
| to our real-robot | setup, and | train the robot | in  | simulation |                |            |     |             |     |        |             |
| ----------------- | ---------- | --------------- | --- | ---------- | -------------- | ---------- | --- | ----------- | --- | ------ | ----------- |
|                   |            |                 |     |            | based throwing | approaches |     | demonstrate |     | better | performance |
initially. Afterwards, the learned policy is used in real-robot than analytical methods[10], [11]. In [10], a deep predictive
settingsdirectly.Weextensivelyevaluatetheperformanceof
|     |     |     |     |     | policy training | architecture |     | (DPPT) | is  | presented | to teach |
| --- | --- | --- | --- | --- | --------------- | ------------ | --- | ------ | --- | --------- | -------- |
our approach in both simulation and real-robot using three a PR2 robot object-grasping and ball-throwing tasks. They
| different tasks | with ascending | levels of difficulties. |     | Exper- |     |     |     |     |     |     |     |
| --------------- | -------------- | ----------------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
showedDPPTissuccessfulinbothsimulatedandrealrobots.
| imental results | show that | the proposed | method | produces |            |       |       |        |                 |     |             |
| --------------- | --------- | ------------ | ------ | -------- | ---------- | ----- | ----- | ------ | --------------- | --- | ----------- |
|                 |           |              |        |          | In another | work, | Kober | et al. | [11] introduced |     | an RL-based |
throws that are more accurate than baseline alternatives. In method for dart throwing task based on a kernelized version
| summary, | our key contributions | are threefold: |     |     |                        |     |     |             |         |     |              |
| -------- | --------------------- | -------------- | --- | --- | ---------------------- | --- | --- | ----------- | ------- | --- | ------------ |
|          |                       |                |     |     | of the reward-weighted |     |     | regression. | In both | of  | these works, |
Tothebestofourknowledge,wearethefirstgroupthat the properties of the object (ball and dart) are known a-
•
addresses object tossing while obstacles are present in priori. In contrast to both of these approaches, we do not
the environment and the target basket is moving. make assumptions about the physical properties of objects
| Despiteonlytrainedusingsimulationdata,theproposed |     |     |     |     | that are thrown. |     |     |     |     |     |     |
| ------------------------------------------------- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- |
•
approach can be directly applied to real-robot. Further- In some other works, researchers tried to combine the
more, it shows impressive generalization capability to potential of analytical and learning approaches for robotic
new target locations and unseen objects. throwing tasks. In particular, analytical models are used to
• Our experiments show that the trained policy could approximate the initial control parameters, and a learning-
achieve above 80% object throwing accuracy for the basedmodelisusedtoestimateresidualparameterstoadjust
most difficult task (i.e., throwing object into the basket the initial parameters. Such approaches are called residual
while there is an obstacle obstructing the path) in both physics. For instance, [4] proposed TossingBot, an end-to-
|            |                |               |     |     | end self-supervised |      | learning | method   | for              | learning | to throw     |
| ---------- | -------------- | ------------- | --- | --- | ------------------- | ---- | -------- | -------- | ---------------- | -------- | ------------ |
| simulation | and real robot | environments. |     |     |                     |      |          |          |                  |          |              |
|            |                |               |     |     | arbitrary objects   | with | residual |          | physics. Similar |          | to our work, |
|            |                |               |     |     | their approach      | was  | able     | to throw | an object        | into     | a basket.    |
II. RELATEDWORK
|     |     |     |     |     | Unlike our | approach, | they | used | an analytical |     | approach for |
| --- | --- | --- | --- | --- | ---------- | --------- | ---- | ---- | ------------- | --- | ------------ |
Theroboticscommunityhaslongbeeninterestedingiving estimating initial control parameters, and then used an end-
service robots the ability to throw objects [2], [3], [4], [5], to-endformulationforlearningresidualvelocityforthrowing
[6]. Throwing formulae were mostly influenced by analyt- motion primitives. We formulate the throwing task as an RL
ical models in the late 1990s and early 2000s [7], while problem that modulates the parameters of a kernel motion
such formulations are increasingly moving toward learning generator. In contrast to all reviewed works, our formulation
approaches today [8], [4]. In the following subsections, we allows the robot to throw the object into a moving basket
briefly review these approaches. whileavoidingobstacles,whereas,inallreviewedworks,the
throwingtaskhasbeenconsideredinanobstacle-freedomain
|               |            |     |     |     | where the target | pose | is  | static | and known | in advance. |     |
| ------------- | ---------- | --- | --- | --- | ---------------- | ---- | --- | ------ | --------- | ----------- | --- |
| A. Analytical | Approaches |     |     |     |                  |      |     |        |           |             |     |
Earlier throwing systems relied on handcrafting or me- III. METHOD
chanical analysis and then optimizing control parameters to In this section, the preliminaries are briefly reviewed, fol-
execute a throw such that the projectile (typically a ball) lowed by a discussion of how we formulate object throwing
landsatatargetlocation.Aswepreviouslyhighlighted,pre-
|     |     |     |     |     | as an RL problem. |     | The | perception | that represents |     | the world |
| --- | --- | --- | --- | --- | ----------------- | --- | --- | ---------- | --------------- | --- | --------- |
cisely modeling of dynamics is difficult because it calls for model at each time step is the subject of the last subsection.
| knowledge | about the physical | characteristics | of  | the object, |     |     |     |     |     |     |     |
| --------- | ------------------ | --------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
A. Preliminaries
gripperandenvironment,whicharehardtoquantify[7].For
instance, Y. Gai et al., derived an analytical approach for Markov Decision Process (MDP): An MDP can be
throwing a ball using a manipulator with a single flexible described as a tuple containing four basic elements:
link through Hamilton’s principle [3]. This is an example (s ,a ,p(s |s ,a ),r(s |s ,a )),wherethes anda are
|     |     |     |     |     | t t t+1 | t   | t   | t+1 t | t   |     | t t |
| --- | --- | --- | --- | --- | ------- | --- | --- | ----- | --- | --- | --- |
of tuning for a single object, a ball in this case. In another the continuous state and action at time step t, respectively.
work, Hu et al., [2] discussed a stereo vision system for p(s t+1 |s t ,a t ) shows the transition probability function to
throwing a ball into a basket. They calculated the ball- reach to the next state s given the current state s and
|     |     |     |     |     |     |     |     | t+1 |     |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
throwing transformation for a specific ball object based on action a . The r(s |s ,a ) denotes the immediate reward
|     |     |     |     |     | t   |     | t+1 t | t   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- |
cubic polynomial. In [9], an analytical approach is used to received from the environment after the state transition.
predict the end-effector velocity (magnitude and direction) Off-policy RL: In online RL, an agent continuously
aswellasadurationmovementforunderhandthrowingtask interactswiththeenvironmenttoaccumulateexperiencesfor
byahumanoidrobot.Suchapproachestosomeextendwork learningtheoptimalpolicyπ∗.Theagentseekstomaximize
for specific scenario but have difficulties generalizing over the expected future return R = E[ (cid:80)∞ γi−tr ] with a
|     |     |     |     |     |     |     |     | t   | i=t |     | i+1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
changing dynamics and various objects. discountedfactorγ ∈[0,1]weightingthefutureimportance.
3052
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore.  Restrictions apply.

The expected return under a policy π after taking action a and action respectively. It should be noted that since the
in the state s is computed by a corresponding action-value transition function is unknown, our off-policy reinforcement
|     | Qπ(s,a) |     | E[R |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
function = t |s t = s,a t = a]. By following learning framework is also model-free.
policyπ,wecancomputeQπ throughtheBellmanequation: 4) Rewards: A success is reached if the thrown object
fallsintothetargetbasketafterexecutingthethrowingaction.
|         |     |      |         |     |     |     |             | In particular, | we  | calculate | the | absolute | distance | between | the |
| ------- | --- | ---- | ------- | --- | --- | --- | ----------- | -------------- | --- | --------- | --- | -------- | -------- | ------- | --- |
| Qπ(s ,a | )=E | [r(s | ,a )+γE |     | [Q  | (s  | ,a )]], (1) |                |     |           |     |          |          |         |     |
t t st+1∼p t t at+1∼A π t+1 t+1 objectandthegoaldis(o,g)<d,whereddenotestheradius
where A states the action space. Consider Q∗(s,a) is the of a cylindrical space that is fitted inside the basket.
|         |              |         |           |          |            |           |            | As a result   | of       | the current |      | state and     | the action | taken,     | if the |
| ------- | ------------ | ------- | --------- | -------- | ---------- | --------- | ---------- | ------------- | -------- | ----------- | ---- | ------------- | ---------- | ---------- | ------ |
| optimal | action-value |         | function. | RL       | algorithms | aim       | to find an |               |          |             |      |               |            |            |        |
|         |              |         |           |          |            |           |            | thrown object | collides |             | with | the obstacle, | severe     | punishment |        |
| optimal | policy       | π∗ such | that      | Qπ∗(s,a) |            | = Q∗(s,a) | for all    |               |          |             |      |               |            |            |        |
states and actions. is given by a negative reward r = −10. It should be noted
|            |             |     |     |     |     |     |     | that the   | collision | information |     | can only | be    | obtained   | after the |
| ---------- | ----------- | --- | --- | --- | --- | --- | --- | ---------- | --------- | ----------- | --- | -------- | ----- | ---------- | --------- |
| B. Problem | Formulation |     |     |     |     |     |     |            |           |             |     |          |       |            |           |
|            |             |     |     |     |     |     |     | action has | been      | executed.   | If  | the next | state | results in | success   |
Given the start pose p (i.e., grasp synthesis of an object) (the thrown object falls into the basket), we encourage such
s
and the goal pose p (i.e., pose of the basket), the throwing behaviorbysettingtherewardr =1.Inthecaseofthrowing
g
task is defined as the problem of finding proper parameters the object outside the target basket is also penalized by
tothrowtheobjectintothebasketwhileavoidingobstacleso calculating rewards based on distance r = − dis(o,g). An
|                     |     |     |       |            |     |     |            | episode | is terminated |     | after | executing | a throwing |     | action. It |
| ------------------- | --- | --- | ----- | ---------- | --- | --- | ---------- | ------- | ------------- | --- | ----- | --------- | ---------- | --- | ---------- |
| in the environment. |     | A   | fully | observable | MDP | can | be used to |         |               |     |       |           |            |     |            |
representthistask,andtheoff-policyreinforcementlearning should be noted that successful attempts are recorded for
framework can be used to solve it. We discuss the detailed later use in behavioral cloning.
| RL formulation |     | in the | following | subsections. |     |     |     |     |     |     |     |     |     |     |     |
| -------------- | --- | ------ | --------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
C. Perception
| 1) States:   | Afeaturevectorisusedtodescribethecontin- |          |             |                  |     |           |             |          |        |                 |     |       |            |         |          |
| ------------ | ---------------------------------------- | -------- | ----------- | ---------------- | --- | --------- | ----------- | -------- | ------ | --------------- | --- | ----- | ---------- | ------- | -------- |
|              |                                          |          |             |                  |     |           |             | Due to   | safety | considerations, |     |       | we trained | the     | proposed |
| uous state,  | including                                |          | the robot’s | proprioception,  |     |           | the pose of |          |        |                 |     |       |            |         |          |
|              |                                          |          |             |                  |     |           |             | throwing | policy | in simulation   |     | first | and then   | use the | learned  |
| the obstacle | and                                      | the goal | in          | the environment, |     | releasing | time,       |          |        |                 |     |       |            |         |          |
durationoftrajectoryexecution,andthedistancebetweenthe policy on our real-robot platform. In the case of simula-
|     |     |     |     |     |     |     |     | tion, we | developed | an  | interface | that | provides | all | necessary |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --------- | --- | --------- | ---- | -------- | --- | --------- |
thrownobjectandthegoal.Inparticular,weformedakernel
trajectory for throwing an object in a straight way using information based on Gazebo’s services, while for real-
|              |        |              |     |           |          |              |             | robot experiments, |     | we      | exploit | our      | robotic   | setup | to detect |
| ------------ | ------ | ------------ | --- | --------- | -------- | ------------ | ----------- | ------------------ | --- | ------- | ------- | -------- | --------- | ----- | --------- |
| trial and    | error. | We then      | let | the robot | learns   | the          | initial and |                    |     |         |         |          |           |       |           |
|              |        |              |     |           |          |              |             | and track          | the | pose of | the     | objects, | recognize | them, | and       |
| final values | for    | the shoulder |     | joint,    | duration | of executing | the         |                    |     |         |         |          |           |       |           |
trajectory(speed),andreleasingtimeinthelearningprocess. predict stable grasp syntheses for each of the object in
|                                                |              |                          |           |             |             |                  |              | 3D space | [12][13][14]. |        | In particular, |             | our | real robot  | uses an |
| ---------------------------------------------- | ------------ | ------------------------ | --------- | ----------- | ----------- | ---------------- | ------------ | -------- | ------------- | ------ | -------------- | ----------- | --- | ----------- | ------- |
| In particular,                                 |              | at each                  | step      | t, we       | record      | the              | initial and  |          |               |        |                |             |     |             |         |
|                                                |              |                          |           |             |             |                  |              | RGB-D    | Asus Xtion    | camera |                | to perceive | the | environment | by      |
| final shoulder                                 |              | joint                    | values    | (j and      | j )         | in radians       | as the       |          |               |        |                |             |     |             |         |
|                                                |              |                          |           | i           | f           |                  |              |          |               |        |                |             |     |             |         |
| proprioception:                                |              | proprio                  | = (j      | ,j )        | ∈ R2.       | Then,            | we estimate  |          |               |        |                |             |     |             |         |
|                                                |              |                          |           | i f         |             |                  |              |          |               |        |                |             |     |             |         |
| the obstacle’s                                 |              | position                 | in        | task space  | and         | describe         | it as        |          |               |        |                |             |     |             |         |
| obs =                                          | (xo,yo,zo)   |                          | ∈ R3.     | We describe |             | the position     | and          |          |               |        |                |             |     |             |         |
| the speed                                      | of goal      | (i.e,                    | center    | of the      | box)        | as a             | point in the |          |               |        |                |             |     |             |         |
| task space:                                    | goal         | = (xg,yg,zg,x˙g,y˙g,z˙g) |           |             |             | ∈ R6.            | We also      |          |               |        |                |             |     |             |         |
| consider                                       | the absolute |                          | distance  | of the      | thrown      | object           | relative     |          |               |        |                |             |     |             |         |
| totheobstacleandgoal,andalsothedistancesintheX |              |                          |           |             |             |                  | and          |          |               |        |                |             |     |             |         |
| Y axes,                                        | dist =       | (dg,dg,dg,do,do,do)      |           |             | ∈           | R6. Furthermore, |              |          |               |        |                |             |     |             |         |
|                                                |              |                          | x y       | x           | y           |                  |              |          |               |        |                |             |     |             |         |
| we record                                      | two          | timing                   | profiles, | the         | duration    | of               | executing    |          |               |        |                |             |     |             |         |
| the throwing                                   | trajectory   |                          | τ,        | and the     | time        | for releasing    | the          |          |               |        |                |             |     |             |         |
| object,                                        | t , where    | t                        | < τ       | and         | time        | = (t ,τ)         | ∈ R2.        |          |               |        |                |             |     |             |         |
|                                                | r            | r                        |           |             |             | r                |              |          |               |        |                |             |     |             |         |
| Finally,                                       | the state    | space                    | can       | be          | represented | as               | a vector:    |          |               |        |                |             |     |             |         |
s=(proprio,obs,goal,dist,time)∈R19.
| 2) Actions:    |     | Each          | action     | is        | denoted  | by       | a vector     |     |     |     |     |     |     |     |     |
| -------------- | --- | ------------- | ---------- | --------- | -------- | -------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
| a∈([−1,1])4,   |     | which         | represents | (i)       | the      | initial  | and (ii) the |     |     |     |     |     |     |     |     |
| final shoulder |     | joint values, |            | (iii) the | duration | of       | trajectory   |     |     |     |     |     |     |     |     |
| execution,     | and | (iv) the      | releasing  | object    | time.    |          |              |     |     |     |     |     |     |     |     |
| 3) Transition  |     | function:     | In         | each      | training | episode, | we set       |     |     |     |     |     |     |     |     |
the pose of the goal randomly and then, set the pose of Fig.2:Visualizingtheoutputofourperceptionmoduleforan
examplescene:itprovidesworldmodelinformationinterms
| the obstacle    | between       |         | the robot   |           | and the       | goal     | pose with  |                      |         |           |         |             |             |          |           |
| --------------- | ------------- | ------- | ----------- | --------- | ------------- | -------- | ---------- | -------------------- | ------- | --------- | ------- | ----------- | ----------- | -------- | --------- |
|                 |               |         |             |           |               |          |            | of object            | pose,   | size, and | label.  | In          | particular, | the      | estimated |
| ±5cm randomness |               | on      | the x-axis. |           | Therefore,    | the      | transition |                      |         |           |         |             |             |          |           |
|                 |               |         |             |           |               |          |            | object’s             | pose is | shown     | by      | a reference | frame,      | the      | object’s  |
| function        | is determined |         | by          | executing | the           | throwing | trajec-    |                      |         |           |         |             |             |          |           |
|                 |               |         |             |           |               |          |            | size is demonstrated |         | by        | a green | bounding    |             | box, and | the label |
| tory given      | the           | sampled | parameters. |           | Specifically, |          | the next   |                      |         |           |         |             |             |          |           |
state, s , can be computed after executing the action f ; of the object is highlighted on top of the object’s Z axis by
| i+1         |         |           |       |         |         |             | s     |          |           |      |          |     |          |          |     |
| ----------- | ------- | --------- | ----- | ------- | ------- | ----------- | ----- | -------- | --------- | ---- | -------- | --- | -------- | -------- | --- |
|             |         |           |       |         |         |             |       | red, and | the grasp | pose | is shown | by  | a yellow | gripper. |     |
| i.e., s i+1 | =f s (s | i ,a i ), | where | s i and | a i are | the current | state |          |           |      |          |     |          |          |     |
3053
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore.  Restrictions apply.

Fig. 3: Our experimental setups in (left) simulation and (right) real-robot settings: our dual-arm robot consists of two UR5e
manipulators, and an Asus Xiton RGB-D camera to perceive the environment. We have developed a simulation environment
in Gazebo very similar to our real robot to reduce the gap between the simulation and the real robot and facilitate transfer
learning. Due to safety constraints, we initially trained the robot in the simulation environment, and then directly transferred
| the learned | policy | to the | real | robot without | fine-tuning. |     |     |     |     |     |     |     |     |     |
| ----------- | ------ | ------ | ---- | ------------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
capturingpointclouddataat30Hz.Generally,apointcloud environment in Gazebo utilizing an ODE physics engine,
consists of a set of points, p i : i ∈ {1,...,n}, where each which is very similar to our real-robot setup. Our setup
point is described by its 3D coordinates [x,y,z] and RGB- consists of an Asus xtion camera, two Universal Robots
D information. To track the pose and speed of the target (UR5e) equipped with two-fingered Robotiq 2F-140 gripper,
object,weuseaparticlefilterthatconsidersshapeandcolor and a user interface to start and stop the experiments.
data[15].Itisworthmentioningthatforthiswork,weforce Due to safety consideration, we trained the proposed
the system to grasp the object from above near the center throwing policy in the simulation. After training phase, we
of mass [14]. In particular, our perception system provides conducted experiments in both simulation and real-robot
a world model service that the agent can call at each time setups to assess the performance of the learned policy in
step to receive the current state of the world, which includes throwing various objects into a box located outside the
the unique ID, pose, speed, label, and grasp synthesis of robot’sreachablearea.Forevaluationpurposes,wedesigned
each object. Figure 2 shows the outputs of our perception three tasks with ascending difficulty levels:
system regarding object detection (i.e., highlighted by green Task1:obstacle-freeobjectthrowingintoastaticbasket
•
| bounding | boxes | and | reference | frames), | object | recognition |     |          |        |     |       |        |        |            |
| -------- | ----- | --- | --------- | -------- | ------ | ----------- | --- | -------- | ------ | --- | ----- | ------ | ------ | ---------- |
|          |       |     |           |          |        |             |     | randomly | placed | in  | front | of the | robot. | An example |
(i.e., highlighted by red on top of each object), and pre- of this task in simulation environment is shown in
| grasp pose | (i.e., | highlighted |     | by yellow | gripper). | For | more |     |     |     |     |     |     |     |
| ---------- | ------ | ----------- | --- | --------- | --------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
Fig. 3 (left);
information about our perception and grasping pipelines, • Task2: obstacle-free object throwing into a moving
please refer to our earlier works [16][13][14]. basket. An example of this task in real-robot setup is
|              | IV. | EXPERIMENTSANDRESULTS |        |     |             |     |      | shown  | in Fig. | 3 (right); |       |             |     |               |
| ------------ | --- | --------------------- | ------ | --- | ----------- | --- | ---- | ------ | ------- | ---------- | ----- | ----------- | --- | ------------- |
|              |     |                       |        |     |             |     |      | Task3: | object  | throwing   | while | an obstacle |     | obstructs the |
| We performed |     | multiple              | rounds | of  | experiments | in  | both | •      |         |            |       |             |     |               |
simulation and real-robot settings to validate our method. throwing path as shown in Fig. 1.
|               |     |             |     |             |     |              |     | The dimension | of  | the basket | was | 0.30×0.30×0.15 |     | (W × |
| ------------- | --- | ----------- | --- | ----------- | --- | ------------ | --- | ------------- | --- | ---------- | --- | -------------- | --- | ---- |
| In this work, |     | we evaluate | the | performance | of  | the proposed |     |               |     |            |     |                |     |      |
approach based on the throwing success rate, which is L × H) and obstacle was 0.15×0.10×0.22. We used 10
|            |        |             |     |               |        |         |       | simulated  | daily-life | objects | with   | different | materials,     | shapes, |
| ---------- | ------ | ----------- | --- | ------------- | ------ | ------- | ----- | ---------- | ---------- | ------- | ------ | --------- | -------------- | ------- |
| calculated | as     | the number  | of  | times a       | thrown | object  | lands |            |            |         |        |           |                |         |
|            |        |             |     |               |        |         |       | sizes, and | weight,    | and     | 5 real | objects.  | In particular, | five    |
| into the   | target | box divided |     | by the number | of     | trials. | More  |            |            |         |        |           |                |         |
specifically, we tried to investigate the following questions: simulated objects were used during training (i.e., milk box,
cokecan,banana,bottle,apple)andtheotherfivesimulated
| (1) Which | of  | the RL | approach | outperforms | other | baselines |     |     |     |     |     |     |     |     |
| --------- | --- | ------ | -------- | ----------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
in terms of throwing success rate when used in the same objects were used for testing (i.e., beer can, peach, soap,
|     |     |     |     |     |     |     |     | pringles,mustard | bottle).Fortherealrobotexperiments,we |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | ------------------------------------- | --- | --- | --- | --- | --- |
environmentandtasksettings?(2)Doesourmethodlearnto
usedfivehouseholdobjectsthataredistinctinsizeandshape
| safely throw | an  | object | into a | target basket | while | there | is an |     |     |     |     |     |     |     |
| ------------ | --- | ------ | ------ | ------------- | ----- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
obstacle obstructing the path? (3) Can the policy learned in from the simulated objects used during the training phase
|                 |           |       |           |                |       |       |     | (i.e., ugly | toy, hello | kitty, | small | box, juice | box,             | hand soap). |
| --------------- | --------- | ----- | --------- | -------------- | ----- | ----- | --- | ----------- | ---------- | ------ | ----- | ---------- | ---------------- | ----------- |
| simulation      | transfers | well  | to        | our real-robot | where | noise | and |             |            |        |       |            |                  |             |
| uncertainty     | exist?    |       |           |                |       |       |     |             |            |        |       |            |                  |             |
|                 |           |       |           |                |       |       |     | B. Baseline | Methods    |        |       |            |                  |             |
| A. Experimental |           | setup | and tasks | settings       |       |       |     |             |            |        |       |            |                  |             |
|                 |           |       |           |                |       |       |     | We employed | two        | sample |       | efficient  | state-of-the-art | off-        |
Our experimental setups in simulation and real-robot are policy RL algorithms to train the robot: Deep Determin-
depicted in Fig. 3. In particular, we developed a simulation istic Policy Gradient (DDPG) [17], [18], and Soft Actor-
3054
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore.  Restrictions apply.

Fig. 4: Four consecutive snapshots showing our simulated robot successfully thrown a milk box into the basket (Task1). In
this round of experiments, the basket is not reachable by the robot, and the robot should grasp the target object first, and
then, based on the pose of the basket, infers proper parameters to throw the object into the basket successfully.
Critic (SAC) [19], [20] via stable baseline3 [21]. The the robot, on average, obtained up to 91% (91 success out
architectures of the neural networks for both SAC and of 100 attempts) throwing success rate in simulation and
DDPG consist of two hidden layers with the size of 256 90% (18/20) in real-robot setting. In particular, the robot
neron per layers with the ReLU activation functions. The with SAC method, on average, showed the best throwing
hyper-parameters of SAC are listed in Table. I, and the performance in real and simulation experiments, and out-
hyper-parameters of DDPG that are not listed in Table I performedDDPGandBCmethodswithalargemargin.The
are reported in the robotwithDDPGachievedthesecond-bestthrowingsuccess
Experiments section of TABLEI:SAChyper-parameters rate(real:85%(17/20),simulation:81%),whereasusingBC
ourpreviouswork[22]. approachitcouldget70%(14/20)and81%accuracyinreal
Parameter Value
We also considered and simulation experiments respectively. As expected, the
#hiddenlayers(allnetworks) 2
Behavior Cloning #hiddenunitsperlayer 256 success rate of throwing unseen objects is moderately lower
(BC), which directly #samplesperminibatch 256 for all policies. These results showed that the learned policy
optimizer Adam
learns a policy (i.e., learningrate 3×10−4 performs well both in simulation and for the real robot.
similar to the actor batchsize 256 In the second round of experiments (Task2) the robot
#epochs 50K
network) by using discount(γ) 0.99 should learn to throw a target object into a moving basket.
supervised learning on replaybuffersize 106 Inthecaseofsimulationexperiments,werandomlyselected
nonlinearity ReLU
observation-actionpairs targetupdaterate(τ) 0.005 the direction and a linear speed for moving the basket while
from 25k successful targetupdateinterval 1 in real-robot experiments, a human user moved the basket
gradientsteps 1
trials, collected during using a stick (see Fig. 5). Similar to the previous round of
train and test phases. experiments,therobotwiththeSACpolicyobtainedthebest
results for both seen and unseen objects. More specifically,
C. Results
for the seen objects, the robot obtained 91% with the SAC
For each of the proposed tasks, we trained the model for
strategy, whereas its throwing performance with DDPG and
50,000 steps in simulation using the five training objects. In
BC dropped to 89% and 79%, respectively. When tossing
the case of simulation, for each object the robot’s throwing
unseen objects in simulation, the robot was 86% accurate
performancewastestedfor100stepsusingthelearnedpolicy
with SAC and DDPG policies, compared to 73% accuracy
twice: once with the five test (unseen) objects and once
with BC. Intriguingly, in contrast to simulation results, the
with the five train (seen) objects. We also used the same
robot with DDPG policy does not perform as competitively
learnedpolicyforrealrobotexperimentsandtesteachofthe
to SAC in the real-world.
test objects for 20 times. As opposed to “unseen objects”,
Task 3 is much more complex than the previous tasks
which is a mixed set of objects not seen during training,
as the robot should learn to infer an obstacle-free path to
“seen objects” is a mixed set of objects that were used
throw the object into the basket. An example of such expri-
during training. The average throwing success rates for each
approach is reported in Table II.
TABLE II: Object Throwing Performance (Mean %). The
In the case of Task 1 (i.e., obstacle-free object throwing
S/U-S/R shows the setup configuration: the first token refers
into a randomly placed static basket), we observed that for
to the Seen or Unseen objects, and the second one denotes
the seen objects the robot with SAC policy could throw
Simulation or Real experiments.
the objects into the basket successfully with 94% accuracy,
while with DDPG and BC (on average) achieved 91% and Task1 Task2 Task3
Methods
86% success rate, respectively. A sequence of snapshots S-S U-S U-R S-S U-S U-R S-S U-S U-R
SAC 94 91 90 91 86 85 86 83 80
demonstrating our robot successfully throwing a milk box DDPG 92 81 85 89 86 75 85 77 65
intoabasketisdepictedinFig.4.Regardingunseenobjects, BC 86 81 70 79 73 65 72 67 55
3055
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore. Restrictions apply.

Fig. 5: A sequence of snapshots showing our real robot successfully thrown a milk box into the moving basket (Task2): In
this round of experiments, the basket is not reachable by the robot, and a human user moves the basket using an aluminum
stick. The robot should first estimate the direction and velocity of the basket based on visual information, and then infer
| proper parameters |     | to throw | the milk | box into | the basket | successfully. |     |     |     |     |     |     |     |
| ----------------- | --- | -------- | -------- | -------- | ---------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
Fig.6:Tossingobjectsintothebasketwhileavoidingtheobstacle:therobotshouldfirstdetecttheposeoftheobject,basket
and obstacle, and then infers appropriate parameters to throw the object into the basket without colliding with the obstacle.
ments is shown in Fig. 6. Similar to the previous round of controller first, and then the controller sends the command
experiments,therobotwithSACpolicyoutperformedDDPG to the gripper, which depends on the status of network
and BC for both seen and unseen objects. By comparing all and the robot. (iii) Selecting an unstable grasp pose was
experiments, we observed that the difference between SAC the third reason for failure. Furthermore, we found that for
policy and others is larger when the task is more difficult. objects with heterogeneous shapes like ugly toy, hand soap,
This becomes more visible in the case of unseen objects and hello kitty, the trajectory of the thrown object varied
in real-robot experiments. In particular, the robot with SAC depending on the grasp pose. In contrast, for homogeneous
policy could better model the solution space and handle un- objects, e.g., milk box, the trajectory of the thrown object
modeled physical parameters presented in real-world objects was not dependent on the grasp pose.
(e.g., aerodynamics, materials, or stiffness). Therefore, in V. CONCLUSIONS
simulationexperiments,therobotusingSACpolicyobtained
|     |     |     |     |     |     |     | In this paper, | we  | trained | a policy | to adjust | the | parameters |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------- | -------- | --------- | --- | ---------- |
86%and83%throwingaccuracyforseenandunseenobjects
|     |     |     |     |     |     |     | of the throwing | kernel | based | on  | RL to enable | robots | to pre- |
| --- | --- | --- | --- | --- | --- | --- | --------------- | ------ | ----- | --- | ------------ | ------ | ------- |
respectively,whileinreal-robotexperimentsitcouldachieve
ciselythrowanobjectintoamovingbasketevenwhenthere
| 80% success | rate. | While | the performance | of  | the robot | with |     |     |     |     |     |     |     |
| ----------- | ----- | ----- | --------------- | --- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- |
DDPGpolicywasmarginallylowerthanSACpolicyforseen is an obstacle in the way. In particular, our method learns
|                 |        |           |                   |           |         |         | to handle     | a wide range | of            | situations | and           | avoid colliding | the      |
| --------------- | ------ | --------- | ----------------- | --------- | ------- | ------- | ------------- | ------------ | ------------- | ---------- | ------------- | --------------- | -------- |
| objects (i.e.,  | 86%    | vs. 85%), | the difference    | increased |         | signif- |               |              |               |            |               |                 |          |
|                 |        |           |                   |           |         |         | thrown object | with         | the obstacle. |            | With our      | formulation,    | the      |
| icantly for     | unseen | objects   | (i.e., simulation |           | 83% vs. | 77%,    |               |              |               |            |               |                 |          |
|                 |        |           |                   |           |         |         | robot could   | iteratively  | learn         | the        | aspects of    | dynamics        | that are |
| and in real     | 80%    | vs 65%).  | Experimental      | results   | showed  | that    |               |              |               |            |               |                 |          |
|                 |        |           |                   |           |         |         | difficult to  | model        | analytically. |            | Due to safely | constraints,    | we       |
| our formulation |        | maintains | the flexibility   | needed    | to      | express |               |              |               |            |               |                 |          |
trainedthethrowingpolicyinsimulationanddirectlyapplied
| complicated | dynamics | system | while | also making |     | learning |     |     |     |     |     |     |     |
| ----------- | -------- | ------ | ----- | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
thelearnedpolicyinreal-robot.Weperformedextensivesets
| easier through | trial | and | error. |     |     |     |                |     |                 |     |          |       |           |
| -------------- | ----- | --- | ------ | --- | --- | --- | -------------- | --- | --------------- | --- | -------- | ----- | --------- |
|                |       |     |        |     |     |     | of experiments | in  | both simulation |     | and real | robot | setups in |
threescenarioswithascendinglevelofdifficultiesincluding,
| D. Failure | Cases |     |     |     |     |     |                 |      |       |        |         |             |         |
| ---------- | ----- | --- | --- | --- | --- | --- | --------------- | ---- | ----- | ------ | ------- | ----------- | ------- |
|            |       |     |     |     |     |     | tossing objects | into | a (i) | static | basket, | (ii) moving | basket, |
Themainreasonforfailureforallapproachesinthesimu- and (iii) obstacle avoidance. Experimental results showed
lation experiments was the inaccurate parameters prediction that the proposed approach enables robot to precisely throw
and throwing of the object near the basket. In the case the object into the basket. We also observed that the learnt
of real robot experiments, apart from inaccurate parameters policy could be directly applied to the real robot even
prediction,weobservedthreeothertypeoffailures:(i)Inac- thoughithadonlybeentrainedinsimulation.Italsoshowed
curate Tracking was one of the main reasons. In particular, outstandinggeneralizationcapabilitiestonewtargetlocations
when the user move the object really fast, the tracking and unknown objects. In the continuation of this work,
could not follow the pose of the object immediately; (ii) the we would like to investigate the possibility of enhancing
second primary reason was the lag in executing the gripper throwing performance by taking into account additional
commands on time. In particular, the gripper is controlled sensory modalities (e.g., force or tactile sensors), as this
throughtherobot’scontrollerandwecannotopenandclose could help the robot grasp objects steadily and adjust its
it directly, i.e., we need to send the command to the robot’s throwing parameters accordingly.
3056
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore.  Restrictions apply.

REFERENCES [22] S. Luo, H. Kasaei, and L. Schomaker, “Accelerating reinforcement
learningforreachingusingcontinuouscurriculumlearning,”in2020
[1] J.E.Kuhn,“Throwing,theshoulder,andhumanevolution.”American InternationalJointConferenceonNeuralNetworks(IJCNN). IEEE,
JournalofOrthopedics(BelleMead,NJ),vol.45,no.3,pp.110–114, 2020,pp.1–8.
2016.
[2] J.-S.Hu,M.-C.Chien,Y.-J.Chang,S.-H.Su,andC.-Y.Kai,“Aball-
throwingrobotwithvisualfeedback,”in2010IEEE/RSJInternational
ConferenceonIntelligentRobotsandSystems. IEEE,2010,pp.2511–
2512.
[3] Y.Gai,Y.Kobayashi,Y.Hoshino,andT.Emaru,“Motioncontrolofa
ballthrowingrobotwithaflexibleroboticarm,”InternationalJournal
ofComputerandInformationEngineering,vol.7,no.7,pp.937–945,
2013.
[4] A.Zeng,S.Song,J.Lee,A.Rodriguez,andT.Funkhouser,“Tossing-
bot:Learningtothrowarbitraryobjectswithresidualphysics,”IEEE
TransactionsonRobotics,vol.36,no.4,pp.1307–1319,2020.
[5] A. Takahashi, M. Sato, and A. Namiki, “Dynamic compensation in
throwing motion with high-speed robot hand-arm,” in 2021 IEEE
InternationalConferenceonRoboticsandAutomation(ICRA). IEEE,
2021,pp.6287–6292.
[6] K.PloegerandJ.Peters,“Controllingthecascade:Kinematicplanning
forn-balltossjuggling,”arXivpreprintarXiv:2207.01414,2022.
[7] M.T.MasonandK.M.Lynch,“Dynamicmanipulation,”inProceed-
ingsof1993IEEE/RSJInternationalConferenceonIntelligentRobots
andSystems(IROS’93),vol.1. IEEE,1993,pp.152–159.
[8] J.Kober,K.Muelling,andJ.Peters,“Learningthrowingandcatching
skills,” in 2012 IEEE/RSJ International Conference on Intelligent
RobotsandSystems. IEEE,2012,pp.5167–5168.
[9] D.M.Lofaro,R.Ellenberg,P.Oh,andJ.-H.Oh,“Humanoidthrowing:
Design of collision-free trajectories with sparse reachable maps,” in
2012 IEEE/RSJ International Conference on Intelligent Robots and
Systems. IEEE,2012,pp.1519–1524.
[10] A. Ghadirzadeh, A. Maki, D. Kragic, and M. Bjo¨rkman, “Deep
predictivepolicytrainingusingreinforcementlearning.in2017ieee,”
in RSJ International Conference on Intelligent Robots and Systems
(IROS),pp.2351–2358.
[11] J. Kober, E. Oztop, and J. Peters, “Reinforcement learning to adjust
robot movements to new situations,” in Twenty-Second International
JointConferenceonArtificialIntelligence,2011.
[12] S. H. Kasaei, N. Shafii, L. S. Lopes, and A. M. Tome´, “Interactive
open-endedobject,affordanceandgrasplearningforroboticmanipu-
lation,”in2019InternationalConferenceonRoboticsandAutomation
(ICRA). IEEE,2019,pp.3747–3753.
[13] H. Kasaei, S. Luo, R. Sasso, and M. Kasaei, “Simultaneous multi-
viewobjectrecognitionandgraspinginopen-endeddomains,”arXiv
preprintarXiv:2106.01866,2021.
[14] H.KasaeiandM.Kasaei,“Mvgrasp:Real-timemulti-view3dobject
graspinginhighlyclutteredenvironments,”RoboticsandAutonomous
Systems,vol.160,p.104313,2023.
[15] S. Kasaei, J. Sock, L. S. Lopes, A. M. Tome´, and T.-K. Kim,
“Perceiving, learning, and recognizing 3d objects: An approach to
cognitiveservicerobots,”inProceedingsoftheAAAIConferenceon
ArtificialIntelligence,vol.32,no.1,2018.
[16] S.H.Kasaei,M.Oliveira,G.H.Lim,L.S.Lopes,andA.M.Tome´,
“Towardslifelongassistiverobotics:Atightcouplingbetweenobject
perceptionandmanipulation,”Neurocomputing,vol.291,pp.151–166,
2018.
[17] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa,
D.Silver,andD.Wierstra,“Continuouscontrolwithdeepreinforce-
mentlearning,”arXivpreprintarXiv:1509.02971,2015.
[18] D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Ried-
miller, “Deterministic policy gradient algorithms,” in International
conferenceonmachinelearning. PMLR,2014,pp.387–395.
[19] T.Haarnoja,A.Zhou,P.Abbeel,andS.Levine,“Softactor-critic:Off-
policymaximumentropydeepreinforcementlearningwithastochastic
actor,” in International conference on machine learning. PMLR,
2018,pp.1861–1870.
[20] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan,
V. Kumar, H. Zhu, A. Gupta, P. Abbeel et al., “Soft actor-critic
algorithmsandapplications,”arXivpreprintarXiv:1812.05905,2018.
[21] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and
N. Dormann, “Stable-baselines3: Reliable reinforcement learning
implementations,” Journal of Machine Learning Research, vol. 22,
no. 268, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/
v22/20-1364.html
3057
Authorized licensed use limited to: University of Groningen. Downloaded on June 12,2026 at 12:43:02 UTC from IEEE Xplore. Restrictions apply.
