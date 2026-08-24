# **Learning coordinated badminton skills for legged manipulators**

**YUNTAO MA** 1∗ **, ANDREI CRAMARIUC**<sup>1</sup> **, FARBOD FARSHIDIAN**<sup>2</sup> **, MARCO HUTTER**<sup>1</sup>

- <sup>1</sup>*Robotic Systems Lab, ETH Zurich, 8092 Zurich, Switzerland.*
- <sup>2</sup>*Currently at Robotics and AI Institute, 145 Broadway, Cambridge MA, USA.*
- <sup>∗</sup>*Corresponding author. Email: mayuntao94@gmail.com*

*Accepted on 29 April, 2025*

Coordinating the motion between lower and upper limbs and aligning limb control with perception are substantial challenges in robotics, particularly in dynamic environments. To this end, we introduce an approach for enabling legged mobile manipulators to play badminton, a task that requires precise coordination of perception, locomotion, and arm swinging. We propose a unified reinforcement learning-based control policy for whole-body visuomotor skills involving all degrees of freedom to achieve effective shuttlecock tracking and striking. This policy is informed by a perception noise model that utilizes real-world camera data, allowing for consistent perception error levels between simulation and deployment and encouraging learned active perception behaviors. Our method includes a shuttlecock prediction model, constrained reinforcement learning for robust motion control, and integrated system identification techniques to enhance deployment readiness. Extensive experimental results in a variety of environments validate the robot's capability to predict shuttlecock trajectories, navigate the service area effectively, and execute precise strikes against human players, demonstrating the feasibility of using legged mobile manipulators in complex and dynamic sports scenarios.

# **INTRODUCTION**

Human sport competitions like badminton pose substantial challenges for players due to the complex interplay required between footwork and upper limb movements. Competent players must develop advanced loco-manipulation skills to effectively cover the extensive court area, complemented by precise hand-eye coordination to anticipate and correctly hit the shuttlecock. Players must predict the shuttlecock's most likely incoming trajectory to synchronize the timing, location, angle, and velocity for strikes that achieve a successful return.

This complex interplay of perception, locomotion, and manipulation makes such sports applications a formidable challenge for developing advanced unified skills for legged mobile manipulation systems as it tackles deficiencies in current paradigms for controllers and hardware. The main challenge for such robots involves balancing rapid and responsive locomotion with accurate arm movements. Although the robot's high number of degrees of freedom (DoFs) in principle allows for agile movements, realizing such potential in practice depends heavily on the control algorithms.

This balancing act is further complicated by the limitations of commercial onboard camera systems, which often must compromise between frame rate, angular resolution, field of view (FOV), and transmission delay, in contrast to the human eye, which has much better motion stabilization, adjustable focus and more sophisticated information processing. To attain similar performance, digital cameras on robots require a perception-aware controller that moves the camera using smooth motions while keeping the target in the FOV.

Various methods have been used for athletic robot control in prior works. Model-based control methods have been utilized for executing complex maneuvers such as front/backflips [\[1–](#page-12-0)[3\]](#page-12-1) and object throwing [\[4,](#page-12-2) [5\]](#page-12-3). These techniques either require model simplifications or only allow local feedback control around pre-optimized trajectories. Recently, noteworthy progress in Reinforcement Learning (RL)-based robot running [\[6–](#page-12-4)[9\]](#page-12-5) and parkour [\[10–](#page-12-6)[13\]](#page-12-7) has demonstrated the potential of learning-based methods to advance robot agility. However, these developments are primarily restricted to locomotion in static scenes. Moreover, the exploration of pedipulation – namely employing robots' feet as manipulators – through RL has uncovered promising approaches to enhance robots' applicability in interactive sports [\[14,](#page-12-8) [15\]](#page-12-9). Despite its potential, this method is generally characterized by a limited operational reach, potentially curtailing its utility in activities where extensive spatial interaction is essential.

In terms of application, one of the closest sports is table tennis, which has been extensively researched for both accuracy [\[16](#page-12-10)[–18\]](#page-13-0) and strategy [\[16,](#page-12-10) [19\]](#page-13-1), primarily using fixed-base or gantry manipulators with external vision systems. In contrast, our work emphasizes wholebody visuomotor skills and relies solely on onboard perception, integrating both legged locomotion and arm swinging — an approach that more closely mirrors human gameplay.

Specifically for legged manipulators, to date, the integration of locomotion and manipulation within the realm of legged manipulation controllers often sees these tasks being treated as distinct entities [\[20–](#page-13-2) [26\]](#page-13-3). This decoupled approach simplifies the optimization process; however, it imposes substantial constraints on the robot's range of motion and its agility, which are critical factors in dynamic environments. Several recent studies have deviated from this traditional decoupling paradigm, albeit with modifications that either maintain a slow-moving state [\[27\]](#page-13-4) or focus solely on self-manipulation tasks [\[28\]](#page-13-5), which does not fully exploit the manipulation capabilities of legged robots in more

**Fig. 1.** Deployment of the badminton control policy on our legged manipulator. Our system operates entirely on onboard perception and computation for shuttlecock detection, trajectory prediction, and limb control. It has been tested in various environments, including the lab, a historic machine hall, and outdoor settings.

dynamic settings. Conversely, there has been noteworthy progress in achieving dynamic motions through imitation learning. Techniques developed for simulated agents playing tennis [\[29\]](#page-13-6) or humanoid robots engaging in generic imitative behaviors [\[30,](#page-13-7) [31\]](#page-13-8) have showcased the potential of using demonstration data to inform robotic control systems while requiring demonstrations from agents of similar morphology. A recent approach involving task-space imitation learning on quadrupedal mobile manipulator platforms has demonstrated dynamic manipulation capabilities with the arms and grippers while not fully leveraging the locomotive potential of the legs in hardware deployments [\[32\]](#page-13-9).

Another major challenge is the trade-off between active perception behavior and agile motion control. Privileged learning has been employed in scenarios where a student policy reconstructs privileged teacher observations based on past interactions and perception history [\[33](#page-13-10)[–35\]](#page-13-11). In this framework, the teacher policy observes privileged information and is not incentivized to learn active perception behaviors, namely, visiting trajectories are informative for decoding the privileged observations. This results in an information gap between the teacher and student policies. Some recent works incorporating perception directly into the RL training loop represent an advancement. For instance, neural rendering techniques utilize a learned perception model where the latent representations can be efficiently integrated within the training process [\[10,](#page-12-6) [36\]](#page-13-12). This approach has shown efficacy in structured and static scenes, enabling the potential learning of active exploration behaviors. Furthermore, the emergence of active perception behaviors has been documented in recent research [\[37,](#page-13-13) [38\]](#page-13-14). However, the emergent active perception behavior was not quantitatively evaluated in these works.

Additional practical deployment challenges include the constraints on electrical current supply to robot actuators, the necessity for precise system identification for accurate simulation modeling, and the need to mitigate perception and communication delays. These elements collectively pose barriers to developing robots that can play badminton at a level comparable to human players.

To address the complex control challenges encountered in real-scale badminton games, we developed a unified RL-based controller trained in simulation, that controls both the base locomotion and the arm manipulative actions. Extending previous work [\[39\]](#page-13-15), this integrated approach leverages all DoFs of the robot to track target End-Effector (EE) states at specified points in time, enabling effective responses to the shuttlecock's incoming trajectory. To prepare the control policy for consecutive shuttlecock hits and learn post-swing follow-through behaviors, we implemented multiple swing targets that are 2 s apart per learning episode. To ensure that the Markov Decision Process (MDP) is well-posed for the critic to estimate the value function, we employed an asymmetric actor-critic formulation [\[40\]](#page-13-16). We incorporated a parameterized shuttlecock perception model based on real-world camera data in the training loop. This model captures the effect of robot motion on perception quality by accounting for both single-frame object tracking errors and final interception predictions, which reduces the perception sim-to-real gap and allows the robot to learn perception-driven behaviors that are also effective on the hardware. This perception model enabled us to reward the policy based on the final perception error and avoids the need for hard-coded orientation strategies, preserving motion efficiency. As an example, the robot may pitch up to keep the shuttlecock in the camera FOV until it needs to pitch down again to swing the racket. The RL algorithm balances the trade-off between agile control and accurate shuttlecock perception by optimizing the policy's overall ability to hit the shuttlecock in simulation. Furthermore, the system integrates shuttlecock prediction [\[41\]](#page-13-17), constrained RL[\[42\]](#page-13-18), state estimation[\[43,](#page-13-19) [44\]](#page-13-20), and system identification on the manipulator dynamics[\[45–](#page-13-21)[47\]](#page-13-22) to facilitate the hardware deployment of the trained control policy.

Movie 1. Summary of the results and the method. The video demonstrates our approach for enabling a legged mobile manipulator robot to play badminton through coordinated control. It showcases how our training pipeline produced policies that balance mobility requirements with rapid arm movements for successful shuttlecock returns. The video also highlights our active perception framework, which incorporated real-world camera noise models into reinforcement learning to develop perception-aware behaviors. This allowed the robot to track fast-moving shuttlecocks while traversing the court for interception and return shots. Various experiments with human players illustrated the system's capabilities across different gameplay scenarios.

In this Article, we demonstrate a quadrupedal mobile manipulator that autonomously plays badminton with human opponents using only onboard perception. Through an integrated RL approach that coordinates whole-body motion with perception, the robot adapts its gait patterns based on time and distance constraints to track and intercept shuttlecocks with swing velocities up to 12.06 m s−<sup>1</sup> . Extensive hardware and simulation experiments validate the system's ability to maintain long rallies in collaborative matches with humans, showcasing emergent active perception behaviors and consistent shuttlecock interception within the court, all while balancing stability with agile arm swings.

## **RESULTS**

Movie 1 summarizes the results of the presented work. Our experiments demonstrated the robot's ability to autonomously track, intercept, and return shuttlecocks during gameplay with human opponents. The system successfully coordinated whole-body movements, adapting its posture and gait patterns based on the shuttlecock's trajectory, while maintaining effective visual tracking. We evaluated the robot's performance in aspects including interception success rates, swing velocity tracking, active perception capabilities, and adaptive locomotion strategies.

### **System overview**

The quadrupedal mobile manipulator robot used in this work consisted of the ANYmal-D base [\[48\]](#page-13-23) and the DynaArm. The robot was equipped with a ZED X stereo camera with global shutters for shuttlecock perception (Fig. 2A). The badminton racket was oriented at a 45-degree angle with respect to the wrist joint, which proved to be the most effective configuration based on early simulation tests of various orientations.

For deployment, the robot's state estimation operated at a frequency of 400 Hz, and the robot control policy updated observations and sent

**Fig. 2. System overview.** The legged mobile manipulator consists of a quadrupedal base and a dynamic arm. We additionally mounted a stereo camera with global shutters. The robot controller system receives the shuttle position computed in the camera frame, predicts the interception position, and feeds it to the RL policy along with the robot proprioception observations. The policy controls all 18 robot drives by producing joint commands.

joint position commands at a rate of 100 Hz, as illustrated in Fig. 2. The system's perception included shuttle position measurement, state estimation, and trajectory prediction. It ran asynchronously at 60 Hz on a Jetson AGX Orin module. Further details are available in the Materials and Method section.

## Collaborative game with human players

Collaborative games were held between the robot and amateur players to validate the system's capability to play real-scale badminton and maintain long rallies, as shown in Movie 1. Throughout the games, ANYmal was able to respond appropriately to the incoming shuttle with various velocity and landing positions, albeit with some failures to return them. The perception module took on average 0.357 s after the opponent had hit the shuttlecock to register trajectories for interception. This left on average 0.654 s until the shuttle trajectory crossed the target interception height of 1.25 m above the robot base height. The fastest hit from the policy was 0.367 s after the interception position was computed.

The robot was capable of consecutive hits. Multiple rallies were documented in Movie 1, with a streak of 10 shots in a single rally. The policy also demonstrated the emergent behavior of moving back near the center of the court after each hit, similar to how human players prepare for the next hits.

#### **Badminton motor skills**

To evaluate the effectiveness of our control policy, we measured the overall success rate of the robot in hitting shuttlecocks that would land at various distances. Fig. 3A indicates the simulated and hardware success rates in different parts of the badminton court, with the robot always starting at the center. The shuttlecock trajectories in this evaluation followed the same initial velocity, and the initial positions were shifted such that the landing locations were distributed as in the figure. We assessed in simulation the policy's ability to intercept the shuttlecock with increasing levels of difficulty, each level corresponding to a qualitative improvement of the badminton skill.

#### Level I - position tracking

Given ground truth perception, we evaluated the percentage of hits that reached the interception position within 0.1 m - the approximate distance between the racket's center and its edge - at the commanded swing time. In the service area, the simulated results indicated the robot could intercept the incoming shuttlecock with a negligible failure rate.

#### Level II - perception error

To assess the robot's ability to predict the shuttle trajectory and intercept it successfully, we added the perception error to the success criteria for this level of evaluation. The perception error measures the position difference between the ground truth shuttlecock position in the simulation and the Extended Kalman Filter (EKF)-estimated position at the time of the swing. To achieve a small error, the control policy had to command the robot to maintain sight on the shuttlecock for a substantial duration based on the measurement noise model while accurately tracking the racket state at the commanded point in time. This level represents the robot's ability to predict and intercept the shuttlecock successfully. This task became particularly challenging at the borders of the service court and when the shuttlecock landed directly behind the robot, as indicated by the lower success rate in Fig. 3A in these regions. We attributed this difficulty to the robot's rectangular FOV, which synergized well with base tilting maneuvers (shown in Fig. 1) to extend visual tracking of the shuttlecock. However, when the shuttlecock approached from directly overhead or behind, the robot had to pitch directly upwards, making it substantially more challenging to maintain continuous visual contact.

Additionally, we reported the EE velocity tracking error at interception when the robot attempted to hit shuttlecocks landing further from the court center along lateral, longitudinal (front-back), and diagonal directions. The velocity tracking accuracy degraded when shuttlecock landings occurred beyond 2.5 m from the court center in the lateral and diagonal directions and beyond 2.0 m in the longitudinal direction.

We validated the success rate evaluation on the hardware by examining how effectively the robot could hit the shuttlecock back over the net. The hardware evaluation was conducted with the robot facing the net while intercepting shuttlecocks approaching from both lateral and frontal directions. This validation confirmed the effectiveness of our deployed control method. Notably, the robot maintained stability and avoided the arm current consumption constraint throughout the hardware experiments, demonstrating its robustness. Video documentation of this evaluation is available in the supplementary movie S1.

#### Fast and accurate racket swing

We further assessed the system's ability to track swing velocity and position commands on hardware with the setup shown in supplementary movie S2. The robot was commanded to swing at varying target velocities to reach position targets at the center of the court at a height of 1.3 m above the starting base height. In Fig. 3B, the executed EE velocity and maximum base angular velocity were plotted against

**Fig. 3.** Assessing the controlled racket swings. (A) We quantitatively evaluated the control system's success rate with progressively more difficult conditions in simulation and partly on the hardware. The EE position error condition evaluates the robot's ability to reach EE targets across the court; the additional perception error condition assesses the robot's ability to perform coordinated perception and control. On the hardware, we evaluated the robot's ability to hit the shuttlecock back across the net at intervals of 0.5 m. The target and robot's initial positions are drawn to their corresponding position on the badminton court. (B) The EE velocity tracking error at interception increases as the robot attempts to hit shuttlecocks landing further from the court center. The error bars represent the standard deviation of the error across 6600 trials. (C) The EE swing velocity and maximum base angular velocity are plotted against the commanded EE velocity, with the error bar spanning from the minimum to maximum values. The robot is able to reach a maximum executed swing velocity of 12.06 ms<sup>-1</sup>, with generally increased base angular velocities at higher EE velocity commands. (D) The distance between the racket sweet spot and the target impact position over relative time. The end-effector minimizes this distance precisely at the commanded impact time. The evaluations in (C-D) are conducted on the hardware.

the commanded velocities. The executed swing velocity generally tracked the commanded velocity below  $10\,\mathrm{m~s^{-1}}$ , with diminishing accuracy at higher velocities. The robot achieved a peak executed velocity of  $12.06\,\mathrm{m~s^{-1}}$  when commanded to swing at  $19\,\mathrm{m~s^{-1}}$ . By comparison, amateur badminton players can reach swing velocity between  $20\text{--}30\,\mathrm{m~s^{-1}}$ , and a recent study on robot table tennis [16] reported an average swing velocity of  $6.83\,\mathrm{m~s^{-1}}$  for their fastest low-level skill. As detailed in the Materials and Methods section, the system operated near its current and joint velocity limits to achieve these commands. Additionally, higher commanded velocities led to increased base angular velocities, indicating a coupling between base attitude control and manipulator swing.

Fig. 3C shows the distance between the racket and the target position around the impact time, with the racket reaching its closest point

precisely at the commanded impact moment. At the commanded swing of  $12 \, \mathrm{m \, s^{-1}}$ , the robot executed swing of mean  $10.8 \, \mathrm{m \, s^{-1}}$ , with a mean position error of  $0.117 \, \mathrm{m}$ , which, in terms of timing, was equivalent to a mere offset of  $0.0108 \, \mathrm{s}$  as the racket moved at the target velocity.

# Active perception behavior

The coordinated interplay between perception and movement was essential for the robot to successfully track and respond to the shuttlecock during gameplay. This section discusses the policy's emergent active perception behavior, evaluates its influences, and compares it to baseline methods. We evaluated the effectiveness of the proposed active perception training and compared our method with two baselines with diverse incoming shuttle trajectories in simulation with the perception noise model. The first baseline, which we call the FOV Reward policy,

**Fig. 4.** Learned Active Perception. (A) A comparison of the perception error heatmap at the time of desired shuttlecock contacts. The policy trained with ground truth shuttlecock states is not incentivized to actively track the shuttlecock, leading to a larger perception error than the other two methods. The heatmaps are overlaid with the badminton court. (B) The perception error statistics collected based on different areas in the badminton court. The line graph shows the mean error, and the error bars represent the standard deviation, computed from 1650 trajectories per cell in each region. (C) The mechanical power required to execute racket swings, normalized by end-effector travel distance. Our proposed method achieves similar mechanical power efficiency to the policy trained without a perception reward, with both outperforming the FOV reward policy. The FOV reward results in inefficient base attitude and locomotion control, as it prioritizes keeping the shuttlecock within the FOV at the expense of energy efficiency. (D-E) An example robot pitch trajectory during a hit. (i) ANYmal observes the shuttlecock circled in red. (ii) ANYmal pitches down while keeping the shuttlecock in the FOV. (iii) ANYmal pitches up to observe the shuttlecock for longer. (iv) ANYmal successfully hits back the shuttlecock, making it reappear in the FOV. Meanwhile, ANYmal returns to the stance posture.

was trained with an explicit reward for keeping the shuttlecock within the FOV, with this reward term tuned to a scale similar to our proposed perception error reward for meaningful comparisons. The second baseline, the No Perception Behavior policy, observed the ground truth shuttlecock states and hence the ground truth interception position, making it not incentivized to learn perception-driven behaviors.

The perception error in this section refers to the mean *L*2-error between the true shuttlecock position and the estimated position from the EKF at the time of the desired racket swing based on the measurement noise model. Overall, the perception error for all three methods

depended on the landing location of the shuttlecock (Fig. 4A), with larger errors occurring near the edge of the badminton service area. Notably, the baseline policy trained with ground truth (GT) perception demonstrated no active perception behavior, resulting in substantially larger errors than the other methods.

We measured the mean and standard deviation of perception errors by regions (a, b, c) of the badminton court (Fig. 4B). Both the proposed method and the policy trained with the FOV reward notably outperformed the no-perception-behavior policy in region c, where the shuttlecock was more likely to exit the robot's FOV. The similarity in perception errors between our proposed method and the-FOV reward baseline was due to the scaling of the FOV reward, which was adjusted to ensure that the policy still performed reasonable locomotion and racket swings. Although we could have increased the FOV reward scaling arbitrarily to further reduce its perception error, doing so would have led to an unfair comparison, as it would have overemphasized FOV tracking at the cost of other important behaviors. Nonetheless, the similar perception error between the two methods suggested that our approach could achieve active perception without relying on explicit FOV rewards.

Another comparison we made was concerning normalized mechanical power, which we define as

$$\varphi = \sum_{\text{all joints}} [\tau \dot{q}]^+ / d \,, \tag{1}$$

where *τ* denotes the joint torque, *q*˙ is the joint velocity, and *d* is the target distance. This metric provided an indication of the policies' energy efficiency. On this scale, our method performed comparably to the no-perception-behavior baseline, with both outperforming the FOVreward baseline (Fig. 4C). Important to note is that the no-perceptionbehavior baseline represented an upper bound of the mechanical power performance, as it solved a subset of the tasks of our proposed method in this comparison. The previously presented metrics indicated that our method balanced between active perception and efficient movement, optimizing both mechanical power and EE tracking.

Fig. 4D and 4E illustrate an instance of the learned active perception behavior observed in our system. The robot started in a stationary position (Fig. 4D, i). Once the interception target was registered (Fig. 4D, ii), the robot first pitched down while keeping the shuttlecock in the upper part of the FOV. Then, it pitched up (Fig. 4D, iii) to reduce the shuttle angular velocity with respect to the camera frame, thus reducing motion blur and keeping the shuttlecock in sight for longer. As soon as the shuttlecock exited the FOV, the robot pitched down again (Fig. 4D, iv) to adjust the robot posture for the racket swing. In this instance, the active perception behavior led to 0.10 s of additional sight of the shuttle flight.

# **Gait adaptation**

Gait adaptation played a critical role in the robot's ability to intercept and return the shuttlecock effectively under varying distances and time constraints. This section discusses the robot's gait patterns in response to different task conditions, as illustrated in Fig. 5. The figure showcases some of the emergent adaptive behavior, with additional comparisons available in the supplementary material and the full video documentation in supplementary movie S3.

We first examined the relationship between gait and the distance the robot needed to cover to intercept the shuttlecock. For this, we conducted hardware experiments where we swept across increasing distances with a fixed time to reach the target (1.6 s) while keeping track of the foot contacts.

At short distances of 0.5 m, no locomotion was necessary. The robot slightly lifted its left front (LF), right front (RF), and right hind (RH) legs to reorient the base while keeping the left hind (LH) in contact with the ground, focusing on precise positioning of the EE for the swing. By the time of the swing, all feet were in contact with the ground (Fig. 5B).

At medium distances of 1.5 m, the robot moved with irregular gait patterns, engaging all four legs in swing phases. The right-side legs, being farther from the target, had notably longer air time than the left. At the time of the swing, three feet remained in contact with the ground (Fig. 5C).

At longer distances of 2.2 m, the robot employed a high-frequency gait resembling galloping between 1.6 s and 0.6 s before the commanded swing. As the swing time approached, the robot adjusted its gait and prepared to lift the right legs (Fig. 5A). The extended flight phase of the right legs enabled an arm extension of 1.0 m in the direction of the target at the time of the swing. One second after swinging, the robot recovered from the dynamic pose and had all four feet in contact.

We also analyzed the gait pattern's dependency on the time the robot had to execute a maneuver. When faced with increased urgency from imminent swing targets, the robot demonstrated adaptive gaits to reach the target while prioritizing safety. The targets in this comparison were located 2 m from the robot's initial base position in *y*-direction.

Furthermore, we observed that the emergent coordination between leg and arm motion emerged under the influence of motion regularization penalties during training. In our training framework, we applied uniform joint torque and acceleration penalty scales across all joints, resulting in the robot prioritizing base tilting and arm usage for hitting nearby targets. By reducing regularization weights on the legs, we could encourage more dynamic leg movements, as demonstrated in the supplementary material fig. S7.

When given 0.8 s to reach the target, the robot stepped with high frequency with the LF, RF, and LH feet while only making one long step with the RH leg. By extending the arm it managed to successfully reach the swing target in time. Under harder time constraints of 0.4 s, it was physically impossible for the robot to reach the target. Despite its attempt, the robot failed to reach the required position, resulting in a missed hit. However, it managed to avoid excessive base limb motion, showcasing the policy's robustness even when faced with unreachable commands and demonstrating awareness of its current physical limitations.

# **DISCUSSION**

We present a legged manipulator system capable of playing badminton using only onboard perception. This system showcases advancements in coordinating legged locomotion with manipulation and balancing limb agility with perception accuracy, highlighting its potential in dynamic and competitive human sports. Through the use of multitarget training and asymmetric actor-critic RL, coupled with a perception model, the robot was able to develop sophisticated human-like badminton behaviors. These include follow-through after hitting the shuttlecock and active perception to enhance shuttle state estimation, achieved without explicit training heuristics.

The robot's performance was extensively evaluated through various hardware experiments, including success rate assessments, tasks involving targets at different distances and under varying time constraints, and verification of active perception behavior. During multiple collaborative games with humans in different environments, the system demonstrated its ability to respond to shuttle shots with varying angles, speeds, and landing locations, achieving ten consecutive shots in a single rally under mildly windy outdoor conditions.

Incorporating the noisy perception model and using the same EKF for both training and deployment establishes a consistent mapping between motion history and expected perception outcomes across sim-

**Fig. 5.** Gait adaptation based on distance and time urgency. (A) The robot reaches targets 2.2 m away from the starting base position in 1.6 s using a galloping-like gait. Near the swing, the longer right foot lift phase and increased *y*-coordinate difference between the base and racket indicate a gait adjustment. (B-C) When targeting nearby positions, the robot barely lifts its feet. It steps to locomote when the target is out of reach. (D-E) Under tight time constraints, the control policy balances between maintaining safety and tracking the target accurately.

ulation and hardware. This provides a means to address a known limitation of privileged learning (teacher-student training): the information gap between the teacher policy trained with perfect perception and the student policy for deployment. In such a framework, the teacher policy has no incentive to learn active perception behaviors because it already has access to perfect observations. The student policy – trained through behavior cloning – only mimics these actions based on partial observations and a latent vector reconstructed from proprioception and perception histories. As a result, neither policy develops active perception behaviors, and a discrepancy arises in the information used for control between the two policies. Our method bridges this gap by encouraging active behaviors through the aforementioned motion-perception mapping of incorporating the EKF also in training. This approach could be further extended by replacing the regressed model with learned perception models to enhance generalizability. Although this may introduce additional training complexity and computational overhead, it presents an exciting direction for future research in improving active perception learning within reinforcement learning frameworks.

We identify several other promising extensions to further enhance the robot's athletic capabilities. Currently, a set of configurable rules determines the swing height, velocity, and orientation. Although the robot's high degree of freedom offers substantial potential for more nuanced racket control, these configurable rules underutilize this capacity. A high-level badminton command policy that adapts swing commands based on the opponent's body movements could improve the robot's ability to maintain rallies and increase its chances of winning. Furthermore, the current control policy is trained to hit interception targets between 0.9-1.4 m above the robot's base using the same side of the racket. Diversifying the swing motion by extending the training scheme could further enhance performance. Moreover, although the policy performs well across shot directions, success rates are lower when returning shuttlecocks that land behind the robot. This limitation stems primarily from perception constraints, as giving the robot ground-truth perception, as shown in Fig. 3A, makes the performance almost symmetric. Having to maintain the shuttlecock within the FOV becomes notably harder when walking backward. A wider FOV camera or an actuated camera pitch joint could mitigate this issue.

Additionally, the current system relies heavily on an EKF applied to a single off-the-shelf stereo camera for shuttlecock state estimation. This approach could be refined by integrating additional sensing modalities, such as torque and sound for impact detection, or incorporating extra RGB, depth, or event-based cameras to enhance the robot's response to physical interactions during more intense gameplay—such as when trying to hit back smash shots. Since human players often predict shuttlecock trajectories by observing their opponents' movements, human pose estimation could also be a valuable modality for improving policy performance.

In conclusion, our research demonstrates that a legged mobile manipulator can autonomously play with human players in a full-scale sport by tightly coupling whole-body maneuver with perception inside a single RL framework. Embedding the parameterized perception model and the same EKF used on hardware in training allows the robot learns to reduce observation error while executing agile strikes. Further simulated experiments hint potential extension of the proposed framework to other legged manipulator morphologies, such as humanoids (supplementary movie S5). Beyond badminton, the method offers a template for deploying legged manipulators in other dynamic tasks where accurate sensing and rapid, whole-body responses are both critical.

# **MATERIALS AND METHODS**

The primary goal of our system was to perceive the shuttlecock, compute the swing target, and execute the swing motion. An overview of our method is presented in Fig. 6.

### **RL-based dynamic whole-body visuomotor skills**

We trained the robot's whole-body maneuvering policy using RL in a high-fidelity simulated environment. This environment included detailed robot dynamics, such as manipulator transmission modeling, joint actuator modeling, and dynamics parameters obtained through system identification. Additionally, constrained RL was used to enforce hardware constraints specific to the robot. To further improve transferability to the physical robot, we applied domain randomization techniques, such as varying friction coefficients, adding base masses, and introducing occasional random pushes. More detailed explanations of the training environment implementation are provided later in this section and in the supplementary material.

To achieve EE swing tracking and allow the policy to learn postswing follow-through behaviors, we simulated six swing targets per episode. However, this implementation made the state value function dependent on the number of hits remaining, though this information should not be made available to the deployed policy actor. To address this, we employed an asymmetric actor-critic approach [\[40\]](#page-13-16) with timebased rewards [\[39\]](#page-13-15) for training, as shown in Fig. 6A. In this setup, the critic network was provided with additional information, such as the number of remaining hits in the episode, to better learn the value function. The actor network only received the robot states and swing target data, which included simulated noise. Table S2 in the supplementary materials presents a detailed list of observations.

We categorized the additional critic observations into two types: enhanced sensor data and preemptive knowledge. Enhanced sensor data included higher-quality versions of existing observations, such as noiseless base and joint states, as well as the EE states. However, the EE states were also derivable via forward kinematics from joint states, albeit with noise. Preemptive knowledge referred to information used to define the MDP that was unavailable during deployment as it depended on the opponent's action, such as the number of remaining targets and the distance between the current target and the next target.

Both categories of observations enhanced the critic network's accuracy in estimating the state value function. Enhanced sensor data reduced stochasticity in the Normalized Penalized Proximal Policy Optimization (N-P3O) policy gradient, and preemptive knowledge completed the set of state variables upon which the value function relied. As detailed in the following section, the training environment was structured around consecutive target swings, with rewards primarily based on EE state tracking. The number of remaining swings in an episode substantially influenced the expected return from the current state. Additionally, the distance between consecutive targets provided the critic with insight into the anticipated motion vigor and tracking precision required, supplying predictive information that further refined return estimation and, consequently, improved policy training.

Individual ablation studies (Fig. 7A and Fig. 7B) show that preemptive knowledge contributed to both the learning process and the final policy performance in EE tracking. The proposed observation format outperformed symmetric privileged training (teacher training in privileged learning [\[33\]](#page-13-10)). This improvement was due to the actor policy in symmetric training developing unnecessary time-dependent behaviors, which introduced additional challenges for value function estimation.

The critic's value function predictions closely match the trajectory return computed from the discounted rewards cumulatively summed from the end, validating the effectiveness of the enhanced critic training

**Fig. 6.** Overview of the training method. (A) Joint control policy training with RL. The policy is trained with an asymmetric actor-critic, with privileged environment states and MDP information given only to the critic network. The policy receives the noised proprioceptive states and shuttlecock perception with simulated noise. (B) The shuttlecock perception module in simulation. The time-adaptive EKF and the path predictor are reused during the deployment. During the training, we used a regressed perception noise model based on real camera noise collected from the hardware. (C) Object perception noise model. The object is in the simulation with the regressed detection probability and measurement noise if it is in the camera FOV.

(Fig. 7C). In contrast, removing the additional critic observations resulted in larger value function errors, leading to worse learning outcomes, as shown in Fig. 7B.

During training, we modeled the perception detection probability and noise as functions of the distance to the shuttlecock, the robot's angular velocity, and whether the shuttlecock was within the robot's FOV. This perception model was regressed from data collected using the robot hardware. We then utilized the same EKF and shuttlecock trajectory prediction module both during training and deployment, ensuring consistency, as depicted in Fig. 6B.

### **Swing tracking**

We employed a time-based swing reward mechanism to incentivize accurate and timely racket swings. This included rewards for position, orientation, and velocity, activated for a single timestep per swing. The orientation reward specifically targeted the angle difference from the

**Fig. 7.** Validation of the training method. (A-B) Ablation studies comparing different observation configurations show that our method consistently outperforms baseline training setups in terms of both convergence time and final end-effector tracking performance. (C) Multiple sample trajectories (represented by different colors) demonstrate that our value prediction closely aligns with the computed trajectory return. (D) The current consumption constraint is respected by policies trained using the N-P3O formulation. (E) The arm joint velocities over the sample swings. (F) The leg joint torques over the sample swings. (D-F) The box and line plots use data from all target positions on the court, each totalling 199,650 trajectories.

normal direction to the racket face.

During training, the robot observed the swing target position as the relative position between the EE and swing target, expressed in the base frame of the robot. This observation formulation helped maintain robust tracking when domain randomization was applied to the arm dynamics, which suggested smaller sim-to-real transfer challenges. As we expected flat terrain during the deployment, we did not implement advanced terrain types or curriculum during the training. An overall flat terrain with small unevenness was used to encourage higher foot-lifting, which helped the sim-to-real transfer. The supplementary materials provide additional training details, including other observations, network architecture, and hyperparameters.

# Perception noise model

To fit the perception noise model, we collected data where the camera moved around while observing a fixed shuttlecock at a known location. The camera's position was tracked using a motion capture system, allowing us to compute the distance and angular velocity between the camera and the shuttlecock. Supplementary movie S4 shows the data collection procedure. The detection probability and the shuttlecock position error were regressed as a linear function of shuttle distance and angular velocity, providing a noise model that could be deployed during the training with negligible computation overhead. An example of the regression is depicted in Fig. 6C.

Both our shuttlecock trajectory generation and the EKF followed the shuttlecock dynamics model [41] with a measured aerodynamic length L of 4.1 m. This aerodynamic length is defined as  $L=2m/\rho SC_D$ , where m is the projectile mass,  $\rho$  is the air density, S is the cross-sectional area, and  $C_D$  is the air drag coefficient.

$$m\frac{d\mathbf{v}}{dt} = m\mathbf{g} - m||\mathbf{v}||\frac{\mathbf{v}}{L}$$
 (2)

where m is the mass of the shuttlecock,  $\mathbf{v}$  is its velocity, and  $\mathbf{g}$  is gravitational acceleration. The process noise and measurement noise configurations are available in the supplementary material.

During training, we integrated the perception noise model, the EKF, and shuttlecock trajectories into the learning loop. We first generated shuttlecock trajectories with random initial states and aerodynamic lengths. For each target swing in the training environment, we sampled a target swing height and a shuttle trajectory from the saved trajectory pool. We padded and translated the trajectory so the shuttle reached the target swing height at the commanded swing time. The shuttle detection and measurement error was sampled using the regressed noise model (Fig.6C) and given to the EKF. Because the same EKF and trajectory prediction module are employed during training and on hardware, the single-frame noise introduced here is filtered identically in both cases, capturing not just per-step measurement errors but also the final interception prediction error. This enabled direct penalization of observation error, rather than imposing hard-coded FOV constraints, allowing the RL algorithm to naturally balance perception accuracy and motion control for active perception learning.

To avoid the computation cost required to rollout the full shuttle prediction, we computed the final target offset linearized with respect to the current state estimation error based on the EKF estimation and the noiseless shuttlecock trajectory. Details on the trajectory distribution and the target offset approximation are presented in the supplementary material.

This approach modeled the perception noise and reused the filter and prediction modules deployed on the hardware to reflect real-world conditions. Note that the perception noise level is subject to the testing site's light conditions and ambient color. We acknowledged that this is a limitation of our approach and that there is an expected perception error difference when deploying the trained policy in a new environment.

However, the learned active perception behavior shown in Fig. 4. would still qualitatively transfer to different environments and decrease the perception error.

### **Training**

The training process used the IsaacGym simulator [49] with the legged\_gym framework [49, 50]. The training used the N-P3O [42], a constrained variant of Proximal Policy Optimization (PPO) algorithm [51]. The policy approached the maximum training reward after around 7500 iterations (Fig. 7A), corresponding to 4.81 h wall-clock time on a single RTX 2080Ti GPU. For deployment, the policy was usually trained for 1 - 2 days for better convergence.

## Perception deployment

In the deployment phase, we implemented several key components to ensure the accurate tracking and striking of the shuttlecock. We utilized a color-based filtering approach for effective shuttlecock tracking in the camera frame. Specifically, we employed the Hue-Saturation-Value (HSV) scale to filter out the shuttlecock's orange color by setting an upper and lower range for the HSV values. This enabled us to isolate the shuttlecock from the background effectively. Using the stereo information provided by the ZED X camera, we then transformed the filtered positions from 2D image coordinates to the robot's map frame.

The map frame – a globally consistent world frame generated by the robot's simultaneous localization and mapping (SLAM) pipeline, distinct from the odometry frame, which can accumulate drift over time – essential for accurate localization and tracking, was derived through the integration of modular sensor fusion (MSF) [44] and CompSLAM [43]. A stable and accurate map frame that properly accounted for the robot's movements was critical, as the shuttlecock's state estimation was filtered within this frame. Any drift in the map frame would have resulted in a noisy shuttlecock estimation or directly led to incorrect interception positions when transformed into the base frame, thereby affecting the swing command observations.

Due to notable base angular velocity during the swing preparation phase, accurate timing information on the shuttle's position was required to determine its position in the world frame. For this purpose, the ZED X camera firmware provided synchronized image timestamps. Camera selection also played a key role in our perception system. We opted for a narrower FOV instead of a wide-angle camera to enhance angular resolution. This choice reduced measurement noise and improved the accuracy of shuttlecock tracking, particularly during high-speed motions where precise angular data is critical. For our perception system, we measured a total of 60-160 ms delay between the camera shutter time and when the shuttle's positions could be computed.

Once the shuttlecock's position in the map frame was obtained, it was processed by an EKF with parameters identical to those used during training. The EKF output a filtered shuttlecock state estimate, enabling trajectory prediction and interception point computation. In both simulation and real-world deployment, if the shuttlecock exited the robot's FOV, the system maintained the last predicted interception position for up to 2 s, during which the robot attempted to strike based on this estimate.

#### Sim-to-real practicalities

Several practical considerations were addressed during hardware deployment. Unlike the actuator network model used for the legs [45], we applied a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize the hardware model parameters [46] in IsaacGym. This approach was chosen because the manipulator we used did not provide accurate torque measurements, and it used quasi direct drives (QDDs), which have more transparent dynamics compared to the series elastic

actuators (SEAs) on the legs. We collected robot joint position trajectories resulting from sine wave command trajectories of varying frequency and swing command trajectories on the hardware. We optimized for the model parameters, including the joint friction, damping, and armature, to match the hardware joint position trajectory under identical commands.

The arm current consumption was limited to 8 A on the ANYmal robot by a fuse. Constrained RL technique N-P3O [\[42\]](#page-13-18) were used to enforce arm current consumption constraints on the robot.

<span id="page-12-11"></span>
$$|I_{\text{total}}| < 8 \,\text{A}\,,\tag{3}$$

where the total current *I*total is calculated by summing the contributions from resistive power, derived from the motor constant *K<sup>m</sup>* and the mechanical power, computed from the torque *τ<sup>i</sup>* and motor velocity *ω<sup>i</sup>* , and dividing by the voltage *V*

$$I_{\text{total}} = \sum_{i=1}^{N} \left(\frac{\tau_i}{K_m}\right)^2 \frac{1}{V} + \sum_{i=1}^{N} \frac{\tau_i \cdot \omega_i}{V}.$$
 (4)

The policy avoided the current limit with the N-P3O constraint implementation, in contrast, the baseline policy without the constraint violated the constraint even with soft over-current penalties. During the hardware deployment, the policy trained with N-P3O never violated the constraint.

The actuator torque and velocity constraints were included as soft penalties in the reward function, as these were also treated as soft constraints on the hardware. Although exceeding these limits wouldn't cause immediate failures, frequent violations could lead to wear or damage. The distributions of maximum arm drive velocity and leg drive torque in our test scenario are shown in Fig. 7E and Fig. 7F, respectively. Fig. 7E indicates fast motion is observed in the wrist drives both in the backswing phase and during the swing, whereas the leg torque usage remained more consistent throughout the phases. During all phases, these values approached their limits but remained within them.

# **Statistical analysis**

Statistical analyses were performed in Python using the NumPy library to compute means and standard deviations. Data were sampled at 100 Hz for simulated experiments and 400 Hz for hardware experiments. The analyses shown in Figs. 3A-B, 4A-C, and 7D-F used 1650 trajectories per target position, with perturbed initial robot joint configurations. For the normalized mechanical power computation (Fig. 4C), trajectories with zero target distance were excluded to prevent division by zero. The mean training rewards shown in Fig. 7A-B were averaged across 4096 parallel training environments using three random seeds. All plots in Fig. 7 were generated from simulated data using Matplotlib, with a convolution filter of window size 50 applied to Fig. 7A for smoothing.

# **ACKNOWLEDGMENTS**

Acknowledgements: We thank Changan Chen and Kaixian Qu for dedicating extensive time as the robot's opponents. We also thank Nikita Rudin for initial project discussions and Fabian Tischhouser for extensive hardware engineering support. We are grateful to Dylan Vogal for his insightful feedback on Movie 1. Additional thanks to Junzhe He, Tianxu An, Jan Preisig, Fan Yang, Mayank Mittal, Yanqing Shen, Takahiro Miki, Filip Bjelonic, and Jia-Ruei Chiu for their assistance in experiments and data collection, and to Eris Sako for hardware insights. We acknowledge Kento Kawaharazuka, David Hoeller, and Joonho Lee for project discussions, and Emre Elbir, Ennio Schnieder, Andreas Binkert, Andri Graf, Johann Schwabe, Flurin Schindele, and Laurin Schmid for CAD and infrastructure support. We also used ChatGPT

and DeepSeek to assist with revising and refining the language in this paper. Funding: This work was supported by Intel Labs, the Max Planck ETH Center for Learning Systems, and the National Centre of Competence in Research Robotics (NCCR Robotics). Additionally, this work was conducted as part of ANYmal Research, a community dedicated to advancing legged robotics. Author contributions: Y.M.: Conceptualization, simulation, sensor selection, data collection, policy training, experiments, investigation, analysis, visualization, writing. A.C.: Sensor selection, substantial revision. F.F.: Initial project discussions, revision. M.H.: Initial task scope, sensor selection, resources, supervision, revision. Competing interests: There are no competing interests to declare. Data and materials availability: All (other) data needed to evaluate the conclusions in the paper are present in the paper or the Supplementary Materials. The data for this study have been deposited in the database DOI: [10.5281/zenodo.15242151.](https://doi.org/10.5281/zenodo.15242151)

# **REFERENCES**

- <span id="page-12-0"></span>1. M. H. Raibert, *Legged robots that balance* (MIT press) (1986).
- 2. S. Le Cleac'h, T. A. Howell, S. Yang, C.-Y. Lee, J. Zhang, A. Bishop, M. Schwager, Z. Manchester, Fast contact-implicit model predictive control. *IEEE Transactions on Robotics* **40**, 1617–1629 (2024).
- <span id="page-12-1"></span>3. B. Katz, J. Di Carlo, S. Kim, Mini cheetah: A platform for pushing the limits of dynamic quadruped control, in *IEEE International Conference on Robotics and Automation* (IEEE, 2019), pp. 6295–6301.
- <span id="page-12-2"></span>4. Y. Liu, A. Billard, Tube acceleration: robust dexterous throwing against release uncertainty. *IEEE Transactions on Robotics* (2024).
- <span id="page-12-3"></span>5. J.-R. Chiu, J.-P. Sleiman, M. Mittal, F. Farshidian, M. Hutter, A Collision-Free MPC for Whole-Body Dynamic Locomotion and Manipulation, in *IEEE International Conference on Robotics and Automation* (IEEE, 2022), pp. 4686–4693.
- <span id="page-12-4"></span>6. D. Crowley, J. Dao, H. Duan, K. Green, J. Hurst, A. Fern, Optimizing bipedal locomotion for the 100m dash with comparison to human running, in *IEEE International Conference on Robotics and Automation* (IEEE, 2023), pp. 12205–12211.
- 7. G. Ji, J. Mun, H. Kim, J. Hwangbo, Concurrent training of a control policy and a state estimator for dynamic and robust legged locomotion. *IEEE Robotics and Automation Letters* **7** (2), 4630–4637 (2022).
- 8. G. B. Margolis, G. Yang, K. Paigwar, T. Chen, P. Agrawal, Rapid locomotion via reinforcement learning. *The International Journal of Robotics Research* **43** (4), 572–587 (2024).
- <span id="page-12-5"></span>9. T. He, C. Zhang, W. Xiao, G. He, C. Liu, G. Shi, Agile but safe: Learning collision-free high-speed legged locomotion, in *Robotics: Science and Systems* (Delft, Netherlands, 2024).
- <span id="page-12-6"></span>10. D. Hoeller, N. Rudin, D. Sako, M. Hutter, Anymal parkour: Learning agile navigation for quadrupedal robots. *Science Robotics* **9** (88), eadi7566 (2024).
- 11. X. Cheng, K. Shi, A. Agarwal, D. Pathak, Extreme parkour with legged robots, in *IEEE International Conference on Robotics and Automation* (IEEE, 2024), pp. 11443–11450.
- 12. Z. Zhuang, Z. Fu, J. Wang, C. Atkeson, S. Schwertfeger, C. Finn, H. Zhao, Robot Parkour Learning, in *Conference on Robot Learning* (2023).
- <span id="page-12-7"></span>13. K. Caluwaerts, A. Iscen, J. C. Kew, W. Yu, T. Zhang, D. Freeman, K.-H. Lee, L. Lee, S. Saliceti, V. Zhuang, *et al.*, Barkour: Benchmarking animal-level agility with quadruped robots. *arXiv:2305.14654* (2023).
- <span id="page-12-8"></span>14. Y. Ji, G. B. Margolis, P. Agrawal, Dribblebot: Dynamic legged manipulation in the wild, in *IEEE International Conference on Robotics and Automation* (IEEE, 2023), pp. 5155–5162.
- <span id="page-12-9"></span>15. T. Haarnoja, B. Moran, G. Lever, S. H. Huang, D. Tirumala, J. Humplik, M. Wulfmeier, S. Tunyasuvunakool, N. Y. Siegel, R. Hafner, M. Bloesch, K. Hartikainen, A. Byravan, L. Hasenclever, Y. Tassa, F. Sadeghi, N. Batchelor, F. Casarini, S. Saliceti, C. Game, N. Sreendra, K. Patel, M. Gwira, A. Huber, N. Hurley, F. Nori, R. Hadsell, N. Heess, Learning agile soccer skills for a bipedal robot with deep reinforcement learning. *Science Robotics* **9** (89), eadi8022 (2024).
- <span id="page-12-10"></span>16. D. B. D'Ambrosio, S. Abeyruwan, L. Graesser, A. Iscen, H. B. Amor, A. Bewley, B. J. Reed, K. Reymann, L. Takayama, Y. Tassa, K. Choro-

- manski, E. Coumans, D. Jain, N. Jaitly, N. Jaques, S. Kataoka, Y. Kuang, N. Lazic, R. Mahjourian, S. Q. Moore, K. Oslund, A. Shankar, V. Sindhwani, V. Vanhoucke, G. Vesom, P. Xu, P. R. Sanketi, Achieving human level competitive robot table tennis. *Computing Research Repository (CoRR)* **abs/2408.03906** (2024).
- 17. T. Ding, L. Graesser, S. Abeyruwan, D. B. D'Ambrosio, A. Shankar, P. Sermanet, P. R. Sanketi, C. Lynch, Learning high speed precision table tennis on a physical robot, in *IEEE/RSJ International Conference on Intelligent Robots and Systems* (IEEE, 2022), pp. 10780–10787.
- <span id="page-13-0"></span>18. H. Ma, D. Büchler, B. Schölkopf, M. Muehlebach, Reinforcement learning with model-based feedforward inputs for robotic table tennis. *Autonomous Robots* **47** (8), 1387–1403 (2023).
- <span id="page-13-1"></span>19. S. W. Abeyruwan, L. Graesser, D. B. D'Ambrosio, A. Singh, A. Shankar, A. Bewley, D. Jain, K. M. Choromanski, P. R. Sanketi, i-sim2real: Reinforcement learning of robotic policies in tight human-robot interaction loops, in *Conference on Robot Learning* (2023), pp. 212–224.
- <span id="page-13-2"></span>20. H. Ferrolho, V. Ivan, W. Merkt, I. Havoutis, S. Vijayakumar, Roloma: Robust loco-manipulation for quadruped robots with arms. *Autonomous Robots* **47** (8), 1463–1481 (2023).
- 21. S. Zimmermann, R. Poranne, S. Coros, Go fetch!-dynamic grasps using boston dynamics spot with external robotic arm, in *IEEE International Conference on Robotics and Automation* (IEEE, 2021), pp. 4488–4494.
- 22. Y. Ma, F. Farshidian, T. Miki, J. Lee, M. Hutter, Combining learningbased locomotion policy with model-based manipulation for legged mobile manipulators. *IEEE Robotics and Automation Letters* **7** (2), 2377–2384 (2022).
- 23. J.-P. Sleiman, F. Farshidian, M. V. Minniti, M. Hutter, A unified mpc framework for whole-body dynamic locomotion and manipulation. *IEEE Robotics and Automation Letters* **6** (3), 4688–4695 (2021).
- 24. N. Yokoyama, A. Clegg, J. Truong, E. Undersander, T.-Y. Yang, S. Arnaud, S. Ha, D. Batra, A. Rai, Asc: Adaptive skill coordination for robotic mobile manipulation. *IEEE Robotics and Automation Letters* **9** (1), 779–786 (2023).
- 25. M. Liu, Z. Chen, X. Cheng, Y. Ji, R. Qiu, R. Yang, X. Wang, Visual Whole-Body Control for Legged Loco-Manipulation, in *Conference on Robot Learning* (2024).
- <span id="page-13-3"></span>26. J. Dao, H. Duan, A. Fern, Sim-to-Real Learning for Humanoid Box Loco-Manipulation, in *IEEE International Conference on Robotics and Automation* (2024), pp. 16930–16936.
- <span id="page-13-4"></span>27. Z. Fu, X. Cheng, D. Pathak, Deep whole-body control: learning a unified policy for manipulation and locomotion, in *Conference on Robot Learning* (2023), pp. 138–149.
- <span id="page-13-5"></span>28. Y. Ma, F. Farshidian, M. Hutter, Learning arm-assisted fall damage reduction and recovery for legged mobile manipulators, in *IEEE International Conference on Robotics and Automation* (IEEE, 2023), pp. 12149–12155.
- <span id="page-13-6"></span>29. H. Zhang, Y. Yuan, V. Makoviychuk, Y. Guo, S. Fidler, X. B. Peng, K. Fatahalian, Learning Physically Simulated Tennis Skills from Broadcast Videos. *ACM Trans. Graph.* (2023).
- <span id="page-13-7"></span>30. Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, C. Finn, HumanPlus: Humanoid Shadowing and Imitation from Humans, in *Conference on Robot Learning* (2024).
- <span id="page-13-8"></span>31. X. Cheng, Y. Ji, J. Chen, R. Yang, X. Wang, Expressive whole-body control for humanoid robots. *arXiv:2402.16796* (2024).
- <span id="page-13-9"></span>32. H. Ha, Y. Gao, Z. Fu, J. Tan, S. Song, UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers. *arXiv:2407.10353* (2024).
- <span id="page-13-10"></span>33. J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, M. Hutter, Learning quadrupedal locomotion over challenging terrain. *Science robotics* **5** (47), eabc5986 (2020).
- 34. T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, M. Hutter, Learning robust perceptive locomotion for quadrupedal robots in the wild. *Science robotics* **7** (62), eabk2822 (2022).
- <span id="page-13-11"></span>35. A. Agarwal, A. Kumar, J. Malik, D. Pathak, Legged locomotion in challenging terrains using egocentric vision, in *Conference on Robot Learning* (2023), pp. 403–415.
- <span id="page-13-12"></span>36. D. Hoeller, N. Rudin, C. Choy, A. Anandkumar, M. Hutter, Neural scene representation for locomotion on structured terrain. *IEEE Robotics and*

- *Automation Letters* **7** (4), 8667–8674 (2022).
- <span id="page-13-13"></span>37. C. Schwarke, V. Klemm, M. Van der Boon, M. Bjelonic, M. Hutter, Curiosity-driven learning of joint locomotion and manipulation tasks, in *Conference on Robot Learning* (2023), vol. 229, pp. 2594–2610.
- <span id="page-13-14"></span>38. O. E. L. Team, A. Stooke, A. Mahajan, C. Barros, C. Deck, J. Bauer, J. Sygnowski, M. Trebacz, M. Jaderberg, M. Mathieu, *et al.*, Openended learning leads to generally capable agents. *arXiv:2107.12808* (2021).
- <span id="page-13-15"></span>39. N. Rudin, D. Hoeller, M. Bjelonic, M. Hutter, Advanced skills by learning locomotion and local navigation end-to-end, in *IEEE/RSJ International Conference on Intelligent Robots and Systems* (IEEE, 2022), pp. 2497– 2503.
- <span id="page-13-16"></span>40. L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, P. Abbeel, Asymmetric Actor Critic for Image-Based Robot Learning, in *Robotics: Science and Systems* (2018).
- <span id="page-13-17"></span>41. C. Cohen, B. D. Texier, D. Quéré, C. Clanet, The physics of badminton. *New Journal of Physics* **17** (6), 063001 (2015).
- <span id="page-13-18"></span>42. J. Lee, L. Schroth, V. Klemm, M. Bjelonic, A. Reske, M. Hutter, Evaluation of constrained reinforcement learning algorithms for legged locomotion. *arXiv:2309.15430* (2023).
- <span id="page-13-19"></span>43. S. Khattak, H. Nguyen, F. Mascarich, T. Dang, K. Alexis, Complementary multi–modal sensor fusion for resilient robot pose estimation in subterranean environments, in *International Conference on Unmanned Aircraft Systems* (IEEE, 2020), pp. 1024–1029.
- <span id="page-13-20"></span>44. S. Lynen, M. W. Achtelik, S. Weiss, M. Chli, R. Siegwart, A robust and modular multi-sensor fusion approach applied to MAV navigation, in *IEEE/RSJ International Conference on Intelligent Robots and Systems* (IEEE, 2013), pp. 3923–3929.
- <span id="page-13-21"></span>45. J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, M. Hutter, Learning agile and dynamic motor skills for legged robots. *Science Robotics* **4** (26), eaau5872 (2019).
- <span id="page-13-27"></span>46. F. Bjelonic, F. Tischhouser, M. Hutter, Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots. *arXiv preprint arXiv:2509.06342* (2025).
- <span id="page-13-22"></span>47. J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, V. Vanhoucke, Sim-to-real: Learning agile locomotion for quadruped robots. *arXiv:1804.10332* (2018).
- <span id="page-13-23"></span>48. M. Hutter, C. Gehring, D. Jud, A. Lauber, C. D. Bellicoso, V. Tsounis, J. Hwangbo, K. Bodie, P. Fankhauser, M. Bloesch, R. Diethelm, S. Bachmann, A. Melzer, M. Hoepflinger, ANYmal - a highly mobile and dynamic quadrupedal robot, in *IEEE/RSJ International Conference on Intelligent Robots and Systems* (2016), pp. 38–44.
- <span id="page-13-24"></span>49. V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, G. State, Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning, in *Neural Information Processing Systems Datasets and Benchmarks Track* (2021).
- <span id="page-13-25"></span>50. N. Rudin, D. Hoeller, P. Reist, M. Hutter, Learning to walk in minutes using massively parallel deep reinforcement learning, in *Conference on Robot Learning* (2022), pp. 91–100.
- <span id="page-13-26"></span>51. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, Proximal Policy Optimization Algorithms. *arXiv:1707.06347* (2017).
- <span id="page-13-28"></span>52. G. Bradski, The OpenCV Library. *Dr. Dobb's Journal of Software Tools* (2000).
- <span id="page-13-29"></span>53. M. Andrychowicz, A. Raichuk, P. Stanczyk, M. Orsini, S. Girgin, ´ R. Marinier, L. Hussenot, M. Geist, O. Pietquin, M. Michalski, *et al.*, What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study, in *International Conference on Learning Representations* (2021).

# **SUPPLEMENTARY MATERIALS**

# **Nomenclature**

| Symbol     | Discription                                     |
|------------|-------------------------------------------------|
| q, q˙      | Joint position, joint velocity                  |
| ˙qˆ<br>qˆ, | Motor position, motor velocity                  |
| I          | Current                                         |
| V          | Voltage                                         |
| τ          | Torque                                          |
| Km         | Motor constant                                  |
| p          | Body position                                   |
| v          | Body velocity                                   |
| a          | Body acceleration                               |
| ω          | Angular velocity                                |
| r          | reward                                          |
| n          | Normal vector                                   |
| T          | Racket swinging time                            |
| tk         | Current time                                    |
| σ          | Reward sensitivity factor                       |
| SC         | Cosine similarity                               |
| L          | Aerodynamic length                              |
| ϵ          | Perception error                                |
| φ          | Normalized mechanical power                     |
| D          | Detecting the shuttlecock, binary               |
| T          | The environment is given a swing target, binary |

#### **Deployment Interception criteria**

During hardware experiments, we qualified the predicted shuttlecock trajectory to intercept if it intersected both rectangles (i) and (ii) in fig. S1. In this context, only the orange trajectory qualified, as it crossed both rectangles, whereas the other two trajectories only intersected one rectangle each. Rectangle (i) was positioned 1.55 m above the ground, corresponding to the height of a standard badminton net, and rectangle (ii) was at ground level. Extending the rectangles to cover the entire single-player court would have aligned the scoring qualifications with formal human matches. However, we limited the region to the service area as we expected the robot to have a higher probability of successfully returning the shuttlecock in this area (Fig. 3).

# **Belt Transmission Modeling**

The manipulator used in this project, DynaArm (shown in fig. [S2\)](#page-14-0), features a belt transmission system that controls the elbow flexion joint via a motor located at the shoulder with a gear ratio of 1:1. This transmission setup is not natively supported in the IsaacGym simulator and introduces major changes to the robot's dynamics. To address this, we implemented a serial-DynaArm conversion to account for these dynamic differences. Here, *δq*˜ and *δ* ˜*q*˙ represent the motor errors, which are distinct from the joint errors *δq* and *δq*˙. In our case, the desired joint velocities *q*˙ *des* were set to zero for all joints.

**Fig. S1.** Shuttlecock trajectory qualification heuristics for hardware deployment. The robot only hits back shuttlecocks with flight trajectories crossing the badminton service area both at ground height and 1.55m above the ground (e.g. the orange trajectory). The swing time is computed based on the time that the trajectory crosses a configurable height.

<span id="page-14-0"></span>**Fig. S2.** Motor names of the DynaArm manipulator.

<span id="page-15-0"></span>**Table S1.** Perception Parameters

|                 | Parameter            | Value                                    |
|-----------------|----------------------|------------------------------------------|
|                 | Camera fps           | 60                                       |
| Camera          |                      | 1e-2 s (indoor, only one floodlight)     |
|                 | Exposure             | 4.17e-3 s (indoor, good light condition) |
|                 |                      | 8.33e-5 s (outdoor, sunny)               |
|                 | Gain                 | 100                                      |
| Color<br>Filter | Hue                  | <5 or >176                               |
|                 | Saturation           | >60                                      |
|                 | Value                | >160                                     |
| EKF             | Process noise std    | 2e-3                                     |
|                 | Measurement noise    | 4e-2                                     |
|                 | Reset time threshold | 0.2 s                                    |

<span id="page-15-1"></span>**Fig. S3.** Distribution of the precomputed shuttlecock trajectory duration.

$$\delta \tilde{q}_{EL\_FLE} = (q_{EL\_FLE}^{des} - q_{EL\_FLE}) + (q_{SH\_FLE}^{des} - q_{SH\_FLE}) \quad \textbf{(S1)}$$

$$\delta \dot{q}_{EL\_FLE} = (\dot{q}_{EL\_FLE}^{des} - \dot{q}_{EL\_FLE}) + (\dot{q}_{SH\_FLE}^{des} - \dot{q}_{SH\_FLE}) \quad (S2)$$

Once the torques were computed from the correct motor errors, we added the motor torque from EL\_FLE motor to SH\_FLE to retrieve the effective joint torque in serial configuration.

## **Perception Parameters**

The parameters for the camera, shuttlecock HSV filter, and the EKF are listed in table [S1.](#page-15-0) The HSV values are specified according to the OpenCV convention [\[52\]](#page-13-28).

# **Shuttle Trajectory Sampling**

For policy training, shuttlecock trajectories were pre-sampled randomly, allowing us to efficiently determine the interception location and shuttle position at each timestep during training episodes with minimal computational cost. These trajectories were sampled from the following distribution (in SI units):

<span id="page-15-2"></span>**Table S2.** Training Rewards

| Reward term              | Scale |
|--------------------------|-------|
| EE position tracking     | 6400  |
| EE orientation tracking  | 1200  |
| EE swing velocity        | 1200  |
| perception error         | 3     |
| face the net             | 0.5   |
| torques                  | -1e-5 |
| joint acceleration       | -1e-6 |
| action rate              | -0.03 |
| collision                | -2    |
| joint position limit     | -1    |
| joint torque limit       | -1e-3 |
| stand still if no target | 16    |

$$\begin{aligned} p_{x,t_0} &\sim U(6,7) \\ p_{y,t_0} &\sim U(-2,2) \\ p_{z,t_0} &\sim U(-0.5,2.5) \\ v_{x,t_0} &\sim U(-19,-13) \\ v_{y,t_0} &\sim U(-3,3) \\ v_{z,t_0} &\sim U(9,15) \end{aligned}$$

This distribution was designed so that shuttlecock trajectories originated near the center of the opponent's court and crossed to the robot's side at an average position of (*px*,*T*, *py*,*T*) ≈ (0, 0) with a height of 1.8 meters. The landing positions were spread across the entire court. The distribution of trajectory durations, from shuttle launch to interception, is shown in fig. [S3.](#page-15-1)

For the evaluations in the Results section, we used the mean values of the distribution to simulate a nominal shuttlecock flight, adjusting the starting position to assess interception performance across different court locations.

### **Training Rewards**

We trained the badminton policy in simulation using the rewards and corresponding scaling factors listed below. Selected reward formulations are provided in Eq. [S3](#page-12-11) to [S6.](#page-16-0) Other reward terms are based on their corresponding *L*2-norms. The reward terms and scales are shown in table [S2.](#page-15-2)

# *Task Rewards*

The EE position tracking reward encouraged the racket's sweet spot to match the target interception point at the time of the swing. It was activated only for a single step per shuttle interception, at *t<sup>k</sup>* = *T*.

$$r_p = \delta(t_k - T) \frac{1}{1 + ||p_{EE}^* - p_{EE}||_2 / \sigma_p}$$
 (S3)

Similarly, the EE orientation tracking reward and the swing velocity reward encouraged accurate orientation and velocity tracking at the same interception timestep. The orientation reward penalized the squared cosine distance between the commanded racket-facing direction and the executed direction. The swing velocity reward was

<span id="page-16-1"></span>**Fig. S4.** The policy's action distribution entropy decreases as the time approaches the swing at timestep=200. Plotted with 16 independent swings.

computed based on the squared difference between the commanded and actual racket sweet-spot velocity.

$$r_q = \delta(t_k - T) \frac{1}{1 + S_C(n_{EE}^* - n_{EE})^2}$$
 (S4)

$$r_v = \delta(t_k - T) \frac{1}{1 + (v_{EE}^* - v_{EE})^2 / \sigma_v}$$
 (S5)

The perception error reward was a temporally dense reward function, calculated based on the ground-truth interception position and the interception position estimated by the EKF shuttlecock state.

<span id="page-16-0"></span>
$$r_{\epsilon} = \frac{1}{1 + ||p_i - p_i^*||_2} \tag{S6}$$

#### *Regularization Rewards*

The impact reward was designed to reduce stomping behavior when the robot traversed the badminton court. It lowered the reward if excessive acceleration occurred in the robot links along the *z*-axis.

$$r_{impact} = \sum_{l}^{links} ||a_{z,link}||^2$$
 (S7)

Additionally, in environments where no shuttlecock trajectories were assigned, the robot was rewarded for standing with the default joint configuration. This encouraged improved behavior during hardware deployment when the shuttle was not observed.

$$r_{stand} = e^{-\frac{1}{\sigma N_{joints}}|q - q*|}$$
 (S8)

# **Observations**

We provided the policy actor with observations that were available on the robot, and additional observations to the critic to help reduce value estimation error. The observations and their descriptions are listed in table [S3.](#page-17-0) Among the critic-only observations, those that provide complete MDP information are highlighted in the grey cells.

## **Training Hyperparameters**

The remaining training hyperparameters are presented in table [S4.](#page-18-0)

### **State-dependent Action Standard Deviation**

We used a state-dependent action standard deviation [\[53\]](#page-13-29) for N-P3O. Our evaluation shows that the action distribution entropy decreased in the timesteps leading up to the swing (fig. [S4\)](#page-16-1), resulting in a minor reward improvement during training.

<span id="page-16-3"></span>**Fig. S5.** Distribution of the time between a human serving the shuttlecock and the completion of our system's perception loop to generate an EE target.

### **Approximating the Estimated Interception**

During training, we estimated the shuttlecock interception position by linearizing the shuttle trajectory prediction [\[41\]](#page-13-17) with respect to the state estimation error at the ground truth shuttlecock state. This simplification reduced the computational cost of full trajectory prediction based on shuttlecock state estimation. The process is outlined in Eq. [S9.](#page-16-2)

<span id="page-16-2"></span>
$$\hat{p}_T \approx p_T + (\hat{p}_{t_k} - p_{t_k}) + (1 - \frac{2\Delta t}{L})(\hat{v}_{t_k} - v_{t_k})(T - t_k)$$
 (S9)

We rejected higher order ∆*t* term due to their small magnitudes.

# **Deflection Error**

The deflection error was calculated based on the difference in outgoing shuttlecock velocity between the expected result of the commanded racket swing (orientation and velocity) and the actual result from the executed swing. For this computation, we assumed a fixed nominal incoming shuttlecock velocity of (*vx*, *vy*, *vz*) = (−4.5, 0.0, −4.5).

We computed the velocity difference between the incoming shuttlecock and the commanded racket velocity as follows:

$$\mathbf{v}_{\text{impact}} = \mathbf{v}_{\text{incoming}} - \mathbf{v}_{\text{target}}$$
 (S10)

Next, we calculated the outgoing shuttlecock velocity, assuming an elastic collision where the reflected racket inertia was much larger than that of the shuttlecock:

$$\begin{aligned} \mathbf{v}_{\text{racket\_normal}} &= (\mathbf{v}_{\text{racket}} \cdot \mathbf{n}_{\text{racket}}) \mathbf{n}_{\text{racket}} \\ \mathbf{v}_{\text{shuttle\_normal}} &= (\mathbf{v}_{\text{impact}} \cdot \mathbf{n}_{\text{racket}}) \mathbf{n}_{\text{racket}} \\ \mathbf{v}_{\text{outgoing}} &= \mathbf{v}_{\text{incoming}} - 2 \mathbf{v}_{\text{shuttle\_normal}} + 2 \mathbf{v}_{\text{racket\_normal}} \end{aligned}$$

The same computation was applied to the executed swing, substituting the measured racket orientation and velocity to calculate the difference in the shuttlecock's outgoing velocity.

#### **Perception Reaction Time**

As noted in the main manuscript, the mean duration for our system to complete the perception loop and produce an EE target was 0.375 s. The distribution of this duration is shown in the histogram in fig. [S5,](#page-16-3) with a minimum duration of 0.217 s and a maximum of 0.517 s.

<span id="page-17-0"></span>**Table S3.** Policy observations

|             | Observation            | Description                                                                                                            |
|-------------|------------------------|------------------------------------------------------------------------------------------------------------------------|
|             | vb                     | Base linear velocity                                                                                                   |
|             | !b                     | Base angular velocity                                                                                                  |
|             | gb                     | Gravity vector in base frame                                                                                           |
|             | h                      | Robot heading vector in court frame                                                                                    |
|             | cmd                    | Racket intercpetion command                                                                                            |
|             | t                      | Time until the interception                                                                                            |
| shared      | q                      | Joint position offset from default configuration                                                                       |
|             | ˙q                     | Joint velocity                                                                                                         |
|             | aprev                  | Previous policy action output                                                                                          |
|             | court<br>p<br>B        | Robot base position (x,y) in the court frame                                                                           |
|             | cam<br>p<br>shuttle    | Shuttlecock position in camera frame                                                                                   |
|             | D                      | Detected the shuttle in this timestep                                                                                  |
|             | thidden                | Time until the shuttlecock is launched                                                                                 |
|             | thiddennext            | Time until the next shuttle is launched                                                                                |
|             | court<br>p<br>EE       | Racket sweet-spot position in the court frame                                                                          |
|             | court<br>v<br>ee       | Racket sweet-spot velocity in the court frame                                                                          |
|             | court<br>n<br>x,racket | Racket facing direction                                                                                                |
|             | cmdnext                | The next interception position, velocity and orientation target                                                        |
| critic-only | Targets remaining      | Number of interception targets remaining in the episode                                                                |
|             | T                      | Whether the robot is commanded to intercept a target or stand still                                                    |
|             | T next                 | Whether the robot is commanded to intercept a target or stand still after the current hit                              |
|             | Domain randomization   | Randomly added base mass, robot friction coefficient                                                                   |
|             | ϵ                      | The difference between the target interception position and the one predicted from the EKF<br>shuttle state estimation |
|             | Is in FOV              | Whether the shuttlecock is in the camera FOV                                                                           |

<span id="page-18-1"></span>**Fig. S6.** Foot contact schedule adaptation to various distances and duration.

<span id="page-18-0"></span>**Table S4.** Training Hyperparameters

| Parameter                   | Value           |
|-----------------------------|-----------------|
| discount factor             | 0.995           |
| GAE lambda                  | 0.975           |
| learning rate               | adaptive        |
| KLD target                  | 0.01            |
| entropy coefficient         | 0.0016          |
| entropy coeff. decay        | 0.99993         |
| num. targets per episode    | 6               |
| control dt (s)              | 0.01            |
| terrain max height diff (m) | 0.06            |
| num. envs                   | 4096            |
| standing env. ratio         | 10%             |
| actor MLP size              | (512, 256, 128) |
| critic MLP size             | (512, 256, 128) |
| network activation          | elu             |
| optimizer                   | AdamW           |

<span id="page-18-2"></span>**Table S5.** Humanoid Training Configurations

| Config        | Changes             |
|---------------|---------------------|
| Robot         | Unitree G1          |
| Num. DoFs     | 23                  |
| Current limit | Removed             |
| Motor model   | From unitree_rl_gym |

### **Additional Gait Adaptation**

Due to space constraints, only a portion of the gait adaptation plot is shown in the main manuscript. The complete plots are provided in fig. [S6.](#page-18-1)

### **Potential Extensions to Other Robots**

We applied our training framework to humanoid robots (Unitree G1) in simulation for badminton. The humanoids demonstrated agile, coordinated whole-body visuomotor skills comparable to ANYmal, as shown in movie S5. We acknowledge that deploying this on hardware presents notable challenges, and this experiment serves only as an early indication of potential future directions. Table [S5](#page-18-2) summarizes the adjusted training configurations; all other settings remained unchanged.

### **Alternative Joint Regularization Settings**

We observed that the robot prioritized different joints if the joint torque and acceleration regularization rewards were scaled differently, resulting in distinct behaviors. As shown in Figure [S7,](#page-19-0) when the policy was trained with reduced leg joint regularization, the robot exhibited greater base displacements for nearby targets, though distant targets that inherently required dynamic leg motion remained largely unaffected. Both regularization terms were reduced to 1/3 of their original values for the reduced regularization setting.

<span id="page-19-0"></span>**Fig. S7.** A comparison of base displacement for various target distances between the default training rewards and reduced leg regularization. (A) Different policies lead to distinct base travel when hitting the same target from the same initial position. (B) The policy with less regularized leg motion shows larger base displacement when hitting nearby targets compared to the default weighting. The line plots and error bars show the mean and standard deviation of the base displacement in 4096 test trajectories.