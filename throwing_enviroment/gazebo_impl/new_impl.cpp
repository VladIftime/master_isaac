bool executeThrowLikeMotion(
    const race_basic_motion_control::srv::Behaviors::Request & req,
    race_basic_motion_control::srv::Behaviors::Response & res,
    const bool do_grasp_and_release,
    const std::string & behavior_name)
  {
    // Local copies (ROS2 req is const; also keeps ROS1 “defaulting” behavior)
    double initial_joint_value = req.initial_joint_value; // 0 => not set
    double final_joint_value   = req.final_joint_value;   // 0 => not set
    double duration            = req.duration;            // 0 => not set
    double releasing_time      = req.releasing_time;      // can be 0 too

    // Choose presets by arm
    std::vector<double> init_joints_pose;
    std::vector<double> initial_joints_pose;
    std::vector<double> end_joints_pose;

    if (req.robot == "left") {
      // ROS1 LEFT
      init_joints_pose = {-1.6, -1.435, -2.3313358465777796, -1.0,  1.5987361113177698, 0.0};

      // if initial_joint_value != 0 => override joint0 and use -1.3 on joint1 (as your ROS1 left)
      if (initial_joint_value != 0.0) {
        initial_joints_pose = {initial_joint_value, -1.3, -2.3313358465777796, -1.0, 1.5987361113177698, 0.0};
      }

      if (final_joint_value == 0.0) final_joint_value = -1.6;

      end_joints_pose = {final_joint_value, -1.881, -0.8647, -0.9, 1.5743544737445276, 0.0};
    }
    else if (req.robot == "right") {
      // ROS1 RIGHT
      init_joints_pose = { 1.6, -1.7235609493651332,  2.3313358465777796, -2.0628696880736292, -1.5987361113177698, 0.0};

      if (initial_joint_value != 0.0) {
        initial_joints_pose = {initial_joint_value, -1.7235609493651332, 2.3313358465777796, -2.0628696880736292, -1.5987361113177698, 0.0};
      }

      if (final_joint_value == 0.0) final_joint_value = 1.6;

      end_joints_pose = {final_joint_value, -1.2773774427226563, 0.8647106329547327, -2.1965824566283167, -1.5743544737445276, 0.0};
    }
    else {
      RCLCPP_WARN(get_logger(), "[%s] Only 'left' or 'right' supported. robot='%s'",
                  behavior_name.c_str(), req.robot.c_str());
      res.result = false;
      return false;
    }

    if (duration == 0.0) duration = 1.0;

    // 1) go to init pose
    directlySetAllJoints(init_joints_pose, req.robot, (req.robot == "right") ? 0.15 : 0.1);
    waiting(0.15);

    // 2) optional grasp
    if (do_grasp_and_release) {
      try {
        std::string picked = req.object_name;
        bool grasp_ok = false;

        if (!picked.empty()) {
          grasp_ok = graspObjectInGazebo(picked, req.robot);
        } else {
          // grasp_ok = graspObjectInGazeboAuto(req.robot, picked);
          // RCLCPP_INFO(get_logger(), "[%s] auto-picked object: [%s]",
          //             behavior_name.c_str(), picked.c_str());
          RCLCPP_WARN(get_logger(), "[%s] No object specified for grasp and auto-grasp not implemented. Skipping grasp.",
                      behavior_name.c_str());

        }

        if (!grasp_ok) {
          RCLCPP_WARN(get_logger(), "[%s] graspObjectInGazebo failed / not implemented",
                      behavior_name.c_str());
          res.result = false;
          return false;
        }
      } catch (const std::exception & ex) {
        RCLCPP_ERROR(get_logger(), "[%s] Exception while grasping: %s",
                    behavior_name.c_str(), ex.what());
        res.result = false;
        return false;
      }
    }

    // 3) optional initial pose override
    if (!initial_joints_pose.empty()) {
      directlySetAllJoints(initial_joints_pose, req.robot, (req.robot == "right") ? 0.25 : 0.1);
      waiting((req.robot == "right") ? 0.25 : 0.1);
    }

    // 4) throw motion
    rclcpp::Time start_time = this->now();
    directlySetAllJoints(end_joints_pose, req.robot, duration);

    // 5) optional release during motion
    if (do_grasp_and_release) {
      while ((this->now() - start_time).seconds() < releasing_time) {
        rclcpp::sleep_for(std::chrono::milliseconds(1));
      }

      try {
        bool rel_ok = releaseObjectInGazebo(req.object_name);
        if (!rel_ok) {
          RCLCPP_WARN(get_logger(), "[%s] releaseObjectInGazebo failed / not implemented",
                      behavior_name.c_str());
        }
      } catch (const std::exception & ex) {
        RCLCPP_ERROR(get_logger(), "[%s] Exception while releasing: %s",
                    behavior_name.c_str(), ex.what());
        res.result = false;
        return false;
      }
    }

    // 7) finish + return to init
    waiting(duration);
    directlySetAllJoints(init_joints_pose, req.robot, 1.1);
    waiting((req.robot == "right") ? 0.1 : 0.15);

    res.result = true;
    return true;
  }[8:54 PM]left arm

init_joints_pose = {-1.6, -1.435, -2.3313358465777796, -1.0,  1.5987361113177698, 0.0};

end_joints_pose = {-1.6, -1.881, -0.8647, -0.9, 1.5743544737445276, 0.0};

 
[8:54 PM]These, together with grasping and realizing behaviour, defined the tossing behavior. duration refer to how fast the robot should execute the trajectory (edited) 
[8:58 PM]bool setAllJointsSim(
    const std::vector<double> & joints,
    const std::string & robot,
    double duration = 0.0)
  {
    if (joints.size() != 6) {
      RCLCPP_WARN( get_logger(), "setAllJointsSim: expected 6 joints, got %zu", joints.size());
      return false;
    }

    // Select action client
    rclcpp_action::Client<FollowJT>::SharedPtr client;
    std::vector<std::string> joint_names;

    if (robot == "left") {
      client = left_arm_client_;
      joint_names = left_joint_names_;
    }
    else if (robot == "right") {
      client = right_arm_client_;
      joint_names = right_joint_names_;
    }
    else {
      RCLCPP_WARN( get_logger(), "SetAllJointsSim: unknown robot '%s'", robot.c_str());
      return false;
    }

    if (!client) {
      RCLCPP_ERROR(get_logger(), "Action client is null");
      return false;
    }

    if (!client->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "Action server not available for %s arm", robot.c_str());
      return false;
    }

    const double min_duration = std::max(0.02, sim_joint_auto_min_duration_s_);
    const double max_duration = std::max(min_duration, sim_joint_auto_max_duration_s_);
    double commanded_duration = duration;
    if (commanded_duration <= 0.0) {
      std::vector<double> current_joints;
      if (getArmJointPositions(robot, current_joints) && current_joints.size() == joints.size()) {
        double max_delta = 0.0;
        for (std::size_t i = 0; i < joints.size(); ++i) {
          max_delta = std::max(max_delta, std::abs(joints[i] - current_joints[i]));
        }

        const double speed = std::max(0.1, sim_joint_auto_max_speed_rad_s_);
        commanded_duration = max_delta / speed;
        commanded_duration = std::clamp(commanded_duration, min_duration, max_duration);
      } else {
        commanded_duration = min_duration;
      }
    } else {
      commanded_duration = std::clamp(commanded_duration, min_duration, max_duration);
    }

    RCLCPP_INFO(
      get_logger(),
      "[IK timing] setAllJointsSim command robot='%s' duration=%.3f s",
      robot.c_str(),
      commanded_duration);

    // Build action goal
    FollowJT::Goal goal;
    goal.trajectory.joint_names = joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = joints;
    point.time_from_start = rclcpp::Duration::from_seconds(commanded_duration);

    goal.trajectory.points.push_back(point);

    // Send goal (fire-and-forget, ROS1-style)
    client->async_send_goal(goal);

    return true;
  }

bool directlySetAllJoints(const std::vector<double>& joints,
                                              const std::string& robot,
                                              double duration = 0.0,
                                              double vel = 1.0,
                                              double acc = 1.0)
  {
    if (joints.size() != 6) {
      RCLCPP_WARN(get_logger(),
        "directlySetAllJoints: expected 6 joints, got %zu", joints.size());
      return false;
    }

    if (runtime_mode_ == RuntimeMode::SIM) {
      return setAllJointsSim(joints, robot, duration);
    } else {
      return setAllJointsReal(robot, joints, vel, acc);
    }
  }


you should first check if the robot can execute this joints' value. 
you may also need to have the directlySetAllJoints() function