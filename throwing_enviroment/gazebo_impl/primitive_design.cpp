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
  }

  bool tossingObject(
    const race_basic_motion_control::srv::Behaviors::Request & req,
    race_basic_motion_control::srv::Behaviors::Response & res)
  {
    return executeThrowLikeMotion(req, res, true, "tossing_object");
  }