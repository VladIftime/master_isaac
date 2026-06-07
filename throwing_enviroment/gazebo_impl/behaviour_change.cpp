bool pingpong2Behavior(
    const race_basic_motion_control::srv::Behaviors::Request & req,
    race_basic_motion_control::srv::Behaviors::Response & res)
  {
    if (req.robot != "right") {
      RCLCPP_WARN(get_logger(), "[pingpong2] Only 'right' arm is supported. robot='%s'", req.robot.c_str());
      res.result = false;
      return false;
    }



    const std::vector<double> initial_joints_pose = {1.6, -1.6, 2.0, 0.0, 0.0, 1.5}; //{1.6, -1.4, 2.2, 0.0, 0.0, 1.5};
    std::vector<double> end_joints_pose;
    double error = 0.0;

    if (!getIkJointSolutionForPose(req.right_target_pose.pose, "right", end_joints_pose, error)) {
      res.result = false;
      return false;
    }

    if (end_joints_pose.size() != initial_joints_pose.size()) {
      RCLCPP_WARN(get_logger(),
                  "[pingpong2] IK returned %zu joints, expected %zu",
                  end_joints_pose.size(), initial_joints_pose.size());
      res.result = false;
      return false;
    }

    if (req.right_target_pose.pose.position.x < 0.0) {
      end_joints_pose[4] = 0.0;
      end_joints_pose[5] = 1.5;
    } else {
      end_joints_pose[4] = 2.0;
      end_joints_pose[5] = 1.5;
    }

    if (!req.right_target_joints.empty()) {
      if (req.right_target_joints.size() != initial_joints_pose.size()) {
        RCLCPP_WARN(get_logger(),
                    "[pingpong2] right_target_joints must contain 6 residual joint values, got %zu",
                    req.right_target_joints.size());
        res.result = false;
        return false;
      }

      for (size_t i = 0; i < initial_joints_pose.size(); ++i) {
        end_joints_pose[i] += req.right_target_joints[i];
      }
    }

    const double duration = (req.duration > 0.0) ? req.duration : 1.0;
    const double return_duration = 1.5;

    if (!directlySetAllJoints(end_joints_pose, req.robot, duration)) {
      res.result = false;
      return false;
    }

    waiting(duration);
    if (!directlySetAllJoints(initial_joints_pose, req.robot, return_duration)) {
      res.result = false;
      return false;
    }
    waiting(return_duration);

    res.right_target_joints = end_joints_pose;
    res.result = true;
    return true;
  }