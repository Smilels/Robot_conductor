#include <../include/arm_track_human_hand.h>

ros::Publisher arm_vel_pub_;
// std::map<std::string, std::vector<double>>* shared_imu_data;

HandTrack::HandTrack(ros::NodeHandle &nh) : nh_(nh) {
    ros::NodeHandle private_node("~");
    // test mode or real robot
    private_node.param<bool>("vel_method", vel_method_, "false");
    private_node.param<bool>("demo_test", demo_test_, "false");

    // which goals to consider
    private_node.param<bool>("wrist_pos", wrist_pos_, "false");
    private_node.param<bool>("wrist_rot", wrist_rot_, "false");
    private_node.param<bool>("wrist_pose", wrist_pose_, "false");
    private_node.param<std::string>("move_group_name", move_group_name_, "left_arm");

    // delta min threshold
    private_node.getParam("delta_min_threshold", DELTA_MIN_THRESHOLD);

    // Maximum joint-space velocity, just for safety. You should mainly rely on
    V(private_node.getParam("left_velocity_factor", left_velocity_factor));
    // Control frequency
    V(private_node.getParam("frequency", frequency));

    robot_model_ = rml_.getModel();
    scene_ = new planning_scene::PlanningScene(robot_model_);
    base_frame_ = robot_model_->getModelFrame();  // base_footprint
    ROS_INFO("Reference frame: %s", base_frame_.c_str());
//    robot_state_ = new moveit::core::RobotState(robot_model_);

    // std::string end_effector_link_ = mgi_->getEndEffectorLink();
    // ROS_INFO("End-effector link: %s", end_effector_link_.c_str());
    mgi_ = new moveit::planning_interface::MoveGroupInterface(move_group_name_);
    joint_model_group_ = robot_model_->getJointModelGroup(move_group_name_);
    previous_joint_values = mgi_->getCurrentJointValues();

    base_camera_tf = get_camera_transform();

    if (demo_test_) {
        inital_state = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.16825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.13565, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.349938515025, 0.0, 0.0, 0.0, 0.349938515025, 0.0, 0.0, 0.0,
                        0.349938515025, 0.0, 0.0, 0.0, 0.0, 0.349938515025,
                        0.0, 0.0, 0.0, 0.0, 0.349938515025, 0.0, 0.0, 0.0, 0.0,
                        -1.13565, -1.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        joint_state_names = {"fl_caster_rotation_joint", "fl_caster_l_wheel_joint", "fl_caster_r_wheel_joint",
                             "fr_caster_rotation_joint",
                             "fr_caster_l_wheel_joint", "fr_caster_r_wheel_joint", "bl_caster_rotation_joint",
                             "bl_caster_l_wheel_joint",
                             "bl_caster_r_wheel_joint", "br_caster_rotation_joint", "br_caster_l_wheel_joint",
                             "br_caster_r_wheel_joint",
                             "torso_lift_joint", "torso_lift_motor_screw_joint", "head_pan_joint", "head_tilt_joint",
                             "laser_tilt_mount_joint", "r_shoulder_pan_joint", "r_shoulder_lift_joint",
                             "r_upper_arm_roll_joint",
                             "r_forearm_roll_joint", "r_elbow_flex_joint", "rh_WRJ2", "rh_WRJ1", "rh_FFJ4", "rh_FFJ3",
                             "rh_FFJ2",
                             "rh_FFJ1", "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1", "rh_RFJ4", "rh_RFJ3", "rh_RFJ2",
                             "rh_RFJ1",
                             "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1", "rh_THJ5", "rh_THJ4", "rh_THJ3",
                             "rh_THJ2",
                             "rh_THJ1", "l_shoulder_pan_joint", "l_shoulder_lift_joint", "l_upper_arm_roll_joint",
                             "l_forearm_roll_joint",
                             "l_elbow_flex_joint", "l_wrist_flex_joint", "l_wrist_roll_joint",
                             "l_HandTrack_motor_slider_joint",
                             "l_HandTrack_motor_screw_joint", "l_HandTrack_l_finger_joint", "l_HandTrack_r_finger_joint",
                             "l_HandTrack_l_finger_tip_joint", "l_HandTrack_r_finger_tip_joint", "l_HandTrack_joint"};
        joint_pub_ = nh_.advertise<sensor_msgs::JointState>("joint_states", 1000);
        // pre_joints.assign(7,0);
    }

    if (vel_method_) {
        ROS_WARN("Have you switched to velocity controllers");
        previous_transform_.setData(tf2::Transform(tf2::Quaternion(0, 0, 0, 1), tf2::Vector3(0, 0, 0)));
        arm_vel_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/l_arm_joint_group_velocity_controller/command",
                                                                    1);
    }

    shared_hand_data.resize(3);
    left_hand_data.resize(3);
    std::string topic_name = "rosOpenpose/right_hand_point";
    subscriber_ = nh_.subscribe<std_msgs::Float64MultiArray>(topic_name, 1, &HandTrack::callback, this);

    // note: sleep 1s to wait data!
    ros::Duration(1).sleep();

    std::vector <std::thread> threads;
    threads.push_back(std::thread(&HandTrack::arm_track, this));
    threads.push_back(std::thread(&HandTrack::stopFlag, this));

    for (auto &thread : threads)
        thread.join();
}

void HandTrack::arm_track() {
    tf2::Transform base_hand_tf, base_robot_tf, camera_hand_tf, camera_robot_tf;
    geometry_msgs::TransformStamped transformStamped;
    ros::Rate rate(frequency);

    previous_shared_hand_data = shared_hand_data;
    while (ros::ok()) {
        // std::cout<< "I am in arm tracking" <<std::endl;
        ros::Time begin = ros::Time::now();
        // if the depth is too far away (such as more than 2 meter), then this depth data is wrong
        if (shared_hand_data[2] > 2)
            continue;

        // get current hand data if the depth value difference is within 20cm
        // need to be evulated by robot experiments
        std::vector<double> human_hand_pos;
        //if (shared_hand_data[2] - previous_shared_hand_data[2] > 0.2)
        //    human_hand_pos = previous_shared_hand_data;
        //else
            human_hand_pos = shared_hand_data;
        // std::cout << "human hand pos :" << human_hand_pos[0] << " " << human_hand_pos[1]<< " " << human_hand_pos[2] << std::endl;
        // broadcast hand frame
        camera_hand_tf.setOrigin(tf2::Vector3(human_hand_pos[0], human_hand_pos[1],
                                               human_hand_pos[2]));
        // todo: consider orientation of the human hand in future
        camera_hand_tf.setRotation(tf2::Quaternion(0, 0, 0, 1));

        // std::cout<< "I am in arm tracking" <<std::endl;
        // tf2::Stamped<tf2::Transform> base_camera_tf = get_camera_transform();
        base_hand_tf = base_camera_tf * camera_hand_tf;

        tf2::Stamped <tf2::Transform> base_hand_tfStamped(base_hand_tf, ros::Time::now(), base_frame_);
        transformStamped = tf2::toMsg(base_hand_tfStamped);
        transformStamped.header.frame_id = base_frame_;
        transformStamped.child_frame_id = "human_hand_frame";
        tf_broadcaster_.sendTransform(transformStamped);

        base_robot_tf = base_hand_tf;
        base_robot_tf.setOrigin(tf2::Vector3(base_hand_tf.getOrigin().getX(),
                                base_hand_tf.getOrigin().getY(),
                                base_hand_tf.getOrigin().getZ() - 0.3));

        // broadcast robot reach frame
        tf2::Stamped <tf2::Transform> base_robot_tfStamped(base_robot_tf, ros::Time::now(), base_frame_);
        transformStamped = tf2::toMsg(base_robot_tfStamped);
        transformStamped.header.frame_id = base_frame_;
        transformStamped.child_frame_id = "robot_reach_frame";
        tf_broadcaster_.sendTransform(transformStamped);

        std::vector<double> joint_values;
        bioik_method(joint_values, joint_model_group_, base_robot_tf);

        // set previous shared hand data as the human hand pose at this time
        previous_shared_hand_data = human_hand_pos;

        rate.sleep();
        ros::Time end = ros::Time::now();
        //std::cout << "time Duration " << end.toSec() - begin.toSec() << std::endl;
    }
}

void HandTrack::bioik_method(std::vector<double> &joint_values,
                                const robot_state::JointModelGroup *joint_model_group_,
                                const tf2::Transform &base_robot_tf) {
    bio_ik::BioIKKinematicsQueryOptions ik_options;
    ik_options.replace = true;
    ik_options.return_approximate_solution = true;

    robot_state::RobotState robot_state_(robot_model_);
    wrist_link_ = "l_gripper_tool_frame";
    robot_state_.setJointGroupPositions(joint_model_group_, previous_joint_values);

    if (wrist_pos_){
        ik_options.goals.emplace_back(new bio_ik::PositionGoal(wrist_link_, base_robot_tf.getOrigin(), 1));
        //ik_options.fixed_joints.push_back("l_wrist_flex_joint");
        //ik_options.fixed_joints.push_back("l_wrist_roll_joint");
    }
    if (wrist_rot_)
        ik_options.goals.emplace_back(new bio_ik::OrientationGoal(wrist_link_, base_robot_tf.getRotation(), 1));
    if (wrist_pose_)
        ik_options.goals.emplace_back(
                new bio_ik::PoseGoal(wrist_link_, base_robot_tf.getOrigin(), base_robot_tf.getRotation(), 1));
    // regularizationGoal tries to keep the joint-space IK solution as close as possible to the given robot seed configuration
    ik_options.goals.emplace_back(new bio_ik::RegularizationGoal(0.5));
//    ik_options.goals.emplace_back(new bio_ik::MinimalDisplacementGoal(1));

    bool found_ik = robot_state_.setFromIK(
            joint_model_group_,
            EigenSTL::vector_Isometry3d(),
            std::vector<std::string>(),
            0.5/frequency,
            moveit::core::GroupStateValidityCallbackFn(
                    [&](robot_state::RobotState *state,
                        const robot_state::JointModelGroup *group,
                        const double *ik_solution) {
                        state->setJointGroupPositions(group, ik_solution);
                        state->update();
                        bool collision = (bool) (scene_->isStateColliding(
                                *state, group->getName(), true));
                        if (collision) {
                            ROS_ERROR_STREAM("collision, solution rejected");
                        }
                        return !collision;
                    }),
            ik_options);

    if (found_ik) {
        robot_state_.copyJointGroupPositions(joint_model_group_, joint_values);
        // keeps boundaries -M_PI < publishPosition <= M_PI
        for (int j = 0; j <joint_values.size(); j++) {
            if (M_PI < joint_values[j]){
                joint_values[j] = joint_values[j] - 2 * M_PI;
                ROS_WARN("keeps boundaries for publishPosition <= M_PI: %f",joint_values[j]);
            }
            if (-M_PI >= joint_values[j]){
                joint_values[j] = joint_values[j] + 2 * M_PI;
                ROS_WARN("keeps boundaries for publishPosition > -M_PI: %f",joint_values[j]);
            }
        }

        std::vector<double> joint_feedforward_diff;
        std::transform(joint_values.begin(), joint_values.end(), previous_joint_values.begin(), std::back_inserter(joint_feedforward_diff),
                     [&](double l, double r) {
                         return (l - r);
                     });
        double max_delta = *max_element(joint_feedforward_diff.begin(), joint_feedforward_diff.end());

        //if (max_delta > DELTA_MIN_THRESHOLD){
        //    ROS_ERROR( "IK solution too far from current joint state, canceled. delta= %8.4lf", max_delta);
        //    zero_Velcity();
       // }
       // else
       {
            if (demo_test_)
                joint_state_publisher(joint_values);
            if (vel_method_)
                controller_vel_method(joint_values, joint_feedforward_diff);
        }
      // robot_state_.copyJointGroupPositions(joint_model_group_, joint_values);
      // robot_state_.update(); // if i use some catesian pose, then maybe need to consider update the robot state
    }
    else{
        ROS_WARN("NO IK SOLUTION");
        zero_Velcity();
    }
}

void HandTrack::controller_vel_method(const std::vector<double> &joint_values, const std::vector<double> &joint_feedforward_diff) {
    if (joint_values.size() > 0) {
      std::vector<double> start = mgi_->getCurrentJointValues();
      std_msgs::Float64MultiArray arm_joints_vel;
      arm_joints_vel.data.assign(7, 0);

      // joint_feedback_diff measures the different of human motion and current robot state
      std::vector<double> joint_feedback_diff;
      std::transform(joint_values.begin(), joint_values.end(), start.begin(), std::back_inserter(joint_feedback_diff),
                     [&](double l, double r) {
                         return (l - r);
                     });

      std::vector<double> joint_diff(joint_values.size(), 0);
      for (int j = 0; j < joint_values.size(); j++) {
          double diff = joint_feedforward_diff[j] * frequency * 0.8 + joint_feedback_diff[j] * left_velocity_factor;
          if (M_PI < diff)
              diff = diff - 2 * M_PI;
          if (-M_PI > diff)
              diff = diff + 2 * M_PI;
          joint_diff[j] = diff;
          // ROS_INFO_STREAM("joint" << j << " " << "diff is "<<diff);
      }

      // shoulder_pan_joint
      arm_joints_vel.data[0] = clip(joint_diff[0], 2.1, -2.1);
      // shoulder_lift_joint
      arm_joints_vel.data[1] = clip(joint_diff[1], 2.1, -2.1);
      // upper_arm_roll_joint
      // arm_joints_vel.data[2] = clip(joint_diff[2], 3.27, -3.27);
      // elbow_flex_joint
      arm_joints_vel.data[3] = clip(joint_diff[3], 3.3, -3.3);
      // forearm_roll_joint
      // arm_joints_vel.data[4] = clip( joint_diff[4], 3.6, -3.6);
      // wrist_flex_joint
      arm_joints_vel.data[5] = clip(joint_diff[5], 3.1, -3.1);
      // wrist_roll_joint
      // arm_joints_vel.data[6] = clip(joint_diff[6], 3.6, -3.6);
      //arm_vel_pub_.publish(arm_joints_vel);
      previous_joint_values = joint_values;
    }
}

void HandTrack::joint_state_publisher( const std::vector<double> &goal) {
    sensor_msgs::JointState joint_states;
    joint_states.header.stamp = ros::Time::now();

    joint_states.name = joint_state_names;
    joint_states.position = inital_state;

    if (goal.size() > 0) {
        joint_states.position[46] = goal[0];
        joint_states.position[47] = goal[1];
        joint_states.position[48] = goal[2];
        joint_states.position[49] = goal[4];
        joint_states.position[50] = goal[3];
        joint_states.position[51] = goal[5];
        joint_states.position[52] = goal[6];
        previous_joint_values = goal;
    }

    joint_pub_.publish(joint_states);
}

void HandTrack::callback(const std_msgs::Float64MultiArrayConstPtr &hand_data) {
    for (int j = 0; j < 3; j++) {
        if (std::isnan(hand_data->data[j]))
            break;
        shared_hand_data.at(j) = hand_data->data[j];
    }
    for (int j = 3; j < 6; j++) {
        if (std::isnan(hand_data->data[j]))
            break;
        left_hand_data.at(j-3) = hand_data->data[j];
    }
}

double HandTrack::clip(double x, double maxv = 0, double minv = 0) {
    if (x > maxv)
        x = maxv;
    if (x < minv)
        x = minv;
    return x;
}

tf2::Stamped <tf2::Transform> HandTrack::get_camera_transform() {
    tf2::Stamped <tf2::Transform> base_camera_tf;

    tf2_ros::TransformListener tfListener(tfBuffer);
    geometry_msgs::TransformStamped tfGeom;
    try {
      tfGeom = tfBuffer.lookupTransform(base_frame_, "head_mount_kinect2_ir_optical_frame",
                                        ros::Time(0), ros::Duration(5.0));
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s", ex.what());
      ros::Duration(1.0).sleep();
    }
    tf2::convert(tfGeom, base_camera_tf);
    return base_camera_tf;
}

void HandTrack::zero_Velcity() {
    std_msgs::Float64MultiArray reset_arm_joints_vel;

    reset_arm_joints_vel.data.assign(7, 0);
    arm_vel_pub_.publish(reset_arm_joints_vel);
}

void HandTrack::stopFlag(){
  while(ros::ok())
  {
    try{
      if(left_hand_data[1]<-0.1 )
        {
          std::cout << "Receive stop command, stop now!" << std::endl;
          zero_Velcity();
          ros::shutdown();
        }
   }
  catch(const std::exception& e)
    {
      ROS_ERROR("No left hand data");
    }
  }
 }

void STOP_VEL_CONTROLLER(int sig) {
    std::cout << "singal handler (SIGINT/SIGKILL) started" << std::endl;

    std_msgs::Float64MultiArray reset_arm_joints_vel;

    reset_arm_joints_vel.data.assign(7, 0);
    arm_vel_pub_.publish(reset_arm_joints_vel);    // delete scene_;
    // delete r_joint_model_group_;
    // delete l_joint_model_group_;
    // delete shared_imu_data;
    ros::shutdown();
}

int main(int argc, char **argv) {
    signal(SIGINT, STOP_VEL_CONTROLLER);
    ros::init(argc, argv, "arm_track_human_hand_node", 1); // 1=no NoSigintHandler
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(4);
    spinner.start();

    HandTrack neuron_teleop(nh);

    return 0;
}
