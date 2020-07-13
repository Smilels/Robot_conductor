#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Scalar.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>

#include <cstdlib>
#include <signal.h>
#include <algorithm>
#include <thread>

#include <bio_ik/bio_ik.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/RobotState.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <control_msgs/FollowJointTrajectoryAction.h>


// Helper macro, throws an exception if a statement fails
#define VXSTR(s) VSTR(s)
#define VSTR(s) #s
#define V(x)                                                                   \
  if (!(x)) {                                                                  \
    ros::Duration(1.0).sleep();                                                \
    throw std::runtime_error(VXSTR(x));                                        \
  }
#define pi 3.1415926

typedef actionlib::SimpleActionClient <pr2_controllers_msgs::Pr2GripperCommandAction> GripperClient;
typedef actionlib::SimpleActionClient< control_msgs::FollowJointTrajectoryAction > TrajClient;

class HandTrack {
public:
    // RobotStatePublisher ik_pub = RobotStatePublisher("ik_test");

    HandTrack(ros::NodeHandle &nh);

    void arm_track();

    ~HandTrack() {
    };

private:
    double clip(double x, double maxv, double minv);

    tf2::Transform get_camera_transform();

    void callback(const std_msgs::Float64MultiArrayConstPtr & bone_data);
    // void callback(const std_msgs::Float64MultiArrayConstPtr & bone_data);

    void bioik_method(std::vector<double> & joint_values,
                      const robot_state::JointModelGroup* joint_model_group_, const tf2::Transform & base_wrist_tf);

    void controller_vel_method(const std::vector<double> & joint_values,
                                const std::vector<double> &joint_feedforward_diff);

    void joint_state_publisher(const std::vector<double> & goal);
    void zero_Velcity();

    ros::Subscriber subscriber_;
    std::vector<double> shared_hand_data;
    ros::Publisher joint_pub_;
    ros::Publisher trajectory_publisher_;
    ros::NodeHandle nh_;

    moveit::planning_interface::MoveGroupInterface *mgi_;
    robot_model_loader::RobotModelLoader rml_;
    robot_model::RobotModelPtr robot_model_;
    planning_scene::PlanningScene *scene_;
    robot_state::JointModelGroup *joint_model_group_;
//    moveit::core::RobotState robot_state_;
//    moveit::core::RobotState previous_state_;

    tf2::Stamped <tf2::Transform> previous_transform_;
    std::vector<double> pre_joints;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    std::string base_frame_;
    std::vector<double> inital_state;
    std::vector <std::string> joint_state_names;

    std::string wrist_link_;

    bool vel_method_;
    bool demo_test_;
    bool wrist_pos_;
    bool wrist_rot_;
    bool wrist_pose_;
    std::string move_group_name_;

    std::vector<double> joint_values;
    std::vector<double> previous_joint_values;
    double previous_rh_wrist1_eular;
    double previous_rh_wrist2_eular;
    double rh_wrist1_eular;
    double rh_wrist2_eular;

    double left_velocity_factor;
    double DELTA_MIN_THRESHOLD;
    std::vector<double> max_arm_velocity;
    int frequency;
};
