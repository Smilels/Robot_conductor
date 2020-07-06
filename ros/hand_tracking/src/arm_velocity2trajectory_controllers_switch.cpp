#include <ros/ros.h>
#include <controller_manager_msgs/SwitchController.h>
#include <moveit/move_group_interface/move_group_interface.h>


void switch_arm_vel_controllers(ros::NodeHandle &nh, bool left_teleop, bool right_teleop) {

    ros::ServiceClient switchClient = nh.serviceClient<controller_manager_msgs::SwitchController>(
            "pr2_controller_manager/switch_controller");

    controller_manager_msgs::SwitchController switchSrv;
    if (left_teleop) {
        switchSrv.request.stop_controllers.push_back("l_arm_joint_group_velocity_controller");
        switchSrv.request.start_controllers.push_back("l_arm_controller");
    }
    if (right_teleop) {
        switchSrv.request.stop_controllers.push_back("r_arm_joint_group_velocity_controller");
        switchSrv.request.start_controllers.push_back("r_arm_controller");
    }

    switchSrv.request.strictness = 2; //controller_manager_msgs::SwitchController::STRICT;
    switchClient.call(switchSrv);

}


int main(int argc, char **argv) {
    ros::init(argc, argv, "arm_trajectory2velocity_controllers_switch");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::NodeHandle pnh("~");
    bool right_arm;
    bool left_arm;
    pnh.param<bool>("right_arm", right_arm, "false");
    pnh.param<bool>("left_arm", left_arm, "true");
    switch_arm_vel_controllers(nh, left_arm, right_arm);
    ros::Rate rate(5.0);
    std::cout << "Switch velocity controller to arm controller!" << std::endl;
    return 0;
}
