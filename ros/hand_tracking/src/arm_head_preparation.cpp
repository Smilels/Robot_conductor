#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <controller_manager_msgs/SwitchController.h>


void velocity2trajectory_controllers(ros::NodeHandle &nh, bool left_teleop, bool right_teleop) {

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

void trajectory2velocity_controllers(ros::NodeHandle &nh, bool left_teleop, bool right_teleop) {

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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_head_preparation");
    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::NodeHandle nh;

    while (ros::ok())
    {
        std::cout << "------------------Choose one of the following action------------------" << std::endl;
        std::cout  << "Press 1: Head On please" << std::endl;
        std::cout << "Press 2: Left arm pose is ready" << std::endl;
        std::cout << "Press 3: Left arm from trajectory control to velocity control" << std::endl;
        std::cout << "Press 4: Right arm from trajectory control to velocity control" << std::endl;
        std::cout << "Press 5: Left arm from velocity control to trajectory control" << std::endl;
        std::cout << "Press 6: Rgiht arm from velocity control to trajectory control" << std::endl;
        std::cout << "Press q : quit this program" << std::endl;

        char imput_command;
        std::cin >> imput_command;
        switch (imput_command){
        case '1':
        {
            moveit::planning_interface::MoveGroupInterface *head_mgi_ = new moveit::planning_interface::MoveGroupInterface("head");
            std::vector<double> pr2_head_position{0.32, 0.51};
            head_mgi_->setJointValueTarget(pr2_head_position);
            head_mgi_->move();
            std::cout << "Head On please" << std::endl;
            break;
        }
        case '2':{
            moveit::planning_interface::MoveGroupInterface *left_arm_mgi_ = new moveit::planning_interface::MoveGroupInterface("left_arm");
            std::vector<double> left_arm_start_position{0.0, 0.63, 0.2, -0.15, 0.0, -0.31, -0.0};
            left_arm_mgi_->setJointValueTarget(left_arm_start_position);
            left_arm_mgi_->move();
            std::cout << "Left arm pose is ready" << std::endl;
            break;}
        case '3':{
            velocity2trajectory_controllers(nh, "true", "false");
            std::cout << "Left arm from trajectory control to velocity control" << std::endl;
            break;}
        case '4':{
            velocity2trajectory_controllers(nh, "false", "true");
            std::cout << "Right arm from trajectory control to velocity control" << std::endl;
            break;}
        case '5':{
            trajectory2velocity_controllers(nh, "true", "false");
            std::cout << "Left arm from velocity control to trajectory control" << std::endl;
            break;}
        case '6':{
            trajectory2velocity_controllers(nh, "false", "true");
            std::cout << "Right arm from velocity control to trajectory control" << std::endl;
            break;}
        case 'q' :{
            ros::shutdown();
            break;}
        default:{
		    ROS_ERROR("Invalid Selection. Please enter '1-6'. ");
		    break;}
        }
        std::cin.clear();
    }
    return 0;
}
