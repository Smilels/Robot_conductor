#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pr2head_move");
    ros::AsyncSpinner spinner(2);
    spinner.start();
    
    moveit::planning_interface::MoveGroupInterface *head_mgi_ = new moveit::planning_interface::MoveGroupInterface("head");
    std::vector<double> pr2_head_position{0.32, 0.51};
    head_mgi_->setJointValueTarget(pr2_head_position);
    head_mgi_->move();
    std::cout << "Head On please" << std::endl;
    

    moveit::planning_interface::MoveGroupInterface *left_arm_mgi_ = new moveit::planning_interface::MoveGroupInterface("left_arm");
    std::vector<double> left_arm_start_position{0.0, 0.63, 0.2, -0.15, 0.0, -0.31, -0.0};
    left_arm_mgi_->setJointValueTarget(left_arm_start_position);
    left_arm_mgi_->move();
    std::cout << "Left arm is ready" << std::endl;

    return 0;
}
