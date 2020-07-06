#include <ros/ros.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit_msgs/MoveGroupActionResult.h>
#include <moveit_msgs/RobotState.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pluginlib/class_loader.h>
#include <moveit/kinematics_base/kinematics_base.h>

#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <memory>
#include <sstream>

#include <getopt.h>

using namespace std;

class RobotStatePublisher
{
    string prefix;
    tf2_ros::TransformBroadcaster tf_broadcaster;
    // tf::TransformBroadcaster tf_broadcaster;
    vector<geometry_msgs::TransformStamped> msgs;
public:
    RobotStatePublisher(const string& prefix) : prefix(prefix)
    {
    }
    void publish(const robot_state::RobotState& robot_state)
    {
        if(prefix.empty()) return;
        auto robot_model = robot_state.getRobotModel();
        auto time = ros::Time::now();
        msgs.clear();
        for(auto& link_name : robot_model->getLinkModelNames())
        {
            auto f = robot_state.getFrameTransform(link_name);
            Eigen::Quaterniond q(f.rotation());
            msgs.emplace_back();
            auto& msg = msgs.back();
            msg.header.stamp = time;
            if(link_name == robot_model->getRootLinkName())
                msg.header.frame_id = robot_model->getRootLinkName();
            else
                msg.header.frame_id = prefix + "/" + robot_model->getRootLinkName();
            msg.child_frame_id = prefix + "/" + link_name;
            msg.transform.translation.x = f.translation().x();
            msg.transform.translation.y = f.translation().y();
            msg.transform.translation.z = f.translation().z();
            msg.transform.rotation.x = q.x();
            msg.transform.rotation.y = q.y();
            msg.transform.rotation.z = q.z();
            msg.transform.rotation.w = q.w();
        }
        tf_broadcaster.sendTransform(msgs);
    }
};
