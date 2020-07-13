#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     :
# Purpose       :
# Creation Date :
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]
import rospy
import tf
import yaml
import os
import rospkg


def get_transform_from_yaml(path_to_ext):
    pos = yaml.load(open(path_to_ext, "r"))
    trans = (pos["translation"]["x"], pos["translation"]["y"], pos["translation"]["z"])
    rotation = (pos["rotation"]["x"], pos["rotation"]["y"], pos["rotation"]["z"], pos["rotation"]["w"])
    return trans, rotation


if __name__ == "__main__":
    rospy.init_node("fixed_camera_broadcaster")
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    trans1, rotation1 = get_transform_from_yaml(os.path.join(rospkg.RosPack().get_path("easy_handeye"),
                                                             "config/pr2_handeye.yaml"))

    while not rospy.is_shutdown():
        br.sendTransform(trans1, rotation1, rospy.Time.now(), "realsense_left_arm_optical_frame", "l_gripper_tool_frame")
        rate.sleep()
