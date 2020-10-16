# RobotConductor

This package includes two parts: one is tracking human hand and estimating human hand pose by pointclouds, 
the other one is estimating human hand joint

## Prerequisites
- Ubuntu 18.04
- Open3D0.9
     ```
        conda install open3d=0.10 -c open3d-admin
     ```

## Demo
1. bringup Trixi and make sure Kinect2 work well

2. run openpose on TAMS225, subscribing the /kinect2/sd/image_color_rect and /kinect2/remapped/sd/image_depth_rect
two topics from Trixi
    ```
        roslaunch ros_openpose run.launch
    ```
3. move Trixi arm and head to proper tracking position
    ```
        rosrun hand_tracking arm_head_preparation
    ```
4. run left arm tracking demo
    ```
        roslaunch hand_tracking arm_track_human_hand.launch
    ```
