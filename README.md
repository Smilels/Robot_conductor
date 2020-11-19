# RobotConductor

This package includes two parts: one is tracking human hand and estimating human hand pose by pointclouds, 
the other one is estimating human hand joint

## Prerequisites
- Ubuntu 18.04
- Open3D0.9
     ```
        conda install open3d=0.10 -c open3d-admin
     ```

## Data preprocess
In /HandPoints/preprocess folder:
- robot_pose_generation.py is used for generation the goal positions for the robot hand.
 For example, the fingertip positions of each finger, and the proximal positions of the fingers.
- human_pc_preprocess.py is used for generating, downsampling, normalizing the pointclouds
of human hand.
- robot_pc_preprocess.py is used for generating, downsampling, normalizing the pointclouds
of robot hand.

## Training 
- train.py is used for bringup the training process
- config/config.yaml provide the setting of gpus, which model to use, and further trainning
optimizer parameters
- model/pointnet2_ssg_handjoints.py sets up the model layers.
    - we define few PointnetSAModules, and each of them has 5 parameters: npoints, radius, nsample, mlp, use_xyz
    -  after few PointnetSAModule, follows some fc_layer
- pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py stores how the models 
initialize and forward.
    - sampling layer: uses iterative RPS to choose a subset of points (npoint)
    - grouping layer: uses these points as the centroieds to find their group in the whole input pointclouds.
     In each group, there are nsample points.
    - PointNet layer: based on the mlps, it outputs #npoint points with new features 
- pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py stores the tool functions, such as
 QueryAndGroup, furthest_point_sample, BallQuery

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
