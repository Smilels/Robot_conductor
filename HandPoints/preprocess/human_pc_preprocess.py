#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: bighand depth to normalized pointclouds
# Date       : 15/09/2020: 17:41
# File Name  : human_pc_preprocess.py


import numpy as np
import os, sys
import cv2
import open3d as o3d
from IPython import embed
import multiprocessing as mp
from utils import depth2pc, pca_rotation, down_sample, get_normal, normalization_unit, FPS_idx, normalization_mean
from scipy.spatial.transform import Rotation as R


save_norm = 0
show_bbx = 0
do_pca = 0

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079

DOWN_SAMPLE_NUM = 2048
FPS_SAMPLE_NUM = 1024
 

if sys. argv[1] == "tams108":
    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    img_path = os.path.join(base_path, "Human_label/human_full_test/")
    tf_path = os.path.join(base_path, "human_pca_tf/")
    pose_path = os.path.join(base_path, "human_pose/")
    points_path = os.path.join(base_path, "human_points/")
    gt_file = "Human_label/text_annotation.txt"

elif sys. argv[1] == "server":
    base_path = "/data/sli/Bighand2017/"
    img_path = base_path + "images/"
    tf_path = os.path.join(base_path, "points_no_pca/human_pca_tf/")
    pose_path = os.path.join(base_path, "human_pose/")
    points_path = base_path + "points_no_pca/human_points/"
    show_bbx = 0
    gt_file = "groundtruth/Training_Annotation.txt"

mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])


def get_human_points(line):
    # read the groundtruth and the image
    frame = line.split(' ')[0].replace("\t", "")

    # image path depends on the location of your training dataset
    try:
        img = cv2.imread(img_path + str(frame), cv2.IMREAD_ANYDEPTH)
        # n = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow(frame, n)
        # key = cv2.waitKey()
    except:
        print("no Image", frame)
        return

    label_source = line.split('\t')[1:]
    label = [float(l.replace(" ", "")) for l in label_source[0:63]]
    keypoints = np.array(label).reshape(21, 3)

    # 1 get local wrist frame
    tf_palm = keypoints[1] - keypoints[0]
    ff_palm = keypoints[2] - keypoints[0]
    mf_palm = keypoints[3] - keypoints[0]
    rf_palm = keypoints[4] - keypoints[0]
    lf_palm = keypoints[5] - keypoints[0]
    palm = np.array([ff_palm, mf_palm, rf_palm, lf_palm])

    wrist_z = np.mean(palm, axis=0)
    wrist_z /= np.linalg.norm(wrist_z)
    wrist_y = np.cross(ff_palm, rf_palm)
    wrist_y /= np.linalg.norm(wrist_y)
    wrist_x = np.cross(wrist_y, wrist_z)
    if np.linalg.norm(wrist_x) != 0:
        wrist_x /= np.linalg.norm(wrist_x)

    hand_frame = np.vstack([wrist_x, wrist_y, wrist_z, ])
    # r = R.from_matrix(hand_frame)
    # axisangle = r.as_rotvec()
    hand_pose = np.hstack([hand_frame, keypoints[0].reshape(3, 1)])

    # 2 get hand points
    padding = 80
    points_raw = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)
    x_min_max = [np.min(keypoints[:, 0] - padding / 2), np.max(keypoints[:, 0]) + padding / 2]
    y_min_max = [np.min(keypoints[:, 1] - padding / 2), np.max(keypoints[:, 1]) + padding / 2]
    z_min_max = [np.min(keypoints[:, 2] - padding / 2), np.max(keypoints[:, 2]) + padding / 2]
    points = points_raw[np.where((points_raw[:, 0] > x_min_max[0]) & (points_raw[:, 0] < x_min_max[1]) &
                                 (points_raw[:, 1] > y_min_max[0]) & (points_raw[:, 1] < y_min_max[1]) &
                                 (points_raw[:, 2] > z_min_max[0]) & (points_raw[:, 2] < z_min_max[1]))]
    if len(points) < 300:
        print("%s hand points is %d, which is less than 300. Maybe it's a broken image" % (frame, len(points)))
        return

    # 3 PCA rotation
    if do_pca:
        points_pca, pc_transfrom = pca_rotation(points)
    else:
        points_pca = points

    # 4 downsampling
    points_pca_sampled, rand_ind = down_sample(points_pca, DOWN_SAMPLE_NUM)

    # 7 FPS Sampling
    points_pca_fps_sampled, farthest_pts_idx = FPS_idx(points_pca_sampled, FPS_SAMPLE_NUM)

    # 8 compute surface normal
    normals_pca = get_normal(points_pca)
    normals_pca_sampled = normals_pca[rand_ind]
    normals_pca_fps_sampled = normals_pca_sampled[farthest_pts_idx]

    # 9 normalize point cloud
    # points_normalized, max_bb3d_len, offset = normalization_unit(points_pca_fps_sampled)
    points_normalized, points_mean = normalization_mean(points_pca_fps_sampled)

    if show_bbx:
        pcd = o3d.geometry.PointCloud()
        pcd_hand = o3d.geometry.PointCloud()
        pcd_pca = o3d.geometry.PointCloud()
        pcd_pca_sample = o3d.geometry.PointCloud()
        pcd_key = o3d.geometry.PointCloud()
        pcd_normalized = o3d.geometry.PointCloud()
        pcd_fps_sample = o3d.geometry.PointCloud()

        # the original points which contains the whole scenario
        pcd.points = o3d.utility.Vector3dVector(points_raw)
        # the original hand points extracted from the whole image
        pcd_hand.points = o3d.utility.Vector3dVector(points)
        pcd_hand.paint_uniform_color([0.9, 0.1, 0.1])  # red
        # keypoints label
        pcd_key.points = o3d.utility.Vector3dVector(keypoints)
        pcd_key.paint_uniform_color([0.1, 0.1, 0.7])
        # points after pca transformation
        pcd_pca.points = o3d.utility.Vector3dVector(points_pca)
        pcd_pca.paint_uniform_color([0.1, 0.9, 0.5])  # green
        # random downsampling of pca points
        pcd_pca_sample.points = o3d.utility.Vector3dVector(points_pca_sampled)
        pcd_pca_sample.paint_uniform_color([0.1, 0.1, 0.7])  # blue
        # fps sampled points
        pcd_fps_sample.points = o3d.utility.Vector3dVector(points_pca_fps_sampled + np.array([200, 0, 0]))
        pcd_fps_sample.paint_uniform_color([0.1, 0.1, 0.7])  # green

        # the values of the normalized points are in [-0.5,0.5],
        # so they cannot visualize together with other non-normalized points)
        pcd_normalized.points = o3d.utility.Vector3dVector(points_normalized)

        # the world frame
        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=150, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([pcd_normalized], point_show_normal=False)

    if not os.path.exists(points_path):
        os.makedirs(points_path)
    if save_norm:
        # data = np.array([points_normalized, axisangle, max_bb3d_len, offset, normals_pca_fps_sampled],
        #                 dtype=object)
        data = np.array([points_normalized, points_mean, normals_pca_fps_sampled],
                        dtype=object)
        np.save(os.path.join(points_path, frame[:-4] + '.npy'), data)
    else:
        data = np.array([points_normalized],
                        dtype=object)
        # data = np.array([points_normalized, max_bb3d_len, offset],
        #                 dtype=object)
        np.save(os.path.join(points_path, frame[:-4] + '.npy'), data)

    if do_pca:
        if not os.path.exists(tf_path):
            os.makedirs(tf_path)
        np.save(os.path.join(tf_path, frame[:-4] + '.npy'), pc_transfrom)

    #if not os.path.exists(pose_path):
    #    os.makedirs(pose_path)
    #np.save(os.path.join(pose_path, frame[:-4] + '.npy'), hand_pose)


def main():
    datafile = open(base_path + gt_file, "r")
    lines = datafile.read().splitlines()
    lines.sort()

    if sys.argv[1] == "tams108":
        for line in lines:
            get_human_points(line)
    elif sys.argv[1] == "server":
        cores = mp.cpu_count()
        pool = mp.Pool(processes=cores)
        pool.map(get_human_points, lines[:400000])

    datafile.close()


if __name__ == '__main__':
    main()
