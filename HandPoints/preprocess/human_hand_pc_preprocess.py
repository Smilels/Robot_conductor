#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: bighand depth to normalized pointclouds
# Date       : 15/09/2020: 17:41
# File Name  : hand_preprocess.py


import numpy as np
import os
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from IPython import embed
import multiprocessing as mp
from utils import depth2pc, pca_rotation, down_sample, get_normal, normalization_unit, farthest_point_sampling_fast, FPS


save_points = True
show_bbx = False
save_local_frame = False

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079

SAMPLE_NUM = 1024
SAMPLE_NUM_level1 = 512
SAMPLE_NUM_level2 = 128

base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
img_path = base_path + "Human_label/human_full_test/"
points_path = base_path + "points_human/"
local_frame_path = base_path + "local_frame/"
mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])


def get_human_points(line):
    # 1 read groundtruth and image
    frame = line.split(' ')[0].replace("\t", "")
    # print(frame)

    # image path depends on the location of your training dataset
    try:
        img = cv2.imread(img_path + str(frame), cv2.IMREAD_ANYDEPTH)
    except:
        print("no Image", frame)
        return

    label_source = line.split('\t')[1:]
    label = []
    label.append([float(l.replace(" ", "")) for l in label_source[0:63]])
    keypoints = np.array(label).reshape(21, 3)

    # 2 get hand points
    padding = 80
    points = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)

    x_min_max = [np.min(keypoints[:, 0] - padding / 2), np.max(keypoints[:, 0]) + padding / 2]
    y_min_max = [np.min(keypoints[:, 1] - padding / 2), np.max(keypoints[:, 1]) + padding / 2]
    z_min_max = [np.min(keypoints[:, 2] - padding / 2), np.max(keypoints[:, 2]) + padding / 2]
    hand_points = points[np.where((points[:, 0] > x_min_max[0]) & (points[:, 0] < x_min_max[1]) &
                                  (points[:, 1] > y_min_max[0]) & (points[:, 1] < y_min_max[1]) &
                                  (points[:, 2] > z_min_max[0]) & (points[:, 2] < z_min_max[1]))]
    if len(hand_points) < 300:
        print("hand points is %d, isless than 300, maybe it's a broken image" % len(hand_points))
        return

    # 3 PCA rotation
    hand_points_pca = pca_rotation(hand_points)

    # 4 downsampling
    hand_points_pca_sampled, rand_ind = down_sample(hand_points_pca, SAMPLE_NUM)

    # 5 compute surface normal
    normals_pca = get_normal(hand_points_pca)
    normals_pca_sampled = normals_pca[rand_ind]
    embed()
    # 6 normalize point cloud
    # hand_points_normalized_sampled = normalization(hand_points_pca, hand_points_pca_sampled, SAMPLE_NUM)

    # 7 FPS Sampling
    # pfs_points = FPS(hand_points_normalized_sampled, SAMPLE_NUM_level1)

    pc = np.concatenate([hand_points_pca_sampled, normals_pca_sampled], axis=1)

    if show_bbx:
        pcd = o3d.geometry.PointCloud()
        pcd_hand = o3d.geometry.PointCloud()
        pcd_pca = o3d.geometry.PointCloud()
        pcd_pca_sample = o3d.geometry.PointCloud()
        pcd_key = o3d.geometry.PointCloud()
        pcd_normalized = o3d.geometry.PointCloud()
        # pcd_fps_sample = o3d.geometry.PointCloud()

        # the original points which contains the whole scenario
        pcd.points = o3d.utility.Vector3dVector(points)
        # the original hand points extracted from the whole image
        pcd_hand.points = o3d.utility.Vector3dVector(hand_points)
        pcd_hand.paint_uniform_color([0.9, 0.1, 0.1])  # red
        # keypoints label
        pcd_key.points = o3d.utility.Vector3dVector(keypoints)
        pcd_key.paint_uniform_color([0.1, 0.1, 0.7])
        # points after pca transformation
        pcd_pca.points = o3d.utility.Vector3dVector(hand_points_pca)
        pcd_pca.paint_uniform_color([0.1, 0.9, 0.5])  # blue
        # random downsampling of pca points
        pcd_pca_sample.points = o3d.utility.Vector3dVector(hand_points_pca_sampled)
        pcd_pca_sample.paint_uniform_color([0.1, 0.1, 0.7])

        # pcd_normalized.points = o3d.utility.Vector3dVector(hand_points_normalized_sampled)
        # pcd_fps_sample.points = o3d.utility.Vector3dVector(pfs_points + np.array([200, 0, 0]))
        # pcd_fps_sample.paint_uniform_color([0.1, 0.9, 0.5])  # blue

        o3d.visualization.draw_geometries([pcd_fps_sample, pcd_fps_sample], point_show_normal=False)
    if save_points:
        if not os.path.exists(points_path):
            os.makedirs(points_path)
        np.save(os.path.join(points_path, frame[:-4] + '_points.npy'), pc)
    if save_local_frame:
        # local wrist frame build
        tf_palm = keypoints[1] - keypoints[0]
        ff_palm = keypoints[2] - keypoints[0]
        mf_palm = keypoints[3] - keypoints[0]
        rf_palm = keypoints[4] - keypoints[0]
        lf_palm = keypoints[5] - keypoints[0]
        # palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])
        palm = np.array([ff_palm, mf_palm, rf_palm, lf_palm])

        wrist_z = np.mean(palm, axis=0)
        wrist_z /= np.linalg.norm(wrist_z)
        wrist_y = np.cross(ff_palm, rf_palm)
        wrist_y /= np.linalg.norm(wrist_y)
        wrist_x = np.cross(wrist_y, wrist_z)
        if np.linalg.norm(wrist_x) != 0:
            wrist_x /= np.linalg.norm(wrist_x)

        # local coordinate matrix
        hand_frame = np.vstack([wrist_x, wrist_y, wrist_z])
        if not os.path.exists(local_frame_path):
            os.makedirs(local_frame_path)
        np.save(os.path.join(local_frame_path, frame[:-4] + '_localframe.npy'), hand_frame)


def main():
    DataFile = open(base_path + "Human_label/text_annotation.txt", "r")
    lines = DataFile.read().splitlines()
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    # pool.map(get_human_points, lines)
    for line in lines:
        get_human_points(line)
    DataFile.close()


if __name__ == '__main__':
    main()
