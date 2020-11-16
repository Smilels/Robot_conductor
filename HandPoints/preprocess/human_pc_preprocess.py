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
from utils import depth2pc, pca_rotation, down_sample, get_normal, normalization_unit, FPS, FPS_idx


save_points = True
show_bbx = 0

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079

DOWN_SAMPLE_NUM = 2048
FPS_SAMPLE_NUM = 512

base_path = "./data/"
img_path = base_path + "images/"
points_path = base_path + "points_human/"
mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])


def get_human_points(line):
    # 1 read the groundtruth and the image
    frame = line.split(' ')[0].replace("\t", "")
    print(frame)

    # image path depends on the location of your training dataset
    try:
        img = cv2.imread(img_path + str(frame), cv2.IMREAD_ANYDEPTH)
    except:
        print("no Image", frame)
        return

    label_source = line.split('\t')[1:]
    label = [float(l.replace(" ", "")) for l in label_source[0:63]]
    keypoints = np.array(label).reshape(21, 3)

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
    points_pca = pca_rotation(points)

    # 4 downsampling
    points_pca_sampled, rand_ind = down_sample(points_pca, DOWN_SAMPLE_NUM)

    # 7 FPS Sampling
    points_pca_fps_sampled, farthest_pts_idx = FPS_idx(points_pca_sampled, FPS_SAMPLE_NUM)

    # 8 compute surface normal
    normals_pca = get_normal(points_pca)
    normals_pca_sampled = normals_pca[rand_ind]
    normals_pca_fps_sampled = normals_pca_sampled[farthest_pts_idx]

    # 9 normalize point cloud
    points_normalized, max_bb3d_len, offset = normalization_unit(points_pca_fps_sampled)

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

        o3d.visualization.draw_geometries([world_frame_vis, pcd_normalized], point_show_normal=False)

    if save_points:
        if not os.path.exists(points_path):
            os.makedirs(points_path)
        data = np.array([points_normalized, normals_pca_fps_sampled, max_bb3d_len, offset],
                        dtype=object)
        np.save(os.path.join(points_path, frame[:-4] + '_points.npy'), data)


def main():
    datafile = open(base_path + "groundtruth/Training_Annotation.txt", "r")
    lines = datafile.read().splitlines()
    lines.sort()
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    pool.map(get_human_points, lines)
    # for line in lines:
    #     get_human_points(line)
    datafile.close()


if __name__ == '__main__':
    main()
