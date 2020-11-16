#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: bighand depth to normalized pointclouds
# Date       : 15/09/2020: 17:41
# File Name  : hand_preprocess.py


import os
import cv2
import open3d as o3d
import math
from utils import depth2pc, pca_rotation, down_sample, get_normal, normalization_unit, FPS, FPS_idx
import glob
import numpy as np
import multiprocessing as mp
from IPython import embed


save_points = 1
vis_points = 0
DOWN_SAMPLE_NUM = 2048
FPS_SAMPLE_NUM = 512

image_width = 640
image_height = 480
near = 0.1  * 1000
max_depth = 5.0 * 1000
fov_y = 1.02974
e = 1.0 / math.tan(fov_y / 2)
# focal_length can be calculated as f = max(u_res, v_res) / tan(FoV/2),
# where FoV is the perspective angle (from the vision sensor properties)
focalLengthX = e * image_width
focalLengthY = e * image_width

centerX = image_width/2
centerY = image_height/2

root_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
image_path = os.path.join(root_path, "depth_shadow")
points_path = os.path.join(root_path, "points_shadow")


def show_points(hand_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hand_points)
    o3d.visualization.draw_geometries([pcd])


def get_shadow_points(item):
    print(item[-19::])
    img = cv2.imread(item, cv2.IMREAD_ANYDEPTH)
    if vis_points:
        points = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)
        print("ddd")
        show_points(points)

    if save_points:
        # 1 get hand points
        img[img == 1000] = 0
        points = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)
        # print("size of hand point is: ", len(hand_points))

        if len(points) < 300:
            print("hand points is %d, isless than 300, maybe it's a broken image" % len(points))
            return
        # 2 PCA rotation
        points_pca = pca_rotation(points)

        # 3 downsampling
        points_pca_sampled, rand_ind = down_sample(points_pca, DOWN_SAMPLE_NUM)
        print("size of hand points after downsampling is: ", len(points_pca_sampled))

        # 4 FPS
        points_pca_fps_sampled, farthest_pts_idx = FPS_idx(points_pca_sampled, FPS_SAMPLE_NUM)
        # show_points(hand_points_pca_sampled)

        # 5 compute surface normal
        normals_pca = get_normal(points_pca)
        normals_pca_sampled = normals_pca[rand_ind]
        normals_pca_fps_sampled = normals_pca_sampled[farthest_pts_idx]

        # 6 normalize point cloud
        points_normalized, max_bb3d_len, offset = normalization_unit(points_pca_fps_sampled)
        # show_points(hand_points_normalized_sampled)

        embed()
        if not os.path.exists(points_path):
            os.makedirs(points_path)
        # np.save(os.path.join(points_path, item[-19:-4] + '.npy'), hand_points_normalized_sampled)
        data = np.array([points_normalized, normals_pca_fps_sampled, max_bb3d_len, offset])
        np.save(os.path.join(points_path, item[-19:-4] + '.npy'), data)


def main():
    image_lists = glob.glob(image_path + "/*.png")
    image_lists.sort()
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    # pool.map(get_shadow_points, image_lists)
    for item in image_lists:
        get_shadow_points(item)


if __name__ == '__main__':
    main()