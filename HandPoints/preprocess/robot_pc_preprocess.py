#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: bighand depth to normalized pointclouds
# Date       : 15/09/2020: 17:41
# File Name  : robot_pc_preprocess.py


import os, sys
import cv2
import open3d as o3d
import math
from utils import depth2pc, robot_pca_rotation, down_sample, get_normal, normalization_unit, FPS_idx
import glob
import numpy as np
import multiprocessing as mp
from IPython import embed


save_norm = 0
vis_points = 0
do_pca = 1

DOWN_SAMPLE_NUM = 2048
FPS_SAMPLE_NUM = 512

image_width = 640
image_height = 480
near = 0.1 * 1000
max_depth = 2.0 * 1000
fov_y = 1.02974
e = 1.0 / math.tan(fov_y / 2)
# focal_length can be calculated as f = max(u_res, v_res) / (2*tan(fov/2),
# where FoV is the perspective angle (from the vision sensor properties)
focalLengthX = e * image_width / 2.0
focalLengthY = e * image_width / 2.0

centerX = image_width/2.0
centerY = image_height/2.0


if sys. argv[1] == "tams108":
    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    img_path = os.path.join(base_path, "depth_shadow/")
    tf_path = os.path.join(base_path, "human_pca_tf/")
    points_path = os.path.join(base_path, "points_shadow/")
elif sys. argv[1] == "server":
    base_path = "./data/"
    img_path = base_path + "depth_shadow/"
    tf_path = base_path + "points_pca/human_pca_tf/"
    points_path = base_path + "points_pca/points_shadow/"
    vis_points = 0

if do_pca:
    tf_lists = os.listdir(tf_path)
    tf_lists.sort()
    f_index = {}
    for ind, line in enumerate(tf_lists):
        f_index[line[:-4]] = ind


def get_shadow_points(item):
    print(item[-19::])
    # embed()
    if do_pca:
        try:
            line = f_index[item[-19:-4]]
            pc_transfrom = np.load(tf_path + item[-19:-4] + ".npy")
        except:
            print("%s does not have reasonable human data" % (item[-19::]))
            return

    img = cv2.imread(item, cv2.IMREAD_ANYDEPTH)

    # 1 get hand points
    img[img == 1000] = 0
    points = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)

    if len(points) < 300:
        print("%s hand points is %d, which is less than 300. Maybe it's a broken image" % (item[-19::], len(points)))
        return

    # 2 PCA rotation
    if do_pca:
        points_pca = robot_pca_rotation(points, pc_transfrom)
    else:
        points_pca = points

    # 3 downsampling
    points_pca_sampled, rand_ind = down_sample(points_pca, DOWN_SAMPLE_NUM)

    # 4 FPS
    points_pca_fps_sampled, farthest_pts_idx = FPS_idx(points_pca_sampled, FPS_SAMPLE_NUM)

    # 5 compute surface normal
    normals_pca = get_normal(points_pca)
    normals_pca_sampled = normals_pca[rand_ind]
    normals_pca_fps_sampled = normals_pca_sampled[farthest_pts_idx]

    # 6 normalize point cloud
    points_normalized, max_bb3d_len, offset = normalization_unit(points_pca_fps_sampled)

    if vis_points:
        pcd = o3d.geometry.PointCloud()
        pcd_hand = o3d.geometry.PointCloud()
        pcd_pca = o3d.geometry.PointCloud()
        pcd_pca_sample = o3d.geometry.PointCloud()
        pcd_key = o3d.geometry.PointCloud()
        pcd_normalized = o3d.geometry.PointCloud()
        pcd_fps_sample = o3d.geometry.PointCloud()

        # the original hand points extracted from the whole image
        pcd_hand.points = o3d.utility.Vector3dVector(points)
        pcd_hand.paint_uniform_color([0.9, 0.1, 0.1])  # red
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

        o3d.visualization.draw_geometries([pcd_hand], point_show_normal=False)

    if not os.path.exists(points_path):
        os.makedirs(points_path)
    if save_norm:
        data = np.array([points_normalized, normals_pca_fps_sampled, max_bb3d_len, offset],
                        dtype=object)
        np.save(os.path.join(points_path, item[-19:-4] + '.npy'), data)
    else:
        np.save(os.path.join(points_path, item[-19:-4] + '.npy'), points_normalized)


def main():
    image_lists = glob.glob(img_path + "*.png")
    image_lists.sort()
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    pool.map(get_shadow_points, image_lists)
    # for item in image_lists:
    #     get_shadow_points(item)


if __name__ == '__main__':
    main()
