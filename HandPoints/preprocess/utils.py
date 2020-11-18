#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: 
# Date       : 04/11/2020: 18:49
# File Name  : utils

import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
from IPython import embed
import os, glob
import cv2


def depth2pc(depth, centerX, centerY, focalLengthX, focalLengthY):
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            Z = int(depth[v, u])
            if Z == 0:
                continue
            X = int((u - centerX) * Z / focalLengthX)
            Y = int((v - centerY) * Z / focalLengthY)
            points.append([X, Y, Z])
    points_np = np.array(points)
    return points_np


def pca_rotation(points):
    hand_points = points.copy()
    hand_points_mean = hand_points.mean(axis=0)
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(hand_points)
    hand_points_pca = pca.transform(hand_points) + hand_points_mean
    return hand_points_pca, pca.components_


def robot_pca_rotation(points, transform):
    hand_points = points.copy()
    hand_points_mean = hand_points.mean(axis=0)
    hand_points_pca = np.dot(hand_points, transform.T) + hand_points_mean
    return hand_points_pca


def down_sample(points, SAMPLE_NUM):
    hand_points = points.copy()
    if len(hand_points) > SAMPLE_NUM:
        rand_ind = np.random.choice(len(hand_points), size=SAMPLE_NUM, replace=False)
        hand_points_sampled = hand_points[rand_ind]
    else:
        rand_ind = np.random.choice(len(hand_points), size=SAMPLE_NUM, replace=True)
        hand_points_sampled = hand_points[rand_ind]
    return hand_points_sampled, rand_ind


def get_normal(points, radius=0.1, max_nn=30):
    hand_points = points.copy()
    # the unit of points should be m
    if max(hand_points.min(), hand_points.max(), key=abs) > 200:
        hand_points = hand_points/1000.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hand_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_norm = np.asanyarray(pcd.normals).astype(np.float32)
    return pcd_norm


def normalization_unit(hand_points_pca_sampled):
    offset = np.mean(hand_points_pca_sampled, axis=0)
    hand_points_normalized_sampled = hand_points_pca_sampled - offset

    min = hand_points_normalized_sampled.min(axis=0)
    max = hand_points_normalized_sampled.max(axis=0)
    bb3d_x_len = max-min
    scale = 1.2
    max_bb3d_len = scale * np.max(bb3d_x_len)
    # max_bb3d_len = scale * bb3d_x_len[0]
    hand_points_normalized_sampled /= max_bb3d_len
    return hand_points_normalized_sampled, max_bb3d_len, offset


def normalization_view(hand_points_pca, hand_points_pca_sampled, SAMPLE_NUM):
    min = hand_points_pca_sampled.min(axis=0)
    max = hand_points_pca_sampled.max(axis=0)
    bb3d_x_len = max-min
    scale = 1.2
    max_bb3d_len = scale * np.max(bb3d_x_len)
    # max_bb3d_len = scale * bb3d_x_len[0]
    hand_points_normalized_sampled = hand_points_pca_sampled / max_bb3d_len
    if len(hand_points_pca) < SAMPLE_NUM:
        offset = np.mean(hand_points_pca) / max_bb3d_len
    else:
        offset = np.mean(hand_points_normalized_sampled)
    hand_points_normalized_sampled = hand_points_normalized_sampled - offset
    return hand_points_normalized_sampled


def normalization_mean(hand_points_pca_sampled):
    hand_points_normalized_sampled = hand_points_pca_sampled - hand_points_pca_sampled.mean(axis=0)
    return hand_points_normalized_sampled, hand_points_pca_sampled.mean(axis=0)


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS(points, K):
    pts = points.copy()
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def FPS_idx(points, K):
    pts = points.copy()
    farthest_pts = np.zeros((K, 3))
    farthest_pts_idx = np.zeros(K)
    upper_bound = pts.shape[0] - 1
    first_idx = np.random.randint(0, upper_bound)
    farthest_pts[0] = pts[first_idx]
    farthest_pts_idx[0] = first_idx
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_idx[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, farthest_pts_idx.astype(np.int64)


def show_paired_depth_images():
    from random import shuffle
    base_path = "/data/shuang_data/Bighand2017/"
    file_list = glob.glob(base_path+'depth_shadow/*.png')
    # file_list.sort()
    shuffle(file_list)
    file_number = len(file_list)

    for i in range(file_number):
        img = cv2.imread(file_list[i], cv2.IMREAD_ANYDEPTH)
        img[img == 1000] = 0
        if img is None:
                continue
        human_img = cv2.imread(base_path+'images/'+file_list[i][-20:-4]+'.png', cv2.IMREAD_ANYDEPTH)

        com = np.hstack([img, human_img])
        n = cv2.normalize(com, com, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow( file_list[i][:-4], n)
        key = cv2.waitKey()

        if key == 27:    # Esc key to stop
            break
        elif key==ord('a'):  # normally -1 returned,so don't print it
            cv2.destroyAllWindows()
            continue


def consequent_same_image_check():
    base_path = "/data/shuang_data/Bighand2017/"
    file_list = glob.glob(base_path+'depth_shadow/*.png')
    file_list.sort()
    file_number = len(file_list)
    for i in range(file_number-1):
     img = cv2.imread(file_list[i], cv2.IMREAD_ANYDEPTH)
     img_ = cv2.imread(file_list[i+1], cv2.IMREAD_ANYDEPTH)
     if (img == img_).all():
        print(file_list[i+1][-20:])


def human_shadow_points_check():
    pcd_human = o3d.geometry.PointCloud()
    pcd_shadow = o3d.geometry.PointCloud()

    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    points_lists = glob.glob(base_path+'points_human/*.npy')
    points_lists.sort()
    # print(len(points_lists))
    for item in points_lists[65:]:
        print(item[-19::])
        points_human = np.load(item)
        pcd_human.points = o3d.utility.Vector3dVector(points_human)
        pcd_human.paint_uniform_color([0.9, 0.1, 0.1])  # red

        try:
            points_shadow = np.load("/homeL/shuang/ros_workspace/tele_ws/src/dataset/points_shadow/" + item[-19::])
            pcd_shadow.points = o3d.utility.Vector3dVector(points_shadow + np.array([0, 0, 0]))
            pcd_shadow.paint_uniform_color([0.1, 0.1, 0.7])  # blue
            world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([world_frame_vis, pcd_human, pcd_shadow])
        except:
            print("%s does not have corresponding robot data, maybe because the collision hand pose" % (item[-19::]))


def human_shadow_points_check2():
    pcd_human = o3d.geometry.PointCloud()
    pcd_shadow = o3d.geometry.PointCloud()

    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    points_lists = glob.glob(base_path+'Human_label/human_full_test/*.png')
    points_lists.sort()
    print(len(points_lists))
    for item in points_lists:
        img = cv2.imread(item, cv2.IMREAD_ANYDEPTH)
        focalLengthX = 475.065948
        focalLengthY = 475.065857
        centerX = 315.944855
        centerY = 245.287079
        print(centerX, centerY, focalLengthX, focalLengthY)

        points_human = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)
        pcd_human.points = o3d.utility.Vector3dVector(points_human)
        pcd_human.paint_uniform_color([0.9, 0.1, 0.1])  # red

        img_shadow = cv2.imread("/homeL/shuang/ros_workspace/tele_ws/src/dataset/depth_shadow/" + item[-19::], cv2.IMREAD_ANYDEPTH)
        image_width = 640
        image_height = 480
        near = 0.1 * 1000
        max_depth = 2.0 * 1000
        fov_y = 1.02974
        import math
        e = 1.0 / math.tan(fov_y / 2)
        # focal_length can be calculated as f = max(u_res, v_res) / (2*tan(fov/2),
        # where FoV is the perspective angle (from the vision sensor properties)
        focalLengthX = e * image_width / 2
        focalLengthY = e * image_width / 2

        centerX = image_width / 2
        centerY = image_height / 2
        print(centerX, centerY, focalLengthX, focalLengthY)
        img_shadow[img_shadow == 1000] = 0
        points_shadow = depth2pc(img_shadow, centerX, centerY, focalLengthX, focalLengthY)
        pcd_shadow.points = o3d.utility.Vector3dVector(points_shadow + np.array([200, 0, 0]))
        pcd_shadow.paint_uniform_color([0.1, 0.1, 0.7])  # blue
        # the world frame
        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=150, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([world_frame_vis, pcd_human, pcd_shadow])


if __name__ == "__main__":
    # N, K = 80, 40
    # pts = np.random.random_sample((N, 2))
    # farthest_pts = FPS(pts, K)
    # show_paired_depth_images()
    # pass
    human_shadow_points_check()
