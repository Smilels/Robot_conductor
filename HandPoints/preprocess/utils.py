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


def pca_rotation(hand_points):
    hand_points_norm = hand_points - hand_points.mean(axis=0)
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(hand_points_norm)
    hand_points_pca = pca.transform(hand_points_norm) + hand_points.mean(axis=0)
    return hand_points_pca


def down_sample(hand_points, SAMPLE_NUM):
    if len(hand_points) > SAMPLE_NUM:
        rand_ind = np.random.choice(len(hand_points), size=SAMPLE_NUM, replace=False)
        hand_points_sampled = hand_points[rand_ind]
    else:
        rand_ind = np.random.choice(len(hand_points), size=SAMPLE_NUM, replace=True)
        hand_points_sampled = hand_points[rand_ind]
    return hand_points_sampled, rand_ind


def get_normal(points: np.ndarray, radius=0.1, max_nn=30):
    # the unit of points should be m
    if max(points.min(), points.max(), key=abs) > 200:
        points /= 1000
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_norm = np.asanyarray(pcd.normals).astype(np.float32)
    return pcd_norm


def normalization(hand_points_pca, hand_points_pca_sampled, SAMPLE_NUM):
    # 5 normalize point cloud
    x_min_max = [np.min(hand_points_pca_sampled[:, 0]), np.max(hand_points_pca_sampled[:, 0])]
    y_min_max = [np.min(hand_points_pca_sampled[:, 1]), np.max(hand_points_pca_sampled[:, 1])]
    z_min_max = [np.min(hand_points_pca_sampled[:, 2]), np.max(hand_points_pca_sampled[:, 2])]
    scale = 1.2
    bb3d_x_len = scale * (x_min_max[1] - x_min_max[0])
    bb3d_y_len = scale * (y_min_max[1] - y_min_max[0])
    bb3d_z_len = scale * (z_min_max[1] - z_min_max[0])
    max_bb3d_len = bb3d_x_len
    hand_points_normalized_sampled = hand_points_pca_sampled / max_bb3d_len
    if len(hand_points_pca) < SAMPLE_NUM:
        offset = np.mean(hand_points_pca) / max_bb3d_len
    else:
        offset = np.mean(hand_points_normalized_sampled)
    hand_points_normalized_sampled = hand_points_normalized_sampled - offset
    return hand_points_normalized_sampled


def farthest_point_sampling_fast(point_cloud, sample_num):
    # point_cloud: N*3
    pc_num = point_cloud.shape[0]

    if pc_num <= sample_num:
        # if pc_num <= sample_num, expand it to reach the amount requirement
        sampled_idx_part1 = np.arange(pc_num)
        sampled_idx_part2 = np.random.choice(pc_num, size=sample_num - pc_num, replace=False)
        sampled_idx = np.concatenate((sampled_idx_part1, sampled_idx_part2))
    else:
        sampled_idx = np.zeros(sample_num).astype(np.int32)
        farthest = np.random.randint(pc_num)
        min_dist = np.ones(pc_num) * 1e10

        for idx in range(sample_num):
            sampled_idx[idx] = farthest
            diff = point_cloud - point_cloud[sampled_idx[idx]].reshape(1, 3)
            dist = np.sum(np.multiply(diff, diff), axis=1)

            # update distances to record the minimum distance of each point
            # in the sample from all existing sample points
            # and not the sample points themselves
            mask = (min_dist > dist) & (min_dist > 1e-8)
            min_dist[mask] = dist[mask]
            farthest = np.argmax(min_dist)

    return np.unique(sampled_idx)
