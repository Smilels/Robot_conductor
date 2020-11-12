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
    hand_points_norm = hand_points - hand_points.mean(axis=0)
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(hand_points_norm)
    hand_points_pca = pca.transform(hand_points_norm) + hand_points.mean(axis=0)
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
        hand_points /= 1000
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hand_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_norm = np.asanyarray(pcd.normals).astype(np.float32)
    return pcd_norm


def normalization_unit(hand_points_pca, hand_points_pca_sampled, SAMPLE_NUM):
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
    return hand_points_normalized_sampled


def farthest_point_sampling_fast(points, sample_num):
    hand_points = points
    # hand_points: N*3
    pc_num = hand_points.shape[0]

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
            diff = hand_points - hand_points[sampled_idx[idx]].reshape(1, 3)
            dist = np.sum(np.multiply(diff, diff), axis=1)

            # update distances to record the minimum distance of each point
            # in the sample from all existing sample points
            # and not the sample points themselves
            mask = (min_dist > dist) & (min_dist > 1e-8)
            min_dist[mask] = dist[mask]
            farthest = np.argmax(min_dist)

    return np.unique(sampled_idx)


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
     if (img==img_).all():
        print(file_list[i+1][-20:])


if __name__ == "__main__":
    # N, K = 80, 40
    # pts = np.random.random_sample((N, 2))
    # farthest_pts = FPS(pts, K)
    show_paired_depth_images()
    # consequent_same_image_check()
