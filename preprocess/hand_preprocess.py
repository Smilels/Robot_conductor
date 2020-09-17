#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: 
# Date       : 15/09/2020: 17:41
# File Name  : hand_preprocess.py


import numpy as np
import os
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from IPython import embed
from sklearn.decomposition import PCA

save_points = True
show_bbx = True
save_local_frame = True

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079

SAMPLE_NUM = 1024
SAMPLE_NUM_level1 = 512
SAMPLE_NUM_level2 = 128


def depth2pc(depth):
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


def get_normal(points: np.ndarray, radius=0.1, max_nn=30):
    # the unit of points should be m
    if max(points.min(), points.max(), key=abs) > 200:
        points /= 1000
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_norm = np.asanyarray(pcd.normals).astype(np.float32)
    return pcd_norm


def farthest_point_sampling_fast(point_cloud, sample_num):
    # point_cloud: N*3
    pc_num = point_cloud.shape[0]

    if pc_num <= sample_num:
        # if pc_num <= sample_num, expand it to reach the amount requirement
        sampled_idx_part1 = np.arange(pc_num)
        sampled_idx_part2 = np.random.choice(pc_num, size=sample_num-pc_num, replace=False)
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


def main():
    """save hand points"""
    base_path = "../data/"
    img_path = base_path + "bighand2017_test/"
    points_path = base_path + "points/"
    local_frame_path = base_path + "local_frame/"
    DataFile = open(base_path + "bighand2017_test_annotation.txt", "r")
    lines = DataFile.read().splitlines()
    # camera center coordinates and focal length
    mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])

    try:
        for line in lines:
            # 1 read groundtruth and image
            frame = line.split(' ')[0].replace("\t", "")
            # print(frame)

            # 1.1 image path depends on the location of your training dataset
            try:
                img = cv2.imread(img_path + str(frame), cv2.IMREAD_ANYDEPTH)
            except RuntimeError:
                print("no Image")
                continue

            label_source = line.split('\t')[1:]
            label = []
            label.append([float(l.replace(" ", "")) for l in label_source[0:63]])
            keypoints = np.array(label).reshape(21, 3)

            # 1.3 get hand points
            padding = 80
            points = depth2pc(img)

            x_min_max = [np.min(keypoints[:, 0] - padding / 2), np.max(keypoints[:, 0]) + padding / 2]
            y_min_max = [np.min(keypoints[:, 1] - padding / 2), np.max(keypoints[:, 1]) + padding / 2]
            z_min_max = [np.min(keypoints[:, 2] - padding / 2), np.max(keypoints[:, 2]) + padding / 2]
            hand_points = points[np.where((points[:, 0] > x_min_max[0]) & (points[:, 0] < x_min_max[1]) &
                                               (points[:, 1] > y_min_max[0]) & (points[:, 1] < y_min_max[1]) &
                                               (points[:, 2] > z_min_max[0]) & (points[:, 2] < z_min_max[1]))]

            # 2 PCA rotation
            hand_points_norm = hand_points - hand_points.mean(axis=0)
            pca = PCA(n_components=3, svd_solver='full')
            pca.fit(hand_points_norm)
            hand_points_pca = pca.transform(hand_points_norm) + hand_points.mean(axis=0)

            # 3 downsampling
            if len(hand_points_pca) > SAMPLE_NUM:
                rand_ind = np.random.choice(len(hand_points_pca), size=SAMPLE_NUM, replace=False)
                hand_points_pca_sampled = hand_points_pca[rand_ind]
            elif len(hand_points_pca) > 500:
                rand_ind = np.random.choice(len(hand_points_pca), size=SAMPLE_NUM, replace=True)
                hand_points_pca_sampled = hand_points_pca[rand_ind]
            else:
                print("hand points less than 500, maybe it's a broken image")
                break

            # 4 compute surface normal
            normals_pca = get_normal(hand_points_pca)
            normals_pca_sampled = normals_pca[rand_ind]

            # 5 normalize point cloud
            x_min_max = [np.min(hand_points_pca_sampled[:,0]), np.max(hand_points_pca_sampled[:,0])]
            y_min_max = [np.min(hand_points_pca_sampled[:,1]), np.max(hand_points_pca_sampled[:,1])]
            z_min_max = [np.min(hand_points_pca_sampled[:,2]), np.max(hand_points_pca_sampled[:,2])]
            scale = 1.2
            bb3d_x_len = scale*(x_min_max[1]-x_min_max[0])
            bb3d_y_len = scale*(y_min_max[1]-y_min_max[0])
            bb3d_z_len = scale*(z_min_max[1]-z_min_max[0])
            max_bb3d_len = bb3d_x_len
            hand_points_normalized_sampled = hand_points_pca_sampled/max_bb3d_len
            if len(hand_points_pca) < SAMPLE_NUM:
                offset = np.mean(hand_points_pca)/max_bb3d_len
            else:
                offset = np.mean(hand_points_normalized_sampled)
            hand_points_normalized_sampled = hand_points_normalized_sampled - offset

            # 6 FPS Sampling
            pc = np.concatenate([hand_points_normalized_sampled, normals_pca_sampled], axis=1)
            # 1st level
            sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, SAMPLE_NUM_level1)
            other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1)
            new_idx = np.concatenate((sampled_idx_l1, other_idx))
            pc = pc[new_idx]
            # 2nd level
            sampled_idx_l2 = farthest_point_sampling_fast(hand_points_normalized_sampled[:SAMPLE_NUM_level1],
                                                          SAMPLE_NUM_level2)
            other_idx = np.setdiff1d(np.arange(SAMPLE_NUM_level1), sampled_idx_l2)
            new_idx = np.concatenate((sampled_idx_l2, other_idx))
            pc[:SAMPLE_NUM_level1, :] = pc[new_idx]

            if show_bbx:
                pcd = o3d.geometry.PointCloud()
                pcd_hand = o3d.geometry.PointCloud()
                pcd_pca = o3d.geometry.PointCloud()
                pcd_pca_sample = o3d.geometry.PointCloud()
                pcd_key = o3d.geometry.PointCloud()
                pcd_normalized = o3d.geometry.PointCloud()
                pcd_normalized_sample = o3d.geometry.PointCloud()

                pcd_pca_sample.points = o3d.utility.Vector3dVector(hand_points_pca_sampled)
                pcd_key.points = o3d.utility.Vector3dVector(keypoints)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd_hand.points = o3d.utility.Vector3dVector(hand_points)
                pcd_pca.points = o3d.utility.Vector3dVector(hand_points_pca)

                pcd_normalized.points = o3d.utility.Vector3dVector(hand_points_normalized_sampled)
                pcd_normalized_sample.points = o3d.utility.Vector3dVector(pc[:SAMPLE_NUM_level1, :][:, 0:3] + np.array([0,0,0.1]))

                pcd_hand.paint_uniform_color([0.9, 0.1, 0.1])
                pcd_key.paint_uniform_color([0.1, 0.1, 0.7])
                pcd_pca.paint_uniform_color([0.1, 0.9, 0.1])
                pcd_pca_sample.paint_uniform_color([0.1, 0.1, 0.7])
                o3d.visualization.draw_geometries([pcd, pcd_key, pcd_pca, pcd_pca_sample], point_show_normal=False)
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

    except KeyboardInterrupt:
        exit()
    DataFile.close()


if __name__ == '__main__':
    main()
