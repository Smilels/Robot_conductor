#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: 
# Date       : 02/02/2021: 18:49
# File Name  : utils

import numpy as np
from IPython import embed


def cal_hand_pose(keypoints):
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
    hand_frame = np.hstack([wrist_x, wrist_y, wrist_z])
    return hand_frame


def pc2uvd(keypoints, mat):
    uvd = np.random.randn(21, 3).astype(np.float32)
    for i in range(0, len(keypoints)):
        uvd[i] = ((1.0 / keypoints[i][2]) * mat @ keypoints[i])
        uvd[i, 2] = keypoints[i][2]
    return uvd


def crop_depth_img(img_ori, uvd, padding):
    img = img_ori.copy()
    top = np.min(uvd[:, 1])
    bottom = np.max(uvd[:, 1])
    left = np.min(uvd[:, 0])
    right = np.max(uvd[:, 0])
    width, height = right - left, bottom - top
    if height > width:
        left_padding = float(height - width)
        top_padding = 0
    else:
        left_padding = 0
        top_padding = float(width - height)

    x_min = int(max(0, int(top - top_padding / 2 - padding / 2)))
    x_max = int(min(img.shape[0] - 1, int(bottom + top_padding / 2 + padding / 2)))
    y_min = int(max(0, int(left - left_padding / 2 - padding / 2)))
    y_max = int(min(img.shape[1] - 1, int(right + left_padding / 2 + padding / 2)))
    img = img[x_min:x_max, y_min:y_max]

    trans = uvd[0].copy().reshape(1, 3)
    trans[:, 0] = trans[:, 0] - float(y_min)
    trans[:, 1] = trans[:, 1] - float(x_min)
    # trans = trans.round().astype(np.int32)
    return img, trans, np.array([x_max, x_min, y_max, y_min])


def uvd2pc(uvd_ori, centerX, centerY, focalLengthX, focalLengthY):
    uvd = uvd_ori.copy()
    v = uvd[0][1]
    u = uvd[0][0]
    Z = uvd[0][2]
    X = int((u - centerX) * Z / focalLengthX)
    Y = int((v - centerY) * Z / focalLengthY)
    points_np = np.array([X, Y, Z]).reshape(3, 1)
    return points_np


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


def spilt():
    import os
    base_path = "/export/home/sli/code/Robot_conductor/data/depth_pose/"
    if os.path.isfile(base_path + "human_pose.npy"):
        pose_np = np.load(base_path + "human_pose.npy")
    else:
        pose_list = os.listdir(base_path + "human_pose/")
        pose_list.sort()
        pose_np = np.array(pose_list)
        np.save(base_path + "human_pose.npy", pose_np)

    label = pose_np[:100000]
    train_sample = int(len(label) * 0.9)
    train = label[:train_sample]
    print(train.shape)
    test = label[train_sample:]
    print(test.shape)
    np.save(base_path + "human_pose_10k.npy", pose_np[:100000])
    np.save(base_path + "train_10k.npy", train)
    np.save(base_path + "test_10k.npy", test)


if __name__ == "__main__":
    # N, K = 80, 40
    # pts = np.random.random_sample((N, 2))
    # farthest_pts = FPS(pts, K)
    # show_paired_depth_images()
    # pass
    spilt()
