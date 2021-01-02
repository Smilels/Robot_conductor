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
from utils import depth2pc

show_bbx = 1

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079

DOWN_SAMPLE_NUM = 2048
FPS_SAMPLE_NUM = 1024

if sys.argv[1] == "tams108":
    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    img_path = os.path.join(base_path, "Human_label/human_full_test/")
    tf_path = os.path.join(base_path, "human_pca_tf/")
    points_path = os.path.join(base_path, "points_human/")
elif sys.argv[1] == "server":
    base_path = "./data/"
    img_path = base_path + "images/"
    tf_path = os.path.join(base_path, "points_no_pca/human_pca_tf/")
    points_path = base_path + "points_no_pca/points_human/"
    show_bbx = 0

mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])


def get_human_points(line):
    # read the groundtruth and the image
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
    uv = np.random.randn(21, 2)

    for i in range(0, len(keypoints)):
        uv[i] = ((1 / keypoints[i][2]) * mat @ keypoints[i])[0:2]

    top = np.min(uv[:, 1])
    bottom = np.max(uv[:, 1])
    left = np.min(uv[:, 0])
    right = np.max(uv[:, 0])
    width, height = right - left, bottom - top
    padding = 10
    if height > width:
        left_padding = float(height - width)
        top_padding = 0
    else:
        left_padding = 0
        top_padding = float(width - height)
    x_min = int(max(0, top - top_padding / 2 - padding / 2))
    x_max = int(min(img.shape[0] - 1, bottom + top_padding / 2 + padding / 2))
    y_min = int(max(0, left - left_padding / 2 - padding / 2))
    y_max = int(min(img.shape[1] - 1, right + left_padding / 2 + padding / 2))
    img = img[x_min:x_max, y_min:y_max]

    # 2 get hand points
    points_raw = depth2pc(img, centerX, centerY, focalLengthX, focalLengthY)

    h, w = img.shape
    angle = np.random.randint(-180, 180)
    M = cv2.getRotationMatrix2D((h / 2.0, w/ 2.0), angle, 1)
    img_rot1 = cv2.warpAffine(img, M, (h, w), cv2.INTER_AREA)
    img_rot2 = cv2.warpAffine(img, M, (h, w), cv2.INTER_CUBIC)
    img_rot3 = cv2.warpAffine(img, M, (h, w), cv2.INTER_LINEAR)
    points_rot1 = depth2pc(img_rot1, centerX, centerY, focalLengthX, focalLengthY)
    points_rot2 = depth2pc(img_rot2, centerX, centerY, focalLengthX, focalLengthY)
    points_rot3 = depth2pc(img_rot3, centerX, centerY, focalLengthX, focalLengthY)

    if show_bbx:
        pcd = o3d.geometry.PointCloud()
        pcd_hand1 = o3d.geometry.PointCloud()
        pcd_hand2 = o3d.geometry.PointCloud()
        pcd_hand3 = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points_raw)
        pcd_hand1.points = o3d.utility.Vector3dVector(points_rot1 + np.array([200, 0, 0]))
        pcd_hand1.paint_uniform_color([0.9, 0.1, 0.1])  # red
        pcd_hand2.points = o3d.utility.Vector3dVector(points_rot2 + np.array([400, 0, 0]))
        pcd_hand2.paint_uniform_color([0.9, 0.1, 0.1])  # red
        pcd_hand3.points = o3d.utility.Vector3dVector(points_rot3 + np.array([600, 0, 0]))
        pcd_hand3.paint_uniform_color([0.9, 0.1, 0.1])  # red

        o3d.visualization.draw_geometries([pcd, pcd_hand1, pcd_hand2, pcd_hand3], point_show_normal=False)


def main():
    # datafile = open(base_path + "groundtruth/Training_Annotation.txt", "r")
    datafile = open(base_path + "Human_label/text_annotation.txt", "r")
    lines = datafile.read().splitlines()
    lines.sort()
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    # pool.map(get_human_points, lines[:30000])
    for line in lines:
        get_human_points(line)
    datafile.close()


if __name__ == '__main__':
    main()
