#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: save raw points
# Date       : 15/09/2020: 17:41
# File Name  : human_pc.py


import numpy as np
import os, sys
import cv2
import open3d as o3d
import multiprocessing as mp
from utils import depth2pc

show_bbx = 1

focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079

if sys. argv[1] == "tams108":
    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    img_path = os.path.join(base_path, "Human_label/human_full_test/")
    tf_path = os.path.join(base_path, "human_pca_tf/")
    pose_path = os.path.join(base_path, "human_pose/")
    points_path = os.path.join(base_path, "human_points/")
    gt_file = "Human_label/text_annotation.txt"

elif sys. argv[1] == "server":
    base_path = "/data/sli/Bighand2017/"
    img_path = base_path + "images/"
    points_path = base_path + "points_keypoints/human_points/"
    show_bbx = 0
    gt_file = "groundtruth/Training_Annotation.txt"

mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])


def get_human_points(line):
    # read the groundtruth and the image
    frame = line.split(' ')[0].replace("\t", "")

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

    if show_bbx:
        pcd = o3d.geometry.PointCloud()
        pcd_hand = o3d.geometry.PointCloud()
        pcd_key = o3d.geometry.PointCloud()

        # the original points which contains the whole scenario
        pcd.points = o3d.utility.Vector3dVector(points_raw)
        # the original hand points extracted from the whole image
        pcd_hand.points = o3d.utility.Vector3dVector(points)
        pcd_hand.paint_uniform_color([0.9, 0.1, 0.1])  # red
        # keypoints label
        pcd_key.points = o3d.utility.Vector3dVector(keypoints)
        pcd_key.paint_uniform_color([0.1, 0.1, 0.7])
        # the world frame
        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=150, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([pcd_hand, pcd_key], point_show_normal=False)

    if not os.path.exists(points_path):
        os.makedirs(points_path)

    data = np.array([points, keypoints],
                    dtype=object)
    np.save(os.path.join(points_path, frame[:-4] + '.npy'), data)


def main():
    datafile = open(base_path + gt_file, "r")
    lines = datafile.read().splitlines()
    lines.sort()

    if sys.argv[1] == "tams108":
        for line in lines:
            get_human_points(line)
    elif sys.argv[1] == "server":
        cores = mp.cpu_count()
        pool = mp.Pool(processes=cores)
        pool.map(get_human_points, lines[:400000])

    datafile.close()


if __name__ == '__main__':
    main()
