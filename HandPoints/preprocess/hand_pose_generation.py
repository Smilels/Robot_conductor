#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : bbx_generate_save
# Purpose : generate, visualize, and save 2d and 3d bbx
# Creation Date : 20-06-2020
# Created By : Shuang Li

import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import csv


hand_bbx_3d = False
save = True
show_bbx = False

focalLengthX = 475.065948
focalLengthY = 475.065857

centerX = 315.944855
centerY = 245.287079


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


def main():
    """crop human hand images to 100*100"""
    base_path = "../data/"
    DataFile = open(base_path + "bighand2017_test_annotation.txt", "r")
    csvSum = open(base_path + "shadow_pose2017.csv", "w")

    lines = DataFile.read().splitlines()
    # camera center coordinates and focal length
    mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])
    writer = csv.writer(csvSum)
    if hand_bbx_3d and show_bbx:
        # Initialize Visualizer and start animation callback
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # set viewpoint. I want the 0,0,0 viewpoint,maybe need to rotate to camera coord
        ctr = vis.get_view_control()

        pcd = o3d.geometry.PointCloud()
        img = cv2.imread(base_path + "bighand2017_test/" + 'image_D00000001.png', cv2.IMREAD_ANYDEPTH)
        whole_points = depth2pc(img)
        pcd.points = o3d.utility.Vector3dVector(whole_points)

        pcd_key = o3d.geometry.PointCloud()
        pcd_key.points = o3d.utility.Vector3dVector(np.random.rand(21, 3))
        pcd_key.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(21)])

        # world frame
        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=150, origin=[-400, -400, 300])

        hand_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100, origin=[0, 0, 0])

        vis.add_geometry(pcd)
        vis.add_geometry(pcd_key)
        vis.add_geometry(world_frame_vis)
        vis.add_geometry(hand_frame_vis)

    for line in lines:
        frame = line.split(' ')[0].replace("\t", "")
        # print(frame)

        label_source = line.split('\t')[1:]
        label = []
        label.append([float(l.replace(" ", "")) for l in label_source[0:63]])

        keypoints = np.array(label)
        keypoints = keypoints.reshape(21, 3)
        keypoints[:, 2] = keypoints[:, 2]  # - 20

        # get 2d keypoints uv values
        uv = np.random.randn(21, 2).astype(np.float32)
        for i in range(0, len(keypoints)):
            uv[i] = ((1 / keypoints[i][2]) * mat @ keypoints[i])[0:2]

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
            hand_frame = np.hstack([wrist_x, wrist_y, wrist_z])

            if show_bbx:
                # image path depends on the location of your training dataset
                img = cv2.imread(base_path + "bighand2017_test/" + str(frame), cv2.IMREAD_ANYDEPTH)
                whole_points = depth2pc(img)
                pcd.points = o3d.utility.Vector3dVector(whole_points)
                pcd_key.points = o3d.utility.Vector3dVector(keypoints)
                hand_frame_vis.rotate(hand_frame.T, center=(0, 0, 0))
                hand_frame_vis.translate(keypoints[0], relative=False)

                vis.update_geometry(pcd)
                vis.update_geometry(pcd_key)
                vis.update_geometry(world_frame_vis)
                vis.update_geometry(hand_frame_vis)
                vis.poll_events()
                vis.update_renderer()
                hand_frame_vis.rotate(np.linalg.inv(hand_frame.T), center=(0, 0, 0))

            if save:
                pose_bbx = [keypoints[0], hand_frame]
                writer.writerow(pose_bbx)

    csvSum.close()
    if show_bbx:
        vis.destroy_window()


if __name__ == '__main__':
    main()