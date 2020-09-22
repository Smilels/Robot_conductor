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


bbx_2d = False
hand_bbx_3d = True
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
    DataFile = open(base_path + "full_annotation/Subject_1/301375_loc_shift_made_by_qi_20180112_v2.txt", "r")
    lines = DataFile.read().splitlines()
    # camera center coordinates and focal length
    mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])

    if hand_bbx_3d and show_bbx:
        # Initialize Visualizer and start animation callback
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # set viewpoint. I want the 0,0,0 viewpoint,maybe need to rotate to camera coord
        ctr = vis.get_view_control()

        pcd = o3d.geometry.PointCloud()
        img = cv2.imread(base_path + "76150/" + 'image_D' + '00000000' + '.png', cv2.IMREAD_ANYDEPTH)
        whole_points = depth2pc(img)
        pcd.points = o3d.utility.Vector3dVector(whole_points)

        pcd_key = o3d.geometry.PointCloud()
        pcd_key.points = o3d.utility.Vector3dVector(np.random.rand(21, 3))
        pcd_key.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(21)])

        lines3d = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [3, 7],
            [2, 6],
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.random.rand(8, 3).tolist())
        line_set.lines = o3d.utility.Vector2iVector(lines3d)
        line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 1] for _ in range(len(lines3d))])

        # world frame
        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=150, origin=[-400, -400, 300])

        hand_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100, origin=[0, 0, 0])

        vis.add_geometry(pcd)
        vis.add_geometry(pcd_key)
        vis.add_geometry(line_set)
        vis.add_geometry(world_frame_vis)
        vis.add_geometry(hand_frame_vis)

    for line in lines[9940:]:
        frame = line.split(' ')[0].replace("\t", "")
        # print(frame)

        label_source = line.split('\t')[1:]
        label = []
        label.append([float(l.replace(" ", "")) for l in label_source[0:63]])

        keypoints = np.array(label)
        keypoints = keypoints.reshape(21, 3)
        keypoints[:, 2] = keypoints[:, 2]  # - 20

        # image path depends on the location of your training dataset
        img = cv2.imread(base_path + "301375/" + 'image_D' + str(frame) + '.png', cv2.IMREAD_ANYDEPTH)

        # get 2d keypoints uv values
        uv = np.random.randn(21, 2).astype(np.float32)
        for i in range(0, len(keypoints)):
            uv[i] = ((1 / keypoints[i][2]) * mat @ keypoints[i])[0:2]

        if bbx_2d:
            # Image coordinates: origin at the top-left corner, u axis going right and v axis going down
            padding = 50

            top = np.min(uv[:, 1]) - padding / 2
            bottom = np.max(uv[:, 1]) + padding / 2
            left = np.min(uv[:, 0]) - padding / 2
            right = np.max(uv[:, 0]) + padding / 2

            # hint: top, bottom left, right are in the pixel coordinate, which is as same as the numpy array order
            x_min = int(max(0, top))
            x_max = int(min(img.shape[0] - 1, bottom))
            y_min = int(max(0, left))
            y_max = int(min(img.shape[1] - 1, right))

            if show_bbx:
                fig = plt.figure(1)

                # palm
                x = [uv[0][0]]
                y = [uv[0][1]]
                plt.scatter(x, y, s=60, c='green')

                # tf
                x = [uv[0][0], uv[1][0], uv[6][0], uv[7][0], uv[8][0]]
                y = [uv[0][1], uv[1][1], uv[6][1], uv[7][1], uv[8][1]]
                plt.scatter(x, y, s=25, c='cyan')
                plt.plot(x, y, 'c', linewidth=1)

                # ff
                x = [uv[0][0], uv[2][0], uv[9][0], uv[10][0], uv[11][0]]
                y = [uv[0][1], uv[2][1], uv[9][1], uv[10][1], uv[11][1]]
                plt.scatter(x, y, s=25, c='blue')
                plt.plot(x, y, 'b', linewidth=1)

                # mf
                x = [uv[0][0], uv[3][0], uv[12][0], uv[13][0], uv[14][0]]
                y = [uv[0][1], uv[3][1], uv[12][1], uv[13][1], uv[14][1]]
                plt.scatter(x, y, s=25, c='red')
                plt.plot(x, y, 'r', linewidth=1)

                # rf
                x = [uv[0][0], uv[4][0], uv[15][0], uv[16][0], uv[17][0]]
                y = [uv[0][1], uv[4][1], uv[15][1], uv[16][1], uv[17][1]]
                plt.scatter(x, y, s=25, c='yellow')
                plt.plot(x, y, 'y', linewidth=1)

                # lf
                x = [uv[0][0], uv[5][0], uv[18][0], uv[19][0], uv[20][0]]
                y = [uv[0][1], uv[5][1], uv[18][1], uv[19][1], uv[20][1]]
                plt.scatter(x[0:], y[0:], s=25, c='magenta')
                plt.plot(x, y, 'm', linewidth=1.5)

                # draw boundingbox
                # hint: here the coordinate system is normal cartersian system,
                # which x axis going right, y axis going up
                x = [x_min, x_max, x_max, x_min, x_min]
                y = [y_max, y_max, y_min, y_min, y_max]
                plt.scatter(y, x, s=25, c='cyan')
                plt.plot(y, x, 'c', linewidth=1)

                plt.imshow(img)
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.000001)
                plt.clf()

            if save:
                b = y_max -y_min
                w = x_max - x_min
                np.save(frame +'_2dbbx.npy', np.array([x_max, y_max, b, w]))

        if hand_bbx_3d:
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

            hand_frame = np.vstack([wrist_x, wrist_y, wrist_z])
            local_keypoints = np.dot((keypoints - keypoints[0]), hand_frame.T)
            r = R.from_matrix(hand_frame)
            axisangle = r.as_rotvec()

            padding = 80

            # hint: z is going inside, x is going right, y is going down
            x_min = np.min(local_keypoints[:, 0]) - padding / 2
            x_max = np.max(local_keypoints[:, 0]) + padding / 2
            y_min = np.min(local_keypoints[:, 1]) - padding / 2
            y_max = np.max(local_keypoints[:, 1]) + padding / 2
            z_min = np.min(local_keypoints[:, 2]) - padding / 2
            z_max = np.max(local_keypoints[:, 2]) + padding / 2

            if show_bbx:
                whole_points = depth2pc(img)

                # draw local 3d boundingbox
                x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
                y = [y_max, y_max, y_min, y_min, y_max, y_max, y_min, y_min]
                z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]
                bbx_keypoints_local = np.hstack(
                    [np.array(x).reshape(8, 1), np.array(y).reshape(8, 1), np.array(z).reshape(8, 1)])
                bbx_keypoints_camera = np.dot(bbx_keypoints_local, np.linalg.inv(hand_frame.T)) + keypoints[0]

                pcd.points = o3d.utility.Vector3dVector(whole_points)
                line_set.points = o3d.utility.Vector3dVector(bbx_keypoints_camera)
                pcd_key.points = o3d.utility.Vector3dVector(keypoints)
                hand_frame_vis.rotate(hand_frame.T, center=(0, 0, 0))
                hand_frame_vis.translate(keypoints[0], relative=False)

                vis.update_geometry(pcd)
                vis.update_geometry(pcd_key)
                vis.update_geometry(line_set)
                vis.update_geometry(world_frame_vis)
                vis.update_geometry(hand_frame_vis)
                vis.poll_events()
                vis.update_renderer()
                hand_frame_vis.rotate(np.linalg.inv(hand_frame.T), center=(0, 0, 0))

            if save:
                b = y_max -y_min
                w = x_max - x_min
                h = z_max - z_min
                pose_bbx = np.hstack([keypoints[0], axisangle, np.array([x_max, y_max, z_max, b, w, h])])
                np.save(frame +'_3dbbx.npy', pose_bbx)

    DataFile.close()
    if show_bbx:
        vis.destroy_window()


if __name__ == '__main__':
    main()
