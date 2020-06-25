#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name :depth_image_crop
# Purpose : crop dataset to 100*100
# Creation Date : 01-08-2018
# Created By : Shuang Li

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from IPython import embed
from mayavi import mlab
import open3d as o3d
import time


save_2d_bbx = False
save_3d_bbx = False
save_3d_hand_bbx = True

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


def pcshow(points_np, scale_factor=1, color=(1, 1, 1)):
        mlab.points3d(points_np[:, 0], points_np[:, 1], points_np[:, 2], color=color, scale_factor=scale_factor)


def main():
    """crop human hand images to 100*100"""
    base_path = "../data/"
    DataFile = open(base_path + "full_annotation/Subject_1/76150_loc_shift_made_by_qi_20180112_v2.txt", "r")
    lines = DataFile.read().splitlines()
    # camera center coordinates and focal length
    mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])

    # Initialize Visualizer and start animation callback
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # set viewpoint. I want the 0,0,0 viewpoint,maybe need to rotate to camera coord
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=0)
    
    pcd = o3d.geometry.PointCloud()
    img = cv2.imread(base_path + "76150/" + 'image_D' + '00000000' + '.png', cv2.IMREAD_ANYDEPTH)
    whole_points = depth2pc(img)
    pcd.points = o3d.utility.Vector3dVector(whole_points)
    pcd_key = o3d.geometry.PointCloud()
    pcd_key.points = o3d.utility.Vector3dVector(np.random.rand(21, 3))

    lines3d = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 1],
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

    vis.add_geometry(pcd)
    vis.add_geometry(pcd_key)
    vis.add_geometry(line_set)

    for line in lines[10740:]:
        frame = line.split(' ')[0].replace("\t", "")
        label_source = line.split('\t')[1:]
        label = []
        label.append([float(l.replace(" ", "")) for l in label_source[0:63]])

        keypoints = np.array(label)
        keypoints = keypoints.reshape(21, 3)
        keypoints[:, 2] = keypoints[:, 2] # - 20

        # image path depends on the location of your training dataset
        img = cv2.imread(base_path + "76150/" + 'image_D' + str(frame) + '.png', cv2.IMREAD_ANYDEPTH)

        # get 2d keypoints uv values
        uv = np.random.randn(21, 2).astype(np.float32)
        for i in range(0, len(keypoints)):
            uv[i] = ((1 / keypoints[i][2]) * mat @ keypoints[i])[0:2]
        # embed()
        # uvd = np.hstack([uv, keypoints[:, 2].reshape(21, 1)])

        if save_2d_bbx:
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

            # draw boundingbox
            # hint: here the coordinate system is normal cartersian system, which x axis going right, y axis going up
            x = [x_min, x_max, x_max,  x_min, x_min]
            y = [y_max, y_max, y_min, y_min, y_max]
            plt.scatter(y, x, s=25, c='cyan')
            plt.plot(y, x, 'c', linewidth=1)

            plt.imshow(img)
            # fig = plt.figure(2)
            # img_crop = img[x_min:x_max, y_min:y_max]
            # plt.imshow(img_crop)

            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.000001)
            plt.clf()

            # save 2d bbx
            # np.save(frame +'_2dbbx.npy', np.array([x_max, x_min, y_max, y_min]))

        if save_3d_bbx:
            mlab.figure()
            whole_points = depth2pc(img)
            pcshow(whole_points)

            padding = 20
            z_padding = 50

            # hint: z is going inside, x is going right, y is going down
            x_min = np.min(keypoints[:, 0]) - padding / 2
            x_max = np.max(keypoints[:, 0]) + padding / 2
            y_min = np.min(keypoints[:, 1]) - padding / 2
            y_max = np.max(keypoints[:, 1]) + padding / 2
            z_min = np.min(keypoints[:, 2]) - z_padding / 2
            z_max = np.max(keypoints[:, 2]) + padding / 2

            # draw 3d boundingbox
            x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
            y = [y_max, y_max, y_min, y_min, y_max, y_max, y_min, y_min]
            z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]
            mlab.plot3d(x, y, z, color=(0.5, 1, 1), tube_radius=2)
            x = [x_min, x_min]
            y = [y_min, y_min]
            z = [z_min, z_max]
            mlab.plot3d(x, y, z, color=(0.5, 1, 1), tube_radius=2)
            x = [x_max, x_max]
            y = [y_max, y_max]
            z = [z_min, z_max]
            mlab.plot3d(x, y, z, color=(0.5, 1, 1), tube_radius=2)
            x = [x_max, x_max]
            y = [y_min, y_min]
            z = [z_min, z_max]
            mlab.plot3d(x, y, z, color=(0.5, 1, 1), tube_radius=2)

            # draw keypoints
            x1 = keypoints[:, 0]
            y1 = keypoints[:, 1]
            z1 = keypoints[:, 2]
            mlab.points3d(x1, y1, z1, color=(0.5, 0.5, 1), scale_factor=10)
            mlab.show()

            # save 3d bbx
            # np.save(frame +'_2dbbx.npy', np.array([x_max, x_min, y_max, y_min,z_max, z_min]))

        if save_3d_hand_bbx:
            # mlab.clf()
            # mlab.figure(size=(1280, 960))

            whole_points = depth2pc(img)
            pcd.points = o3d.utility.Vector3dVector(whole_points)

            # pcshow(whole_points)

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
            wrist_y = np.cross(lf_palm, rf_palm)
            wrist_y /= np.linalg.norm(wrist_y)
            wrist_x = np.cross(wrist_y, wrist_z)
            if np.linalg.norm(wrist_x) != 0:
                wrist_x /= np.linalg.norm(wrist_x)
            #
            # mlab.quiver3d(keypoints[0][0], keypoints[0][1], keypoints[0][2], wrist_x[0], wrist_x[1], wrist_x[2],
            #               scale_factor=50, line_width=0.5, color=(1, 0, 0), mode='arrow')
            # mlab.quiver3d(keypoints[0][0], keypoints[0][1], keypoints[0][2], wrist_y[0], wrist_y[1], wrist_y[2],
            #               scale_factor=50, line_width=0.5, color=(0, 1, 0), mode='arrow')
            # mlab.quiver3d(keypoints[0][0], keypoints[0][1], keypoints[0][2], wrist_z[0], wrist_z[1], wrist_z[2],
            #               scale_factor=50, line_width=0.5, color=(0, 0, 1), mode='arrow')

            local_frame = np.vstack([wrist_x, wrist_y, wrist_z])
            local_keypoints = np.dot((keypoints - keypoints[0]), local_frame.T)

            padding = 80

            # hint: z is going inside, x is going right, y is going down
            x_min = np.min(local_keypoints[:, 0]) - padding / 2
            x_max = np.max(local_keypoints[:, 0]) + padding / 2
            y_min = np.min(local_keypoints[:, 1]) - padding / 2
            y_max = np.max(local_keypoints[:, 1]) + padding / 2
            z_min = np.min(local_keypoints[:, 2]) - padding / 2
            z_max = np.max(local_keypoints[:, 2]) + padding / 2

            # draw 3d boundingbox
            x = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min]
            y = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min]
            z = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_max]
            bbx_keypoints_local = np.hstack([np.array(x).reshape(10,1), np.array(y).reshape(10,1), np.array(z).reshape(10,1)])
            bbx_keypoints_camera = np.dot(bbx_keypoints_local, np.linalg.inv(local_frame.T)) + keypoints[0]

            line_set.points = o3d.utility.Vector3dVector(bbx_keypoints_camera.tolist())
            line_set.lines = o3d.utility.Vector2iVector(lines3d)
            colors = [[0.5, 0.5, 0] for i in range(len(lines3d))]
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # draw keypoints
            pcd_key.points = o3d.utility.Vector3dVector(keypoints)
            pcd_key.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0] for i in range(len(keypoints))])

            vis.update_geometry(pcd)
            vis.update_geometry(pcd_key)
            vis.update_geometry(line_set)
            vis.poll_events()
            vis.update_renderer()

            # save 3d bbx
            # np.save(frame +'_2dbbx.npy', np.array([x_max, x_min, y_max, y_min,z_max, z_min]))


    DataFile.close()
    vis.destroy_window()

if __name__ == '__main__':
    main()
