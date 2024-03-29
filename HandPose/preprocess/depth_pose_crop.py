#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np
import multiprocessing as mp
from IPython import embed
import cv2
import sys
import os
from utils import pc2uvd, crop_depth_img, cal_hand_pose, uvd2pc, depth2pc
from seg import seg_hand_depth
import open3d as o3d


focalLengthX = 475.065948
focalLengthY = 475.065857
centerX = 315.944855
centerY = 245.287079
mat = np.array([[focalLengthX, 0, centerX], [0, focalLengthY, centerY], [0, 0, 1]])
output_size = 96
show_2d = 0
show_3d = 0

if sys.argv[1] == "tams108":
    base_path = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/"
    img_path = os.path.join(base_path, "Human_label/human_full_test/")
    pose_path = os.path.join(base_path, "human_pose/")
    crop_img_path = os.path.join(base_path, "human_crop_seg_img/")
    gt_file = "Human_label/text_annotation.txt"

elif sys. argv[1] == "server":
    base_path = "/data/sli/Bighand2017/"
    img_path = base_path + "images/"
    pose_path = os.path.join(base_path, "depth_pose/human_pose/")
    crop_info_path = os.path.join(base_path, "depth_pose/crop_info/")
    crop_img_path = base_path + "depth_pose/human_crop_seg_img/"
    gt_file = "groundtruth/Training_Annotation.txt"


def get_img_pose(line):
    frame = line.split(' ')[0].replace("\t", "")
    # print(frame)
    img = cv2.imread(img_path + str(frame), cv2.IMREAD_ANYDEPTH)
    if img is None or img.size == 0:
        print(frame, " no Image find")
        return
    h, w = img.shape

    label_source = line.split('\t')[1:]
    label = [float(l.replace(" ", "")) for l in label_source[0:63]]
    keypoints = np.array(label).reshape(21, 3).astype(np.float32)
    rot = cal_hand_pose(keypoints).reshape(1, -1)

    uvd = pc2uvd(keypoints, mat)
    padding = 10
    crop_img, trans_uvd, crop_data1 = crop_depth_img(img, uvd, padding)
    if crop_img is None or crop_img.size == 0:
        print(frame, "wrong label or empty image")
        return
    if np.max(crop_img) == np.min(crop_img):
          print(frame, ' max and min are same, bad!')

    try:
        output, trans_uvd, crop_data2 = seg_hand_depth(crop_img, 500, 1000, 10, output_size, 4, 4, 250, True, 300, label=trans_uvd)
    except:
        print(frame, ' seg failed')
        return
    if trans_uvd[0, 0] > output_size or trans_uvd[0, 0] > output_size:
        print(frame, ' uv more than outputsize, wrong label')
        return
    # trans = uvd2pc(trans_uvd, centerX, centerY, focalLengthX, focalLengthY)
    hand_pose = np.hstack([rot, trans_uvd.reshape(1, 3)])

    if show_2d:
        cv2.circle(output, (int(trans_uvd[0, 0]), int(trans_uvd[0, 1])), 5, (0, 255, 0), -1)
        n = cv2.normalize(output, output, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow(frame, n)
        key = cv2.waitKey()

        if key == ord('q'):
            exit()
        if key == ord('a'):
            cv2.destroyAllWindows()
            return

    if show_3d:
        points_raw = depth2pc(output, centerY * output_size/float(h), centerY * output_size/float(h),
                              focalLengthX * output_size/float(h), focalLengthY * output_size/float(w))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_raw)
        pcd.paint_uniform_color([0.1, 0.1, 0.7])
        pcd_wrist = o3d.geometry.PointCloud()
        # trans = uvd2pc(trans_uvd, centerY * output_size/float(h), centerY * output_size/float(h),
        #                focalLengthX * output_size/float(h), focalLengthY * output_size/float(w))
        pcd_wrist.points = o3d.utility.Vector3dVector(trans_uvd.reshape(1, 3))
        pcd_wrist.paint_uniform_color([0.9, 0.1, 0.1])
        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, pcd_wrist, world_frame_vis], point_show_normal=False)

    if not os.path.exists(pose_path):
        os.makedirs(pose_path)
    if not os.path.exists(crop_info_path):
        os.makedirs(crop_info_path)
    np.save(os.path.join(pose_path, frame[:-4] + '.npy'), hand_pose)
    np.save(os.path.join(crop_info_path, frame[:-4] + '_crop1.npy'), crop_data1)
    np.save(os.path.join(crop_info_path, frame[:-4] + '_crop2.npy'), crop_data2)
    if not os.path.exists(crop_img_path):
        os.makedirs(crop_img_path)
    cv2.imwrite(os.path.join(crop_img_path, frame), output)


def main():
    datafile = open(base_path + gt_file, "r")
    lines = datafile.read().splitlines()
    lines.sort()
    
    if sys.argv[1] == "tams108":
        for line in lines:
            get_img_pose(line)
    elif sys.argv[1] == "server":
        cores = mp.cpu_count()
        pool = mp.Pool(processes=cores)
        pool.map(get_img_pose, lines)
        #for line in lines:
        #    get_img_pose(line)

    datafile.close()


if __name__ == '__main__':
    main()
