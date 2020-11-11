import numpy as np
import csv
import os
from mayavi import mlab
from IPython import embed
from pathlib import Path

class Map_Loader(object):
    def __init__(self, base_path="./data/"):
        # load data
        self.base_path = base_path
        # DataFile = open(base_path + "bighand2017_test_annotation.txt", "r")
        DataFile = open(base_path + "Training_Annotation.txt", "r")

        lines = DataFile.read().splitlines()
        self.framelist = [ln.split(' ')[0].replace("\t", "") for ln in lines]
        label_source = [ln.split('\t')[1:] for ln in lines]
        self.label = []
        for ln in label_source:
            ll = ln[0:63]
            self.label.append([float(l.replace(" ", "")) for l in ll])

        self.label = np.array(self.label)
        DataFile.close()

    def map(self, start):
        # the joint order is
        # [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP,
        # TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]
        # for index in range(start, start + batch_size):
        keypoints = self.label[start]
        frame = self.framelist[start]
        keypoints = keypoints.reshape(21, 3)

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
        wrist_y = np.cross(lf_palm, ff_palm)
        wrist_y /= np.linalg.norm(wrist_y)
        wrist_x = np.cross(wrist_y, wrist_z)
        if np.linalg.norm(wrist_x) != 0:
            wrist_x /= np.linalg.norm(wrist_x)
        local_frame = np.vstack([wrist_x, wrist_y, wrist_z])
        local_points = np.dot((keypoints - keypoints[0]), local_frame.T)

        # the pip-mcp direction of each finger
        tf_pip_mcp = local_points[6] - local_points[1]
        ff_pip_mcp = local_points[9] - local_points[2]
        mf_pip_mcp = local_points[12] - local_points[3]
        rf_pip_mcp = local_points[15] - local_points[4]
        lf_pip_mcp = local_points[18] - local_points[5]
        pip_mcp = np.hstack([tf_pip_mcp, ff_pip_mcp, mf_pip_mcp, rf_pip_mcp, lf_pip_mcp])

        # the tip-pip direction of thumb
        tf_tip_pip = local_points[8] - local_points[6]

        # TTIP-PTIP,TPIP-PPIP
        position_goal = np.hstack([
            local_points[8], local_points[11], local_points[14], local_points[17], local_points[20],
            local_points[6], local_points[9], local_points[12], local_points[15], local_points[18]])

        # position_goal = np.hstack([
        #     keypoints[8], keypoints[11], keypoints[14], keypoints[17], keypoints[20],
        #     keypoints[6], keypoints[9], keypoints[12], keypoints[15], keypoints[18]])

        goals = np.hstack([position_goal, pip_mcp, tf_tip_pip])/1000.0
        return frame, keypoints, goals, local_points, keypoints[0]/1000.0, np.hstack([wrist_x, wrist_y, wrist_z])


def show_line(un1, un2, color='g', scale_factor=1):
    # for shadow and human scale_factor=1
    if color == 'b':
        color_f = (0.8, 0, 0.9)
    elif color == 'r':
        color_f = (0.3, 0.2,0.7)
    elif color == 'p':
        color_f = (0.1, 1, 0.8)
    elif color == 'y':
        color_f = (0.5, 1, 1)
    elif color == 'g':
        color_f = (1, 1, 0)
    elif isinstance(color, tuple):
        color_f = color
    else:
        color_f = (1, 1, 1)
    mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)


def show_points(point, color='b', scale_factor=5):
    # for shadow and human scale_factor=5
    if color == 'b':
        color_f = (0, 0, 1)
    elif color == 'r':
        color_f = (1, 0, 0)
    elif color == 'g':
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    if point.size == 3:  # vis for only one point
        mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
    else:  # vis for multiple points
        mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)


def show_hand(points, type='human'):
    show_points(points)
    if type == "human":
        show_line(points[0], points[1], color='r')
        show_line(points[1], points[6], color='r')
        show_line(points[7], points[6], color='r')
        show_line(points[7], points[8], color='r')

        show_line(points[0], points[2], color='y')
        show_line(points[9], points[2], color='y')
        show_line(points[9], points[10], color='y')
        show_line(points[11], points[10], color='y')

        show_line(points[0], points[3], color='g')
        show_line(points[12], points[3], color='g')
        show_line(points[12], points[13], color='g')
        show_line(points[14], points[13], color='g')

        show_line(points[0], points[4], color='b')
        show_line(points[15], points[4], color='b')
        show_line(points[15], points[16], color='b')
        show_line(points[17], points[16], color='b')

        show_line(points[0], points[5], color='p')
        show_line(points[18], points[5], color='p')
        show_line(points[18], points[19], color='p')
        show_line(points[20], points[19], color='p')
    elif type == "shadow":
        show_line(points[0], points[1], color='r')
        show_line(points[1], points[6], color='r')
        show_line(points[7], points[6], color='r')
        show_line(points[7], points[8], color='r')

        show_line(points[0], points[2], color='y')
        show_line(points[9], points[2], color='y')
        show_line(points[9], points[10], color='y')
        show_line(points[11], points[10], color='y')

        show_line(points[0], points[3], color='g')
        show_line(points[12], points[3], color='g')
        show_line(points[12], points[13], color='g')
        show_line(points[14], points[13], color='g')

        show_line(points[0], points[4], color='b')
        show_line(points[15], points[4], color='b')
        show_line(points[15], points[16], color='b')
        show_line(points[17], points[16], color='b')

        show_line(points[0], points[5], color='p')
        show_line(points[18], points[5], color='p')
        show_line(points[18], points[19], color='p')
        show_line(points[20], points[19], color='p')
    else:
        show_line(points[0], points[11], color='r')
        show_line(points[11], points[6], color='r')
        show_line(points[6], points[1], color='r')

        show_line(points[0], points[12], color='y')
        show_line(points[12], points[7], color='y')
        show_line(points[7], points[2], color='y')

        show_line(points[0], points[13], color='g')
        show_line(points[13], points[8], color='g')
        show_line(points[8], points[3], color='g')

        show_line(points[0], points[14], color='b')
        show_line(points[14], points[9], color='b')
        show_line(points[9], points[4], color='b')

        show_line(points[0], points[15], color='p')
        show_line(points[15], points[10], color='p')
        show_line(points[10], points[5], color='p')

    tf_palm = points[1] - points[0]
    ff_palm = points[2] - points[0]
    mf_palm = points[3] - points[0]
    rf_palm = points[4] - points[0]
    lf_palm = points[5] - points[0]
    # palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])
    palm = np.array([ff_palm, mf_palm, rf_palm, lf_palm])

    # local wrist frame build
    wrist_z = np.mean(palm, axis=0)
    wrist_z /= np.linalg.norm(wrist_z)
    wrist_y = np.cross(rf_palm, ff_palm)
    wrist_y /= np.linalg.norm(wrist_y)
    wrist_x = np.cross(wrist_y, wrist_z)
    if np.linalg.norm(wrist_x) != 0:
        wrist_x /= np.linalg.norm(wrist_x)

    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_x[0], wrist_x[1], wrist_x[2],
                  scale_factor=50, line_width=0.5, color=(1, 0, 0), mode='arrow')
    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_y[0], wrist_y[1], wrist_y[2],
                  scale_factor=50, line_width=0.5, color=(0, 1, 0), mode='arrow')
    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_z[0], wrist_z[1], wrist_z[2],
                  scale_factor=50, line_width=0.5, color=(0, 0, 1), mode='arrow')


def cartesian_pos_show(base_path):
    DataFile = open(base_path + "cartesian_world_pos_file.csv", "r")
    lines = DataFile.read().splitlines()
    for ln in lines:
        label_source = ln.split(',')[1:46]
        label = np.array([float(ll) for ll in label_source])
        keypoints = np.vstack((np.array([0,0,0]), label.reshape(15, 3)))
        # mlab.clf
        mlab.figure(bgcolor=(1,1,1),size=(1280,960))
        show_hand(keypoints, 'shadow_real')
        mlab.show()
    DataFile.close()


if __name__ == '__main__':
    batch_size = 1
    base_path = "../data/"
    # cartesian_pos_show(base_path)
    map_loader = Map_Loader(base_path)
    filename = base_path + "shadow_hand_mapping_pose_file.csv"
    filename = "/homeL/shuang/ros_workspace/tele_ws/src/dataset/" + "shadow_hand_mapping_pose_file.csv"

    # The method unlink() removes (deletes) the file path.
    Path(filename).unlink(missing_ok=True)
    csvSum = open(filename, "w")
    writer = csv.writer(csvSum)

    for i in range(len(map_loader.framelist)):
        frame, keypoints, goals, local_points, wrist_pos, local_frame = map_loader.map(i)
        # save key
        # embed()
        result = np.hstack([frame, goals, wrist_pos, local_frame]).tolist()
        writer.writerow(result)

        # mlab.figure(bgcolor=(1, 1, 1), size=(1280, 960))
        # show_hand(local_points, 'human')
        # show_hand(keypoints, 'shadow')
        # mlab.savefig(filename="../data/tams_handshape/" + frame)
        # mlab.close()
        # mlab.show()
    csvSum.close()
