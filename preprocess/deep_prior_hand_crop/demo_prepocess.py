#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : bighand_preprocess.py
# Purpose : Provides importer classes for importing data
# Last Modified:01/07/20
# Created By :Shuang Li

'''
this method cannot directly apply on bighand dataset because there are many images which the hand is not the closest object to the camera.
but is can be used on images from human96 which have roughly cropped by the groudtruth
'''
import scipy.io
import numpy as np
from PIL import Image
import os
import progressbar as pb
import struct
from basetypes import DepthFrame, NamedImgSequence
from handdetector import HandDetector
from transformations import transformPoints2D
from IPython import embed
from sensor_msgs.msg import JointState
import rospy
import moveit_commander
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


basepath = '/homeL/demo/ros_workspace/pr2_shadow_ws/src/TeachNet_Teleoperation/ros/src/shadow_teleop/data/Human_label'
# the annotation txt file name
txtname = 'text_annotation'
# the depth image folder
image_path = '/test_data'


class DepthImporter(object):
    def __init__(self):
        """
        Initialize object
        x: focal length in x direction
        fy: focal length in y direction
        ux: principal point in x direction
        uy: principal point in y direction
        """

        self.fx = 475.065948
        self.fy = 475.065857
        self.ux = 315.944855
        self.uy = 245.287079
        self.depth_map_size = (640, 3480)
        self.refineNet = None
        self.crop_joint_idx = 0
        self.numJoints = 21

        self.bridge = CvBridge()

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(), frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:4, 0], frame.gtcrop[0:4, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[4:7, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[4:7, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[7:10, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[7:10, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[10:13, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[10:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:16, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:16, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

    def load_file(self):
        while True:
                img_data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
                rospy.loginfo("Got an image ^_^")
                try:
                    img = self.bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
                except CvBridgeError as e:
                    rospy.logerr(e)

                # adjust a proper cube size
                config = {'cube': (200, 200, 200)}
                # loadDepthMap(dptFileName)
                # img.show()
                dpt = np.asarray(img, np.float32)
                dpt[np.where(dpt > 450)] = 0

                # Detect hand
                hd = HandDetector(dpt, self.fx, self.fy, refineNet=None, importer=self)
                if not hd.checkImage(1):
                    print("Skipping image {}, no content")
                    continue
                try:
                    # we can use one value of uv of 96*96 as com
                    dpt, M, com = hd.cropArea3D(com=None, size=config['cube'], dsize=(96, 96), docom=True)
                except UserWarning:
                    print("Skipping image {}, no hand detected")
                    continue

                com3D = self.jointImgTo3D(com)

                # print("{}".format(gt3Dorig))
                # self.showAnnotatedDepth(DepthFrame(dpt, gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName, frame,
                #                                    'right', ''))

                img = dpt.astype(np.float32)
                img = img / 255. * 2. - 1

                n = cv2.resize(img, (0, 0), fx=2, fy=2)
                n = cv2.normalize(n, n, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("segmented human hand", n)
                cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('human_teleop_shadow')
    importer = DepthImporter()
    while not rospy.is_shutdown():
        importer.load_file()
