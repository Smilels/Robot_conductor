#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: load hand point data
# Date       : 17/09/2020: 15:52
# File Name  : dataset

import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import data_utils as d_utils
from torchvision import transforms

import scipy.io as sio
import pdb

SAMPLE_NUM = 1024
JOINT_NUM = 21


class HandPointDataset(data.Dataset):
    def __init__(self, root_path, opt, train=True):
        self.dir_AB = os.path.join(opt.dataroot, 'joints_img')  # get the image directory

        self.human_points_path = os.path.join(opt.dataroot, '../human_points/')
        self.shadow_points_path = os.path.join(opt.dataroot, '../shadow_points/')
        if opt.phase == "train":
           self.label = np.load(os.path.join(opt.dataroot, 'train.npy'))
        elif opt.phase == "test":
            self.label = np.load(os.path.join(opt.dataroot, 'test.npy'))

        self.transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRotate(axis=np.array([0, 0, 1])),
                d_utils.PointcloudScale(lo=0.85, hi=1.15),
                d_utils.PointcloudTranslate(translate_range=0.015),
                # d_utils.PointcloudJitter(),
            ]
        )

    def __getitem__(self, index):
        tag = self.label[index]
        fname = tag[0]
        target = tag[3:].astype(np.float32)
        human_points = np.load(os.path.join(self.human_points_path, fname[:-4]+'.npy')).astype(np.float32)
        shadow_points = np.load(os.path.join(self.shadow_points_path, fname[:-4]+'.npy')).astype(np.float32)

        if self.transforms is not None:
            human_points = self.transforms(human_points)

        return human_points, shadow_points, target

    def __len__(self):
        return len(self.label)