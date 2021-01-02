#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: load hand point data
# Date       : 17/09/2020: 15:52
# File Name  : handpoints_dataset

import torch
import torch.utils.data as data
from torchvision import transforms
import os
import os.path
import numpy as np
import lmdb
import msgpack_numpy
import tqdm
import dataloader.data_utils as d_utils, pca_rotation, down_sample, get_normal, normalization_unit, FPS_idx, normalization_mean
from dataloader.base_dataset import BaseDataset, cv2_transform
from IPython import embed
import cv2
from scipy.spatial.transform import Rotation as R


SAMPLE_NUM = 1024
JOINT_NUM = 21


class JointDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self._cache = os.path.join(opt.dataroot, "sampled")  # get the image directory
        self.data_path = os.path.join(opt.dataroot, "human_crop_seg_img/")

        if not os.path.exists(self._cache):
            os.makedirs(self._cache)
            for split in ["train", "test"]:
                self.label = np.load(os.path.join(opt.dataroot, split+".npy"))
                with lmdb.open(
                    os.path.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.label)):
                        tag = self.label[i]
                        fname = tag[0].decode("utf-8")
                        pose =np.load(os.path.join(
                            self.pose_path, fname[:-4] + ".png")).astype(np.float32)
                        human = cv2.imread(os.path.join(
                            self.data_path, fname[:-4] + ".png"), cv2.IMREAD_ANYDEPTH)
                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(frame=fname, human=human, pose=pose), use_bin_type=True
                            ),
                        )
        self._lmdb_file = os.path.join(self._cache, "train" if self.transforms is not None else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None
        self.num_points = opt.num_points

    def __getitem__(self, index):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(index).encode()), raw=False)

        human = np.array(ele["human"])
        human = cv2_transform(self.opt, human, is_a=True)
        pose = np.array(ele["pose"])
        rot = pose[:3, :3]
        r = R.from_matrix(rot)
        axisangle = r.as_rotvec()
        trans = pose[:, 3:]
        if self.opt.phase == "train":
            self.preprocess(human, trans)
        return ele['frame'], human, axisangle, trans

    def preprocess(self, img, trans):
        img = img.astype(np.float32)
        if 'resize' in self.opt.preprocess:
            img = cv2.resize(img, (self.opt.load_size, self.opt.load_size))
        if self.opt.phase == "train":
            if 'rotate' in self.opt.preprocess:
                angle = np.random.randint(-180, 180)
                M = cv2.getRotationMatrix2D(((self.opt.load_size - 1) / 2.0, (self.opt.load_size - 1) / 2.0), angle, 1)
                img = cv2.warpAffine(img, M, (self.opt.load_size, self.opt.load_size))

                trans_out = np.ones((1, 3))
                trans_out[:, :2] = trans[:, :2].copy()
                trans_out = np.matmul(M, trans_out.transpose())
                trans_out = trans_out.transpose()
                trans_out[:, 2] = trans[:, 2]

            if 'jitter' in self.opt.preprocess:
                min_img = np.min(img[img != 255.])
                max_img = np.max(img[img != 255.])
                delta = np.random.rand() * (255. - max_img + min_img) - min_img
                img[img != 255.] += delta
                img = img.clip(max=255., min=0.)

        return img, trans_out

    def __len__(self):
        return self._len


if __name__ == "__main__":
    dset = JointDataset(train=True)
    print(dset[1000][0])
    print(dset[1000][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
