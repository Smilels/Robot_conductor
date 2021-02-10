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
import dataloader.data_utils as d_utils
from dataloader.base_dataset import BaseDataset
from IPython import embed
from utils import depth2pc, pca_rotation, down_sample, get_normal, normalization_unit, FPS_idx, normalization_mean
import dataloader.data_utils as d_utils


DOWN_SAMPLE_NUM = 2048
FPS_SAMPLE_NUM = 1024


class KeypointsDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self._cache = os.path.join(opt.dataroot, "sampled_20k")  # get the image directory
        self.human_points_path = os.path.join(opt.dataroot, "human_points/")
        self.shadow_points_path = os.path.join(opt.dataroot, "points_shadow/")

        if opt.phase == "train":
            self.transforms = transforms.Compose(
                [
                    d_utils.PointcloudToTensor(),
                    d_utils.PointcloudScale(),
                    d_utils.PointcloudRotate(axis=np.array([0.0, 0.0, 1.0])),
                    d_utils.PointcloudTranslate(translate_range=0.015)
                ]
            )
        elif opt.phase == "test":
            self.transforms = None

        if not os.path.exists(self._cache):
            os.makedirs(self._cache)

            for split in ["train_20k", "test_20k"]:
                self.label = np.load(os.path.join(opt.dataroot, split+".npy"))
                with lmdb.open(
                    os.path.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.label)):
                        tag = self.label[i]
                        fname = tag[0].decode("utf-8")
                        human_points, target = np.load(os.path.join(
                            self.human_points_path, fname[:-4] + ".npy"), allow_pickle=True).astype(np.float32)
                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(frame=fname, pc=human_points, lbl=target), use_bin_type=True
                            ),
                        )
        self._lmdb_file = os.path.join(self._cache, "train_20k" if self.transforms is not None else "test_20k")
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

        points = np.squeeze(np.array(ele["pc"]))
        target = np.array(ele["lbl"])

        # downsampling
        points_sampled, rand_ind = down_sample(points, DOWN_SAMPLE_NUM)

        # FPS Sampling
        points_fps_sampled, farthest_pts_idx = FPS_idx(points_sampled, self.num_points)

        # normalization
        points_normalized = points_fps_sampled - points_fps_sampled.mean(axis=0)
        target = target - points_fps_sampled.mean(axis=0)

        # shuffle
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        human_points = points_normalized[pt_idxs, :]

        if self.transforms is not None:
            human_points = self.transforms(human_points)

        return ele['frame'], human_points, target, points_fps_sampled.mean(axis=0)

    def __len__(self):
        return self._len


if __name__ == "__main__":
    dset = KeypointsDataset(train=True)
    print(dset[1000][0])
    print(dset[1000][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
