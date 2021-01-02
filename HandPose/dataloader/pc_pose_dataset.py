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
from dataloader.base_dataset import BaseDataset
from IPython import embed

SAMPLE_NUM = 1024
JOINT_NUM = 21


class JointDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self._cache = os.path.join(opt.dataroot, "sampled")  # get the image directory
        self.human_points_path = os.path.join(opt.dataroot, "points_human/")
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

            for split in ["train", "test"]:
                self.label = np.load(os.path.join(opt.dataroot, split+".npy"))
                with lmdb.open(
                    os.path.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.label)):
                        tag = self.label[i]
                        fname = tag[0].decode("utf-8")
                        # todo
                        rot = tag[3:].astype(np.float32)
                        trans = tag[3:].astype(np.float32)
                        human_points = np.load(os.path.join(
                            self.human_points_path, fname[:-4] + ".npy")).astype(np.float32)
                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(frame=fname, pc=human_points, rot=rot, trans=trans), use_bin_type=True
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

        points = np.array(ele["pc"])
        points = self.preprocess(points, )
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        human_points = points[pt_idxs, :]

        rot = np.array(ele["rot"])
        trans = np.array(ele["trans"])

        if self.transforms is not None:
            if self.opt.mormal:
                human_points, rot, trans, normals = self.transforms(human_points, rot, trans)
            else:
                human_points, rot, trans = self.transforms(human_points, rot, trans)

        return ele['frame'], human_points, rot, trans

    def preprocess(self, points, rot, trans):
        # PCA rotation
        if self.opt.do_pca:
            points_pca, pc_transfrom = pca_rotation(points)
            rot = np.dot(rot, pc_transfrom.T)
            trans = np.dot(trans, pc_transfrom.T)
        else:
            points_pca = points

        # downsampling
        points_pca_sampled, rand_ind = down_sample(points_pca, self.opt.downsample_num)

        # FPS Sampling
        points_pca_fps_sampled, farthest_pts_idx = FPS_idx(points_pca_sampled, self.opt.fps_sample_num)

        # normalize point cloud
        if self.opt.normalization == "mean":
            points_normalized, points_mean = normalization_mean(points_pca_fps_sampled)

        elif self.opt.normalization == "unit":
            points_normalized, max_bb3d_len, offset = normalization_unit(points_pca_fps_sampled)
            trans = (trans - offset)/max_bb3d_len

        # compute surface normal
        if self.opt.mormal:
            normals_pca = get_normal(points_pca)
            normals_pca_sampled = normals_pca[rand_ind]
            normals_pca_fps_sampled = normals_pca_sampled[farthest_pts_idx]
            return points_normalized, rot, trans, normals_pca_fps_sampled
        else:
            return points_normalized, rot, trans

    def __len__(self):
        return self._len


if __name__ == "__main__":
    dset = JointDataset(train=True)
    print(dset[1000][0])
    print(dset[1000][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
