#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: load hand point data
# Date       : 17/09/2020: 15:52
# File Name  : handpoints_dataset

import torch.utils.data as data
import os
import os.path
import shutil
import numpy as np
import lmdb
import msgpack_numpy
import tqdm

SAMPLE_NUM = 1024
JOINT_NUM = 21
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class HandPointsDataset(data.Dataset):
    def __init__(self, transforms=None, train=True):
        self._cache = os.path.join(BASE_DIR, "sampled")  # get the image directory
        if not os.path.exists(self._cache):
            os.makedirs(self._cache)
        self.human_points_path = os.path.join(BASE_DIR, "../human_points/")
        self.shadow_points_path = os.path.join(BASE_DIR, "../shadow_points/")
        self.train = train

        for split in ["train", "test"]:
            self.label = np.load(os.path.join(BASE_DIR, split+".npy"))
            with lmdb.open(
                os.path.join(self._cache, split), map_size=1 << 36
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                for i in tqdm.trange(len(self.label)):
                    tag = self.label[i]
                    fname = tag[0]
                    target = tag[3:].astype(np.float32)
                    human_points = np.load(os.path.join(
                        self.human_points_path, fname[:-4] + ".npy")).astype(np.float32)

                    txn.put(
                        str(i).encode(),
                        msgpack_numpy.packb(
                            dict(pc=human_points, lbl=target), use_bin_type=True
                        ),
                    )
            shutil.rmtree(self.data_dir)
        self._lmdb_file = os.path.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None
        self.transforms = transforms

    def __getitem__(self, index):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(index).encode()), raw=False)

        human_points = ele["pc"]

        if self.transforms is not None:
            human_points = self.transforms(human_points)

        return human_points, ele["lbl"]

    def __len__(self):
        return self._len

if __name__== "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
                    [
                                    d_utils.PointcloudToTensor(),
                                    d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
                                    d_utils.PointcloudScale(),
                                    d_utils.PointcloudTranslate(),
                                    d_utils.PointcloudJitter(),
                    ])
    dset = ModelNet40Cls(16, train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
