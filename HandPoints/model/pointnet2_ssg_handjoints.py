#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: Handjoint model from PoineNet2
# Date       : 17/09/2020: 15:50
# File Name  : pointnet2_ssg_handpoints

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader
from torchvision import transforms

import HandPoints.dataset.data_utils as d_utils
from HandPoints.dataset import HandPointsDataset
from HandPoints.model.pointnet2_ssg_cls import PointNet2ClassificationSSG
from IPython import embed


class PointNet2HandJointSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.15,
                nsample=64,
                mlp=[0, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.15,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.hparams["model.use_xyz"]
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 22),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.fc_layer(features.squeeze(-1))

    def training_step(self, batch, batch_idx):
        pc, labels = batch
        self.joint_upper_range = self.joint_upper_range.to(pc.get_device())
        self.joint_lower_range = self.joint_lower_range.to(pc.get_device())

        joints = self.forward(pc)
        joints = joints * (self.joint_upper_range - self.joint_lower_range) + self.joint_lower_range
        loss = F.mse_loss(joints, labels)
        with torch.no_grad():
            acc = (torch.sum(torch.abs(joints - labels) < 0.1, dim=-1) == 22).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, labels = batch
        joints = self.forward(pc)
        self.joint_upper_range = self.joint_upper_range.to(pc.get_device())
        self.joint_lower_range = self.joint_lower_range.to(pc.get_device())
        joints = joints * (self.joint_upper_range - self.joint_lower_range) + self.joint_lower_range
        loss = F.mse_loss(joints, labels)
        acc = (torch.sum(torch.abs(joints - labels) < 0.1, dim=-1) == 22).float().mean()
        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                # d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                # d_utils.PointcloudJitter(),
                # d_utils.PointcloudRandomInputDropout(),
            ]
        )
        self.train_dset = HandPointsDataset(transforms=train_transforms, train=True)
        self.val_dset = HandPointsDataset(transforms=None, train=False)
