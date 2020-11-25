#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: 
# Date       : 25/11/2020: 19:51
# File Name  : model

import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


def break_up_pc(pc):
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

    return xyz, features


class PointNet2HandJointSSG(nn.Module):
    def __init__(self):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.15,
                nsample=30,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.15,
                nsample=30,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512], use_xyz=True
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 22),
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
        xyz, features = break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.fc_layer(features.squeeze(-1))
