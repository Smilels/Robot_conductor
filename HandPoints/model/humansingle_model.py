#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: Handjoint model from PoineNet2
# Date       : 17/09/2020: 15:50
# File Name  : humansingle_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import networks
from .base_model import BaseModel
from IPython import embed


class HumansingleModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['J_L2']
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.joint_angles = None
        self.netG = networks.define_G(opt.netG, opt.norm, opt.init_type,
                                      opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                               1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                               1.571, 1.047, 1.222, 0.209, 0.524, 1.571]).to(self.device)
        self.joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                               -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0]).to(self.device)

    def set_input(self, input):
        pc, label = input
        self.pc = pc.to(self.device)
        self.label = label.to(self.device)

    def forward(self):
        self.joint_angles =  self.netG(self.pc)

    def optimize_parameters(self):
        self.forward()
        self.joint_angles = self.joint_angles * (self.joint_upper_range - self.joint_lower_range) + self.joint_lower_range
        self.loss_J_L2 = F.mse_loss(self.joint_angles, self.label)
        self.loss_J_L2.backward()
        self.optimizer_G.step()
