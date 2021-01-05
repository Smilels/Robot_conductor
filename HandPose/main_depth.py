#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description:
# Date       : 22/11/2020: 10:55
# File Name  : main.py

import time
import random
import copy
from collections import OrderedDict

from config.train_options import TrainOptions
from model import create_model
from model.depth_model import NaiveTeleModel
from utils.visualizer import Visualizer
from dataloader import create_dataset

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import os
from IPython import embed

args = TrainOptions().parse()  # get training args
logger = SummaryWriter(os.path.join('./log/', args.display_env))
np.random.seed(int(time.time()))

total_iters = 0  # the total number of training iterations
input_size = args.load_size 
embedding_size = 128

train_loader = create_dataset(args)
print('The number of training images = %d' % len(train_loader))
args_test = copy.deepcopy(args)
args_test.phase = 'test'
args_test.isTrain = False
test_loader = create_dataset(args_test)
print("test data number is: ", len(test_loader.dataset))

model = NaiveTeleModel(input_size=input_size, embedding_size=embedding_size)

if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])
   model.to(args.gpu_ids[0])
   model = nn.DataParallel(model, device_ids=args.gpu_ids).cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)


def train(model, loader, epoch):
    model.train()
    torch.set_grad_enabled(True)
    train_error_rot = 0
    train_error_trans = 0
    for batch_idx, (fname, human, label_rot, label_trans) in enumerate(loader):
        human, label_rot, label_trans = human.cuda(), label_rot.cuda(), label_trans.cuda()
        optimizer.zero_grad()
        pred_rot, pred_trans = model(human)
        # pred_trans[:, 2] = pred_trans[:, 2] 
        loss_rot = F.mse_loss(pred_rot, label_rot)
        loss_trans = F.mse_loss(pred_trans, label_trans)
        train_loss = loss_rot + loss_trans
        train_loss.backward()
        optimizer.step()

        # compute average angle error
        train_error_rot += F.l1_loss(pred_rot, label_rot, reduction="sum")
        train_error_trans += F.l1_loss(pred_trans, label_trans, reduction="sum")

        if batch_idx % args.print_freq == 0:
            if isinstance(train_loss, float):
                train_loss = torch.zeros(1)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Loss_rot: {:.6f} Loss_trans: {:.6f} \t{}'.format(
                epoch, batch_idx * args.batch_size, len(loader.dataset),
                100. * batch_idx * args.batch_size / len(loader.dataset),
                train_loss.item(), loss_rot.item(), loss_trans.item(),
                args.display_env))

            logger.add_scalar('train_rot_loss', train_loss.item(),
                              batch_idx + epoch * len(loader))
            logger.add_scalar('train_rot_loss', loss_rot.item(),
                              batch_idx + epoch * len(loader))
            logger.add_scalar('train_trans_loss', loss_trans.item(),
                              batch_idx + epoch * len(loader))

    train_error_rot /= len(loader.dataset)
    train_error_trans /= len(loader.dataset)
    
    scheduler.step()
    return train_error_rot, train_error_trans


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_error_rot = 0
    test_error_trans = 0
    test_loss_rot = 0
    test_loss_trans = 0
    for batch_idx, (fname, human, label_rot, label_trans) in enumerate(loader):
        human, label_rot, label_trans = human.cuda(), label_rot.cuda(), label_trans.cuda()

        pred_rot, pred_trans = model(human)
        # pred_trans[:, :2] = pred_trans[:, :2] * float(input_size) 
        test_loss_rot += F.mse_loss(pred_rot, label_rot)
        test_loss_trans += F.mse_loss(pred_trans, label_trans)

        # compute average angle error
        test_error_rot += F.l1_loss(pred_rot, label_rot, reduction="sum")
        test_error_trans += F.l1_loss(pred_trans, label_trans, reduction="sum")

    test_loss_rot /= len(loader.dataset)
    test_loss_trans /= len(loader.dataset)
    test_loss = test_loss_rot + test_loss_trans
    test_error_rot /= len(loader.dataset)
    test_error_trans /= len(loader.dataset)

    return test_loss_rot, test_loss_trans, test_loss, test_error_rot, test_error_trans


def main():
    if args.phase== 'train':
        for epoch in range(args.epoch_count, 400 + 1):
            train_error_rot, train_error_trans = train(model, train_loader, epoch)
            print('Train done, rot_error={}, trans_error={}'.format(
                train_error_rot, train_error_trans))
            test_loss_rot, test_loss_trans, test_loss, test_error_rot, test_error_trans = test(model, test_loader)
            print(
                'Test done, test_loss_rot ={}, test_loss_trans={}, '
                'test_loss={}, test_error_rot={}, test_error_trans={}'.format(test_loss_rot,
                                                                              test_loss_trans, test_loss,
                                                                              test_error_rot, test_error_trans))
            logger.add_scalar('train_error_rot', train_error_rot, epoch)
            logger.add_scalar('train_error_trans', train_error_trans, epoch)

            logger.add_scalar('test_error_rot', test_error_rot, epoch)
            logger.add_scalar('test_error_trans', test_error_trans, epoch)
            logger.add_scalar('test_loss', test_loss, epoch)
            logger.add_scalar('test_loss_rot', test_loss_rot, epoch)
            logger.add_scalar('test_loss_trans', test_loss_trans, epoch)

            if epoch % args.save_epoch_freq == 0:
                path = os.path.join(args.checkpoints_dir, args.display_env + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        test_loss_rot, test_loss_trans, test_loss, test_error_rot, test_error_trans = test(model, test_loader)
        print(
            'Test done, test_loss_rot ={}, test_loss_trans={}, '
            'test_loss={}, test_error_rot={}, test_error_trans={}'.format(test_loss_rot,
                                                                          test_loss_trans, test_loss,
                                                                          test_error_rot, test_error_trans))


if __name__ == "__main__":
    main()
