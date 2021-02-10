#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description:
# Date       : 22/11/2020: 10:55
# File Name  : main.py

import time
import copy

from config.train_options import TrainOptions
from model.pct import Pct
from dataloader import create_dataset
from IPython import embed

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import os

args = TrainOptions().parse()  # get training argsions
logger = SummaryWriter(os.path.join('./log/', args.display_env))
np.random.seed(int(time.time()))

total_iters = 0  # the total number of training iterations
thresh_acc = [4, 5, 6, 7]
keypoints_aize = 63

train_loader = create_dataset(args)
print('The number of training images = %d' % len(train_loader))
args_test = copy.deepcopy(args)
args_test.phase = 'test'
args_test.isTrain = False
test_loader = create_dataset(args_test)
print("test data number is: ", len(test_loader.dataset))

model = Pct(output_channels=keypoints_aize)

if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])
   model.to(args.gpu_ids[0])
   model = nn.DataParallel(model, device_ids=args.gpu_ids).cuda()

#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
scheduler = CosineAnnealingLR(optimizer, 300, eta_min=args.lr)


def train(model, loader, epoch):
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    train_error_human = 0
    correct_human = [0, 0, 0, 0]
    for batch_idx, (fname, human, target, mean) in enumerate(loader):
        human, target = human.cuda(), target.cuda()
        human = human.transpose(2, 1)
        # shadow part
        optimizer.zero_grad()
        keypoints_human = model(human)
        loss_human = F.mse_loss(keypoints_human, target)

        loss_human.backward()
        optimizer.step()

        # compute acc
        res_human = [np.sum(np.sum(abs(keypoints_human.cpu().data.numpy() -
                                       target.cpu().data.numpy()) < thresh,
                                       axis=-1) == keypoints_aize) for thresh in thresh_acc]
        correct_human = [c + r for c, r in zip(correct_human, res_human)]

        # compute average angle error
        train_error_human += F.l1_loss(keypoints_human, target, size_average=False) / keypoints_aize

        if batch_idx % args.print_freq == 0:
            if isinstance(loss_human, float):
                loss_human = torch.zeros(1)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * args.batch_size, len(loader.dataset),
                       100. * batch_idx * args.batch_size / len(loader.dataset), loss_human.item(), args.display_env))

            logger.add_scalar('train_loss', loss_human.item(),
                              batch_idx + epoch * len(loader))

    train_error_human /= len(loader.dataset)
    acc_human = [float(c) / float(len(loader.dataset)) for c in correct_human]

    return acc_human, train_error_human


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss_human = 0
    correct_human = [0, 0, 0, 0]
    test_error_human = 0
    for fname, human, target, mean in loader:
        human, target = human.cuda(), target.cuda()

        human = human.transpose(2, 1)
        # human part
        keypoints_human = model(human)
        keypoints_human += mean
        test_loss_human += F.mse_loss(keypoints_human, target, size_average=False).item()

        # compute acc
        res_human = [np.sum(np.sum(abs(keypoints_human.cpu().data.numpy()
                                       - target.cpu().data.numpy()) < thresh,
                                       axis=-1) == keypoints_aize) for thresh in thresh_acc]
        correct_human = [c + r for c, r in zip(correct_human, res_human)]

        # compute average angle error
        test_error_human += F.l1_loss(keypoints_human, target, size_average=False) / keypoints_aize

    test_loss_human /= len(loader.dataset)
    test_error_human /= len(loader.dataset)

    acc_human = [float(c) / float(len(loader.dataset)) for c in correct_human]
    # f = open('input.csv', 'w')
    # for batch in res:
    #     for name, joint in zip(batch[0], batch[1]):
    #         buf = [name, '0.0', '0.0'] + [str(i) for i in joint.cpu().data.numpy()]
    #         f.write(','.join(buf) + '\n')

    return acc_human, test_error_human, test_loss_human


def main():
    if args.phase== 'train':
        for epoch in range(args.epoch_count, args.n_epochs + 1):
            acc_train_human, train_error_human = train(model, train_loader, epoch)
            print('Train done, acc_human={}, train_error_human={}'.format(
                acc_train_human, train_error_human))
            acc_test_human, test_error_human, loss_test_human = test(model, test_loader)
            print(
                'Test done, acc_human={}, error_human ={}, loss_test_human={}'.
                    format(acc_test_human, test_error_human, loss_test_human))
            logger.add_scalar('train_acc_human4', acc_train_human[0], epoch)
            logger.add_scalar('train_acc_human5', acc_train_human[1], epoch)
            logger.add_scalar('train_acc_human6', acc_train_human[2], epoch)
            logger.add_scalar('train_acc_human7', acc_train_human[3], epoch)

            logger.add_scalar('test_acc_human4', acc_test_human[0], epoch)
            logger.add_scalar('test_acc_human5', acc_test_human[1], epoch)
            logger.add_scalar('test_acc_human6', acc_test_human[2], epoch)
            logger.add_scalar('test_acc_human7', acc_test_human[3], epoch)

            logger.add_scalar('test_error_human', test_error_human, epoch)
            logger.add_scalar('test_loss_human', loss_test_human, epoch)

            if epoch % args.save_epoch_freq== 0:
                path = os.path.join(args.checkpoints_dir, args.display_env + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc_test_human, test_error_human, loss_test_human = test(model, test_loader)
        print(
            'Test done, acc_human={}, error_human ={},loss_test_human={}'.
                format(acc_test_human, test_error_human, loss_test_human))


if __name__ == "__main__":
    main()
