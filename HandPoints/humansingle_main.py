#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: 
# Date       : 22/11/2020: 10:55
# File Name  : main.py

import os
import time
import copy
import torch.utils.data
import numpy as np
from tensorboardX import SummaryWriter
from collections import OrderedDict

from config.train_options import TrainOptions
from model import create_model
from utils.visualizer import Visualizer
from dataloader import create_dataset

if __name__ == '__main__':
    args = TrainOptions().parse()  # get training argsions

    dataset = create_dataset(args)  # create a dataset given args.dataset_mode and other argsions
    print('The number of training images = %d' % len(dataset))
    args_test = copy.deepcopy(args)
    args_test.phase = 'test'
    args_test.isTrain = False
    dataset_test = create_dataset(args_test)  # create a dataset given args.dataset_mode and other argsions
    print("test data number is: ", len(dataset_test.dataset))

    model = create_model(args)  # create a model given args.model and other argsions
    model.setup(args)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(args)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    thresh_acc = [0.2, 0.25, 0.3]
    acc_train = OrderedDict()
    acc_test = OrderedDict()
    for epoch in range(args.epoch_count,
                       args.n_epochs + args.n_epochs_decay + 1):
        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        correct_shadow = [0, 0, 0]
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        model.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % args.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += args.batch_size
            epoch_iter += args.batch_size
            model.set_input(data)
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            joint_acc_error = model.get_current_error()
            # compute acc
            res_shadow = [np.sum(np.sum(abs(joint_acc_error.cpu().data.numpy()) < thresh,
                                        axis=-1) == 22) for thresh in thresh_acc]
            correct_shadow = [c + r for c, r in zip(correct_shadow, res_shadow)]

            if total_iters % args.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / args.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if args.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)

            if total_iters % args.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if args.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % args.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        acc_shadow = [float(c) / float(len(dataset.dataset)) for c in correct_shadow]
        acc_train['0.2'] = acc_shadow[0]
        acc_train['0.25'] = acc_shadow[1]
        acc_train['0.3'] = acc_shadow[2]
        visualizer.plot_current_train_acc(epoch, acc_train)

        print('End of epoch %d / %d \t 0.2 rad accuracy: %.3f\t Time Taken: %d sec' % (
        epoch, args.n_epochs + args.n_epochs_decay, acc_shadow[0], time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

        # eval
        test_epoch_start_time = time.time()  # timer for entire epoch
        model.eval()
        for i, data in enumerate(dataset_test):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            joint_acc_error = model.get_current_error()
            # compute acc
            res_shadow = [np.sum(np.sum(abs(joint_acc_error.cpu().data.numpy()) < thresh,
                                        axis=-1) == 22) for thresh in thresh_acc]
            correct_shadow = [c + r for c, r in zip(correct_shadow, res_shadow)]

        acc_shadow = [float(c) / float(len(dataset_test.dataset)) for c in correct_shadow]
        acc_test['0.2'] = acc_shadow[0]
        acc_test['0.25'] = acc_shadow[1]
        acc_test['0.3'] = acc_shadow[2]
        visualizer.plot_current_test_acc(epoch, acc_test)

        print('Test end of epoch %d / %d \t 0.2 rad accuracy: %.3f ' % (
        epoch, args.n_epochs + args.n_epochs_decay, acc_shadow[0]))
