#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Shuang Li
# E-mail     : sli@informatik.uni-hamburg.de
# Description: 
# Date       : 17/09/2020: 15:52
# File Name  : utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
from pointnet2_ops import pointnet2_utils


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

def group_points(points, opt):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(points)
    inputs1_diff = points[:, :, 0:3].transpose(1, 2).unsqueeze(1).expand(cur_train_size, opt.sample_num_level1, 3,
                                                                         opt.SAMPLE_NUM) \
                   - points[:, 0:opt.sample_num_level1, 0:3].unsqueeze(-1).expand(cur_train_size, opt.sample_num_level1,
                                                                                  3,
                                                                                  opt.SAMPLE_NUM)  # B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)  # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)  # B * 512 * 1024
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False,
                                    sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64

    # ball query
    invalid_map = dists.gt(opt.ball_radius)  # B * 512 * 64
    for jj in range(opt.sample_num_level1):
        inputs1_idx[:, jj, :][invalid_map[:, jj, :]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size, opt.sample_num_level1 * opt.knn_K, 1).expand(cur_train_size,
                                                                                                      opt.sample_num_level1 * opt.knn_K,
                                                                                                      opt.INPUT_FEATURE_NUM)
    inputs_level1 = points.gather(1, idx_group_l1_long).view(cur_train_size, opt.sample_num_level1, opt.knn_K,
                                                             opt.INPUT_FEATURE_NUM)  # B*512*64*6

    inputs_level1_center = points[:, 0:opt.sample_num_level1, 0:3].unsqueeze(2)  # B*512*1*3
    inputs_level1[:, :, :, 0:3] = inputs_level1[:, :, :, 0:3] - inputs_level1_center.expand(cur_train_size,
                                                                                            opt.sample_num_level1,
                                                                                            opt.knn_K, 3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1, 4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1, 1, opt.sample_num_level1, 3).transpose(1,
                                                                                                             3)  # B*3*512*1
    return inputs_level1, inputs_level1_center
    # inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1


def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)
    inputs1_diff = points[:, 0:3, :].unsqueeze(1).expand(cur_train_size, sample_num_level2, 3, sample_num_level1) \
                   - points[:, 0:3, 0:sample_num_level2].transpose(1, 2).unsqueeze(-1).expand(cur_train_size,
                                                                                              sample_num_level2, 3,
                                                                                              sample_num_level1)  # B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)  # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)  # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False,
                                    sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64

    # ball query
    invalid_map = dists.gt(ball_radius)  # B * 128 * 64, invalid_map.float().sum()
    # pdb.set_trace()
    for jj in range(sample_num_level2):
        inputs1_idx.data[:, jj, :][invalid_map.data[:, jj, :]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size, 1, sample_num_level2 * knn_K).expand(cur_train_size,
                                                                                              points.size(1),
                                                                                              sample_num_level2 * knn_K)
    inputs_level2 = points.gather(2, idx_group_l1_long).view(cur_train_size, points.size(1), sample_num_level2,
                                                             knn_K)  # B*131*128*64

    inputs_level2_center = points[:, 0:3, 0:sample_num_level2].unsqueeze(3)  # B*3*128*1
    inputs_level2[:, 0:3, :, :] = inputs_level2[:, 0:3, :, :] - inputs_level2_center.expand(cur_train_size, 3,
                                                                                            sample_num_level2,
                                                                                            knn_K)  # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1
