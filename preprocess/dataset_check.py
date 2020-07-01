#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : dataset_check.py
# Purpose : find and delete continuous same groundtruth and images
# Creation Date : 26-06-2020
# Created By : Shuang Li

import numpy as np
import cv2
import os
import sys


subject = 'Subject' + sys.argv[1]
view_folder = '76150'

base_path = ''
path = os.path.join(base_path, subject, view_folder)
file_list = os.listdir(path)
print('This folder have ', len(file_list), 'images')
file_list.sort()
img_pre = np.zeros([480, 640])

# check continuous same images
a = []
for file in file_list:
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_ANYDEPTH)
    if (img == img_pre).all():
        a += [file]
        print(file, len(a))
    img_pre = img
np.save(subject + '_' + view_folder + '_dataset_same_images.npy', np.array(a))
#todo: delete same images


# check continuous same groundtruth
filename = view_folder + '_loc_shift_made_by_qi_20180112_v2.txt'
DataFile = open(filename, "r")
lines = DataFile.read().splitlines()

label = []
for line in lines:
    frame = line.split(' ')[0].replace("\t", "")
    label_source = line.split('\t')[1:]
    label.append(frame)
    for l in label_source[0:63]:
        label.append(float(l.replace(" ", "")))

data = np.array(label).reshape(-1, 64)
unq, indices = np.unique(data[:, 1:], return_index=True, axis=0)
full_indices = np.array(list(range(data.shape[0]+1)))
bad_gt = np.setdiff1d(full_indices, indices)
good_gt = data[indices]

#todo: save good gt and only visulaize good data

np.save('dataset_same_gt.npy', good_gt)

