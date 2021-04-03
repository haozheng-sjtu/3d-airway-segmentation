#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:11:20 2018

@author: zhenghao
"""

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import os
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings('ignore')


class AirwayData(Dataset):
    def __init__(self, path, train = True):
        self.path = path
        self.path_list = os.listdir(self.path)
        self.path_list.sort()
        self.datalen = len(self.path_list)//3
        self.train = train
        
    def __len__(self):
        return self.datalen
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.path_list[3*idx])
        label_path = os.path.join(self.path, self.path_list[3*idx+1])
        patient = [self.path_list[3*idx].split('_')[0]]
        
        img = np.load(img_path)
        img = img[np.newaxis,:]
       
        label = np.load(label_path)
        label = label[np.newaxis,:]
        
        
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(label.astype(np.float32)), \
    torch.from_numpy(img.astype(np.float32)), patient
    
