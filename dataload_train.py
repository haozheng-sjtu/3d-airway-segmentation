# -*- coding: utf-8 -*-
"""
Dataloader
"""


import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import os
import scipy.ndimage as ndimage
np.random.seed(777) #numpy


class AirwayData(Dataset):
    def __init__(self, path, train = True):
        self.path = path
        self.path_list = os.listdir(self.path)
        self.path_list.sort()
        self.sample_num = 16
        self.datalen = (len(self.path_list)//6)*self.sample_num
        self.train = train
           
    def __len__(self):
        return self.datalen
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.path_list[6*(idx//self.sample_num)])        #images after preprocessing
        dist_path = os.path.join(self.path, self.path_list[6*(idx//self.sample_num)+1])     #distance-based weight
        label_path = os.path.join(self.path, self.path_list[6*(idx//self.sample_num)+2])    #ground truth
        label_s_path = os.path.join(self.path, self.path_list[6*(idx//self.sample_num)+3])  #extract the small branches according to the diameter
        pred_path = os.path.join(self.path, self.path_list[6*(idx//self.sample_num)+4])     #predictions of the network trained with Dice loss, which is used in hard skeleton sampling
        skeleton_path = os.path.join(self.path, self.path_list[6*(idx//self.sample_num)+5]) #airway skeleton
        patient = [self.path_list[6*(idx//self.sample_num)]]
        img0 = np.load(img_path)        
        dist = np.load(dist_path)
        label0 = np.load(label_path)
        label_s = np.load(label_s_path)
        pred = np.load(pred_path)
        skeleton = np.load(skeleton_path)
        
        p = np.random.random()
        if p > 0.5:
            img, label, dist = small_airway_sample(img0, label0, label_s, dist, [150, 150, 150], 0.5) #small airway sampling
        else:
            img, label, dist = skeleton_sample(img0, label0, pred, skeleton, dist, [150, 150, 150]) #hard skeleton sampling
    
        img, label, dist = random_rotate(img, label, dist, angle=15, train=self.train, threshold=0.7) #threshold=0.7 in stage1 and 0.9 in stage2
        img, label, dist = central_crop(img, label, dist, [128, 128, 128])
          
        img = img[np.newaxis,:]
        label = label[np.newaxis,:] 
        dist = dist[np.newaxis,:] 
        
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(label.astype(np.float32)), \
    torch.from_numpy(dist.astype(np.float32)), patient
    

def small_airway_sample(sample, label, label_s, dist, crop_size, p=0.5):
    origin_size = sample.shape
    crop_size = np.array(crop_size)
    for i in range(3):
        if crop_size[i] >= origin_size[i]:
            pad_num = (crop_size[i] - origin_size[i])//2 + 1
            sample = np.pad(sample, pad_num, 'constant')
            label = np.pad(label, pad_num, 'constant')
            label_s = np.pad(label_s, pad_num, 'constant')
            dist = np.pad(dist, pad_num, 'constant')
    origin_size = sample.shape
    #factor = origin_size/crop_size
    start = [np.random.randint(0,origin_size[0]-crop_size[0]),np.random.randint(0,origin_size[1]-crop_size[1]),np.random.randint(0,origin_size[2]-crop_size[2])]
    sample2 = sample[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
    label2 = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    label_s2 = label_s[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    dist2 = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    
    reject = np.random.random()
    if reject<p:
        while((label_s2.sum()==0) or (label2-label_s2).sum()>200):
            start = [np.random.randint(0,origin_size[0]-crop_size[0]),np.random.randint(0,origin_size[1]-crop_size[1]),np.random.randint(0,origin_size[2]-crop_size[2])]
            sample2 = sample[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
            label2 = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
            label_s2 = label_s[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
            dist2 = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    
    return sample2, label2, dist2

def skeleton_sample(img, label, pred, skeleton, dist, crop_size):
    origin_size = img.shape
    crop_size = np.array(crop_size)
    for i in range(3):
        if crop_size[i] >= origin_size[i]:
            pad_num = (crop_size[i] - origin_size[i])//2 + 1
            img = np.pad(img, pad_num, 'constant')
            label = np.pad(label, pad_num, 'constant')
            pred = np.pad(pred, pad_num, 'constant')
            skeleton = np.pad(skeleton, pad_num, 'constant')
            dist = np.pad(dist, pad_num, 'constant')
    origin_size = img.shape
    if (pred*skeleton).sum() == skeleton.sum():
        start = [np.random.randint(0,origin_size[0]-crop_size[0]),np.random.randint(0,origin_size[1]-crop_size[1]),np.random.randint(0,origin_size[2]-crop_size[2])]
        img2 = img[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
        label2 = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        dist2 = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    else:
        loc = np.where(skeleton*(1-pred))
        random_loc = np.random.randint(len(loc[0]))
        start = [np.random.randint(max(0,loc[0][random_loc]-crop_size[0]), loc[0][random_loc]),
                 np.random.randint(max(0,loc[1][random_loc]-crop_size[1]), loc[1][random_loc]),
                 np.random.randint(max(0,loc[2][random_loc]-crop_size[2]), loc[2][random_loc])]
        for i in range(3):
            if (start[i]+crop_size[i]) > origin_size[i]:
                start[i] = origin_size[i] - crop_size[i] 
        img2 = img[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
        label2 = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        dist2 = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    
    return img2, label2, dist2
        

def central_crop(sample, label, dist, crop_size):
    origin_size = sample.shape
    crop_size = np.array(crop_size)
    start = (origin_size - crop_size)//2
    sample = sample[start[0]:(start[0]+crop_size[0]), start[1]:(start[1]+crop_size[1]), start[2]:(start[2]+crop_size[2])]
    label = label[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    dist = dist[start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
    
    return sample, label, dist

def random_rotate(img, label, dist, angle, train=True, threshold):
    if train:
        rotate_angle = np.random.randint(angle)*np.sign(np.random.random()-0.5)
        rotate_axes = [(0,1),(1,2),(0,2)]
        k = np.random.randint(0,3)
        img = ndimage.interpolation.rotate(img, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
        label = label.astype(np.float32)
        label = ndimage.interpolation.rotate(label, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
        threshold = threshold   #threshold=0.7 in stage1 and 0.9 in stage2
        label[label>=threshold] = 1 
        label[label<threshold] = 0
        label = label.astype(np.uint8)
        
        dist = dist.astype(np.float32)
        dist = ndimage.interpolation.rotate(dist, angle=rotate_angle, axes=rotate_axes[k], reshape=False)
        dist[dist>1] = 1
        dist[dist<0] = 0
        img[img<0] = 0
        img[img>255] = 255
        img = img.astype(np.uint8)
    return img, label, dist



