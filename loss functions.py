"""
pred: network prediction
target: ground truth
dist: distance-based weight

"""

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy,sigmoid,binary_cross_entropy

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = ((iflat) * tflat).sum()
    
    return 1-((2. * intersection + smooth)/((iflat).sum() + (tflat).sum() + smooth))


def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1 - alpha
    intersection = (pred*target).sum()
    FP = (pred*(1-target)).sum()
    FN = ((1-pred)*target).sum()
    return 1-(intersection + smooth)/(intersection + alpha*FP + beta*FN + smooth)


def general_union_loss(pred, target, dist):
    weight = dist*target + (1-target)
    #when weight = 1, this loss becomes Root Tversky loss
    smooth = 1.0
    alpha = 0.1 #alpha=0.1 in stage1 and 0.2 in stage2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target*sigma1 + (1-target)*sigma2
    intersection = (weight*((pred+weight_i)**0.7)*target).sum()
    intersection2 = (weight*(alpha*pred + beta*target)).sum()
    return 1-(intersection + smooth)/(intersection2 + smooth)
    





