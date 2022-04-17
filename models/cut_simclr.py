import logging
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn 
from tqdm import tqdm
import numpy as np 

# LET US TRY AND IMPLEMENT CUTMIX 

def rand_bbox(size, lam):
    """extract bounding box"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def create_cutmix_img(imgs, labels, beta):
    pass 
