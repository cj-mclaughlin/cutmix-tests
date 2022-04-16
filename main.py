"""
Self-Supervised Learning for Object Detection

Results table: 

Will use Faster RCNN, R50 
Also use Mask RCNN

Finetune on 24k on VOC 
Pretraining on CIFAR10 (except for 1 supervised, which we use imaganet, which is easy to get)
+------------------------------------------
|Pre-train|             AP50 | AP | AP75 | 
+------------------------------------------
|Random init.                |    |      |
+------------------------------------------
| Supervised (CIFAR10)       |    |      |
+------------------------------------------
| Supervised (IM-1M)         |    |      |
+------------------------------------------
| MoCo V2 (100ep ) |         |    |      |
+------------------------------------------
| SimCLR |                   |    |      |
+------------------------------------------
with cutmix
+----------------------------------------------------
|Pre-train|                        AP50 | AP | AP75 | 
+----------------------------------------------------
| Supervised (CIFAR10) w/ CutMix        |    |      |
+----------------------------------------------------
| MoCo V2 + CutMix                      |    |      |
+----------------------------------------------------
| SimCLR + CutMix                       |    |      |
+----------------------------------------------------
| Ours                                  |    |      |
+---------------------------------------------------

WE can also compare to other objection detection-focused pre-training techniques,
notably:
InsLoc (https://github.com/limbo0000/InstanceLoc)
UnMix (https://github.com/szq0214/Un-Mix)
DetCo (https://arxiv.org/pdf/2102.04803.pdf)

MAYBE look at the redundancy reduction techniques, e.g. Barlow Twins, VICReg, Whitening, Zero-CL

"""

import os 
import shutil 
import time 
import cv2 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim 
import modules.resnet as RN
import modules.preresnet as PRN
import argparse # for later
import torch.backends.cudnn as cudnn

from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torchvision import transforms, models
from modules.transform import TrainTransform

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='SimCLR for CIFAR10')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

def main():
    args = parser.parse_args()
    
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    transforms = TrainTransform() 
    
    train_loader = DataLoader(
        CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    num_classes = 10

    print("=> Creating resnet model...")
    model = RN.ResNet("cifar10", 32, num_classes)
    
    print('=> The number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=args.momentum,
                                weight_decay=args.wd, nesterov=True)

    cudnn.benchmark = True 

