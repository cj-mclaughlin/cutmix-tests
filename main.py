import os 
import shutil 
import time 
import cv2 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim 
import models.modules.resnet as RN
import models.modules.preresnet as PRN
import argparse # for later
import torch.backends.cudnn as cudnn

from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torchvision import transforms

def main():
    # parameters
    batch_size = 64 
    lr = 0.1 
    momentum = 0.9
    weight_decay = 1e-4

    # transform data 
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = DataLoader(
        CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CIFAR10('../data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, pin_memory=True)
    num_classes = 10

    print("=> Creating resnet model...")
    model = RN.ResNet("cifar10", 32, num_classes)
    
    print('=> The number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)

    cudnn.benchmark = True 