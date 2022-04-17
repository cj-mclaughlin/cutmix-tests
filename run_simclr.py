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
import torch
import torch.optim 
import argparse 
import torch.backends.cudnn as cudnn

from torchvision.datasets import CIFAR10, CIFAR100
from models.simclr.simclr import ResNetSimCLR, SimCLR
from models.simclr.pair_generator import PairGenerator
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torchvision import models
from modules.transform import TrainTransform

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='SimCLR for CIFAR10')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dname', default='cifar10',
                    help='dataset name', choices=['cifar10', 'cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

# training parameter
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=10, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

# gpu
parser.add_argument('--disable-cuda', action='store_false',
                    help='Disable CUDA')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')                 
parser.add_argument('--mixed-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

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
    train_dataset = CIFAR10(root=args.data,
                                train=True,
                                transform=PairGenerator(transforms),
                                download=True)

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              pin_memory=True, 
                              drop_last=True)
        
    print("=> Starting SimCLR...")
    model = ResNetSimCLR(base_model=args.arch, 
                        dataset=args.dname, 
                        out_dim=args.out_dim)
    
    print('=> The number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-3, last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        print('=> Training SimCLR on CIFAR10...')
        simclr.train(train_loader)

    cudnn.benchmark = True 

if __name__ == "__main__":
    main()