"""SimCLR in Pytorch
modified code from: https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/models/resnet_simclr.py#L7
"""

import logging
import os
import sys
import torch
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from modules.loss import InfoNCE


class ResNetSimCLR(nn.Module):
    """SimCLR model"""
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self.resnet_dict[base_model] 
        dim_mlp = self.backbone.fc.in_features # num inputs to linear layer

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

class SimCLR(object):
    """training code for SimCLR"""
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter() # data visualization 
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = InfoNCE(self.args.batch_size, self.args.device, self.args.temperature, self.args.n_views)

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.mixed_precision)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.mixed_precision):
                    features = self.model(images)
                    loss = self.criterion(features)

                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if n_iter % self.args.log_every_n_steps == 0:
                        # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        self.writer.add_scalar('loss', loss, global_step=n_iter)
                        # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    n_iter += 1
                
                # warmup for the first 10 epochs
            if epoch  >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch}\tLoss: {loss}")

        logging.info("Training has finished.")

        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")