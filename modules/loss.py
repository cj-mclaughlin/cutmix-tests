import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
InfoNCE loss functions
Referenced from: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
and https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
"""

class InfoNCE(nn.Module):
    def __init__(self, batch_size, device, temperature, n_views):
        self.batch_size = batch_size
        self.n_views = n_views
        self.device = device
        self.temperature = temperature 

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
       
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        return logits, labels

    def forward(self, features):
        logits, labels = self.info_nce_loss(features) 
        return F.cross_entropy(logits, labels).to(self.device)