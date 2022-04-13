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
    def __init__(self, batch_size, temperature=0.5):
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # we normalize the initial augmentations
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        labels = torch.cat((z_i, z_j), dim=0)

        similarity_matrix = F.cosine_similarity(labels.unsqueeze(1), labels.unsqueeze(0), dim=2)
        
        # extract diagonals to mask out
        sim_i_j = torch.diag(similarity_matrix, self.batch_size)
        sim_j_i = torch.diag(similarity_matrix, -self.batch_size)

        # create 2N samples
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)
        negative_samples = similarity_matrix[self.mask]
        
        # calculate loss
        numerator = torch.exp(positive_samples / self.temperature) 
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        
        return loss
       