import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # targets -= (targets == 1) * random.randrange(0,20)*0.01
        # targets += (targets == 0) * random.randrange(0,10)*0.01
        bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets.float(), reduction=self.reduction)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss