from tabnanny import verbose
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import FocalLoss


def cosine_annealing(step, total_steps, lr_max, lr_min, lr):
    warmup_steps = total_steps * 0.15
    if step<warmup_steps:
        warmup_percent = step/warmup_steps
        return lr*warmup_percent
    else:
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def warmup_cosine_decay(step, total_steps, lr_min, lr):
    warmup_steps = total_steps * 0.15
    smooth_steps = total_steps * 0.4
    if step<warmup_steps:
        warmup_percent = step/warmup_steps
        learning_rate = lr*warmup_percent + lr_min
    elif step>=warmup_steps and step<=smooth_steps:
        learning_rate = lr+lr_min 
    else:
        learning_rate = (np.cos((step-smooth_steps)/(total_steps-smooth_steps) * np.pi)+1)/2 * lr+lr_min
    return learning_rate*1e3

class BaseTrainer:
    def __init__(self,
                 pretrained: nn.Module,
                 net: nn.Module,
                 train_loader: DataLoader,
                 text_token: torch.tensor,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100) -> None:
        self.pretrained = pretrained
        self.net = net
        self.train_loader = train_loader
        self.text_token = text_token.cuda()
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            self.learning_rate,
            weight_decay=weight_decay,
            amsgrad=True
        )

        # self.optimizer = torch.optim.Adam(
        #     net.parameters(),
        #     self.learning_rate,
        #     weight_decay=0.0001
        # )
        # self.optimizer = torch.optim.SGD(
        #     net.parameters(),
        #     learning_rate,
        #     momentum=momentum,
        #     weight_decay=weight_decay,
        #     nesterov=True,
        # )

        print(f"epochs * len(train_loader):{epochs * len(train_loader)}")
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_cosine_decay(
                step = step,
                total_steps = epochs * len(train_loader),
                lr_min = 1e-6,
                lr = self.learning_rate
            )
        )

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        # criterion = FocalLoss()
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['soft_label'].cuda()
            # forward
            feas = self.pretrained.visual.token_forward(data.type(torch.HalfTensor).cuda()).detach()
            logits = self.net(feas.type(torch.FloatTensor).cuda(), self.text_token.type(torch.FloatTensor).cuda())
            loss = criterion(logits, target)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg
        print(f"lr:{self.scheduler.get_last_lr()}")
        

        return metrics
