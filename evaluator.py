import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import FocalLoss


class Evaluator:
    def __init__(
        self,
        pretrain: nn.Module,
        net: nn.Module,
        text_token: torch.tensor,
        k: int,
    ):
        self.pretrain = pretrain
        self.net = net
        self.text_token = text_token.cuda()
        self.k = k

    def eval_recall(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()
        # criterion = FocalLoss()
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        loss_avg = 0.0
        pred_list, gt_list = [], []
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].cuda()
                feas = self.pretrain.visual.token_forward(data.type(torch.HalfTensor).cuda())
                logits = self.net(feas.type(torch.FloatTensor).cuda(), self.text_token.type(torch.FloatTensor).cuda())
                prob = torch.sigmoid(logits)
                target = batch['soft_label'].cuda()
                loss = criterion(prob, target)
                loss_avg += float(loss.data)
                # gather prediction and gt
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
                for soft_label in batch['soft_label']:
                    gt_label = (soft_label == 1).nonzero(as_tuple=True)[0]\
                                .cpu().detach().tolist()
                    gt_list.append(gt_label)

        # compute mean recall
        score_list = np.zeros([56, 2], dtype=int) # first column: gt; second column: pred
        for gt, pred in zip(gt_list, pred_list):
            for gt_id in gt:
                # pos 0 for counting all existing relations
                score_list[gt_id][0] += 1
                if gt_id in pred:
                    # pos 1 for counting relations that is recalled
                    score_list[gt_id][1] += 1
        score_list = score_list[6:]
        # to avoid nan
        score_list[:, 0][score_list[:, 0] == 0] = 1
        meanrecall = np.mean(score_list[:, 1] / score_list[:, 0])

        metrics = {}
        metrics['test_loss'] = loss_avg / len(data_loader)
        metrics['mean_recall'] = meanrecall

        return metrics

    def submit(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()

        pred_list = []
        with torch.no_grad():
            for batch in data_loader:

                data = batch['data'].cuda()
                feas = self.pretrain.visual.token_forward(data.type(torch.HalfTensor).cuda())
                logits = self.net(feas.type(torch.FloatTensor).cuda(), self.text_token.type(torch.FloatTensor).cuda())
                prob = torch.sigmoid(logits)

                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
        return pred_list
