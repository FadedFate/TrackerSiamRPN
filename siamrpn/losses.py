from __future__ import absolute_import, division
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# import fitlog
# fitlog.commit(__file__)

# __all__ = ['BalancedLoss', 'FocalLoss', 'GHMCLoss', 'OHNMLoss','SmoothL1']
# For rpn_right now , only
__all__ = ['BalancedCELoss', 'SmoothL1']

def log_sigmoid(x):
    # for x > 0: 0 - log(1 + exp(-x))
    # for x < 0: x - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)


def log_minus_sigmoid(x):
    # for x > 0: -x - log(1 + exp(-x))
    # for x < 0:  0 - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)


class BalancedCELoss(nn.Module):

    def __init__(self):
        super(BalancedCELoss, self).__init__()
        # self.neg_weight = neg_weight
    
    def forward(self, inputs, target , num_pos , num_neg):  # inputs [N , 1805 , H ] N = 8 , H = 2  target [N , 1805]
        loss_all = []
        for batch_id in range(target.shape[0]):
            min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
            min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg))
            pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist() 
            neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist() 

            pos_index_random = random.sample(pos_index, min_pos)
            if len(pos_index) > 0:
                pos_loss_id_final = F.cross_entropy(input=inputs[batch_id][pos_index_random],
                                                     target=target[batch_id][pos_index_random], reduction='none')
            else:
                pos_loss_id_final = torch.FloatTensor([0]).cuda()

            neg_index_random = random.sample(neg_index, min_neg)
            if len(neg_index) > 0:
                neg_loss_id_final = F.cross_entropy(input=inputs[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduction='none')
            else:
                neg_loss_id_final = torch.FloatTensor([0]).cuda()

            loss_id = (pos_loss_id_final.mean() + neg_loss_id_final.mean()) / 2
            loss_all.append(loss_id)
        # print(loss_all)
        # print(loss_all)
        return torch.stack(loss_all).mean()

class SmoothL1(nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()
    def forward(self, inputs, target , label , num_pos):  # inputs [N , 1805 , H ] N = 8 , H = 4  target [N , 1805 , 4]  label : [pos , neg , ]
        loss_all = []
        for batch_id in range(target.shape[0]): 
            pos_index = np.where(label[batch_id].cpu() == 1)[0]   # 仅对证样本进行计算
            min_pos = min(len(pos_index), num_pos)   
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss_id = F.smooth_l1_loss(inputs[batch_id][pos_index], target[batch_id][pos_index])
                # print(loss_id)
            else:
                loss_id = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_id.mean())
        # print(loss_all)
        return torch.stack(loss_all).mean()

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)

        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)

        loss = -(target * pos_weight * pos_log_sig + \
            (1 - target) * neg_weight * neg_log_sig)
        
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        loss /= avg_weight.mean()

        return loss.mean()


class GHMCLoss(nn.Module):
    
    def __init__(self, bins=30, momentum=0.5):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [t / bins for t in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
    
    def forward(self, input, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        tot = input.numel()
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights /= weights.mean()

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        
        return loss


class OHNMLoss(nn.Module):
    
    def __init__(self, neg_ratio=3.0):
        super(OHNMLoss, self).__init__()
        self.neg_ratio = neg_ratio
    
    def forward(self, input, target):
        pos_logits = input[target > 0]
        pos_labels = target[target > 0]

        neg_logits = input[target == 0]
        neg_labels = target[target == 0]

        pos_num = pos_logits.numel()
        neg_num = int(pos_num * self.neg_ratio)
        neg_logits, neg_indices = neg_logits.topk(neg_num)
        neg_labels = neg_labels[neg_indices]

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits, neg_logits]),
            torch.cat([pos_labels, neg_labels]),
            reduction='mean')
        
        return loss
