from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os 
import sys
from torch.optim.lr_scheduler import ExponentialLR
from collections import namedtuple
from got10k.trackers import Tracker
from torch.utils.data import DataLoader
from . import ops
from .losses import BalancedCELoss , SmoothL1
from .transforms import SiamRPNTransforms
from .net import SiamRPN
from .datasets import Pair

# import fitlog
_all__ = ['TrackerSiamRPN']


class SiamRPNLoss(nn.Module):

    def __init__(self, lamda = 1.0 , num_pos = 16, num_neg = 16):
        super(SiamRPNLoss, self).__init__()
        self.lamda = lamda
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.LossCls = BalancedCELoss()
        self.LossRpn = SmoothL1()
    def forward(self, cls_out , reg_out , cls_target , reg_target):  
        loss_cls = self.LossCls(cls_out , cls_target , num_pos = self.num_pos , num_neg = self.num_neg)
        loss_rpn = self.LossRpn(reg_out , reg_target , cls_target , num_pos = self.num_pos)
        return loss_cls + self.lamda * loss_rpn

class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, fit_log = False , **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)
        self.parse_args(fit_log , **kargs)

        #GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:1' if self.cuda else 'cpu')

        # setup model
        self.net = SiamRPN()
        ops.init_weights(self.net)
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # loss func 
        self.criterion = SiamRPNLoss(lamda = self.cfg.lamda ,num_pos=self.cfg.num_pos, num_neg=self.cfg.num_neg )
        # optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        # lr schedule
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, fit_log = False , **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295,
            # train para
            'batch_size' : 8,
            "clip" : 10,
            'num_workers': 16,
            'epoch_num': 60,
            'initial_lr': 3e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'lamda': 5,
            'num_pos' : 16,
            'num_neg' : 48,
            }
        # if fit_log:
        #     record_key = self.cfg.copy()
            # fitlog.add_other(record_key , name =  "used cfg")
        for key, val in kargs.items():
            self.cfg.update({key: val})
            # print(self.cfg)
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    @torch.no_grad()
    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = ops.create_anchors(self.cfg , self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = ops.crop_and_resize(
            image, self.center, self.z_sz,
            self.cfg.exemplar_sz, border_value = self.avg_color)

        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)

    @torch.no_grad()
    def update(self, image):
        image = np.asarray(image)
        
        # search image
        instance_image = ops.crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, border_value = self.avg_color)

        # classification and regression outputs
        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls)
        
        # offsets
        offsets = out_reg.permute(
            1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)

        # response
        response = F.softmax(out_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        
        # peak location
        best_id = np.argmax(response)
        offset = offsets[:, best_id] * self.z_sz / self.cfg.exemplar_sz

        # update center
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])

        # update scale
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None, save_dir='pretrained'):
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        # train data loader
        transforms = SiamRPNTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        if not val_seqs == None:
            val_dataset = Pair(seqs=val_seqs, transforms=transforms)
            val_dataloader = DataLoader(val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cuda,
                drop_last=True)
        step = 0
        beststep = 0 
        bestloss = 1000000
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)
            for it, (batch , cls_label , reg_label)  in enumerate(dataloader):
                # print(cls_label.dtype)
                loss = self.train_step(batch , cls_label , reg_label , backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
                # if step % 20:
                #     fitlog.add_loss(value = loss ,  step = step, name = "Training Loss" , epoch = epoch)    
                step = step + 1
                
            # validation set 
            if not val_seqs == None :
                val_loss = []
                for it, (batch , cls_label , reg_label) in enumerate(val_dataloader):
                    vloss = self.train_step(batch, cls_label , reg_label , backward=False)
                    val_loss.append(vloss)
                val_loss_mean =  np.mean(val_loss)
                print('Val loss --- Epoch: {} Loss: {:.5f}'.format(
                        epoch + 1, val_loss_mean))
                sys.stdout.flush()
                # fitlog.add_loss(value = val_loss_mean ,  step = epoch, name = "Testing Loss" )    
                if val_loss_mean < bestloss:
                    bestloss = val_loss_mean
                    beststep = epoch
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            net_path = os.path.join(save_dir, 'siamrpn_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
        # fitlog.add_best_metric({"Training":{"Val Loss":bestloss,"Step":beststep}})

    @torch.enable_grad()
    def train_step(self, batch, cls_label , reg_label , backward=True):
        # set network mode
        if backward:
            self.net.train()
        else : 
            self.net.eval()
        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        # print(z.shape)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        cls_label = cls_label.to(self.device , non_blocking = self.cuda)
        reg_label = reg_label.to(self.device , non_blocking = self.cuda)
        # print("cls: "  ,  cls_label.shape)  
        # print("reg:" , reg_label.shape)
        with torch.set_grad_enabled(backward):
            # inference
            out_reg, out_cls = self.net(z, x)
            # calculate loss
            # print(out_reg.shape , out_cls.shape)     [8 , 20 , 19  , 19]  [8 , 5 , 19 , 19 ]
            # print(reg_label.shape , cls_label.shape)  [1805  , 4]  [1805 , ]
            out_cls = out_cls.view(self.cfg.batch_size , 2 , -1).permute(0,2,1)
            out_reg = out_reg.view(self.cfg.batch_size , 4 , -1).permute(0,2,1)
            # print("cls_net " , out_cls.shape)
            
            # cls_label = np.tile()
            loss = self.criterion(out_cls , out_reg , cls_label , reg_label)
            # a = input()
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip)
                self.optimizer.step()
        
        return loss.item()    

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)
        
        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
            self.cfg.penalty_k)

        return penalty
