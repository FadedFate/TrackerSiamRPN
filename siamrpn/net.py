from __future__ import absolute_import, division
import torch
import torch.nn as nn
import torch.nn.functional as F
# import fitlog
class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        kernel_reg, kernel_cls = self.learn(z)
        return self.inference(x, kernel_reg, kernel_cls)

    def learn(self, z):
        N = z.size(0)  # batch_num  :  8 (training)
        # print(N)
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)
        # print(kernel_reg.shape , kernel_cls.shape)    #  [8 , 10240 , 4, 4]  [8 , 5120 , 4 , 4]
        k = kernel_reg.size()[-1]   #    22 
        kernel_reg = kernel_reg.view(N , 4 * self.anchor_num, 512, k, k) # [8 , 4 * 5 , 512 , 4 , 4]
        kernel_cls = kernel_cls.view(N , 2 * self.anchor_num, 512, k, k) # [8 , 2 * 5 , 512 , 4 , 4]
        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        N = x.size(0)
        x = self.feature(x)   #  [8 , 512 , 24 , 24]
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)
        # print(x_reg.shape , x_cls.shape)   #  [8 , 512 , 22 , 22]  [8 , 512 , 22 , 22]
        zk = kernel_reg.size()[-1] 
        xk = x_reg.size()[-1]
        mid = F.conv2d(x_reg.view(1 , -1 , xk , xk) , kernel_reg.view(-1 , 512 , zk , zk) ,  groups=N)
        mid_k = mid.size()[-1] 
        out_reg = self.adjust_reg(mid.view(N , -1 , mid_k , mid_k))

        mid = F.conv2d(x_cls.view(1 , -1 , xk , xk) , kernel_cls.view(-1 , 512 , zk , zk) ,  groups=N)
        out_cls = mid.view(N , -1 , mid_k , mid_k)   
        # print(out_reg.shape , out_cls.shape)  [8 , 20 , 19 , 19]  [8 , 10 , 19 ,19]
        return out_reg, out_cls