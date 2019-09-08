###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize
from models import BaseNet
from torch.nn import Module, Parameter, Softmax

__all__ = ['DANet','PAM_Module','CAM_Module']

class DANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    Reference:
        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015
    """
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, _, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)
        
class DANetHead(Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

class PAM_Module(Module):
    def __init__(self,in_dim):
        super(PAM_Module,self).__init__()

        self.query_conv=Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv=Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv=Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax=Softmax(dim=-1)

    def forward(self,x):
        batch_size, C, height, width = x.size)()
        proj_query = self.query_conv(x).view(batch_size,-1,height*width).permute(0,2,1)
        proj_key = self.key_conv(x).view(batch_size,-1,height*width)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size,-1,height*width)

        out = torch.bmm(proj_value,attention)
        out = out.view(batch_size,C,height,width)

        out = self.gamma*out+x

        return out

class CAM_Module(Module):
    def __init__(self,in_dim):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = x.view(batch_size,C,-1)
        proj_key = x.view(batch_size,C,-1).permute(0,2,1)
        energy = torch.bmm(proj_query,proj_key)
        new_energy = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(new_energy)
        proj_value = x.view(batch_size,C,-1)

        out = torch.bmm(attention,proj_value)
        out = out.view(batch_size,C,height, width)

        out = self.gamma*out+x

        return out

