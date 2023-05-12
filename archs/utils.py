#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:22:05 2021

@author: xiaohui
"""

import math
import torch
import torch.nn as nn
from basicsr.archs.arch_util import default_init_weights

class act_MFM(object):
    def __init__(self):
        pass

    def __call__(self, x):
        shape = list(x.shape)
        res = x.reshape(-1, shape[1], shape[2], 2, shape[-1] // 2)  # potential bug in conv_net
        res = torch.max(res, dim=3)
        return res


class act_MFM_FC(object):
    """docstring for act_MFM_FC"""

    def __init__(self):
        pass

    def __call__(self, x):
        shape = list(x.shape)
        res = torch.max(x.reshape(-1, 2, shape[-1] // 2), dim=1)
        return res


class act_swish(object):
    def __init__(self):
        self.swish = nn.Sigmoid()

    def __call__(self, x):
        return x * self.swish(x)

class Activation(nn.Module):
    def __init__(self, param, **kwargs):
        super(Activation, self).__init__()
        self.param = param
        self.kwargs = kwargs

        if self.param == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.param == 'leakyrelu':
            act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.param == 'elu':
            act = nn.ELU(inplace=True)
        elif self.param == 'tanh':
            act = nn.Tanh()
        elif self.param == 'mfm':
            act = act_MFM()
        elif self.param == 'mfm_fc':
            act = act_MFM_FC()
        elif self.param == 'sigmoid':
            act = nn.Sigmoid()
        elif self.param == 'swish':
            act = act_swish()
        elif self.param == 'prelu':
            act = nn.PReLU()
            # a separate aa is used for each input channel.
            # weight decay should not be used when learning a for good performance
            # channel dim is the 2nd dim of input when input has dims<2 then there is no
            # channel dim  and the number of channels = 1
        else:
            act = lambda x : x

        self.act = act

    def __call__(self, x):
        act = self.act(x)
        return act


class HGTNet(nn.Module):
    def __init__(self, upscale, embed_dim, bias=True):
        super().__init__()
        self.upscale = upscale
        if upscale == 2:
            self.hgt_net = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1, bias=bias),
                nn.PixelUnshuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(128, embed_dim, 3, 1, 1, bias=bias),
            )
        elif upscale == 4:
            self.hgt_net = nn.Sequential(
                nn.Conv2d(1, 8, 3, 1, 1, bias=bias),
                nn.PixelUnshuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=bias),
                nn.PixelUnshuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(128, embed_dim, 3, 1, 1, bias=bias),
            )
        elif upscale == 5:
            self.hgt_net = nn.Sequential(
                nn.Conv2d(1, 6, 3, 1, 1, bias=bias),
                nn.PixelUnshuffle(5),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(150, embed_dim, 3, 1, 1, bias=bias),
            )
        elif upscale == 10:
            self.hgt_net = nn.Sequential(
                nn.Conv2d(1, 2, 3, 1, 1, bias=bias),
                nn.PixelUnshuffle(5),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(50, 50, 3, 1, 1, bias=bias),
                nn.PixelUnshuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(200, embed_dim, 3, 1, 1, bias=bias),
            )

    def forward(self, x):
        return self.hgt_net(x)

class ResidualGroup3DBlockNoBN(nn.Module):
    def __init__(self, groups, num_feat=64, res_scale=1, pytorch_init=False, bias=True):
        super(ResidualGroup3DBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), padding=(0, 1, 1), groups=groups, bias=bias)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), padding=(0, 1, 1), groups=groups, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualGroup2DBlockNoBN(nn.Module):
    def __init__(self, groups, num_feat=64, res_scale=1, pytorch_init=False, bias=True):
        super(ResidualGroup2DBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=groups, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1, bias=bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(3))
        elif scale == 5:
            m.append(nn.Conv2d(num_feat, 25 * num_feat, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(5))
        elif scale == 10:
            m.append(nn.Conv2d(num_feat, 25 * num_feat, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(5))
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(2))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
