import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rcan_arch import RCAN, ChannelAttention
from .utils import Activation, Upsample
from basicsr.archs.arch_util import make_layer, default_init_weights

@ARCH_REGISTRY.register()
class RCANClimate(RCAN):
    def __init__(self, num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 activation='sigmoid'):
        super(RCANClimate, self).__init__(num_in_ch,
                 num_out_ch,
                 num_feat,
                 num_group,
                 num_block,
                 squeeze_factor,
                 4,
                 res_scale,
                 img_range,
                 rgb_mean)
        self.upsample = Upsample(upscale, num_feat)
        self.act = Activation(activation)

    def forward(self, x):
        x = x['lq']
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))

        return self.act(x)