import torch
import numpy as np
import os
import math
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from .climatesr_model import ClimateSRModel, ClimateSRAddHGTModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from Plot.pcolor_map_one import pcolor_map_one_python as pcolor_map_one
from basicsr.utils.dist_util import get_dist_info
from torch import distributed as dist

def expand2square(timg, factor=16.0):
    h, w = timg.shape[-2:]
    rsp = timg.shape[:-2]

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(*rsp,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(rsp[0],1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[..., ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[..., ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1.0)

    return img, mask

def expand2square2(timg, factor=16.0):
    h, w = timg.shape[-2:]
    rsp = timg.shape[:-2]

    X = int(math.ceil(max(h,w)/float(factor))*factor)
    mod_pad_w = X - w
    mod_pad_h = X - h
    if timg.ndim == 5:
        img = F.pad(timg, (0, mod_pad_w, 0, mod_pad_h, 0, 0), 'reflect')
    elif timg.ndim == 4:
        img = F.pad(timg, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    mask = torch.zeros(rsp[0],1,X,X).type_as(timg)
    mask[..., :h,:w].fill_(1.0)

    return img, mask

@MODEL_REGISTRY.register()
class ClimateUformerModel(ClimateSRAddHGTModel):
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale')
        h, w = self.lq['lq'].shape[-2:]
        self.lq['lq'], mask = expand2square2(self.lq['lq'], factor=window_size * 4)
        if h == w and h % (window_size * 4) == 0:
            mask = None
        if self.lq['hgt'].shape[-1]:
            self.lq['hgt'], _ = expand2square2(self.lq['hgt'], factor=window_size * 4 * scale)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, None)
            self.net_g.train()

        self.output = self.output[..., :h*scale, :w*scale]

@MODEL_REGISTRY.register()
class ClimateUformerMultiscaleFuseModel(ClimateSRAddHGTModel):
    def test(self):        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale')
        h, w = self.lq['lq'].shape[-2:]
        self.lq['lq'], mask = expand2square2(self.lq['lq'], factor=window_size * 4)
        if h == w and h % (window_size * 4) == 0:
            mask = None
        if self.lq['hgt'].shape[-1]:
            self.lq['hgt'], _ = expand2square2(self.lq['hgt'], factor=window_size * 4 * scale)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, None)
            self.net_g.train()

        self.output = self.output[0][..., :h*scale, :w*scale]

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            self.output = self.net_g(self.lq)
        self.output = [v.float() for v in self.output]
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        scale = self.opt.get('scale')
        if self.cri_pix:
            l_pix = self.cri_pix(self.output[0], self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_pix_multiscale:
            gt0 = F.interpolate(self.gt, scale_factor=1/(scale*4), mode='bilinear', align_corners=False)
            gt1 = F.interpolate(self.gt, scale_factor=1/(scale*2), mode='bilinear', align_corners=False)
            gt2 = F.interpolate(self.gt, scale_factor=1/(scale), mode='bilinear', align_corners=False)
            gt3 = F.interpolate(self.gt, scale_factor=1/5, mode='bilinear', align_corners=False)
            l_pix_0 = self.cri_pix_multiscale(self.output[1], gt0)
            l_pix_1 = self.cri_pix_multiscale(self.output[2], gt1)
            l_pix_2 = self.cri_pix_multiscale(self.output[3], gt2)
            l_pix_3 = self.cri_pix_multiscale(self.output[4], gt3)
            l_pix_multi = l_pix_0 + l_pix_1 + l_pix_2 + l_pix_3
            l_total += l_pix_multi
            loss_dict['l_pix_multi'] = l_pix_multi

        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
