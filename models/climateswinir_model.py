import torch
import torch.nn.functional as F
from .climatesr_model import ClimateSRModel,ClimateSRAddHGTModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ClimateSwinIRHGTModel(ClimateSRAddHGTModel):
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale')
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq['lq'].size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        self.lq['lq'] = F.pad(self.lq['lq'], (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if self.lq['hgt'].shape[-1]:
            self.lq['hgt'] = F.pad(self.lq['hgt'], (0, mod_pad_w * scale, 0, mod_pad_h * scale), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

