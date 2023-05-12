import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rcan_arch import ResidualGroup
from basicsr.archs.arch_util import make_layer, default_init_weights
from einops import rearrange
from archs.utils import Activation, HGTNet
from archs.utils import Upsample as SR_Upsample
from archs.uformer_module import *

class MyUformer_depth5(nn.Module):
    def __init__(self, img_size=32, in_chans=128,
                 embed_dim=128, depths=[2, 2, 2, 2, 2], num_heads=[4, 8, 16, 16, 8],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[2]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim*4, embed_dim*2)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[3]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[3:4]):sum(depths[3:5])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        #Encoder
        conv0 = self.encoderlayer_0(y,mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        pool1 = self.dowsample_1(conv1)

        # Bottleneck
        conv2 = self.conv(pool1, mask=mask)

        #Decoder
        up0 = self.upsample_0(conv2)
        deconv0 = torch.cat([up0,conv1],-1)
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv0],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)

        # Output Projection
        y = self.output_proj(deconv1)
        return y


class MyUformer_depth7(nn.Module):
    def __init__(self, img_size=32, in_chans=128,
                 embed_dim=128, depths=[2, 2, 2, 2, 2, 2, 2], num_heads=[2, 4, 8, 16, 16, 8, 4],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[3]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim*8, embed_dim*4)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[4]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        #Encoder
        conv0 = self.encoderlayer_0(y,mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1,mask=mask)
        pool2 = self.dowsample_2(conv2)

        # Bottleneck
        conv3 = self.conv(pool2, mask=mask)

        #Decoder
        up0 = self.upsample_0(conv3)
        deconv0 = torch.cat([up0,conv2],-1)
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv1],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2,conv0],-1)
        deconv2 = self.decoderlayer_2(deconv2,mask=mask)

        # Output Projection
        y = self.output_proj(deconv2)
        return y


@ARCH_REGISTRY.register()
class ClimateUformer(nn.Module):
    def __init__(self,
                 add_hgt,
                 num_in_ch=3,
                 num_out_ch=3,
                 upscale=5,
                 img_size=32,
                 embed_dim=128,
                 depths=[2, 2, 2, 2, 2],
                 num_heads=[4, 8, 16, 16, 8],
                 window_size=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='leff',
                 activation='none'
                 ):
        super(ClimateUformer, self).__init__()
        self.upscale = upscale
        self.add_hgt = add_hgt

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        if self.add_hgt:
            self.conv_before_body = nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
            self.hgt_net = HGTNet(upscale, embed_dim)

        if len(depths) == 5:
            Block = MyUformer_depth5
        elif len(depths) == 7:
            Block = MyUformer_depth7
        else:
            raise ValueError('Unsupported depths: ', depths)

        # reconstruction
        self.reconstruction_network = Block(
            img_size=img_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            win_size=window_size,
            mlp_ratio=mlp_ratio,
            qk_scale=qk_scale,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=False
        )

        # upsample
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = SR_Upsample(upscale, 64)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)
        # activation function
        self.act = Activation(activation)

    def forward(self, x_in, mask=None):
        x = x_in['lq']
        hgt = x_in['hgt']

        feat = self.conv_first(x)
        if self.add_hgt:
            hgt_feat = self.hgt_net(hgt)
            feat = self.conv_before_body(torch.cat([feat, hgt_feat], dim=1))

        out = self.conv_after_body(self.reconstruction_network(feat, mask)) + feat
        out = self.conv_before_upsample(out)
        out = self.upsample(out)
        out = self.conv_last(out)
        #base = F.interpolate(x_center, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        #out += base
        return self.act(out)

@ARCH_REGISTRY.register()
class ClimateUformerMultiScaleHGT(nn.Module):
    def __init__(self,
                 add_hgt,
                 num_in_ch=3,
                 num_out_ch=3,
                 upscale=5,
                 img_size=32,
                 embed_dim=128,
                 depths=[2, 2, 2, 2, 2],
                 num_heads=[4, 8, 16, 16, 8],
                 window_size=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='leff',
                 activation='none',
                 dowsample=Downsample,
                 upsample=Upsample,
                 se_layer=False,
                 multi_add_pos='decoder', #encoder, decoder, xcoder
                 has_input_output_proj=False
                 ):
        super(ClimateUformerMultiScaleHGT, self).__init__()
        self.upscale = upscale
        self.add_hgt = add_hgt
        self.multi_add_pos = multi_add_pos
        self.has_input_output_proj = has_input_output_proj

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = window_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[2]
        dec_dpr = enc_dpr[::-1]

        if self.has_input_output_proj:
            self.input_proj = InputProj(in_channel=embed_dim, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
            self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim*4, embed_dim*2)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[3]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[3:4]):sum(depths[3:5])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        if self.add_hgt:
            self.hgt_layer = nn.ModuleList()
            self.hgt_layer.append(nn.Sequential(
                nn.Conv2d(1, 2, 3, 1, 1),
                nn.PixelUnshuffle(5),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(50, embed_dim // 4, 3, 1, 1),
                nn.PixelUnshuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))
            self.hgt_layer.append(nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim*2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))
            self.hgt_layer.append(nn.Sequential(
                nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))

            self.hgt_conv = nn.ModuleList()
            #self.hgt_conv.append(nn.Conv2d(50, 64, 3, 1, 1))
            self.hgt_conv.append(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            self.hgt_conv.append(nn.Conv2d(embed_dim*2, embed_dim*2, 3, 1, 1))
            self.hgt_conv.append(nn.Conv2d(embed_dim*4, embed_dim*4, 3, 1, 1))

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # upsample
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        #self.upsample = SR_Upsample(upscale, 64)
        self.sa_upsample_1 = nn.Sequential(
            nn.Conv2d(64, 4 * 64, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.sa_upsample_2 = nn.Sequential(
            nn.Conv2d(64, 25 * 64, 3, 1, 1),
            nn.PixelShuffle(5)
        )
        if self.has_input_output_proj:
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)
        # activation function
        self.act = Activation(activation)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x_in, mask=None):
        x = x_in['lq']
        hgt = x_in['hgt']
        b, c, h, w = x.shape

        feat = self.conv_first(x)

        if self.add_hgt:
            hgt_feat = []
            for layer in self.hgt_layer:
                hgt = layer(hgt)
                hgt_feat.append(hgt)
            for i, layer in enumerate(self.hgt_conv):
                hgt_feat[i] = layer(hgt_feat[i])

        if self.has_input_output_proj:
            feat1 = self.input_proj(feat)
        else:
            feat1 = feat.flatten(2).transpose(1, 2).contiguous()
        if self.add_hgt and self.multi_add_pos == 'encoder':
            feat1 = feat1 + hgt_feat[0].flatten(2).transpose(1, 2).contiguous()
        #Encoder
        conv0 = self.encoderlayer_0(feat1,mask=mask)
        if self.add_hgt and self.multi_add_pos == 'xcoder':
            conv0 = conv0 + hgt_feat[0].flatten(2).transpose(1, 2).contiguous()
        pool0 = self.dowsample_0(conv0)
        if self.add_hgt and self.multi_add_pos == 'encoder':
            pool0 = pool0 + hgt_feat[1].flatten(2).transpose(1, 2).contiguous()
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        if self.add_hgt and self.multi_add_pos == 'xcoder':
            conv1 = conv1 + hgt_feat[1].flatten(2).transpose(1, 2).contiguous()
        pool1 = self.dowsample_1(conv1) #embed_dim*4
        if self.add_hgt and self.multi_add_pos == 'encoder':
            pool1 = pool1 + hgt_feat[2].flatten(2).transpose(1, 2).contiguous()
        if self.add_hgt and self.multi_add_pos == 'decoder':
            pool1 = pool1 + hgt_feat[3].flatten(2).transpose(1, 2).contiguous()
        # Bottleneck
        conv2 = self.conv(pool1, mask=mask)
        if self.add_hgt and self.multi_add_pos == 'xcoder':
            conv2 = conv2 + hgt_feat[2].flatten(2).transpose(1, 2).contiguous()

        #Decoder
        up0 = self.upsample_0(conv2) #embed_dim*2
        if self.add_hgt and self.multi_add_pos == 'decoder':
            up0 = up0 + hgt_feat[2].flatten(2).transpose(1, 2).contiguous()
        deconv0 = torch.cat([up0,conv1],-1)
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)

        up1 = self.upsample_1(deconv0) # embed_dim*1
        if self.add_hgt and self.multi_add_pos == 'decoder':
            up1 = up1 + hgt_feat[1].flatten(2).transpose(1, 2).contiguous()
        deconv1 = torch.cat([up1,conv0],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)

        if self.has_input_output_proj:
            deconv1 = self.output_proj(deconv1)
        else:
            B, L, C = deconv1.shape
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
            deconv1 = deconv1.transpose(1, 2).view(B, C, H, W)

        out = self.conv_after_body(deconv1) + feat
        out = self.conv_before_upsample(out)
        out = self.sa_upsample_1(out)
        if self.add_hgt and self.multi_add_pos == 'decoder':
            out = out + hgt_feat[0]
        out = self.sa_upsample_2(out)
        #out = self.upsample(out)
        out = self.conv_last(out)
        return self.act(out)


@ARCH_REGISTRY.register()
class ClimateUformerMultiScaleHGTMultiScaleOut(nn.Module):
    def __init__(self,
                 add_hgt,
                 num_in_ch=3,
                 num_out_ch=3,
                 upscale=5,
                 img_size=32,
                 embed_dim=128,
                 depths=[2, 2, 2, 2, 2],
                 num_heads=[4, 8, 16, 16, 8],
                 window_size=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='leff',
                 activation='none',
                 dowsample=Downsample,
                 upsample=Upsample,
                 se_layer=False,
                 multi_add_pos='decoder', #encoder, decoder, xcoder
                 has_input_output_proj=False
                 ):
        super(ClimateUformerMultiScaleHGTMultiScaleOut, self).__init__()
        self.upscale = upscale
        self.add_hgt = add_hgt
        self.multi_add_pos = multi_add_pos
        self.has_input_output_proj = has_input_output_proj

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = window_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[2]
        dec_dpr = enc_dpr[::-1]

        if self.has_input_output_proj:
            self.input_proj = InputProj(in_channel=embed_dim, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
            self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.scale0_outconv = nn.Conv2d(embed_dim * 4, 64, 3, 1, 1)

        # Decoder
        self.upsample_0 = upsample(embed_dim*4, embed_dim*2)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[3]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.scale1_outconv = nn.Conv2d(embed_dim * 4, 64, 3, 1, 1)
        self.upsample_1 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[3:4]):sum(depths[3:5])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.scale2_outconv = nn.Conv2d(embed_dim * 2, 64, 3, 1, 1)

        if self.add_hgt:
            self.hgt_layer = nn.ModuleList()
            self.hgt_layer.append(nn.Sequential(
                nn.Conv2d(1, 2, 3, 1, 1),
                nn.PixelUnshuffle(5),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(50, embed_dim // 4, 3, 1, 1),
                nn.PixelUnshuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))
            self.hgt_layer.append(nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim*2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))
            self.hgt_layer.append(nn.Sequential(
                nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ))

            self.hgt_conv = nn.ModuleList()
            #self.hgt_conv.append(nn.Conv2d(50, 64, 3, 1, 1))
            self.hgt_conv.append(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            self.hgt_conv.append(nn.Conv2d(embed_dim*2, embed_dim*2, 3, 1, 1))
            self.hgt_conv.append(nn.Conv2d(embed_dim*4, embed_dim*4, 3, 1, 1))

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # upsample
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        #self.upsample = SR_Upsample(upscale, 64)
        self.sa_upsample_1 = nn.Sequential(
            nn.Conv2d(64, 4 * 64, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        #self.scale3_outconv = nn.Conv2d(64, 64, 3, 1, 1)
        self.sa_upsample_2 = nn.Sequential(
            nn.Conv2d(64, 25 * 64, 3, 1, 1),
            nn.PixelShuffle(5)
        )
        if self.has_input_output_proj:
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
        self.scale4_outconv = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)

        self.conv_fuse = nn.Conv2d(64 * 4, 64, 3, 1, 1)
        # activation function
        self.act = Activation(activation)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x_in, mask=None):
        x = x_in['lq']
        hgt = x_in['hgt']
        b, c, h, w = x.shape

        feat = self.conv_first(x)

        if self.add_hgt:
            hgt_feat = []
            for layer in self.hgt_layer:
                hgt = layer(hgt)
                hgt_feat.append(hgt)
            for i, layer in enumerate(self.hgt_conv):
                hgt_feat[i] = layer(hgt_feat[i])

        if self.has_input_output_proj:
            feat1 = self.input_proj(feat)
        else:
            feat1 = feat.flatten(2).transpose(1, 2).contiguous()
        if self.add_hgt and self.multi_add_pos == 'encoder':
            feat1 = feat1 + hgt_feat[0].flatten(2).transpose(1, 2).contiguous()
        #Encoder
        conv0 = self.encoderlayer_0(feat1,mask=mask)
        if self.add_hgt and self.multi_add_pos == 'xcoder':
            conv0 = conv0 + hgt_feat[0].flatten(2).transpose(1, 2).contiguous()
        pool0 = self.dowsample_0(conv0)
        if self.add_hgt and self.multi_add_pos == 'encoder':
            pool0 = pool0 + hgt_feat[1].flatten(2).transpose(1, 2).contiguous()
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        if self.add_hgt and self.multi_add_pos == 'xcoder':
            conv1 = conv1 + hgt_feat[1].flatten(2).transpose(1, 2).contiguous()
        pool1 = self.dowsample_1(conv1) #embed_dim*4
        if self.add_hgt and self.multi_add_pos == 'encoder':
            pool1 = pool1 + hgt_feat[2].flatten(2).transpose(1, 2).contiguous()
        if self.add_hgt and self.multi_add_pos == 'decoder':
            pool1 = pool1 + hgt_feat[3].flatten(2).transpose(1, 2).contiguous()
        # Bottleneck
        conv2 = self.conv(pool1, mask=mask)
        if self.add_hgt and self.multi_add_pos == 'xcoder':
            conv2 = conv2 + hgt_feat[2].flatten(2).transpose(1, 2).contiguous()

        #Decoder
        B, L, C = conv2.shape
        out0 = self.scale0_outconv(conv2.transpose(1, 2).reshape(B, C, int(math.sqrt(L)), int(math.sqrt(L))))
        up0 = self.upsample_0(conv2) #embed_dim*2
        if self.add_hgt and self.multi_add_pos == 'decoder':
            up0 = up0 + hgt_feat[2].flatten(2).transpose(1, 2).contiguous()
        deconv0 = torch.cat([up0,conv1],-1)
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)

        B, L, C = deconv0.shape
        out1 = self.scale1_outconv(deconv0.transpose(1, 2).reshape(B, C, int(math.sqrt(L)), int(math.sqrt(L))))
        up1 = self.upsample_1(deconv0) # embed_dim*1
        if self.add_hgt and self.multi_add_pos == 'decoder':
            up1 = up1 + hgt_feat[1].flatten(2).transpose(1, 2).contiguous()
        deconv1 = torch.cat([up1,conv0],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)

        B, L, C = deconv1.shape
        out2 = self.scale2_outconv(deconv1.transpose(1, 2).reshape(B, C, int(math.sqrt(L)), int(math.sqrt(L))))

        if self.has_input_output_proj:
            deconv1 = self.output_proj(deconv1)
        else:
            B, L, C = deconv1.shape
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
            deconv1 = deconv1.transpose(1, 2).view(B, C, H, W)

        
        out = self.conv_after_body(deconv1) + feat
        out = self.conv_before_upsample(out)
        out = self.sa_upsample_1(out)
        #out3 = self.scale3_outconv(out)
        if self.add_hgt and self.multi_add_pos == 'decoder':
            out = out + hgt_feat[0]
        out = self.sa_upsample_2(out)
        
        #out = self.upsample(out)
        out4 = self.scale4_outconv(out)
        
        out0 = F.interpolate(out0, out4.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, out4.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, out4.shape[2:], mode='bilinear', align_corners=False)
        #out3 = F.interpolate(out3, out4.shape[2:], mode='bilinear', align_corners=False)
        out = self.conv_fuse(torch.cat([out0, out1, out2, out4], dim=1))
        out = self.conv_last(out)
        return self.act(out), self.act(out0), self.act(out1), self.act(out2) #, self.act(out3)
