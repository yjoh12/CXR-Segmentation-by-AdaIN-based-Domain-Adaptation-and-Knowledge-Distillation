"""
Modified by Yujin Oh
https://github.com/yjoh12/CXR-Segmentation-by-AdaIN-based-Domain-Adaptation-and-Knowledge-Distillation.git

Forked from StarGAN v2, Copyright (c) 2020-preeent NAVER Corp.
https://github.com/clovaai/stargan-v2.git
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

tf_style_dim = 16


class Blk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, upsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        if self.upsample:
            self.conv1 = nn.ConvTranspose2d(dim_in, dim_out, 2, 2, 0)
        else:
            self.conv1 = nn.Conv2d(dim_in, dim_out, 2, 2, 0)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.dropout = nn.Dropout(0.5)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
            self.self_norm1 = AdaIN(style_dim, dim_in)
            self.self_norm2 = AdaIN(style_dim, dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.dropout = nn.Dropout(0.5)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x, s=None):
        if self.normalize:
            if s is not None:
                x = self.self_norm1(x, s)
            else:
                x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            if s is not None:
                x = self.self_norm2(x, s)
            else:
                x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        # if self.downsample:
        #     x = F.avg_pool2d(x, 2)
        return x

    def forward(self, x, s=None):
        if s is not None:
            x = self._shortcut(x) + self._residual(x, s=s) # + self._shortcut(x) 
        else:
            x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
    
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.isnorm1 = nn.InstanceNorm2d(dim_in)
        self.norm1 = AdaIN(style_dim, dim_in)
        
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.isnorm2 = nn.InstanceNorm2d(dim_out)        
        self.norm2 = AdaIN(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.dropout = nn.Dropout(0.5)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, skip=None, seg=False):
        if seg:
            x = self.isnorm1(x)
        else:
            x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if seg:
            x = self.isnorm2(x)
        else:
            x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, skip=None, seg=False):
        out = self._residual(x, s, seg=seg)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(self, args, max_conv_dim=512): #1024
        super().__init__()
        self.num_domains = args.num_domains
        dim_in = 2**14 // args.img_size
        img_size = args.img_size
        input_dim = args.input_dim
        seg_cls = args.seg_class
        style_dim = args.style_dim
        
        self.input = nn.ModuleList()
        self.input += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]
        self.input += [nn.Conv2d(seg_cls, dim_in, 3, 1, 1)]

        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
    
        self.output = nn.ModuleList()
        self.output += [nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, input_dim, 1, 1, 0))]
        self.output += [nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, seg_cls, 1, 1, 0))]

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, style_dim=style_dim, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, style_dim=style_dim, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))
            
            
    def forward(self, x, y_x, y_t, s, masks=None, seg=False, self_cons=None, att=False, ref=None):

        x = self.input[torch.div(y_x[[0]], (self.num_domains-1), rounding_mode='trunc')](x)
        cache = {}
        list_skip = []

        # encode
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            
            if (self_cons is not None):
                x = block(x, self_cons)
            else:
                x = block(x)
            
        list_skip.reverse()
        count = 0

        # decode
        for block in self.decode:
            count += 1
            x = block(x, s, seg=seg)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        
        return self.output[torch.div(y_t[[0]], (self.num_domains-1), rounding_mode='trunc')](x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2, hidden=512):
        super().__init__()
        self.num_domains = num_domains

        layers = []
        layers += [nn.Linear(latent_dim, hidden)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden, hidden)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y[[0]]]  # (batch)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512, seg_cls=2, input_dim=1):
        super().__init__()
        self.num_domains = num_domains

        dim_in = 2**14 // img_size
        blocks = []
        self.input = nn.ModuleList()
        self.input += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]
        self.input += [nn.Conv2d(seg_cls, dim_in, 3, 1, 1)]
        # blocks += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        x = self.input[torch.div(y[[0]], (self.num_domains-1), rounding_mode='trunc')](x)
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y[[0]]]  # (batch)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, seg_cls=2, input_dim=1):
        super().__init__()
        self.num_domains = num_domains
        dim_in = 2**14 // img_size
        blocks = []
        self.input = nn.ModuleList()
        self.input += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]
        self.input += [nn.Conv2d(seg_cls, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        x = self.input[torch.div(y[[0]], (self.num_domains-1), rounding_mode='trunc')](x)
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y[[0]]]  # (batch)
        return out


def build_model(args):

    generator = Generator(args) 
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains+1, hidden=args.hidden_dim)
    discriminator = Discriminator(args.img_size, args.num_domains, seg_cls=args.seg_class, input_dim=args.input_dim)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, seg_cls=args.seg_class, input_dim=args.input_dim)

    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                mapping_network=mapping_network,
                style_encoder=style_encoder,
                discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                    mapping_network=mapping_network_ema,
                    style_encoder=style_encoder_ema)
                    
    return nets, nets_ema
