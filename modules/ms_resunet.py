from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class RSB(nn.Module):
    """Residual sequential block: 1x1 reduce -> 3x3 -> 1x1 restore, residual sum. Output channels == input channels."""
    def __init__(self, channels, bottleneck=None):
        super().__init__()
        mid = bottleneck or max(8, channels // 4)

        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.relu(out + residual)
        return out


class IRB(nn.Module):
    """Improved residual block: like RSB core, but changes channels and (optionally) spatial scale. Has 1x1+BN projection shortcut."""
    def __init__(self, in_ch, out_ch, stride=2, bottleneck=None):
        super().__init__()
        mid = bottleneck or max(8, out_ch // 4)

        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.proj(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.relu(out + residual)
        return out
        
class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super().__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x
    
stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}

class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super().__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
    
    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return x

class UpTo(nn.Module):
    def __init__(self, ch, mode="deconv"):
        super().__init__()
        self.mode = mode
        if mode == "deconv":
            self.op = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
        elif mode == "pixelshuffle":
            self.op = nn.Sequential(
                nn.Conv2d(ch, ch * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(2),
            )
        elif mode == "bilinear":
            self.op = None
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")

    def forward(self, x, ref, crop_like_fn):
        if self.mode == "bilinear":
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)
            return x
        x = self.op(x)
        return crop_like_fn(x, ref)

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_rsb, downsample):
        super().__init__()
        stride = 2 if downsample else 1
        self.irb = IRB(in_ch, out_ch, stride=stride)
        self.rsb = nn.Sequential(*[RSB(out_ch) for _ in range(n_rsb)])

    def forward(self, x):
        x = self.irb(x)
        x = self.rsb(x)
        return x
    
class RefineStage(nn.Module):
    def __init__(self, in_ch, proj_ch, out_ch, up_mode=None):
        super().__init__()
        self.proj = conv3x3(in_ch, proj_ch, bias=False)
        self.adapt = nn.Sequential(
            RCUBlock(proj_ch, proj_ch, n_blocks=2, n_stages=2),
            conv3x3(proj_ch, proj_ch, bias=False),
        )
        self.crp = CRPBlock(proj_ch, proj_ch, n_stages=4)
        self.mflow = RCUBlock(proj_ch, proj_ch, n_blocks=3, n_stages=2)
        self.out = conv3x3(proj_ch, out_ch, bias=False)
        self.up = UpTo(out_ch, up_mode) if up_mode is not None else None

    def forward(self, feat, prev=None, ref=None, crop_like_fn=None):
        x = self.proj(feat)
        x = self.adapt(x)

        if prev is not None:
            x = F.relu(x + prev)
        else:
            x = F.relu(x)

        x = self.crp(x)
        x = self.mflow(x)
        x = self.out(x)

        if self.up is not None:
            x = self.up(x, ref, crop_like_fn)
        return x

class RefineNet(nn.Module):
    def __init__(self, layers, in_ch=1, channels=32, conv_kernel_size=5, conv_padding=2,
                 planes=(32, 64, 128, 256), up_mode="deconv"):
        super().__init__()
        self.inplanes = channels

        self.conv1 = nn.Conv2d(in_ch, channels, kernel_size=conv_kernel_size, stride=1, padding=conv_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        p1, p2, p3, p4 = planes
        self.enc1 = EncoderStage(channels, p1, layers[0], downsample=False)
        self.enc2 = EncoderStage(p1,      p2, layers[1], downsample=True)
        self.enc3 = EncoderStage(p2,      p3, layers[2], downsample=True)
        self.enc4 = EncoderStage(p3,      p4, layers[3], downsample=True)
        
        c1, c2, c3, c4 = planes

        c1 = planes[0]
        c2 = planes[1]
        c3 = planes[2]
        c4 = planes[3]

        # l4 сначала "толще" (256), потом всё в 128
        stage_specs = [
            ("l4", c4, 256, 128, up_mode),      # -> up to l3
            ("l3", c3, 128, 128, up_mode),      # -> up to l2
            ("l2", c2, 128, 128, up_mode),      # -> up to l1
            ("l1", c1, 128, 128, None),         # final, no up
        ]
        self.stages = nn.ModuleDict({
            name: RefineStage(in_c, proj_c, out_c, u)
            for (name, in_c, proj_c, out_c, u) in stage_specs
        })

        self.clf_conv1 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.clf_conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def _crop_like(self, x, ref):
        """
        Центр-кроп x под spatial размер ref.
        Используем, если ConvTranspose дал размер чуть больше, чем у skip-фичей.
        """
        _, _, h, w = x.size()
        _, _, hr, wr = ref.size()
    
        if h == hr and w == wr:
            return x
    
        # предполагаем h >= hr, w >= wr
        dh = h - hr
        dw = w - wr
    
        x = x[:, :, dh // 2 : h - (dh - dh // 2),
                   dw // 2 : w - (dw - dw // 2)]
        return x

    def forward(self, x):
        x  = self.relu(self.bn1(self.conv1(x)))
        
        l1 = self.enc1(x)
        l2 = self.enc2(l1)
        l3 = self.enc3(l2)
        l4 = self.enc4(l3)

        # проход сверху вниз одним циклом
        y4 = self.stages["l4"](l4, prev=None, ref=l3, crop_like_fn=self._crop_like)
        y3 = self.stages["l3"](l3, prev=y4, ref=l2, crop_like_fn=self._crop_like)
        y2 = self.stages["l2"](l2, prev=y3, ref=l1, crop_like_fn=self._crop_like)
        y1 = self.stages["l1"](l1, prev=y2, ref=None, crop_like_fn=self._crop_like)

        out = self.clf_conv2(self.clf_conv1(y1))
        return out

def MS_ResUNet():
    return RefineNet(layers=[3, 4, 3, 3], in_ch=1, planes=(32, 64, 128, 256))