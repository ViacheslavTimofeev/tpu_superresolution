from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

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

class Bottleneck(nn.Module):        #--->(Bottleneck(16, 16, stride=1, downsample))
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)     #downsample     x-->(nn.Conv2d+nn.BatchNorm2d)-->residual

        out += residual
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, block, layers):     #---->(Bottleneck, [3, 4, 23, 3])
        super().__init__()
        
        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.upCT4 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.upCT3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.upCT2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])                  #---->(Bottleneck, 16, [3])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)        #---->(Bottleneck, 32, [4)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)        #---->(Bottleneck, 64, [23])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)       #---->(Bottleneck,128, [3])
        
        self.p_ims1d2_outl1_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage1_b = self._make_rcu(256, 256, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g1_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(256, 128, bias=False)
        self.up_ps4 = nn.Conv2d(128, 128 * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps4 = nn.PixelShuffle(2)
        
        self.p_ims1d2_outl2_dimred = conv3x3(512, 128, bias=False)
        self.adapt_stage2_b = self._make_rcu(128, 128, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(128, 128, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(128, 128, 4)
        self.mflow_conv_g2_b = self._make_rcu(128, 128, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(128, 128, bias=False)
        self.up_ps3 = nn.Conv2d(128, 128 * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps3 = nn.PixelShuffle(2)

        self.p_ims1d2_outl3_dimred = conv3x3(256, 128, bias=False)
        self.adapt_stage3_b = self._make_rcu(128, 128, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(128, 128, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(128, 128, 4)
        self.mflow_conv_g3_b = self._make_rcu(128, 128, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(128, 128, bias=False)
        self.up_ps2 = nn.Conv2d(128, 128 * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps2 = nn.PixelShuffle(2)

        self.p_ims1d2_outl4_dimred = conv3x3(128, 128, bias=False)
        self.adapt_stage4_b = self._make_rcu(128, 128, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(128, 128, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(128, 128, 4)
        self.mflow_conv_g4_b = self._make_rcu(128, 128, 3, 2)
        
        self.clf_conv1 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.clf_conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=2, bias=True)

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
        
    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):   #---->(Bottleneck, 16, [3])
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  #inplanes=16  Bottleneck.expansion=4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))   #(Bottleneck(16, 16, stride=1, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  #[1,3]
            layers.append(block(self.inplanes, planes))                   #Bottleneck(16, 16)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        
        #x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4) # Bilinear
        #x4 = self.upCT4(x4) # ConvTranspose
        x4 = self.up_ps4(x4)  # PixelShuffle # B × 512 × H/8 × W/8
        x4 = self.ps4(x4)  #PixelShuffle   # B × 128 × H/4 × W/4
        x4 = self._crop_like(x4, l3) # ConvTranspose, PixelShuffle

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        
        #x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3) # Bilinear
        #x3 = self.upCT3(x3) # ConvTranspose
        x3 = self.up_ps3(x3)  # PixelShuffle
        x3 = self.ps3(x3)  # PixelShuffle
        x3 = self._crop_like(x3, l2) # ConvTranspose, PixelShuffle

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        
        #x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)
        #x2 = self.upCT2(x2) # ConvTranspose
        x2 = self.up_ps2(x2)  # PixelShuffle
        x2 = self.ps2(x2)  # PixelShuffle
        x2 = self._crop_like(x2, l1) # ConvTranspose, PixelShuffle

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)

        out = self.clf_conv1(x1)
        out = self.clf_conv2(out)
        return out


def MS_ResUNet():
    model = RefineNet(Bottleneck, [3, 4, 3, 3])
    return model