import math

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from functools import partial
from timm.models.layers import trunc_normal_tf_
# from timm.layers.pool2d_same import GeneralizedMeanPooling

class DABBlock(nn.Module):
    def __init__(self, in_channels):
        """
        初始化 DAB 模块（通道注意力+空间注意力+像素注意力）
        :param in_channels: 输入特征图的通道数
        """
        super(DABBlock, self).__init__()

        # 通道注意力（CA）
        self.ca_global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.ca_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),  # 全连接层
            nn.Sigmoid()
        )

        # 空间注意力（SA）
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)  # 7x7 卷积提取空间特征
        self.sa_sigmoid = nn.Sigmoid()

        # 像素注意力（PA）：本质是逐点卷积（1x1卷积）
        self.pa_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.pa_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征图，形状为 (batch_size, in_channels, H, W)
        :return: 融合注意力后的特征图
        """
        b, c, h, w = x.size()

        # --------------------- 通道注意力（CA） ---------------------
        # 全局平均池化 -> [b, c, 1, 1]
        ca_pooled = self.ca_global_pool(x).view(b, c)
        # 全连接层生成通道权重 -> [b, c]
        ca_weight = self.ca_fc(ca_pooled).view(b, c, 1, 1)
        # 通道加权
        ca_out = x * ca_weight  # [b, c, h, w]

        # --------------------- 空间注意力（SA） ---------------------
        # 对通道维度求平均 -> [b, 1, h, w]
        sa_pooled = x.mean(dim=1, keepdim=True)
        # 7x7 卷积提取空间特征 -> [b, 1, h, w]
        sa_weight = self.sa_conv(sa_pooled)
        sa_weight = self.sa_sigmoid(sa_weight)
        # 空间加权
        sa_out = x * sa_weight  # [b, c, h, w]

        # --------------------- 像素注意力（PA） ---------------------
        # 1x1 卷积生成像素权重 -> [b, c, h, w]
        pa_weight = self.pa_conv(x)
        pa_weight = self.pa_sigmoid(pa_weight)
        pa_out = x * pa_weight  # [b, c, h, w]

        # --------------------- 融合三种注意力 ---------------------
        out = ca_out + sa_out + pa_out
        return out
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels,scale_factor=2, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            # 转置卷积：stride=scale_factor 实现上采样，padding确保尺寸精确
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=scale_factor,  # 核大小=缩放因子，保证尺寸对齐
                stride=scale_factor,  # 步长=缩放因子，实现上采样
                padding=0,
                groups=in_channels,  # 深度可分离转置卷积，保持轻量化
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True)
        )
        # self.up_dwc = nn.Sequential(
        #     nn.Upsample(scale_factor=scale_factor),
        #     nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
        #               padding=kernel_size // 2, groups=self.in_channels, bias=False),
        #     nn.BatchNorm2d(self.in_channels),
        #     act_layer(activation, inplace=True)
        # )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')
        self.dab = DABBlock(out_channels)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):

        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        # x = self.dab(x)
        return x