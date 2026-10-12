from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.fuse import Fusion_module
from basicsr.archs.Agent_cross import DualCrossAgentAttention,LKSFM
from math import exp
from basicsr.archs.new_fusion import SDFM
from basicsr.archs.PSF import SIM
from .utils.CDC import cdcconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine1, IFP
import torch.nn.init as init
import os
import cv2
import numbers
from einops import rearrange
from .Muti_MMO import CrossScaleDeformableBlock
from .eucb import EUCB
import torchvision.models as models
import numpy
from .contmix import ContMixBlock
from timm.models.layers import DropPath
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3
class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        if p is None:
            # 如果卷积核大小为整数，则使用卷积核的一半作为自动填充值；
            # 若卷积核大小为列表，则对每个卷积核尺寸计算其一半作为对应的填充值。
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False) # 初始化二维卷积层
        self.bn = nn.BatchNorm2d(c2) # 初始化批归一化层
        self.act = nn.SiLU()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.bn(self.conv(x)) # V1 对输入张量应用卷积、批归一化
        # return self.act(self.bn(self.conv(x))) # V2 对输入张量应用卷积、批归一化以及激活函数
# class Star_Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, g=dim) # 深度可分离卷积层
#
#         self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1) # 在希望保留输入数据的空间结构时，可以用1x1卷积来代替全连接层。
#         self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1) # 在希望保留输入数据的空间结构时，可以用1x1卷积来代替全连接层。
#
#         self.g = ConvBN(mlp_ratio * dim, dim, 1) # Star
#         self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim) # 另一个深度可分离卷积层
#
#         self.act = nn.ReLU6() # 使用ReLU6作为激活函数
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # 随机丢弃路径以防止过拟合
#
#     def forward(self, x):
#         input = x # 保存原始输入
#
#         x = self.dwconv(x) # 应用第一个深度可分离卷积
#         x1, x2 = self.f1(x), self.f2(x) # 分别通过两个全连接层
#
#         x = self.act(x1) * x2 # 将经过激活的第一个输出与第二个输出相乘
#
#         x = self.g(x) # star Conv
#         x = self.dwconv2(x) # 轻量级的多阶通道重分配模块，可以自适应地重分配高维隐藏空间通道特征，允许模型自适应地调整每个通道的重要性，从而优化整个网络信息流动。
#
#         x = input + self.drop_path(x) # 添加原始输入，并可能随机丢弃部分路径
#         return x


class DualScaleDWConv(nn.Module):
    """
    创新点1: 双尺度深度卷积
    一支 3x3 负责保留可见光的边缘细节
    一支 7x7 (或5x5) 负责提取红外图的热目标轮廓
    """

    def __init__(self, dim):
        super().__init__()
        self.dw3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        # self.dw7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        self.dw5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        # 简单的融合权值，或者直接相加
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x3 = self.dw3x3(x)
        x7 = self.dw5x5(x)
        return self.norm(x3 + x7)  # 多尺度特征融合
class SRB(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = DualScaleDWConv(dim)

        # --- 宽展维度 ---
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.gate_spatial = nn.Conv2d(mlp_ratio * dim, mlp_ratio * dim, 3, 1, 1,
                                      groups=mlp_ratio * dim, bias=False)

        # Star 操作后的聚合
        self.g = ConvBN(mlp_ratio * dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

        self.act = nn.SiLU()  # 推荐 SiLU 替换 ReLU6
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)

    def forward(self, x):
        input = x

        # 1. 多尺度特征提取
        x = self.dwconv(x)

        # 2. 分支映射
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2

        # 4. 特征聚合
        x = self.g(x)
        x = self.dwconv2(x)

        # 5. 残差连接
        x = input + self.drop_path(x)
        return x
class DenseBlockMscale(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier'):
        super(DenseBlockMscale, self).__init__()
        self.ops = DenseBlock(channel_in, channel_out, init)
        self.fusepool = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc1 = nn.Sequential(nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fuse = nn.Conv2d(3*channel_out,channel_out,1,1,0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops(x1)
        x2 = self.ops(x2)
        x3 = self.ops(x3)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        xattw = self.fusepool(x1+x2+x3)
        xattw1 = self.fc1(xattw)
        xattw2 = self.fc2(xattw)
        xattw3 = self.fc3(xattw)
        # x = x1*xattw1+x2*xattw2+x3*xattw3
        x = self.fuse(torch.cat([x1*xattw1,x2*xattw2,x3*xattw3],1))

        return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlockMscale(channel_in, channel_out, init)
            else:
                return DenseBlockMscale(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None
    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out
    
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class highOrderInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(highOrderInteraction, self).__init__()
        self.low_fuse2 = SDFM(128, 64)
        self.low_fuse = SDFM(16, 16)
        self.low_fuse1 = SDFM(64, 32)
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(channelin * 3, channelin, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        self.SIM1 = SIM(norm_nc=32, label_nc=64, nhidden=32)
        self.SIM2 = SIM(norm_nc=16, label_nc=32, nhidden=32)
        # self.eucb=EUCB(32,channelout)
        # self.eucb1=EUCB(64,channelout,scale_factor=4)
        # self.MMO=CrossScaleDeformableBlock(channels=16,  window_size=16)
        # self.MMO1=CrossScaleDeformableBlock(channels=64,  window_size=16)
        # self.MMO2=CrossScaleDeformableBlock(channels=128, window_size=16)
        # self.postprocess = nn.Sequential(InvBlock(DenseBlock, 16,16),
        #                                  nn.Conv2d(16, 16, 1, 1, 0))
        # self.postprocess = nn.Sequential(InvBlock(DenseBlock, 3 * channelin, 3 * channelin // 2),
        #                                  nn.Conv2d(3 * channelin, channelin, 1, 1, 0))
    def forward(self, vis_y, inf,mask_logits, i, j):
        vis_y1,vis_y2,vis_y3=vis_y
        inf1,inf2,inf3=inf
        # inf1=self.MMO(inf1,vis_y1)
        # inf2=self.MMO1(inf2,vis_y2)
        # inf3=self.MMO2(inf3,vis_y3)
        out =  self.low_fuse(vis_y1, inf1,mask_logits=mask_logits)
        out1=   self.low_fuse1 (vis_y2, inf2,mask_logits=mask_logits)
        out2 =  self.low_fuse2(vis_y3, inf3,mask_logits=mask_logits)
        rec_f1 = self.SIM1(out1,out2, mask_logits)
        rec_f2 = self.SIM2(out, rec_f1,mask_logits)

        return rec_f2,vis_y1,inf

class LowPassBlur(nn.Module):
    """
    Fixed depthwise low-pass filter.
    Inspired by BlurPool / anti-aliased CNN:
    Richard Zhang, "Making Convolutional Networks Shift-Invariant Again", ICML 2019.
    """
    def __init__(self, channels, kernel_size=5):
        super().__init__()

        if kernel_size == 3:
            filt = torch.tensor([1., 2., 1.])
        elif kernel_size == 5:
            filt = torch.tensor([1., 4., 6., 4., 1.])
        elif kernel_size == 7:
            filt = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError("kernel_size should be 3, 5, or 7")

        kernel = filt[:, None] * filt[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)

        self.register_buffer("kernel", kernel)
        self.groups = channels
        self.pad = kernel_size // 2

    # def forward(self, x):
    #     return F.conv2d(
    #         x,
    #         self.kernel,
    #         stride=1,
    #         padding=self.pad,
    #         groups=self.groups
    #     )
    def forward(self, x):
        x=F.pad(x,[self.pad,self.pad,self.pad,self.pad],mode='reflect')
        return F.conv2d(x,self.kernel,stride=1,padding=0,groups=self.groups)
class EdgeBlock(nn.Module):
    def __init__(self, channelin, channelout):
        super(EdgeBlock, self).__init__()
        self.process = nn.Conv2d(channelin,channelout,3,1,1)
        self.Res = SRB(channelout)
        self.lowpass = LowPassBlur(channelout, kernel_size=7)
        # self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
        #     nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
        self.CDC = cdcconv(channelin, channelout)

    def forward(self,x):

        f = self.process(x)
        low_feat = self.lowpass(f)
        derb_out = self.Res(low_feat)
        x1=self.CDC(x)
        out = derb_out + x1

        return out, derb_out,x1

class RobustFeatureExtract(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(RobustFeatureExtract, self).__init__()
        # 128x128 -> 128x128
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            EdgeBlock(out_channels // 2, out_channels // 2)
        )
        self.channel_mapper1 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1  # 保持空间尺寸不变（128×128）
        )
        # self.stage2 = nn.Sequential(
        #     nn.Conv2d(out_channels // 2, out_channels, 3, 2, 1),  # Stride=2
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     EdgeBlock(out_channels, out_channels)
        # )

    def forward(self, x):
        x,x_low,x_high = self.stage1(x)  # [B, 16, 128, 128] -> 留着做 Skip Connection
        x_low1 = self.channel_mapper1(x)  # [B, 64, 128, 128] -> 喂给 restormer_block
        return x, x_low1,x_low,x_high
# class FeatureExtract(nn.Module):
#     def __init__(self, channelin, channelout):
#         super(FeatureExtract, self).__init__()
#         self.conv = nn.Conv2d(channelin,channelout,1,1,0)
#         self.block1 = EdgeBlock(channelout,channelout)
#         self.block2 = EdgeBlock(channelout, channelout)
#
#     def forward(self,x):
#         xf = self.conv(x)
#         xf1 = self.block1(xf)
#         x1 = self.block2(xf1)
#         return x1
class S2M(nn.Module):
    '''
    Scene Specific Mask
    '''
    def __init__(self, channels, r=4):
        super(S2M, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_block = nn.BatchNorm2d(channels)

    def forward(self, x):
        # spatial attention
        local_w = self.local_att(x) ## local attention
        ## channel attention
        global_w = self.global_att(x)
        mask = self.sigmoid(local_w * global_w)
        masked_feature = mask * x
        output = self.conv_block(masked_feature)
        return output
class SFP(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels, img=False):
        super(SFP, self).__init__()
        self.mask = S2M(channels[0])
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        x = self.mask(x)
        return (self.conv_block(x) + 1) / 2
@ARCH_REGISTRY.register()
class highOrderInteractionFusion(nn.Module):
    def __init__(self, vis_channels=1, inf_channels=1,
                 n_feat=16):
        super(highOrderInteractionFusion,self).__init__()
        self.vis = RobustFeatureExtract(vis_channels, 32)
        self.inf = RobustFeatureExtract(inf_channels, 32)
        self.num_resnet_layers = 34
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
        self.interaction1 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4
        self.pred_fusion = IFP([n_feat, 1])
        self.i = 0
        # self.vis_reg = SFP([n_feat, 1])
        # self.inf_reg = SFP([n_feat, 1])
        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 64, 64 // 2),
                                     nn.Conv2d(64, 32, 1, 1, 0))
    def forward(self, image_vis, image_ir,mask_logits):

        vis_y = image_vis[:,:1]
        inf = image_ir

        vis_h, vis_l,vis_low,vis_high = self.vis(vis_y)
        inf_h, inf_l,inf_low,inf_high = self.inf(inf)
        x = self.encoder_rgb_bn1(vis_l)  # (240, 320)
        x = self.encoder_rgb_relu(x)
        x_pool = self.encoder_rgb_maxpool(x)
        rgb1 = self.encoder_rgb_layer1(x_pool)
        rgb2 = self.encoder_rgb_layer2(rgb1)
        # (120, 160)
        y = self.encoder_thermal_bn1(inf_l)  # (120, 160)
        y=self.encoder_thermal_relu(y)
        y_pool=self.encoder_thermal_maxpool(y)
        thermal1=self.encoder_thermal_layer1(y_pool)# (60, 80)
        thermal2 = self.encoder_thermal_layer2(thermal1)  # (60, 80)
        vis_y=(vis_h,rgb1,rgb2)
        inf=(inf_h,thermal1,thermal2)
        vis_y_feat,vis_,inf_ = self.interaction1(vis_y, inf,mask_logits, self.i, j=1)
        # fused = self.postprocess(torch.cat([fused_deep, high_fuse], 1))
        # vis = self.vis_reg(vis_y_feat)
        # inf = self.inf_reg(vis_y_feat)
        fused1 = self.pred_fusion(vis_y_feat)
        self.i += 1

        return fused1,vis_,inf_,vis_low,vis_high,inf_low,inf_high

