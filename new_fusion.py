import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.PSF import GuidedFusion
from basicsr.archs.Agent_cross import DualCrossAgentAttention
# ==========================================
# Part 1: 基础组件 (Basic Blocks)
# ==========================================

class BBasicConv2d(nn.Module):
    """ 标准卷积块: Conv + BN + ReLU """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BBasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LayerNorm(nn.Module):
    """ 支持 [B, C, H, W] 的 LayerNorm """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ==========================================
# Part 2: 共享编码器 (Restormer Encoder)
# ==========================================

class MDTA(nn.Module):
    def __init__(self, dim, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, self.dim_head, -1)
        k = k.view(b, self.num_heads, self.dim_head, -1)
        v = v.view(b, self.num_heads, self.dim_head, -1)
        q, k = q.softmax(dim=-1), k.softmax(dim=-1)  # Softmax on Channel dim
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).view(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super(GDFN, self).__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x):
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SharedRestormerEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_blocks=2):
        super().__init__()
        self.shallow_extract = nn.Conv2d(in_channels, embed_dim, 3, 1, 1, bias=False)
        self.body = nn.Sequential(*[RestormerBlock(embed_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.shallow_extract(x)
        return self.body(x) + x


class SDFM(nn.Module):
    """
    Structure Detail Feature Module - Upgraded
    将原来的普通融合模块替换为 mask-guided attention fusion
    """

    def __init__(self, in_C, out_C):
        super(SDFM, self).__init__()

        # 特征预处理
        self.RGBobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGBspr = BBasicConv2d(out_C, out_C, 3, 1, 1)

        self.Infobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Infspr = BBasicConv2d(out_C, out_C, 3, 1, 1)

        # 核心替换: 使用 UniversalFeatureFusion
        self.obj_fuse = UniversalFeatureFusion(
            in_channels=out_C,
            out_channels=out_C,
            embed_dim=out_C
        )

    def forward(self, rgb, depth, mask_logits=None):
        # 1. 特征提取分支
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))
        rgb_obj = self.RGBspr(rgb_sum)

        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))
        Inf_obj = self.Infspr(Inf_sum)

        # 2. 融合 (传入 mask_logits)
        out = self.obj_fuse(rgb_obj, Inf_obj, mask_logits)

        return out


# ==========================================
# Part 6: 完整整合模型 (Integrated Model)
# ==========================================
class SEBlock(nn.Module):
    """
    通道注意力裁判员 (Squeeze-and-Excitation)
    它的作用是：看一眼所有通道，觉得哪些通道是垃圾（比如由烂Mask产生的），就把它权重降为0。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UniversalFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=64):
        super().__init__()
        if in_channels != embed_dim:
            self.proj_a = nn.Conv2d(in_channels, embed_dim, 1)
            self.proj_b = nn.Conv2d(in_channels, embed_dim, 1)
        else:
            self.proj_a = nn.Identity()
            self.proj_b = nn.Identity()
        self.fg_stream = DualCrossAgentAttention(embed_dim)
        self.fusion=GuidedFusion(out_channels)

    def forward(self, A, B, mask_logits=None):
        """
        A, B: [B, C, H, W] -> 输入特征
        mask_logits: [B, 2, H, W] -> SAM 的输出 (还没 softmax)
                     或者 [B, 1, H, W] -> 已经是 mask
        """
        Bsz, _, H, W = A.shape

        # --- 1. 预处理与对齐 ---
        curr_A = self.proj_a(A)  # [B, embed_dim, H, W]
        curr_B = self.proj_b(B)
        out1, out2 = self.fg_stream(curr_A, curr_B,mask_logits=mask_logits)
        # out=torch.cat([out1,out2],dim=1)
        # out=out2+out1
        # out = out + self.refine(out)
        out=self.fusion(out1,out2)
        # 通道筛选
        # out = self.channel_gate(out)

        return out

# ==========================================
# Part 7: 测试代码 (Verification)
# ==========================================

if __name__ == "__main__":
    print("=== 开始测试完整融合模块 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 定义模型
    model = UniversalFeatureFusion(in_channels=1, embed_dim=64).to(device)

    # 2. 创建假数据 (Batch=2, Channel=1, Size=128x128)
    ir = torch.randn(2, 1, 128, 128).to(device)
    vi = torch.randn(2, 1, 128, 128).to(device)

    # 3. 前向传播
    fused_img, mask_logits = model(ir, vi)

    print(f"输入尺寸: {ir.shape}")
    print(f"融合输出尺寸: {fused_img.shape}")
    print(f"掩码输出尺寸: {mask_logits.shape}")

    # 4. 模拟 Loss 计算
    # 假设我们有 SAM 生成的 Ground Truth Mask
    sam_mask_gt = torch.randint(0, 2, (2, 128, 128)).long().to(device)

    # Seg Loss (模仿 SAM)
    loss_seg = F.cross_entropy(mask_logits, sam_mask_gt)
    # Fusion Loss (模仿图像)
    loss_fus = F.l1_loss(fused_img, ir)

    total_loss = loss_fus + 0.1 * loss_seg

    # 5. 反向传播
    try:
        total_loss.backward()
        print("反向传播: 成功 (梯度连通)")
    except Exception as e:
        print(f"反向传播: 失败 {e}")

    print("=== 测试通过 ===")