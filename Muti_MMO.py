import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================================
# 跨尺度配准子模块（上一个版本）
# ==========================================================
class GlobalAlignmentGate(nn.Module):
    """
    Global consistency estimation.
    Decide whether local geometric alignment should be activated.
    """
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Sigmoid()
        )

    def forward(self, moving_feat, fixed_feat):
        # Global average pooling
        g_m = F.adaptive_avg_pool2d(moving_feat, 1).flatten(1)
        g_f = F.adaptive_avg_pool2d(fixed_feat, 1).flatten(1)
        w = self.mlp(torch.cat([g_m, g_f], dim=1))  # [B, 1]
        return w
class CrossScaleDeformableBlock(nn.Module):
    """
    Global-guided Residual Local Deformable Alignment

    - Local window-based attention for correspondence
    - Deformable warping for local alignment
    - Global gate to control whether alignment is necessary
    - Residual formulation to avoid over-alignment
    """

    def __init__(
        self,
        channels=64,
        window_size=16,
        max_disp=8.0
    ):
        super().__init__()

        self.window_size = window_size
        self.max_disp = max_disp

        # QKV projection
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)

        # Local flow regression
        self.reg_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1)
        )

        # Flow refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1)
        )

        # Global gate
        self.global_gate = GlobalAlignmentGate(channels)

    # -------------------------------------------------------
    # Local window-based attention (lightweight, no N^2)
    # -------------------------------------------------------
    def local_attention(self, q, k, v):
        B, C, H, W = q.shape
        ws = self.window_size
        out = torch.zeros_like(v)

        for i in range(0, H, ws):
            for j in range(0, W, ws):
                q_patch = q[:, :, i:i + ws, j:j + ws]
                k_patch = k[:, :, i:i + ws, j:j + ws]
                v_patch = v[:, :, i:i + ws, j:j + ws]

                if q_patch.numel() == 0:
                    continue

                Bp, Cp, Hp, Wp = q_patch.shape
                qf = q_patch.reshape(Bp, Cp, -1)
                kf = k_patch.reshape(Bp, Cp, -1)
                vf = v_patch.reshape(Bp, Cp, -1)

                attn = torch.bmm(qf.transpose(1, 2), kf) / math.sqrt(Cp)
                attn = torch.softmax(attn, dim=-1)
                agg = torch.bmm(attn, vf.transpose(1, 2)).transpose(1, 2)
                agg = agg.reshape(Bp, Cp, Hp, Wp)

                out[:, :, i:i + Hp, j:j + Wp] = agg

        return out

    def warp(self, feat, flow):
        B, C, H, W = feat.shape
        ys = torch.linspace(-1, 1, H, device=feat.device)
        xs = torch.linspace(-1, 1, W, device=feat.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0] = flow[:, 0] / (W / 2.0)
        flow_norm[:, 1] = flow[:, 1] / (H / 2.0)
        flow_norm = flow_norm.permute(0, 2, 3, 1)

        grid = base_grid + flow_norm
        return F.grid_sample(
            feat, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
    def forward(self, moving_feat, fixed_feat):
        """
        moving_feat: feature to be aligned (source)
        fixed_feat : reference feature
        """

        B, C, Hm, Wm = moving_feat.shape
        Hf, Wf = fixed_feat.shape[-2:]

        # 1️⃣ Scale alignment (engineering necessity, not semantic resize)
        fixed_up = F.interpolate(
            fixed_feat, size=(Hm, Wm),
            mode='bilinear', align_corners=False
        )
        q = self.to_q(fixed_up)
        k = self.to_k(moving_feat)
        v = self.to_v(moving_feat)
        agg = self.local_attention(q, k, v)
        x = agg + fixed_up
        flow = self.reg_net(x)
        flow_ref = self.refine(torch.cat([x, flow], dim=1))
        flow = flow + flow_ref
        flow = torch.tanh(flow) * self.max_disp
        aligned = self.warp(moving_feat, flow)
        w = self.global_gate(moving_feat, fixed_feat)
        w = w.view(B, 1, 1, 1)
        out = moving_feat + w * (aligned - moving_feat)

        return out
# class CrossScaleDeformableBlock(nn.Module):
#     def __init__(self, channels=16, scale_factor=2, window_size=16):
#         super(CrossScaleDeformableBlock, self).__init__()
#         self.scale_factor = scale_factor
#         self.window_size = window_size
#
#         self.to_q = nn.Conv2d(channels, channels, 1)
#         self.to_k = nn.Conv2d(channels, channels, 1)
#         self.to_v = nn.Conv2d(channels, channels, 1)
#
#         self.reg_net = nn.Sequential(
#             nn.Conv2d(channels, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 2, 3, padding=1)
#         )
#
#         self.refine = nn.Sequential(
#             nn.Conv2d(channels + 2, channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 2, 3, padding=1)
#         )
#
#     def local_attention(self, q, k, v, window_size=16):
#         B, C, H, W = q.shape
#         out = torch.zeros_like(v)
#         for i in range(0, H, window_size):
#             for j in range(0, W, window_size):
#                 q_patch = q[:, :, i:i + window_size, j:j + window_size]
#                 k_patch = k[:, :, i:i + window_size, j:j + window_size]
#                 v_patch = v[:, :, i:i + window_size, j:j + window_size]
#
#                 qf = q_patch.reshape(B, C, -1)
#                 kf = k_patch.reshape(B, C, -1)
#                 vf = v_patch.reshape(B, C, -1)
#
#                 attn = torch.bmm(qf.permute(0, 2, 1), kf) / (C ** 0.5)
#                 attn = torch.softmax(attn, dim=-1)
#                 agg = torch.bmm(attn, vf.permute(0, 2, 1)).permute(0, 2, 1)
#                 agg = agg.reshape(B, C, *q_patch.shape[2:])
#                 out[:, :, i:i + window_size, j:j + window_size] = agg
#         return out
#
#     def warp(self, feat, flow):
#         B, C, H, W = feat.shape
#         ys, xs = torch.linspace(-1, 1, H, device=feat.device), torch.linspace(-1, 1, W, device=feat.device)
#         grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
#         base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
#         flow_norm = torch.cat([
#             flow[:, 0:1] / (W / 2.0),
#             flow[:, 1:2] / (H / 2.0)
#         ], dim=1).permute(0, 2, 3, 1)
#         grid = base_grid + flow_norm
#         return F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
#
#     def forward(self, moving_feat, fixed_feat):
#         B, C, H1, W1 = moving_feat.shape
#         H2, W2 = fixed_feat.shape[-2:]
#
#         # 1️⃣ 将 fixed_feat 上采样到 moving_feat 尺度
#         fixed_up = F.interpolate(fixed_feat, size=(H1, W1), mode='bilinear', align_corners=False)
#
#         # 2️⃣ 注意力计算
#         q = self.to_q(fixed_up)
#         k = self.to_k(moving_feat)
#         v = self.to_v(moving_feat)
#         agg = self.local_attention(q, k, v, self.window_size)
#
#         # 3️⃣ flow 预测 + refine
#         x = agg + fixed_up
#         flow = self.reg_net(x)
#         flow_ref = self.refine(torch.cat([x, flow], dim=1))
#         flow = flow + flow_ref
#
#         # 4️⃣ warp 对齐
#         aligned = self.warp(moving_feat, flow)
#         return aligned
#
#
# # ==========================================================
# # 多尺度配准主模块
# # ==========================================================
# class MultiScaleRegistration(nn.Module):
#     def __init__(self, channels=16):
#         super(MultiScaleRegistration, self).__init__()
#
#         # 同尺度
#         self.reg1 = CrossScaleDeformableBlock(channels)
#         self.reg2 = CrossScaleDeformableBlock(channels)
#         self.reg3 = CrossScaleDeformableBlock(channels)
#
#         # 跨尺度
#         self.cross12 = CrossScaleDeformableBlock(channels)
#         self.cross23 = CrossScaleDeformableBlock(channels)
#
#         # 最终融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(channels * 5, channels, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, ir_feats, vi_feats):
#         """
#         ir_feats = [xf1_ir, xf2_ir, xf3_ir]
#         vi_feats = [xf1_vi, xf2_vi, xf3_vi]
#         """
#         xf1_ir, xf2_ir, xf3_ir = ir_feats
#         xf1_vi, xf2_vi, xf3_vi = vi_feats
#
#         # 同尺度配准
#         xf1_align = self.reg1(xf1_ir, xf1_vi)
#         xf2_align = self.reg2(xf2_ir, xf2_vi)
#         xf3_align = self.reg3(xf3_ir, xf3_vi)
#
#         # 跨尺度配准
#         xf12_align = self.cross12(xf1_ir, xf2_vi)
#         xf23_align = self.cross23(xf2_ir, xf3_vi)
#
#         # 上采样并融合
#         xf2_up = F.interpolate(xf2_align, size=xf1_align.shape[-2:], mode='bilinear', align_corners=False)
#         xf3_up = F.interpolate(xf3_align, size=xf1_align.shape[-2:], mode='bilinear', align_corners=False)
#         xf23_up = F.interpolate(xf23_align, size=xf1_align.shape[-2:], mode='bilinear', align_corners=False)
#         xf12_up = F.interpolate(xf12_align, size=xf1_align.shape[-2:], mode='bilinear', align_corners=False)
#
#         fused = torch.cat([xf1_align, xf2_up, xf3_up, xf12_up, xf23_up], dim=1)
#         fused = self.fusion(fused)
#
#         return fused

def test_multiscale_registration():
    # ------------------------
    # 模拟输入特征
    # ------------------------
    B = 2         # batch size
    C = 16        # 特征通道数
    H1, W1 = 1024, 768   # xf1 特征大小
    H2, W2 = 64, 64     # xf2 特征大小
    H3, W3 = 32, 32     # xf3 特征大小

    # 随机生成红外特征
    xf1_ir = torch.randn(B, C, H1, W1)
    xf2_ir = torch.randn(B, C, H2, W2)
    xf3_ir = torch.randn(B, C, H3, W3)

    # 随机生成可见光特征
    xf1_vi = torch.randn(B, C, H1, W1)
    xf2_vi = torch.randn(B, C, H2, W2)
    xf3_vi = torch.randn(B, C, H3, W3)

    ir_feats = [xf1_ir, xf2_ir, xf3_ir]
    vi_feats = [xf1_vi, xf2_vi, xf3_vi]

    # ------------------------
    # 初始化多尺度配准模块
    # ------------------------
    ms_reg = MultiScaleRegistration(channels=C)

    # ------------------------
    # 前向验证
    # ------------------------
    fused_feats = ms_reg(ir_feats, vi_feats)

    print("输出特征形状:", fused_feats.shape)

if __name__ == "__main__":
    test_multiscale_registration()
