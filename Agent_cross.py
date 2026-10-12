# cross_agent_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DepthwiseConv2d(nn.Module):
    def __init__(self, dim, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel, padding=kernel//2, groups=dim)
    def forward(self, x):
        return self.conv(x)  # x: B,C,H,W

# class AgentPool(nn.Module):
#     """
#     Pool tokens -> agent tokens via adaptive avg pool.
#     agent_num: number of agents (M). Will be laid out in sqrt grid if possible.
#     """
#     def __init__(self, in_dim, agent_num=49, mode='avg'):
#         super().__init__()
#         self.agent_num = agent_num
#         self.mode = mode
#         # small projection to agent embedding
#         self.proj = nn.Linear(in_dim, in_dim)
#     def forward(self, x, H, W):
#         # x: B, N, C  (N=H*W)
#         B, N, C = x.shape
#         assert N == H * W
#         x2 = x.transpose(1,2).reshape(B, C, H, W)
#         s = int(self.agent_num ** 0.5)
#         if s*s != self.agent_num:
#             # fallback: global avg then replicate
#             pooled = x2.mean(dim=(2,3), keepdim=True).reshape(B,1,C)
#             pooled = pooled.repeat(1, self.agent_num, 1)  # B, M, C
#         else:
#             pooled = F.adaptive_avg_pool2d(x2, (s, s))  # B,C,s,s
#             pooled = pooled.reshape(B, C, s*s).transpose(1,2)  # B, M, C
#         return self.proj(pooled)  # B, M, C
# class BiModalConvLocalAttention(nn.Module):
#     """
#     双模态拼接的卷积局部注意力（红外+可见光）
#     核心：输入拼接后的双模态特征，分别学习红外/可见光的局部权重，再融合
#     """
#
#     def __init__(self, channels, inter_channels=None,):
#         super().__init__()
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#         self.Recalibrate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2 * inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2 * channels),
#             nn.Sigmoid(),
#         )
#         self.channel_agg = nn.Sequential(
#             nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x_ir, x_vi):
#         """
#         x_ir: [B, C, H, W] 红外特征
#         x_vi: [B, C, H, W] 可见光特征
#         """
#         # 步骤1：拼接双模态特征（核心：让局部注意力感知两个模态）
#         x_bi = torch.cat([x_ir, x_vi], dim=1)  # [B, 2C, H, W
#         # 步骤3：双模态特征投影（2C→C，保留融合信息）
#         recal_w = self.Recalibrate(x_bi)
#         recal_input = recal_w * x_bi
#
#         # 步骤4：提取局部空间特征
#         spatial_feat = self.channel_agg(recal_input)  # [B, C, H, W]
#         local_w = self.local_att(spatial_feat)
#
#         return local_w
class BiModalConvLocalAttention(nn.Module):
    """
    双模态拼接的卷积局部注意力（红外+可见光）
    核心：输入拼接后的双模态特征，分别学习红外/可见光的局部权重，再融合
    """

    def __init__(self, channels, inter_channels=None, kernel_size=3, padding=1):
        super().__init__()
        # channels：单模态通道数 → 拼接后是2*channels
        self.channels = channels
        inter_channels = inter_channels or channels // 4

        # 1. 双模态特征投影（将拼接后的2C特征投影回C，保留双模态信息）
        self.bi_modal_proj = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # 2. 空间分支：提取双模态融合后的局部空间特征（3x3深度卷积）
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # 3. 双模态通道分支（核心：分别学习红外/可见光的局部权重）
        # 分支1：红外局部权重（从拼接特征中提取红外模态的局部信息）
        self.ir_channel_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding),
            nn.Conv2d(2 * channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        # 分支2：可见光局部权重（从拼接特征中提取可见光模态的局部信息）
        self.vi_channel_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding),
            nn.Conv2d(2 * channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # 4. 融合分支：整合双模态局部特征
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # 可学习的模态平衡参数（自适应调整红外/可见光局部权重）
        self.alpha_modal = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_ir, x_vi):
        """
        x_ir: [B, C, H, W] 红外特征
        x_vi: [B, C, H, W] 可见光特征
        """
        # 步骤1：拼接双模态特征（核心：让局部注意力感知两个模态）
        x_bi = torch.cat([x_ir, x_vi], dim=1)  # [B, 2C, H, W]

        # 步骤2：分别生成红外/可见光的局部通道权重
        ir_weight = self.ir_channel_branch(x_bi)  # [B, C, H, W] 红外局部权重
        vi_weight = self.vi_channel_branch(x_bi)  # [B, C, H, W] 可见光局部权重

        # 步骤3：双模态特征投影（2C→C，保留融合信息）
        x_proj = self.bi_modal_proj(x_bi)  # [B, C, H, W]

        # 步骤4：提取局部空间特征
        spatial_feat = self.spatial_branch(x_proj)  # [B, C, H, W]

        # 步骤5：双模态自适应加权（红外局部特征 + 可见光局部特征）
        # 红外：spatial_feat × 红外权重 × 模态平衡参数
        # 可见光：spatial_feat × 可见光权重 × (1-模态平衡参数)
        weighted_feat = spatial_feat * (
                self.alpha_modal * ir_weight + (1 - self.alpha_modal) * vi_weight
        )

        # 步骤6：残差融合（保留原始双模态投影特征）
        out = self.fusion(weighted_feat + x_proj)

        return out

class AgentPool(nn.Module):
    """
    Improved AgentPool for IVIF
    Features:
    1. Dual Pooling (Avg + Max) -> Preserves IR thermal targets better.
    2. Positional Encoding via DWConv before pooling.
    3. Robust handling of agent shapes.
    修改后：输出形状从 (B, agent_num, in_dim) → (B, in_dim, grid_size, grid_size)
    """

    def __init__(self, in_dim, agent_num=49):
        super().__init__()
        self.agent_num = agent_num
        self.grid_size = int(agent_num ** 0.5)

        # 强制检查平方数，避免逻辑崩塌
        assert self.grid_size * self.grid_size == agent_num, \
            f"Agent number must be a square number (e.g., 49, 64), got {agent_num}."
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.GELU()
        )

        # 2. 投影层：输入是 Avg+Max 的拼接，所以输入维度翻倍
        self.proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, x):
        """
        x: 输入形状 (B, C, H, W) → 示例：(8,16,H,W)
        return: 输出形状 (B, C, grid_size, grid_size) → 示例：(8,16,7,7)
        """
        # 保留原始批次维度
        B, C, H, W = x.shape

        # 1. 原有预处理+双池化逻辑
        x_spatial = x
        x_spatial = x_spatial + self.pre_conv(x_spatial)  # (B,C,H,W)
        x_avg = F.adaptive_avg_pool2d(x_spatial, (self.grid_size, self.grid_size))  # (B,C,7,7)
        x_max = F.adaptive_max_pool2d(x_spatial, (self.grid_size, self.grid_size))  # (B,C,7,7)
        x_cat = torch.cat([x_avg, x_max], dim=1)  # (B, 2*C, 7,7) → 示例：(8,32,7,7)

        # 2. 原有投影逻辑（调整维度顺序适配Linear层）
        # 拉平空间维度：(B,2C,7,7) → (B,2C,49) → (B,49,2C)
        agents = x_cat.flatten(2).transpose(1, 2)  # (B,49,2C) → 示例：(8,49,32)
        out = self.proj(agents)  # (B,49,C) → 示例：(8,49,16)

        # ========== 核心修改：维度转换 ==========
        # 步骤1：调换维度顺序 (B,49,C) → (B,C,49)
        out = out.transpose(1, 2)  # (8,16,49)
        # 步骤2：还原空间维度 (B,C,49) → (B,C,7,7)
        out = out.reshape(B, C, self.grid_size, self.grid_size)  # (8,16,7,7)

        return out
class MaskGuidedAgentCrossAttention(nn.Module):
    """
    Simplified MG-ACA (移除位置偏置，直接广播Agent_v到Q)
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        agent_num=49,
        mask_channels=1,
        attn_drop=0.0,
        proj_drop=0.0,
        use_pe=True
    ):
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.agent_num = agent_num
        self.pool_size = int(agent_num ** 0.5)

        # QKV投影
        self.q_A = nn.Linear(dim, dim)
        self.k_B = nn.Linear(dim, dim)
        self.v_B = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Agent池化
        self.agent_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        # 掩码编码器
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(mask_channels, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.dwc = BiModalConvLocalAttention(dim,dim)
        # 可学习平衡参数
        self.alpha_global = nn.Parameter(torch.tensor(0.0))
        self.alpha_local = nn.Parameter(torch.tensor(0.0))
        self.align_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, stride=1, padding=0),  # 通道对齐
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim),  # 空间对齐
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.dynamic_weight = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, stride=1, padding=0),  # 拼接全局+局部特征
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1, stride=1, padding=0),  # 输出2个权重（全局/局部）
            nn.Softmax(dim=1)  # 逐像素归一化，保证权重和为1
        )

        self.peg = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) if use_pe else None

    def forward(self, A, B, mask=None):
        """
        A: Infrared Feature [B, C, H, W] (Query Source)
        B: Visible Feature [B, C, H, W] (Key/Value Source)
        mask: [B, H, W] or [B, 1, H, W]
        """
        Bsz, C, H, W = A.shape
        N = H * W
        M = self.agent_num
        # === 2. 掩码引导 Agent 生成 ===
        A_agent_src = A
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(H, W), mode='nearest')
            mask_weight = self.mask_encoder(mask)
            # 使用 mask 增强 A，使 agent 更关注显著区域
            A_agent_src = A * (1.0 +0.5*mask_weight)
            # A_agent_src = A * (1.0 +0.5*mask_weight)

        agents = self.agent_pool(A_agent_src)  # [B, C, pool_h, pool_w]
        # 给 Agent 加入位置信息 (PEG)
        if self.peg is not None:
            agents = agents + self.peg(agents)
        agent_tokens = agents.flatten(2).transpose(1, 2)  # [B, M, C]

        # === 3. 准备 Attention 变量 ===
        A_seq = A.flatten(2).transpose(1, 2)
        q = self.q_A(A_seq).view(Bsz, N, self.num_heads, self.head_dim).transpose(1, 2)

        B_seq = B.flatten(2).transpose(1, 2)
        k = self.k_B(B_seq).view(Bsz, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_B(B_seq).view(Bsz, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_spatial = v.transpose(1, 2).reshape(Bsz, N, C).transpose(1, 2).reshape(Bsz, C, H, W)
        q_spatial = q.transpose(1, 2).reshape(Bsz, N, C).transpose(1, 2).reshape(Bsz, C, H, W)
        local_out = self.dwc(q_spatial,v_spatial).flatten(2).transpose(1, 2)  # [B, N, C]
        agent_tokens_head = agent_tokens.view(Bsz, M, self.num_heads, self.head_dim).transpose(1, 2)

        # === 4. Agent Cross Attention ===

        # Step A: Agent 聚合全局信息 (Agents 看着 全局 B)
        # B, h, M, N
        attn_ab = (agent_tokens_head * self.scale) @ k.transpose(-2, -1)
        attn_ab = F.softmax(attn_ab, dim=-1)
        attn_ab = self.attn_drop(attn_ab)

        # agent_v 包含了由 Agent 聚合的 B 的全局语义
        agent_v = attn_ab @ v  # [B, h, M, d]

        # Step B: 广播回 Q (全局 A 看着 Agents)
        # 注意：这里用 agent_tokens (来自A) 作为 Key，保证了空间结构与 A 一致
        q_attn = (q * self.scale) @ agent_tokens_head.transpose(-2, -1)
        q_attn = F.softmax(q_attn, dim=-1)
        q_attn = self.attn_drop(q_attn)

        global_out = q_attn @ agent_v  # [B, h, N, d]

        global_out = global_out.transpose(1, 2).reshape(Bsz, N, C)
        global_out = self.proj(global_out)
        global_out = self.proj_drop(global_out)

        # === 5. 最终融合 ===
        # Output = Infrared Base + Weighted Global Fusion + Weighted Local Fusion
        out = A_seq +torch.sigmoid(self.alpha_global) * global_out + torch.sigmoid(self.alpha_local) * local_out
        # out = A_seq + torch.sigmoid(self.alpha_global) * global_out
        out = out.transpose(1, 2).reshape(Bsz, C, H, W)
        return out
    # def forward(self, A, B, mask=None):
    #     Bsz, C, H, W = A.shape
    #     N = H * W
    #     M = self.agent_num
    #
    #     # 1. 掩码引导的代理生成
    #     A_agent_src = A
    #     if mask is not None:
    #         if mask.dim() == 3:
    #             mask = mask.unsqueeze(1)
    #         mask = F.interpolate(mask, size=(H, W), mode='nearest')
    #         mask_weight = self.mask_encoder(mask)
    #         A_agent_src = A * (1.0 + 0.5 * mask_weight)
    #
    #     agents = self.agent_pool(A_agent_src)
    #     agent_tokens = agents.flatten(2).transpose(1, 2)  # [B, M, C]
    #
    #     # 2. 特征重塑为多头
    #     A_seq = A.flatten(2).transpose(1, 2)
    #     q = self.q_A(A_seq).view(Bsz, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
    #
    #     B_seq = B.flatten(2).transpose(1, 2)
    #     k = self.k_B(B_seq).view(Bsz, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
    #     v = self.v_B(B_seq).view(Bsz, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
    #
    #     agent_tokens = agent_tokens.view(Bsz, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]
    #
    #     # 3. Agent ↔ B Cross Attention (无位置偏置)
    #     attn_ab = (agent_tokens * self.scale) @ k.transpose(-2, -1)
    #     attn_ab = F.softmax(attn_ab, dim=-1)
    #     attn_ab = self.attn_drop(attn_ab)
    #     agent_v = attn_ab @ v  # [B, h, M, d]
    #     # Q=agent_tokens [64,8,49,8], K=k [64,8,16384,8] → 反向注意力：B的V加权后池化到Agent尺寸
    #     # 步骤1：用正向注意力权重（已归一化）直接加权B的V，得到B的全局特征
    #     b_global_v = attn_ab.transpose(-2, -1) @ agent_tokens  # [B,h,N,d] = [64,8,16384,8]
    #     # 步骤2：将B的全局特征重塑为空间特征，池化到Agent尺寸
    #     b_global_v = b_global_v.transpose(1, 2).reshape(Bsz, N, C)  # [B,N,C] = [64,16384,64]
    #     b_global_v = b_global_v.transpose(1, 2).reshape(Bsz, C, H, W)  # [B,C,H,W] = [64,64,128,128]
    #     b_global_agent = self.agent_pool(b_global_v)  # [B,C,7,7] = [64,64,7,7]
    #     # 步骤3：重塑为多头格式，匹配agent_v维度
    #     b_global_agent = b_global_agent.flatten(2).transpose(1, 2)  # [B,M,C] = [64,49,64]
    #     b_global_agent = b_global_agent.view(Bsz, M, self.num_heads, self.head_dim).transpose(1,
    #                                                                                           2)  # [B,h,M,d] = [64,8,49,8]
    #     agent_v = agent_v + b_global_agent * 0.5
    #     q_attn = (q * self.scale) @ agent_tokens.transpose(-2, -1)
    #     q_attn = F.softmax(q_attn, dim=-1)
    #     q_attn = self.attn_drop(q_attn)
    #     global_out = q_attn @ agent_v  # [B, h, N, d]
    #     global_out = global_out.transpose(1, 2).reshape(Bsz, N, C)
    #     global_out = self.proj(global_out)
    #     global_out = self.proj_drop(global_out)
    #     global_out_spatial = global_out.transpose(1, 2).reshape(Bsz, C, H, W)
    #     global_out_spatial = self.align_conv(global_out_spatial)  # 空间对齐
    #     # 2. 全局特征归一化
    #     # global_out_spatial = self.channel_norm(global_out_spatial)
    #     # global_out_spatial = torch.sigmoid(global_out_spatial)
    #
    #     # 3. 局部特征计算+掩码引导+归一化
    #     local_spatial = self.local_att(A, B)
    #     # local_spatial = self.channel_norm(local_spatial)
    #     # local_spatial = torch.sigmoid(local_spatial)
    #
    #     # 4. 动态权重融合
    #     weight_feat = torch.cat([global_out_spatial, local_spatial], dim=1)
    #     dynamic_weights = self.dynamic_weight(weight_feat)
    #     alpha_g = dynamic_weights[:, 0:1, :, :]
    #     alpha_l = dynamic_weights[:, 1:2, :, :]
    #     out_spatial = A + alpha_g * global_out_spatial + alpha_l * local_spatial
    #     # out = out_spatial.flatten(2).transpose(1, 2).reshape(Bsz, C, H, W)
    #
    #     return out_spatial


# class MaskGuidedAgentCrossAttention(nn.Module):
#     """
#     Cross-Agent Attention layer (single-direction): agents from modality A aggregate modality B,
#     then broadcast back to modality A queries.
#
#     Steps:
#       1) Compute agent tokens A = pool(Q_A)
#       2) Agent aggregation: agent_attn = softmax( (A W_q) (K_B W_k)^T ) -> agent_v = agent_attn @ (V_B W_v)
#       3) Broadcast: q_attn = softmax( (Q_A W_q2) agent_v^T ) -> out = q_attn @ agent_v
#     Multi-head implemented by splitting channels.
#     """
#     def __init__(self, dim, agent_num=49, num_heads=8, use_dwc=True, head_dim=None):
#         super().__init__()
#         self.dim = dim
#         self.agent_num = agent_num
#         self.num_heads = num_heads
#         if head_dim is None:
#             assert dim % num_heads == 0, "dim must be divisible by num_heads"
#             self.head_dim = dim // num_heads
#         else:
#             self.head_dim = head_dim
#         self.scale = (self.head_dim) ** -0.5
#
#         # projections
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#         self.pool = AgentPool(dim, agent_num=agent_num)
#         self.use_dwc = use_dwc
#         if use_dwc:
#             self.dwc = DepthwiseConv2d(dim, kernel=3)
#
#     def forward(self, Q_A, K_B, V_B, H, W):
#         """
#         Q_A: B, N, C  (queries from modality A)
#         K_B, V_B: B, N, C  (keys/values from modality B) -- assumed same spatial grid
#         H, W: spatial dims
#         returns: out_A: B, N, C (A influenced by B via agents)
#         """
#         B, N, C = Q_A.shape
#         A_sp = Q_A.permute(0,2,1).reshape(B, C, H, W)
#         # projections
#         qA = self.q_proj(Q_A)  # B,N,C
#         kB = self.k_proj(K_B)
#         vB = self.v_proj(V_B)
#
#         # form agent tokens from qA (modality A)
#         agents = self.pool(A_sp, H, W)  # B, M, C
#         # reshape for multi-head
#         # shape -> (B, heads, seq, head_dim)
#         def split_heads(x):
#             B, L, C = x.shape
#             x = x.view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
#             return x  # B, H, L, D
#
#         qA_h = split_heads(qA)    # B, h, N, d
#         kB_h = split_heads(kB)    # B, h, N, d
#         vB_h = split_heads(vB)    # B, h, N, d
#         agents_h = agents.view(B, self.agent_num, self.num_heads, self.head_dim).permute(0,2,1,3)  # B,h,M,d
#
#         # Agent aggregation: agents attend to K_B/V_B
#         # attn shape: B, h, M, N
#         agent_attn_logits = torch.matmul(agents_h * self.scale, kB_h.transpose(-2,-1))  # B,h,M,N
#         agent_attn = torch.softmax(agent_attn_logits, dim=-1)
#         agent_v = torch.matmul(agent_attn, vB_h)  # B,h,M,d
#
#         # Broadcast: Q_A attends to agent_v
#         # q_attn logits: B,h,N,M
#         q_attn_logits = torch.matmul(qA_h * self.scale, agent_v.transpose(-2,-1))
#         q_attn = torch.softmax(q_attn_logits, dim=-1)
#         out_h = torch.matmul(q_attn, agent_v)  # B,h,N,d
#
#         # merge heads
#         out = out_h.permute(0,2,1,3).reshape(B, N, C)  # B,N,C
#         out = self.out_proj(out)  # final linear
#
#         # optional dwconv residual from V_B to keep local texture (reshape vB to spatial)
#         if self.use_dwc:
#             # v_sp = vB.permute(0,2,1).reshape(B, C, H, W)  # B,C,H,W
#             # dw = self.dwc(v_sp).reshape(B, C, -1).permute(0,2,1)  # B,N,C
#             # out = out + dw * 0.2
#         out = out.transpose(1, 2).reshape(B, C, H, W)
#         return out  # B,N,C

class DualCrossAgentAttention(nn.Module):
    """
    Full Dual Cross-Agent Attention for two modalities (A,B). It performs:
       A_agents <- pool(A); A_agg_from_B = agents_A aggregate K_B/V_B; A_out = broadcast(A,Q_A, A_agg_from_B)
       and symmetric for B

    Returns updated features for both modalities.
    Supports gating fusion after if desired.
    """
    def __init__(self, dim, agent_num=49, num_heads=8, gate_mode='none'):
        """
        gate_mode: 'none' | 'channel' | 'spatial'  (applied when you want fused single output)
        """
        super().__init__()
        self.dim = dim
        self.agent_num = agent_num
        self.num_heads = num_heads
        self.layer_A = MaskGuidedAgentCrossAttention(dim, agent_num=agent_num, num_heads=num_heads)
        self.layer_B = MaskGuidedAgentCrossAttention(dim, agent_num=agent_num, num_heads=num_heads)
        # gating to merge A_out and B_out into single fused feature (optional)
        self.gate_mode = gate_mode
        if gate_mode == 'channel':
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim//4),
                nn.ReLU(inplace=True),
                nn.Linear(dim//4, dim),
                nn.Sigmoid()
            )
        elif gate_mode == 'spatial':
            self.gate_sp = nn.Sequential(
                nn.Conv2d(dim, dim//4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim//4, 1, 1),
                nn.Sigmoid()
            )
        # final small fusion projector when wanting single output
        self.fuse_proj = nn.Linear(dim, dim)
        self.gamma_A = nn.Parameter(torch.tensor(0.3))
        self.gamma_B = nn.Parameter(torch.tensor(0.7))
    def forward(self, A, B,mask_logits=None, H=None, W=None):
        Bsz, C, H0, W0 = A.shape
        H = H or H0
        W = W or W0
        N = H*W
        A_t = A.reshape(Bsz, C, N).permute(0,2,1)  # B,N,C
        B_t = B.reshape(Bsz, C, N).permute(0,2,1)

        A_from_B = self.layer_A(A, B,mask_logits)  # B,N,C
        B_from_A = self.layer_B(B, A,mask_logits)  # B,N,C
        # A_B=A_from_B+B_from_A
        # A_B = A_B.reshape(Bsz, C, N).permute(0, 2, 1)#b.n,c
        # V=self.layer_A(A_B,B_t,B_t,H,W)
        # I=self.layer_A(A_B,A_t,A_t,H,W)
        # A_from_B1 = self.layer_A(A_from_B, B, mask_logits)
        # B_from_A1 = self.layer_B(B_from_A, A, mask_logits)
        # A_out = A_t + self.gamma_A * (A_from_B1 - A_t)
        # B_out = B_t + self.gamma_B * (B_from_A1 - B_t)
        #
        # A_out = A + A_from_B  # residual
        # B_out = B + B_from_A
        # A_sp = A_from_B.permute(0, 2, 1).reshape(B, C, H, W)
        # B_sp = B_from_A.permute(0, 2, 1).reshape(B, C, H, W)
        # reshape back
        # A_out_sp = A_from_B1.permute(0,2,1).reshape(Bsz, C, H, W)
        # B_out_sp = B_from_A1.permute(0,2,1).reshape(Bsz, C, H, W)

        if self.gate_mode == 'none':
            return A_from_B, B_from_A

        if self.gate_mode == 'channel':
            # compute gate per token per channel
            # gate_input: average
            gate_input = 0.5 * (A_out + B_out)  # B,N,C
            g = self.gate_mlp(gate_input)  # B,N,C (sigmoid)
            fused_tok = g * A_out + (1 - g) * B_out
            fused = fused_tok.permute(0,2,1).reshape(Bsz, C, H, W)
            fused = self.fuse_proj(fused_tok).permute(0,2,1).reshape(Bsz, C, H, W)
            return fused

#
# class LocalSpatialInteraction(nn.Module):
#     """
#     用于底层 (Layer1) 的局部空间交互模块。
#     1. 使用 7x7 大核卷积感知局部纹理。
#     2. 保持双流输出 (Dual-stream Output)，以便后续连接 Channel Interaction。
#     """
#
#     def __init__(self, dim):
#         super().__init__()
#
#         # 1. 预处理：简单的特征变换
#         self.proj_vis = nn.Conv2d(dim, dim, 1)
#         self.proj_inf = nn.Conv2d(dim, dim, 1)
#
#         # 2. 空间感知：使用 7x7 DWConv (大感受野，参数少)
#         # 这一步模拟了 Attention 的"看周围"的能力
#         self.spatial_context = nn.Sequential(
#             nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, dim, 1)  # 通道融合
#         )
#
#         # 3. 门控生成器
#         # 输入: 拼接后的上下文 -> 输出: 互补的门控权重
#         self.gate_generator = nn.Sequential(
#             nn.Conv2d(dim * 2, dim, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, vis, inf):
#         # A. 提取空间上下文 (Context)
#         vis_ctx = self.spatial_context(vis)
#         inf_ctx = self.spatial_context(inf)
#
#         # B. 计算共享的互补门控 (Shared Complementary Gate)
#         # 网络通过对比两者的上下文，决定哪里该听可见光的，哪里该听红外的
#         cat_ctx = torch.cat([vis_ctx, inf_ctx], dim=1)
#         g = self.gate_generator(cat_ctx)  # [B, C, H, W], 值在 0~1
#
#         # C. 交叉注入 (Cross Injection)
#         # 这里的逻辑是：
#         # 可见光特征 = 原始可见光 + (红外信息 * 互补系数)
#         # 红外特征   = 原始红外   + (可见光信息 * 互补系数)
#         # 这样既保留了原始特征，又把对方有用的信息"借"了过来
#
#         vis_out = vis + self.proj_vis(vis_ctx * g)  # 注入红外觉得重要的区域
#         inf_out = inf + self.proj_inf(inf_ctx * (1 - g))  # 注入可见光觉得重要的区域
#
#         return vis_out, inf_out


class LKSFM(nn.Module):
    '''
    Large Kernel Spatial Fusion Module (LKSFM)
    结合了 SDFM 的稳定性和 Large Kernel 的纹理感知能力
    '''

    def __init__(self, channels=64):
        super(LKSFM, self).__init__()

        # 1. Recalibrate (预校准): 类似 SE-Block，先从通道上对齐
        # 保持你原有的逻辑，这部分对于抑制噪声很有用
        self.recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2 * channels, 1),
            nn.Sigmoid()
        )

        # 2. 特征压缩
        self.reduce = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 3. === 关键修改: 大核空间注意力 (Large Kernel Spatial Attention) ===
        # 替代原有的 1x1 local_att
        # 使用 7x7 DWConv 来感知纹理和边缘
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),  # 融合通道信息
            nn.BatchNorm2d(channels)
        )

        # 4. 全局通道注意力 (保持不变)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1: Vis, x2: Inf
        input_cat = torch.cat([x1, x2], dim=1)

        # Step 1: 通道预校准 (Recalibration)
        weights = self.recalibrate(input_cat)
        x1_w, x2_w = torch.split(weights, x1.shape[1], dim=1)
        x1 = x1 * (1 + x1_w)  # 这里的残差连接 (1 + w) 比单纯乘法更稳
        x2 = x2 * (1 + x2_w)

        # Step 2: 生成注意力图
        feat = self.reduce(torch.cat([x1, x2], dim=1))

        # 空间看纹理 (7x7)
        att_spatial = self.spatial_att(feat)
        # 通道看全局 (1x1)
        att_channel = self.channel_att(feat)

        # 结合
        attn_map = self.sigmoid(att_spatial + att_channel)

        # Step 3: 软融合 (Soft Fusion)
        # 这种互补式融合 (w * x1 + (1-w) * x2) 对于底层特征非常自然
        out = attn_map * x1 + (1 - attn_map) * x2

        return out
# ----------------- Example usage (smoke test) -----------------
if __name__ == "__main__":
    # small smoke test
    B, C, H, W = 8, 16, 1024, 650  # try small first
    model = DualCrossAgentAttention(dim=C, agent_num=16, num_heads=8, use_dwc=True, gate_mode='channel')
    A = torch.randn(B, C, H, W)
    Bv = torch.randn(B, C, H, W)
    A_out, B_out = model(A, Bv, H, W)
    print("A_out:", A_out.shape, "B_out:", B_out.shape)  # expect B,C,H,W
