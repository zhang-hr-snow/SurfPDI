# src/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.utils import to_dense_batch

# src/layers.py 修改版

class CrossScaleInteractionModule(nn.Module):
    """
    纯特征驱动的跨尺度交互 (Feature-Based Cross-Scale Interaction)
    去除显式距离偏置，仅基于特征相似度进行筛选。
    """

    def __init__(self, residue_dim, point_dim, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.residue_dim = residue_dim
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        # 1. 映射层
        self.W_q = nn.Linear(residue_dim, hidden_dim)
        self.W_k = nn.Linear(point_dim, hidden_dim)
        self.W_v = nn.Linear(point_dim, hidden_dim)

        # (删除了 distance_proj)

        # 2. 输出层 (LayerNorm + Dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, res_feats, point_feats, point_to_residue, residue_batch):
        """
        注意：现在 forward 接口不再需要 res_coords 和 point_coords
        """

        # --- 1. Q, K, V 映射 ---
        Q = self.W_q(res_feats)  # (N_res, H)
        K = self.W_k(point_feats)  # (N_pts, H)
        V = self.W_v(point_feats)  # (N_pts, H)

        # --- 2. 广播 Q ---
        Q_expanded = Q[point_to_residue]  # (N_pts, H)

        # --- 3. 计算特征相似度分数 (Dot Product) ---
        # 纯粹看“残基需求”和“点特征”的匹配度
        score = (Q_expanded * K).sum(dim=-1, keepdim=True) * self.scale

        # 钳制数值防止 NaN (保留这个好习惯)
        score = torch.clamp(score, min=-10, max=10)

        # --- 4. 稳定的 Softmax ---
        out_dim = res_feats.shape[0]

        # 找最大值 (Max trick)
        score_max = torch.full((out_dim, 1), -1e9, device=score.device)
        score_max = score_max.scatter_reduce(0, point_to_residue.unsqueeze(1), score, reduce="amax", include_self=False)

        # exp
        score_stable = score - score_max[point_to_residue]
        exp_score = torch.exp(score_stable)

        # sum
        sum_exp = torch.zeros(out_dim, 1, device=score.device)
        sum_exp.index_add_(0, point_to_residue, exp_score)

        # norm
        attention_weights = exp_score / (sum_exp[point_to_residue] + 1e-6)

        # --- 5. 加权聚合 ---
        V_weighted = attention_weights * V
        aggregated = torch.zeros(out_dim, self.hidden_dim, device=res_feats.device)
        aggregated.index_add_(0, point_to_residue, V_weighted)

        # 后处理
        aggregated = self.dropout(self.norm(aggregated))

        # --- 6. 拼接输出 ---
        # 依然使用 Concat，保留最大信息量
        # 输出维度: residue_dim + hidden_dim
        final_out = torch.cat([res_feats, aggregated], dim=-1)

        return final_out

    def scatter_softmax(self, src, index, dim=0):
        """
        数值稳定的 Softmax 实现
        src: (N_pts, 1) - Attention Scores
        index: (N_pts,) - 归属的残基索引
        """
        # 1. 找到每个残基组内的最大值 (Numerical Stability Trick)
        # 这一步是为了防止 exp() 溢出
        out_dim = index.max().item() + 1

        # 初始化为很小的数
        src_max = torch.full((out_dim, src.shape[1]), -1e9, device=src.device)

        # 使用 scatter_reduce 找最大值 (PyTorch 1.12+ 支持)
        # 如果你的 PyTorch 版本老，可以用 index_reduce 或循环(慢)
        # 这里用一种兼容性写法：
        try:
            # 尝试使用 scatter_max (需要安装 torch_scatter)
            from torch_scatter import scatter_max
            src_max, _ = scatter_max(src, index, dim=dim, dim_size=out_dim)
        except ImportError:
            # 手写简易版 max (利用 scatter_reduce_ 如果 pytorch > 1.11)
            # 或者简单的 hack: 既然找不到库，我们假设数据没那么极端，
            # 但为了安全，我们还是要做这一步。
            # 如果实在没法高效求max，可以跳过减max步骤，但要加 clamp
            pass

            # 如果成功获取了 max，广播回去
        if src.shape == src_max[index].shape:
            src = src - src_max[index]  # 指数偏移，防止爆炸
        else:
            # 如果没有 torch_scatter，直接截断 src 防止过大
            src = torch.clamp(src, max=20)

            # 2. 正常计算 exp
        src_exp = torch.exp(src)

        # 3. 计算分母
        sum_exp = torch.zeros(out_dim, src.shape[1], device=src.device)
        sum_exp.index_add_(dim, index, src_exp)

        # 4. 广播分母并归一化
        sum_exp_expanded = sum_exp[index]

        # 加一个 epsilon 防止除零
        return src_exp / (sum_exp_expanded + 1e-10)

class FeatureFusionModule(nn.Module):
    """
    特征融合模块
    支持: Concat, Gated, Attention 等多种融合方式
    """

    def __init__(self, method='concat', feat1_dim=256, feat2_dim=256,
                 output_dim=None, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.method = method
        self.output_dim = output_dim or (feat1_dim + feat2_dim)

        if method == 'concat':
            self.fusion = self._ConcatFusion()
        elif method == 'gated':
            self.fusion = self._GatedFusion(feat1_dim, feat2_dim, self.output_dim, hidden_dim, dropout)
        elif method == 'attention':
            self.fusion = self._AttentionFusion(feat1_dim, feat2_dim, self.output_dim, hidden_dim, num_heads, dropout)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, feat1, feat2):
        return self.fusion(feat1, feat2)

    # --- 内部实现类 ---
    class _ConcatFusion(nn.Module):
        def forward(self, feat1, feat2):
            return torch.cat([feat1, feat2], dim=-1)

    class _GatedFusion(nn.Module):
        def __init__(self, d1, d2, out_d, h_d, drop):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(d1 + d2, h_d), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h_d, 2), nn.Softmax(dim=-1)
            )
            self.p1 = nn.Linear(d1, out_d // 2)
            self.p2 = nn.Linear(d2, out_d // 2)

        def forward(self, f1, f2):
            weights = self.gate_net(torch.cat([f1, f2], dim=-1))
            return weights[:, 0:1] * self.p1(f1) + weights[:, 1:2] * self.p2(f2)

    class _AttentionFusion(nn.Module):
        def __init__(self, d1, d2, out_d, h_d, heads, drop):
            super().__init__()
            self.heads = heads;
            self.hd = h_d // heads
            self.scale = 1.0 / math.sqrt(self.hd)
            self.q = nn.Linear(d1, h_d);
            self.k = nn.Linear(d2, h_d);
            self.v = nn.Linear(d2, h_d)
            self.out = nn.Linear(h_d, out_d)
            self.drop = nn.Dropout(drop)

        def forward(self, f1, f2):
            B = f1.size(0)
            # Reshape for multi-head attention (B, Heads, 1, HeadDim)
            q = self.q(f1).view(B, self.heads, 1, self.hd)
            k = self.k(f2).view(B, self.heads, 1, self.hd)
            v = self.v(f2).view(B, self.heads, 1, self.hd)

            attn = (q * k).sum(dim=-1, keepdim=True) * self.scale
            attn = F.softmax(attn, dim=1)
            attn = self.drop(attn)

            out = (attn * v).view(B, -1)
            return self.out(out)


def aggregate_points_to_residues(point_features, point_to_residue, residue_batch):
    """
    独立函数: 将表面点特征聚合到残基
    """
    n_residues = residue_batch.shape[0]
    # 保护性索引修正
    idx = torch.clamp(point_to_residue, 0, n_residues - 1)

    try:
        from torch_scatter import scatter_mean
        return scatter_mean(point_features, idx, dim=0, dim_size=n_residues)
    except ImportError:
        # Fallback implementation
        out = torch.zeros(n_residues, point_features.shape[1], device=point_features.device)
        counts = torch.zeros(n_residues, 1, device=point_features.device)
        out.index_add_(0, idx, point_features)
        counts.index_add_(0, idx, torch.ones_like(counts).expand_as(point_features[:, :1]))
        return out / counts.clamp(min=1.0)


def init_weights_xavier(module):
    """独立函数: 权重初始化"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


class SurfaceAttentionPooling(nn.Module):
    """
    交叉模态注意力池化：利用残基特征 Q 去 Key-Value 搜索表面点特征
    """

    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)

        self.scale = (hidden_dim ** 0.5)

    def forward(self, res_feats, point_feats, point_to_res):
        """
        res_feats: [L, query_dim] - 残基支路输出
        point_feats: [M, key_dim] - 表面点支路输出
        point_to_res: [M] - 映射索引
        """
        # 1. 映射为 Q, K, V
        Q = self.q_proj(res_feats)  # [L, H]
        K = self.k_proj(point_feats)  # [M, H]
        V = self.v_proj(point_feats)  # [M, H]

        # 2. 计算点对点的注意力得分
        # 获取每个表面点对应的残基 Query
        # q_expanded 形状变为 [M, H]
        q_expanded = Q[point_to_res]

        # 点积注意力 (Dot-product attention)
        # scores 形状 [M]
        scores = (q_expanded * K).sum(dim=-1) / self.scale

        # 3. 在残基内部进行 Softmax 归一化
        # 使用 scatter_softmax 确保每个残基所属的点权重之和为 1
        attn_weights = scatter_softmax(scores, point_to_res, dim=0)  # [M]

        # 4. 加权求和得到残基级的表面特征
        # weighted_v 形状 [M, H]
        weighted_v = V * attn_weights.unsqueeze(-1)

        # 聚合回残基: [L, H]
        from torch_scatter import scatter_sum
        res_surf_refined = scatter_sum(weighted_v, point_to_res, dim=0, dim_size=res_feats.size(0))

        return res_surf_refined


class SurfaceMeanPooling(nn.Module):
    """Uniformly pool mapped surface points as a non-attention fusion control."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        use_residual_gate: bool = False,
    ) -> None:
        super().__init__()
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.use_residual_gate = bool(use_residual_gate)
        self.residual_gate = nn.Sequential(
            nn.Linear(query_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        res_feats: torch.Tensor,
        point_feats: torch.Tensor,
        point_to_res: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.v_proj(point_feats)
        n_residues = res_feats.size(0)
        pooled = values.new_zeros((n_residues, values.size(-1)))
        pooled.index_add_(0, point_to_res, values)
        counts = values.new_zeros((n_residues, 1))
        counts.index_add_(0, point_to_res, values.new_ones((values.size(0), 1)))
        correction = self.norm(self.output_proj(pooled / counts.clamp_min(1.0)))
        if self.use_residual_gate:
            gate = torch.sigmoid(self.residual_gate(torch.cat([res_feats, correction], dim=-1)))
        else:
            gate = correction.new_ones((n_residues, 1))
        return correction, gate


class SurfaceToResidueMessage(nn.Module):
    """One-way messages from every mapped surface point to its residue."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        use_geometry: bool = False,
        num_rbf: int = 8,
        rbf_radius: float = 6.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_geometry = bool(use_geometry)
        geometry_dim = num_rbf + 3 if self.use_geometry else 0
        self.point_message = nn.Sequential(
            nn.Linear(key_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residue_projection = nn.Linear(query_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        centers = torch.linspace(0.0, rbf_radius, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 1.0 / max((rbf_radius / max(num_rbf - 1, 1)) ** 2, 1e-6)

    def forward(
        self,
        res_feats: torch.Tensor,
        point_feats: torch.Tensor,
        point_to_res: torch.Tensor,
        residue_pos: torch.Tensor,
        point_pos: torch.Tensor,
    ) -> torch.Tensor:
        message_input = point_feats
        if self.use_geometry:
            relative = point_pos - residue_pos[point_to_res]
            distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            direction = relative / distance
            rbf = torch.exp(-self.rbf_gamma * (distance - self.rbf_centers.view(1, -1)).square())
            message_input = torch.cat([point_feats, rbf, direction], dim=-1)

        messages = self.point_message(message_input)
        n_residues = res_feats.size(0)
        pooled = scatter_sum(messages, point_to_res, dim=0, dim_size=n_residues)
        counts = scatter_sum(
            messages.new_ones((messages.size(0), 1)),
            point_to_res,
            dim=0,
            dim_size=n_residues,
        )
        pooled = pooled / counts.clamp_min(1.0)
        residue_context = self.residue_projection(res_feats)
        return self.norm(self.update(torch.cat([residue_context, pooled], dim=-1)))


class GeometryAwareSparseCSRA(nn.Module):
    """Geometry-aware sparse residue-surface interaction.

    All surface points are encoded by the surface EGNN before this module.
    For every residue, this module scores its assigned surface points using
    feature compatibility plus relative distance/direction, keeps the learned
    top-k candidates, and returns a gated residual correction.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        topk: int = 32,
        num_rbf: int = 8,
        rbf_radius: float = 6.0,
        dropout: float = 0.1,
        use_geometry: bool = True,
        use_sparse_topk: bool = True,
        use_residual_gate: bool = True,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.geometry_bias = nn.Sequential(
            nn.Linear(num_rbf + 3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(query_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.topk = int(topk)
        self.use_geometry = bool(use_geometry)
        self.use_sparse_topk = bool(use_sparse_topk)
        self.use_residual_gate = bool(use_residual_gate)
        self.scale = hidden_dim ** -0.5
        centers = torch.linspace(0.0, rbf_radius, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 1.0 / max((rbf_radius / max(num_rbf - 1, 1)) ** 2, 1e-6)

    def forward(
        self,
        res_feats: torch.Tensor,
        point_feats: torch.Tensor,
        point_to_res: torch.Tensor,
        residue_pos: torch.Tensor,
        point_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(res_feats)
        k = self.k_proj(point_feats)
        values = self.v_proj(point_feats)

        content_scores = (q[point_to_res] * k).sum(dim=-1) * self.scale
        if self.use_geometry:
            relative = point_pos - residue_pos[point_to_res]
            distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            direction = relative / distance
            rbf = torch.exp(-self.rbf_gamma * (distance - self.rbf_centers.view(1, -1)).square())
            geometry = torch.cat([rbf, direction], dim=-1)
            scores = content_scores + self.geometry_bias(geometry).squeeze(-1)
        else:
            scores = content_scores
        scores = scores.clamp(min=-20.0, max=20.0)

        # Learned top-k is applied only after every point has passed through
        # the surface encoder and received a content-plus-geometry score.
        if self.use_sparse_topk:
            point_ids = torch.arange(scores.size(0), device=scores.device)
            dense_scores, _ = to_dense_batch(scores, point_to_res, fill_value=-1e4)
            dense_ids, _ = to_dense_batch(point_ids, point_to_res, fill_value=-1)
            k_per_residue = min(self.topk, dense_scores.size(1))
            _, selected_columns = dense_scores.topk(k_per_residue, dim=1)
            selected_ids = dense_ids.gather(1, selected_columns)
            selected_valid = selected_ids >= 0
            keep_mask = torch.zeros(scores.size(0), dtype=torch.bool, device=scores.device)
            keep_mask[selected_ids[selected_valid]] = True
            sparse_scores = scores.masked_fill(~keep_mask, -1e4)
        else:
            sparse_scores = scores

        attention = scatter_softmax(sparse_scores, point_to_res, dim=0)
        aggregated = scatter_sum(values * attention.unsqueeze(-1), point_to_res, dim=0, dim_size=res_feats.size(0))
        correction = self.norm(self.output_proj(aggregated))
        gate = torch.sigmoid(self.residual_gate(torch.cat([res_feats, correction], dim=-1))) if self.use_residual_gate else correction.new_ones((res_feats.size(0), 1))
        return correction, gate
