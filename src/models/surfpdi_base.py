# src/model.py

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from src.modules.egnn_pytorch_geometric import EGNN_Sparse_Network
from torch_geometric.nn import TransformerConv
from src.modules.layers import CrossScaleInteractionModule, aggregate_points_to_residues, init_weights_xavier, SurfaceAttentionPooling, SurfaceMeanPooling, SurfaceToResidueMessage, GeometryAwareSparseCSRA

"""
        几何注入式全局推理架构 (Local-to-Global Reasoning)
        【学术描述】：
        该架构遵循“微观感知引导宏观决策”的设计哲学。
        1. 特征提取层：表面分支与结构分支并行运行 2层 EGNN，分别在 表面点云（微观）和 残基点（宏观）尺度上
           建立旋转平移等变的局部空间表征。
        2. 几何注入层：引入 CSRA（跨模态交叉注意力）模块。每一个残基节点作为 Query，自适应地从其
           所属的表面点云中吸纳高分辨率的几何纹理与物理场信号。
        3. 全局精修层：在融合后的残基空间内，引入原生的 Vanilla Transformer。利用全局自注意力机制
           打破 GNN 的感受野限制，捕捉蛋白质远端残基间的长程协同效应（如变构效应），从全局视角
           对结合界面的连通性进行逻辑重构。
"""


class ProteinInterfacePredictor(nn.Module):
    def __init__(
        self,
        node_feat_dim=14,
        edge_attr_dim=3,
        res_edge_dim=4,
        hidden_dim=128,
        surface_hidden_dim=None,
        surface_message_dim=16,
        surface_num_layers=2,
        residue_num_layers=2,
        surface_norm_coors=False,
        surface_coordinate_weight_clamp=None,
        surface_update_coors=True,
        use_surface_branch=True,
        use_surface_egnn=True,
        surface_feature_mode="full",
        surface_edge_attr_mode="full",
        residue_edge_attr_mode="full",
        csra_type="base",
        csra_topk=32,
        csra_use_geometry=True,
        csra_use_sparse_topk=True,
        csra_use_residual_gate=True,
        use_global_transformer=True,
        transformer_dropout=0.1,
        classifier_dropout=0.3,
    ):
        super().__init__()

        # --- [1] 表面分支：2层 EGNN (保持 14 维以节省显存) ---
        self.use_surface_branch = bool(use_surface_branch)
        self.use_surface_egnn = bool(use_surface_egnn)
        self.use_global_transformer = bool(use_global_transformer)
        self.surface_feature_mode = str(surface_feature_mode)
        self.surface_edge_attr_mode = str(surface_edge_attr_mode)
        self.residue_edge_attr_mode = str(residue_edge_attr_mode)
        self.csra_type = csra_type
        if self.surface_feature_mode not in {"full", "coordinates_only", "geometry", "physicochemical"}:
            raise ValueError(f"Unsupported surface_feature_mode: {self.surface_feature_mode}")
        if self.surface_edge_attr_mode not in {"full", "zero"}:
            raise ValueError(f"Unsupported surface_edge_attr_mode: {self.surface_edge_attr_mode}")
        if self.residue_edge_attr_mode not in {"full", "zero"}:
            raise ValueError(f"Unsupported residue_edge_attr_mode: {self.residue_edge_attr_mode}")
        if self.use_surface_branch:
            surface_hidden_dim = int(surface_hidden_dim or node_feat_dim)
            self.surface_input = (
                nn.Identity()
                if surface_hidden_dim == node_feat_dim
                else nn.Sequential(
                    nn.Linear(node_feat_dim, surface_hidden_dim),
                    nn.LayerNorm(surface_hidden_dim),
                    nn.GELU(),
                )
            )
            self.surface_egnn = EGNN_Sparse_Network(
                n_layers=int(surface_num_layers), feats_dim=surface_hidden_dim, pos_dim=3,
                edge_attr_dim=edge_attr_dim, m_dim=surface_message_dim, aggr='mean',
                norm_coors=surface_norm_coors,
                update_coors=surface_update_coors,
                coor_weights_clamp_value=surface_coordinate_weight_clamp,
            )
            self.surface_norm = nn.LayerNorm(surface_hidden_dim + 3)

        # --- [2] 结构分支：2层 EGNN ---
        self.esm_proj = nn.Linear(2560, hidden_dim)
        self.structure_egnn = EGNN_Sparse_Network(
            n_layers=int(residue_num_layers), feats_dim=hidden_dim, pos_dim=3,
            edge_attr_dim=res_edge_dim, m_dim=32, aggr='mean'
        )
        self.struct_norm = nn.LayerNorm(hidden_dim + 3)

        # --- [3] 交叉模态融合 (CSRA) ---
        # Query: 结构特征(H+3), Key: 表面特征(14+3)
        if self.use_surface_branch:
            if csra_type == "base":
                self.csra = SurfaceAttentionPooling(
                    query_dim=hidden_dim + 3,
                    key_dim=surface_hidden_dim + 3,
                    hidden_dim=hidden_dim
                )
            elif csra_type in {"mean_concat", "mean_gated_residual"}:
                self.csra = SurfaceMeanPooling(
                    query_dim=hidden_dim + 3,
                    key_dim=surface_hidden_dim + 3,
                    hidden_dim=hidden_dim,
                    use_residual_gate=csra_type == "mean_gated_residual",
                )
            elif csra_type == "geometry_sparse_gated":
                self.csra = GeometryAwareSparseCSRA(
                    query_dim=hidden_dim + 3,
                    key_dim=surface_hidden_dim + 3,
                    hidden_dim=hidden_dim,
                    topk=csra_topk,
                    use_geometry=csra_use_geometry,
                    use_sparse_topk=csra_use_sparse_topk,
                    use_residual_gate=csra_use_residual_gate,
                )
            elif csra_type in {"surface_message", "surface_message_geometry"}:
                self.csra = SurfaceToResidueMessage(
                    query_dim=hidden_dim + 3,
                    key_dim=surface_hidden_dim + 3,
                    hidden_dim=hidden_dim,
                    use_geometry=csra_type == "surface_message_geometry",
                )
            else:
                raise ValueError(f"Unsupported csra_type: {csra_type}")

        # --- [4] 【创新核心】：原生 Vanilla Transformer ---
        # 融合特征维度 = 结构分支(H+3) + 表面聚合(H)
        self.fusion_dim = hidden_dim * 2 if self.use_surface_branch else hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,  # 正好是 512，完美被 4 整除
            nhead=4,
            dim_feedforward=self.fusion_dim * 2,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 2)
        )
        init_weights_xavier(self)

    def forward(self, data):
        # 1. 表面分支 (2层 EGNN)
        if self.use_surface_branch:
            surface_attributes = data['x'][:, 3:]
            if self.surface_feature_mode == "coordinates_only":
                surface_attributes = torch.zeros_like(surface_attributes)
            elif self.surface_feature_mode == "geometry":
                surface_attributes = torch.cat(
                    [surface_attributes[:, :6], torch.zeros_like(surface_attributes[:, 6:])],
                    dim=-1,
                )
            elif self.surface_feature_mode == "physicochemical":
                surface_attributes = torch.cat(
                    [torch.zeros_like(surface_attributes[:, :6]), surface_attributes[:, 6:]],
                    dim=-1,
                )
            s_features = self.surface_input(surface_attributes)
            s_input = torch.cat([data['x'][:, :3], s_features], dim=-1)
            surface_edge_attr = data['edge_attr']
            if self.surface_edge_attr_mode == "zero":
                surface_edge_attr = torch.zeros_like(surface_edge_attr)
            s_out = (
                self.surface_egnn(s_input, data['edge_index'], data['batch'], surface_edge_attr)
                if self.use_surface_egnn
                else s_input
            )
            s_out = self.surface_norm(s_out)

        # 2. 结构分支 (2层 EGNN)
        r_h = self.esm_proj(data['esmc_features'].float())
        r_in = torch.cat([data['residue_pos'], r_h], dim=-1)
        residue_edge_attr = data['residue_edge_attr']
        if self.residue_edge_attr_mode == "zero":
            residue_edge_attr = torch.zeros_like(residue_edge_attr)
        r_out = self.structure_egnn(r_in, data['residue_edge_index'], data['residue_batch'],
                                    edge_attr=residue_edge_attr)
        r_out = self.struct_norm(r_out)

        # 3. CSRA 融合 (将微观信息注入宏观节点)
        # D. 【关键动作】：只提取结构分支的特征部分 (去掉前3列坐标)
        res_feats_structure = r_out[:, 3:]  # [L, hidden]

        # E. 拼接融合：[L, hidden + hidden] = [L, 2hidden]
        if self.use_surface_branch:
            if self.csra_type in {"surface_message", "surface_message_geometry"}:
                correction = self.csra(
                    res_feats=r_out,
                    point_feats=s_out,
                    point_to_res=data['point_to_residue'],
                    residue_pos=data['residue_pos'],
                    point_pos=data['x'][:, :3],
                )
                corrected_residue = res_feats_structure + correction
                fused_init = torch.cat([res_feats_structure, corrected_residue], dim=-1)
            elif self.csra_type in {"geometry_sparse_gated", "mean_concat", "mean_gated_residual"}:
                correction, gate = self.csra(
                    res_feats=r_out,
                    point_feats=s_out,
                    point_to_res=data['point_to_residue'],
                    **(
                        {
                            "residue_pos": data['residue_pos'],
                            "point_pos": data['x'][:, :3],
                        }
                        if self.csra_type == "geometry_sparse_gated"
                        else {}
                    ),
                )
                if self.csra_type == "mean_concat":
                    fused_init = torch.cat([res_feats_structure, correction], dim=-1)
                else:
                    corrected_residue = res_feats_structure + gate * correction
                    fused_init = torch.cat([res_feats_structure, corrected_residue], dim=-1)
            else:
                res_surf_info = self.csra(
                    res_feats=r_out,
                    point_feats=s_out,
                    point_to_res=data['point_to_residue']
                )  # [L, hidden]
                fused_init = torch.cat([res_feats_structure, res_surf_info], dim=-1)
        else:
            fused_init = res_feats_structure

        # F. 原生 Transformer 全局推理
        if self.use_global_transformer:
            res_dense, mask = to_dense_batch(fused_init, data['residue_batch'])
            refined_dense = self.global_transformer(res_dense, src_key_padding_mask=~mask)
            final_res_h = refined_dense[mask]
        else:
            final_res_h = fused_init

        # G. 分类
        logits = self.classifier(final_res_h)

        return logits.squeeze(-1)
