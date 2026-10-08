"""
Input:
    A full-resolution collated SurfPDI batch. Labels are intentionally ignored.

Output:
    Masked surface-attribute reconstruction, a boolean mask identifying the
    reconstructed points, and global residue/surface embeddings for contrast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src.modules.egnn_pytorch_geometric import EGNN_Sparse_Network


class FullSurfaceMultimodalPretrainer(nn.Module):
    """Memory-conscious full-surface pretrainer with a small 32-D surface trunk."""

    def __init__(
        self,
        surface_feature_dim: int = 14,
        surface_hidden_dim: int = 32,
        surface_message_dim: int = 16,
        residue_hidden_dim: int = 128,
        residue_message_dim: int = 32,
        projection_dim: int = 64,
        coordinate_weight_clamp: float | None = 0.10,
        contrast_teacher_mode: str = "frozen_esmc",
        use_local_alignment: bool = False,
        shared_private_surface: bool = False,
        local_alignment_shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.surface_mask_token = nn.Parameter(torch.zeros(surface_feature_dim))
        self.surface_input = nn.Sequential(
            nn.Linear(surface_feature_dim, surface_hidden_dim),
            nn.LayerNorm(surface_hidden_dim),
            nn.GELU(),
        )
        self.surface_egnn = EGNN_Sparse_Network(
            n_layers=2,
            feats_dim=surface_hidden_dim,
            pos_dim=3,
            edge_attr_dim=3,
            m_dim=surface_message_dim,
            aggr="mean",
            norm_coors=True,
            coor_weights_clamp_value=coordinate_weight_clamp,
        )
        self.surface_norm = nn.LayerNorm(surface_hidden_dim + 3)
        self.use_local_alignment = bool(use_local_alignment)
        self.shared_private_surface = bool(shared_private_surface)
        self.local_alignment_shuffle = bool(local_alignment_shuffle)
        if self.shared_private_surface and surface_hidden_dim % 2 != 0:
            raise ValueError("shared_private_surface requires an even surface_hidden_dim")
        self.shared_dim = surface_hidden_dim // 2 if self.shared_private_surface else surface_hidden_dim
        self.private_dim = surface_hidden_dim // 2 if self.shared_private_surface else surface_hidden_dim

        self.surface_decoder = nn.Sequential(
            nn.Linear(self.private_dim, surface_hidden_dim),
            nn.GELU(),
            nn.Linear(surface_hidden_dim, surface_feature_dim),
        )

        self.contrast_teacher_mode = contrast_teacher_mode
        if contrast_teacher_mode == "frozen_esmc":
            # ESM-C is precomputed and fixed. Only this small adapter learns,
            # avoiding a second coordinate-updating GNN in the contrast path.
            self.residue_teacher_projection = nn.Sequential(
                nn.Linear(2560, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        elif contrast_teacher_mode == "residue_egnn":
            self.residue_input = nn.Sequential(
                nn.Linear(2560, residue_hidden_dim),
                nn.LayerNorm(residue_hidden_dim),
                nn.GELU(),
            )
            self.residue_egnn = EGNN_Sparse_Network(
                n_layers=2,
                feats_dim=residue_hidden_dim,
                pos_dim=3,
                edge_attr_dim=4,
                m_dim=residue_message_dim,
                aggr="mean",
                norm_coors=True,
                coor_weights_clamp_value=coordinate_weight_clamp,
            )
            self.residue_norm = nn.LayerNorm(residue_hidden_dim + 3)
            self.residue_global_projection = nn.Sequential(
                nn.Linear(residue_hidden_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            raise ValueError(f"Unsupported contrast_teacher_mode: {contrast_teacher_mode}")

        self.surface_global_projection = nn.Sequential(
            nn.Linear(self.shared_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        if self.use_local_alignment:
            self.surface_local_projection = nn.Sequential(
                nn.Linear(self.shared_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
            self.residue_local_projection = nn.Sequential(
                nn.Linear(2560, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )

    def forward(self, batch: dict[str, torch.Tensor], mask_ratio: float) -> dict[str, torch.Tensor]:
        surface_coordinates = batch["x"][:, :3].float()
        surface_attributes = batch["x"][:, 3:].float()
        mask = torch.rand(surface_attributes.shape[0], device=surface_attributes.device) < mask_ratio
        masked_attributes = torch.where(
            mask.unsqueeze(-1),
            self.surface_mask_token.unsqueeze(0),
            surface_attributes,
        )
        surface_features = self.surface_input(masked_attributes)
        surface_output = self.surface_egnn(
            torch.cat([surface_coordinates, surface_features], dim=-1),
            batch["edge_index"],
            batch["batch"],
            batch["edge_attr"],
        )
        surface_output = self.surface_norm(surface_output)
        surface_hidden = surface_output[:, 3:]
        if self.shared_private_surface:
            surface_shared, surface_private = surface_hidden.split(
                [self.shared_dim, self.private_dim], dim=-1
            )
        else:
            surface_shared = surface_hidden
            surface_private = surface_hidden
        reconstructed_attributes = self.surface_decoder(surface_private)

        surface_global = scatter_mean(surface_shared, batch["batch"], dim=0)
        if self.contrast_teacher_mode == "frozen_esmc":
            residue_global = scatter_mean(batch["esmc_features"].float(), batch["residue_batch"], dim=0)
            residue_embedding = F.normalize(
                self.residue_teacher_projection(residue_global), dim=-1, eps=1e-6
            )
        else:
            residue_features = self.residue_input(batch["esmc_features"].float())
            residue_output = self.residue_egnn(
                torch.cat([batch["residue_pos"].float(), residue_features], dim=-1),
                batch["residue_edge_index"],
                batch["residue_batch"],
                batch["residue_edge_attr"],
            )
            residue_output = self.residue_norm(residue_output)
            residue_hidden = residue_output[:, 3:]
            residue_global = scatter_mean(residue_hidden, batch["residue_batch"], dim=0)
            residue_embedding = F.normalize(
                self.residue_global_projection(residue_global), dim=-1, eps=1e-6
            )
        output = {
            "target_attributes": surface_attributes,
            "reconstructed_attributes": reconstructed_attributes,
            "mask": mask,
            # eps=1e-6 stays representable in float16 and avoids unstable
            # division when a newly initialized projection is nearly zero.
            "surface_embedding": F.normalize(self.surface_global_projection(surface_global), dim=-1, eps=1e-6),
            "residue_embedding": residue_embedding,
        }
        if self.use_local_alignment:
            n_residues = int(batch["residue_pos"].shape[0])
            point_to_residue = batch["point_to_residue"].long()
            counts = torch.bincount(point_to_residue, minlength=n_residues)
            valid_residues = counts > 0
            surface_local = scatter_mean(
                surface_shared,
                point_to_residue,
                dim=0,
                dim_size=n_residues,
            )[valid_residues]
            residue_local = batch["esmc_features"].float()[valid_residues]
            residue_positions = batch["residue_pos"].float()[valid_residues]
            residue_local_batch = batch["residue_batch"].long()[valid_residues]
            if self.local_alignment_shuffle and surface_local.size(0) > 1:
                shuffled = surface_local.clone()
                for protein_index in residue_local_batch.unique(sorted=True):
                    protein_mask = residue_local_batch == protein_index
                    protein_rows = protein_mask.nonzero(as_tuple=False).squeeze(1)
                    if protein_rows.numel() > 1:
                        permutation = torch.randperm(protein_rows.numel(), device=surface_local.device)
                        shuffled[protein_rows] = surface_local[protein_rows[permutation]]
                surface_local = shuffled
            output.update(
                {
                    "surface_local_embedding": F.normalize(
                        self.surface_local_projection(surface_local), dim=-1, eps=1e-6
                    ),
                    "residue_local_embedding": F.normalize(
                        self.residue_local_projection(residue_local), dim=-1, eps=1e-6
                    ),
                    "local_residue_positions": residue_positions,
                    "local_residue_batch": residue_local_batch,
                    "num_local_pairs": int(valid_residues.sum().item()),
                }
            )
        if self.shared_private_surface:
            shared_centered = surface_shared.float() - surface_shared.float().mean(dim=0, keepdim=True)
            private_centered = surface_private.float() - surface_private.float().mean(dim=0, keepdim=True)
            cross_covariance = shared_centered.T @ private_centered / max(surface_hidden.size(0), 1)
            output["decorrelation_loss"] = cross_covariance.square().mean()
        else:
            output["decorrelation_loss"] = surface_hidden.sum() * 0.0
        return output
