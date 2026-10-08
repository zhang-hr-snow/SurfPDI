from __future__ import annotations

import torch


def surface_edge_attr(pos: torch.Tensor, normals: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    norm_sim = (normals[row] * normals[col]).sum(dim=-1, keepdim=True)
    rel_pos = pos[col] - pos[row]
    dist = rel_pos.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit_rel_pos = rel_pos / dist
    dr_angle_i = (normals[row] * unit_rel_pos).sum(dim=-1, keepdim=True)
    dr_angle_j = (normals[col] * unit_rel_pos).sum(dim=-1, keepdim=True)
    return torch.cat([norm_sim, dr_angle_i, dr_angle_j], dim=-1)


def residue_edge_attr(residue_pos: torch.Tensor, residue_ca_pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    res_vecs = residue_pos - residue_ca_pos
    res_vecs = res_vecs / res_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dist = torch.norm(residue_pos[row] - residue_pos[col], dim=-1, keepdim=True)
    f1 = 1.0 / (dist + 1e-6)

    seq_dist = torch.abs(row - col).float().unsqueeze(-1)
    f2 = 1.0 / (seq_dist + 1e-6)

    f3 = (res_vecs[row] * res_vecs[col]).sum(dim=-1, keepdim=True)

    rel_pos = residue_pos[col] - residue_pos[row]
    unit_rel_pos = rel_pos / rel_pos.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    f4 = (res_vecs[row] * unit_rel_pos).sum(dim=-1, keepdim=True)

    return torch.cat([f1, f2, f3, f4], dim=-1)

