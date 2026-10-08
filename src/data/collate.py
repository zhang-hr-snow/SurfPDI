"""
Input:
    A list of SurfPDI sample dicts loaded from sample.pt files.

Output:
    A single batched dict accepted by src.models.surfpdi_base.ProteinInterfacePredictor.
    Surface and residue edge indices are offset independently.
"""

from __future__ import annotations

import torch


TENSOR_KEYS_TO_FLOAT = {
    "x",
    "edge_attr",
    "residue_pos",
    "residue_edge_attr",
}


def _cat_or_empty(values: list[torch.Tensor], shape_tail: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if values:
        return torch.cat(values, dim=0)
    return torch.empty((0, *shape_tail), dtype=dtype)


def surfpdi_collate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    x_list: list[torch.Tensor] = []
    edge_index_list: list[torch.Tensor] = []
    edge_attr_list: list[torch.Tensor] = []
    surface_batch_list: list[torch.Tensor] = []
    point_to_residue_list: list[torch.Tensor] = []

    residue_pos_list: list[torch.Tensor] = []
    residue_edge_index_list: list[torch.Tensor] = []
    residue_edge_attr_list: list[torch.Tensor] = []
    residue_batch_list: list[torch.Tensor] = []
    esmc_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    protein_ids: list[str] = []
    splits: list[str] = []
    sample_paths: list[str] = []
    residue_slices: list[tuple[int, int]] = []

    surface_offset = 0
    residue_offset = 0
    for batch_idx, sample in enumerate(samples):
        n_surface = int(sample["x"].shape[0])
        n_residue = int(sample["residue_pos"].shape[0])

        x_list.append(sample["x"].float())
        edge_index_list.append(sample["edge_index"].long() + surface_offset)
        edge_attr_list.append(sample["edge_attr"].float())
        surface_batch_list.append(torch.full((n_surface,), batch_idx, dtype=torch.long))
        point_to_residue_list.append(sample["point_to_residue"].long() + residue_offset)

        residue_pos_list.append(sample["residue_pos"].float())
        residue_edge_index_list.append(sample["residue_edge_index"].long() + residue_offset)
        residue_edge_attr_list.append(sample["residue_edge_attr"].float())
        residue_batch_list.append(torch.full((n_residue,), batch_idx, dtype=torch.long))
        esmc_list.append(sample["esmc_features"])
        label_list.append(sample["residue_labels"].long())

        protein_ids.append(str(sample["protein_id"]))
        splits.append(str(sample["split"]))
        sample_paths.append(str(sample.get("sample_path", "")))
        residue_slices.append((residue_offset, residue_offset + n_residue))

        surface_offset += n_surface
        residue_offset += n_residue

    batch = {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "batch": torch.cat(surface_batch_list, dim=0),
        "point_to_residue": torch.cat(point_to_residue_list, dim=0),
        "residue_pos": torch.cat(residue_pos_list, dim=0),
        "residue_edge_index": torch.cat(residue_edge_index_list, dim=1),
        "residue_edge_attr": torch.cat(residue_edge_attr_list, dim=0),
        "residue_batch": torch.cat(residue_batch_list, dim=0),
        "esmc_features": torch.cat(esmc_list, dim=0),
        "residue_labels": torch.cat(label_list, dim=0),
        "protein_ids": protein_ids,
        "splits": splits,
        "sample_paths": sample_paths,
        "residue_slices": residue_slices,
    }
    return batch


def move_batch_to_device(batch: dict, device: torch.device | str) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved
