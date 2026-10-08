"""
Input:
    data/processed/<processed_version>/processed_manifest.csv
    data/processed/<processed_version>/samples/<protein_id>.pt

Output:
    One sample dict per protein. Large tensors are loaded lazily in __getitem__,
    so full-resolution surface data is not held in memory at dataset creation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


class SurfPDIDataset(Dataset):
    """Lazy dataset for one-protein-per-file SurfPDI processed samples."""

    def __init__(
        self,
        processed_root: str | Path,
        splits: Iterable[str] | None = None,
        max_samples: int | None = None,
        records: list[dict[str, str]] | None = None,
        override_root: str | Path | None = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.manifest_path = self.processed_root / "processed_manifest.csv"
        self.override_root = Path(override_root) if override_root is not None else None
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing processed manifest: {self.manifest_path}")

        if records is None:
            split_set = set(splits) if splits is not None else None
            loaded_records: list[dict[str, str]] = []
            with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if split_set is not None and row["split"] not in split_set:
                        continue
                    loaded_records.append(row)
        else:
            loaded_records = list(records)

        if max_samples is not None:
            loaded_records = loaded_records[:max_samples]
        if not loaded_records:
            raise ValueError(f"No samples found in {self.manifest_path} for splits={splits}")

        self.records = loaded_records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_sample_path(self, record: dict[str, str]) -> Path:
        manifest_path = Path(record.get("sample_path", ""))
        candidates = [manifest_path]
        if not manifest_path.is_absolute():
            candidates.append(self.processed_root / manifest_path)
        candidates.append(self.processed_root / "samples" / f"{record['protein_id']}.pt")

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Missing sample file for {record['protein_id']}; checked: {checked}"
        )

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sample_path = self._resolve_sample_path(record)

        try:
            sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        except TypeError:
            sample = torch.load(sample_path, map_location="cpu")

        if self.override_root is not None:
            override_path = self.override_root / f"{record['protein_id']}.pt"
            if not override_path.is_file():
                raise FileNotFoundError(
                    f"Missing experimental override for {record['protein_id']}: {override_path}"
                )
            try:
                override = torch.load(override_path, map_location="cpu", weights_only=False)
            except TypeError:
                override = torch.load(override_path, map_location="cpu")
            sample = self._apply_override(sample, override, override_path)

        sample["sample_path"] = str(sample_path)
        return sample

    @staticmethod
    def _apply_override(sample: dict, override: dict, override_path: Path) -> dict:
        """Apply an explicitly prepared intervention without duplicating ESM-C tensors."""

        allowed = {
            "x",
            "surface_attributes",
            "edge_index",
            "edge_attr",
            "point_to_residue",
            "residue_edge_index",
            "residue_edge_attr",
        }
        unknown = set(override) - allowed - {"metadata"}
        if unknown:
            raise KeyError(f"Unsupported keys in {override_path}: {sorted(unknown)}")

        modified = dict(sample)
        if "surface_attributes" in override:
            attributes = override["surface_attributes"]
            if attributes.shape != sample["x"][:, 3:].shape:
                raise ValueError(
                    f"surface_attributes shape mismatch in {override_path}: "
                    f"{tuple(attributes.shape)} vs {tuple(sample['x'][:, 3:].shape)}"
                )
            modified["x"] = sample["x"].clone()
            modified["x"][:, 3:] = attributes
        for key in allowed - {"surface_attributes"}:
            if key in override:
                modified[key] = override[key]
        modified["experimental_override"] = str(override_path)
        return modified

    def residue_label_counts(self) -> tuple[int, int]:
        """Return negative and positive residue counts from the lightweight manifest."""

        negatives = 0
        positives = 0
        for row in self.records:
            n_res = int(row["num_residues"])
            n_pos = int(row["num_positive_residues"])
            positives += n_pos
            negatives += n_res - n_pos
        return negatives, positives
