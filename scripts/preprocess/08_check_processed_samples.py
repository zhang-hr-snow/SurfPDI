"""
Step 8: Sanity-check packed per-protein sample.pt files.

Environment:
  bioai

Inputs:
  data/processed/<processed-version>/
    samples/<protein_id>.pt
    processed_manifest.csv

Outputs:
  data/processed/<processed-version>/sample_check_report.json

Checks:
  - required keys exist
  - residue label length equals residue_pos and ESMC length
  - point_to_residue indices are within residue count
  - surface and residue edge indices are within node counts
  - edge_attr row counts match edge_index column counts
  - tensor values do not contain NaN/Inf
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm


REQUIRED_KEYS = [
    "protein_id",
    "split",
    "sequence",
    "x",
    "edge_index",
    "edge_attr",
    "point_to_residue",
    "residue_pos",
    "residue_edge_index",
    "residue_edge_attr",
    "esmc_features",
    "residue_labels",
    "metadata",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def tensor_has_bad_values(value: torch.Tensor) -> bool:
    if not torch.is_floating_point(value):
        return False
    return bool(torch.isnan(value).any().item() or torch.isinf(value).any().item())


def check_one(path: Path) -> tuple[dict, list[str]]:
    sample = torch.load(path, map_location="cpu", weights_only=False)
    errors = []
    for key in REQUIRED_KEYS:
        if key not in sample:
            errors.append(f"missing key: {key}")
    if errors:
        return {"sample_path": str(path)}, errors

    num_surface = int(sample["x"].shape[0])
    num_residue = int(sample["residue_pos"].shape[0])
    num_surface_edges = int(sample["edge_index"].shape[1])
    num_residue_edges = int(sample["residue_edge_index"].shape[1])

    if sample["x"].shape[1] != 17:
        errors.append(f"x dim should be 17, got {tuple(sample['x'].shape)}")
    if sample["residue_labels"].shape[0] != num_residue:
        errors.append("residue_labels length != residue_pos length")
    if sample["esmc_features"].shape[0] != num_residue:
        errors.append("esmc_features length != residue_pos length")
    if len(sample["sequence"]) != num_residue:
        errors.append("sequence length != residue_pos length")
    if sample["point_to_residue"].shape[0] != num_surface:
        errors.append("point_to_residue length != surface point count")
    if sample["point_to_residue"].numel() and int(sample["point_to_residue"].max().item()) >= num_residue:
        errors.append("point_to_residue index out of range")
    if sample["edge_index"].numel() and int(sample["edge_index"].max().item()) >= num_surface:
        errors.append("surface edge_index out of range")
    if sample["residue_edge_index"].numel() and int(sample["residue_edge_index"].max().item()) >= num_residue:
        errors.append("residue edge_index out of range")
    if sample["edge_attr"].shape[0] != num_surface_edges:
        errors.append("surface edge_attr rows != surface edge count")
    if sample["residue_edge_attr"].shape[0] != num_residue_edges:
        errors.append("residue_edge_attr rows != residue edge count")

    for key in [
        "x",
        "edge_attr",
        "residue_pos",
        "residue_edge_attr",
        "esmc_features",
    ]:
        if tensor_has_bad_values(sample[key]):
            errors.append(f"NaN/Inf in {key}")

    info = {
        "protein_id": sample["protein_id"],
        "split": sample["split"],
        "sample_path": str(path),
        "num_surface_points": num_surface,
        "num_residues": num_residue,
        "num_surface_edges": num_surface_edges,
        "num_residue_edges": num_residue_edges,
        "num_positive_residues": int(sample["residue_labels"].sum().item()),
    }
    return info, errors


def summarize(values: list[int]) -> dict:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "min": int(values_sorted[0]),
        "median": int(values_sorted[n // 2]),
        "max": int(values_sorted[-1]),
        "mean": float(sum(values_sorted) / n),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.processed_root / "processed_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    infos = []
    errors = []
    for row in tqdm(rows, desc=f"check {args.processed_root.name}"):
        info, sample_errors = check_one(Path(row["sample_path"]))
        infos.append(info)
        if sample_errors:
            errors.append(
                {
                    "protein_id": info.get("protein_id", row.get("protein_id")),
                    "sample_path": info.get("sample_path", row["sample_path"]),
                    "errors": sample_errors,
                }
            )

    report = {
        "processed_root": str(args.processed_root),
        "num_samples": len(infos),
        "num_error_samples": len(errors),
        "surface_points": summarize([info["num_surface_points"] for info in infos if "num_surface_points" in info]),
        "residues": summarize([info["num_residues"] for info in infos if "num_residues" in info]),
        "surface_edges": summarize([info["num_surface_edges"] for info in infos if "num_surface_edges" in info]),
        "residue_edges": summarize([info["num_residue_edges"] for info in infos if "num_residue_edges" in info]),
        "example_errors": errors[: args.max_examples],
    }
    output_path = args.processed_root / "sample_check_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
