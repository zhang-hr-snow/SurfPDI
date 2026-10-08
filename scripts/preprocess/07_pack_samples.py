"""
Step 7: Pack final per-protein training samples.

Environment:
  bioai

Inputs:
  data/manifest/manifest.csv
  data/intermediate/foundation/<protein_id>/
  data/surface_sampled/<sample-version>/<protein_id>/
  data/surface_graphs/<surface-graph-version>/<protein_id>/
  data/residue_graphs/<residue-graph-version>/<protein_id>/
  ESMC features from the old project, usually D:/Codes/SurfPDI/npy*/<protein_id>/esmc_features.pt

Outputs:
  data/processed/<processed-version>/samples/<protein_id>.pt
  data/processed/<processed-version>/processed_manifest.csv
  data/processed/<processed-version>/processed_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.io_utils import read_manifest, write_json


SPLIT_TO_OLD_DIR = {
    "train573": "npy573",
    "test129": "npy129",
    "test181": "npy181",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-version", required=True)
    parser.add_argument("--sample-version", required=True)
    parser.add_argument("--surface-graph-version", required=True)
    parser.add_argument("--residue-graph-version", default="r10")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest" / "manifest.csv")
    parser.add_argument("--foundation-root", type=Path, default=ROOT / "data" / "intermediate" / "foundation")
    parser.add_argument("--sampled-root", type=Path, default=ROOT / "data" / "surface_sampled")
    parser.add_argument("--surface-graph-root", type=Path, default=ROOT / "data" / "surface_graphs")
    parser.add_argument("--residue-graph-root", type=Path, default=ROOT / "data" / "residue_graphs")
    parser.add_argument("--old-project-root", type=Path, default=Path(r"D:/Codes/SurfPDI"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed")
    return parser.parse_args()


def normalize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return ((features - mean) / std).astype(np.float32)


def find_esmc(row: dict, old_project_root: Path) -> Path:
    old_dir = SPLIT_TO_OLD_DIR[row["split"]]
    path = old_project_root / old_dir / row["protein_id"] / "esmc_features.pt"
    if not path.exists():
        raise FileNotFoundError(f"ESMC feature not found: {path}")
    return path


def pack_one(row: dict, args: argparse.Namespace, samples_dir: Path) -> dict:
    protein_id = row["protein_id"]
    labels = torch.tensor([int(ch) for ch in row["label_string"]], dtype=torch.long)

    foundation_dir = args.foundation_root / protein_id
    sampled_dir = args.sampled_root / args.sample_version / protein_id
    surface_graph_dir = args.surface_graph_root / args.surface_graph_version / protein_id
    residue_graph_dir = args.residue_graph_root / args.residue_graph_version / protein_id

    surf_xyz = np.load(sampled_dir / "surf_xyz.npy").astype(np.float32)
    surf_curv = np.load(sampled_dir / "surf_curv.npy").astype(np.float32)
    surf_onehot = np.load(sampled_dir / "surf_onehot.npy").astype(np.float32)
    surf_hbond = np.load(sampled_dir / "surf_hbond.npy").astype(np.float32)
    surf_hphob = np.load(sampled_dir / "surf_hphob.npy").astype(np.float32)
    surf_res_idx = np.load(sampled_dir / "surf_res_idx.npy").astype(np.int64)

    raw_surface_features = np.concatenate(
        [surf_curv, surf_onehot, surf_hbond[:, None], surf_hphob[:, None]],
        axis=-1,
    )
    surface_features = normalize_features(raw_surface_features)
    surface_x = np.concatenate([surf_xyz, surface_features], axis=-1).astype(np.float32)

    residue_pos = torch.from_numpy(np.load(foundation_dir / "residue_pos.npy")).float()
    esmc_features = torch.load(find_esmc(row, args.old_project_root), map_location="cpu", weights_only=False)

    if len(labels) != residue_pos.shape[0]:
        raise ValueError(f"Label/residue length mismatch for {protein_id}: {len(labels)} vs {residue_pos.shape[0]}")
    if esmc_features.shape[0] != residue_pos.shape[0]:
        raise ValueError(f"ESMC/residue length mismatch for {protein_id}: {esmc_features.shape[0]} vs {residue_pos.shape[0]}")
    if surf_res_idx.max(initial=0) >= residue_pos.shape[0]:
        raise ValueError(f"surf_res_idx exceeds residue count for {protein_id}")

    sample = {
        "protein_id": protein_id,
        "split": row["split"],
        "sequence": row["sequence"],
        "x": torch.from_numpy(surface_x).float(),
        "edge_index": torch.from_numpy(np.load(surface_graph_dir / "surf_edge_index.npy")).long(),
        "edge_attr": torch.from_numpy(np.load(surface_graph_dir / "surf_edge_attr.npy")).float(),
        "point_to_residue": torch.from_numpy(surf_res_idx).long(),
        "residue_pos": residue_pos,
        "residue_edge_index": torch.from_numpy(np.load(residue_graph_dir / "residue_edge_index.npy")).long(),
        "residue_edge_attr": torch.from_numpy(np.load(residue_graph_dir / "residue_edge_attr.npy")).float(),
        "esmc_features": esmc_features,
        "residue_labels": labels,
        "metadata": {
            "processed_version": args.processed_version,
            "sample_version": args.sample_version,
            "surface_graph_version": args.surface_graph_version,
            "residue_graph_version": args.residue_graph_version,
            "num_surface_points": int(surface_x.shape[0]),
            "num_residues": int(residue_pos.shape[0]),
            "num_positive_residues": int(labels.sum().item()),
        },
    }
    out_path = samples_dir / f"{protein_id}.pt"
    torch.save(sample, out_path)
    return {
        "protein_id": protein_id,
        "split": row["split"],
        "sample_path": (Path("samples") / out_path.name).as_posix(),
        "num_residues": int(residue_pos.shape[0]),
        "num_surface_points": int(surface_x.shape[0]),
        "num_positive_residues": int(labels.sum().item()),
    }


def main() -> None:
    args = parse_args()
    out_root = args.output_root / args.processed_version
    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    processed_rows = []
    errors = []
    for row in tqdm(rows, desc="pack samples"):
        try:
            processed_rows.append(pack_one(row, args, samples_dir))
        except Exception as exc:
            errors.append({"protein_id": row["protein_id"], "error": str(exc)})

    if processed_rows:
        with (out_root / "processed_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(processed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(processed_rows)

    stats = {
        "processed_version": args.processed_version,
        "sample_version": args.sample_version,
        "surface_graph_version": args.surface_graph_version,
        "residue_graph_version": args.residue_graph_version,
        "num_samples": len(processed_rows),
        "num_errors": len(errors),
        "total_residues": int(sum(row["num_residues"] for row in processed_rows)),
        "total_surface_points": int(sum(row["num_surface_points"] for row in processed_rows)),
        "errors": errors,
    }
    write_json(out_root / "processed_stats.json", stats)
    print(f"Processed root: {out_root}")
    print(f"Packed samples: {len(processed_rows)}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
