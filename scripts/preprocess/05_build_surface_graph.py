"""
Step 5: Build radius graph and edge attributes for sampled surface points.

Environment:
  bioai

Inputs:
  data/surface_sampled/<sample-version>/<protein_id>/
    surf_xyz.npy
    surf_normals.npy

Outputs:
  data/surface_graphs/<sample-version>_r<radius>/<protein_id>/
    surf_edge_index.npy
    surf_edge_attr.npy
    graph_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch_cluster import radius_graph
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.graph_features import surface_edge_attr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-version", required=True)
    parser.add_argument("--radius", type=float, default=4.5)
    parser.add_argument("--sampled-root", type=Path, default=ROOT / "data" / "surface_sampled")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "surface_graphs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.sampled_root / args.sample_version
    graph_version = f"{args.sample_version}_r{args.radius:g}"
    out_root = args.output_root / graph_version
    config = {"sample_version": args.sample_version, "surface_edge_radius": args.radius}
    errors = []
    for protein_dir in tqdm([p for p in input_root.iterdir() if p.is_dir()], desc="surface graph"):
        try:
            pos = torch.from_numpy(np.load(protein_dir / "surf_xyz.npy")).float()
            normals = torch.from_numpy(np.load(protein_dir / "surf_normals.npy")).float()
            edge_index = radius_graph(pos, r=args.radius, loop=False)
            edge_attr = surface_edge_attr(pos, normals, edge_index)
            out_dir = out_root / protein_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "surf_edge_index.npy", edge_index.numpy().astype(np.int64))
            np.save(out_dir / "surf_edge_attr.npy", edge_attr.numpy().astype(np.float32))
            (out_dir / "graph_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception as exc:
            errors.append({"protein_id": protein_dir.name, "error": str(exc)})
    (out_root / "surface_graph_errors.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    (out_root / "graph_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Surface graph root: {out_root}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()

