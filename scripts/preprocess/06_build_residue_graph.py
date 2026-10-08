"""
Step 6: Build residue graph and residue edge attributes.

Environment:
  bioai

Inputs:
  data/intermediate/foundation/<protein_id>/
    residue_pos.npy
    residue_ca_xyz.npy

Outputs:
  data/residue_graphs/r<radius>/<protein_id>/
    residue_edge_index.npy
    residue_edge_attr.npy
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

from src.preprocessing.graph_features import residue_edge_attr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-root", type=Path, default=ROOT / "data" / "intermediate" / "foundation")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "residue_graphs")
    parser.add_argument("--radius", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_version = f"r{args.radius:g}"
    out_root = args.output_root / graph_version
    config = {"residue_edge_radius": args.radius}
    errors = []
    for protein_dir in tqdm([p for p in args.foundation_root.iterdir() if p.is_dir()], desc="residue graph"):
        try:
            residue_pos = torch.from_numpy(np.load(protein_dir / "residue_pos.npy")).float()
            residue_ca = torch.from_numpy(np.load(protein_dir / "residue_ca_xyz.npy")).float()
            edge_index = radius_graph(residue_pos, r=args.radius, loop=False)
            edge_attr = residue_edge_attr(residue_pos, residue_ca, edge_index)
            out_dir = out_root / protein_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "residue_edge_index.npy", edge_index.numpy().astype(np.int64))
            np.save(out_dir / "residue_edge_attr.npy", edge_attr.numpy().astype(np.float32))
            (out_dir / "graph_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception as exc:
            errors.append({"protein_id": protein_dir.name, "error": str(exc)})
    (out_root / "residue_graph_errors.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    (out_root / "graph_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Residue graph root: {out_root}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()

