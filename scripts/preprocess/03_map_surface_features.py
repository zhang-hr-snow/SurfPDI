"""
Step 3: Map physicochemical features from atoms/residues to surface points.

Environment:
  bioai

Inputs:
  data/intermediate/foundation/<protein_id>/
    atom_xyz.npy
    atom_types.npy
    residue_names.npy
  data/surface_raw/<surface_version>/<protein_id>/
    surf_xyz.npy
    surf_res_idx.npy

Outputs:
  data/intermediate/surface_features/<surface_version>/<protein_id>/
    surf_onehot.npy
    surf_hbond.npy
    surf_hphob.npy
    feature_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KD_SCALE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9, "ALA": 1.8,
    "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3, "PRO": -1.6,
    "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-version", required=True, help="Example: resolution_1 or resolution_1.5")
    parser.add_argument("--foundation-root", type=Path, default=ROOT / "data" / "intermediate" / "foundation")
    parser.add_argument("--surface-root", type=Path, default=ROOT / "data" / "surface_raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "intermediate" / "surface_features")
    return parser.parse_args()


def process(protein_id: str, foundation_dir: Path, surface_dir: Path, output_dir: Path) -> None:
    surf_xyz = torch.from_numpy(np.load(surface_dir / "surf_xyz.npy")).float()
    atom_xyz = torch.from_numpy(np.load(foundation_dir / "atom_xyz.npy")).float()
    atom_types = torch.from_numpy(np.load(foundation_dir / "atom_types.npy")).float()
    residue_names = np.load(foundation_dir / "residue_names.npy")
    surf_res_idx = np.load(surface_dir / "surf_res_idx.npy").astype(np.int64)

    hphob_table = np.asarray([KD_SCALE.get(str(name), 0.0) for name in residue_names], dtype=np.float32)
    surf_hphob = hphob_table[np.clip(surf_res_idx, 0, len(hphob_table) - 1)]

    is_oxygen = atom_types[:, 2] == 1
    is_nitrogen = atom_types[:, 3] == 1
    polar_mask = is_oxygen | is_nitrogen
    if polar_mask.any():
        polar_xyz = atom_xyz[polar_mask]
        polar_sign = torch.where(atom_types[polar_mask, 2] == 1, -1.0, 1.0)
        dists = torch.cdist(surf_xyz, polar_xyz)
        surf_hbond = (polar_sign * torch.exp(-dists)).sum(dim=1).numpy().astype(np.float32)
    else:
        surf_hbond = np.zeros(len(surf_xyz), dtype=np.float32)

    dist_to_atoms = torch.cdist(surf_xyz, atom_xyz)
    nearest_atom = dist_to_atoms.argmin(dim=1)
    surf_onehot = atom_types[nearest_atom].numpy().astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "surf_onehot.npy", surf_onehot)
    np.save(output_dir / "surf_hbond.npy", surf_hbond)
    np.save(output_dir / "surf_hphob.npy", surf_hphob.astype(np.float32))
    (output_dir / "feature_config.json").write_text(json.dumps({"protein_id": protein_id}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    surface_version_root = args.surface_root / args.surface_version
    out_version_root = args.output_root / args.surface_version
    errors = []
    for surface_dir in tqdm([p for p in surface_version_root.iterdir() if p.is_dir()], desc="map surface features"):
        protein_id = surface_dir.name
        try:
            process(protein_id, args.foundation_root / protein_id, surface_dir, out_version_root / protein_id)
        except Exception as exc:
            errors.append({"protein_id": protein_id, "error": str(exc)})
    (out_version_root / "surface_feature_errors.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    print(f"Surface feature root: {out_version_root}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()

