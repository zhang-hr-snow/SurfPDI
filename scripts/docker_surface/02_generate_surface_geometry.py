"""
Step 2: Generate molecular surface points, normals, curvatures, and point-to-residue mapping.

Environment:
  Docker dMaSIF environment. This step requires geometry_processing.py and pykeops.

Inputs:
  data/intermediate/foundation/<protein_id>/
    atom_xyz.npy
    atom_types.npy
    atom_res_map.npy

Outputs:
  data/surface_raw/resolution_<value>/<protein_id>/
    surf_xyz.npy
    surf_normals.npy
    surf_curv.npy
    surf_res_idx.npy
    surface_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geometry_processing import atoms_to_points_normals, curvatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-root", type=Path, default=PROJECT_ROOT / "data" / "intermediate" / "foundation")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "surface_raw")
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--distance", type=float, default=1.1)
    parser.add_argument("--sup-sampling", type=int, default=40)
    parser.add_argument("--curvature-scales", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protein-id", default=None, help="Optional single protein id for a smoke test.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def process_protein(protein_dir: Path, out_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    atom_xyz = torch.from_numpy(np.load(protein_dir / "atom_xyz.npy")).float().to(device)
    atom_types = torch.from_numpy(np.load(protein_dir / "atom_types.npy")).float().to(device)
    atom_res_map = torch.from_numpy(np.load(protein_dir / "atom_res_map.npy")).long().to(device)
    atom_batch = torch.zeros(len(atom_xyz), dtype=torch.long, device=device)

    surf_xyz, surf_normals, surf_batch = atoms_to_points_normals(
        atoms=atom_xyz,
        batch=atom_batch,
        atomtypes=atom_types,
        resolution=args.resolution,
        distance=args.distance,
        sup_sampling=args.sup_sampling,
    )
    surf_curv = curvatures(
        vertices=surf_xyz,
        normals=surf_normals,
        batch=surf_batch,
        scales=args.curvature_scales,
    )

    dist = torch.cdist(surf_xyz, atom_xyz)
    nearest_atom = dist.argmin(dim=1)
    surf_res_idx = atom_res_map[nearest_atom]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "surf_xyz.npy", surf_xyz.detach().cpu().numpy().astype(np.float32))
    np.save(out_dir / "surf_normals.npy", surf_normals.detach().cpu().numpy().astype(np.float32))
    np.save(out_dir / "surf_curv.npy", surf_curv.detach().cpu().numpy().astype(np.float32))
    np.save(out_dir / "surf_res_idx.npy", surf_res_idx.detach().cpu().numpy().astype(np.int64))
    (out_dir / "surface_config.json").write_text(
        json.dumps(
            {
                "resolution": args.resolution,
                "distance": args.distance,
                "sup_sampling": args.sup_sampling,
                "curvature_scales": args.curvature_scales,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    version = f"resolution_{args.resolution:g}"
    out_root = args.output_root / version
    proteins = [p for p in args.foundation_root.iterdir() if p.is_dir() and (p / "atom_xyz.npy").exists()]
    if args.protein_id:
        proteins = [p for p in proteins if p.name == args.protein_id]
        if not proteins:
            raise ValueError(f"Protein id not found under foundation root: {args.protein_id}")
    if args.limit is not None:
        proteins = proteins[:args.limit]
    errors = []
    for protein_dir in tqdm(proteins, desc=f"surface {version}"):
        try:
            process_protein(protein_dir, out_root / protein_dir.name, args, device)
        except Exception as exc:
            errors.append({"protein_id": protein_dir.name, "error": str(exc)})
    (out_root / "surface_generation_errors.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    print(f"Surface raw root: {out_root}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
