"""
Step 4: Build full or sampled surface-point versions.

Environment:
  bioai

Inputs:
  data/surface_raw/<surface_version>/<protein_id>/
    surf_xyz.npy
    surf_normals.npy
    surf_curv.npy
    surf_res_idx.npy
  data/intermediate/surface_features/<surface_version>/<protein_id>/
    surf_onehot.npy
    surf_hbond.npy
    surf_hphob.npy
  data/intermediate/foundation/<protein_id>/residue_pos.npy

Outputs:
  data/surface_sampled/<output-version>/<protein_id>/
    surf_xyz.npy
    surf_normals.npy
    surf_curv.npy
    surf_res_idx.npy
    surf_onehot.npy
    surf_hbond.npy
    surf_hphob.npy
    selected_indices.npy
    sampling_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.sampling import residue_wise_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-version", required=True)
    parser.add_argument("--method", choices=["full", "fps", "random", "nearest"], default="full")
    parser.add_argument("--max-points-per-residue", type=int, default=None)
    parser.add_argument("--output-version", required=True, help="Example: res1.0_full or res1.0_fps_k32")
    parser.add_argument("--surface-root", type=Path, default=ROOT / "data" / "surface_raw")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data" / "intermediate" / "surface_features")
    parser.add_argument("--foundation-root", type=Path, default=ROOT / "data" / "intermediate" / "foundation")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "surface_sampled")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_selected(source_arrays: dict, indices: np.ndarray, output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in source_arrays.items():
        np.save(output_dir / f"{name}.npy", array[indices])
    np.save(output_dir / "selected_indices.npy", indices.astype(np.int64))
    (output_dir / "sampling_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = args.output_root / args.output_version
    surface_version_root = args.surface_root / args.surface_version
    feature_version_root = args.feature_root / args.surface_version
    config = {
        "surface_version": args.surface_version,
        "method": args.method,
        "max_points_per_residue": args.max_points_per_residue,
        "seed": args.seed,
    }
    errors = []
    for surface_dir in tqdm([p for p in surface_version_root.iterdir() if p.is_dir()], desc="sample surface"):
        protein_id = surface_dir.name
        try:
            feature_dir = feature_version_root / protein_id
            arrays = {
                "surf_xyz": np.load(surface_dir / "surf_xyz.npy"),
                "surf_normals": np.load(surface_dir / "surf_normals.npy"),
                "surf_curv": np.load(surface_dir / "surf_curv.npy"),
                "surf_res_idx": np.load(surface_dir / "surf_res_idx.npy").astype(np.int64),
                "surf_onehot": np.load(feature_dir / "surf_onehot.npy"),
                "surf_hbond": np.load(feature_dir / "surf_hbond.npy"),
                "surf_hphob": np.load(feature_dir / "surf_hphob.npy"),
            }
            residue_pos = np.load(args.foundation_root / protein_id / "residue_pos.npy")
            indices = residue_wise_sample(
                arrays["surf_xyz"],
                arrays["surf_res_idx"],
                residue_pos,
                args.method,
                args.max_points_per_residue,
                seed=args.seed,
            )
            save_selected(arrays, indices, out_root / protein_id, config)
        except Exception as exc:
            errors.append({"protein_id": protein_id, "error": str(exc)})
    (out_root / "sampling_errors.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    (out_root / "sampling_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Sampled surface root: {out_root}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()

