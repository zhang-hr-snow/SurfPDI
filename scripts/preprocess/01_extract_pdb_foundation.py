"""
Step 1: Extract atom-level and residue-level foundation arrays from PDB files.

Environment:
  bioai

Inputs:
  data/manifest/manifest.csv
  PDB files listed by the manifest column `pdb_path`

Outputs:
  data/intermediate/foundation/<protein_id>/
    atom_xyz.npy
    atom_types.npy
    atom_res_map.npy
    residue_names.npy
    residue_pos.npy
    residue_ca_xyz.npy
    sequence_info.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.io_utils import ensure_dir, read_manifest, write_json


ELE2NUM = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest" / "manifest.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "intermediate" / "foundation")
    parser.add_argument("--strict-chain", action="store_true")
    return parser.parse_args()


def sidechain_centroid(residue):
    coords = [atom.get_coord() for atom in residue.get_atoms() if atom.get_name() not in BACKBONE_ATOMS]
    if coords:
        return np.mean(coords, axis=0)
    if "CA" in residue:
        return residue["CA"].get_coord()
    return None


def standard_residues(structure, chain_id: str, strict_chain: bool):
    candidate_chains = []
    for model in structure:
        candidate_chains.extend(list(model.get_chains()))
        break
    if chain_id:
        selected = [chain for chain in candidate_chains if chain.id == chain_id]
        if selected:
            candidate_chains = selected
        elif strict_chain:
            raise ValueError(f"Chain {chain_id} not found.")
    for chain in candidate_chains:
        for residue in chain:
            if residue.get_id()[0] == " ":
                yield residue


def process_row(row: dict, output_root: Path, strict_chain: bool) -> None:
    protein_id = row["protein_id"]
    pdb_path = Path(row["pdb_path"])
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found for {protein_id}: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_id, str(pdb_path))

    atom_xyz = []
    atom_types = []
    atom_res_map = []
    residue_names = []
    residue_pos = []
    residue_ca_xyz = []

    for res_idx, residue in enumerate(standard_residues(structure, row["chain_id"], strict_chain)):
        centroid = sidechain_centroid(residue)
        if centroid is None:
            continue
        residue_names.append(residue.get_resname())
        residue_pos.append(centroid)
        residue_ca_xyz.append(residue["CA"].get_coord() if "CA" in residue else centroid)
        for atom in residue.get_atoms():
            element = atom.element.upper()
            if element in ELE2NUM:
                atom_xyz.append(atom.get_coord())
                atom_types.append(ELE2NUM[element])
                atom_res_map.append(len(residue_names) - 1)

    if not atom_xyz:
        raise ValueError(f"No valid atoms extracted for {protein_id}")

    out_dir = ensure_dir(output_root / protein_id)
    np.save(out_dir / "atom_xyz.npy", np.asarray(atom_xyz, dtype=np.float32))
    atom_onehot = np.zeros((len(atom_types), len(ELE2NUM)), dtype=np.float32)
    atom_onehot[np.arange(len(atom_types)), np.asarray(atom_types, dtype=np.int64)] = 1.0
    np.save(out_dir / "atom_types.npy", atom_onehot)
    np.save(out_dir / "atom_res_map.npy", np.asarray(atom_res_map, dtype=np.int64))
    np.save(out_dir / "residue_names.npy", np.asarray(residue_names))
    np.save(out_dir / "residue_pos.npy", np.asarray(residue_pos, dtype=np.float32))
    np.save(out_dir / "residue_ca_xyz.npy", np.asarray(residue_ca_xyz, dtype=np.float32))
    write_json(
        out_dir / "sequence_info.json",
        {
            "protein_id": protein_id,
            "split": row["split"],
            "sequence_length": int(row["length"]),
            "label_positive_count": int(row["positive_count"]),
            "pdb_residue_count": len(residue_names),
            "pdb_path": str(pdb_path),
        },
    )


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    errors = []
    for row in tqdm(rows, desc="extract foundation"):
        try:
            process_row(row, args.output_root, args.strict_chain)
        except Exception as exc:
            errors.append({"protein_id": row["protein_id"], "error": str(exc)})
    write_json(args.output_root / "foundation_errors.json", {"errors": errors})
    print(f"Foundation root: {args.output_root}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()

