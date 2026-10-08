"""
Optional Step 0b: Copy PDB files from the old dMaSIF project into the new raw layout.

Environment:
  bioai or normal Python

Inputs:
  data/manifest/manifest.csv
  Old PDB folders:
    D:/Codes/dmasif/PDB573
    D:/Codes/dmasif/PDB129
    D:/Codes/dmasif/pdb181

Outputs:
  data/raw/pdb/train573/<protein_id>.pdb
  data/raw/pdb/test129/<protein_id>.pdb
  data/raw/pdb/test181/<protein_id>.pdb

After running this script, rerun:
  python scripts/preprocess/00_build_manifest.py
so that `pdb_exists` in the manifest is refreshed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.io_utils import read_manifest


DEFAULT_SOURCE_ROOTS = {
    "train573": Path(r"D:/Codes/dmasif/PDB573"),
    "test129": Path(r"D:/Codes/dmasif/PDB129"),
    "test181": Path(r"D:/Codes/dmasif/pdb181"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest" / "manifest.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "raw" / "pdb")
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="copy")
    return parser.parse_args()


def find_source_pdb(row: dict) -> Path | None:
    source_root = DEFAULT_SOURCE_ROOTS[row["split"]]
    candidates = [
        source_root / f"{row['protein_id']}.pdb",
        source_root / f"{row['pdb_id']}_{row['chain_id']}.pdb",
        source_root / f"{row['pdb_id']}.pdb",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    copied = 0
    missing = []
    for row in tqdm(rows, desc="prepare pdb"):
        source = find_source_pdb(row)
        if source is None:
            missing.append(row["protein_id"])
            continue
        target = args.output_root / row["split"] / f"{row['protein_id']}.pdb"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            copied += 1
            continue
        if args.copy_mode == "link":
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        copied += 1

    report = {"prepared": copied, "missing": missing, "num_missing": len(missing)}
    (args.output_root / "prepare_pdb_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PDB prepared: {copied}")
    print(f"Missing: {len(missing)}")
    print(f"Report: {args.output_root / 'prepare_pdb_report.json'}")


if __name__ == "__main__":
    main()

