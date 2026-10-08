"""
Step 0: Build a unified dataset manifest.

Environment:
  bioai

Inputs:
  data/raw/txt/DNA-573_Train.txt
  data/raw/txt/DNA-129_Test.txt
  data/raw/txt/DNA_Test_181.txt
  Optional PDB root, usually data/raw/pdb/<split>/<protein_id>.pdb

Outputs:
  data/manifest/manifest.csv
  data/manifest/dataset_stats.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.io_utils import SPLIT_FILES, read_label_txt, write_json, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-root", type=Path, default=ROOT / "data" / "raw" / "txt")
    parser.add_argument("--pdb-root", type=Path, default=ROOT / "data" / "raw" / "pdb")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifest" / "manifest.csv")
    parser.add_argument("--stats-output", type=Path, default=ROOT / "data" / "manifest" / "dataset_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for split, filename in SPLIT_FILES.items():
        txt_path = args.txt_root / filename
        for record in read_label_txt(txt_path):
            protein_id = record["protein_id"]
            split_pdb = args.pdb_root / split / f"{protein_id}.pdb"
            flat_pdb = args.pdb_root / f"{protein_id}.pdb"
            pdb_path = split_pdb if split_pdb.exists() else flat_pdb
            rows.append(
                {
                    **record,
                    "split": split,
                    "txt_file": str(txt_path),
                    "pdb_path": str(pdb_path),
                    "pdb_exists": str(pdb_path.exists()),
                }
            )

    write_manifest(args.output, rows)
    stats = {
        "num_proteins": len(rows),
        "splits": {},
        "total_residues": int(sum(int(row["length"]) for row in rows)),
        "total_positive_residues": int(sum(int(row["positive_count"]) for row in rows)),
    }
    for split in SPLIT_FILES:
        split_rows = [row for row in rows if row["split"] == split]
        stats["splits"][split] = {
            "num_proteins": len(split_rows),
            "total_residues": int(sum(int(row["length"]) for row in split_rows)),
            "total_positive_residues": int(sum(int(row["positive_count"]) for row in split_rows)),
            "missing_pdb": int(sum(row["pdb_exists"] != "True" for row in split_rows)),
        }
    write_json(args.stats_output, stats)
    print(f"Manifest saved to: {args.output}")
    print(f"Stats saved to: {args.stats_output}")


if __name__ == "__main__":
    main()

