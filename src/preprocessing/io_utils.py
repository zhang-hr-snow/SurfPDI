from __future__ import annotations

import csv
import json
from pathlib import Path


SPLIT_FILES = {
    "train573": "DNA-573_Train.txt",
    "test129": "DNA-129_Test.txt",
    "test181": "DNA_Test_181.txt",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_label_txt(path: Path) -> list[dict]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"Label txt must contain records of 3 lines: {path}")

    records = []
    for i in range(0, len(lines), 3):
        protein_id = lines[i].removeprefix(">").strip()
        sequence = lines[i + 1].strip()
        label_string = lines[i + 2].strip()
        if len(sequence) != len(label_string):
            raise ValueError(
                f"Sequence/label length mismatch for {protein_id}: "
                f"{len(sequence)} vs {len(label_string)}"
            )
        if any(ch not in "01" for ch in label_string):
            raise ValueError(f"Label string contains non-binary values for {protein_id}")
        pdb_id, chain_id = split_protein_id(protein_id)
        records.append(
            {
                "protein_id": protein_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "sequence": sequence,
                "label_string": label_string,
                "length": len(sequence),
                "positive_count": label_string.count("1"),
            }
        )
    return records


def split_protein_id(protein_id: str) -> tuple[str, str]:
    if "_" not in protein_id:
        return protein_id[:4], ""
    pdb_id, chain_id = protein_id.split("_", 1)
    return pdb_id, chain_id


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty manifest.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

