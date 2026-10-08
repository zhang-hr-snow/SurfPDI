"""
Input:
    A completed DNA-573-only five-fold run selected as the final model.

Output:
    One frozen evaluation on DNA-129 and DNA-181 using each fold's validation
    threshold, residue-level predictions, and the six USPDB metrics.

This script refuses to run unless --confirm-final-model is explicit and the
source run declares that independent benchmark sets were not loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import move_batch_to_device, surfpdi_collate
from src.data.dataset import SurfPDIDataset
from src.models.surfpdi_base import ProteinInterfacePredictor
from src.training.metrics import metrics_from_probabilities, positive_probabilities


PAPER_METRICS = ("auc", "f1", "mcc", "precision", "recall", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection", default="best_auc", choices=("best_auc", "best_mcc"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--confirm-final-model", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def make_loader(dataset: SurfPDIDataset, loader_cfg: dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        collate_fn=surfpdi_collate,
    )


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(batch)
        logits_cpu = logits.detach().float().cpu()
        labels_cpu = batch["residue_labels"].detach().cpu()
        logits_all.append(logits_cpu)
        labels_all.append(labels_cpu)
        for protein_id, (start, end) in zip(batch["protein_ids"], batch["residue_slices"]):
            for residue_index, global_index in enumerate(range(start, end)):
                rows.append(
                    {
                        "protein_id": protein_id,
                        "residue_index": residue_index,
                        "label": int(labels_cpu[global_index].item()),
                    }
                )
    return {"logits": torch.cat(logits_all), "labels": torch.cat(labels_all), "rows": rows}


def save_predictions(
    path: Path,
    prediction: dict[str, Any],
    probabilities: np.ndarray,
    threshold: float,
    fold: int,
    selection: str,
    checkpoint: Path,
) -> None:
    fields = [
        "fold", "selection", "checkpoint", "protein_id", "residue_index",
        "label", "probability", "prediction", "threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(prediction["rows"], probabilities):
            writer.writerow(
                {
                    "fold": fold,
                    "selection": selection,
                    "checkpoint": str(checkpoint),
                    **row,
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "threshold": threshold,
                }
            )


def summarize(folds: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    out = {}
    for metric in PAPER_METRICS:
        values = np.asarray([fold["tests"][dataset][metric] for fold in folds], dtype=float)
        out[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    if not args.confirm_final_model:
        raise RuntimeError("Refusing benchmark evaluation without --confirm-final-model.")

    run_dir = resolve_path(args.run_dir)
    config_path = run_dir / "config.yaml"
    summary_path = run_dir / "cv_summary.json"
    if not config_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Incomplete selected run: {run_dir}")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = str(source_summary.get("protocol", ""))
    if "were not loaded" not in protocol:
        raise RuntimeError(
            "The selected run is not a DNA-573-only run. Select an output generated with "
            "--skip-independent-tests."
        )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    processed_root = resolve_path(cfg["data"]["processed_root"])
    device_name = cfg["run"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    amp = bool(cfg["train"].get("amp", True))
    loaders = {
        "test129": make_loader(SurfPDIDataset(processed_root, ["test129"]), cfg.get("loader", {})),
        "test181": make_loader(SurfPDIDataset(processed_root, ["test181"]), cfg.get("loader", {})),
    }

    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / "final_evaluation" / f"frozen_{run_dir.name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_dirs = sorted(path for path in run_dir.glob("fold_*") if path.is_dir())
    if len(fold_dirs) != 5:
        raise RuntimeError(f"Expected five completed folds, found {len(fold_dirs)} in {run_dir}")
    fold_outputs: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        fold = int(fold_dir.name.split("_")[-1])
        checkpoint_path = fold_dir / f"{args.selection}.pt"
        threshold_path = fold_dir / f"threshold_tuning_{args.selection}.json"
        if not checkpoint_path.exists() or not threshold_path.exists():
            raise FileNotFoundError(f"Missing selected checkpoint or threshold in {fold_dir}")
        threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        model = ProteinInterfacePredictor(**cfg.get("model", {})).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        fold_result = {
            "fold": fold,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "threshold": threshold,
            "tests": {},
        }
        for dataset_name, loader in loaders.items():
            prediction = predict(model, loader, device, amp)
            probabilities = positive_probabilities(prediction["logits"]).numpy()
            metrics = metrics_from_probabilities(probabilities, prediction["labels"].numpy(), threshold)
            fold_result["tests"][dataset_name] = {key: metrics[key] for key in PAPER_METRICS}
            save_predictions(
                output_dir / f"predictions_{dataset_name}_fold{fold}.csv",
                prediction,
                probabilities,
                threshold,
                fold,
                args.selection,
                checkpoint_path,
            )
            print(
                f"fold={fold} dataset={dataset_name} "
                + " ".join(f"{key}={metrics[key]:.4f}" for key in PAPER_METRICS)
            )
        fold_outputs.append(fold_result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final = {
        "protocol": "Frozen final-model evaluation; thresholds were selected on DNA-573 validation folds.",
        "source_run": str(run_dir),
        "selection": args.selection,
        "paper_metrics": list(PAPER_METRICS),
        "folds": fold_outputs,
        "summary": {dataset: summarize(fold_outputs, dataset) for dataset in loaders},
    }
    (output_dir / "final_benchmark_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(final["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
