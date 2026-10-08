"""
Input:
    A YAML config such as configs/train/base_full_cv.yaml.

Output:
    results/training/<run_name>_<timestamp>/
        config.yaml
        fold_0..fold_4/
            fold_split.json
            history.json
            best.pt
            final_metrics.json
        cv_summary.json

Protocol:
    DNA-573 is used for 5-fold cross-validation only.
    DNA-129 and DNA-181 are independent test sets and are never used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import move_batch_to_device, surfpdi_collate
from src.data.dataset import SurfPDIDataset
from src.models.surfpdi_base import ProteinInterfacePredictor
from src.training.trainer import Trainer
from src.training.metrics import metrics_from_probabilities, positive_probabilities, tune_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Override run.name for an identifiable output directory.")
    parser.add_argument("--output-root", type=str, default=None, help="Override run.output_root, useful for self-contained queues.")
    parser.add_argument("--dry-run", action="store_true", help="Run one train batch from fold 0 and exit.")
    parser.add_argument("--dry-run-largest", action="store_true", help="Use the largest fold-0 train sample for dry-run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override loader.batch_size while preserving the remaining config.",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--skip-independent-tests",
        action="store_true",
        help="Run DNA-573 model selection only; DNA-129 and DNA-181 are not loaded.",
    )
    parser.add_argument(
        "--pretrain-run",
        type=str,
        default=None,
        help="Pretraining run directory; fold_f/best_pretrain.pt initializes only fold f's surface trunk.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("supervised", config)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, loader_cfg: dict, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        collate_fn=surfpdi_collate,
    )


def class_weights_from_dataset(dataset: SurfPDIDataset, loss_cfg: dict) -> torch.Tensor | None:
    mode = loss_cfg.get("class_weight", "clipped_balanced")
    if mode in {"none", None, False}:
        return None
    negatives, positives = dataset.residue_label_counts()
    if positives == 0 or negatives == 0:
        return torch.ones(2, dtype=torch.float32)
    total = positives + negatives
    balanced = torch.tensor([total / (2.0 * negatives), total / (2.0 * positives)], dtype=torch.float32)
    if mode == "balanced":
        return balanced
    if mode == "clipped_balanced":
        max_positive = float(loss_cfg.get("max_positive_class_weight", 1.0))
        return torch.tensor([1.0, min(float(balanced[1]), max_positive)], dtype=torch.float32)
    raise ValueError(f"Unsupported loss.class_weight: {mode}")


def positive_ratio_bins(records: list[dict[str, str]], n_bins: int) -> np.ndarray | None:
    ratios = np.array(
        [int(r["num_positive_residues"]) / max(int(r["num_residues"]), 1) for r in records],
        dtype=np.float64,
    )
    edges = np.unique(np.quantile(ratios, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 2:
        return None
    labels = np.digitize(ratios, edges[1:-1], right=True)
    counts = np.bincount(labels)
    return labels if counts.min() >= n_bins else None


def make_folds(records: list[dict[str, str]], cfg: dict):
    n_splits = int(cfg["cv"].get("n_splits", 5))
    seed = int(cfg["cv"].get("seed", cfg["run"].get("seed", 42)))
    indices = np.arange(len(records))
    if cfg["cv"].get("stratify", "positive_ratio_bins") == "positive_ratio_bins":
        labels = positive_ratio_bins(records, n_splits)
        if labels is not None:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(indices, labels)), "stratified_positive_ratio_bins"
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(indices)), "kfold"


def build_output_dir(cfg: dict) -> Path:
    run_name = cfg["run"].get("name", "surfpdi_cv")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(cfg["run"].get("output_root", "results/training")) / f"{run_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def metric_mean_std(rows: list[dict], selection_name: str, test_name: str) -> dict:
    keys = ["specificity", "recall", "precision", "f1", "auc", "aupr", "mcc", "accuracy", "loss"]
    out = {}
    for key in keys:
        values = np.array(
            [row["selections"][selection_name]["tests"][test_name][key] for row in rows],
            dtype=np.float64,
        )
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


def validation_metric_mean_std(rows: list[dict], selection_name: str) -> dict:
    keys = ["specificity", "recall", "precision", "f1", "auc", "aupr", "mcc", "accuracy", "loss"]
    out = {}
    for key in keys:
        values = np.array(
            [row["selections"][selection_name]["validation"][key] for row in rows],
            dtype=np.float64,
        )
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


def metrics_from_prediction(prediction: dict, threshold: float, class_weights: torch.Tensor | None) -> dict:
    logits = prediction["logits"].float()
    labels = prediction["labels"].long()
    probabilities = positive_probabilities(logits).numpy()
    metrics = metrics_from_probabilities(probabilities, labels.numpy(), threshold=threshold)
    weight_cpu = class_weights.detach().cpu() if class_weights is not None else None
    metrics["loss"] = float(F.cross_entropy(logits, labels, weight=weight_cpu).item())
    return metrics


def save_prediction_csv(path: Path, prediction: dict, threshold: float) -> None:
    logits = prediction["logits"]
    labels = prediction["labels"]
    probabilities = positive_probabilities(logits).numpy()
    rows = prediction["rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["protein_id", "residue_index", "label", "probability", "prediction", "threshold"],
        )
        writer.writeheader()
        for row, label, probability in zip(rows, labels.numpy(), probabilities):
            writer.writerow(
                {
                    "protein_id": row["protein_id"],
                    "residue_index": row["residue_index"],
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "threshold": float(threshold),
                }
            )


def load_surface_pretraining(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    """Load only compatible surface-encoder weights from one SSL fold."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    prefixes = ("surface_input.", "surface_egnn.", "surface_norm.")
    loaded, skipped = [], []
    for key, value in source.items():
        if not key.startswith(prefixes):
            continue
        if key in target and target[key].shape == value.shape:
            target[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    if not loaded:
        raise RuntimeError(
            f"No compatible surface weights in {checkpoint_path}. "
            "Use the matching 32-D pretrained dual-branch configuration."
        )
    model.load_state_dict(target, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "pretrain_epoch": checkpoint.get("epoch"),
        "pretrain_val_reconstruction": checkpoint.get("val_reconstruction"),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
    }


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)
    if args.run_name:
        cfg.setdefault("run", {})["name"] = args.run_name
    if args.output_root:
        cfg.setdefault("run", {})["output_root"] = args.output_root
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        cfg.setdefault("loader", {})["batch_size"] = args.batch_size
    set_seed(int(cfg["run"].get("seed", 42)))

    device_name = cfg["run"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    processed_root = resolve_path(cfg["data"]["processed_root"])
    override_root = cfg["data"].get("override_root")
    override_root = resolve_path(override_root) if override_root else None
    dataset_kwargs = {"override_root": override_root}
    train_all = SurfPDIDataset(
        processed_root,
        cfg["data"].get("cv_split", ["train573"]),
        **dataset_kwargs,
    )
    folds, fold_method = make_folds(train_all.records, cfg)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    output_dir = build_output_dir(cfg)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    independent_description = "not_loaded" if args.skip_independent_tests else "129,181"
    print(
        f"protocol=573_{len(folds)}fold_cv independent_tests={independent_description} "
        f"fold_method={fold_method} device={device} output_dir={output_dir}"
    )

    loader_cfg = cfg.get("loader", {})
    test_loaders = {}
    if not args.skip_independent_tests:
        test_loaders = {
            "test129": make_loader(
                SurfPDIDataset(processed_root, ["test129"], args.max_test_samples, **dataset_kwargs),
                loader_cfg,
                shuffle=False,
            ),
            "test181": make_loader(
                SurfPDIDataset(processed_root, ["test181"], args.max_test_samples, **dataset_kwargs),
                loader_cfg,
                shuffle=False,
            ),
        }
    fold_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_records = [train_all.records[i] for i in train_idx]
        val_records = [train_all.records[i] for i in val_idx]
        if args.dry_run_largest:
            dry_batch_size = int(loader_cfg.get("batch_size", 1))
            train_records = sorted(
                train_records,
                key=lambda row: int(row["num_surface_points"]),
                reverse=True,
            )[:dry_batch_size]
        train_set = SurfPDIDataset(
            processed_root,
            records=train_records,
            max_samples=args.max_train_samples,
            **dataset_kwargs,
        )
        val_set = SurfPDIDataset(
            processed_root,
            records=val_records,
            max_samples=args.max_val_samples,
            **dataset_kwargs,
        )

        split_info = {
            "fold": fold_id,
            "train_count": len(train_set),
            "val_count": len(val_set),
            "train_proteins": [r["protein_id"] for r in train_set.records],
            "val_proteins": [r["protein_id"] for r in val_set.records],
            "independent_tests": list(test_loaders),
        }
        (fold_dir / "fold_split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

        train_loader = make_loader(train_set, loader_cfg, shuffle=True)
        val_loader = make_loader(val_set, loader_cfg, shuffle=False)
        model = ProteinInterfacePredictor(**cfg.get("model", {})).to(device)
        if args.pretrain_run:
            checkpoint_path = resolve_path(args.pretrain_run) / f"fold_{fold_id}" / "best_pretrain.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing fold-specific pretraining checkpoint: {checkpoint_path}")
            pretrain_info = load_surface_pretraining(model, checkpoint_path)
            (fold_dir / "pretrain_initialization.json").write_text(
                json.dumps(pretrain_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"fold={fold_id} loaded_surface_pretraining="
                f"epoch={pretrain_info['pretrain_epoch']} keys={len(pretrain_info['loaded_keys'])}"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["optim"].get("lr", 1e-4)),
            weight_decay=float(cfg["optim"].get("weight_decay", 1e-4)),
        )
        weights = class_weights_from_dataset(train_set, cfg.get("loss", {}))
        weights = weights.to(device) if weights is not None else None
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            output_dir=fold_dir,
            amp=bool(cfg["train"].get("amp", True)),
            grad_clip_norm=cfg["train"].get("grad_clip_norm", 1.0),
            monitor=cfg["train"].get("monitor", "mcc"),
            monitor_mode=cfg["train"].get("monitor_mode", "max"),
            checkpoint_monitors=cfg["train"].get("checkpoint_monitors", ["auc", "mcc"]),
        )

        print(f"fold={fold_id} train={len(train_set)} val={len(val_set)} class_weights={weights}")
        if args.dry_run:
            batch = move_batch_to_device(next(iter(train_loader)), device)
            model.train()
            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda"):
                logits = model(batch)
                loss = criterion(logits, batch["residue_labels"])
            loss.backward()
            print(
                "dry_run_ok "
                f"fold={fold_id} proteins={batch['protein_ids']} residues={batch['residue_labels'].numel()} "
                f"surface_points={batch['x'].shape[0]} loss={float(loss.detach().cpu()):.4f}"
            )
            return

        fit_summary = trainer.fit(
            train_loader,
            val_loader,
            epochs=int(args.epochs or cfg["train"].get("epochs", 100)),
            patience=cfg["train"].get("patience", 20),
        )
        selections = {}
        primary_selection = f"best_{fit_summary['primary_monitor']}"
        for monitor_name, checkpoint_info in fit_summary["best"].items():
            selection_name = f"best_{monitor_name}"
            checkpoint = torch.load(checkpoint_info["best_path"], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])

            val_prediction = trainer.predict(val_loader)
            val_probs = positive_probabilities(val_prediction["logits"]).numpy()
            val_labels = val_prediction["labels"].numpy()
            threshold_info = tune_threshold(
                val_probs,
                val_labels,
                metric=cfg.get("threshold", {}).get("metric", "mcc"),
                num_thresholds=int(cfg.get("threshold", {}).get("num_thresholds", 200)),
            )
            tuned_threshold = float(threshold_info["threshold"])
            validation_metrics = metrics_from_prediction(
                val_prediction,
                threshold=tuned_threshold,
                class_weights=weights,
            )
            (fold_dir / f"threshold_tuning_{selection_name}.json").write_text(
                json.dumps(threshold_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            save_prediction_csv(
                fold_dir / f"predictions_validation_{selection_name}.csv",
                val_prediction,
                tuned_threshold,
            )

            tests = {}
            tests_at_0_5 = {}
            for name, loader in test_loaders.items():
                prediction = trainer.predict(loader)
                tests_at_0_5[name] = metrics_from_prediction(prediction, threshold=0.5, class_weights=weights)
                tests[name] = metrics_from_prediction(prediction, threshold=tuned_threshold, class_weights=weights)
                save_prediction_csv(fold_dir / f"predictions_{name}_{selection_name}.csv", prediction, tuned_threshold)

            selections[selection_name] = {
                "checkpoint": checkpoint_info,
                "threshold_tuning": threshold_info,
                "validation": validation_metrics,
                "tests": tests,
                "tests_at_0.5": tests_at_0_5,
            }

        fold_final = {
            "fold": fold_id,
            "fit": fit_summary,
            "primary_selection": primary_selection,
            "selections": selections,
            "tests": selections[primary_selection]["tests"],
            "tests_at_0.5": selections[primary_selection]["tests_at_0.5"],
        }
        (fold_dir / "final_metrics.json").write_text(json.dumps(fold_final, indent=2, ensure_ascii=False), encoding="utf-8")
        fold_rows.append(fold_final)

        del model, optimizer, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selection_summary = {}
    selection_names = list(fold_rows[0]["selections"].keys()) if fold_rows else []
    for selection_name in selection_names:
        selection_summary[selection_name] = {
            "validation": validation_metric_mean_std(fold_rows, selection_name),
        }
        for test_name in test_loaders:
            selection_summary[selection_name][test_name] = metric_mean_std(
                fold_rows,
                selection_name,
                test_name,
            )

    summary = {
        "protocol": (
            "DNA-573 5-fold CV model selection only; DNA-129 and DNA-181 were not loaded."
            if args.skip_independent_tests
            else "DNA-573 5-fold CV followed by DNA-129 and DNA-181 benchmark evaluation."
        ),
        "fold_method": fold_method,
        "primary_selection": f"best_{cfg['train'].get('monitor', 'mcc')}",
        "folds": fold_rows,
        "selection_summary": selection_summary,
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
