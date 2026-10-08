"""
Input:
    A YAML config and full-resolution packed SurfPDI samples.

Output:
    Fold-specific self-supervised multimodal checkpoints under results/pretraining.

The pretraining loss does not use residue labels. Five folds are nevertheless
kept aligned with later downstream DNA-573 cross-validation to prevent any
validation or independent-test structure from entering a fold's pretraining set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
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

from experiments.pretrain_multimodal_full.pretrain_model import FullSurfaceMultimodalPretrainer
from src.data.collate import move_batch_to_device, surfpdi_collate
from src.data.dataset import SurfPDIDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-largest", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config.get("pretraining", config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: SurfPDIDataset, loader_cfg: dict, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=False,
        collate_fn=surfpdi_collate,
    )


def positive_ratio_bins(records: list[dict[str, str]], n_splits: int) -> np.ndarray | None:
    ratios = np.array([
        int(row["num_positive_residues"]) / max(int(row["num_residues"]), 1)
        for row in records
    ])
    edges = np.unique(np.quantile(ratios, np.linspace(0.0, 1.0, n_splits + 1)))
    if len(edges) <= 2:
        return None
    labels = np.digitize(ratios, edges[1:-1], right=True)
    return labels if np.bincount(labels).min() >= n_splits else None


def make_folds(records: list[dict[str, str]], cfg: dict):
    n_splits = int(cfg["cv"].get("n_splits", 5))
    indices = np.arange(len(records))
    seed = int(cfg["run"].get("seed", 42))
    labels = positive_ratio_bins(records, n_splits)
    if labels is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices, labels)), "stratified_positive_ratio_bins"
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(indices)), "kfold"


class ContrastiveQueue:
    """Queue negatives make batch-size-one cross-modal contrast possible."""

    def __init__(self, capacity: int, temperature: float, device: torch.device) -> None:
        self.capacity = capacity
        self.temperature = temperature
        self.device = device
        self.residue: list[torch.Tensor] = []
        self.surface: list[torch.Tensor] = []

    def loss(self, residue: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        # Keep the queue and all logits in float32. Autocast's float16 matrix
        # products are unnecessary for this tiny protein-level contrast term
        # and were the first component to become non-finite in full-surface runs.
        residue = F.normalize(residue.float(), dim=-1, eps=1e-6)
        surface = F.normalize(surface.float(), dim=-1, eps=1e-6)
        current_similarity = residue @ surface.T
        if self.residue:
            residue_negatives = torch.cat(self.residue, dim=0).float()
            surface_negatives = torch.cat(self.surface, dim=0).float()
            r_to_s = torch.cat([current_similarity, residue @ surface_negatives.T], dim=1)
            s_to_r = torch.cat([current_similarity.T, surface @ residue_negatives.T], dim=1)
        else:
            r_to_s = current_similarity
            s_to_r = current_similarity.T
        r_to_s = r_to_s / self.temperature
        s_to_r = s_to_r / self.temperature
        target = torch.arange(residue.shape[0], dtype=torch.long, device=self.device)
        return 0.5 * (F.cross_entropy(r_to_s, target) + F.cross_entropy(s_to_r, target))

    def enqueue(self, residue: torch.Tensor, surface: torch.Tensor) -> None:
        self.residue.append(F.normalize(residue.detach().float(), dim=-1, eps=1e-6))
        self.surface.append(F.normalize(surface.detach().float(), dim=-1, eps=1e-6))
        while len(self.residue) > self.capacity:
            self.residue.pop(0)
            self.surface.pop(0)


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve a configured AMP dtype and reject unsupported GPU settings."""
    normalized = name.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP was requested but is not supported by this CUDA device")
        return torch.bfloat16
    raise ValueError(f"Unsupported training.amp_dtype: {name}")


def autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled and device.type == "cuda",
    )


def masked_reconstruction_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = output["mask"]
    if not bool(mask.any()):
        return output["reconstructed_attributes"].sum() * 0.0
    return F.mse_loss(output["reconstructed_attributes"][mask], output["target_attributes"][mask])


def _sample_nonmatching_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    anchors = torch.arange(length, device=device).unsqueeze(1)
    negatives = torch.randint(0, length - 1, (length, count), device=device)
    return negatives + (negatives >= anchors).long()


def _local_negative_indices(
    output: dict[str, torch.Tensor], count: int, mode: str
) -> torch.Tensor:
    length = int(output["residue_local_embedding"].size(0))
    count = min(count, length - 1)
    if mode == "random":
        return _sample_nonmatching_indices(length, count, output["residue_local_embedding"].device)
    if mode != "mixed_spatial":
        raise ValueError(f"Unsupported local_negative_mode: {mode}")

    # Find spatially close negatives from a random candidate pool. This avoids
    # an O(L^2) distance matrix on very large proteins.
    hard_count = count // 2
    random_count = count - hard_count
    candidate_count = min(max(count * 4, hard_count), length - 1)
    candidates = _sample_nonmatching_indices(
        length, candidate_count, output["residue_local_embedding"].device
    )
    positions = output["local_residue_positions"].float()
    candidate_distances = (
        positions.unsqueeze(1) - positions[candidates]
    ).square().sum(dim=-1)
    nearest_columns = candidate_distances.topk(hard_count, dim=1, largest=False).indices
    hard_negatives = candidates.gather(1, nearest_columns)
    random_negatives = _sample_nonmatching_indices(length, random_count, positions.device)
    return torch.cat([hard_negatives, random_negatives], dim=1)


def _single_protein_local_contrastive_loss(
    output: dict[str, torch.Tensor], temperature: float, num_negatives: int, negative_mode: str
) -> torch.Tensor:
    """Symmetric sampled InfoNCE with memory linear in residue count."""
    residue = output.get("residue_local_embedding")
    surface = output.get("surface_local_embedding")
    if residue is None or surface is None or residue.size(0) < 2:
        return output["reconstructed_attributes"].sum() * 0.0
    residue = F.normalize(residue.float(), dim=-1, eps=1e-6)
    surface = F.normalize(surface.float(), dim=-1, eps=1e-6)
    negative_ids = _local_negative_indices(output, num_negatives, negative_mode)
    positive = (residue * surface).sum(dim=-1, keepdim=True)
    residue_to_surface_negatives = (
        residue.unsqueeze(1) * surface[negative_ids]
    ).sum(dim=-1)
    surface_to_residue_negatives = (
        surface.unsqueeze(1) * residue[negative_ids]
    ).sum(dim=-1)
    residue_to_surface = torch.cat([positive, residue_to_surface_negatives], dim=1) / temperature
    surface_to_residue = torch.cat([positive, surface_to_residue_negatives], dim=1) / temperature
    target = torch.zeros(residue.size(0), dtype=torch.long, device=residue.device)
    return 0.5 * (
        F.cross_entropy(residue_to_surface, target)
        + F.cross_entropy(surface_to_residue, target)
    )


def local_contrastive_loss(
    output: dict[str, torch.Tensor], temperature: float, num_negatives: int, negative_mode: str
) -> torch.Tensor:
    local_batch = output.get("local_residue_batch")
    if local_batch is None:
        return output["reconstructed_attributes"].sum() * 0.0
    losses = []
    for protein_index in local_batch.unique(sorted=True):
        mask = local_batch == protein_index
        if int(mask.sum()) < 2:
            continue
        protein_output = {
            "residue_local_embedding": output["residue_local_embedding"][mask],
            "surface_local_embedding": output["surface_local_embedding"][mask],
            "local_residue_positions": output["local_residue_positions"][mask],
            "reconstructed_attributes": output["reconstructed_attributes"],
        }
        losses.append(
            _single_protein_local_contrastive_loss(
                protein_output, temperature, num_negatives, negative_mode
            )
        )
    return torch.stack(losses).mean() if losses else output["reconstructed_attributes"].sum() * 0.0


@torch.no_grad()
def _single_protein_local_retrieval_accuracy(
    output: dict[str, torch.Tensor], num_negatives: int, negative_mode: str
) -> float:
    residue = output.get("residue_local_embedding")
    surface = output.get("surface_local_embedding")
    if residue is None or surface is None or residue.size(0) < 2:
        return 0.0
    residue = F.normalize(residue.float(), dim=-1, eps=1e-6)
    surface = F.normalize(surface.float(), dim=-1, eps=1e-6)
    negative_ids = _local_negative_indices(output, num_negatives, negative_mode)
    positive = (residue * surface).sum(dim=-1, keepdim=True)
    forward = torch.cat(
        [positive, (residue.unsqueeze(1) * surface[negative_ids]).sum(dim=-1)], dim=1
    )
    reverse = torch.cat(
        [positive, (surface.unsqueeze(1) * residue[negative_ids]).sum(dim=-1)], dim=1
    )
    return float((0.5 * ((forward.argmax(dim=1) == 0).float().mean() + (reverse.argmax(dim=1) == 0).float().mean())).cpu())


@torch.no_grad()
def local_retrieval_accuracy(
    output: dict[str, torch.Tensor], num_negatives: int, negative_mode: str
) -> float:
    local_batch = output.get("local_residue_batch")
    if local_batch is None:
        return 0.0
    accuracies = []
    for protein_index in local_batch.unique(sorted=True):
        mask = local_batch == protein_index
        if int(mask.sum()) < 2:
            continue
        protein_output = {
            "residue_local_embedding": output["residue_local_embedding"][mask],
            "surface_local_embedding": output["surface_local_embedding"][mask],
            "local_residue_positions": output["local_residue_positions"][mask],
        }
        accuracies.append(
            _single_protein_local_retrieval_accuracy(
                protein_output, num_negatives, negative_mode
            )
        )
    return float(np.mean(accuracies)) if accuracies else 0.0


def assert_finite(name: str, tensor: torch.Tensor, protein_ids: list[str] | None = None) -> None:
    """Fail at the first invalid tensor instead of writing NaN histories."""
    if torch.isfinite(tensor).all():
        return
    context = f" proteins={protein_ids}" if protein_ids else ""
    finite_ratio = float(torch.isfinite(tensor).float().mean().detach().cpu())
    raise FloatingPointError(f"Non-finite {name}; finite_ratio={finite_ratio:.6f}{context}")


def nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


@torch.no_grad()
def evaluate(model, loader, device: torch.device, cfg: dict, amp: bool, amp_dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    total_reconstruction = 0.0
    total_global = 0.0
    total_local = 0.0
    total_decorrelation = 0.0
    total_retrieval = 0.0
    count = 0
    temperature = float(cfg["loss"].get("temperature", 0.2))
    global_weight = float(cfg["loss"].get("global_contrastive_weight", cfg["loss"].get("contrastive_weight", 0.0)))
    local_weight = float(cfg["loss"].get("local_contrastive_weight", 0.0))
    decorrelation_weight = float(cfg["loss"].get("decorrelation_weight", 0.0))
    reconstruction_weight = float(cfg["loss"].get("reconstruction_weight", 1.0))
    local_negatives = int(cfg["loss"].get("local_negatives", 64))
    local_negative_mode = str(cfg["loss"].get("local_negative_mode", "random"))
    queue = ContrastiveQueue(int(cfg["loss"].get("queue_size", 256)), temperature, device)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with autocast(device, amp, amp_dtype):
            output = model(batch, float(cfg["loss"]["mask_ratio"]))
            reconstruction = masked_reconstruction_loss(output)
        global_contrast = (
            queue.loss(output["residue_embedding"], output["surface_embedding"])
            if global_weight > 0.0
            else reconstruction.new_zeros(())
        )
        local_contrast = (
            local_contrastive_loss(output, temperature, local_negatives, local_negative_mode)
            if local_weight > 0.0
            else reconstruction.new_zeros(())
        )
        decorrelation = output["decorrelation_loss"].float()
        assert_finite("validation reconstruction", reconstruction, batch.get("protein_ids"))
        assert_finite("validation global contrast", global_contrast, batch.get("protein_ids"))
        assert_finite("validation local contrast", local_contrast, batch.get("protein_ids"))
        if global_weight > 0.0:
            queue.enqueue(output["residue_embedding"], output["surface_embedding"])
        total_reconstruction += float(reconstruction.detach().cpu())
        total_global += float(global_contrast.detach().cpu())
        total_local += float(local_contrast.detach().cpu())
        total_decorrelation += float(decorrelation.detach().cpu())
        total_retrieval += local_retrieval_accuracy(output, local_negatives, local_negative_mode)
        count += 1
    divisor = max(count, 1)
    metrics = {
        "val_reconstruction": total_reconstruction / divisor,
        "val_global_contrast": total_global / divisor,
        "val_local_contrast": total_local / divisor,
        "val_decorrelation": total_decorrelation / divisor,
        "val_local_retrieval_accuracy": total_retrieval / divisor,
    }
    metrics["val_total"] = (
        reconstruction_weight * metrics["val_reconstruction"]
        + global_weight * metrics["val_global_contrast"]
        + local_weight * metrics["val_local_contrast"]
        + decorrelation_weight * metrics["val_decorrelation"]
    )
    return metrics


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg["run"].get("seed", 42)))
    device = torch.device("cuda" if cfg["run"].get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    amp = bool(cfg["training"].get("amp", True))
    amp_dtype = resolve_amp_dtype(str(cfg["training"].get("amp_dtype", "float16")), device)
    root = resolve_path(cfg["data"]["processed_root"])
    all_train = SurfPDIDataset(root, cfg["data"].get("cv_split", ["train573"]))
    folds, fold_method = make_folds(all_train.records, cfg)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_path(cfg["run"].get("output_root", "results/pretraining"))
    output_dir = resolve_path(args.output_dir) if args.output_dir else output_root / f"{cfg['run']['name']}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cfg_path, output_dir / "config.yaml")
    print(f"protocol=DNA-573_{len(folds)}fold_ssl_pretraining independent_tests=none fold_method={fold_method} output_dir={output_dir}")

    for fold_id, (train_indices, val_indices) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir()
        train_records = [all_train.records[index] for index in train_indices]
        val_records = [all_train.records[index] for index in val_indices]
        if args.dry_run_largest:
            dry_batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
            train_records = sorted(
                train_records,
                key=lambda row: int(row["num_surface_points"]),
                reverse=True,
            )[:dry_batch_size]
        train_set = SurfPDIDataset(root, records=train_records, max_samples=args.max_train_samples)
        val_set = SurfPDIDataset(root, records=val_records, max_samples=args.max_val_samples)
        train_loader = make_loader(train_set, cfg.get("loader", {}), True)
        val_loader = make_loader(val_set, cfg.get("loader", {}), False)
        (fold_dir / "fold_split.json").write_text(json.dumps({
            "fold": fold_id,
            "train_count": len(train_set),
            "val_count": len(val_set),
            "labels_used_by_pretraining": False,
        }, indent=2), encoding="utf-8")

        model = FullSurfaceMultimodalPretrainer(**cfg["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)))
        scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda" and amp_dtype == torch.float16)
        print(f"fold={fold_id} train={len(train_set)} val={len(val_set)} amp={amp} amp_dtype={amp_dtype}")

        if args.dry_run:
            batch = move_batch_to_device(next(iter(train_loader)), device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, amp, amp_dtype):
                output = model(batch, float(cfg["loss"]["mask_ratio"]))
                reconstruction = masked_reconstruction_loss(output)
            global_weight = float(cfg["loss"].get("global_contrastive_weight", cfg["loss"].get("contrastive_weight", 0.0)))
            local_weight = float(cfg["loss"].get("local_contrastive_weight", 0.0))
            decorrelation_weight = float(cfg["loss"].get("decorrelation_weight", 0.0))
            reconstruction_weight = float(cfg["loss"].get("reconstruction_weight", 1.0))
            local_contrast = local_contrastive_loss(output, float(cfg["loss"].get("temperature", 0.2)), int(cfg["loss"].get("local_negatives", 64)), str(cfg["loss"].get("local_negative_mode", "random"))) if local_weight > 0.0 else reconstruction.new_zeros(())
            loss = reconstruction_weight * reconstruction + local_weight * local_contrast + decorrelation_weight * output["decorrelation_loss"].float()
            scaler.scale(loss).backward()
            peak_memory_gib = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"dry_run_ok proteins={batch['protein_ids']} surface_points={batch['x'].shape[0]} "
                f"residues={batch['residue_pos'].shape[0]} local_pairs={output.get('num_local_pairs', 0)} "
                f"reconstruction={reconstruction.item():.4f} local={local_contrast.item():.4f} "
                f"total={loss.item():.4f} peak_allocated_gib={peak_memory_gib:.2f}"
            )
            return

        best_val = float("inf")
        bad_epochs = 0
        history = []
        max_epochs = int(args.epochs or cfg["training"].get("epochs", 50))
        for epoch in range(1, max_epochs + 1):
            model.train()
            queue = ContrastiveQueue(int(cfg["loss"].get("queue_size", 256)), float(cfg["loss"].get("temperature", 0.1)), device)
            target_global_weight = float(cfg["loss"].get("global_contrastive_weight", cfg["loss"].get("contrastive_weight", 0.0)))
            target_local_weight = float(cfg["loss"].get("local_contrastive_weight", 0.0))
            decorrelation_weight = float(cfg["loss"].get("decorrelation_weight", 0.0))
            reconstruction_weight = float(cfg["loss"].get("reconstruction_weight", 1.0))
            warmup_epochs = int(cfg["loss"].get("contrast_warmup_epochs", 0))
            warmup_scale = 1.0
            if warmup_epochs > 0:
                warmup_scale = min(1.0, max(0.0, (epoch - 1) / warmup_epochs))
            global_weight = target_global_weight * warmup_scale
            local_weight = target_local_weight * warmup_scale
            train_total = 0.0
            train_reconstruction = 0.0
            train_global = 0.0
            train_local = 0.0
            train_decorrelation = 0.0
            nonfinite_gradient_skips = 0
            max_nonfinite_gradient_skips = int(
                cfg["training"].get("max_nonfinite_gradient_skips_per_epoch", 3)
            )
            steps = 0
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, amp, amp_dtype):
                    output = model(batch, float(cfg["loss"]["mask_ratio"]))
                    reconstruction = masked_reconstruction_loss(output)
                # Contrast is deliberately outside autocast: it is a small
                # global loss and float32 is materially more stable here.
                # A zero-weight diagnostic must skip it entirely; multiplying
                # a non-finite contrast term by zero would still yield NaN.
                global_contrast = queue.loss(output["residue_embedding"], output["surface_embedding"]) if global_weight > 0.0 else reconstruction.detach().new_zeros(())
                local_contrast = local_contrastive_loss(output, float(cfg["loss"].get("temperature", 0.2)), int(cfg["loss"].get("local_negatives", 64)), str(cfg["loss"].get("local_negative_mode", "random"))) if local_weight > 0.0 else reconstruction.detach().new_zeros(())
                decorrelation = output["decorrelation_loss"].float()
                loss = reconstruction_weight * reconstruction + global_weight * global_contrast + local_weight * local_contrast + decorrelation_weight * decorrelation
                assert_finite("surface embedding", output["surface_embedding"], batch.get("protein_ids"))
                assert_finite("residue embedding", output["residue_embedding"], batch.get("protein_ids"))
                assert_finite("reconstruction loss", reconstruction, batch.get("protein_ids"))
                assert_finite("global contrastive loss", global_contrast, batch.get("protein_ids"))
                assert_finite("local contrastive loss", local_contrast, batch.get("protein_ids"))
                assert_finite("decorrelation loss", decorrelation, batch.get("protein_ids"))
                assert_finite("total loss", loss, batch.get("protein_ids"))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                bad_gradients = nonfinite_gradient_names(model)
                if bad_gradients:
                    # A rare FP16 overflow is recoverable: GradScaler has
                    # already found it during unscale_ and lowers its scale on
                    # update(). Do not send inf through gradient clipping.
                    if not scaler.is_enabled():
                        raise FloatingPointError(
                            "Non-finite gradients without active GradScaler; "
                            f"first_parameters={bad_gradients[:8]} proteins={batch.get('protein_ids')}"
                        )
                    nonfinite_gradient_skips += 1
                    print(
                        "amp_overflow_skip "
                        f"epoch={epoch:03d} count={nonfinite_gradient_skips} "
                        f"first_parameters={bad_gradients[:4]} proteins={batch.get('protein_ids')}"
                    )
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if nonfinite_gradient_skips > max_nonfinite_gradient_skips:
                        raise FloatingPointError(
                            "Too many non-finite gradient skips in one epoch; "
                            f"count={nonfinite_gradient_skips} limit={max_nonfinite_gradient_skips}"
                        )
                    continue
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["training"].get("grad_clip_norm", 1.0)),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                if global_weight > 0.0:
                    queue.enqueue(output["residue_embedding"], output["surface_embedding"])
                train_total += float(loss.detach().cpu())
                train_reconstruction += float(reconstruction.detach().cpu())
                train_global += float(global_contrast.detach().cpu())
                train_local += float(local_contrast.detach().cpu())
                train_decorrelation += float(decorrelation.detach().cpu())
                steps += 1

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                cfg,
                amp,
                amp_dtype,
            )
            row = {
                "epoch": epoch,
                "train_loss": train_total / max(steps, 1),
                "train_reconstruction": train_reconstruction / max(steps, 1),
                "train_global_contrast": train_global / max(steps, 1),
                "train_local_contrast": train_local / max(steps, 1),
                "train_decorrelation": train_decorrelation / max(steps, 1),
                "global_contrastive_weight": global_weight,
                "local_contrastive_weight": local_weight,
                "nonfinite_gradient_skips": nonfinite_gradient_skips,
                **val_metrics,
            }
            history.append(row)
            (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                "epoch={:03d} train_loss={:.4f} reconstruction={:.4f} global={:.4f} local={:.4f} "
                "global_weight={:.4f} local_weight={:.4f} overflow_skips={} val_reconstruction={:.4f} "
                "val_local={:.4f} val_retrieval={:.4f}".format(
                    epoch, row["train_loss"], row["train_reconstruction"], row["train_global_contrast"],
                    row["train_local_contrast"], global_weight, local_weight, nonfinite_gradient_skips,
                    row["val_reconstruction"], row["val_local_contrast"], row["val_local_retrieval_accuracy"]
                )
            )
            checkpoint_metric = str(cfg["training"].get("checkpoint_metric", "val_reconstruction"))
            checkpoint_value = float(row[checkpoint_metric])
            if checkpoint_value < best_val:
                best_val = checkpoint_value
                bad_epochs = 0
                torch.save({"epoch": epoch, "model": model.state_dict(), **val_metrics, "checkpoint_metric": checkpoint_metric, "config": cfg}, fold_dir / "best_pretrain.pt")
            else:
                bad_epochs += 1
            if bad_epochs >= int(cfg["training"].get("patience", 10)):
                print(f"Early stopping at epoch={epoch}; best_{checkpoint_metric}={best_val:.4f}")
                break

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
