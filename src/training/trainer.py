"""
Input:
    ProteinInterfacePredictor, DataLoaders, optimizer, criterion, and config.

Output:
    Training logs, best checkpoint, and final metric dictionaries.
    The trainer keeps memory use conservative for full surface graphs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from src.data.collate import move_batch_to_device
from src.training.metrics import binary_classification_metrics, positive_probabilities


def _autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        output_dir: str | Path,
        amp: bool = True,
        grad_clip_norm: float | None = 1.0,
        monitor: str = "mcc",
        monitor_mode: str = "max",
        checkpoint_monitors: list[str] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.amp = bool(amp and device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.checkpoint_monitors = list(dict.fromkeys([monitor] + list(checkpoint_monitors or [])))
        self.scaler = _make_grad_scaler(self.amp)
        self.history: list[dict[str, Any]] = []

    def train_one_epoch(self, loader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_residues = 0
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []

        for step, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, self.device)
            labels = batch["residue_labels"]

            self.optimizer.zero_grad(set_to_none=True)
            try:
                with _autocast_context(self.device, self.amp):
                    logits = self.model(batch)
                    loss = self.criterion(logits, labels)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    proteins = ", ".join(batch.get("protein_ids", []))
                    raise RuntimeError(f"CUDA OOM at epoch={epoch}, step={step}, proteins={proteins}") from exc
                raise

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            n = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_residues += n
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

        metrics = binary_classification_metrics(torch.cat(logits_all), torch.cat(labels_all))
        metrics["loss"] = total_loss / max(total_residues, 1)
        return metrics

    @torch.no_grad()
    def predict(self, loader) -> dict[str, Any]:
        self.model.eval()
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []

        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            labels = batch["residue_labels"]
            with _autocast_context(self.device, self.amp):
                logits = self.model(batch)

            logits_cpu = logits.detach().cpu()
            labels_cpu = labels.detach().cpu()
            probs_cpu = positive_probabilities(logits_cpu)
            logits_all.append(logits_cpu)
            labels_all.append(labels_cpu)

            for protein_id, (start, end) in zip(batch["protein_ids"], batch["residue_slices"]):
                for local_idx, global_idx in enumerate(range(start, end)):
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "residue_index": local_idx,
                            "label": int(labels_cpu[global_idx].item()),
                            "probability": float(probs_cpu[global_idx].item()),
                        }
                    )

        return {"logits": torch.cat(logits_all), "labels": torch.cat(labels_all), "rows": rows}

    @torch.no_grad()
    def evaluate(self, loader, threshold: float = 0.5) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_residues = 0
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []

        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            labels = batch["residue_labels"]
            with _autocast_context(self.device, self.amp):
                logits = self.model(batch)
                loss = self.criterion(logits, labels)

            n = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_residues += n
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

        metrics = binary_classification_metrics(torch.cat(logits_all), torch.cat(labels_all), threshold=threshold)
        metrics["loss"] = total_loss / max(total_residues, 1)
        return metrics

    def is_better(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.monitor_mode == "min":
            return current < best
        return current > best

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def fit(self, train_loader, val_loader, epochs: int, patience: int | None = None) -> dict[str, Any]:
        best = {
            monitor: {
                "best_score": None,
                "best_epoch": 0,
                "best_path": str(self.output_dir / f"best_{monitor}.pt"),
                "best_metrics": None,
            }
            for monitor in self.checkpoint_monitors
        }
        primary_best_score: float | None = None
        primary_best_epoch = 0
        bad_epochs = 0
        best_path = self.output_dir / "best.pt"

        for epoch in range(1, epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_started = time.perf_counter()
            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            else:
                peak_memory_gb = 0.0
            epoch_seconds = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "epoch_seconds": epoch_seconds,
                "peak_memory_gb": peak_memory_gb,
            }
            self.history.append(row)
            (self.output_dir / "history.json").write_text(
                json.dumps(self.history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            primary_improved = False
            for monitor in self.checkpoint_monitors:
                score = float(val_metrics[monitor])
                if self.is_better(score, best[monitor]["best_score"]):
                    best[monitor]["best_score"] = score
                    best[monitor]["best_epoch"] = epoch
                    best[monitor]["best_metrics"] = val_metrics
                    self.save_checkpoint(best[monitor]["best_path"], epoch, val_metrics)

            primary_score = float(val_metrics[self.monitor])
            if self.is_better(primary_score, primary_best_score):
                primary_best_score = primary_score
                primary_best_epoch = epoch
                primary_improved = True
                bad_epochs = 0
                self.save_checkpoint(best_path, epoch, val_metrics)
            else:
                bad_epochs += 1

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f} "
                f"val_mcc={val_metrics['mcc']:.4f} "
                f"best_{self.monitor}={primary_best_score:.4f} "
                f"epoch_seconds={epoch_seconds:.1f} peak_memory_gb={peak_memory_gb:.2f}"
            )

            if patience is not None and bad_epochs >= patience:
                print(f"Early stopping at epoch={epoch}; best_{self.monitor}_epoch={primary_best_epoch}.")
                break

        return {
            "primary_monitor": self.monitor,
            "primary_best_epoch": primary_best_epoch,
            "primary_best_score": primary_best_score,
            "primary_best_path": str(best_path),
            "best": best,
        }
