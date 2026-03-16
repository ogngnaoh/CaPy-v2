"""Training loop with validation, checkpointing, and early stopping (FR-7)."""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.evaluation.diagnostics import compute_alignment, compute_uniformity
from src.evaluation.retrieval import (
    compute_all_compound_retrieval_metrics,
    compute_all_retrieval_metrics,
)
from src.models.losses import SigLIPLoss, VICRegLoss, compute_total_loss
from src.utils.config import get_git_hash, save_config_yaml
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODALITY_ORDER = {"mol": 0, "morph": 1, "expr": 2}


class Trainer:
    """Training orchestrator (FR-7.1 through FR-7.4).

    Handles:
    - Training loop with gradient clipping and mixed precision
    - Validation evaluation each epoch
    - Checkpointing on best val mean_R@10
    - Early stopping with configurable patience
    - Collapse detection via uniformity monitoring
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config,
        siglip_loss_fn: SigLIPLoss | dict[str, SigLIPLoss],
        vicreg_loss_fn: VICRegLoss,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.siglip_loss_fn = siglip_loss_fn
        self.vicreg_loss_fn = vicreg_loss_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Move SigLIP instance(s) to device
        if isinstance(self.siglip_loss_fn, dict):
            for fn in self.siglip_loss_fn.values():
                fn.to(self.device)
        else:
            self.siglip_loss_fn.to(self.device)

        # Mixed precision (bfloat16 on CUDA only)
        self.use_amp = config.training.mixed_precision and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        # Early stopping state
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

        # Checkpoint path
        checkpoint_dir = Path(getattr(config, "checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = (
            checkpoint_dir / f"{config.model.name}_seed{config.seed}.pt"
        )

        # Cache config values
        self.vicreg_lambda = config.training.loss.vicreg_lambda
        self.clip_max_norm = config.training.gradient_clip_max_norm
        self.max_epochs = config.training.epochs
        self.patience = config.training.patience
        self.modalities = list(config.model.modalities)

        # Pair weights from config
        pw_cfg = getattr(config.training.loss, "pair_weights", None)
        self.pair_weights = (
            OmegaConf.to_container(pw_cfg, resolve=True) if pw_cfg else {}
        )

        # Epoch history for training curves (FR-8.4)
        self.epoch_history: list[dict] = []

        # VICReg persistence tracking (Edge 5.2)
        self._vicreg_high_epochs: dict[str, int] = {}

    def train(self) -> dict:
        """Run full training loop. Returns dict of best metrics."""
        logger.info(
            "Starting training: %d epochs, patience=%d, device=%s",
            self.max_epochs,
            self.patience,
            self.device,
        )
        best_metrics: dict = {}

        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Scheduler step (per-epoch)
            self.scheduler.step()

            # Record epoch history for training curves
            self.epoch_history.append({"epoch": epoch, **train_metrics, **val_metrics})

            # Early stopping + checkpointing (prefer compound-level metric)
            mean_r10 = val_metrics.get(
                "compound/mean_R@10", val_metrics.get("mean_R@10", 0.0)
            )
            should_stop = self.check_early_stopping(mean_r10)

            # patience_counter == 0 means improvement was detected
            if self.patience_counter == 0:
                self.best_epoch = epoch
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metrics)

            if should_stop:
                logger.info(
                    "Early stopping at epoch %d (patience=%d exhausted)",
                    epoch,
                    self.patience,
                )
                break

            # VICReg persistence tracking (Edge 5.2)
            self._check_vicreg_persistence(train_metrics)

            # Collapse detection
            uniformity_scores = {
                k: v for k, v in val_metrics.items() if k.startswith("uniform_")
            }
            self.check_collapse(uniformity_scores)

        logger.info(
            "Training complete. Best mean_R@10=%.4f at epoch %d. " "Checkpoint: %s",
            self.best_metric,
            self.best_epoch,
            self.checkpoint_path,
        )

        self._save_run_metrics(best_metrics)

        return {
            "best_mean_R@10": self.best_metric,
            "best_epoch": self.best_epoch,
            **best_metrics,
        }

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns loss dict."""
        self.model.train()
        epoch_losses: dict[str, float] = {}
        n_batches = 0

        pair_weights = dict(self.pair_weights)

        for batch in self.train_loader:
            batch_gpu = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            self.optimizer.zero_grad()

            # Extract compound IDs for multi-positive SigLIP (OPEN-1)
            compound_ids = None
            if "metadata" in batch:
                compound_ids = [m["compound_id"] for m in batch["metadata"]]

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                embeddings, encoder_outputs = self.model(batch_gpu)

                total_loss, loss_dict = compute_total_loss(
                    embeddings,
                    self.siglip_loss_fn,
                    self.vicreg_loss_fn,
                    self.vicreg_lambda,
                    encoder_outputs=encoder_outputs,
                    compound_ids=compound_ids,
                    pair_weights=pair_weights,
                )

            # NaN detection (FSD edge case 5.2)
            if torch.isnan(total_loss):
                msg = (
                    f"NaN loss detected at epoch {epoch}, "
                    f"batch {n_batches + 1}. "
                    "Check data preprocessing or reduce learning rate."
                )
                raise RuntimeError(msg)

            total_loss.backward()

            # Gradient clipping
            all_params = list(self.model.parameters()) + list(self._siglip_parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params, self.clip_max_norm
            ).item()

            # Gradient explosion warning (FSD edge case 5.2)
            if grad_norm > 10 * self.clip_max_norm:
                logger.warning(
                    "Gradient explosion detected (norm=%.2f). " "Clipping applied.",
                    grad_norm,
                )

            self.optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            epoch_losses["grad_norm"] = epoch_losses.get("grad_norm", 0.0) + grad_norm
            n_batches += 1

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        logger.info(
            "Epoch %d | train_loss=%.4f | grad_norm=%.4f",
            epoch,
            avg.get("loss_total", 0.0),
            avg.get("grad_norm", 0.0),
        )
        return avg

    def validate(self, epoch: int) -> dict:
        """Run validation evaluation. Returns metrics dict."""
        self.model.eval()
        all_embeddings: dict[str, list[torch.Tensor]] = {m: [] for m in self.modalities}
        all_compound_ids: list[str] = []
        val_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch_gpu = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                # Extract compound IDs for multi-positive SigLIP
                compound_ids = None
                if "metadata" in batch:
                    compound_ids = [m["compound_id"] for m in batch["metadata"]]
                    all_compound_ids.extend(compound_ids)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    embeddings, encoder_outputs = self.model(batch_gpu)
                    total_loss, _ = compute_total_loss(
                        embeddings,
                        self.siglip_loss_fn,
                        self.vicreg_loss_fn,
                        self.vicreg_lambda,
                        encoder_outputs=encoder_outputs,
                        compound_ids=compound_ids,
                        pair_weights=self.pair_weights,
                    )

                for m in self.modalities:
                    all_embeddings[m].append(embeddings[m].cpu())
                val_loss_sum += total_loss.item()
                n_batches += 1

        # Concat embeddings
        all_emb = {m: torch.cat(tensors) for m, tensors in all_embeddings.items()}

        # Row-level retrieval metrics (FR-8.1)
        retrieval_metrics = compute_all_retrieval_metrics(all_emb)

        # Compound-level retrieval (deduped, primary metric)
        compound_retrieval = {}
        if all_compound_ids:
            compound_retrieval = compute_all_compound_retrieval_metrics(
                all_emb, all_compound_ids
            )

        # Alignment + uniformity (FR-8.2)
        diagnostics: dict[str, float] = {}
        modalities = sorted(all_emb.keys(), key=lambda m: _MODALITY_ORDER[m])
        for m_a, m_b in itertools.combinations(modalities, 2):
            diagnostics[f"align_{m_a}_{m_b}"] = compute_alignment(
                all_emb[m_a], all_emb[m_b]
            )
        for m in modalities:
            diagnostics[f"uniform_{m}"] = compute_uniformity(all_emb[m])

        metrics = {
            **retrieval_metrics,
            **compound_retrieval,
            **diagnostics,
            "val_loss": val_loss_sum / max(n_batches, 1),
        }

        # Use compound-level R@10 as primary metric if available
        primary_r10 = metrics.get("compound/mean_R@10", metrics.get("mean_R@10", 0.0))
        logger.info(
            "Epoch %d | val_loss=%.4f | compound_R@10=%.4f | row_R@10=%.4f",
            epoch,
            metrics["val_loss"],
            primary_r10,
            metrics.get("mean_R@10", 0.0),
        )

        # Per-direction R@10 summary
        dir_strs = []
        for key, val in sorted(metrics.items()):
            if key.startswith("compound/") and key.endswith("/R@10"):
                parts = key.split("/")
                if len(parts) == 3 and "->" in parts[1]:
                    direction = parts[1]
                    dir_strs.append(f"{direction}={val:.3f}")
        if dir_strs:
            logger.info("  Per-direction R@10: %s", " | ".join(dir_strs))

        return metrics

    def _siglip_parameters(self) -> list:
        """Collect all SigLIP parameters (works for both single and dict)."""
        if isinstance(self.siglip_loss_fn, dict):
            params = []
            for fn in self.siglip_loss_fn.values():
                params.extend(fn.parameters())
            return params
        return list(self.siglip_loss_fn.parameters())

    def _siglip_state_dicts(self) -> dict:
        """Get SigLIP state dict(s) for checkpointing."""
        if isinstance(self.siglip_loss_fn, dict):
            return {k: fn.state_dict() for k, fn in self.siglip_loss_fn.items()}
        return {"shared": self.siglip_loss_fn.state_dict()}

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save model checkpoint (FR-7.2)."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "siglip_state_dicts": self._siglip_state_dicts(),
                "epoch": epoch,
                "best_metric": self.best_metric,
                "metrics": metrics,
                "config": OmegaConf.to_container(self.config, resolve=True),
                "epoch_history": self.epoch_history,
            },
            self.checkpoint_path,
        )
        # Save config YAML alongside checkpoint (FR-10.1)
        config_path = Path(self.checkpoint_path).with_suffix(".yaml")
        save_config_yaml(self.config, config_path)

        logger.info(
            "Checkpoint saved: %s (epoch=%d, mean_R@10=%.4f)",
            self.checkpoint_path,
            epoch,
            self.best_metric,
        )

    def check_early_stopping(self, metric: float) -> bool:
        """Check early stopping condition (FR-7.3). Returns True if should stop."""
        if metric > self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.patience

    def check_collapse(self, uniformity_scores: dict) -> bool:
        """Log collapse warnings if uniformity > -0.5 (FR-7.4).

        Returns True if collapse detected in any modality.
        """
        collapse_detected = False
        for key, score in uniformity_scores.items():
            if score > -0.5:
                collapse_detected = True
                modality = key.replace("uniform_", "")
                logger.warning(
                    "COLLAPSE WARNING: %s uniformity=%.4f "
                    "(threshold=-0.5). "
                    "Encoder may be collapsing.",
                    modality,
                    score,
                )
        return collapse_detected

    def _check_vicreg_persistence(self, train_metrics: dict) -> None:
        """Warn if VICReg variance loss stays high for >=20 epochs (Edge 5.2)."""
        for key, val in train_metrics.items():
            if not key.startswith("vicreg_"):
                continue
            modality = key.replace("vicreg_", "")
            if val > 0.5:
                self._vicreg_high_epochs[modality] = (
                    self._vicreg_high_epochs.get(modality, 0) + 1
                )
                if self._vicreg_high_epochs[modality] >= 20:
                    logger.warning(
                        "VICReg variance loss persistently high for %s "
                        "(>0.5 for %d epochs). Embeddings may have low "
                        "variance despite regularization.",
                        modality,
                        self._vicreg_high_epochs[modality],
                    )
            else:
                self._vicreg_high_epochs[modality] = 0

    def _save_run_metrics(self, best_metrics: dict) -> None:
        """Save structured per-run metrics JSON to results/ (FR-11.0)."""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        config_name = self.config.model.name
        seed = self.config.seed
        out_path = results_dir / f"{config_name}_seed{seed}_metrics.json"

        payload = {
            "config_name": config_name,
            "seed": seed,
            "git_hash": get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "metrics": best_metrics,
            "config": OmegaConf.to_container(self.config, resolve=True),
            "epoch_history": self.epoch_history,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Run metrics saved: %s", out_path)
