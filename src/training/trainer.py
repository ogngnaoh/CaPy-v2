"""Training loop with validation, checkpointing, and early stopping (FR-7)."""

import itertools
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.evaluation.diagnostics import compute_alignment, compute_uniformity
from src.evaluation.retrieval import compute_all_retrieval_metrics
from src.models.losses import SigLIPLoss, VICRegLoss, compute_total_loss
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
    - W&B logging of all metrics
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config,
        siglip_loss_fn: SigLIPLoss,
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.siglip_loss_fn.to(self.device)

        # Mixed precision (bfloat16 on CUDA only)
        self.use_amp = (
            config.training.mixed_precision and torch.cuda.is_available()
        )
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        # Early stopping state
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

        # Checkpoint path
        checkpoint_dir = Path(
            getattr(config, "checkpoint_dir", "checkpoints")
        )
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

        # Step counter for W&B
        self.global_step = 0

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

            # W&B logging
            self._log_to_wandb(epoch, train_metrics, val_metrics)

            # Early stopping + checkpointing
            mean_r10 = val_metrics.get("mean_R@10", 0.0)
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
                k: v
                for k, v in val_metrics.items()
                if k.startswith("uniform_")
            }
            collapse = self.check_collapse(uniformity_scores)
            if collapse:
                self._log_collapse_to_wandb()

        logger.info(
            "Training complete. Best mean_R@10=%.4f at epoch %d. "
            "Checkpoint: %s",
            self.best_metric,
            self.best_epoch,
            self.checkpoint_path,
        )
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

        for batch in self.train_loader:
            batch_gpu = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            self.optimizer.zero_grad()

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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.siglip_loss_fn.parameters()),
                self.clip_max_norm,
            ).item()

            # Gradient explosion warning (FSD edge case 5.2)
            if grad_norm > 10 * self.clip_max_norm:
                logger.warning(
                    "Gradient explosion detected (norm=%.2f). "
                    "Clipping applied.",
                    grad_norm,
                )

            self.optimizer.step()
            self.global_step += 1

            # Accumulate
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            epoch_losses["grad_norm"] = (
                epoch_losses.get("grad_norm", 0.0) + grad_norm
            )
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
        all_embeddings: dict[str, list[torch.Tensor]] = {
            m: [] for m in self.modalities
        }
        val_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch_gpu = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

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
                    )

                for m in self.modalities:
                    all_embeddings[m].append(embeddings[m].cpu())
                val_loss_sum += total_loss.item()
                n_batches += 1

        # Concat embeddings
        all_emb = {m: torch.cat(tensors) for m, tensors in all_embeddings.items()}

        # Retrieval metrics (FR-8.1)
        retrieval_metrics = compute_all_retrieval_metrics(all_emb)

        # Alignment + uniformity (FR-8.2)
        diagnostics: dict[str, float] = {}
        modalities = sorted(
            all_emb.keys(), key=lambda m: _MODALITY_ORDER[m]
        )
        for m_a, m_b in itertools.combinations(modalities, 2):
            diagnostics[f"align_{m_a}_{m_b}"] = compute_alignment(
                all_emb[m_a], all_emb[m_b]
            )
        for m in modalities:
            diagnostics[f"uniform_{m}"] = compute_uniformity(all_emb[m])

        metrics = {
            **retrieval_metrics,
            **diagnostics,
            "val_loss": val_loss_sum / max(n_batches, 1),
        }

        logger.info(
            "Epoch %d | val_loss=%.4f | mean_R@10=%.4f",
            epoch,
            metrics["val_loss"],
            metrics.get("mean_R@10", 0.0),
        )
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save model checkpoint (FR-7.2)."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "siglip_state_dict": self.siglip_loss_fn.state_dict(),
                "epoch": epoch,
                "best_metric": self.best_metric,
                "config": OmegaConf.to_container(
                    self.config, resolve=True
                ),
            },
            self.checkpoint_path,
        )
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

    def _log_collapse_to_wandb(self) -> None:
        """Log collapse_warning=True to W&B when collapse detected."""
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {"collapse_warning": True}, step=self.global_step
                )
        except ImportError:
            pass

    def _log_to_wandb(
        self, epoch: int, train_metrics: dict, val_metrics: dict
    ) -> None:
        """Log epoch metrics to W&B if available."""
        try:
            import wandb

            if wandb.run is None:
                return
        except ImportError:
            return

        log_dict = {
            "epoch": epoch,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        for k, v in train_metrics.items():
            log_dict[f"train/{k}"] = v
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v
        wandb.log(log_dict, step=self.global_step)
