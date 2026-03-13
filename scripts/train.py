"""Entry point for model training (make train).

Usage:
    python scripts/train.py
    python scripts/train.py model=bi_mol_morph training.batch_size=128 seed=42
"""

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:
    # Hydra changes CWD — restore original so data paths resolve
    os.chdir(hydra.utils.get_original_cwd())

    logger.info("Config:\n%s", OmegaConf.to_yaml(config))

    # Seed all random sources
    seed_everything(config.seed)

    # W&B init (optional)
    try:
        import wandb

        wandb.init(
            project=config.project_name,
            name=f"{config.model.name}_seed{config.seed}",
            config=OmegaConf.to_container(config, resolve=True),
            tags=[
                f"config={config.model.name}",
                f"seed={config.seed}",
                "dataset=lincs",
            ],
        )
    except ImportError:
        logger.warning("wandb not available, skipping experiment tracking")

    # Build dataloaders
    from src.data.dataset import build_dataloaders

    train_loader, val_loader, _ = build_dataloaders(config)

    if len(train_loader) == 0:
        msg = (
            f"batch_size {config.training.batch_size} > dataset size. "
            "Reduce batch_size."
        )
        raise RuntimeError(msg)

    # Override encoder input_dims from actual processed data
    import json

    with open(config.data.output.feature_columns_path) as f:
        feature_cols = json.load(f)

    dim_map = {
        "morph_encoder": len(feature_cols["morph_features"]),
        "expr_encoder": len(feature_cols["expr_features"]),
    }
    with open_dict(config):
        for enc_key, actual_dim in dim_map.items():
            if enc_key.split("_")[0] in config.model.modalities:
                cfg_dim = config.model[enc_key].input_dim
                if cfg_dim != actual_dim:
                    logger.info(
                        "Overriding %s.input_dim: %d -> %d (from data)",
                        enc_key,
                        cfg_dim,
                        actual_dim,
                    )
                    config.model[enc_key].input_dim = actual_dim

    # Build model
    from src.models.capy import CaPyModel

    model = CaPyModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d parameters)", config.model.name, n_params)

    # Build loss functions
    from src.models.losses import SigLIPLoss, VICRegLoss

    siglip_loss_fn = SigLIPLoss(
        bias_init=config.training.loss.siglib_bias_init
    )
    vicreg_loss_fn = VICRegLoss()

    # Build optimizer (model + SigLIP learnable bias)
    all_params = list(model.parameters()) + list(siglip_loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
    )

    # Build scheduler: linear warmup -> cosine decay
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=config.training.scheduler.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs - config.training.scheduler.warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.training.scheduler.warmup_epochs],
    )

    # Build trainer and run
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        siglip_loss_fn=siglip_loss_fn,
        vicreg_loss_fn=vicreg_loss_fn,
    )
    best_metrics = trainer.train()

    logger.info("Best metrics: %s", best_metrics)

    # Finish W&B
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
