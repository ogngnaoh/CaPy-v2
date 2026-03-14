# CaPy v2 — Development Log

> Auto-maintained paper trail of progress, decisions, and changes.
> Newest entries first. See `capy_v2_prd.md` and `capy_v2_fsd.md` for authoritative specs.

---

### 2026-03-14 03:30 — [FIX] R@10 stuck at ~3%: 5 structural issues identified and fixed
- **Branch:** `main` | **Commit:** `9f8219a`
- **Root cause:** R@10=2.7% was NOT a hyperparameter issue — 5 cascading structural problems:
  1. Degenerate evaluation: dose duplication inflated ranks (R@1 structurally impossible, min rank=7)
  2. Degenerate data: compound-level morph aggregation → only ~787 unique (mol,morph) pairs from ~5,250 rows
  3. Massive overcapacity: 8.9M params / 787 unique compounds = 11,309 params/compound (113x overfitting threshold)
  4. No mol augmentation: ECFP deterministic → mol encoder memorized mapping instantly
  5. Phase 1 gate unrealistic: 15% R@10 at ~1,400 compounds inconsistent with literature (CLOOME: 30K→8%)
- **Fixes applied:**
  - Added compound-level retrieval (`compute_compound_retrieval_metrics`) — averages embeddings per compound, re-normalizes, eliminates rank inflation. Early stopping now uses compound-level R@10
  - Treatment-level morph matching — round-robin pairing of per-dose morph profiles instead of compound-level mean. Effective unique pairs: ~787 → ~3,000-5,000
  - Model capacity 7x reduction — encoders from [1024,1024,1024]→[512,256], output_dim 512→256, dropout 0.1→0.3, weight_decay 1e-4→1e-3. Total params: ~8.9M → ~1.2M
  - ECFP bit dropout — 10% of active bits randomly zeroed during training (prevents mol memorization)
  - Phase 1 gate revised: compound-level R@10 > 10% AND alignment < 1.5 (was R@10 > 15%)
- 153 tests passing, 4 new compound-level retrieval tests, lint clean on changed files
- Files: `src/evaluation/retrieval.py`, `src/training/trainer.py`, `src/data/preprocess.py`, `src/data/dataset.py`, `src/models/encoders.py`, `src/models/projections.py`, `configs/model/*.yaml`, `configs/training/default.yaml`, `tests/test_retrieval.py`, `scripts/verify_signal.py`, `CLAUDE.md`, `capy_v2_prd.md`, `capy_v2_fsd.md`

### 2026-03-12 21:36 — [FIX] FR-7 FSD compliance gaps + Colab training notebook
- **Branch:** `main` | **Commit:** `a01e36d`
- `save_checkpoint` now includes full Hydra config via `OmegaConf.to_container()` (FR-7.2)
- `check_collapse` returns `bool` and logs `collapse_warning=True` to W&B when collapse detected (FR-7.4)
- Added `_vicreg_high_epochs` counter: warns when VICReg variance loss stays >0.5 for >=20 consecutive epochs (Edge 5.2)
- Created `notebooks/colab_training.ipynb`: self-contained Colab notebook for Phase 1 gate (clone, install, download, preprocess, train bi-modal + tri-modal, multi-seed runs)
- 3 new tests (checkpoint config key, collapse return True/False), 143 total passing, lint clean
- Files: `src/training/trainer.py`, `tests/test_training.py`, `notebooks/colab_training.ipynb`

### 2026-03-12 21:18 — [MILESTONE] FR-7.0 training loop + FR-8.1/8.2 evaluation complete
- **Branch:** `main` | **Commit:** `828289b`
- `Trainer` (FR-7.1–7.4): train/val loop, gradient clipping, NaN detection, mixed precision (bfloat16), W&B logging, early stopping on val mean_R@10, checkpoint save/load, collapse detection via uniformity monitoring
- `compute_retrieval_metrics` / `compute_all_retrieval_metrics` (FR-8.1): R@1/5/10, MRR for all 6 cross-modal directions + mean_R@10
- `compute_alignment` / `compute_uniformity` (FR-8.2): positive pair alignment, embedding uniformity with collapse threshold at -0.5
- `scripts/train.py`: Hydra entry point with `SequentialLR(LinearLR + CosineAnnealingLR)` warmup schedule, AdamW with SigLIP bias in param group
- 26 new tests (early stopping, collapse warnings, checkpointing, train_epoch loss/grad_norm, validate retrieval/uniformity, full train loop), 140 total passing, lint clean
- Files: `src/training/trainer.py`, `src/evaluation/retrieval.py`, `src/evaluation/diagnostics.py`, `scripts/train.py`, `tests/test_training.py`, `tests/test_retrieval.py`

### 2026-03-12 20:29 — [MILESTONE] FR-6.0 loss functions complete
- **Branch:** `main` | **Commit:** `2e8e25a`
- `SigLIPLoss` (FR-6.1): pairwise contrastive with learnable bias, `F.logsigmoid` for numerical stability, symmetric loss
- `VICRegLoss` (FR-6.2): variance hinge (target std ≥ 1) + covariance penalty (off-diagonal sum / d), eps=1e-4 for NaN safety
- `compute_total_loss` (FR-6.3): assembles per-pair SigLIP + per-modality VICReg with lambda scaling, canonical modality ordering, returns differentiable total + float dict for logging
- 18 new tests (gradient flow, symmetry, collapse detection, tri/bi-modal configs, component sum), all 114 tests passing, lint clean
- Files: `src/models/losses.py`, `tests/test_losses.py`

### 2026-03-12 02:38 — [MILESTONE] FR-5.0 model architecture complete
- **Branch:** `main` | **Commit:** `bacd392`
- `_MLPEncoder` shared base: configurable `hidden_dims` list, BN→ReLU→Dropout per layer, final Linear
- `MolecularEncoder` (2048→[1024×3]→512), `MorphologyEncoder` (~1500→[1024×2]→512), `ExpressionEncoder` (978→[1024×2]→512)
- `ProjectionHead` (512→512→256) with L2 normalization via `F.normalize`
- `CaPyModel`: config-driven assembly using `nn.ModuleDict`, instantiates only active modalities (T1 tri-modal, B4/B5/B6 bi-modal)
- 13 tests passing (encoder shapes, gradient flow, L2 norms, param count ~11.6M, bi/tri-modal configs), lint clean
- Files: `src/models/encoders.py`, `src/models/projections.py`, `src/models/capy.py`, `tests/test_models.py`

### 2026-03-12 02:15 — [MILESTONE] FR-4.0 CaPyDataset & build_dataloaders implemented
- **Branch:** `main` | **Commit:** `b72cc72`
- `CaPyDataset` (FR-4.1): reads processed parquets, featurizes SMILES→ECFP at init, filters invalid SMILES, fills residual NaN with 0.0, returns aligned `{mol, morph, expr, metadata}` dicts
- SCARF augmentation: replaces features with empirical marginal draws at configurable corruption rate, applied to morph/expr only (not mol)
- `collate_fn`: custom collate stacks tensors, collects metadata as list of dicts
- `build_dataloaders` (FR-4.2): wires train/val/test splits with proper shuffle/drop_last, SCARF on train only, pin_memory auto-detected
- 15 tests passing (7 CaPyDataset, 2 collate, 6 build_dataloaders), lint clean
- Files: `src/data/dataset.py`, `tests/test_data.py`, `tests/conftest.py`

### 2026-03-12 01:38 — [MILESTONE] FR-3.0 molecular featurization complete
- **Branch:** `main` | **Commit:** `bd13b44`
- `smiles_to_ecfp`: SMILES → 2048-bit ECFP Morgan fingerprint (float32 tensor), CXSMILES stripping fallback, uses `rdFingerprintGenerator.GetMorganGenerator` (non-deprecated API)
- `featurize_smiles_batch`: batch featurization with dedup and FSD-spec summary logging ("Featurized N/M compounds (K failed)")
- `_strip_cxsmiles` duplicated locally from `preprocess.py` (4 lines, avoids cross-module private import)
- 15 tests passing (11 for `smiles_to_ecfp`, 4 for `featurize_smiles_batch`), lint clean
- Files: `src/data/featurize.py`, `tests/test_featurize.py`

### 2026-03-12 01:18 — [MILESTONE] FR-2.0 data preprocessing pipeline complete
- **Branch:** `main` | **Commit:** `4f348e1`
- Full preprocessing pipeline implemented in `src/data/preprocess.py` (FR-2.1 through FR-2.8)
- Data loaders: morphology feature intersection across batches, GCTX transpose + col_meta merge, metadata loading
- FR-2.1/2.2: Pass-through stubs (data arrives pre-aggregated as consensus MODZ)
- FR-2.3: Treatment matching with compound-level inner join, multi-stage SMILES resolution (metadata → x_smiles → optional PubChem), MOA union sourcing
- FR-2.4: Control removal (DMSO, empty, ctl_vehicle/ctl_untrt, case-insensitive)
- FR-2.5: Two-phase feature QC — phase A (inf/NaN, pre-split), phase B (zero-variance on train, post-split)
- FR-2.6: Scaffold splitting with Bemis-Murcko scaffolds, compound grouping, greedy assignment, seed reproducibility
- FR-2.7: Normalization (RobustScaler morph, conditional StandardScaler expr, clip [-5,5], fit on train only)
- FR-2.8: Parquet output (train/val/test) + feature_columns.json + pipeline orchestrator
- 53 tests passing (37 new), lint clean. Scripts and config updated.

### 2026-03-12 — [MILESTONE] FR-1.0 complete and committed
- **Branch:** `main` | **Commit:** `3023ef2`
- FR-1.1: Morphology download via GitHub LFS HTTPS (2 consensus MODZ files, ~145 MB)
- FR-1.2: Expression download from Figshare (Level 5 primary, Level 4 optional)
- FR-1.3: Metadata download from Drug Repurposing Hub
- FR-1.4: Data audit producing markdown report (morphology, expression, metadata, cross-modal overlap)
- 16 tests passing, 0 failures. Lint clean.
- Spec updates committed separately (`7c82dbc`): resolved decisions, validation annotations in FSD
- Tooling committed separately (`a7d87fe`): DEVLOG, /devlog skill, PostToolUse ruff hook

### 2026-03-11 — [MILESTONE] Project scaffold and data validation complete
- **Branch:** `main` | **Commit:** `464d001`
- Initialized CaPy v2 repo with full project structure, CLAUDE.md, configs, and test scaffold
- Resolved all 5 open questions (OPEN-1 through OPEN-4 + SCARF rate): dose-as-augmentation, Level 5 L1000, SCARF 40%, InfoAlign deferred to P2, pivot strategy defined
- Validated all 3 data sources against FSD expectations:
  - Morphology: Batch 1 (10,752 rows x 1,781 features) + Batch 2 (10,368 rows x 2,198 features) = 21,120 rows, 1,848 compounds. Feature count mismatch requires intersection in FR-2.5
  - Expression: Level 5 col_meta = 9,482 rows (incl 500 DMSO), 1,402 compounds, 7 doses. GCTX readable via cmapPy/h5py
  - Metadata: Repurposing Hub = 13,553 rows, 100% SMILES. NO MOA column — MOA sourced from pert_info.txt (88% coverage)
- Cross-modal overlap: 1,402 compounds (morph ∩ expr), 1,125 tri-modal (with SMILES). ID matching requires BRD truncation to 13 chars
- Updated FSD (FR-1.1 through FR-1.4, FR-2.2) and CLAUDE.md with validation annotations and resolved decisions
- Rewrote download pipeline: S3 sync → GitHub LFS HTTPS (145 MB vs 2.65 GB). All 16 tests pass

### 2026-03-11 — [DECISION] Data source strategy finalized
- **Branch:** `main` | **Commit:** `31d8fb2`
- Morphology: consensus MODZ profiles from GitHub LFS (not S3 per-plate profiles)
- Expression: Level 5 MODZ (not Level 4 replicate-level) — better SNR, 1:1:1 pairing
- Metadata: Drug Repurposing Hub for SMILES/BRD, pert_info.txt for MOA
- Both morph and expr arrive pre-aggregated — FR-2.1/2.2 become pass-through

---
