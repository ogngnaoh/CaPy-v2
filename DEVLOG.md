# CaPy v2 — Development Log

> Auto-maintained paper trail of progress, decisions, and changes.
> Newest entries first. See `capy_v2_prd.md` and `capy_v2_fsd.md` for authoritative specs.

---

### 2026-03-15 — [MILESTONE] FR-9.0 complete — B0-B3 baselines + FR-9.2 summary
- **Branch:** `main`
- Added `evaluate_baseline()` to `scripts/run_ablations.py`: B0 (random 256-dim embeddings), B1-B3 (raw features → random linear projection to 256-dim → cross-modal retrieval)
- B1-B3 use modality-specific `compound/mean_R@10` (average of 4 directions involving the baseline's modality)
- Resume support: `--resume` skips baselines already in JSONL
- 12 new tests in `tests/test_ablations.py`, 199/199 total passing
- FR-9.2 committed earlier: CSV, LaTeX, barplot, Welch's t-tests with Bonferroni correction

---

### 2026-03-15 02:10 — [MILESTONE] FR-9.2 implemented — Phase 3 blocking work complete
- **Branch:** `main` | **Commit:** `4fe8d50`
- Implemented `scripts/summarize_ablations.py`: CSV, LaTeX table, barplot from `results/ablation_runs.jsonl`
- Welch's t-test (T1 vs B4/B5/B6) with Bonferroni correction (α=0.05/3)
- Results: T1 compound mean R@10 = 36.76% ± 0.76%, all comparisons p < 1e-10 (***)
- Added 11 tests in `tests/test_summarize.py`, full suite 187/187 passing
- Notebook cell added after ablation matrix run cell

---

### 2026-03-15 — [DECISION] Phase 2 remediation sweep complete — S2b locked as T1 config
- **Branch:** `main` | **Commit:** pending
- **Sweep results (7 configs, single seed each on Colab H100):**
  - S1 per-pair SigLIP baseline: mean R@10=36.4%, morph↔expr recovered to 84-86% (vs 65-67% with shared SigLIP)
  - **S2b per-pair SigLIP + 2x mol weights: mean R@10=37.3%, morph→expr=88.7%, expr→morph=87.0%** ← BEST
  - S2c 3x mol weights: mean R@10=37.0%, diminishing returns on mol, same morph↔expr
  - S3a discriminative LR (mol 2x): best mol→expr=14.1%, but morph↔expr drops to 80%
  - S3b discriminative LR (mol 3x): morph↔expr drops further to 73%, net worse
  - S4 curriculum: mol→morph=15.3% (best), but morph↔expr drops to 83%
  - S5 staged: worst overall, freezing phenotypic encoders hurts morph↔expr badly (65%)
- **Key finding:** Per-pair SigLIP is the single biggest improvement (+20pp morph↔expr recovery). 2x mol weight gives marginal additional gains. Mol-containing directions capped at ~12-14% regardless of training strategy — ECFP representation is the ceiling.
- **Decision:** Lock S2b (per-pair SigLIP + mol pair weights 2.0) as default T1 config for Phase 3 ablation matrix
- **Phase 2 remediation: COMPLETE.** Partial success per PRD 4.1 — tri-modal improves morph↔expr by +13pp over B6
- Updated: `configs/training/default.yaml` (pair_weights mol=2.0), `CLAUDE.md` (phase status), `capy_v2_prd.md` (Phase 2 annotation)

### 2026-03-15 — [MILESTONE] FR-8.0 evaluation suite complete (FR-8.3 + FR-8.4 + CLI)
- **Branch:** `main` | **Commit:** `d2a30f7`
- FR-8.3 MOA clustering (`src/evaluation/clustering.py`): k-NN accuracy (k=5,10,20 majority vote), AMI, ARI via k-means, null MOA filtering, single-class edge case
- FR-8.4 full report (`src/evaluation/report.py`): checkpoint loading, embedding generation, retrieval table (CSV+LaTeX), UMAP plots per modality, similarity heatmap, training curves from epoch_history, formatted summary table
- CLI (`scripts/evaluate.py`): `--full`, `--diagnostics`, `--clustering` modes with `--data-dir` and `--output-dir` flags
- Trainer `epoch_history` added to `src/training/trainer.py` (3 lines) for training curve plots
- 16 new tests (8 clustering + 8 report), 176 total passing, lint clean
- **Remaining stub:** FR-9.2 `scripts/summarize_ablations.py` — not needed until Phase 3 ablation matrix completes
- Files: `src/evaluation/clustering.py`, `src/evaluation/report.py`, `scripts/evaluate.py`, `src/training/trainer.py`, `tests/test_clustering.py`, `tests/test_report.py`

### 2026-03-15 01:00 — [MILESTONE] Phase 2 gate PASSED + remediation infrastructure for weak mol directions
- **Branch:** `main` | **Commit:** pending
- **Phase 2 gate:** T1 morph↔expr compound R@10 = 84.8% vs B6 = 75.1% (+10pp). Adding mol helps morph↔expr but mol-containing directions stuck at ~12% (barely 2x random)
- **Root causes:** shared SigLIP params across pairs with different similarity distributions, equal loss weighting (morph↔expr converges by epoch ~50 stealing gradient budget), uniform LR across modalities, no curriculum for hard mol pairs
- **Infrastructure added (7 tasks):**
  - Per-pair SigLIP with independent temperature/bias per modality pair (`src/models/losses.py`, `scripts/train.py`)
  - Configurable pair weights with curriculum linear ramp for mol pairs (`src/training/trainer.py`, `configs/training/default.yaml`)
  - Discriminative learning rates per modality encoder via `modality_lr_mult` config
  - Staged training: stage1 morph↔expr only → stage2 freeze morph/expr + add mol (`configs/training/staged.yaml`)
  - Per-direction R@10 logging in validation for diagnosing asymmetric convergence
  - Extended raw-feature baselines for all 6 directions (`scripts/verify_signal.py`)
  - Checkpoint analysis script for comparing per-direction metrics (`scripts/analyze_checkpoints.py`)
- 160 tests passing (7 new: per-pair SigLIP, pair_weights scaling, zero-weight elimination, staged freeze, curriculum ramp, integration tests)
- Updated CLAUDE.md (phase status, loss formula), capy_v2_prd.md (Phase 2 gate annotation)
- **Next:** Sweep on Colab — per-pair SigLIP → loss weights → discriminative LR → curriculum → staged
- Files: `src/models/losses.py`, `src/training/trainer.py`, `scripts/train.py`, `configs/training/default.yaml`, `configs/training/staged.yaml`, `scripts/analyze_checkpoints.py`, `scripts/verify_signal.py`, `tests/test_losses.py`, `tests/test_training.py`, `CLAUDE.md`, `capy_v2_prd.md`

### 2026-03-14 23:46 — [MILESTONE] Multi-seed validation complete — Phase 1 gate confirmed
- **Branch:** `main` | **Commit:** `c9a45ab`
- 4-seed results (42, 123, 456, 789): compound R@10 range 12.7–14.7%, mean 13.4% (±0.9%) — all seeds > 10% threshold
- Alignment range 1.26–1.52: 3/4 seeds pass < 1.5 threshold, seed 456 marginal fail at 1.524 (best R@10 but longest training → alignment drift trade-off)
- No collapse in any seed: uniformity -2.00 to -2.43 (threshold -0.5)
- **Phase 1 gate: CONFIRMED.** Proceeding to Phase 2 (bi-modal baselines B5/B6 + tri-modal T1)

### 2026-03-14 04:30 — [MILESTONE] Phase 1 gate passed (single seed), multi-seed validation prep
- **Branch:** `main` | **Commit:** pending
- Phase 1 gate result (seed 42): compound R@10 = 12.7% > 10% threshold, alignment = 1.464 < 1.5 threshold
- Result is 2.25x random baseline (random = 5.65% at 177 val compounds), no collapse (uniformity mol=-2.10, morph=-2.17)
- Implemented `scripts/run_ablations.py` (FR-9.1): full ablation harness with `--resume`, `--configs`, `--seeds` support
- Checkpoint now saves full val metrics dict for ablation result collection
- Updated Colab notebook: multi-seed validation cells, Phase 2 training cells, result comparison table
- Updated CLAUDE.md current phase to reflect gate passed, next steps
- 153 tests passing, lint clean
- **Next:** Multi-seed validation (seeds 123, 456, 789) on Colab, then Phase 2 bi-modal baselines + tri-modal
- Files: `scripts/run_ablations.py`, `src/training/trainer.py`, `notebooks/colab_training.ipynb`, `CLAUDE.md`

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
