# CLAUDE.md — CaPy v2

This file provides guidance to Claude Code when working in this repository.

**CaPy v2** (Contrastive Alignment of Phenotypic Yields) is a tri-modal contrastive
learning framework for drug discovery. It learns a shared 256-dim embedding space
across molecular structure, cell morphology, and gene expression — then rigorously
tests whether tri-modal alignment outperforms any bi-modal pair.

Full specifications: `capy_v2_prd.md` (requirements) and `capy_v2_fsd.md` (functional spec).
These two documents are the authoritative source of truth for all requirements.
They can be updated when the user requests changes — they are not read-only.

---

## STOP — Do NOT Build These (NG1–NG9)

Before implementing anything, check this list. These are **permanent non-goals**:

- **NG1:** Do NOT build a web UI, Streamlit app, or visualization server.
- **NG2:** Do NOT process raw microscopy images. All morphology input is pre-extracted CellProfiler features (~1,500 dims).
- **NG3:** Do NOT implement any GNN encoder (GIN, GAT, SchNet). Molecular encoder is ECFP + MLP only.
- **NG4:** Do NOT implement InfoAlign's context graph, random walk sampling, or decoder-based information bottleneck.
- **NG5:** Do NOT combine data from different cell lines. CaPy uses only A549 LINCS data.
- **NG6:** Do NOT implement molecule generation, SMILES generation, or any generative model.
- **NG7:** Do NOT implement model serving, inference API, or ONNX export.
- **NG8:** Do NOT add authentication, user accounts, or multi-tenancy.
- **NG9:** Do NOT use hyperparameter optimization frameworks (Optuna, Ray Tune). Use Hydra multirun with manual grids.

---

## Locked-In Decisions

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Molecular encoder | ECFP (2048-bit) + MLP | GIN (collapsed at N<2000, uniformity=−0.22) | Benchmarks show ECFP matches/beats neural encoders |
| Contrastive loss | SigLIP pairwise | InfoNCE (batch-size limited with log K bound) | SigLIP is batch-size agnostic |
| Regularization | VICReg (variance + covariance) | None (v1 had encoder collapse) | Variance term directly prevents collapse |
| Dataset | LINCS cpg0004 (A549) | CDRP-bio (U2OS, 15× less data) | Better cross-modality alignment per Rosetta paper |
| Morphology source | Consensus MODZ profiles from lincs-cell-painting repo (Git LFS, ~145 MB) | Per-plate S3 profiles (2.65 GB, 1620 files) | Pre-aggregated, treatment-level, full features for our own QC |
| L1000 level | Level 5 (treatment-level MODZ) | Level 4 (replicate-level, noisier) | Better SNR, matches 1:1:1 treatment pairing |
| Config management | Hydra | Raw omegaconf / argparse | Structured overrides, multirun, config groups |
| Experiment tracking | Weights & Biases | TensorBoard | Better sweep management, team features |
| Normalization | BatchNorm in encoders | LayerNorm | Standard for MLP encoders in contrastive learning |
| Split strategy | Scaffold-based (Bemis-Murcko) | Random split | Prevents molecular structure leakage |
| GPU environment | Google Colab H100 (80 GB HBM3) | Local T4/V100 | Ample VRAM; code stays portable to smaller GPUs |

---

## Resolved Decisions (formerly Open)

All open decisions resolved 2026-03-11. FR-2.6 and FR-1.2 are now unblocked.

| ID | Decision | Resolution |
|----|----------|------------|
| OPEN-1 | Dose as augmentation with quality filtering | SupCon-style positive pairing across doses, but only for treatments passing replicate-correlation filter (90th pctl DMSO null). Effective N ~3,000–5,000. |
| OPEN-2 | L1000 Level 5 (treatment-level MODZ) | Better SNR, 1:1:1 pairing. Level 4 available as optional backup. |
| OPEN-3 | SCARF corruption 40% default | Sweep 20/40/60% on morph↔expr in Week 2 (~90 min). Cell Painting features are highly correlated so higher rates tolerable. |
| Q2 | InfoAlign comparison deferred to P2 | Asymmetric arch (2/6 dirs), different cell line. Frame as out-of-distribution if included. |
| Q5 | Pivot = per-MOA complementarity analysis | Break by MOA class to show where tri-modal helps vs. doesn't. Conditional analysis is the contribution. |

---

## Commands

> **Environment note:** This machine uses `python3`, not `python`. Makefile and scripts assume `python` — run via `make` targets or use `python3` directly.

```bash
# Setup & data
make setup              # pip install -e ".[dev]" + download data
make preprocess          # QC, normalize, split → data/processed/

# Training & evaluation
make train               # train default tri-modal config
make evaluate            # evaluate best checkpoint

# Code quality
make test                # pytest with coverage
make lint                # ruff check + black check

# Scripts (direct)
python3 scripts/download.py --source morphology|expression|metadata
python3 scripts/preprocess.py
python3 scripts/train.py model=bi_mol_morph training.batch_size=256 seed=42
python3 scripts/evaluate.py --checkpoint checkpoints/best.pt --full
python3 scripts/run_ablations.py --matrix core
python3 scripts/summarize_ablations.py

# Formatting
ruff check src/ tests/ scripts/ --fix
black src/ tests/ scripts/
```

---

## Tech Stack

**Using:**
- Python 3.10+, PyTorch (training framework)
- Hydra + OmegaConf (config management)
- Weights & Biases (experiment tracking)
- RDKit (SMILES → ECFP fingerprints)
- pandas, numpy, scipy, scikit-learn (data processing)
- pyarrow (parquet I/O)
- pubchempy (SMILES lookup fallback)
- umap-learn, matplotlib, seaborn (visualization)
- pytest, ruff, black, pre-commit (dev tools)

**NOT using (removed from v1):**
- torch-geometric — no GNN encoders
- scanpy — no single-cell analysis
- gseapy — no gene set enrichment (P2 at best)
- raw omegaconf — replaced by Hydra

---

## Repository Structure

```
CaPy-v2/
├── CLAUDE.md                        # This file
├── capy_v2_prd.md                   # Product requirements (authoritative)
├── capy_v2_fsd.md                   # Functional spec (authoritative)
├── pyproject.toml                   # Dependencies + tool config
├── Makefile                         # CLI entry points
├── .pre-commit-config.yaml          # Ruff, black, data blocker
├── configs/
│   ├── default.yaml                 # Top-level defaults
│   ├── data/lincs.yaml              # Dataset URLs, paths, QC thresholds
│   ├── model/
│   │   ├── tri_modal.yaml           # T1: all 3 modalities
│   │   ├── bi_mol_morph.yaml        # B4
│   │   ├── bi_mol_expr.yaml         # B5
│   │   └── bi_morph_expr.yaml       # B6
│   ├── training/default.yaml        # Hyperparameters
│   └── ablation/core.yaml           # 8-config × 5-seed matrix
├── src/
│   ├── data/
│   │   ├── download.py              # FR-1: data download
│   │   ├── audit.py                 # FR-1.4: data audit report
│   │   ├── preprocess.py            # FR-2: QC, normalize, split
│   │   ├── featurize.py             # FR-3: SMILES → ECFP
│   │   └── dataset.py              # FR-4: CaPyDataset + DataLoader
│   ├── models/
│   │   ├── encoders.py              # FR-5.1–5.3: MLP encoders
│   │   ├── projections.py           # FR-5.4: projection heads
│   │   ├── capy.py                  # FR-5.5: model assembly
│   │   └── losses.py               # FR-6: SigLIP + VICReg
│   ├── training/
│   │   └── trainer.py               # FR-7: train/val loop
│   ├── evaluation/
│   │   ├── retrieval.py             # FR-8.1: R@K, MRR
│   │   ├── diagnostics.py           # FR-8.2: alignment/uniformity
│   │   ├── clustering.py            # FR-8.3: MOA clustering
│   │   └── report.py               # FR-8.4: full evaluation report
│   └── utils/
│       ├── config.py                # FR-10: Hydra utilities
│       ├── logging.py               # FR-11: logger setup
│       └── seeding.py               # FR-10.2: seed_everything()
├── scripts/
│   ├── download.py                  # make setup entry point
│   ├── preprocess.py                # make preprocess entry point
│   ├── train.py                     # make train entry point
│   ├── evaluate.py                  # make evaluate entry point
│   ├── run_ablations.py             # FR-9.1: ablation harness
│   └── summarize_ablations.py       # FR-9.2: ablation summary
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_retrieval.py
│   └── test_featurize.py
├── data/
│   ├── raw/{morphology,expression,metadata}/  # Downloaded (gitignored)
│   ├── processed/                              # QC'd + normalized (gitignored)
│   └── reports/                                # Data audit reports
├── results/                         # Generated outputs (gitignored)
└── checkpoints/                     # Model checkpoints (gitignored)
```

---

## Architecture Summary

### Encoders

| Modality | Input | Architecture | Output |
|----------|-------|-------------|--------|
| Molecular | ECFP 2048-bit | MLP [2048→512→256→256] + BN + ReLU + Dropout(0.3) | 256-dim |
| Morphology | ~1,500 CellProfiler features | MLP [~1500→512→256→256] + BN + ReLU + Dropout(0.3) | 256-dim |
| Expression | 978 L1000 landmark genes | MLP [978→512→256→256] + BN + ReLU + Dropout(0.3) | 256-dim |
| Projection | 256-dim encoder output | MLP [256→256→256] + L2-normalize | 256-dim (unit norm) |

### Loss

```
L_total = w₁·SigLIP₁(emb_mol,emb_morph) + w₂·SigLIP₂(emb_mol,emb_expr) + w₃·SigLIP₃(emb_morph,emb_expr)
        + λ · (VICReg(enc_mol) + VICReg(enc_morph) + VICReg(enc_expr))
```

Each SigLIP instance has its own learnable temperature and bias (per-pair).
SigLIP operates on L2-normalized embeddings (post-projection).
VICReg operates on encoder outputs (pre-projection, pre-normalization) —
applying VICReg to L2-normalized vectors causes the variance hinge to saturate.
w₁ = w₂ = 2.0 (mol pairs), w₃ = 1.0 (morph↔expr). λ = 0.1.
For bi-modal configs, only the relevant pair is included (weight = 1.0).
Curriculum mode (disabled by default) linearly ramps mol-pair weights from 0 to target over warmup epochs.

### Training Hyperparameters

AdamW (weight_decay=1e-3), LR=5e-4, cosine annealing + 10-epoch warmup,
batch_size=256, epochs=200, early_stopping patience=30 on val compound-level mean R@10,
SCARF corruption=40% (morph/expr) + ECFP 10% bit dropout (mol), gradient clip max_norm=1.0.

---

## Code Standards

- **Formatter:** black (line-length 88)
- **Linter:** ruff
- **Tests:** pytest, target ≥80% coverage on src/
- **Type hints:** required on all public functions
- **Logging:** NEVER use `print()` — use `src/utils/logging.py` logger with format `{timestamp} | {module} | {level} | {message}`
- **Seeding:** ALL random sources (Python, numpy, torch, CUDA) via `src/utils/seeding.py`
- **Config:** ALL hyperparameters via Hydra config, never hardcode
- **Embeddings:** always 256-dim, always L2-normalized in contrastive space

---

## Terminology

Use these terms consistently (from FSD Section 2):

| Term | Meaning |
|------|---------|
| Treatment | One compound at one dose in one cell line (atomic training unit) |
| Compound | Unique chemical entity identified by InChIKey |
| Modality | mol (molecular), morph (morphology), expr (expression) |
| Profile | Numeric feature vector for one treatment in one modality |
| Embedding | 256-dim L2-normalized vector from encoder + projection head |
| Retrieval direction | Ordered pair e.g. mol→morph. 6 total directions |
| Ablation config | One of B0–B6 or T1 in the 8-condition matrix |
| Run | One training with specific config + seed. 40 runs in core matrix |

---

## Agent Workflow Guidance

Before writing code for any feature:
1. **Check FR:** Find the relevant functional requirement in `capy_v2_fsd.md` (FR-1 through FR-11). Follow the spec exactly.
2. **Write test alongside:** Every module gets a corresponding test. Write tests that verify the "Verified when" clause from the FR.
3. **Check non-goals:** Re-read NG1–NG9 above before starting. If your implementation touches a non-goal, stop.

When debugging:
- Check alignment/uniformity metrics first — collapse (uniformity > −0.5) is the most common failure mode.
- Check per-modality loss components — asymmetric convergence is expected and not an error.
- Check data shapes at module boundaries — mismatched feature dims are the second most common issue.

Subagent patterns:
- Data download: parallelize across 3 sources (morphology, expression, metadata)
- Multi-seed training: parallelize across seeds for same config
- Module implementation: parallelize independent modules (e.g., encoders + losses + evaluation)

Use context7 MCP for up-to-date PyTorch, Hydra, and RDKit documentation.

---

## Current Phase

**Phase 1 — Foundation** (Weeks 1–3) — **GATE PASSED** (multi-seed confirmed)

Result (4 seeds): compound R@10 range 12.7–14.7%, mean 13.4% (±0.9%), all seeds > 10% threshold.

**Phase 2 — Tri-modal** (Weeks 4–6) — **GATE PASSED**

Gate: Tri-modal beats best bi-modal on at least one metric category.

Result: T1 morph↔expr compound R@10 = 84.8% vs B6 morph↔expr = 75.1% (+10pp). Adding mol helps morph↔expr alignment. However, mol-containing directions remain at ~12% (barely 2x random).

**Phase 2 Remediation — COMPLETE.** Sweep of 7 configs on Colab (per-pair SigLIP, loss weights, discriminative LR, curriculum, staged). Best config: **S2b (per-pair SigLIP + 2x mol pair weights)**.

S2b results (single seed): compound mean R@10 = 37.3% (6.6x random). morph→expr = 88.7% (+13.6pp vs B6), expr→morph = 87.0% (+13.0pp vs B6). Mol-containing directions ~11-14% (≈ bi-modal baselines). Locked as default T1 config.

**Current focus: Phase 3 — Ablations & Rigor.** Run 40-run ablation matrix (8 configs × 5 seeds), statistical analysis, complementarity analysis.

Next: `python scripts/run_ablations.py --matrix core` on Colab → `scripts/summarize_ablations.py` → p-values.
