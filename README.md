# CaPy v2 -- Contrastive Alignment of Phenotypic Yields

[![Tests](https://github.com/ogngnaoh/CaPy-v2/actions/workflows/test.yml/badge.svg)](https://github.com/ogngnaoh/CaPy-v2/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

CaPy v2 is a **tri-modal contrastive learning framework for drug discovery** that
learns a shared 256-dimensional embedding space across three data modalities --
molecular structure (ECFP fingerprints), cell morphology (CellProfiler features from
Cell Painting), and gene expression (L1000 landmark genes). Using pairwise SigLIP
contrastive loss with VICReg regularization, it rigorously tests whether aligning all
three modalities simultaneously outperforms any bi-modal pair, with a full ablation
study across 8 configurations and 5 random seeds (24 runs total).

---

## Key Results

**Core finding:** Adding molecular information to phenotype alignment significantly
improves morph-expr retrieval (+14 percentage points), but mol-containing retrieval
is bottlenecked by the ECFP representation (~11.5%), invariant across configurations.

| Config | Description | Compound Mean R@10 | vs Random |
|--------|------------|--------------------:|----------:|
| **T1** | **Tri-modal (mol + morph + expr)** | **36.8% +/- 0.8%** | **6.1x** |
| B6 | Bi-modal (morph + expr) | 73.4% +/- 0.8% | 12.2x |
| B4 | Bi-modal (mol + morph) | 12.5% +/- 0.5% | 2.1x |
| B5 | Bi-modal (mol + expr) | 11.8% +/- 0.9% | 2.0x |
| B0 | Random embeddings | 6.0% | 1.0x |

> All trained-config comparisons are statistically significant (Welch's t-test,
> p < 1e-10 for all T1 vs B4/B5/B6 pairs).

**Per-direction breakdown (T1 vs best bi-modal):**

| Direction | T1 (5-seed mean) | Best Bi-modal | Delta |
|-----------|------------------:|--------------:|------:|
| morph -> expr | 88.4% | 73.8% (B6) | +14.6pp |
| expr -> morph | 86.9% | 73.0% (B6) | +13.9pp |
| mol -> morph | ~11.5% | ~12.5% (B4) | -- |
| mol -> expr | ~11.5% | ~11.8% (B5) | -- |

![Ablation Results](results/ablation_barplot.png)

---

## Architecture

### Encoders

| Modality | Input Dim | Architecture | Output |
|----------|----------:|-------------|--------|
| Molecular | 2,048 (ECFP) | MLP [2048 -> 512 -> 256 -> 256] + BN + ReLU + Dropout(0.3) | 256-dim |
| Morphology | ~1,500 (CellProfiler) | MLP [N -> 512 -> 256 -> 256] + BN + ReLU + Dropout(0.3) | 256-dim |
| Expression | 978 (L1000 genes) | MLP [978 -> 512 -> 256 -> 256] + BN + ReLU + Dropout(0.3) | 256-dim |
| Projection | 256 | MLP [256 -> 256 -> 256] + L2-normalize | 256-dim |

### Loss Function

```
L = w1 * SigLIP(mol, morph) + w2 * SigLIP(mol, expr) + w3 * SigLIP(morph, expr)
  + lambda * (VICReg(mol) + VICReg(morph) + VICReg(expr))
```

Each SigLIP instance has independent learnable temperature and bias (per-pair
parameterization). Weights: w1 = w2 = 2.0 (mol-containing), w3 = 1.0 (morph-expr).
VICReg lambda = 0.1. SigLIP operates on L2-normalized post-projection embeddings;
VICReg operates on pre-projection encoder outputs.

![Cross-modal Similarity](results/similarity_heatmap.png)

### Embedding Space (UMAP)

| Molecular | Morphology | Expression |
|:---------:|:----------:|:----------:|
| ![mol](results/umap_mol.png) | ![morph](results/umap_morph.png) | ![expr](results/umap_expr.png) |

---

## Quick Start

### Installation

```bash
git clone https://github.com/ogngnaoh/CaPy-v2.git
cd CaPy-v2
pip install -e ".[dev]"
```

### Full Pipeline

```bash
make setup       # install + download data (~300 MB)
make preprocess  # QC, normalize, scaffold split
make train       # train default tri-modal config (T1)
make evaluate    # evaluate best checkpoint
```

### Individual Steps

```bash
# Download specific data sources
python3 scripts/download.py --source morphology
python3 scripts/download.py --source expression
python3 scripts/download.py --source metadata

# Preprocess
python3 scripts/preprocess.py

# Train with overrides
python3 scripts/train.py model=tri_modal training.batch_size=256 seed=42

# Evaluate
python3 scripts/evaluate.py --checkpoint checkpoints/best.pt --full

# Run full ablation study (24 runs)
python3 scripts/run_ablations.py --matrix core
python3 scripts/summarize_ablations.py
```

### Code Quality

```bash
make test        # pytest with coverage
make lint        # ruff check + black check
make format      # auto-format (ruff + black)
make coverage    # coverage report with HTML output
```

---

## Configuration

CaPy v2 uses [Hydra](https://hydra.cc/) for configuration management. All
hyperparameters live in YAML files under `configs/`.

### Config Groups

| Group | Options | Description |
|-------|---------|-------------|
| `model` | `tri_modal`, `bi_mol_morph`, `bi_mol_expr`, `bi_morph_expr` | Modality selection |
| `training` | `default` | Learning rate, batch size, epochs, etc. |
| `data` | `lincs` | Dataset paths, URLs, QC thresholds |
| `ablation` | `core` | 8-config x 5-seed ablation matrix |

### Override Examples

```bash
# Bi-modal training
python3 scripts/train.py model=bi_mol_morph

# Change batch size and seed
python3 scripts/train.py training.batch_size=512 seed=123

# Adjust loss weights
python3 scripts/train.py model.loss_weights.mol_morph=1.0 model.loss_weights.mol_expr=1.0

# Change learning rate
python3 scripts/train.py training.lr=1e-4
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `training.lr` | 5e-4 | Learning rate (AdamW) |
| `training.weight_decay` | 1e-3 | AdamW weight decay |
| `training.batch_size` | 256 | Training batch size |
| `training.epochs` | 200 | Maximum epochs |
| `training.patience` | 30 | Early stopping patience (val R@10) |
| `training.warmup_epochs` | 10 | Cosine annealing warmup |
| `training.grad_clip` | 1.0 | Gradient clipping max norm |
| `model.vicreg_weight` | 0.1 | VICReg regularization weight |
| `model.scarf_corruption` | 0.4 | SCARF feature corruption rate |

---

## Data

CaPy v2 uses the [LINCS](https://lincsproject.org/) cpg0004 dataset, restricted to
the **A549 cell line** (human lung adenocarcinoma).

| Source | Description | Size | Compounds |
|--------|-------------|-----:|----------:|
| Morphology | Consensus MODZ profiles from [lincs-cell-painting](https://github.com/broadinstitute/lincs-cell-painting) | ~145 MB | 1,848 |
| Expression | L1000 Level 5 (treatment-level MODZ) from [Figshare](https://figshare.com/) | ~50 MB | 1,402 |
| Metadata | [Drug Repurposing Hub](https://clue.io/repurposing) | ~5 MB | 13,553 |

**Cross-modal overlap:** 1,402 compounds (morph intersection expr), ~1,125 tri-modal
(with valid SMILES). After quality filtering, approximately 3,000 tri-modal treatments
are used for training.

**Split strategy:** Scaffold-based (Bemis-Murcko) to prevent molecular structure
leakage between train/val/test sets.

---

## Ablation Design

The core ablation matrix tests 8 configurations across 5 random seeds:

| Config | Type | Modalities | Seeds |
|--------|------|------------|------:|
| B0 | Baseline | Random embeddings | 1 |
| B1 | Baseline | Raw mol features | 1 |
| B2 | Baseline | Raw morph features | 1 |
| B3 | Baseline | Raw expr features | 1 |
| B4 | Trained | mol + morph | 5 |
| B5 | Trained | mol + expr | 5 |
| B6 | Trained | morph + expr | 5 |
| T1 | Trained | mol + morph + expr | 5 |

**Total:** 24 runs. All comparisons use Welch's t-test for statistical significance.

Generated outputs in `results/`:
- `ablation_summary.csv` -- per-config aggregated metrics
- `ablation_comparison.tex` -- LaTeX table for publication
- `ablation_barplot.png` -- bar chart with error bars
- `retrieval_table.csv` -- per-direction R@K and MRR

---

## Project Structure

```
CaPy-v2/
├── configs/
│   ├── default.yaml              # Top-level Hydra defaults
│   ├── data/lincs.yaml           # Dataset URLs, paths, QC thresholds
│   ├── model/                    # tri_modal, bi_mol_morph, bi_mol_expr, bi_morph_expr
│   ├── training/default.yaml     # Hyperparameters
│   └── ablation/core.yaml        # 8-config x 5-seed matrix
├── src/
│   ├── data/                     # Download, preprocess, featurize, dataset
│   ├── models/                   # Encoders, projections, CaPy model, losses
│   ├── training/                 # Trainer (train/val loop)
│   ├── evaluation/               # Retrieval, diagnostics, clustering, report
│   └── utils/                    # Config, logging, seeding
├── scripts/                      # CLI entry points (train, evaluate, ablations, etc.)
├── tests/                        # pytest test suite
├── notebooks/
│   └── demo.ipynb                # CPU demo notebook (synthetic data)
├── data/                         # Raw + processed data (gitignored)
├── results/                      # Metrics, plots, tables
├── checkpoints/                  # Model checkpoints (gitignored)
├── Makefile                      # Make targets
└── pyproject.toml                # Dependencies and tool config
```

---

## Reproducibility

CaPy v2 is designed for full reproducibility:

- **Deterministic seeding:** All random sources (Python, NumPy, PyTorch, CUDA) are
  seeded via `src/utils/seeding.py` with `torch.use_deterministic_algorithms` where
  supported.
- **Git hash tracking:** Every training run records the git commit hash in its output
  config, linking results to exact code state.
- **Config snapshots:** The full Hydra config is saved alongside each checkpoint for
  exact reproduction.
- **Results in version control:** `results/` directory is tracked in git so metrics
  and plots persist across environments.
- **Scaffold splits:** Train/val/test splits are deterministic given the same compound
  set and split seed, preventing data leakage.

To reproduce the full 24-run ablation study:

```bash
make setup
make preprocess
python3 scripts/run_ablations.py --matrix core
python3 scripts/summarize_ablations.py
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Framework | Python 3.10+, PyTorch |
| Config | Hydra, OmegaConf |
| Chemistry | RDKit (SMILES to ECFP fingerprints) |
| Data | pandas, NumPy, SciPy, scikit-learn, pyarrow |
| Visualization | UMAP, matplotlib, seaborn |
| Dev tools | pytest, ruff, black, pre-commit |

---

## Citation

If you use CaPy v2 in your research, please cite:

```bibtex
@software{ngo2026capy,
  author    = {Ngo, Hoang},
  title     = {{CaPy v2}: Tri-Modal Contrastive Alignment of Phenotypic Yields},
  year      = {2026},
  url       = {https://github.com/ogngnaoh/CaPy-v2},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
