# CaPy v2 — Functional Specification Document

## Document Control

| Field | Value |
|-------|-------|
| Title | CaPy v2 Functional Specification |
| Author | Hoang Ngo |
| Version | 0.5 (Review) |
| Last Modified | 2026-03-10 |
| Status | Review — open issues must be resolved before Week 1 coding begins |
| Parent PRD | `capy_v2_prd.md` v2.0 |

### Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-03-10 | Hoang | Initial draft from PRD v2.0 |
| 0.5 | 2026-03-10 | Hoang | Full functional spec for Phase 1–4, review-ready |

---

## 1. Overview & Scope

CaPy v2 is a Python command-line research framework that trains contrastive models aligning three biological data modalities — molecular structure, cell morphology, and gene expression — in a shared embedding space, then evaluates whether tri-modal alignment outperforms any bi-modal pair. The user interacts with CaPy through four CLI entry points: `download`, `preprocess`, `train`, and `evaluate`. All configuration is managed via YAML files loaded by Hydra. All experiment metrics are logged to Weights & Biases and to local JSON files.

**In scope:** Data pipeline (download → QC → normalize → split), model training with configurable modality combinations, evaluation suite (retrieval + clustering + diagnostics), ablation harness for systematic experiments, reproducibility infrastructure (Docker, seeding, config hashing).

**Out of scope:** Web UI, serving/inference API, raw image processing, context graph construction (InfoAlign-style), GNN training from scratch. See PRD Section 6 for full non-goals.

---

## 2. Glossary

| Term | Definition | Used Consistently As |
|------|-----------|---------------------|
| Treatment | One compound at one dose in one cell line. The atomic unit of training data. | "treatment" (never "sample" or "data point") |
| Compound | A unique chemical entity identified by InChIKey. Has SMILES, may appear at multiple doses. | "compound" |
| Modality | One of three data types: molecular structure (mol), cell morphology (morph), gene expression (expr). | "modality" with abbreviations mol/morph/expr |
| Profile | A numeric feature vector for one treatment in one modality. | "profile" |
| Embedding | A 256-dim L2-normalized vector produced by an encoder + projection head. | "embedding" |
| Retrieval direction | An ordered pair of modalities (e.g., mol→morph). 6 total directions exist. | "direction" (e.g., "mol→morph direction") |
| Ablation config | One of the 8 experimental conditions (B0–B6, T1) in the ablation matrix. | "config" with ID (e.g., "config B4") |
| Run | One training execution with a specific config + seed. 40 total runs in the core matrix. | "run" |
| Seed | An integer that fully determines all randomness (data splits, init, shuffling). | "seed" |

---

## 3. Scenarios & User Flows

### 3.1 Scenario: Hoang runs the full pipeline for the first time

Hoang opens the repo in Google Colab with an H100 GPU runtime. He runs:

```
make setup          # installs deps, downloads data
make preprocess     # runs QC, normalization, splitting
make train          # trains the default tri-modal config
make evaluate       # runs full evaluation suite on best checkpoint
```

**Happy path:** Each command completes with a summary printed to stdout. `make setup` downloads ~350 MB of data and reports "Downloaded 3/3 sources. Data audit: 7,842 treatments available." `make preprocess` reports "Processed: 5,523 train / 1,184 val / 1,184 test treatments. Saved to data/processed/." `make train` reports "Training complete. Best mean_R@10 = 0.312 at epoch 47. Checkpoint: checkpoints/tri_modal_seed42.pt." `make evaluate` produces a results table and saves figures to `results/`.

**Unhappy path 1 — network failure during download:** `make setup` fails on the L1000 Figshare download. The system prints "ERROR: Failed to download L1000 profiles from Figshare (HTTP 503). Retry with: python scripts/download.py --source l1000". Previously downloaded files (Cell Painting profiles) are preserved, not re-downloaded on retry.

**Unhappy path 2 — data audit reveals too few treatments:** `make preprocess` runs the data audit and finds only 2,100 treatments survive QC (below the 5,000 target). The system prints "WARNING: Only 2,100 treatments survived QC (target: ≥5,000). Check data/reports/lincs_audit.md for details. Proceeding with available data." Training continues but logs a persistent warning.

**Unhappy path 3 — GPU out of memory:** `make train` crashes with CUDA OOM at batch_size=256. The system prints the standard PyTorch OOM message. Hoang edits `configs/default.yaml` to set `training.batch_size: 128` and reruns. No data reprocessing is needed. (Note: unlikely on Colab H100 with 80 GB HBM3, but handled for portability.)

### 3.2 Scenario: Hoang runs the full ablation matrix

Hoang wants to execute all 24 runs (4 baselines × 1 seed + 4 trained × 5 seeds) in the core ablation matrix:

```
python scripts/run_ablations.py --matrix core
```

**Happy path:** The system launches runs sequentially (or via Hydra multirun if configured). Each run logs to W&B with tags `config=B4, seed=42`. After all 24 runs complete, the system generates `results/ablation_summary.csv` containing all metrics for all runs, and prints a comparison table to stdout showing mean ± std for each config across seeds.

**Unhappy path — a run crashes mid-execution:** The system logs which runs completed and which failed. On rerun with `--resume`, it skips completed runs (identified by config+seed matching existing checkpoints) and only runs the remaining ones.

### 3.3 Scenario: Hoang investigates a failing modality pair

After training, Hoang notices mol→expr R@10 is only 0.11. He runs:

```
python scripts/evaluate.py --checkpoint checkpoints/best.pt --diagnostics
```

The system produces per-direction retrieval tables, alignment/uniformity metrics per modality, a cosine similarity heatmap, and UMAP visualizations saved to `results/diagnostics/`. Hoang sees that mol uniformity = −0.8 (healthy) but mol-expr alignment = 1.95 (poor), confirming the molecular and expression embeddings are well-distributed but not aligned with each other.

---

## 4. Functional Requirements

### FR-1.0: Data Download

**FR-1.1: Cell Painting profile download**
Trigger: User runs `python scripts/download.py --source morphology` or `make setup`.
Input: None (URLs in `configs/data/lincs.yaml`).
Behavior:
- The system SHALL download pre-aggregated consensus MODZ profiles from the `broadinstitute/lincs-cell-painting` GitHub repository (Git LFS) via HTTPS.
- Two files are downloaded: Batch 1 (`2016_04_01_a549_48hr_batch1_consensus_modz.csv.gz`, ~66 MB) and Batch 2 (`2017_12_05_Batch2_consensus_modz.csv.gz`, ~78 MB). Total ~145 MB.
- These are treatment-level profiles already aggregated via MODZ with full CellProfiler features (~1,500 columns before QC).
- NOTE: The raw S3 bucket (`s3://cellpainting-gallery/cpg0004-lincs/broad/workspace/profiles/`) contains per-plate profiles (270 plates × 6 variants = 1,620 files, 2.65 GB). We do NOT download those — the consensus files are what we need.
- IF a file already exists at the target path, the system SHALL skip that file and print "Skipping [filename] — already exists."
- The system SHALL save files to `data/raw/morphology/` and log the file count after download.
- IF download fails after 3 retries (30s timeout each), the system SHALL log an error and exit with code 1.
Verified when: `data/raw/morphology/` contains 2 gzipped CSV files with combined ≥15,000 rows and ≥500 feature columns.
**Validated 2026-03-11:** Batch 1 = 10,752 rows × 1,781 features (+ 9 metadata cols). Batch 2 = 10,368 rows × 2,198 features (+ 9 metadata cols). Combined = 21,120 rows, 1,848 unique compounds. Note: feature column counts differ between batches — FR-2.5 must intersect features before QC.

**FR-1.2: L1000 expression profile download**
Trigger: User runs `python scripts/download.py --source expression` or `make setup`.
Input: None (URLs in `configs/data/lincs.yaml`).
Behavior:
- The system SHALL download Level 5 (treatment-level MODZ aggregated) profiles from Figshare. Level 4 (replicate-level) is NOT downloaded by default (OPEN-2 resolved: Level 5 chosen for better SNR).
- Three primary files: the Level 5 GCTX data file, its column metadata, and perturbation info.
- Same retry, skip, and error behavior as FR-1.1.
- The system SHALL save to `data/raw/expression/`.
Verified when: `data/raw/expression/` contains the Level 5 GCTX file with ≥5,000 rows and 978 columns.
**Validated 2026-03-11:** Level 5 col_meta has 9,482 rows (8,982 treatments + 500 DMSO controls), 1,402 unique compounds, 7 dose levels (0.04–20 µM). GCTX readable via cmapPy or h5py. Column `pert_id` uses truncated 13-char BRD format. Column `x_smiles` provides SMILES for all treatments.

**FR-1.3: Compound metadata download**
Trigger: User runs `python scripts/download.py --source metadata` or `make setup`.
Input: None.
Behavior:
- The system SHALL download the Drug Repurposing Hub samples file from `s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt`.
- The system SHALL save to `data/raw/metadata/repurposing_samples.txt`.
Verified when: File contains ≥5,000 rows with columns including `broad_id` and `smiles`.
**Validated 2026-03-11:** 13,553 rows, 6,806 unique compounds (by truncated BRD), 100% SMILES coverage, has InChIKey and pubchem_cid. NOTE: This file has NO MOA column — MOA comes from L1000 `pert_info.txt` (990/1,125 tri-modal compounds have MOA, 88%) and morphology `Metadata_moa` column.

**FR-1.4: Data audit**
Trigger: Runs automatically after all downloads complete, OR manually via `python scripts/audit.py`.
Input: All downloaded raw files.
Behavior:
- The system SHALL produce a markdown report at `data/reports/lincs_audit.md` containing:
  - Morphology: row count, column count, NaN rate per feature (histogram), inf count, metadata columns identified
  - Expression: row count, column count, mean/std statistics, probe ID count
  - Metadata: compound count, SMILES coverage, MOA annotation coverage
  - Cross-modal overlap: number of treatments present in BOTH morphology and expression
- The system SHALL print a one-line summary: "Data audit complete: [N] treatments with paired morph+expr profiles. Report: data/reports/lincs_audit.md"
- IF paired treatment count < 5,000, the system SHALL print "WARNING: Only [N] paired treatments found (target: ≥5,000)."
Verified when: `data/reports/lincs_audit.md` exists and contains all five sections listed above.
**Validated 2026-03-11:** Cross-modal overlap = 1,402 compounds (morph ∩ expr), 1,125 compounds (all three modalities with SMILES). Target ≥5,000 paired treatments is met. ID matching requires truncating morph `Metadata_broad_sample` (22-char) to 13 chars to match expr `pert_id` (13-char).

---

### FR-2.0: Data Preprocessing

**FR-2.1: Replicate correlation filtering**
Trigger: Invoked during `python scripts/preprocess.py` or `make preprocess`.
Input: Raw morphology and expression profiles.
Behavior:
- For each treatment with ≥2 replicates, the system SHALL compute the mean pairwise Pearson correlation across feature vectors.
- The system SHALL compute a null distribution by sampling 10,000 random pairs of DIFFERENT treatments and computing their Pearson correlation.
- The system SHALL compute the 90th percentile of the null distribution as the quality threshold.
- The system SHALL remove treatments whose median replicate correlation falls below the threshold.
- The system SHALL log: "Replicate filter: kept [N]/[M] treatments (threshold=[T])."
- IF no control replicates are found for null computation, the system SHALL use all-treatment random pairs and log "WARNING: No DMSO controls found. Using all-treatment null distribution."
Verified when: Output contains fewer treatments than input, threshold is between 0.1 and 0.6, and log message is printed.

**FR-2.2: Replicate aggregation**
Trigger: After replicate filtering.
Input: Filtered replicate-level profiles.
Behavior:
- The system SHALL aggregate replicates to treatment-level using MODZ (moderated z-score) aggregation: weight each replicate by its mean Spearman correlation with other replicates of the same treatment.
- IF a treatment has only 1 replicate, the system SHALL use that replicate directly (weight = 1.0).
- The system SHALL log: "Aggregation: [N] replicates → [M] treatments (method=MODZ)."
- **NOTE (2026-03-11):** The consensus MODZ morphology profiles are already aggregated at the treatment level. FR-2.1 and FR-2.2 apply only to L1000 if using Level 4 data. With Level 5 (resolved), both modalities arrive pre-aggregated. These steps become pass-through but are retained for pipeline completeness and future Level 4 experiments.
Verified when: Output has exactly one row per unique treatment identifier. No duplicate treatment IDs exist.

**FR-2.3: Treatment matching across modalities**
Trigger: After aggregation of both morphology and expression.
Input: Treatment-level morphology profiles, treatment-level expression profiles, compound metadata.
Behavior:
- The system SHALL normalize all BRD identifiers to truncated 13-char format (e.g., `BRD-K76022557-003-28-9` → `BRD-K76022557`) before matching. Morphology uses full 22-char BRD IDs (`Metadata_broad_sample`); expression uses 13-char (`pert_id`).
- The system SHALL match treatments across morphology and expression tables using the normalized compound ID + dose.
- The system SHALL match compounds to SMILES using a three-stage fallback: (1) truncated BRD match against Repurposing Hub `broad_id`, (2) match against L1000 `x_smiles` column from col_meta, (3) compound name → PubChem CID → SMILES lookup via `pubchempy`.
- The system SHALL source MOA annotations from L1000 `pert_info.txt` (`moa` column, 88% coverage) and morphology `Metadata_moa` column (union). The Drug Repurposing Hub does NOT contain MOA.
- The system SHALL remove treatments where ANY of the three modalities (morph, expr, SMILES) is missing.
- The system SHALL log: "Matched: [N] treatments with all 3 modalities. SMILES coverage: [X]/[Y] via BRD, [Z] via x_smiles, [W] via PubChem. MOA coverage: [A]/[N]."
Verified when: Every row in the output has non-null values for morph features, expr features, and a valid SMILES string parseable by RDKit.
**Validated 2026-03-11:** Expected ~1,125 compounds (tri-modal with SMILES). ID truncation verified: morph 22-char → 13-char matches expr `pert_id`. SMILES available from both Repurposing Hub (100% of 6,806 compounds) and L1000 `x_smiles` (all 9,482 treatments). MOA from `pert_info.txt` covers 990/1,125 (88%).

**FR-2.4: Control removal**
Trigger: After matching.
Input: Matched treatment table.
Behavior:
- The system SHALL remove all treatments identified as DMSO or vehicle controls (matching patterns: `DMSO`, `dmso`, `EMPTY`, `empty` in compound name or perturbation type columns).
- The system SHALL log: "Removed [N] control treatments. [M] active treatments remain."
Verified when: No remaining treatments have DMSO/vehicle identifiers.

**FR-2.5: Feature QC**
Trigger: After control removal.
Input: Active treatment table with feature columns.
Behavior:
- **Feature intersection (morphology only):** Batch 1 has 1,781 CellProfiler features, Batch 2 has 2,198. The system SHALL compute the intersection of feature column names across all morphology files and retain ONLY the shared columns. The system SHALL log: "Feature intersection: [X] shared morph features from [N] files (Batch 1: [A], Batch 2: [B])."
- The system SHALL replace all `inf` and `-inf` values with `NaN`.
- The system SHALL identify morphology feature columns using CellProfiler naming prefixes: `Cells_`, `Cytoplasm_`, `Nuclei_`.
- The system SHALL identify expression feature columns using probe ID suffix: `_at`.
- The system SHALL remove features with >5% NaN values across all treatments.
- The system SHALL remove features with zero variance (std = 0) computed on training set only.
- The system SHALL log: "Feature QC: morph [X] → [Y], expr [A] → [B]. Removed [Z] features total."
- The system SHALL save the list of retained feature column names to `data/processed/feature_columns.json`.
Verified when: Output contains no `inf` values, no features with >5% NaN, and `feature_columns.json` exists.
**Validated 2026-03-11:** Batch 1 has 1,781 features, Batch 2 has 2,198 features. Intersection required before downstream QC. Expression is fixed at 978 landmark genes (no intersection needed).

**FR-2.6: Scaffold splitting**
Trigger: After feature QC.
Input: Active treatments with SMILES.
Behavior:
- The system SHALL compute Bemis-Murcko scaffolds for all compounds using RDKit.
- The system SHALL group all doses of the same compound into the same split (no compound appears in more than one split).
- The system SHALL assign 70% of compounds to train, 15% to val, 15% to test, maintaining scaffold grouping.
- IF MOA annotations are available for ≥50 compounds, the system SHALL stratify by MOA within the 70/15/15 allocation.
- The system SHALL log: "Scaffold split: train=[N] ([X]%), val=[M] ([Y]%), test=[K] ([Z]%) treatments."
- The system SHALL save split assignments to each output file (a `split` column).
Verified when: No compound (by InChIKey) appears in more than one split. Split proportions are within ±3% of targets.

**FR-2.7: Normalization**
Trigger: After splitting.
Input: Split treatment data.
Behavior:
- Morphology: The system SHALL fit a RobustScaler (median/IQR) on training data only, then transform all splits. After scaling, the system SHALL clip values to [−5, 5].
- Expression: The system SHALL verify that expression values are approximately z-scored (mean within ±0.1 of 0, std within ±0.3 of 1.0 across genes). IF not, the system SHALL fit a StandardScaler on training data and transform all splits. After normalization, the system SHALL clip to [−5, 5].
- The system SHALL log per-split statistics: "Train morph: mean=[X], std=[Y]. Train expr: mean=[A], std=[B]."
Verified when: Training set morph features have median ≈ 0 and IQR ≈ 1. Expression features have mean ≈ 0 and std between 0.7 and 1.3. No values exceed ±5.

**FR-2.8: Output files**
Trigger: After normalization.
Input: Fully processed data.
Behavior:
- The system SHALL save three parquet files: `data/processed/train.parquet`, `data/processed/val.parquet`, `data/processed/test.parquet`.
- Each file SHALL contain: all morphology feature columns, all expression feature columns, compound metadata columns (SMILES, compound ID, dose, MOA if available), and a `split` column.
- The system SHALL save `data/processed/feature_columns.json` with keys `morph_features` (list of strings) and `expr_features` (list of strings).
- The system SHALL print: "Saved: train=[N], val=[M], test=[K] treatments to data/processed/."
Verified when: All three parquet files are readable by pandas, column counts match `feature_columns.json`, and total row count equals the sum reported in the log.

---

### FR-3.0: Molecular Featurization

**FR-3.1: SMILES to ECFP conversion**
Trigger: During dataset construction (lazy, at DataLoader time, or eager during preprocessing — configurable).
Input: SMILES string for one compound.
Behavior:
- The system SHALL parse the SMILES string using `RDKit.Chem.MolFromSmiles()`.
- IF parsing fails (returns None), the system SHALL attempt to sanitize the SMILES by stripping extended SMILES (CXSMILES) annotations (everything after ` |` including the pipe) and retry.
- IF parsing still fails, the system SHALL log "WARNING: Could not parse SMILES: [smiles]" and exclude the treatment from the dataset.
- For valid molecules, the system SHALL compute a 2048-bit count-based Morgan fingerprint at radius 2 using `RDKit.Chem.AllChem.GetMorganFingerprintAsBitVect()`.
- The system SHALL convert the fingerprint to a float32 tensor of shape `[2048]`.
- The system SHALL log once at startup: "Featurized [N]/[M] compounds ([K] failed)."
Verified when: For SMILES `CCO` (ethanol), the output tensor has shape `[2048]`, dtype float32, and at least 3 nonzero bits. For an invalid SMILES like `INVALID`, parsing failure is logged and the treatment is excluded.

---

### FR-4.0: Dataset & DataLoader

**FR-4.1: CaPy dataset class**
Trigger: Instantiated during training/evaluation.
Input: Path to a processed parquet file, path to `feature_columns.json`, featurization mode.
Behavior:
- The system SHALL load the parquet file and extract morph features, expr features, and SMILES.
- Each `__getitem__` call SHALL return a dict: `{"morph": Tensor[morph_dim], "expr": Tensor[expr_dim], "mol": Tensor[2048], "metadata": {...}}`.
- IF SCARF augmentation is enabled (training only), the system SHALL corrupt `morph` and `expr` tensors by replacing a random `corruption_rate` fraction of features with draws from the per-feature empirical marginal distribution (computed once at dataset init from the training set).
- The system SHALL NOT apply SCARF augmentation during validation or testing.
- IF a treatment's SMILES failed featurization, that treatment SHALL be excluded and logged at init time.
Verified when: Dataset length equals parquet row count minus failed SMILES. Each item's tensors have correct shapes and dtypes (float32).

**FR-4.2: DataLoader construction**
Trigger: During training/evaluation setup.
Input: Dataset, batch_size, num_workers from config.
Behavior:
- Training DataLoader SHALL shuffle data each epoch.
- Validation and test DataLoaders SHALL NOT shuffle (deterministic ordering).
- The system SHALL drop the last incomplete batch during training (to maintain consistent batch size for SigLIP).
- The system SHALL NOT drop the last batch during validation/testing.
Verified when: Training DataLoader produces `ceil(len(dataset) / batch_size) - 1` batches (last dropped). Val/test produce `ceil(len(dataset) / batch_size)` batches.

---

### FR-5.0: Model Architecture

**FR-5.1: Morphology encoder**
Trigger: Forward pass with morph features.
Input: Tensor of shape `[batch_size, morph_dim]`.
Behavior:
- The system SHALL process the input through an MLP with architecture: `Linear(morph_dim, 1024) → BatchNorm → ReLU → Dropout(p) → Linear(1024, 1024) → BatchNorm → ReLU → Dropout(p) → Linear(1024, 512)`.
- Dropout rate `p` SHALL be configurable (default: 0.1).
- Output shape SHALL be `[batch_size, 512]`.
Verified when: Given input of shape `[32, 1500]`, output has shape `[32, 512]` and gradients flow through all parameters.

**FR-5.2: Expression encoder**
Trigger: Forward pass with expr features.
Input: Tensor of shape `[batch_size, expr_dim]`.
Behavior: Identical architecture to FR-5.1 but with `expr_dim` as input dimension.
Verified when: Given input of shape `[32, 978]`, output has shape `[32, 512]`.

**FR-5.3: Molecular encoder**
Trigger: Forward pass with ECFP fingerprint tensors.
Input: Tensor of shape `[batch_size, 2048]`.
Behavior:
- The system SHALL process the input through an MLP: `Linear(2048, 1024) → BatchNorm → ReLU → Dropout(p) → Linear(1024, 1024) → BatchNorm → ReLU → Dropout(p) → Linear(1024, 1024) → BatchNorm → ReLU → Dropout(p) → Linear(1024, 512)`.
- Output shape SHALL be `[batch_size, 512]`.
Verified when: Given input of shape `[32, 2048]`, output has shape `[32, 512]`.

**FR-5.4: Projection heads**
Trigger: After each encoder's forward pass.
Input: Encoder output of shape `[batch_size, 512]`.
Behavior:
- Each modality SHALL have a separate projection head: `Linear(512, 512) → BatchNorm → ReLU → Linear(512, 256)`.
- The output SHALL be L2-normalized to unit length along the feature dimension.
- Output shape SHALL be `[batch_size, 256]` with L2 norm = 1.0 (±1e-6) per row.
Verified when: Output norms are all within [0.9999, 1.0001]. Shape is `[batch_size, 256]`.

**FR-5.5: Modality selection**
Trigger: At model construction time, based on config.
Input: `config.model.modalities` — a list from `[mol, morph, expr]`.
Behavior:
- The system SHALL instantiate encoders and projection heads ONLY for modalities listed in the config.
- For config B4 (`modalities: [mol, morph]`), only mol and morph encoders/heads are created.
- For config T1 (`modalities: [mol, morph, expr]`), all three are created.
- The model's `forward()` SHALL accept a batch dict and return embeddings only for configured modalities.
Verified when: Config B4 model has no `expr_encoder` attribute. Config T1 model has all three.

---

### FR-6.0: Loss Functions

**FR-6.1: SigLIP pairwise loss**
Trigger: Called during training with two embedding tensors.
Input: `z_a` of shape `[N, 256]`, `z_b` of shape `[N, 256]`, learnable bias scalar.
Behavior:
- The system SHALL compute the pairwise cosine similarity matrix: `sim = z_a @ z_b.T` of shape `[N, N]`.
- The system SHALL construct a target matrix `targets` of shape `[N, N]` where `targets[i,j] = +1` if `i == j` (positive pair) and `targets[i,j] = -1` otherwise (negative pair).
- The system SHALL compute the loss as: `mean(-log_sigmoid(targets * sim + bias))` where `bias` is a learnable parameter initialized to 0.
- The system SHALL return a scalar loss value.
Verified when: For identical `z_a = z_b` (perfect alignment), loss is near 0. For random orthogonal embeddings, loss is near `log(2) ≈ 0.693`.

**FR-6.2: VICReg regularization**
Trigger: Called per-modality during training.
Input: Embedding tensor `z` of shape `[N, 256]`.
Behavior:
- **Variance term:** For each feature dimension, compute std across the batch. Apply hinge loss: `max(0, 1 - std)`. Average across dimensions.
- **Covariance term:** Compute the covariance matrix of `z` (shape `[256, 256]`). Sum the squared off-diagonal entries divided by 256.
- The system SHALL return: `variance_loss + covariance_loss` (weighted equally within VICReg; the outer λ weight is applied by the caller).
Verified when: For embeddings that are all identical (collapsed), variance loss > 0 and covariance loss ≈ 0. For well-distributed embeddings, variance loss ≈ 0.

**FR-6.3: Total tri-modal loss**
Trigger: Each training step.
Input: Embeddings `z_mol`, `z_morph`, `z_expr` (whichever are active per config).
Behavior:
- The system SHALL compute SigLIP loss for every pair of active modalities.
- For tri-modal (T1): `L = SigLIP(mol,morph) + SigLIP(mol,expr) + SigLIP(morph,expr)`.
- For bi-modal (e.g., B4): `L = SigLIP(mol,morph)`.
- The system SHALL add VICReg regularization for each active modality: `L += λ * sum(VICReg(z) for z in active_embeddings)`.
- λ SHALL be configurable (default: 0.1).
- The system SHALL return the total loss scalar and a dict of individual loss components for logging.
Verified when: For config T1, the loss dict contains keys `loss_mol_morph`, `loss_mol_expr`, `loss_morph_expr`, `vicreg_mol`, `vicreg_morph`, `vicreg_expr`, and `loss_total`.

---

### FR-7.0: Training Loop

**FR-7.1: Training execution**
Trigger: User runs `python scripts/train.py --config configs/[name].yaml` or `make train`.
Input: Hydra config specifying all hyperparameters.
Behavior:
- The system SHALL seed all random sources (Python, NumPy, PyTorch, CUDA) with `config.seed`.
- The system SHALL log all config values to W&B at run start.
- For each epoch:
  1. Set model to train mode.
  2. Iterate over training DataLoader batches.
  3. Forward pass → compute loss (FR-6.3) → backward pass → clip gradients (max_norm from config) → optimizer step → scheduler step.
  4. Log per-batch metrics to W&B: `train/loss_total`, individual loss components, `train/grad_norm`, `train/temperature` (if applicable).
  5. Set model to eval mode.
  6. Run validation evaluation (FR-8.1) on val set.
  7. Log per-epoch metrics: all val retrieval metrics, alignment/uniformity, val loss.
  8. Check early stopping (FR-7.3).
  9. Check collapse detection (FR-7.4).
- The system SHALL use mixed precision (bfloat16) if CUDA is available.
Verified when: W&B dashboard shows per-batch and per-epoch metrics. Training completes without NaN losses.

**FR-7.2: Checkpointing**
Trigger: At the end of each epoch where validation metric improves.
Input: Model state, optimizer state, current epoch, current best metric.
Behavior:
- The system SHALL save a checkpoint to `checkpoints/{config_name}_seed{seed}.pt` containing: model state dict, optimizer state dict, epoch number, best metric value, full Hydra config.
- The system SHALL overwrite the previous best checkpoint (keep only the best).
- On training completion, the system SHALL print: "Training complete. Best mean_R@10=[X] at epoch [E]. Checkpoint: [path]."
Verified when: Checkpoint file is loadable and model produces identical outputs when restored.

**FR-7.3: Early stopping**
Trigger: End of each epoch.
Input: Current val `mean_R@10`, historical best, patience counter.
Behavior:
- IF `mean_R@10` exceeds the previous best by any margin, the system SHALL reset the patience counter to 0 and save a checkpoint.
- IF `mean_R@10` does not improve, the system SHALL increment the patience counter.
- IF the patience counter reaches `config.training.patience` (default: 30), the system SHALL stop training and print "Early stopping at epoch [E] (patience=[P] exhausted)."
Verified when: Training stops after 30 epochs without improvement. The reported best metric matches the checkpoint.

**FR-7.4: Collapse detection**
Trigger: After each validation evaluation.
Input: Per-modality uniformity scores.
Behavior:
- For each active modality, IF uniformity > −0.5 (near-collapse), the system SHALL print: "COLLAPSE WARNING: [modality] uniformity=[X] (threshold=−0.5). Encoder may be collapsing."
- The system SHALL log `collapse_warning=True` to W&B for any epoch where a warning fires.
- The system SHALL NOT stop training on collapse detection (VICReg should self-correct).
Verified when: When all embeddings are identical (uniformity ≈ 0), the warning fires. When embeddings are well-distributed (uniformity < −2), no warning fires.

---

### FR-8.0: Evaluation

**FR-8.1: Cross-modal retrieval**
Trigger: Called at end of each validation epoch and during standalone evaluation.
Input: All embeddings for all active modalities on the eval set.
Behavior:
- For each pair of active modalities (a, b), the system SHALL:
  1. Compute cosine similarity matrix `sim[i,j] = cos(z_a[i], z_b[j])` of shape `[N, N]`.
  2. For each query embedding `z_a[i]`, rank all candidates `z_b[j]` by descending similarity.
  3. Compute R@1: fraction of queries where the correct match (`j=i`) is ranked #1.
  4. Compute R@5: fraction where correct match is in top 5.
  5. Compute R@10: fraction where correct match is in top 10.
  6. Compute MRR: mean of `1/rank` of the correct match across all queries.
- The system SHALL compute metrics for BOTH directions of each pair (a→b and b→a).
- The system SHALL compute `mean_R@10` as the average R@10 across all computed directions.
- The system SHALL compute compound-level retrieval metrics by averaging embeddings per compound, re-normalizing, then computing R@K on the deduplicated set. This eliminates rank inflation from dose duplication. Compound-level metrics use keys like `compound/mol->morph/R@10`, `compound/mean_R@10`.
- Early stopping uses `compound/mean_R@10` as the primary metric (falls back to `mean_R@10` if compound IDs are unavailable).
- The system SHALL return a dict with keys like `mol->morph/R@1`, `morph->mol/R@1`, etc.
Verified when: For identical embeddings across modalities, R@1 = 1.0. For random embeddings with N=100, R@10 ≈ 0.10 (±0.03). Compound-level R@1 = 1.0 for perfect alignment even when row-level R@1 < 1.0 due to dose duplication.

**FR-8.2: Alignment and uniformity**
Trigger: Called alongside retrieval evaluation.
Input: Embeddings for all active modalities.
Behavior:
- **Alignment** (per modality pair): mean L2 distance between positive pairs: `mean(||z_a[i] - z_b[i]||^2)`. Lower is better.
- **Uniformity** (per modality): `log(mean(exp(-2 * ||z[i] - z[j]||^2)))` for all pairs i≠j. More negative is better. Values > −0.5 indicate collapse.
- The system SHALL return a dict with keys like `align_mol_morph`, `uniform_mol`, etc.
Verified when: Collapsed embeddings produce uniformity ≈ 0. Well-distributed embeddings produce uniformity < −2.

**FR-8.3: MOA clustering**
Trigger: During standalone evaluation (`python scripts/evaluate.py --clustering`).
Input: Embeddings, MOA labels (where available).
Behavior:
- The system SHALL filter to only compounds with non-null MOA annotations.
- The system SHALL compute k-NN MOA accuracy for k=5,10,20: for each compound, classify by majority vote of neighbors' MOA labels. Report accuracy.
- The system SHALL compute AMI (Adjusted Mutual Information) between k-means clusters (k = number of unique MOAs) and ground-truth MOA labels.
- The system SHALL compute ARI (Adjusted Rand Index) between the same clusters and labels.
- The system SHALL log: "MOA clustering: [N] compounds with MOA labels. AMI=[X], ARI=[Y], kNN-5 acc=[Z]."
Verified when: With random embeddings, AMI ≈ 0 and kNN accuracy ≈ 1/(num_classes). With perfectly clustered embeddings, AMI ≈ 1.0.

**FR-8.4: Full evaluation report**
Trigger: User runs `python scripts/evaluate.py --checkpoint [path] --full`.
Input: Trained model checkpoint, test dataset.
Behavior:
- The system SHALL run FR-8.1 (retrieval), FR-8.2 (alignment/uniformity), and FR-8.3 (MOA clustering).
- The system SHALL generate and save:
  - `results/retrieval_table.csv` — all R@K and MRR values per direction
  - `results/retrieval_table.tex` — LaTeX-formatted version
  - `results/umap_{modality}.png` — UMAP visualization for each modality, colored by MOA where available
  - `results/similarity_heatmap.png` — 6-panel heatmap of similarity matrices for all directions
  - `results/training_curves.png` — loss and R@10 over epochs (loaded from W&B or local log)
- The system SHALL print a formatted summary table to stdout.
Verified when: All output files exist and are non-empty. The summary table contains all 6 retrieval directions.

---

### FR-9.0: Ablation Harness

**FR-9.1: Ablation matrix execution**
Trigger: User runs `python scripts/run_ablations.py --matrix core`.
Input: Ablation matrix definition in `configs/ablation/core.yaml`.
Behavior:
- The system SHALL read the matrix definition specifying 8 configs with variable seeds (4 baselines × 1 seed + 4 trained × 5 seeds = 24 runs).
- For each run, the system SHALL:
  1. Check if a matching checkpoint exists (`checkpoints/{config}_{seed}.pt`). If yes and `--resume` flag is set, skip.
  2. Otherwise, execute `train.py` with the specified config overrides and seed.
  3. After training, execute `evaluate.py` on the best checkpoint.
  4. Append results to `results/ablation_runs.jsonl` (one JSON line per run).
- The system SHALL print progress: "Run [X]/24: config=[name], seed=[S]. Status: [running/complete/skipped]."
Verified when: After full execution, `results/ablation_runs.jsonl` has exactly 24 lines, each parseable as JSON with keys `config`, `seed`, `mean_R@10`, and all other metrics.

**FR-9.2: Ablation summary generation**
Trigger: User runs `python scripts/summarize_ablations.py` or automatically after FR-9.1 completes.
Input: `results/ablation_runs.jsonl`.
Behavior:
- The system SHALL compute mean ± std for each metric across seeds, grouped by config.
- The system SHALL compute Welch's t-test (two-sided) between T1 (tri-modal) and each bi-modal config (B4, B5, B6) for `mean_R@10`. Apply Bonferroni correction (3 comparisons, α = 0.05/3 = 0.0167).
- The system SHALL generate:
  - `results/ablation_summary.csv` — mean ± std per config per metric
  - `results/ablation_comparison.tex` — LaTeX table with significance stars (* p<0.05, ** p<0.01, *** p<0.001)
  - `results/ablation_barplot.png` — bar chart of mean_R@10 per config with error bars (std)
- The system SHALL print the comparison table to stdout.
Verified when: CSV has 8 rows (one per config). p-values are between 0 and 1. Significance stars match p-value thresholds.

---

### FR-10.0: Configuration Management

**FR-10.1: Hydra config structure**
Trigger: At application startup.
Input: YAML config files in `configs/`.
Behavior:
- The system SHALL use Hydra for all configuration management.
- Config directory structure SHALL be:
  ```
  configs/
  ├── default.yaml           # Full default config
  ├── data/
  │   └── lincs.yaml         # Dataset-specific settings
  ├── model/
  │   ├── tri_modal.yaml     # T1 config
  │   ├── bi_mol_morph.yaml  # B4 config
  │   ├── bi_mol_expr.yaml   # B5 config
  │   └── bi_morph_expr.yaml # B6 config
  ├── training/
  │   └── default.yaml       # Training hyperparameters
  └── ablation/
      └── core.yaml          # 8-config × 5-seed matrix definition
  ```
- Every training run SHALL log the resolved config to W&B and to `checkpoints/{run_name}_config.yaml`.
- The system SHALL support command-line overrides: `python scripts/train.py model=bi_mol_morph training.batch_size=128 seed=123`.
Verified when: `python scripts/train.py --cfg job` prints the full resolved config. Overrides change the logged config.

**FR-10.2: Reproducibility guarantees**
Trigger: Always.
Behavior:
- The system SHALL seed Python (`random`), NumPy (`np.random`), PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`), and set `torch.backends.cudnn.deterministic = True`.
- The system SHALL log the git commit hash (if in a git repo) to W&B metadata.
- Two runs with identical config + seed + code SHALL produce identical val metrics (within floating-point tolerance of ±1e-5).
Verified when: Running the same config+seed twice produces identical `mean_R@10` values.

---

### FR-11.0: Logging & Monitoring

**FR-11.1: W&B logging**
Trigger: During training.
Behavior:
- The system SHALL log to W&B project `capy-v2` with run name `{config_name}_seed{seed}`.
- Per-batch: `train/loss_total`, `train/loss_{pair}` for each active pair, `train/grad_norm`.
- Per-epoch: all val retrieval metrics (FR-8.1), alignment/uniformity (FR-8.2), val loss, learning rate, epoch number.
- Run metadata: full config, git hash, GPU type, total parameters, dataset sizes.
- The system SHALL tag each run with: `config={config_name}`, `seed={seed}`, `dataset=lincs`.
Verified when: W&B dashboard shows all listed metrics. Runs are filterable by config tag.

**FR-11.2: Console logging**
Trigger: During all operations.
Behavior:
- The system SHALL use Python's `logging` module with format: `{timestamp} | {module} | {level} | {message}`.
- Level SHALL default to INFO. Configurable via `--verbose` (DEBUG) or `--quiet` (WARNING).
- All warnings (collapse detection, data quality issues, missing SMILES) SHALL use WARNING level.
- Errors SHALL use ERROR level and include the specific failure reason and suggested fix.
Verified when: Running with `--quiet` suppresses INFO messages. Running with `--verbose` shows DEBUG messages.

---

## 5. Edge Cases & Boundary Conditions

### 5.1 Data edge cases

| Edge Case | Expected Behavior |
|-----------|-------------------|
| A SMILES string contains CXSMILES extended notation (e.g., `CC |a:1,TLB:...|`) | Strip everything after ` \|` and retry RDKit parsing. Log warning if original parse fails. |
| A compound has only 1 replicate (no pairwise correlation possible) | Include it — replicate correlation filter only applies to compounds with ≥2 replicates. Assign weight=1.0 in MODZ. |
| A compound has replicates in morphology but not expression (or vice versa) | Exclude from final dataset — all three modalities must be present per FR-2.3. |
| All features are NaN for a single treatment | Exclude the treatment. Log: "WARNING: Treatment [ID] has all-NaN morph/expr features. Excluded." |
| A feature has exactly 5.0% NaN (at the threshold) | Keep the feature (threshold is >5%, strictly greater). |
| Batch size > dataset size | Training DataLoader produces 0 batches (all dropped). The system SHALL detect this and error: "ERROR: batch_size [B] > dataset size [N]. Reduce batch_size." |
| Only 1 unique MOA label exists in test set | MOA clustering (FR-8.3) SHALL skip and log: "WARNING: Only 1 MOA class found. Skipping clustering metrics." |
| Two compounds share the same scaffold but different SMILES | Both go in the same split (scaffold grouping is correct). They are separate treatments. |

### 5.2 Training edge cases

| Edge Case | Expected Behavior |
|-----------|-------------------|
| NaN appears in loss | Training SHALL halt immediately. Print: "ERROR: NaN loss detected at epoch [E], batch [B]. Check data preprocessing or reduce learning rate." |
| Gradient norm exceeds 100.0 (10× the clip threshold) | Log: "WARNING: Gradient explosion detected (norm=[X]). Clipping applied." |
| All modality pairs converge except one (e.g., morph↔expr loss → 0.01 but mol↔morph loss → 0.5) | Not an error condition — this is expected. Log individual loss components for diagnosis. |
| VICReg variance loss stays >0.5 for >20 epochs | Log: "WARNING: VICReg variance loss persistently high for [modality]. Embeddings may have low variance despite regularization." |
| Val loss increases while train loss decreases (overfitting) | Not an error — early stopping (FR-7.3) handles this. The best checkpoint by R@10 is preserved. |

### 5.3 Evaluation edge cases

| Edge Case | Expected Behavior |
|-----------|-------------------|
| Test set has only 10 compounds (very small) | Retrieval metrics are computed but noisy. Log: "WARNING: Small test set (N=10). Retrieval metrics may be unreliable." |
| Multiple compounds share identical embeddings | Retrieval ties are broken by index order (first match wins). This indicates potential collapse. |
| A modality pair has R@10 < random baseline (< 10/N) | Log: "WARNING: [direction] R@10 = [X] is below random baseline ([Y]). Model may be worse than random for this direction." |

---

## 6. Non-Functional Constraints

| Constraint | Specification | Rationale |
|-----------|--------------|-----------|
| Single training run time | SHALL complete in ≤30 minutes on a Google Colab H100 GPU | 24-run matrix must finish in ~1 day |
| Peak GPU memory | SHALL not exceed 16 GB for batch_size=128 | Compatible with Colab H100 (80 GB HBM3); constraint kept conservative for portability to T4/V100 |
| Data download | SHALL complete in ≤30 minutes on a 100 Mbps connection | ~350 MB total download |
| Reproducibility | Two identical runs SHALL produce identical val metrics (±1e-5) | Scientific rigor requirement |
| Code quality | `ruff check src/` SHALL produce 0 errors; `black --check src/` SHALL pass | Engineering standard |
| Test coverage | `pytest --cov=src/` SHALL report ≥80% line coverage on `src/data/`, `src/models/`, `src/evaluation/` | Critical paths must be tested |

---

## 7. Non-Goals & Out of Scope

These items are explicitly excluded. AI agents and developers SHALL NOT implement them without explicit approval.

- **NG1:** Do NOT implement a web UI, Streamlit app, or any interactive visualization server.
- **NG2:** Do NOT implement raw image processing or any vision backbone. All morphology input is pre-extracted CellProfiler features.
- **NG3:** Do NOT implement any GNN encoder (GIN, GAT, SchNet, etc.). The molecular encoder is ECFP + MLP only.
- **NG4:** Do NOT implement InfoAlign's context graph, random walk sampling, or decoder-based information bottleneck.
- **NG5:** Do NOT combine data from different cell lines (U2OS + A549) in the same training run.
- **NG6:** Do NOT implement molecule generation, SMILES generation, or any generative model.
- **NG7:** Do NOT implement model serving, inference API, or ONNX export.
- **NG8:** Do NOT add authentication, user accounts, or multi-tenancy.
- **NG9:** Do NOT implement hyperparameter optimization frameworks (Optuna, Ray Tune). Use Hydra multirun with manually specified grids.

---

## 8. Open Issues & Decision Log

### 8.1 Open issues

All issues resolved as of 2026-03-11. See Decision Log below.

### 8.2 Decision log

| Date | Decision | Decided By | Rationale |
|------|----------|-----------|-----------|
| 2026-03-10 | Use SigLIP loss instead of InfoNCE | Hoang | InfoNCE is information-theoretically limited with small batches (log K bound). SigLIP operates pairwise with no global softmax, eliminating batch-size sensitivity. |
| 2026-03-10 | Use ECFP + MLP instead of GIN | Hoang | GIN collapsed at N=500 (uniformity=−0.22). Benchmarks show ECFP matches/beats neural encoders. CLOOME validated this approach. |
| 2026-03-10 | Add VICReg regularization | Hoang | CaPy v1 showed encoder collapse. VICReg variance term directly prevents this by maintaining embedding std > 1. |
| 2026-03-10 | Use LINCS (A549) instead of CDRP-bio (U2OS) | Hoang | Rosetta paper shows LINCS has markedly better modality alignment. 15× more data after QC. Single cell line avoids confounding. |
| 2026-03-10 | Scaffold split grouped by compound (all doses in same split) | Hoang | Prevents data leakage from structurally similar molecules across splits. Different doses of the same compound would leak molecular identity. |
| 2026-03-11 | OPEN-1: Dose as augmentation with quality filtering | Hoang | SupCon-style positive pairing across doses, but only treatments passing replicate-correlation filter (90th percentile DMSO null). Effective N ~3,000–5,000. Excludes near-DMSO and cytotoxic profiles. |
| 2026-03-11 | OPEN-2: L1000 Level 5 (treatment-level) | Hoang | Better SNR via MODZ aggregation. Level 4 replicates are not independent. Contrastive framework needs 1:1:1 treatment pairing. Level 4 downloaded as optional backup. |
| 2026-03-11 | OPEN-3: SCARF 40% default, sweep 20/40/60% in Week 2 | Hoang | Cell Painting features highly correlated — higher corruption tolerable. Quick sweep on morph↔expr (1 seed each) to validate. |
| 2026-03-11 | OPEN-4: InfoAlign comparison deferred to P2 | Hoang | Asymmetric architecture (2/6 directions), different cell line (U2OS vs A549). Frame as out-of-distribution evaluation if included. |

---

## Appendix: Repository Structure

```
capy/
├── Makefile                     # setup, preprocess, train, evaluate, test, lint
├── Dockerfile                   # Reproducible environment
├── pyproject.toml               # Dependencies
├── CLAUDE.md                    # Claude Code instructions
├── PRD.md                       # Product requirements
├── FSD.md                       # This document
├── configs/
│   ├── default.yaml
│   ├── data/lincs.yaml
│   ├── model/{tri_modal,bi_mol_morph,...}.yaml
│   ├── training/default.yaml
│   └── ablation/core.yaml
├── data/
│   ├── raw/                     # Downloaded (gitignored)
│   ├── processed/               # QC'd + normalized (gitignored)
│   └── reports/                 # Data audit reports
├── src/
│   ├── data/
│   │   ├── download.py          # FR-1.x
│   │   ├── audit.py             # FR-1.4
│   │   ├── preprocess.py        # FR-2.x
│   │   ├── featurize.py         # FR-3.x
│   │   └── dataset.py           # FR-4.x
│   ├── models/
│   │   ├── encoders.py          # FR-5.1–5.3
│   │   ├── projections.py       # FR-5.4
│   │   ├── capy.py              # FR-5.5 (model assembly)
│   │   └── losses.py            # FR-6.x
│   ├── training/
│   │   └── trainer.py           # FR-7.x
│   ├── evaluation/
│   │   ├── retrieval.py         # FR-8.1
│   │   ├── diagnostics.py       # FR-8.2
│   │   ├── clustering.py        # FR-8.3
│   │   └── report.py            # FR-8.4
│   └── utils/
│       ├── config.py            # FR-10.x
│       ├── logging.py           # FR-11.x
│       └── seeding.py           # FR-10.2
├── scripts/
│   ├── download.py              # CLI entry: make setup
│   ├── preprocess.py            # CLI entry: make preprocess
│   ├── train.py                 # CLI entry: make train
│   ├── evaluate.py              # CLI entry: make evaluate
│   ├── run_ablations.py         # FR-9.1
│   └── summarize_ablations.py   # FR-9.2
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_retrieval.py
│   └── test_featurize.py
├── results/                     # Generated outputs (gitignored)
└── checkpoints/                 # Model checkpoints (gitignored)
```
