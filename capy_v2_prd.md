# CaPy v2 — Product Requirements Document

**Contrastive Alignment of Phenotypic Yields: Tri-Modal Contrastive Learning for Drug Discovery**

| Field | Value |
|-------|-------|
| Author | Hoang Ngo |
| Version | 2.0 |
| Date | March 2026 |
| Status | Implementation-Ready |

---

## 1. Problem Statement

When pharmaceutical companies treat cells with drug candidates, they generate data in three fundamentally different formats: molecular structures (what the drug looks like as a chemical), cell morphology images (how the cell physically changes), and gene expression profiles (which genes turn on or off). Today, each data type lives in a silo. A chemist examines molecular fingerprints, a microscopist examines Cell Painting images, a genomicist examines transcriptomic profiles. No unified computational framework exists that learns a shared representation across all three modalities simultaneously, and no published work has rigorously demonstrated whether combining all three produces better drug similarity search and mechanism-of-action prediction than any pair alone.

This fragmentation has a concrete cost: drug discovery teams cannot ask cross-modal questions like "given this gene expression signature, which molecular scaffolds and morphological phenotypes are most similar?" without building separate, brittle pipelines for each direction. InfoAlign (ICLR 2025) partially addresses this for molecular representations, but its asymmetric, molecule-centric design cannot encode morphology or expression independently, cannot perform symmetric cross-modal retrieval, and conflates data from different cell lines (U2OS and A549) without accounting for the cell-line dependence of cross-modality alignment.

**CaPy v2 fills this gap:** a symmetric tri-modal contrastive framework that learns a shared 256-dimensional embedding space where molecular structure, cell morphology, and gene expression are jointly aligned, enabling any-to-any retrieval across all six cross-modal directions.

---

## 2. Why Now / Strategic Context

### 2.1 The field is converging on multi-modal bio-ML

Four papers in the last 18 months signal that multi-modal molecular representation learning is a top-tier research direction: InfoAlign (ICLR 2025), CellCLIP (NeurIPS 2025), CLOOME (Nature Communications 2023), and CHMR (arXiv Nov 2025). Yet none has rigorously answered the most basic question: does tri-modal alignment actually beat bi-modal? InfoAlign sidesteps it with a decoder-based architecture. CellCLIP only handles two modalities. CLOOME aligns molecules with morphology but ignores gene expression entirely.

### 2.2 Alignment with insitro's platform thesis

insitro's core thesis is that purpose-built multi-modal cellular data combined with ML creates a predictive "data tensor" for drug behavior. Their CellPaint-POSH platform (Nature Communications 2025) generates Cell Painting data, their ContrastiveVI+ handles transcriptomics, and their molecular libraries provide chemical structure. CaPy directly tests whether unifying these three data types in a shared embedding space outperforms any pair — addressing a question insitro's own publications identify as important but haven't yet answered.

### 2.3 Cost of not doing this

Without a rigorous tri-modal ablation study, the field will continue building increasingly complex systems (InfoAlign's context graph has 277K nodes) without knowing whether the marginal value of each modality justifies the engineering investment. CaPy's contribution is both a scientific result (quantified tri-modal improvement) and an engineering artifact (a clean, reproducible framework).

---

## 3. Target Users & Use Cases

### 3.1 Primary persona: ML Engineer at a phenomics-driven drug discovery company

An ML engineer at insitro, Recursion, or a pharma computational biology group who needs to evaluate whether investing in tri-modal data integration (which costs more than bi-modal) yields measurably better representations. They have access to Cell Painting, transcriptomic, and molecular data but lack a framework to systematically compare modality combinations.

### 3.2 Secondary persona: Computational biology researcher

A researcher studying the relationship between molecular structure, cellular morphology, and gene expression who needs interpretable embeddings showing which morphological features align with which gene sets.

### 3.3 Use cases

- **UC1 (Cross-modal retrieval):** "As an ML engineer, I want to query the embedding space with a molecular structure and retrieve the most similar morphological and expression profiles, so I can predict a compound's phenotypic effect without running the assay."
- **UC2 (Modality ablation):** "As a platform architect, I want quantified evidence of whether adding gene expression to a molecule+morphology alignment improves retrieval by a statistically significant margin, so I can justify the data acquisition cost."
- **UC3 (MOA clustering):** "As a biologist, I want compounds with the same mechanism of action to cluster together in the tri-modal embedding space, so I can use embedding similarity as a proxy for functional similarity."

---

## 4. Success Metrics

Every metric has a baseline (current CaPy v1 or published comparable) and a target.

| Metric | Baseline | Target | Rationale |
|--------|----------|--------|-----------|
| **Mean R@10 (6 dirs)** | 0.25 (CaPy v1) | **0.30+** | 3.4× above random (8.93%) |
| **Tri > best bi-modal R@10** | No published comparison | **+5% relative** | Core thesis validation |
| **MOA clustering AMI** | ~0.05 (random) | **0.15+** | Biological meaningfulness |
| **Statistical significance** | N/A | **p < 0.05 (5 seeds)** | Tri vs each bi-modal pair |

**Leading indicator:** Training loss convergence without val divergence by epoch 30 (currently val loss diverges by epoch 5).

**Lagging indicator:** Downstream linear probing AUC on activity prediction improves with tri-modal embeddings vs. bi-modal.

### 4.1 Success / failure decision criteria

- **Clear success:** Tri-modal R@10 exceeds best bi-modal by ≥2% absolute AND p < 0.05 across 5 seeds on at least 4 of 6 retrieval directions.
- **Partial success:** Tri-modal improves on ≥1 metric category (retrieval OR MOA clustering OR linear probing) with statistical significance. Pivot narrative to "complementarity analysis — the third modality helps for specific MOA classes."
- **Null result:** No statistically significant improvement on any metric. Still publishable as a rigorous negative result with the ablation methodology as the contribution. Reframe as "quantifying the ceiling of contrastive alignment for biological multi-modal data."

---

## 5. Proposed Solution / Key Features

### 5.1 P0 (Must Have) — Core Framework

**F1: LINCS dataset pipeline.**
Download, QC, normalize, and scaffold-split the Rosetta LINCS dataset (1,327 compounds × 6 doses, A549 cells) with matched Cell Painting + L1000 profiles.
WHY: 15× more paired data than CDRP-bio with 3–5× better cross-modality alignment quality.

**F2: Symmetric tri-modal contrastive model.**
Three MLP encoders (ECFP molecular fingerprints, CellProfiler features, L1000 z-scores) projecting into a shared 256-dim space, trained with pairwise SigLIP loss + VICReg regularization.
WHY: SigLIP eliminates InfoNCE's batch-size sensitivity; VICReg prevents the encoder collapse observed in v1.

**F3: 8-condition ablation matrix.**
Random, 3 single-modality, 3 bi-modal, 1 tri-modal — each with 5 seeds (40 total runs).
WHY: This is the core scientific contribution; no published work has done systematic tri vs. bi comparison.

**F4: Comprehensive evaluation suite.**
R@1/5/10, MRR across all 6 retrieval directions, MOA clustering (AMI/ARI), alignment/uniformity metrics (Wang & Isola 2020).
WHY: InfoAlign only evaluates molecule-centric retrieval on 80–196 pairs. CaPy evaluates 1,327+ compounds across all 6 directions.

### 5.2 P1 (Should Have) — Engineering Quality

**F5: Hydra config + W&B experiment tracking.**
WHY: Reproducibility and systematic hyperparameter management matching insitro's internal practices.

**F6: SCARF tabular augmentation.**
30–60% feature corruption with empirical marginal replacement for morphology and expression inputs.
WHY: Standard for tabular contrastive learning; effectively doubles positive pairs.

**F7: Staged bi-modal → tri-modal training.**
Pre-train morph↔expr, then add molecular encoder.
WHY: Prevents weakest modality pair from causing collapse during early training.

### 5.3 P2 (Could Have) — Extensions

**F8: ChemBERTa-2 encoder option.**
Frozen pretrained transformer as alternative molecular encoder.
WHY: Upgrade path if ECFP+MLP plateaus.

**F9: Interpretability analysis.**
Feature importance mapping between morphology features and gene sets via GSEA.
WHY: Demonstrates biological curiosity beyond pure ML engineering.

**F10: InfoAlign direct comparison.**
Run InfoAlign's pretrained model on CaPy's evaluation protocol.
WHY: Fair head-to-head comparison on symmetric retrieval.

---

## 6. Scope & Non-Goals

### 6.1 Explicit non-goals for this phase

- **NG1:** We will NOT use raw microscopy images. We use pre-extracted CellProfiler features (~1,500 dims). Using raw images would require a vision backbone and 100× more compute, changing the project scope entirely.
- **NG2:** We will NOT implement InfoAlign's context graph architecture. CaPy's contribution is a simpler contrastive approach, not a reimplementation of their decoder-based information bottleneck. We compare against InfoAlign, not replicate it.
- **NG3:** We will NOT train GNN molecular encoders from scratch. The retrospective analysis proved from-scratch GINs collapse at N<2000. We use fixed ECFP fingerprints or pretrained encoders only.
- **NG4:** We will NOT combine data from different cell lines. Unlike InfoAlign (which mixes U2OS and A549), CaPy uses only A549 LINCS data to avoid cell-line confounding. This is a deliberate scientific choice, not a limitation.
- **NG5:** We will NOT generate novel molecules or claim clinical relevance. This is a methods project demonstrating multi-modal alignment, not a drug candidate pipeline.
- **NG6:** We will NOT build a production deployment or web application. The deliverable is a reproducible research codebase with results, not a serving system.

### 6.2 Future considerations (Icebox)

- Scale to JUMP cpg0016 (~116K compounds) if cell-line-matched L1000 data becomes available
- Replace CellProfiler features with learned image embeddings (e.g., from CellPaint-POSH's CP-DINO)
- Extend to genetic perturbations (CRISPR knockouts via JUMP-ORF)
- Streamlit interactive embedding explorer for compound browsing

---

## 7. Competitive & Market Context

| System | Modalities | Mol Encoder | Cell Line | Symmetric Retrieval | Tri vs Bi Ablation |
|--------|-----------|-------------|-----------|--------------------|--------------------|
| **CaPy v2** | Mol+CP+L1000 | ECFP+MLP | A549 (single) | **All 6 directions** | **Full 8-condition** |
| InfoAlign (ICLR 2025) | Mol+CP+L1000 | GIN (pretrained) | Mixed (U2OS+A549) | Mol-centric only | No |
| CLOOME (NatComm 2023) | Mol+CP | FC on descriptors | U2OS | 2 directions | N/A (bi only) |
| CellCLIP (NeurIPS 2025) | Image+Text | Text encoder | U2OS | 2 directions | N/A (bi only) |
| CHMR (arXiv Nov 2025) | Mol+CP+L1000 | Hierarchical VQ | Mixed | Mol-centric | Partial |

**CaPy's unique positioning:** The only framework that (a) aligns all three modalities symmetrically, (b) uses a single cell line for biological soundness, (c) provides systematic tri-modal vs. bi-modal ablation evidence, and (d) enables all 6 cross-modal retrieval directions.

### 7.1 Five exploitable gaps in InfoAlign

1. **Cell line conflation.** InfoAlign's context graph mixes U2OS Cell Painting data (JUMP-CP, Bray 2017) with A549/multi-cell-line L1000 data. The Rosetta paper shows cross-modality alignment is cell-line dependent. CaPy's single-cell-line approach is biologically sounder.
2. **No systematic tri vs. bi ablation.** InfoAlign's Table 4 removes modalities from decoder targets but never compares pure bi-modal alignment against full tri-modal. CaPy's 8-condition matrix fills this gap.
3. **Asymmetric, molecule-centric design.** InfoAlign cannot encode morphology or expression independently. It cannot perform morph→expr or expr→morph retrieval. CaPy enables all 6 directions.
4. **Limited retrieval evaluation.** InfoAlign evaluates on 80 pairs (ChEMBL2K) and 196 pairs (Broad6K). CaPy evaluates on 1,327+ compounds — far more statistically powered.
5. **Complexity vs. reproducibility.** InfoAlign requires a 277K-node context graph, random walk sampling, and VAE-style encoding. CaPy's simpler architecture signals engineering maturity.

---

## 8. Risks, Assumptions & Dependencies

### 8.1 Key assumptions

- **A1: Tri-modal > bi-modal is achievable.** HYPOTHESIS. If morphology and expression share only ~27% of MOA signal (Way et al., Cell Systems 2022), the third modality provides complementary information. If the two phenotypic modalities are largely redundant, tri-modal may not significantly outperform the best bi-modal pair. MITIGATION: Even a null result is a publishable finding with proper ablation methodology.
- **A2: 1,327 compounds is sufficient for contrastive learning.** VALIDATED BY LITERATURE. CLOOME trained on ~30K but achieved good retrieval. With 6 doses per compound (~7,900 treatment pairs) and SigLIP loss (no batch-size limitation), this is above the ~2,000-sample empirical minimum.
- **A3: ECFP fingerprints are sufficient molecular representations.** VALIDATED BY BENCHMARKS. Praski et al. (2025) showed nearly all neural molecular encoders fail to outperform ECFP across 25 datasets. CLOOME used descriptor-based FC networks for this reason.

### 8.2 Technical risks

| Risk | Severity | Probability | Mitigation |
|------|----------|------------|------------|
| Encoder collapse recurs | HIGH | LOW (VICReg + ECFP) | Monitor alignment/uniformity every epoch; VICReg variance term directly prevents collapse |
| Val loss divergence (overfitting) | HIGH | MEDIUM | SigLIP pairwise loss is less prone to overfitting than InfoNCE; SCARF augmentation; early stopping |
| Tri-modal shows no improvement over bi-modal | MEDIUM | MEDIUM (~40%) | Reframe as complementarity analysis; identify which MOA classes benefit from third modality |
| LINCS data has unexpected quality issues | MEDIUM | LOW | Way et al. already validated this dataset extensively; data audit in Week 1 catches issues early |

### 8.3 Dependencies

- **D1: AWS S3 public access** to Cell Painting Gallery (cpg0004-lincs). No credentials needed (`--no-sign-request`).
- **D2: Figshare access** for matched L1000 profiles. Public download, no credentials.
- **D3: Drug Repurposing Hub** for SMILES metadata. Public download.
- **D4: Google Colab H100 GPU.** Training takes <30 min per run on Colab H100 (80 GB HBM3); full 40-run ablation matrix takes ~20 GPU-hours. Code remains portable to T4/V100/A100.

---

## 9. Milestones & Phasing

### Phase 1: Foundation (Weeks 1–3)

**Gate: Bi-modal mol↔morph beats random baseline (R@10 > 15%).**

- **Week 1 — Data audit + repo scaffold.**
  - Set up repo: Hydra configs, W&B tracking, Docker environment, pre-commit hooks (black, ruff)
  - Download LINCS data from all three sources (CP Gallery S3, Figshare L1000, Repurposing Hub SMILES)
  - Run data audit script: feature distributions, NaN rates, replicate correlations, compound overlap with metadata
  - **Hard deliverable:** `data/reports/lincs_audit.md` confirming usable sample count (target: ≥5,000 treatment pairs after QC)
  - Data exploration notebook

- **Week 2 — Encoders + evaluation.**
  - Implement all three MLP encoders, ECFP featurization via RDKit
  - Implement SCARF augmentation for morphology and expression inputs
  - Build full evaluation suite (R@K, MRR, AMI/ARI, alignment/uniformity)
  - Implement scaffold-based train/val/test splits (70/15/15)
  - Verify each encoder produces non-degenerate embeddings (uniformity < −1.0)
  - **Hard deliverable:** All encoders passing shape/gradient tests

- **Week 3 — First bi-modal baseline.**
  - Implement SigLIP loss + VICReg regularization
  - Train molecule↔morphology alignment (the CLOOME-equivalent)
  - Run random baseline (B0) and all single-modality baselines (B1–B3)
  - Monitor alignment and uniformity metrics throughout training
  - **🚦 GO/NO-GO:** Bi-modal mol↔morph R@10 > 15%. If not, debug data pairing and loss before proceeding.

### Phase 2: Core Development (Weeks 4–6)

**Gate: Tri-modal beats best bi-modal on at least one metric category, OR clear pivot plan defined.**

- **Week 4 — All bi-modal pairs + tri-modal.**
  - Train B5 (Mol↔Expr) and B6 (Morph↔Expr)
  - Implement tri-modal loss (sum of 3 pairwise SigLIP + VICReg)
  - Begin first tri-modal training runs (3 seeds)
  - **Hard deliverable:** All bi-modal baselines complete with logged metrics

- **Week 5 — The critical comparison.**
  - Complete tri-modal training
  - Generate full comparison table: tri-modal vs. all bi-modal pairs across all metrics
  - Produce UMAP visualizations colored by MOA
  - **🚦 CRITICAL GO/NO-GO:** Does tri-modal beat best bi-modal? If yes → proceed. If marginal → investigate staged training + loss weighting. If no → execute pivot (complementarity analysis).

- **Week 6 — Optimization.**
  - Hyperparameter sweep: temperature (0.05, 0.07, 0.1), projection dim (128, 256, 512), LR, batch size
  - Experiment with staged vs. joint training
  - Test DCL as loss alternative
  - Run 5-seed experiments on best configuration
  - **Hard deliverable:** Optimized tri-modal model with quantified improvement margins

### Phase 3: Ablations & Rigor (Weeks 7–9)

**Gate: Complete 40-run matrix with p-values for all key comparisons.**

- **Week 7 — Full ablation execution.**
  - Launch all 40 core runs (8 configs × 5 seeds) via Hydra multirun
  - Begin loss function ablation (SigLIP vs. DCL vs. InfoNCE)
  - Start downstream evaluation (linear probing on activity prediction)
  - **Hard deliverable:** All 40 core experiments submitted and running

- **Week 8 — Statistical analysis.**
  - Compute Welch's t-test with Bonferroni correction for all key comparisons
  - Run encoder choice ablation (ECFP vs. ChemBERTa vs. pretrained GIN) if time permits
  - Ablate contribution of each modality following InfoAlign's Table 4 approach
  - Generate all publication-quality figures and tables
  - **Hard deliverable:** Complete results table with confidence intervals and p-values

- **Week 9 — Robustness + failure analysis.**
  - Dataset size ablation (25%, 50%, 75%, 100%)
  - Analyze which MOA classes benefit most from tri-modal alignment
  - Document failure cases (compound pairs where tri-modal degrades)
  - Compare against InfoAlign pretrained model on shared evaluation protocol (P2)
  - **Hard deliverable:** Robustness analysis document

### Phase 4: Polish & Ship (Weeks 10–12)

**Gate: `make train && make evaluate` reproduces all results from a clean clone.**

- **Week 10 — Code quality.**
  - Complete docstrings, type hints, unit tests (≥80% coverage on `src/`)
  - Comprehensive README with results summary and reproduction instructions
  - Ensure Docker builds and produces identical results

- **Week 11 — Write-up + presentation.**
  - Technical report (2–3 page PDF)
  - Presentation slides (15 min talk)
  - Demo notebook walking through key results

- **Week 12 — Buffer + interview prep.**
  - Final reproducibility check from clean clone
  - Practice explaining tradeoffs under technical questioning
  - Prepare answers for: "Why not use InfoAlign directly?", "What if tri doesn't beat bi?", "How would this scale to 100K compounds?"

---

## 10. Open Questions & Q&A Log

### 10.1 Unresolved questions

All questions resolved as of 2026-03-11. See Section 10.2.

### 10.2 Decisions made

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-10 | Switch from CDRP-bio to LINCS dataset | Rosetta paper shows LINCS has markedly better modality alignment; 15× more data after QC |
| 2026-03-10 | Replace from-scratch GIN with ECFP + MLP | GIN uniformity=−0.22 confirmed collapse; benchmarks show ECFP matches/beats neural encoders at this scale |
| 2026-03-10 | Replace InfoNCE with SigLIP + VICReg | InfoNCE information-theoretically limited with 63 negatives; SigLIP is batch-size agnostic; VICReg prevents collapse |
| 2026-03-10 | Position CaPy to build on InfoAlign, not compete | InfoAlign is ICLR 2025 from the Broad — acknowledge and fill their systematic ablation gap |
| 2026-03-10 | Core contribution = tri > bi proof with ablations, not novel architecture | Signals engineering rigor + scientific methodology over architectural novelty; matches ML Engineering role |
| 2026-03-11 | Q1: Dose as augmentation with quality filtering | Use SupCon-style positive pairing across doses of the same compound, but only for treatments that pass replicate-correlation filter (90th percentile of DMSO null). Effective N ~3,000–5,000. Low-dose/near-DMSO and cytotoxic profiles are excluded before pairing. Dose-aware pairing improvement is a reportable finding. |
| 2026-03-11 | Q4: L1000 Level 5 (treatment-level aggregated) | Level 5 provides better SNR via Broad's MODZ aggregation. Level 4 replicates aren't independent (technical replicates of same condition), so "more samples" is misleading. Contrastive framework pairs treatments 1:1:1 across modalities — Level 4 would require re-aggregation anyway. Download Level 4 as optional backup for future augmentation experiments. |
| 2026-03-11 | Q3: SCARF corruption rate — 40% default, sweep in Week 2 | Cell Painting features are highly correlated (clusters of 20–30 texture features), so higher corruption rates are tolerable. Default 40%, sweep 20/40/60% on morph↔expr (1 seed each, ~90 min total) during Week 2. Check feature distributions before assuming corruption rate semantics match literature. |
| 2026-03-11 | Q2: InfoAlign comparison is P2 stretch goal | InfoAlign is asymmetric (can't do morph→expr), was trained on U2OS (not A549), and only 2/6 retrieval directions are comparable. Cross-cell-line evaluation is confounded. If included, frame as "InfoAlign evaluated out-of-distribution," not a head-to-head comparison. Don't invest time until core results are solid. |
| 2026-03-11 | Q5: Pivot = per-MOA-class complementarity analysis | Break test set by MOA class to show where tri-modal helps vs. doesn't. "Tri-modal alignment provides significant gains for X mechanism classes but not Y, suggesting marginal value depends on biological redundancy between morphology and expression." If contrastive alignment fails broadly, the diagnostic analysis of why is itself a contribution. |

---

## Appendix A: Architecture Specification

### A.1 Encoder specifications

| Component | Specification | Rationale |
|-----------|--------------|-----------|
| Molecular encoder | ECFP (2048-bit, radius 2) → MLP [2048→1024→1024→1024→512] | Zero trainable params for representation; only projection needs learning |
| Morphology encoder | MLP [~1500→1024→1024→512] + SCARF augmentation | Matches CLOOME's proven architecture; SCARF creates positive pairs |
| Expression encoder | MLP [978→1024→1024→512] + SCARF augmentation | Symmetric with morphology encoder |
| Projection head | 2-layer MLP [512→512→256] per modality | Loss computed on projected features; pre-projection features for downstream |
| Embedding dim | 256 | Shared contrastive space |
| Batch normalization | After each hidden layer | Prevents internal covariate shift |

### A.2 Loss formulation

For three modalities (M = molecule, C = Cell Painting, G = gene expression):

```
L_total = L_SigLIP(M,C) + L_SigLIP(M,G) + L_SigLIP(C,G)
        + λ · (L_VICReg(M) + L_VICReg(C) + L_VICReg(G))
```

Where λ = 0.1–0.25 (tuned in Week 6). For bi-modal ablations, drop the irrelevant SigLIP and VICReg terms.

### A.3 Training hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay 1e-4 |
| Learning rate | 1e-3 | All encoders (no separate GIN LR needed) |
| Batch size | 128–256 | SigLIP is batch-size agnostic, but larger is still better for negative diversity |
| Epochs | 200 | Early stopping patience=30 on val mean R@10 |
| LR schedule | Cosine annealing with 10 epoch warmup | |
| SCARF corruption | 40% (tunable) | Replace corrupted features with empirical marginal draws |
| VICReg λ | 0.1 | Variance + covariance regularization weight |
| Gradient clipping | max_norm=1.0 | Stability |

---

## Appendix B: Dataset Specification

### B.1 Primary dataset: Rosetta LINCS (cpg0004-lincs)

| Property | Value |
|----------|-------|
| Cell line | A549 (human lung adenocarcinoma) |
| Perturbation type | Chemical (bioactive compounds) |
| Number of compounds | ~1,327 |
| Doses per compound | 6 |
| Technical replicates | 5 per condition |
| Morphology features | ~1,500 CellProfiler features (after QC) |
| Expression features | 978 L1000 landmark genes |
| Pairing | Matched — same compounds measured in both assays, same cell line |

### B.2 Data sources and download

| Source | URL | Content |
|--------|-----|---------|
| Cell Painting profiles | `s3://cellpainting-gallery/cpg0004-lincs/` | Level 4b normalized, Level 5 feature-selected, spherized/consensus profiles |
| L1000 profiles | `figshare.com/articles/dataset/13181966/2` | Pre-matched to same compounds |
| Analysis code + baselines | `github.com/broadinstitute/lincs-profiling-complementarity` | Full processing pipeline, train/test splits, MOA prediction baselines |
| LINCS Cell Painting repo | `github.com/broadinstitute/lincs-cell-painting` | Processed profiles, metadata |
| SMILES metadata | `broadinstitute.org/drug-repurposing-hub` | BRD IDs, SMILES, InChIKey, MOA annotations |
| InfoAlign data (comparison) | `huggingface.co/datasets/liuganghuggingface/InfoAlign-Data` | CP-JUMP.csv.gz + matched L1000 for fair baseline comparison |

### B.3 Data pipeline order

1. Download pre-normalized profiles from S3 + Figshare
2. Run data audit: NaN rates, feature distributions, replicate correlations
3. Replicate correlation filter (90th percentile of null distribution)
4. Aggregate replicates to treatment-level (MODZ weighted mean)
5. Match treatments across modalities + merge SMILES metadata via BRD ID → InChIKey → PubChem CID
6. Remove DMSO controls
7. Feature QC: drop features with >5% NaN, zero variance, or on Broad blocklist
8. Scaffold split (grouped by Bemis-Murcko scaffold, stratified by MOA where available): 70/15/15
9. Global normalization: RobustScaler (morph, fit on train), verify z-score + clip (expr)
10. Save to parquet

---

## Appendix C: Ablation Matrix

### C.1 Core 8-condition experiment

| ID | Configuration | What It Proves | Seeds |
|----|--------------|----------------|-------|
| B0 | Random embeddings | Absolute baseline (R@10 ≈ 8.93%) | 1 |
| B1 | Mol-only (ECFP features, no training) | Single-modality reference | 1 |
| B2 | Morph-only (CP features, no training) | Single-modality reference | 1 |
| B3 | Expr-only (L1000 features, no training) | Single-modality reference | 1 |
| B4 | Mol↔Morph (bi-modal, trained) | CLOOME-equivalent baseline | 5 |
| B5 | Mol↔Expr (bi-modal, trained) | Structure-expression alignment | 5 |
| B6 | Morph↔Expr (bi-modal, trained) | Phenotype-phenotype alignment | 5 |
| **T1** | **Mol↔Morph↔Expr (tri-modal, trained)** | **CaPy's core claim** | **5** |

Total: 40 runs. ~20 GPU-hours at ~30 min/run.

### C.2 Extended ablations (if time permits)

| Ablation | Conditions | What It Tests |
|----------|-----------|---------------|
| Loss function | SigLIP vs. DCL vs. InfoNCE (T1 config, 3 seeds each) | Best loss for small-batch multi-modal |
| Encoder choice | ECFP vs. ChemBERTa vs. pretrained GIN (T1 config, 3 seeds each) | Molecular encoder impact |
| Embedding dim | 64, 128, 256, 512 (T1 config, 3 seeds each) | Capacity vs. overfitting |
| Dataset size | 25%, 50%, 75%, 100% of data (T1 config, 3 seeds each) | Sample efficiency |
| Staged vs. joint | Pre-train morph↔expr then add mol, vs. all three jointly | Training strategy |

---

## Appendix D: Evaluation Protocol

### D.1 Cross-modal retrieval (primary metric)

For each of 6 retrieval directions (mol→morph, morph→mol, mol→expr, expr→mol, morph→expr, expr→morph):

- **R@1, R@5, R@10:** Is the correct match in the top K results?
- **MRR:** Mean reciprocal rank of the correct match
- **NDCG@10:** Normalized discounted cumulative gain (for InfoAlign comparison)

Report mean across all 6 directions as the headline metric.

### D.2 MOA clustering (secondary metric)

- Embed all test compounds using the trained model
- For each modality and the concatenated tri-modal embedding:
  - k-NN MOA accuracy (k=5,10,20) — classify by majority vote of neighbors' MOA labels
  - Adjusted Mutual Information (AMI) — clustering quality vs. ground-truth MOA
  - Adjusted Rand Index (ARI) — pairwise agreement with ground truth

### D.3 Embedding quality diagnostics

- **Alignment** (Wang & Isola 2020): mean distance between positive pairs. Lower is better. Target: < 1.5 (currently ~1.85).
- **Uniformity** (Wang & Isola 2020): log-average pairwise Gaussian potential. More negative is better. Target: < −3.0. Collapse threshold: > −0.5.
- **Per-modality cosine std:** Standard deviation of pairwise cosine similarities. Higher means more diverse embeddings. Collapse indicator: < 0.1.

### D.4 Downstream transfer (tertiary metric)

- Freeze encoder, train linear probe on activity prediction (binary classification per target)
- Report mean AUC across available targets
- Compare tri-modal vs. best bi-modal embeddings

---

## Appendix E: Key References

1. Haghighi et al. (2022). High-dimensional gene expression and morphology profiles of cells across 28,000 perturbations. *Nature Methods*.
2. Liu et al. (2025). Learning Molecular Representation in a Cell (InfoAlign). *ICLR 2025*.
3. Way et al. (2022). Morphology and gene expression profiling provide complementary information for mapping cell state. *Cell Systems*.
4. Sanchez-Fernandez et al. (2023). CLOOME: Contrastive learning unlocks bioimaging databases for queries with chemical structures. *Nature Communications*.
5. Lu et al. (2025). CellCLIP: Learning perturbation effects in Cell Painting via text-guided contrastive learning. *NeurIPS 2025*.
6. Yeh et al. (2022). Decoupled Contrastive Learning. *ECCV 2022*.
7. Zhai et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). *ICCV 2023*.
8. Bardes et al. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.
9. Bahri et al. (2022). SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption. *ICLR 2022*.
10. Wang & Isola (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *ICML 2020*.
11. Praski et al. (2025). Benchmarking Pretrained Molecular Embedding Models for Molecular Representation Learning. *arXiv:2508.06199*.
