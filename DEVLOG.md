# CaPy v2 — Development Log

> Auto-maintained paper trail of progress, decisions, and changes.
> Newest entries first. See `capy_v2_prd.md` and `capy_v2_fsd.md` for authoritative specs.

---

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
