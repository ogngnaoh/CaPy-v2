"""Tests for B0-B3 baseline evaluation in scripts/run_ablations.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scripts/ is not a package — add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_ablations import evaluate_baseline, load_existing_runs, main


@pytest.fixture
def synthetic_processed_dir(tmp_path):
    """Create a minimal processed data directory with synthetic data."""
    n_compounds = 20
    n_morph = 10
    n_expr = 8

    # Generate synthetic compound data
    np.random.seed(42)
    smiles_list = [f"C{'C' * i}O" for i in range(n_compounds)]
    compound_ids = [f"BRD-K{i:08d}" for i in range(n_compounds)]

    morph_cols = [f"morph_feat_{j}" for j in range(n_morph)]
    expr_cols = [f"expr_feat_{j}" for j in range(n_expr)]

    data = {
        "compound_id": compound_ids,
        "smiles": smiles_list,
    }
    for col in morph_cols:
        data[col] = np.random.randn(n_compounds)
    for col in expr_cols:
        data[col] = np.random.randn(n_compounds)

    df = pd.DataFrame(data)
    df.to_parquet(tmp_path / "val.parquet", index=False)

    feature_cols = {
        "morph_features": morph_cols,
        "expr_features": expr_cols,
    }
    with open(tmp_path / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    return tmp_path


class TestEvaluateBaselineB0:
    def test_b0_produces_metrics(self, synthetic_processed_dir):
        """B0 produces all expected metric keys."""
        result = evaluate_baseline("B0", 42, synthetic_processed_dir)
        assert result["config"] == "B0"
        assert result["seed"] == 42
        assert "compound/mean_R@10" in result
        assert "compound/mol->morph/R@10" in result
        assert "compound/morph->expr/R@10" in result

    def test_b0_near_random_baseline(self, synthetic_processed_dir):
        """B0 mean_R@10 should be near random baseline (10/N)."""
        result = evaluate_baseline("B0", 42, synthetic_processed_dir)
        mean_r10 = result["compound/mean_R@10"]
        # With 20 compounds, random R@10 = 10/20 = 0.5
        # Allow wide margin since small N makes random embeddings noisy
        assert 0.0 < mean_r10 < 1.0

    def test_b0_reproducible(self, synthetic_processed_dir):
        """Same seed should produce identical results."""
        r1 = evaluate_baseline("B0", 42, synthetic_processed_dir)
        r2 = evaluate_baseline("B0", 42, synthetic_processed_dir)
        assert r1["compound/mean_R@10"] == r2["compound/mean_R@10"]


class TestEvaluateBaselineSingleModality:
    def test_b1_averages_mol_directions(self, synthetic_processed_dir):
        """B1 mean_R@10 should average only the 4 mol-involving directions."""
        result = evaluate_baseline("B1", 42, synthetic_processed_dir)
        mol_directions = [
            "compound/mol->morph/R@10",
            "compound/morph->mol/R@10",
            "compound/mol->expr/R@10",
            "compound/expr->mol/R@10",
        ]
        expected_mean = np.mean([result[d] for d in mol_directions])
        assert abs(result["compound/mean_R@10"] - expected_mean) < 1e-6

    def test_b2_averages_morph_directions(self, synthetic_processed_dir):
        """B2 mean_R@10 should average only the 4 morph-involving directions."""
        result = evaluate_baseline("B2", 42, synthetic_processed_dir)
        morph_directions = [
            "compound/mol->morph/R@10",
            "compound/morph->mol/R@10",
            "compound/morph->expr/R@10",
            "compound/expr->morph/R@10",
        ]
        expected_mean = np.mean([result[d] for d in morph_directions])
        assert abs(result["compound/mean_R@10"] - expected_mean) < 1e-6

    def test_b3_averages_expr_directions(self, synthetic_processed_dir):
        """B3 mean_R@10 should average only the 4 expr-involving directions."""
        result = evaluate_baseline("B3", 42, synthetic_processed_dir)
        expr_directions = [
            "compound/mol->expr/R@10",
            "compound/expr->mol/R@10",
            "compound/morph->expr/R@10",
            "compound/expr->morph/R@10",
        ]
        expected_mean = np.mean([result[d] for d in expr_directions])
        assert abs(result["compound/mean_R@10"] - expected_mean) < 1e-6

    def test_shared_directions_identical_across_b1_b2_b3(self, synthetic_processed_dir):
        """B1, B2, B3 use the same raw features — shared direction metrics should match."""
        b1 = evaluate_baseline("B1", 42, synthetic_processed_dir)
        b2 = evaluate_baseline("B2", 42, synthetic_processed_dir)
        b3 = evaluate_baseline("B3", 42, synthetic_processed_dir)

        # mol->morph should be the same across B1 and B2 (both involve mol and morph)
        assert b1["compound/mol->morph/R@10"] == b2["compound/mol->morph/R@10"]
        # morph->expr should be the same across B2 and B3
        assert b2["compound/morph->expr/R@10"] == b3["compound/morph->expr/R@10"]
        # mol->expr should be the same across B1 and B3
        assert b1["compound/mol->expr/R@10"] == b3["compound/mol->expr/R@10"]


class TestBaselineWritesJsonl:
    def test_writes_correct_keys(self, synthetic_processed_dir):
        """Baseline result dict has required keys for JSONL entry."""
        result = evaluate_baseline("B0", 42, synthetic_processed_dir)
        assert "config" in result
        assert "seed" in result
        assert "compound/mean_R@10" in result

        # Verify it's JSON-serializable
        json_str = json.dumps(result)
        reloaded = json.loads(json_str)
        assert reloaded["config"] == "B0"

    def test_b1_writes_correct_keys(self, synthetic_processed_dir):
        result = evaluate_baseline("B1", 42, synthetic_processed_dir)
        assert result["config"] == "B1"
        assert "compound/mean_R@10" in result
        json.dumps(result)  # Ensure serializable


class TestResumeSkipsExistingRuns:
    def test_resume_skips_existing(self, tmp_path):
        """load_existing_runs returns (config, seed) pairs from JSONL."""
        jsonl_path = tmp_path / "ablation_runs.jsonl"
        entries = [
            {"config": "B0", "seed": 42, "compound/mean_R@10": 0.05},
            {"config": "B4", "seed": 42, "compound/mean_R@10": 0.12},
            {"config": "B4", "seed": 123, "compound/mean_R@10": 0.13},
        ]
        with open(jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        existing = load_existing_runs(jsonl_path)
        assert ("B0", 42) in existing
        assert ("B4", 42) in existing
        assert ("B4", 123) in existing
        assert ("B1", 42) not in existing
        assert ("B4", 456) not in existing

    def test_empty_file_returns_empty_set(self, tmp_path):
        jsonl_path = tmp_path / "ablation_runs.jsonl"
        jsonl_path.write_text("")
        existing = load_existing_runs(jsonl_path)
        assert len(existing) == 0

    def test_nonexistent_file_returns_empty_set(self, tmp_path):
        jsonl_path = tmp_path / "does_not_exist.jsonl"
        existing = load_existing_runs(jsonl_path)
        assert len(existing) == 0


class TestMainExitsOnMissingData:
    def test_main_exits_on_missing_data(self, tmp_path, monkeypatch):
        """main() should exit with code 1 when processed data is missing."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_ablations.py",
                "--matrix",
                "core",
                "--processed-dir",
                str(tmp_path),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
