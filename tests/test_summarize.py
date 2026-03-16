"""Tests for scripts/summarize_ablations.py (FR-9.2)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scripts/ is not a package — add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_ablations import (
    compute_summary,
    generate_barplot,
    generate_csv,
    generate_latex,
    load_runs,
    run_statistical_tests,
)


def _make_run(config: str, seed: int, compound_mean_r10: float) -> dict:
    """Create a synthetic run entry for testing."""
    return {
        "config": config,
        "name": f"test_{config}",
        "model": f"model_{config}",
        "seed": seed,
        "best_epoch": 50 + seed % 10,
        "best_metric": compound_mean_r10,
        "mean_R@10": compound_mean_r10 * 0.3,
        "compound/mean_R@10": compound_mean_r10,
        "compound/random_R@10": 0.0565,
        "compound/mol->morph/R@10": compound_mean_r10 * 0.5,
        "compound/morph->mol/R@10": compound_mean_r10 * 0.5,
        "compound/mol->expr/R@10": compound_mean_r10 * 0.4,
        "compound/expr->mol/R@10": compound_mean_r10 * 0.4,
        "compound/morph->expr/R@10": compound_mean_r10 * 1.5,
        "compound/expr->morph/R@10": compound_mean_r10 * 1.4,
        "align_mol_morph": 1.2 + np.random.random() * 0.1,
        "uniform_mol": -2.0 + np.random.random() * 0.1,
        "uniform_morph": -2.0 + np.random.random() * 0.1,
        "val_loss": 0.5 + np.random.random(),
    }


def _make_synthetic_jsonl(path: Path, include_single_seed: bool = False) -> None:
    """Write synthetic JSONL data with B4, B5, B6, T1 (5 seeds each)."""
    np.random.seed(42)
    runs = []
    for config, base_r10 in [("B4", 0.12), ("B5", 0.11), ("B6", 0.73), ("T1", 0.37)]:
        for seed in [42, 123, 456, 789, 1024]:
            noise = np.random.normal(0, 0.01)
            runs.append(_make_run(config, seed, base_r10 + noise))

    if include_single_seed:
        runs.append(_make_run("B0", 42, 0.056))

    with open(path, "w") as f:
        for run in runs:
            f.write(json.dumps(run) + "\n")


@pytest.fixture
def synthetic_jsonl(tmp_path):
    """Create a synthetic JSONL file for testing."""
    path = tmp_path / "ablation_runs.jsonl"
    _make_synthetic_jsonl(path)
    return path


@pytest.fixture
def synthetic_jsonl_with_single_seed(tmp_path):
    """Create a synthetic JSONL with B0 (single seed)."""
    path = tmp_path / "ablation_runs.jsonl"
    _make_synthetic_jsonl(path, include_single_seed=True)
    return path


class TestLoadRuns:
    def test_load_runs(self, synthetic_jsonl):
        df = load_runs(synthetic_jsonl)
        assert len(df) == 20  # 4 configs × 5 seeds
        assert "config" in df.columns
        assert "compound/mean_R@10" in df.columns

    def test_load_runs_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No runs found"):
            load_runs(path)


class TestComputeSummary:
    def test_compute_summary_has_all_configs(self, synthetic_jsonl):
        df = load_runs(synthetic_jsonl)
        summary = compute_summary(df)
        configs_in_summary = summary["config"].unique()
        assert set(configs_in_summary) == {"B4", "B5", "B6", "T1"}

    def test_summary_has_mean_and_std(self, synthetic_jsonl):
        df = load_runs(synthetic_jsonl)
        summary = compute_summary(df)
        primary = summary[summary["metric"] == "compound/mean_R@10"]
        for _, row in primary.iterrows():
            assert not np.isnan(row["mean"])
            assert not np.isnan(row["std"])  # 5 seeds → std should be valid
            assert row["count"] == 5


class TestStatisticalTests:
    def test_welch_t_test_p_values_valid(self, synthetic_jsonl):
        df = load_runs(synthetic_jsonl)
        comparisons = run_statistical_tests(df)
        assert len(comparisons) == 3
        for c in comparisons:
            assert 0 <= c["p_value"] <= 1, f"p_value out of range: {c['p_value']}"

    def test_bonferroni_correction(self, synthetic_jsonl):
        df = load_runs(synthetic_jsonl)
        comparisons = run_statistical_tests(df)
        for c in comparisons:
            expected = min(c["p_value"] * 3, 1.0)
            assert abs(c["p_corrected"] - expected) < 1e-10

    def test_significance_stars(self, synthetic_jsonl):
        df = load_runs(synthetic_jsonl)
        comparisons = run_statistical_tests(df)
        for c in comparisons:
            p = c["p_value"]
            if p < 0.001:
                assert c["stars"] == "***"
            elif p < 0.01:
                assert c["stars"] == "**"
            elif p < 0.05:
                assert c["stars"] == "*"
            else:
                assert c["stars"] == ""


class TestOutputGeneration:
    def test_csv_output(self, synthetic_jsonl, tmp_path):
        df = load_runs(synthetic_jsonl)
        summary = compute_summary(df)
        csv_path = tmp_path / "summary.csv"
        generate_csv(summary, csv_path)
        assert csv_path.exists()
        result = pd.read_csv(csv_path)
        assert "config" in result.columns
        assert len(result) == 4  # B4, B5, B6, T1

    def test_latex_output(self, synthetic_jsonl, tmp_path):
        df = load_runs(synthetic_jsonl)
        comparisons = run_statistical_tests(df)
        tex_path = tmp_path / "comparison.tex"
        generate_latex(df, comparisons, tex_path)
        assert tex_path.exists()
        content = tex_path.read_text()
        assert len(content) > 0
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert "T1" in content

    def test_barplot_output(self, synthetic_jsonl, tmp_path):
        df = load_runs(synthetic_jsonl)
        png_path = tmp_path / "barplot.png"
        generate_barplot(df, png_path)
        assert png_path.exists()
        assert png_path.stat().st_size > 0


class TestEdgeCases:
    def test_single_seed_config(self, synthetic_jsonl_with_single_seed):
        """B0 has 1 seed → std should be NaN, handled gracefully."""
        df = load_runs(synthetic_jsonl_with_single_seed)
        summary = compute_summary(df)

        b0_primary = summary[
            (summary["config"] == "B0") & (summary["metric"] == "compound/mean_R@10")
        ]
        assert len(b0_primary) == 1
        assert np.isnan(b0_primary.iloc[0]["std"])
        assert b0_primary.iloc[0]["count"] == 1

        # CSV generation should not crash
        csv_path = Path(tempfile.mktemp(suffix=".csv"))
        try:
            generate_csv(summary, csv_path)
            result = pd.read_csv(csv_path)
            assert len(result) == 5  # B0, B4, B5, B6, T1
        finally:
            csv_path.unlink(missing_ok=True)

        # Barplot should not crash
        png_path = Path(tempfile.mktemp(suffix=".png"))
        try:
            generate_barplot(df, png_path)
            assert png_path.exists()
        finally:
            png_path.unlink(missing_ok=True)
