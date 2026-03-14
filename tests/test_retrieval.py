"""Tests for evaluation metrics (FR-8.1, FR-8.2)."""

import itertools

import pytest
import torch
import torch.nn.functional as f

from src.evaluation.diagnostics import compute_alignment, compute_uniformity
from src.evaluation.retrieval import (
    compute_all_compound_retrieval_metrics,
    compute_all_retrieval_metrics,
    compute_compound_retrieval_metrics,
    compute_retrieval_metrics,
)

# ── Retrieval tests (FR-8.1) ────────────────────────────────


class TestRetrievalMetrics:
    """Tests for compute_retrieval_metrics (FR-8.1)."""

    def test_identical_embeddings_perfect_recall(self):
        """For z_a = z_b, R@1 should be 1.0."""
        torch.manual_seed(42)
        z = f.normalize(torch.randn(32, 256), dim=-1)
        metrics = compute_retrieval_metrics(z, z)
        assert metrics["R@1"] == pytest.approx(1.0)
        assert metrics["R@5"] == pytest.approx(1.0)
        assert metrics["R@10"] == pytest.approx(1.0)
        assert metrics["MRR"] == pytest.approx(1.0)

    def test_random_embeddings_baseline(self):
        """For random embeddings with N=100, R@10 should be ~0.10."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(100, 256), dim=-1)
        z_b = f.normalize(torch.randn(100, 256), dim=-1)
        metrics = compute_retrieval_metrics(z_a, z_b)
        assert 0.03 < metrics["R@10"] < 0.20

    def test_mrr_range(self):
        """MRR should be in [0, 1]."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(50, 256), dim=-1)
        z_b = f.normalize(torch.randn(50, 256), dim=-1)
        metrics = compute_retrieval_metrics(z_a, z_b)
        assert 0.0 <= metrics["MRR"] <= 1.0

    def test_all_directions_computed(self):
        """compute_all_retrieval_metrics should produce metrics for all 6 directions."""
        torch.manual_seed(42)
        z = f.normalize(torch.randn(16, 256), dim=-1)
        embeddings = {"mol": z, "morph": z.clone(), "expr": z.clone()}
        metrics = compute_all_retrieval_metrics(embeddings)
        # 6 directions x 4 metrics + mean_R@10
        for m_a, m_b in itertools.permutations(["mol", "morph", "expr"], 2):
            assert f"{m_a}->{m_b}/R@1" in metrics
            assert f"{m_a}->{m_b}/R@10" in metrics
            assert f"{m_a}->{m_b}/MRR" in metrics
        assert "mean_R@10" in metrics

    def test_mean_r10_is_average(self):
        """mean_R@10 should equal the arithmetic mean of all direction R@10 values."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(16, 256), dim=-1)
        z_b = f.normalize(torch.randn(16, 256), dim=-1)
        metrics = compute_all_retrieval_metrics({"mol": z_a, "morph": z_b})
        r10_values = [v for k, v in metrics.items() if k.endswith("/R@10")]
        assert metrics["mean_R@10"] == pytest.approx(
            sum(r10_values) / len(r10_values)
        )

    def test_bi_modal_two_directions(self):
        """Bi-modal should produce exactly 2 directions."""
        torch.manual_seed(42)
        z = f.normalize(torch.randn(16, 256), dim=-1)
        metrics = compute_all_retrieval_metrics({"mol": z, "morph": z.clone()})
        r10_keys = [k for k in metrics if k.endswith("/R@10")]
        assert len(r10_keys) == 2


# ── Compound-level retrieval tests ────────────────────────────


class TestCompoundRetrievalMetrics:
    """Tests for compound-level retrieval (deduplication)."""

    def test_dedup_removes_dose_inflation(self):
        """Compound-level R@1 should be achievable when row-level is not."""
        torch.manual_seed(42)
        # 4 compounds, each with 3 doses = 12 rows
        n_compounds = 4
        n_doses = 3
        compound_ids = []
        for i in range(n_compounds):
            compound_ids.extend([f"BRD-{i:04d}"] * n_doses)

        # Perfect alignment: z_a[i] == z_b[i] for matched pairs
        z_base = f.normalize(torch.randn(n_compounds, 256), dim=-1)
        z_a = z_base.repeat_interleave(n_doses, dim=0)
        z_b = z_base.repeat_interleave(n_doses, dim=0)

        # Row-level: R@1 should be < 1.0 due to ties from identical doses
        row_metrics = compute_retrieval_metrics(z_a, z_b)
        assert row_metrics["R@1"] < 1.0

        # Compound-level: R@1 should be 1.0 after dedup
        compound_metrics = compute_compound_retrieval_metrics(
            z_a, z_b, compound_ids
        )
        assert compound_metrics["R@1"] == pytest.approx(1.0)

    def test_unique_ids_matches_row_level(self):
        """When all compound_ids are unique, compound-level == row-level."""
        torch.manual_seed(42)
        n = 20
        z_a = f.normalize(torch.randn(n, 256), dim=-1)
        z_b = f.normalize(torch.randn(n, 256), dim=-1)
        unique_ids = [f"BRD-{i:04d}" for i in range(n)]

        row_metrics = compute_retrieval_metrics(z_a, z_b)
        compound_metrics = compute_compound_retrieval_metrics(
            z_a, z_b, unique_ids
        )
        assert compound_metrics["R@10"] == pytest.approx(
            row_metrics["R@10"], abs=1e-6
        )

    def test_n_compounds_returned(self):
        """Metrics should include n_compounds count."""
        z = f.normalize(torch.randn(12, 256), dim=-1)
        compound_ids = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]
        metrics = compute_compound_retrieval_metrics(z, z, compound_ids)
        assert metrics["n_compounds"] == 4.0

    def test_all_compound_retrieval_keys(self):
        """compute_all_compound_retrieval_metrics should produce expected keys."""
        torch.manual_seed(42)
        z = f.normalize(torch.randn(8, 256), dim=-1)
        embeddings = {"mol": z, "morph": z.clone()}
        compound_ids = ["A", "A", "B", "B", "C", "C", "D", "D"]
        metrics = compute_all_compound_retrieval_metrics(
            embeddings, compound_ids
        )
        assert "compound/mean_R@10" in metrics
        assert "compound/random_R@10" in metrics
        assert "compound/mol->morph/R@10" in metrics
        assert "compound/morph->mol/R@10" in metrics


# ── Diagnostics tests (FR-8.2) ──────────────────────────────


class TestDiagnostics:
    """Tests for alignment and uniformity (FR-8.2)."""

    def test_alignment_identical_pairs(self):
        """For z_a = z_b, alignment should be 0."""
        z = f.normalize(torch.randn(32, 256), dim=-1)
        assert compute_alignment(z, z) == pytest.approx(0.0, abs=1e-6)

    def test_alignment_random_positive(self):
        """Alignment of random pairs should be > 0."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(32, 256), dim=-1)
        z_b = f.normalize(torch.randn(32, 256), dim=-1)
        assert compute_alignment(z_a, z_b) > 0.0

    def test_uniformity_collapsed(self):
        """All-identical embeddings -> uniformity ~ 0 (collapse)."""
        z = torch.ones(32, 256) / (256**0.5)  # constant unit-norm vectors
        assert compute_uniformity(z) > -0.5

    def test_uniformity_distributed(self):
        """Random L2-normed embeddings -> uniformity < -2."""
        torch.manual_seed(42)
        z = f.normalize(torch.randn(256, 256), dim=-1)
        assert compute_uniformity(z) < -2.0

    def test_uniformity_returns_float(self):
        """Uniformity should return a Python float."""
        z = f.normalize(torch.randn(16, 256), dim=-1)
        result = compute_uniformity(z)
        assert isinstance(result, float)

    def test_alignment_returns_float(self):
        """Alignment should return a Python float."""
        z = f.normalize(torch.randn(16, 256), dim=-1)
        result = compute_alignment(z, z)
        assert isinstance(result, float)
