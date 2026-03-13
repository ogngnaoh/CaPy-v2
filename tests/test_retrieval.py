"""Tests for evaluation metrics (FR-8.1, FR-8.2)."""

import itertools

import pytest
import torch
import torch.nn.functional as f

from src.evaluation.diagnostics import compute_alignment, compute_uniformity
from src.evaluation.retrieval import (
    compute_all_retrieval_metrics,
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
