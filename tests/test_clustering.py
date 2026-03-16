"""Tests for MOA clustering evaluation (FR-8.3)."""

import torch
import torch.nn.functional as f

from src.evaluation.clustering import compute_moa_clustering


class TestMOAClustering:
    """Tests for compute_moa_clustering (FR-8.3)."""

    def test_random_embeddings_low_ami(self):
        """Random embeddings should yield AMI near 0."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(100, 256), dim=-1)
        moa_labels = [f"moa_{i % 5}" for i in range(100)]
        metrics = compute_moa_clustering(embeddings, moa_labels)
        assert "AMI" in metrics
        assert -0.1 < metrics["AMI"] < 0.3

    def test_perfect_clusters_high_ami(self):
        """Perfectly separated clusters should yield AMI close to 1.0."""
        torch.manual_seed(42)
        n_per_class = 30
        n_classes = 4
        embeddings_list = []
        labels = []
        for i in range(n_classes):
            # Tight cluster around a distinct center
            center = torch.zeros(256)
            center[i * 64 : (i + 1) * 64] = 10.0
            cluster = center.unsqueeze(0) + torch.randn(n_per_class, 256) * 0.01
            embeddings_list.append(cluster)
            labels.extend([f"moa_{i}"] * n_per_class)

        embeddings = f.normalize(torch.cat(embeddings_list), dim=-1)
        metrics = compute_moa_clustering(embeddings, labels)
        assert metrics["AMI"] > 0.8
        assert metrics["ARI"] > 0.8

    def test_single_moa_class_skips(self):
        """Only 1 unique MOA should return empty dict."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(20, 256), dim=-1)
        moa_labels = ["same_moa"] * 20
        metrics = compute_moa_clustering(embeddings, moa_labels)
        assert metrics == {}

    def test_null_moas_filtered(self):
        """Null MOAs should be filtered out; metrics computed on remainder."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(50, 256), dim=-1)
        moa_labels: list[str | None] = [
            f"moa_{i % 3}" if i < 30 else None for i in range(50)
        ]
        metrics = compute_moa_clustering(embeddings, moa_labels)
        assert "AMI" in metrics
        assert "ARI" in metrics
        assert "kNN_5_acc" in metrics

    def test_partial_moa_coverage(self):
        """Mix of null and valid MOAs — only valid ones used."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(40, 256), dim=-1)
        moa_labels: list[str | None] = []
        for i in range(40):
            if i % 4 == 0:
                moa_labels.append(None)
            else:
                moa_labels.append(f"moa_{i % 3}")
        metrics = compute_moa_clustering(embeddings, moa_labels)
        assert "AMI" in metrics
        assert len(metrics) > 0

    def test_knn_accuracy_keys(self):
        """Output should contain kNN_{k}_acc for all default k values."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(50, 256), dim=-1)
        moa_labels = [f"moa_{i % 5}" for i in range(50)]
        metrics = compute_moa_clustering(embeddings, moa_labels)
        assert "kNN_5_acc" in metrics
        assert "kNN_10_acc" in metrics
        assert "kNN_20_acc" in metrics
        for k in [5, 10, 20]:
            assert 0.0 <= metrics[f"kNN_{k}_acc"] <= 1.0

    def test_custom_k_values(self):
        """Custom k_values should produce matching kNN keys."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(30, 256), dim=-1)
        moa_labels = [f"moa_{i % 3}" for i in range(30)]
        metrics = compute_moa_clustering(embeddings, moa_labels, k_values=[3, 7])
        assert "kNN_3_acc" in metrics
        assert "kNN_7_acc" in metrics
        assert "kNN_5_acc" not in metrics

    def test_k_capped_at_n_minus_1(self):
        """k > n_samples should not crash (capped internally)."""
        torch.manual_seed(42)
        embeddings = f.normalize(torch.randn(8, 256), dim=-1)
        moa_labels = [f"moa_{i % 2}" for i in range(8)]
        # k=100 >> n_samples=8
        metrics = compute_moa_clustering(embeddings, moa_labels, k_values=[100])
        assert "kNN_100_acc" in metrics
        assert 0.0 <= metrics["kNN_100_acc"] <= 1.0
