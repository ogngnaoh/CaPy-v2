"""Tests for reproducibility seeding (FR-10.2)."""

import random

import numpy as np
import torch

from src.utils.seeding import seed_everything


class TestSeedEverything:
    """Tests for seed_everything."""

    def test_deterministic_python_random(self):
        """Python random produces same output after reseeding."""
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_deterministic_numpy(self):
        """NumPy produces same output after reseeding."""
        seed_everything(42)
        a = np.random.randn(10)
        seed_everything(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        """PyTorch produces same output after reseeding."""
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds produce different output."""
        seed_everything(42)
        a = np.random.randn(10)
        seed_everything(99)
        b = np.random.randn(10)
        assert not np.array_equal(a, b)

    def test_cuda_branch(self, monkeypatch):
        """CUDA seeding branch is exercised when cuda is available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # Mock cuda-specific calls
        called = {}
        monkeypatch.setattr(
            torch.cuda, "manual_seed_all", lambda s: called.update({"seed": s})
        )
        seed_everything(42)
        assert called.get("seed") == 42
        assert torch.backends.cudnn.deterministic is True
