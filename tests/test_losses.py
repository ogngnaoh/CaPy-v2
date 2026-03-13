"""Tests for loss functions (FR-6)."""

import pytest
import torch
import torch.nn.functional as f

from src.models.losses import SigLIPLoss, VICRegLoss, compute_total_loss


@pytest.fixture
def siglip_loss():
    """SigLIPLoss instance with default bias_init=0."""
    return SigLIPLoss()


@pytest.fixture
def vicreg_loss():
    """VICRegLoss instance."""
    return VICRegLoss()


# ── SigLIP tests (FR-6.1) ───────────────────────────────────


class TestSigLIPLoss:
    """Tests for SigLIPLoss (FR-6.1)."""

    def test_identical_embeddings_low_loss(self, siglip_loss):
        """For z_a = z_b (perfect alignment), loss should be lower than random baseline."""
        torch.manual_seed(42)
        z = f.normalize(torch.randn(32, 256), dim=-1)
        loss_identical = siglip_loss(z, z).item()
        # Random independent embeddings as baseline
        z_rand = f.normalize(torch.randn(32, 256), dim=-1)
        loss_random = siglip_loss(z, z_rand).item()
        assert loss_identical < loss_random

    def test_random_embeddings_baseline_loss(self, siglip_loss):
        """For random independent embeddings, loss should be near log(2) ≈ 0.693."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(64, 256), dim=-1)
        z_b = f.normalize(torch.randn(64, 256), dim=-1)
        loss = siglip_loss(z_a, z_b).item()
        # All cosine sims ≈ 0 in high dim → -logsigmoid(0) = log(2)
        assert 0.55 < loss < 0.80

    def test_output_is_scalar(self, siglip_loss):
        """Loss should return a scalar tensor."""
        z_a = f.normalize(torch.randn(16, 256), dim=-1)
        z_b = f.normalize(torch.randn(16, 256), dim=-1)
        loss = siglip_loss(z_a, z_b)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    def test_bias_is_learnable(self, siglip_loss):
        """Bias should be a learnable parameter initialized to 0."""
        assert hasattr(siglip_loss, "bias")
        assert siglip_loss.bias.requires_grad is True
        assert siglip_loss.bias.item() == 0.0

    def test_gradient_flows_through_bias(self, siglip_loss):
        """Bias should receive gradients after backward."""
        z_a = f.normalize(torch.randn(8, 256), dim=-1)
        z_b = f.normalize(torch.randn(8, 256), dim=-1)
        loss = siglip_loss(z_a, z_b)
        loss.backward()
        assert siglip_loss.bias.grad is not None

    def test_gradient_flows_through_inputs(self, siglip_loss):
        """Gradients should flow to input embeddings."""
        z_a = f.normalize(torch.randn(8, 256), dim=-1).requires_grad_(True)
        z_b = f.normalize(torch.randn(8, 256), dim=-1).requires_grad_(True)
        loss = siglip_loss(z_a, z_b)
        loss.backward()
        assert z_a.grad is not None
        assert z_b.grad is not None

    def test_symmetric_pairs(self, siglip_loss):
        """SigLIP(z_a, z_b) should equal SigLIP(z_b, z_a)."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(16, 256), dim=-1)
        z_b = f.normalize(torch.randn(16, 256), dim=-1)
        loss_ab = siglip_loss(z_a, z_b)
        loss_ba = siglip_loss(z_b, z_a)
        assert torch.allclose(loss_ab, loss_ba, atol=1e-6)


# ── VICReg tests (FR-6.2) ───────────────────────────────────


class TestVICRegLoss:
    """Tests for VICRegLoss (FR-6.2)."""

    def test_collapsed_embeddings_high_variance_loss(self, vicreg_loss):
        """All-identical embeddings → variance loss > 0."""
        z = torch.ones(32, 256)
        loss = vicreg_loss(z)
        # std=0 for all dims → variance_loss = mean(relu(1 - 0)) = 1.0
        # cov of identical rows = 0 → cov_loss = 0
        assert loss.item() > 0.9

    def test_well_distributed_low_variance_loss(self, vicreg_loss):
        """Well-distributed embeddings → variance loss near 0."""
        torch.manual_seed(42)
        # Large batch to reduce covariance sampling noise
        z = torch.randn(1024, 256)
        loss = vicreg_loss(z)
        # Variance hinge satisfied (std ≈ 1), covariance noise small at n=1024
        assert loss.item() < 0.5

    def test_output_is_scalar(self, vicreg_loss):
        """Loss should return a scalar tensor."""
        z = torch.randn(32, 256)
        loss = vicreg_loss(z)
        assert loss.shape == ()

    def test_gradient_flows(self, vicreg_loss):
        """Gradients should flow to input embeddings."""
        z = torch.randn(32, 256, requires_grad=True)
        loss = vicreg_loss(z)
        loss.backward()
        assert z.grad is not None

    def test_covariance_penalty_on_correlated_features(self, vicreg_loss):
        """Correlated features should produce higher loss than uncorrelated."""
        torch.manual_seed(42)
        # Uncorrelated (large batch to reduce noise)
        z_uncorr = torch.randn(512, 256)
        loss_uncorr = vicreg_loss(z_uncorr).item()
        # Correlated: copy col 0 into many cols to create strong signal
        z_corr = torch.randn(512, 256)
        for i in range(1, 32):
            z_corr[:, i] = z_corr[:, 0]
        loss_corr = vicreg_loss(z_corr).item()
        assert loss_corr > loss_uncorr


# ── Total loss tests (FR-6.3) ───────────────────────────────


class TestTotalLoss:
    """Tests for compute_total_loss (FR-6.3)."""

    def _make_embeddings(self, modalities, batch_size=16, dim=256):
        """Helper: create random L2-normalized embeddings."""
        return {
            m: f.normalize(torch.randn(batch_size, dim), dim=-1) for m in modalities
        }

    def test_tri_modal_loss_keys(self, siglip_loss, vicreg_loss):
        """T1 config should produce all 7 expected keys."""
        embeddings = self._make_embeddings(["mol", "morph", "expr"])
        _, loss_dict = compute_total_loss(embeddings, siglip_loss, vicreg_loss)
        expected_keys = {
            "loss_mol_morph",
            "loss_mol_expr",
            "loss_morph_expr",
            "vicreg_mol",
            "vicreg_morph",
            "vicreg_expr",
            "loss_total",
        }
        assert set(loss_dict.keys()) == expected_keys

    def test_bi_modal_loss_keys(self, siglip_loss, vicreg_loss):
        """B4 config should only produce mol+morph keys."""
        embeddings = self._make_embeddings(["mol", "morph"])
        _, loss_dict = compute_total_loss(embeddings, siglip_loss, vicreg_loss)
        expected_keys = {
            "loss_mol_morph",
            "vicreg_mol",
            "vicreg_morph",
            "loss_total",
        }
        assert set(loss_dict.keys()) == expected_keys

    def test_bi_mol_expr_loss_keys(self, siglip_loss, vicreg_loss):
        """B5 config should produce mol+expr keys."""
        embeddings = self._make_embeddings(["mol", "expr"])
        _, loss_dict = compute_total_loss(embeddings, siglip_loss, vicreg_loss)
        expected_keys = {
            "loss_mol_expr",
            "vicreg_mol",
            "vicreg_expr",
            "loss_total",
        }
        assert set(loss_dict.keys()) == expected_keys

    def test_total_loss_is_differentiable(self, siglip_loss, vicreg_loss):
        """Total loss tensor should support backward()."""
        embeddings = self._make_embeddings(["mol", "morph", "expr"])
        total_loss, _ = compute_total_loss(embeddings, siglip_loss, vicreg_loss)
        assert total_loss.requires_grad is True
        total_loss.backward()
        assert siglip_loss.bias.grad is not None

    def test_loss_dict_values_are_floats(self, siglip_loss, vicreg_loss):
        """All dict values should be Python floats, not tensors."""
        embeddings = self._make_embeddings(["mol", "morph"])
        _, loss_dict = compute_total_loss(embeddings, siglip_loss, vicreg_loss)
        for key, val in loss_dict.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_components_sum_to_total(self, siglip_loss, vicreg_loss):
        """Sum of individual components should equal loss_total."""
        embeddings = self._make_embeddings(["mol", "morph", "expr"])
        _, loss_dict = compute_total_loss(embeddings, siglip_loss, vicreg_loss)
        component_sum = sum(v for k, v in loss_dict.items() if k != "loss_total")
        assert abs(component_sum - loss_dict["loss_total"]) < 1e-4
