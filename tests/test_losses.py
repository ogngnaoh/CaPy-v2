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

    def test_temperature_is_learnable(self, siglip_loss):
        """Temperature should be a learnable parameter."""
        assert hasattr(siglip_loss, "log_temperature")
        assert siglip_loss.log_temperature.requires_grad is True

    def test_gradient_flows_through_bias(self, siglip_loss):
        """Bias and temperature should receive gradients after backward."""
        z_a = f.normalize(torch.randn(8, 256), dim=-1)
        z_b = f.normalize(torch.randn(8, 256), dim=-1)
        loss = siglip_loss(z_a, z_b)
        loss.backward()
        assert siglip_loss.bias.grad is not None
        assert siglip_loss.log_temperature.grad is not None

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

    def test_multi_positive_reduces_loss_for_duplicates(self, siglip_loss):
        """Compound-aware multi-positive should give lower loss when same-compound
        pairs have identical embeddings (no longer penalized as negatives)."""
        torch.manual_seed(42)
        # 8 samples: 4 unique compounds, each appearing twice
        z_base = f.normalize(torch.randn(4, 256), dim=-1)
        z_a = z_base.repeat(2, 1)  # [8, 256] — pairs (0,4), (1,5), etc. are identical
        z_b = z_base.repeat(2, 1)
        compound_ids = ["A", "B", "C", "D", "A", "B", "C", "D"]

        # Without compound_ids: identical duplicates treated as negatives → higher loss
        loss_no_ids = siglip_loss(z_a, z_b).item()
        # With compound_ids: identical duplicates treated as positives → lower loss
        loss_with_ids = siglip_loss(z_a, z_b, compound_ids=compound_ids).item()
        assert loss_with_ids < loss_no_ids

    def test_multi_positive_unique_ids_matches_default(self, siglip_loss):
        """When all compound_ids are unique, multi-positive matches default behavior."""
        torch.manual_seed(42)
        z_a = f.normalize(torch.randn(8, 256), dim=-1)
        z_b = f.normalize(torch.randn(8, 256), dim=-1)
        unique_ids = [f"compound_{i}" for i in range(8)]

        loss_default = siglip_loss(z_a, z_b).item()
        loss_unique = siglip_loss(z_a, z_b, compound_ids=unique_ids).item()
        assert abs(loss_default - loss_unique) < 1e-5


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

    def _make_encoder_outputs(self, modalities, batch_size=16, dim=512):
        """Helper: create random encoder outputs (pre-projection)."""
        return {m: torch.randn(batch_size, dim) for m in modalities}

    def test_tri_modal_loss_keys(self, siglip_loss, vicreg_loss):
        """T1 config should produce all 7 expected keys."""
        mods = ["mol", "morph", "expr"]
        embeddings = self._make_embeddings(mods)
        enc_outputs = self._make_encoder_outputs(mods)
        _, loss_dict = compute_total_loss(
            embeddings, siglip_loss, vicreg_loss, encoder_outputs=enc_outputs
        )
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
        mods = ["mol", "morph"]
        embeddings = self._make_embeddings(mods)
        enc_outputs = self._make_encoder_outputs(mods)
        _, loss_dict = compute_total_loss(
            embeddings, siglip_loss, vicreg_loss, encoder_outputs=enc_outputs
        )
        expected_keys = {
            "loss_mol_morph",
            "vicreg_mol",
            "vicreg_morph",
            "loss_total",
        }
        assert set(loss_dict.keys()) == expected_keys

    def test_bi_mol_expr_loss_keys(self, siglip_loss, vicreg_loss):
        """B5 config should produce mol+expr keys."""
        mods = ["mol", "expr"]
        embeddings = self._make_embeddings(mods)
        enc_outputs = self._make_encoder_outputs(mods)
        _, loss_dict = compute_total_loss(
            embeddings, siglip_loss, vicreg_loss, encoder_outputs=enc_outputs
        )
        expected_keys = {
            "loss_mol_expr",
            "vicreg_mol",
            "vicreg_expr",
            "loss_total",
        }
        assert set(loss_dict.keys()) == expected_keys

    def test_total_loss_is_differentiable(self, siglip_loss, vicreg_loss):
        """Total loss tensor should support backward()."""
        mods = ["mol", "morph", "expr"]
        embeddings = self._make_embeddings(mods)
        enc_outputs = self._make_encoder_outputs(mods)
        total_loss, _ = compute_total_loss(
            embeddings, siglip_loss, vicreg_loss, encoder_outputs=enc_outputs
        )
        assert total_loss.requires_grad is True
        total_loss.backward()
        assert siglip_loss.bias.grad is not None

    def test_loss_dict_values_are_floats(self, siglip_loss, vicreg_loss):
        """All dict values should be Python floats, not tensors."""
        mods = ["mol", "morph"]
        embeddings = self._make_embeddings(mods)
        enc_outputs = self._make_encoder_outputs(mods)
        _, loss_dict = compute_total_loss(
            embeddings, siglip_loss, vicreg_loss, encoder_outputs=enc_outputs
        )
        for key, val in loss_dict.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_components_sum_to_total(self, siglip_loss, vicreg_loss):
        """Sum of individual components should equal loss_total."""
        mods = ["mol", "morph", "expr"]
        embeddings = self._make_embeddings(mods)
        enc_outputs = self._make_encoder_outputs(mods)
        _, loss_dict = compute_total_loss(
            embeddings, siglip_loss, vicreg_loss, encoder_outputs=enc_outputs
        )
        component_sum = sum(v for k, v in loss_dict.items() if k != "loss_total")
        assert abs(component_sum - loss_dict["loss_total"]) < 1e-4

    def test_vicreg_on_encoder_outputs_not_embeddings(self, vicreg_loss):
        """VICReg variance term is satisfiable for unnormalized encoder
        outputs but saturated for L2-normalized embeddings."""
        torch.manual_seed(42)
        # Large batch to minimize covariance noise
        n = 1024
        # L2-normalized: per-dim std ≈ 1/sqrt(256) ≈ 0.06 → hinge ≈ 0.94
        z_normed = f.normalize(torch.randn(n, 256), dim=-1)
        loss_normed = vicreg_loss(z_normed).item()
        # Unnormalized randn: per-dim std ≈ 1.0 → variance hinge ≈ 0
        z_raw = torch.randn(n, 256)
        loss_raw = vicreg_loss(z_raw).item()
        # Unnormalized should have much lower loss (variance term near 0)
        assert loss_raw < loss_normed
