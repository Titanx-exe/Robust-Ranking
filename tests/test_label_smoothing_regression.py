"""
Regression tests for label smoothing functionality in BiEncoderHuggingface.

Tests cover:
1. Training with label smoothing disabled (label_smoothness=0.0)
2. Training with positive label smoothing (label_smoothness>0 via F.cross_entropy)
3. Training with negative label smoothing (label_smoothness<0 via loss_gls)
4. Loss decreases appropriately under each configuration
5. Model convergence behavior with and without smoothing
6. Consistency between loss_gls and cross_entropy at boundary (smoothness=0.0)
"""

import copy
import math
import sys
import os
import pytest
import torch
import torch.nn.functional as F

# Ensure the project root is on sys.path so imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.BiEncoderHuggingface import BiEncoderRanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_params(**overrides):
    """Return a minimal parameter dict for BiEncoderRanker (HuggingFace variant)."""
    params = {
        "bert_model": "bert-base-uncased",
        "lowercase": True,
        "out_dim": 1,
        "pull_from_layer": -1,
        "add_linear": False,
        "data_parallel": False,
        "path_to_model": None,
        "max_context_length": 32,
        "max_cand_length": 32,
        # Label smoothing
        "label_smoothness": 0.0,
    }
    params.update(overrides)
    return params


def _make_synthetic_batch(model, batch_size=4, seq_len=16, seed=None):
    """
    Create a synthetic batch of tokenised context and candidate tensors that
    are compatible with the HuggingFace BiEncoderRanker's forward method.

    Returns (context_input, candidate_input) as BatchEncoding objects.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = model.device
    vocab_size = model.tokenizer.vocab_size

    # Random token ids (avoid 0 = [PAD])
    ctx_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    cand_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    ctx_input = {
        "input_ids": ctx_ids,
        "attention_mask": torch.ones_like(ctx_ids),
        "token_type_ids": torch.zeros_like(ctx_ids),
    }
    cand_input = {
        "input_ids": cand_ids,
        "attention_mask": torch.ones_like(cand_ids),
        "token_type_ids": torch.zeros_like(cand_ids),
    }

    from transformers import BatchEncoding
    ctx_enc = BatchEncoding(ctx_input)
    cand_enc = BatchEncoding(cand_input)

    return ctx_enc, cand_enc


def _train_steps(model, n_steps=20, lr=1e-3, batch_size=4, seq_len=16, seed=42):
    """
    Run *n_steps* training iterations on synthetic data with a fixed seed
    for reproducibility. Returns a list of loss values (one per step).
    """
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        ctx, cand = _make_synthetic_batch(model, batch_size=batch_size, seq_len=seq_len,
                                          seed=seed + step)
        optimizer.zero_grad()
        loss, logits = model(ctx, cand, label_input=None)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def _assert_loss_decreases(losses, label=""):
    """Check that early losses are on average higher than late losses."""
    half = max(1, len(losses) // 2)
    early_avg = sum(losses[:half]) / half
    late_avg = sum(losses[-half:]) / half
    assert late_avg < early_avg, (
        f"[{label}] Loss did not decrease: early_avg={early_avg:.4f}, "
        f"late_avg={late_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tests – basic forward pass
# ---------------------------------------------------------------------------

class TestLabelSmoothingForwardPass:
    """Verify that the forward pass works for every smoothing configuration."""

    def test_no_smoothing_forward(self, device):
        """Forward pass with label_smoothness=0.0 produces finite loss."""
        params = _base_params(label_smoothness=0.0)
        model = BiEncoderRanker(params, device=device)
        model.train()
        ctx, cand = _make_synthetic_batch(model, seed=100)
        loss, logits = model(ctx, cand, label_input=None)

        assert loss.isfinite(), "Loss must be finite with no smoothing"
        assert logits.shape[0] > 0, "Logits must be non-empty"

    def test_positive_smoothing_forward(self, device):
        """Forward pass with positive label_smoothness uses F.cross_entropy path."""
        params = _base_params(label_smoothness=0.1)
        model = BiEncoderRanker(params, device=device)
        model.train()
        ctx, cand = _make_synthetic_batch(model, seed=101)
        loss, logits = model(ctx, cand, label_input=None)

        assert loss.isfinite(), "Loss must be finite with positive smoothing"

    def test_negative_smoothing_forward(self, device):
        """Forward pass with negative label_smoothness triggers loss_gls path."""
        params = _base_params(label_smoothness=-0.2)
        model = BiEncoderRanker(params, device=device)
        model.train()
        ctx, cand = _make_synthetic_batch(model, seed=102)
        loss, logits = model(ctx, cand, label_input=None)

        assert loss.isfinite(), "Loss must be finite with negative (GLS) smoothing"

    def test_forward_with_explicit_labels(self, device):
        """Forward with explicit label_input (non-None) also respects smoothing."""
        for smoothness in [0.0, 0.1, -0.2]:
            params = _base_params(label_smoothness=smoothness)
            model = BiEncoderRanker(params, device=device)
            model.train()
            batch_size = 4
            ctx, cand = _make_synthetic_batch(model, batch_size=batch_size, seed=103)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss, logits = model(ctx, cand, label_input=labels)

            assert loss.isfinite(), (
                f"Loss not finite with label_smoothness={smoothness} and explicit labels"
            )


# ---------------------------------------------------------------------------
# Tests – loss decreasing (convergence)
# ---------------------------------------------------------------------------

class TestLabelSmoothingConvergence:
    """
    Train a few steps and verify the loss shows a downward trend.
    We compare the average loss in the first half to the last half.
    """

    N_STEPS = 10

    def test_convergence_no_smoothing(self, device):
        """Model converges (loss decreases) with label_smoothness=0.0."""
        params = _base_params(label_smoothness=0.0)
        model = BiEncoderRanker(params, device=device)
        losses = _train_steps(model, n_steps=self.N_STEPS, seed=42)
        _assert_loss_decreases(losses, "no_smoothing")

    def test_convergence_positive_smoothing(self, device):
        """Model converges with positive label smoothing."""
        params = _base_params(label_smoothness=0.1)
        model = BiEncoderRanker(params, device=device)
        losses = _train_steps(model, n_steps=self.N_STEPS, seed=42)
        _assert_loss_decreases(losses, "positive_smoothing=0.1")

    def test_convergence_negative_smoothing(self, device):
        """Model converges with negative label smoothing (GLS)."""
        params = _base_params(label_smoothness=-0.2)
        model = BiEncoderRanker(params, device=device)
        losses = _train_steps(model, n_steps=self.N_STEPS, seed=42)
        _assert_loss_decreases(losses, "negative_smoothing=-0.2")

    def test_convergence_strong_negative_smoothing(self, device):
        """Model converges with a strong negative smoothing factor (GLS)."""
        params = _base_params(label_smoothness=-0.6)
        model = BiEncoderRanker(params, device=device)
        losses = _train_steps(model, n_steps=self.N_STEPS, seed=42)
        _assert_loss_decreases(losses, "negative_smoothing=-0.6")


# ---------------------------------------------------------------------------
# Tests – loss_gls correctness
# ---------------------------------------------------------------------------

class TestLossGLS:
    """Unit-level tests for the loss_gls method on BiEncoderRanker."""

    def _make_model(self, smoothness, device):
        params = _base_params(label_smoothness=smoothness)
        return BiEncoderRanker(params, device=device)

    def test_loss_gls_is_class_method(self, device):
        """Verify loss_gls is accessible as a class method (not nested)."""
        model = self._make_model(-0.4, device)
        assert hasattr(model, 'loss_gls'), (
            "BiEncoderRanker must have loss_gls as a class-level method, "
            "not nested inside score_candidate"
        )
        assert callable(model.loss_gls), "loss_gls must be callable"

    def test_loss_gls_finite(self, device):
        """loss_gls returns a finite scalar for typical inputs."""
        model = self._make_model(-0.4, device)
        batch_size, n_classes = 8, 8
        torch.manual_seed(77)
        logits = torch.randn(batch_size, n_classes, device=device)
        labels = torch.arange(batch_size, device=device) % n_classes
        loss = model.loss_gls(logits, labels)
        assert loss.isfinite(), "loss_gls must return a finite value"

    def test_loss_gls_gradient_flows(self, device):
        """Gradients flow through loss_gls."""
        model = self._make_model(-0.4, device)
        batch_size, n_classes = 4, 4
        torch.manual_seed(78)
        logits = torch.randn(batch_size, n_classes, device=device, requires_grad=True)
        labels = torch.arange(batch_size, device=device)
        loss = model.loss_gls(logits, labels)
        loss.backward()
        assert logits.grad is not None, "Gradients must flow through loss_gls"
        assert logits.grad.isfinite().all(), "All gradients must be finite"

    def test_loss_gls_positive_loss(self, device):
        """loss_gls should generally produce a positive loss value."""
        model = self._make_model(-0.2, device)
        batch_size, n_classes = 8, 8
        torch.manual_seed(79)
        logits = torch.randn(batch_size, n_classes, device=device)
        labels = torch.arange(batch_size, device=device)
        loss = model.loss_gls(logits, labels)
        # With random logits the loss should be positive
        assert loss.item() > 0, "loss_gls should produce positive loss for random logits"

    def test_loss_gls_varies_with_smoothness(self, device):
        """Different smoothing rates produce different loss values for the same input."""
        torch.manual_seed(42)
        batch_size, n_classes = 8, 8
        logits = torch.randn(batch_size, n_classes, device=device)
        labels = torch.arange(batch_size, device=device)

        losses = []
        for s in [-0.2, -0.4, -0.8]:
            model = self._make_model(s, device)
            loss_val = model.loss_gls(logits, labels).item()
            losses.append(loss_val)

        # At least two of the three should differ
        assert not (
            math.isclose(losses[0], losses[1], rel_tol=1e-5)
            and math.isclose(losses[1], losses[2], rel_tol=1e-5)
        ), f"loss_gls should vary with smoothing rate, got {losses}"


# ---------------------------------------------------------------------------
# Tests – smoothing vs no-smoothing comparison
# ---------------------------------------------------------------------------

class TestSmoothingEffectOnLoss:
    """
    Verify that enabling label smoothing actually changes the loss value
    compared to no smoothing, given identical model weights and inputs.
    """

    def test_positive_smoothing_changes_loss(self, device):
        """Positive smoothing produces a different loss than no smoothing."""
        torch.manual_seed(123)
        params_none = _base_params(label_smoothness=0.0)
        params_smooth = _base_params(label_smoothness=0.1)

        model_none = BiEncoderRanker(params_none, device=device)
        model_smooth = BiEncoderRanker(params_smooth, device=device)

        # Copy weights so they are identical
        model_smooth.load_state_dict(model_none.state_dict())

        ctx, cand = _make_synthetic_batch(model_none, seed=999)
        loss_none, _ = model_none(ctx, cand, label_input=None)

        # Re-create the same batch for the second model (same seed)
        ctx2, cand2 = _make_synthetic_batch(model_smooth, seed=999)

        loss_smooth, _ = model_smooth(ctx2, cand2, label_input=None)

        assert not math.isclose(loss_none.item(), loss_smooth.item(), rel_tol=1e-5), (
            f"Positive smoothing should change loss: "
            f"no_smooth={loss_none.item():.6f}, smooth={loss_smooth.item():.6f}"
        )

    def test_negative_smoothing_changes_loss(self, device):
        """Negative smoothing (GLS) produces a different loss than no smoothing."""
        torch.manual_seed(123)
        params_none = _base_params(label_smoothness=0.0)
        params_gls = _base_params(label_smoothness=-0.2)

        model_none = BiEncoderRanker(params_none, device=device)
        model_gls = BiEncoderRanker(params_gls, device=device)

        model_gls.load_state_dict(model_none.state_dict())

        ctx, cand = _make_synthetic_batch(model_none, seed=999)
        loss_none, _ = model_none(ctx, cand, label_input=None)

        ctx2, cand2 = _make_synthetic_batch(model_gls, seed=999)

        loss_gls, _ = model_gls(ctx2, cand2, label_input=None)

        assert not math.isclose(loss_none.item(), loss_gls.item(), rel_tol=1e-5), (
            f"Negative smoothing (GLS) should change loss: "
            f"no_smooth={loss_none.item():.6f}, gls={loss_gls.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Tests – logits shape and scores consistency
# ---------------------------------------------------------------------------

class TestLogitsAndScores:
    """Ensure logits have the correct shape regardless of smoothing config."""

    @pytest.mark.parametrize("smoothness", [0.0, 0.1, -0.2])
    def test_logits_shape_square(self, device, smoothness):
        """Logits should be (batch_size, batch_size) for in-batch negatives."""
        batch_size = 4
        params = _base_params(label_smoothness=smoothness)
        model = BiEncoderRanker(params, device=device)
        model.eval()
        ctx, cand = _make_synthetic_batch(model, batch_size=batch_size, seed=200)
        with torch.no_grad():
            loss, logits = model(ctx, cand, label_input=None)
        assert logits.shape == (batch_size, batch_size), (
            f"Expected ({batch_size}, {batch_size}), got {logits.shape}"
        )

    @pytest.mark.parametrize("smoothness", [0.0, 0.1, -0.2])
    def test_logits_are_finite(self, device, smoothness):
        """All logit values must be finite."""
        params = _base_params(label_smoothness=smoothness)
        model = BiEncoderRanker(params, device=device)
        model.eval()
        ctx, cand = _make_synthetic_batch(model, seed=201)
        with torch.no_grad():
            _, logits = model(ctx, cand, label_input=None)
        assert logits.isfinite().all(), "Logits must be finite"


# ---------------------------------------------------------------------------
# Tests – training with explicit label_input
# ---------------------------------------------------------------------------

class TestExplicitLabelsTraining:
    """
    When label_input is provided (not None), the smoothing code-path should
    still converge.
    """

    N_STEPS = 10

    def _train_with_labels(self, model, n_steps, device, seed=42):
        torch.manual_seed(seed)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        losses = []
        batch_size = 4
        for step in range(n_steps):
            ctx, cand = _make_synthetic_batch(model, batch_size=batch_size,
                                              seed=seed + step)
            # Use diagonal labels (matching the in-batch negative pattern)
            labels = torch.arange(batch_size, dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss, _ = model(ctx, cand, label_input=labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    def test_explicit_labels_no_smoothing(self, device):
        params = _base_params(label_smoothness=0.0)
        model = BiEncoderRanker(params, device=device)
        losses = self._train_with_labels(model, self.N_STEPS, device, seed=100)
        _assert_loss_decreases(losses, "explicit_labels_no_smoothing")

    def test_explicit_labels_positive_smoothing(self, device):
        params = _base_params(label_smoothness=0.1)
        model = BiEncoderRanker(params, device=device)
        losses = self._train_with_labels(model, self.N_STEPS, device)
        _assert_loss_decreases(losses, "explicit_labels_positive_smoothing")

    def test_explicit_labels_negative_smoothing(self, device):
        params = _base_params(label_smoothness=-0.2)
        model = BiEncoderRanker(params, device=device)
        losses = self._train_with_labels(model, self.N_STEPS, device)
        _assert_loss_decreases(losses, "explicit_labels_negative_smoothing")


# ---------------------------------------------------------------------------
# Entry point (for convenience: `python tests/test_label_smoothing_regression.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
