"""
Tests for BehavioralCloningModel.

Includes unit tests and property-based tests for:
- Property 1: Model output shape consistency
- Property 2: Model output range constraints
"""

import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from model.bc_model import BehavioralCloningModel


@pytest.fixture
def model():
    """Create a BC model (no pretrained weights for speed)."""
    return BehavioralCloningModel(pretrained=False)


class TestBCModelBasic:
    """Core model tests."""

    def test_output_shapes(self, model):
        x = torch.randn(2, 3, 224, 224)
        steering, throttle = model(x)
        assert steering.shape == (2, 1)
        assert throttle.shape == (2, 1)

    def test_single_sample(self, model):
        x = torch.randn(1, 3, 224, 224)
        steering, throttle = model(x)
        assert steering.shape == (1, 1)
        assert throttle.shape == (1, 1)

    def test_output_ranges(self, model):
        x = torch.randn(4, 3, 224, 224)
        steering, throttle = model(x)
        assert steering.min() >= -1.0
        assert steering.max() <= 1.0
        assert throttle.min() >= 0.0
        assert throttle.max() <= 1.0

    def test_feature_dim(self, model):
        x = torch.randn(1, 3, 224, 224)
        features = model.get_features(x)
        assert features.shape == (1, 512)


class TestBCModelBackboneFreeze:
    """Backbone freeze/unfreeze tests."""

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_fc_head_trainable_when_frozen(self, model):
        """FC head should remain trainable when backbone is frozen."""
        model.freeze_backbone()
        for param in model.fc_head.parameters():
            assert param.requires_grad

    def test_gradient_flow_frozen(self, model):
        """With frozen backbone, only FC head should get gradients."""
        model.freeze_backbone()
        x = torch.randn(1, 3, 224, 224)
        steering, throttle = model(x)
        loss = steering.sum() + throttle.sum()
        loss.backward()

        # Backbone grads should be None
        for param in model.backbone.parameters():
            assert param.grad is None

        # FC head should have grads
        has_grad = any(p.grad is not None for p in model.fc_head.parameters())
        assert has_grad


class TestBCModelDeterminism:
    """Determinism tests."""

    def test_eval_mode_deterministic(self, model):
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        s1, t1 = model(x)
        s2, t2 = model(x)
        assert torch.equal(s1, s2)
        assert torch.equal(t1, t2)


# --- Property-based tests ---

class TestBCModelProperties:
    """Property-based tests for BC model."""

    @given(batch_size=st.integers(min_value=1, max_value=8))
    @settings(
        max_examples=5,
        deadline=60000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_shape_property(self, batch_size):
        """Property 1: Any valid input → exactly 2 outputs."""
        model = BehavioralCloningModel(pretrained=False)
        model.eval()
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            steering, throttle = model(x)
        assert steering.shape == (batch_size, 1)
        assert throttle.shape == (batch_size, 1)

    @given(batch_size=st.integers(min_value=1, max_value=8))
    @settings(
        max_examples=5,
        deadline=60000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_output_range_property(self, batch_size):
        """Property 2: steering ∈ [-1,1], throttle ∈ [0,1]."""
        model = BehavioralCloningModel(pretrained=False)
        model.eval()
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            steering, throttle = model(x)
        assert steering.min() >= -1.0
        assert steering.max() <= 1.0
        assert throttle.min() >= 0.0
        assert throttle.max() <= 1.0
