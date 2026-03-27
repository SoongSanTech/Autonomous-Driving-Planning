"""
Tests for BCInferenceEngine.

Property 5: Inference latency constraint (< 100ms on CPU).
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from model.bc_model import BehavioralCloningModel
from model.checkpoint import CheckpointManager
from model.inference import BCInferenceEngine


@pytest.fixture
def checkpoint_path(tmp_path):
    """Create a BC checkpoint for testing."""
    model = BehavioralCloningModel(pretrained=False)
    optimizer = torch.optim.Adam(model.parameters())
    mgr = CheckpointManager(str(tmp_path))
    return mgr.save(model, optimizer, epoch=1, metrics={}, model_type="bc")


@pytest.fixture
def engine(checkpoint_path):
    return BCInferenceEngine(checkpoint_path, device="cpu")


class TestBCInferenceEngine:
    """Inference engine tests."""

    def test_predict_numpy(self, engine):
        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        steering, throttle, latency = engine.predict(image)
        assert -1.0 <= steering <= 1.0
        assert 0.0 <= throttle <= 1.0
        assert latency > 0

    def test_predict_tensor(self, engine):
        tensor = torch.randn(1, 3, 224, 224)
        steering, throttle, latency = engine.predict_tensor(tensor)
        assert -1.0 <= steering <= 1.0
        assert 0.0 <= throttle <= 1.0

    def test_avg_latency(self, engine):
        for _ in range(5):
            image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            engine.predict(image)
        assert engine.avg_latency_ms > 0

    def test_metadata(self, engine):
        assert engine.metadata["model_type"] == "bc"
        assert engine.metadata["epoch"] == 1


class TestInferenceLatencyProperty:
    """Property 5: Inference latency < 100ms."""

    @given(batch=st.integers(min_value=1, max_value=3))
    @settings(
        max_examples=3,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_latency_under_100ms(self, checkpoint_path, batch):
        """Single-image inference should complete within 100ms on CPU."""
        engine = BCInferenceEngine(checkpoint_path, device="cpu")

        # Warm up
        dummy = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        engine.predict(dummy)

        # Measure
        latencies = []
        for _ in range(batch):
            image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            _, _, latency = engine.predict(image)
            latencies.append(latency)

        avg = sum(latencies) / len(latencies)
        assert avg < 100.0, f"Average latency {avg:.1f}ms exceeds 100ms"
