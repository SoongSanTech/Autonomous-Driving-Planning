"""
Tests for ModelEvaluator.

Tests offline evaluation with synthetic data.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.evaluator import ModelEvaluator


class SimpleModel(nn.Module):
    """Model that always predicts (0.0, 0.5)."""

    def forward(self, x):
        batch = x.size(0)
        return torch.zeros(batch, 1), torch.full((batch, 1), 0.5)


class TestModelEvaluatorOffline:
    """Offline evaluation tests."""

    def test_evaluate_offline(self):
        model = SimpleModel()
        images = torch.randn(10, 3, 224, 224)
        controls = torch.tensor([[0.1, 0.5]] * 10)
        loader = DataLoader(TensorDataset(images, controls), batch_size=4)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_offline(model, loader)

        assert "mae_steering" in metrics
        assert "mae_throttle" in metrics
        assert "inference_time_ms" in metrics
        assert metrics["total_samples"] == 10

    def test_mae_values(self):
        """Model predicts (0, 0.5), ground truth is (0.1, 0.5)."""
        model = SimpleModel()
        images = torch.randn(10, 3, 224, 224)
        controls = torch.tensor([[0.1, 0.5]] * 10)
        loader = DataLoader(TensorDataset(images, controls), batch_size=10)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_offline(model, loader)

        assert metrics["mae_steering"] == pytest.approx(0.1, abs=1e-4)
        assert metrics["mae_throttle"] == pytest.approx(0.0, abs=1e-4)

    def test_latency_measured(self):
        model = SimpleModel()
        images = torch.randn(4, 3, 224, 224)
        controls = torch.tensor([[0.0, 0.5]] * 4)
        loader = DataLoader(TensorDataset(images, controls), batch_size=4)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_offline(model, loader)
        assert metrics["inference_time_ms"] > 0
