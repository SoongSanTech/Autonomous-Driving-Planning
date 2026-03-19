"""
Tests for CheckpointManager.

Includes unit tests and property-based tests for:
- Property 4: Checkpoint round-trip preservation
- Property 15: Checkpoint filename format
- Property 16: Checkpoint content completeness
"""

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from model.checkpoint import CheckpointManager, REQUIRED_KEYS


class SimpleModel(nn.Module):
    """Minimal model for checkpoint testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def ckpt_manager(tmp_path):
    return CheckpointManager(str(tmp_path))


@pytest.fixture
def model_and_optimizer():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


class TestCheckpointManagerSave:
    """Save functionality tests."""

    def test_save_creates_file(self, ckpt_manager, model_and_optimizer):
        model, opt = model_and_optimizer
        path = ckpt_manager.save(model, opt, epoch=1, metrics={"loss": 0.5}, model_type="bc")
        assert Path(path).exists()

    def test_save_creates_checksum(self, ckpt_manager, model_and_optimizer):
        model, opt = model_and_optimizer
        path = ckpt_manager.save(model, opt, epoch=1, metrics={}, model_type="bc")
        assert Path(path).with_suffix(".pth.sha256").exists()

    def test_save_invalid_model_type(self, ckpt_manager, model_and_optimizer):
        model, opt = model_and_optimizer
        with pytest.raises(ValueError, match="model_type"):
            ckpt_manager.save(model, opt, epoch=1, metrics={}, model_type="invalid")

    def test_save_filename_format(self, ckpt_manager, model_and_optimizer):
        """Property 15: Filename contains model_type and timestamp."""
        model, opt = model_and_optimizer
        path = ckpt_manager.save(model, opt, epoch=5, metrics={}, model_type="rl")
        filename = Path(path).name
        assert filename.startswith("rl_")
        assert "epoch5" in filename
        assert filename.endswith(".pth")
        # Check timestamp format YYYYMMDD_HHMMSS
        assert re.match(r"rl_\d{8}_\d{6}_epoch5\.pth", filename)


class TestCheckpointManagerLoad:
    """Load functionality tests."""

    def test_load_restores_weights(self, ckpt_manager, model_and_optimizer):
        """Property 4: Save → Load preserves model weights."""
        model, opt = model_and_optimizer
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        path = ckpt_manager.save(model, opt, epoch=1, metrics={"loss": 0.1}, model_type="bc")

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # Load should restore original
        ckpt_manager.load(path, model, opt)

        for key in original_state:
            assert torch.equal(model.state_dict()[key], original_state[key])

    def test_load_restores_optimizer(self, ckpt_manager):
        model = SimpleModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Do a step to create optimizer state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        original_opt_state = {
            k: v for k, v in opt.state_dict().items()
        }

        path = ckpt_manager.save(model, opt, epoch=1, metrics={}, model_type="bc")

        # Reset optimizer
        opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
        ckpt_manager.load(path, model, opt2)

        # Optimizer state should be restored
        assert len(opt2.state_dict()["state"]) > 0

    def test_load_returns_metadata(self, ckpt_manager, model_and_optimizer):
        """Property 16: Checkpoint contains all required fields."""
        model, opt = model_and_optimizer
        metrics = {"train_loss": 0.5, "val_loss": 0.3}
        path = ckpt_manager.save(model, opt, epoch=3, metrics=metrics, model_type="bc")

        result = ckpt_manager.load(path, model, opt)

        assert result["model_type"] == "bc"
        assert result["epoch"] == 3
        assert result["metrics"] == metrics
        assert result["config"]["input_shape"] == (3, 224, 224)
        assert "timestamp" in result
        for key in REQUIRED_KEYS:
            assert key in result

    def test_load_missing_file(self, ckpt_manager, model_and_optimizer):
        model, opt = model_and_optimizer
        with pytest.raises(FileNotFoundError):
            ckpt_manager.load("/nonexistent/path.pth", model, opt)

    def test_load_corrupted_file(self, ckpt_manager, model_and_optimizer, tmp_path):
        model, opt = model_and_optimizer
        bad_path = tmp_path / "bad.pth"
        bad_path.write_bytes(b"corrupted data")
        with pytest.raises(ValueError):
            ckpt_manager.load(str(bad_path), model, opt)


class TestCheckpointVerify:
    """Integrity verification tests."""

    def test_verify_valid(self, ckpt_manager, model_and_optimizer):
        model, opt = model_and_optimizer
        path = ckpt_manager.save(model, opt, epoch=1, metrics={}, model_type="bc")
        assert ckpt_manager.verify_checkpoint(path) is True

    def test_verify_tampered(self, ckpt_manager, model_and_optimizer):
        model, opt = model_and_optimizer
        path = ckpt_manager.save(model, opt, epoch=1, metrics={}, model_type="bc")

        # Tamper with the file
        with open(path, "ab") as f:
            f.write(b"tampered")

        assert ckpt_manager.verify_checkpoint(path) is False

    def test_verify_missing(self, ckpt_manager):
        assert ckpt_manager.verify_checkpoint("/nonexistent.pth") is False


# --- Property-based tests ---

class TestCheckpointProperties:
    """Property-based tests for checkpoint round-trip."""

    @given(
        epoch=st.integers(min_value=0, max_value=1000),
        model_type=st.sampled_from(["bc", "rl"]),
        loss_val=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(
        max_examples=10,
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_roundtrip_preserves_all(self, tmp_path_factory, epoch, model_type, loss_val):
        """Property 4: Save → Load preserves weights and metadata."""
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(str(tmp_path))

        model = SimpleModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        metrics = {"loss": loss_val}

        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        path = mgr.save(model, opt, epoch=epoch, metrics=metrics, model_type=model_type)

        # Corrupt model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)

        result = mgr.load(path, model, opt)

        # Weights restored
        for key in original_state:
            assert torch.equal(model.state_dict()[key], original_state[key])

        # Metadata preserved
        assert result["epoch"] == epoch
        assert result["model_type"] == model_type
        assert result["metrics"]["loss"] == pytest.approx(loss_val)

    @given(model_type=st.sampled_from(["bc", "rl"]))
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_filename_format_property(self, tmp_path_factory, model_type):
        """Property 15: Filename follows {model_type}_{timestamp}_epoch{N}.pth."""
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(str(tmp_path))
        model = SimpleModel()
        opt = torch.optim.Adam(model.parameters())

        path = mgr.save(model, opt, epoch=42, metrics={}, model_type=model_type)
        filename = Path(path).name

        pattern = rf"^{model_type}_\d{{8}}_\d{{6}}_epoch42\.pth$"
        assert re.match(pattern, filename), f"Filename '{filename}' doesn't match pattern"

    @given(model_type=st.sampled_from(["bc", "rl"]))
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_content_completeness_property(self, tmp_path_factory, model_type):
        """Property 16: Checkpoint contains all required keys."""
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(str(tmp_path))
        model = SimpleModel()
        opt = torch.optim.Adam(model.parameters())

        path = mgr.save(model, opt, epoch=1, metrics={"val": 0.1}, model_type=model_type)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        for key in REQUIRED_KEYS:
            assert key in checkpoint, f"Missing key: {key}"
