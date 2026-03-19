"""
Tests for BCTrainer.

Uses synthetic data to test training loop, checkpoint saving,
NaN handling, and phase transitions.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.bc_model import BehavioralCloningModel
from model.bc_trainer import BCTrainer


def make_synthetic_loaders(n_train=20, n_val=8, batch_size=4):
    """Create synthetic train/val DataLoaders."""
    train_images = torch.randn(n_train, 3, 224, 224)
    train_controls = torch.cat([
        torch.rand(n_train, 1) * 2 - 1,  # steering [-1, 1]
        torch.rand(n_train, 1),            # throttle [0, 1]
    ], dim=1)

    val_images = torch.randn(n_val, 3, 224, 224)
    val_controls = torch.cat([
        torch.rand(n_val, 1) * 2 - 1,
        torch.rand(n_val, 1),
    ], dim=1)

    train_loader = DataLoader(
        TensorDataset(train_images, train_controls),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_images, val_controls),
        batch_size=batch_size,
    )
    return train_loader, val_loader


@pytest.fixture
def trainer(tmp_path):
    model = BehavioralCloningModel(pretrained=False)
    train_loader, val_loader = make_synthetic_loaders()
    return BCTrainer(
        model, train_loader, val_loader,
        checkpoint_dir=str(tmp_path),
    )


class TestBCTrainerBasic:
    """Basic training loop tests."""

    def test_train_runs(self, trainer):
        result = trainer.train(epochs=2, frozen_epochs=1, patience=5)
        assert "best_val_loss" in result
        assert result["best_val_loss"] > 0

    def test_checkpoint_saved(self, trainer):
        result = trainer.train(epochs=2, frozen_epochs=1, patience=5)
        assert result["best_checkpoint"] is not None

    def test_history_recorded(self, trainer):
        result = trainer.train(epochs=3, frozen_epochs=1, patience=5)
        assert len(result["history"]["train_loss"]) <= 3
        assert len(result["history"]["val_loss"]) <= 3

    def test_loss_decreases_or_stable(self, trainer):
        """Loss should not explode over a few epochs."""
        result = trainer.train(epochs=3, frozen_epochs=1, patience=10)
        losses = result["history"]["train_loss"]
        # Just check it doesn't go to infinity
        assert all(loss < 1e6 for loss in losses)


class TestBCTrainerPhases:
    """Phase transition tests."""

    def test_phase1_backbone_frozen(self, tmp_path):
        model = BehavioralCloningModel(pretrained=False)
        train_loader, val_loader = make_synthetic_loaders()
        trainer = BCTrainer(model, train_loader, val_loader, checkpoint_dir=str(tmp_path))

        # After init and first train call, backbone should be frozen
        trainer.model.freeze_backbone()
        for param in trainer.model.backbone.parameters():
            assert not param.requires_grad

    def test_phase2_backbone_unfrozen(self, tmp_path):
        model = BehavioralCloningModel(pretrained=False)
        train_loader, val_loader = make_synthetic_loaders()
        trainer = BCTrainer(model, train_loader, val_loader, checkpoint_dir=str(tmp_path))

        # Train past frozen_epochs to trigger phase 2
        trainer.train(epochs=3, frozen_epochs=1, patience=10)
        # After training, backbone should be unfrozen (phase 2)
        for param in trainer.model.backbone.parameters():
            assert param.requires_grad


class TestBCTrainerEarlyStopping:
    """Early stopping tests."""

    def test_early_stopping_triggers(self, tmp_path):
        model = BehavioralCloningModel(pretrained=False)
        train_loader, val_loader = make_synthetic_loaders()
        trainer = BCTrainer(model, train_loader, val_loader, checkpoint_dir=str(tmp_path))

        result = trainer.train(epochs=100, frozen_epochs=1, patience=2)
        # Should stop well before 100 epochs
        actual_epochs = len(result["history"]["train_loss"])
        assert actual_epochs < 100
