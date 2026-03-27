"""
BCTrainer: Behavioral Cloning training pipeline.

Two-phase training:
  Phase 1: Backbone frozen (10 epochs) — train FC head only
  Phase 2: Full fine-tune — unfreeze backbone

Loss: MSE with steering weight 2.0, throttle weight 1.0
Optimizer: Adam LR 1e-4, ReduceLROnPlateau
Early stopping: patience=10 on val_loss
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.bc_model import BehavioralCloningModel
from model.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class BCTrainer:
    """
    Behavioral Cloning trainer with two-phase training.

    Args:
        model: BehavioralCloningModel instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        lr: Initial learning rate.
        steering_weight: Loss weight for steering (default 2.0).
        throttle_weight: Loss weight for throttle (default 1.0).
        device: Training device ('cuda' or 'cpu').
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: BehavioralCloningModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        steering_weight: float = 2.0,
        throttle_weight: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.steering_weight = steering_weight
        self.throttle_weight = throttle_weight

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()
        self.ckpt_manager = CheckpointManager(checkpoint_dir)
        self._lr = lr

    def train(
        self,
        epochs: int = 50,
        patience: int = 10,
        frozen_epochs: int = 10,
        max_grad_norm: float = 1.0,
    ) -> dict:
        """
        Run two-phase training.

        Args:
            epochs: Maximum total epochs.
            patience: Early stopping patience.
            frozen_epochs: Epochs with backbone frozen (Phase 1).
            max_grad_norm: Gradient clipping max norm.

        Returns:
            Dict with best metrics and checkpoint path.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_path = None
        history = {"train_loss": [], "val_loss": []}

        # Phase 1: freeze backbone
        self.model.freeze_backbone()
        self._reinit_optimizer()
        logger.info("Phase 1: backbone frozen for %d epochs", frozen_epochs)

        for epoch in range(1, epochs + 1):
            # Phase transition
            if epoch == frozen_epochs + 1:
                self.model.unfreeze_backbone()
                self._reinit_optimizer(lr=self._lr * 0.1)
                logger.info("Phase 2: backbone unfrozen, LR reduced")

            train_loss = self._train_epoch(max_grad_norm)
            val_loss, val_metrics = self._validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, "
                "mae_steer: %.4f, mae_throttle: %.4f, lr: %.2e",
                epoch, epochs, train_loss, val_loss,
                val_metrics["mae_steering"], val_metrics["mae_throttle"],
                current_lr,
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_metrics,
                }
                best_path = self.ckpt_manager.save(
                    self.model, self.optimizer, epoch, metrics, model_type="bc",
                    config={"learning_rate": current_lr, "batch_size": self.train_loader.batch_size},
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return {
            "best_val_loss": best_val_loss,
            "best_checkpoint": best_path,
            "history": history,
        }

    def _train_epoch(self, max_grad_norm: float) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, controls in self.train_loader:
            images = images.to(self.device)
            controls = controls.to(self.device)

            steering_gt = controls[:, 0:1]
            throttle_gt = controls[:, 1:2]

            self.optimizer.zero_grad()
            steering_pred, throttle_pred = self.model(images)

            loss_steer = self.criterion(steering_pred, steering_gt) * self.steering_weight
            loss_throttle = self.criterion(throttle_pred, throttle_gt) * self.throttle_weight
            loss = loss_steer + loss_throttle

            # NaN detection
            if math.isnan(loss.item()):
                logger.warning("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple:
        """Run validation. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        total_mae_steer = 0.0
        total_mae_throttle = 0.0
        num_batches = 0

        for images, controls in self.val_loader:
            images = images.to(self.device)
            controls = controls.to(self.device)

            steering_gt = controls[:, 0:1]
            throttle_gt = controls[:, 1:2]

            steering_pred, throttle_pred = self.model(images)

            loss_steer = self.criterion(steering_pred, steering_gt) * self.steering_weight
            loss_throttle = self.criterion(throttle_pred, throttle_gt) * self.throttle_weight
            loss = loss_steer + loss_throttle

            total_loss += loss.item()
            total_mae_steer += (steering_pred - steering_gt).abs().mean().item()
            total_mae_throttle += (throttle_pred - throttle_gt).abs().mean().item()
            num_batches += 1

        n = max(num_batches, 1)
        return total_loss / n, {
            "mae_steering": total_mae_steer / n,
            "mae_throttle": total_mae_throttle / n,
        }

    def _reinit_optimizer(self, lr: Optional[float] = None):
        """Reinitialize optimizer with current trainable parameters."""
        lr = lr or self._lr
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
