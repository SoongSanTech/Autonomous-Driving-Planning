"""
CheckpointManager: Save/load PyTorch model checkpoints.

Checkpoint format: {model_type}_{timestamp}_epoch{N}.pth
Contents: model_type, epoch, model_state_dict, optimizer_state_dict,
          metrics, config, timestamp.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {
    "model_type", "epoch", "model_state_dict",
    "optimizer_state_dict", "metrics", "config", "timestamp",
}


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        model_type: str,
        config: Optional[dict] = None,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: PyTorch model.
            optimizer: Optimizer with state.
            epoch: Current epoch number.
            metrics: Training/validation metrics dict.
            model_type: 'bc' or 'rl'.
            config: Optional config dict (defaults include input_shape).

        Returns:
            Path to saved checkpoint file.
        """
        if model_type not in ("bc", "rl"):
            raise ValueError(f"model_type must be 'bc' or 'rl', got '{model_type}'")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_{timestamp}_epoch{epoch}.pth"
        filepath = self.checkpoint_dir / filename

        default_config = {
            "architecture": "resnet18",
            "input_shape": (3, 224, 224),
        }
        if config:
            default_config.update(config)

        checkpoint = {
            "model_type": model_type,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": default_config,
            "timestamp": timestamp,
        }

        torch.save(checkpoint, filepath)

        # Save checksum for integrity verification
        checksum = self._compute_checksum(filepath)
        checksum_path = filepath.with_suffix(".pth.sha256")
        checksum_path.write_text(checksum)

        logger.info("Checkpoint saved: %s (epoch %d)", filepath, epoch)
        return str(filepath)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> dict:
        """
        Load a checkpoint and restore model/optimizer state.

        Args:
            path: Path to checkpoint file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            device: Device to map tensors to.

        Returns:
            Checkpoint dict with all metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint is corrupted or missing required keys.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if not self.verify_checkpoint(path):
            raise ValueError(f"Checkpoint integrity check failed: {path}")

        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        # Validate required keys
        missing = REQUIRED_KEYS - set(checkpoint.keys())
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            "Checkpoint loaded: %s (model_type=%s, epoch=%d)",
            path, checkpoint["model_type"], checkpoint["epoch"],
        )
        return checkpoint

    def verify_checkpoint(self, path: str) -> bool:
        """
        Verify checkpoint file integrity using SHA-256 checksum.

        Returns True if valid or if no checksum file exists (backward compat).
        """
        filepath = Path(path)
        checksum_path = filepath.with_suffix(".pth.sha256")

        if not filepath.exists():
            return False

        if not checksum_path.exists():
            # No checksum file — try loading to verify
            try:
                torch.load(filepath, map_location="cpu", weights_only=False)
                return True
            except Exception:
                return False

        expected = checksum_path.read_text().strip()
        actual = self._compute_checksum(filepath)
        return expected == actual

    @staticmethod
    def _compute_checksum(filepath: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
