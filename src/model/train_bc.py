"""
CLI script for Behavioral Cloning training.

Usage:
    python -m model.train_bc --data_path src/data/2024-01-01_120000
    python -m model.train_bc --data_path src/data/2024-01-01_120000 --epochs 30 --lr 5e-5
"""

import argparse
import logging
import sys

import torch

from model.bc_model import BehavioralCloningModel
from model.bc_trainer import BCTrainer
from model.dataset import DataLoaderFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BC driving model")
    parser.add_argument("--data_path", required=True, help="Session directory path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frozen_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info("Starting BC training: %s", args)

    # Create data loaders
    train_loader, val_loader = DataLoaderFactory.create_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
    )

    # Create model
    model = BehavioralCloningModel(pretrained=True)

    # Train
    trainer = BCTrainer(
        model, train_loader, val_loader,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    result = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        frozen_epochs=args.frozen_epochs,
    )

    logger.info("Training complete. Best val_loss: %.4f", result["best_val_loss"])
    logger.info("Best checkpoint: %s", result["best_checkpoint"])


if __name__ == "__main__":
    main()
