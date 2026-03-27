"""
DrivingDataset & DataLoaderFactory for Front RGB driving data.

Loads image/label pairs from the data pipeline output structure:
  - Images: src/data/{session}/front/*.png
  - Labels: src/data/{session}/labels/driving_log.csv

DrivingDataset: Resize (800×600 → 224×224) + ImageNet normalization.
DataLoaderFactory: 80/20 train/val split, augmentation, batch loading.
"""

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform() -> transforms.Compose:
    """Default preprocessing: resize to 224×224 + ImageNet normalize."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DrivingDataset(Dataset):
    """
    PyTorch Dataset for front-camera driving data.

    Loads Front RGB images and corresponding steering/throttle labels
    from a single session directory. Corrupted or missing images are
    skipped with a warning log.

    Args:
        session_dir: Path to session directory (e.g. src/data/2024-01-01_120000/).
                     Must contain front/ (images) and labels/driving_log.csv.
        transform: Optional torchvision transform. Defaults to
                   Resize(224,224) + ToTensor + ImageNet normalize.
    """

    def __init__(
        self,
        session_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.session_dir = Path(session_dir)
        self.transform = transform or default_transform()

        self._image_dir = self.session_dir / "front"
        csv_path = self.session_dir / "labels" / "driving_log.csv"

        # Load and validate CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Label file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Build valid sample list, skipping missing/corrupted images
        self._samples: list[Tuple[Path, float, float]] = []
        skipped = 0

        for _, row in df.iterrows():
            image_path = self._image_dir / str(row["image_filename"])

            if not image_path.exists():
                logger.warning("Image file missing, skipping: %s", image_path)
                skipped += 1
                continue

            # Validate image can be opened
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                logger.warning("Corrupted image file, skipping: %s", image_path)
                skipped += 1
                continue

            steering = float(row["steering"])
            throttle = float(row["throttle"])
            self._samples.append((image_path, steering, throttle))

        if skipped > 0:
            logger.warning(
                "Skipped %d samples due to missing/corrupted images "
                "(loaded %d valid samples)",
                skipped,
                len(self._samples),
            )

        if len(self._samples) == 0:
            logger.warning("No valid samples found in %s", self.session_dir)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (3, 224, 224) normalized float tensor
            controls: (2,) tensor [steering, throttle]
        """
        image_path, steering, throttle = self._samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        controls = torch.tensor([steering, throttle], dtype=torch.float32)
        return image, controls


class DrivingAugmentation:
    """
    Data augmentation for driving data.

    Applies horizontal flip (with steering sign inversion),
    brightness adjustment (±20%), and Gaussian noise.
    Augmentation is applied randomly per sample.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_range: float = 0.2,
        noise_std: float = 0.02,
    ):
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.noise_std = noise_std

    def __call__(
        self, image: Image.Image, steering: float, throttle: float
    ) -> Tuple[Image.Image, float, float]:
        """
        Apply augmentations to image and adjust labels accordingly.

        Args:
            image: PIL Image (before tensor conversion)
            steering: steering value
            throttle: throttle value

        Returns:
            (augmented_image, adjusted_steering, throttle)
        """
        # Horizontal flip with steering sign inversion
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            steering = -steering

        # Brightness adjustment ±20%
        factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)

        return image, steering, throttle


class AugmentedDrivingDataset(Dataset):
    """
    Wraps DrivingDataset with data augmentation.

    Applies augmentation before the tensor transform so that
    steering labels are correctly adjusted (e.g., flip → negate steering).
    Gaussian noise is added after tensor conversion.
    """

    def __init__(
        self,
        base_dataset: DrivingDataset,
        augmentation: Optional[DrivingAugmentation] = None,
        noise_std: float = 0.02,
    ):
        self._base = base_dataset
        self._aug = augmentation or DrivingAugmentation()
        self._noise_std = noise_std

        # Build a raw transform (resize + toTensor + normalize) for post-aug
        self._transform = default_transform()

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, steering, throttle = self._base._samples[idx]

        image = Image.open(image_path).convert("RGB")

        # Apply PIL-level augmentations (flip, brightness)
        image, steering, throttle = self._aug(image, steering, throttle)

        # Apply standard transform (resize, toTensor, normalize)
        image = self._transform(image)

        # Add Gaussian noise
        if self._noise_std > 0:
            noise = torch.randn_like(image) * self._noise_std
            image = image + noise

        controls = torch.tensor([steering, throttle], dtype=torch.float32)
        return image, controls


class DataLoaderFactory:
    """
    Creates train/val DataLoaders from a session directory.

    Splits the dataset 80/20, applies augmentation to training set only,
    and returns DataLoaders with configurable batch size and workers.
    """

    @staticmethod
    def create_dataloaders(
        session_dir: str,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        augment: bool = True,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders.

        Args:
            session_dir: Path to session directory.
            batch_size: Batch size for both loaders.
            val_split: Fraction of data for validation (default 0.2).
            num_workers: Number of data loading workers.
            augment: Whether to apply augmentation to training data.
            seed: Random seed for reproducible splits.

        Returns:
            (train_loader, val_loader)
        """
        base_dataset = DrivingDataset(session_dir)
        total = len(base_dataset)

        if total == 0:
            raise ValueError(f"No valid samples in {session_dir}")

        # Deterministic split
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=generator).tolist()

        val_size = max(1, int(total * val_split))
        train_size = total - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Validation: plain dataset with default transform
        val_subset = Subset(base_dataset, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Training: with or without augmentation
        if augment:
            aug_dataset = AugmentedDrivingDataset(base_dataset)
            train_subset = Subset(aug_dataset, train_indices)
        else:
            train_subset = Subset(base_dataset, train_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        logger.info(
            "DataLoaders created: %d train, %d val (batch=%d, augment=%s)",
            train_size, val_size, batch_size, augment,
        )

        return train_loader, val_loader
