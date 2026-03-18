"""
DrivingDataset: PyTorch Dataset for loading Front RGB images and driving labels.

Loads image/label pairs from the data pipeline output structure:
  - Images: src/data/{session}/front/*.png
  - Labels: src/data/{session}/labels/driving_log.csv

Applies resize (800×600 → 224×224) and ImageNet normalization.
Skips corrupted or missing files with warning logs.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
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
