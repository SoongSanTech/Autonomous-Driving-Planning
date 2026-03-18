"""
Tests for DrivingDataset.

Uses temporary directories with synthetic images and CSV files
to validate data loading, preprocessing, and error handling.
"""

import csv
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model.dataset import IMAGENET_MEAN, IMAGENET_STD, DrivingDataset


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """Create a valid session directory with synthetic data."""
    front_dir = tmp_path / "front"
    front_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # Create 5 synthetic 800×600 RGB images
    filenames = []
    for i in range(5):
        fname = f"{1000 + i}.png"
        filenames.append(fname)
        img = Image.fromarray(
            np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        )
        img.save(front_dir / fname)

    # Write CSV
    csv_path = labels_dir / "driving_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
        for i, fname in enumerate(filenames):
            steering = -0.5 + i * 0.25  # range across [-0.5, 0.5]
            throttle = 0.2 + i * 0.15   # range across [0.2, 0.8]
            writer.writerow([fname, 10.0, steering, throttle, 0.0])

    return tmp_path


class TestDrivingDatasetBasic:
    """Core loading and preprocessing tests."""

    def test_len(self, session_dir: Path):
        ds = DrivingDataset(str(session_dir))
        assert len(ds) == 5

    def test_getitem_shapes(self, session_dir: Path):
        ds = DrivingDataset(str(session_dir))
        image, controls = ds[0]
        assert image.shape == (3, 224, 224)
        assert controls.shape == (2,)

    def test_getitem_dtypes(self, session_dir: Path):
        ds = DrivingDataset(str(session_dir))
        image, controls = ds[0]
        assert image.dtype == torch.float32
        assert controls.dtype == torch.float32

    def test_controls_values(self, session_dir: Path):
        """Verify steering/throttle match CSV values."""
        ds = DrivingDataset(str(session_dir))
        _, controls = ds[0]
        assert pytest.approx(controls[0].item(), abs=1e-5) == -0.5
        assert pytest.approx(controls[1].item(), abs=1e-5) == 0.2

    def test_all_samples_accessible(self, session_dir: Path):
        ds = DrivingDataset(str(session_dir))
        for i in range(len(ds)):
            image, controls = ds[i]
            assert image.shape == (3, 224, 224)
            assert controls.shape == (2,)


class TestDrivingDatasetNormalization:
    """ImageNet normalization tests."""

    def test_normalized_range(self, session_dir: Path):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        ds = DrivingDataset(str(session_dir))
        image, _ = ds[0]
        # ImageNet normalized values are typically in [-2.2, 2.7]
        assert image.min() >= -3.0
        assert image.max() <= 3.0

    def test_not_in_0_255_range(self, session_dir: Path):
        """Normalized images should NOT be in [0, 255] range."""
        ds = DrivingDataset(str(session_dir))
        image, _ = ds[0]
        # If properly normalized, max should be well below 255
        assert image.max() < 10.0


class TestDrivingDatasetErrorHandling:
    """Missing/corrupted file handling tests."""

    def test_missing_csv_raises(self, tmp_path: Path):
        """Missing CSV should raise FileNotFoundError."""
        (tmp_path / "front").mkdir()
        with pytest.raises(FileNotFoundError):
            DrivingDataset(str(tmp_path))

    def test_missing_image_skipped(self, tmp_path: Path, caplog):
        """Missing image files should be skipped with warning."""
        front_dir = tmp_path / "front"
        front_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create CSV referencing a non-existent image
        csv_path = labels_dir / "driving_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
            writer.writerow(["nonexistent.png", 10.0, 0.1, 0.5, 0.0])

        with caplog.at_level(logging.WARNING):
            ds = DrivingDataset(str(tmp_path))

        assert len(ds) == 0
        assert "missing" in caplog.text.lower()

    def test_corrupted_image_skipped(self, tmp_path: Path, caplog):
        """Corrupted image files should be skipped with warning."""
        front_dir = tmp_path / "front"
        front_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create a corrupted image file (random bytes)
        corrupted_path = front_dir / "bad.png"
        corrupted_path.write_bytes(b"not a real png file content")

        # Also create one valid image
        valid_img = Image.fromarray(
            np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        )
        valid_img.save(front_dir / "good.png")

        csv_path = labels_dir / "driving_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
            writer.writerow(["bad.png", 10.0, 0.1, 0.5, 0.0])
            writer.writerow(["good.png", 10.0, 0.2, 0.6, 0.0])

        with caplog.at_level(logging.WARNING):
            ds = DrivingDataset(str(tmp_path))

        assert len(ds) == 1
        assert "corrupted" in caplog.text.lower() or "skipped" in caplog.text.lower()

    def test_mixed_valid_and_invalid(self, session_dir: Path, caplog):
        """Dataset with mix of valid + missing images loads only valid ones."""
        # Add a CSV row referencing a missing image
        csv_path = session_dir / "labels" / "driving_log.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ghost.png", 10.0, 0.0, 0.5, 0.0])

        with caplog.at_level(logging.WARNING):
            ds = DrivingDataset(str(session_dir))

        # 5 valid + 1 missing = 5 loaded
        assert len(ds) == 5

    def test_empty_dataset_warning(self, tmp_path: Path, caplog):
        """Empty dataset (all invalid) should log warning."""
        front_dir = tmp_path / "front"
        front_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        csv_path = labels_dir / "driving_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
            writer.writerow(["missing.png", 10.0, 0.0, 0.5, 0.0])

        with caplog.at_level(logging.WARNING):
            ds = DrivingDataset(str(tmp_path))

        assert len(ds) == 0
        assert "no valid samples" in caplog.text.lower()


class TestDrivingDatasetCustomTransform:
    """Custom transform support."""

    def test_custom_transform(self, session_dir: Path):
        """Custom transform should be applied instead of default."""
        from torchvision import transforms

        custom = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        ds = DrivingDataset(str(session_dir), transform=custom)
        image, _ = ds[0]
        assert image.shape == (3, 128, 128)
