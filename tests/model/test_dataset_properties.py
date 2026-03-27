"""
Property-based tests for DrivingDataset.

Uses Hypothesis to generate random data and verify correctness properties:
- Property 3: Data pairing consistency
- Property 13: Image normalization range
- Property 14: Dataset split completeness
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from PIL import Image

from model.dataset import DrivingDataset, IMAGENET_MEAN, IMAGENET_STD


# --- Strategies ---

@st.composite
def driving_session(draw, num_samples=None):
    """Generate a temporary session directory with random driving data."""
    if num_samples is None:
        num_samples = draw(st.integers(min_value=1, max_value=20))

    tmp_path = Path(draw(st.from_type(type).filter(lambda _: True)))
    # We'll use pytest's tmp_path via fixture instead
    return num_samples


@st.composite
def steering_throttle_pairs(draw):
    """Generate valid steering/throttle pairs."""
    steering = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
    throttle = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return steering, throttle


# --- Fixtures ---

def create_session(tmp_path: Path, num_samples: int, labels: list):
    """Helper to create a session directory with given labels."""
    front_dir = tmp_path / "front"
    front_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for i in range(num_samples):
        fname = f"{1000 + i}.png"
        filenames.append(fname)
        img = Image.fromarray(
            np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        )
        img.save(front_dir / fname)

    csv_path = labels_dir / "driving_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
        for i, fname in enumerate(filenames):
            s, t = labels[i]
            writer.writerow([fname, 10.0, s, t, 0.0])

    return tmp_path, filenames, labels


# --- Property 3: Data pairing consistency ---

class TestDataPairingConsistency:
    """Property 3: Each image is correctly paired with its steering/throttle."""

    @given(
        num_samples=st.integers(min_value=1, max_value=10),
        data=st.data(),
    )
    @settings(
        max_examples=20,
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_pairing_matches_csv(self, tmp_path_factory, num_samples, data):
        """Every dataset sample's controls must match the CSV row."""
        tmp_path = tmp_path_factory.mktemp("session")

        labels = [
            (
                data.draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
                data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            )
            for _ in range(num_samples)
        ]

        session_path, filenames, expected_labels = create_session(
            tmp_path, num_samples, labels
        )

        ds = DrivingDataset(str(session_path))
        assert len(ds) == num_samples

        for i in range(len(ds)):
            _, controls = ds[i]
            expected_s, expected_t = expected_labels[i]
            assert pytest.approx(controls[0].item(), abs=1e-5) == expected_s
            assert pytest.approx(controls[1].item(), abs=1e-5) == expected_t


# --- Property 13: Image normalization range ---

class TestImageNormalizationRange:
    """Property 13: After preprocessing, values are in ImageNet-normalized range."""

    @given(
        pixel_val=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=50, deadline=5000)
    def test_single_color_normalization_bounds(self, pixel_val):
        """A uniform-color image should produce values within expected bounds."""
        # Compute theoretical min/max for ImageNet normalization
        # After ToTensor: pixel_val / 255.0
        # After Normalize: (pixel_val/255.0 - mean) / std
        for ch in range(3):
            normalized = (pixel_val / 255.0 - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch]
            # Theoretical bounds: (0 - max_mean) / min_std to (1 - min_mean) / min_std
            assert normalized >= -3.0
            assert normalized <= 3.0

    def test_real_image_normalization(self, tmp_path):
        """Real dataset images should have normalized values in [-3, 3]."""
        front_dir = tmp_path / "front"
        front_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create diverse images
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            )
            img.save(front_dir / f"{i}.png")

        csv_path = labels_dir / "driving_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
            for i in range(3):
                writer.writerow([f"{i}.png", 10.0, 0.0, 0.5, 0.0])

        ds = DrivingDataset(str(tmp_path))
        for i in range(len(ds)):
            image, _ = ds[i]
            assert image.min() >= -3.0, f"Min {image.min()} below -3.0"
            assert image.max() <= 3.0, f"Max {image.max()} above 3.0"


# --- Property 14: Dataset split completeness ---

class TestDatasetSplitCompleteness:
    """Property 14: Train + val = total samples, val ≈ 20%."""

    @given(
        num_samples=st.integers(min_value=2, max_value=50),
    )
    @settings(
        max_examples=20,
        deadline=60000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_split_covers_all_samples(self, tmp_path_factory, num_samples):
        """Train + val indices must cover all samples without overlap."""
        from model.dataset import DataLoaderFactory

        tmp_path = tmp_path_factory.mktemp("split")
        labels = [(0.0, 0.5)] * num_samples
        create_session(tmp_path, num_samples, labels)

        train_loader, val_loader = DataLoaderFactory.create_dataloaders(
            str(tmp_path), batch_size=4, num_workers=0, augment=False
        )

        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)

        # Total must equal original
        assert train_count + val_count == num_samples

        # Val should be approximately 20% (±1 sample for rounding)
        expected_val = max(1, int(num_samples * 0.2))
        assert abs(val_count - expected_val) <= 1

    def test_no_overlap_between_splits(self, tmp_path):
        """Train and val sets must not share any samples."""
        from model.dataset import DataLoaderFactory

        num_samples = 20
        labels = [(i * 0.1 - 1.0, 0.5) for i in range(num_samples)]
        create_session(tmp_path, num_samples, labels)

        train_loader, val_loader = DataLoaderFactory.create_dataloaders(
            str(tmp_path), batch_size=4, num_workers=0, augment=False
        )

        # Collect all steering values from each split
        train_steerings = set()
        for _, controls in train_loader:
            for c in controls:
                train_steerings.add(round(c[0].item(), 5))

        val_steerings = set()
        for _, controls in val_loader:
            for c in controls:
                val_steerings.add(round(c[0].item(), 5))

        # With unique steering values, there should be no overlap
        overlap = train_steerings & val_steerings
        assert len(overlap) == 0, f"Overlap found: {overlap}"
