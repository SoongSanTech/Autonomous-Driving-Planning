"""Unit tests for DataValidator."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiment.data_validator import DataValidator, DataValidationReport


def _create_session(tmp_path, num_frames=10, corrupt_indices=None,
                    steering_values=None, throttle_values=None,
                    timestamp_start=1000, timestamp_interval=100):
    """Helper to create a fake session directory."""
    session = tmp_path / "session"
    images_dir = session / "front"
    labels_dir = session / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    if steering_values is None:
        steering_values = [0.0] * num_frames
    if throttle_values is None:
        throttle_values = [0.5] * num_frames
    if corrupt_indices is None:
        corrupt_indices = set()

    csv_path = labels_dir / "driving_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
        for i in range(num_frames):
            ts = timestamp_start + i * timestamp_interval
            fname = f"{ts}.png"
            img_path = images_dir / fname
            if i in corrupt_indices:
                img_path.write_bytes(b"not a png")
            else:
                img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
                img.save(str(img_path))
            writer.writerow([fname, 10.0, steering_values[i], throttle_values[i], 0.0])

    return str(session)


class TestValidateImages:
    def test_all_valid(self, tmp_path):
        session = _create_session(tmp_path, num_frames=5)
        v = DataValidator()
        result = v.validate_images(session)
        assert result["total"] == 5
        assert result["valid"] == 5
        assert result["corrupted"] == 0

    def test_corrupted_images(self, tmp_path):
        session = _create_session(tmp_path, num_frames=5, corrupt_indices={0, 2})
        v = DataValidator()
        result = v.validate_images(session)
        assert result["total"] == 5
        assert result["valid"] == 3
        assert result["corrupted"] == 2

    def test_empty_session(self, tmp_path):
        session = tmp_path / "empty"
        session.mkdir()
        v = DataValidator()
        result = v.validate_images(str(session))
        assert result["total"] == 0

    def test_zero_byte_file(self, tmp_path):
        session = _create_session(tmp_path, num_frames=3)
        # Overwrite one file with 0 bytes
        images_dir = Path(session) / "front"
        first_png = sorted(images_dir.glob("*.png"))[0]
        first_png.write_bytes(b"")
        v = DataValidator()
        result = v.validate_images(session)
        assert result["corrupted"] == 1


class TestValidateLabels:
    def test_all_in_range(self, tmp_path):
        session = _create_session(tmp_path, num_frames=5)
        v = DataValidator()
        result = v.validate_labels(session)
        assert result["out_of_range_steering"] == 0
        assert result["out_of_range_throttle"] == 0

    def test_out_of_range_steering(self, tmp_path):
        session = _create_session(
            tmp_path, num_frames=5,
            steering_values=[0.0, -1.5, 0.5, 1.2, 0.0],
        )
        v = DataValidator()
        result = v.validate_labels(session)
        assert result["out_of_range_steering"] == 2

    def test_out_of_range_throttle(self, tmp_path):
        session = _create_session(
            tmp_path, num_frames=4,
            throttle_values=[0.5, -0.1, 1.5, 0.8],
        )
        v = DataValidator()
        result = v.validate_labels(session)
        assert result["out_of_range_throttle"] == 2

    def test_timing_anomalies(self, tmp_path):
        # Normal interval 100ms, but we'll use irregular intervals
        session = _create_session(tmp_path, num_frames=5, timestamp_interval=100)
        # Rewrite CSV with irregular timestamps
        csv_path = Path(session) / "labels" / "driving_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
            timestamps = [1000, 1100, 1150, 1400, 1500]  # intervals: 100, 50, 250, 100
            for ts in timestamps:
                writer.writerow([f"{ts}.png", 10.0, 0.0, 0.5, 0.0])
        v = DataValidator()
        result = v.validate_labels(session)
        assert result["timing_anomalies"] == 2  # 50ms and 250ms are out of 80-120ms

    def test_missing_csv_raises(self, tmp_path):
        session = tmp_path / "no_csv"
        session.mkdir()
        (session / "front").mkdir()
        v = DataValidator()
        with pytest.raises(FileNotFoundError):
            v.validate_labels(str(session))


class TestValidateSession:
    def test_full_validation(self, tmp_path):
        session = _create_session(tmp_path, num_frames=20, corrupt_indices={0})
        v = DataValidator()
        report = v.validate_session(session)
        assert isinstance(report, DataValidationReport)
        assert report.total_frames == 20
        assert report.valid_frames == 19
        assert report.corrupted_frames == 1
        assert report.needs_recollection is False  # 1/20 = 5%, not > 5%

    def test_needs_recollection_boundary_5pct(self, tmp_path):
        # Exactly 5% = 1/20 → should NOT trigger (> 5%, not >=)
        session = _create_session(tmp_path, num_frames=20, corrupt_indices={0})
        v = DataValidator()
        report = v.validate_session(session)
        assert report.needs_recollection is False

    def test_needs_recollection_above_5pct(self, tmp_path):
        # 2/20 = 10% → should trigger
        session = _create_session(tmp_path, num_frames=20, corrupt_indices={0, 1})
        v = DataValidator()
        report = v.validate_session(session)
        assert report.needs_recollection is True
        assert len(report.warnings) > 0

    def test_nonexistent_session_raises(self, tmp_path):
        v = DataValidator()
        with pytest.raises(FileNotFoundError):
            v.validate_session(str(tmp_path / "nonexistent"))


class TestAnalyzeDistribution:
    def test_distribution_stats(self, tmp_path):
        steering = [0.1, -0.1, 0.2, -0.2, 0.0]
        session = _create_session(tmp_path, num_frames=5, steering_values=steering)
        v = DataValidator()
        result = v.analyze_distribution(session)
        assert abs(result["steering_mean"] - np.mean(steering)) < 1e-6
        assert abs(result["steering_std"] - np.std(steering)) < 1e-6
        assert "steering_histogram" in result
