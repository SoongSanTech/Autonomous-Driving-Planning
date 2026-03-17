"""
Unit tests for AsyncDataLogger consumer methods (Task 3.3).

Tests cover:
- start() launches background workers
- _writer_loop() writes PNG files to correct path
- _writer_loop() appends CSV rows correctly
- stop() flushes queue and terminates gracefully
- CSV file has correct data after writing multiple frames
"""

import csv
import time

import cv2
import numpy as np
import pytest

from data_pipeline.async_logger import AsyncDataLogger
from data_pipeline.models import VehicleState


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    return str(tmp_path / "test_out")


@pytest.fixture
def sample_image():
    """800x600 RGB image with non-zero content."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)


@pytest.fixture
def sample_vehicle_state():
    return VehicleState(speed=15.3, steering=0.12, throttle=0.5, brake=0.0)


class TestStart:
    """Tests for the start() method."""

    def test_start_sets_running_flag(self, output_dir):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10, num_workers=1)
        logger_inst.start()
        assert logger_inst._running is True
        logger_inst.stop()

    def test_start_creates_futures_for_each_worker(self, output_dir):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10, num_workers=3)
        logger_inst.start()
        assert len(logger_inst._futures) == 3
        logger_inst.stop()


class TestWriterLoop:
    """Tests for the _writer_loop() method — PNG and CSV writing."""

    def test_writes_png_to_correct_path(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10, num_workers=1)
        logger_inst.start()
        logger_inst.enqueue_frame(1000, sample_image, sample_vehicle_state)
        # Give the writer time to process
        time.sleep(0.5)
        logger_inst.stop()

        expected_path = logger_inst._images_dir / "1000.png"
        assert expected_path.exists(), f"Expected PNG at {expected_path}"

    def test_png_is_valid_and_decodable(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10, num_workers=1)
        logger_inst.start()
        logger_inst.enqueue_frame(2000, sample_image, sample_vehicle_state)
        time.sleep(0.5)
        logger_inst.stop()

        img_path = str(logger_inst._images_dir / "2000.png")
        decoded = cv2.imread(img_path)
        assert decoded is not None
        assert decoded.shape == (600, 800, 3)

    def test_appends_csv_row(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10, num_workers=1)
        logger_inst.start()
        logger_inst.enqueue_frame(3000, sample_image, sample_vehicle_state)
        time.sleep(0.5)
        logger_inst.stop()

        with open(logger_inst._csv_path, "r") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 2  # header + 1 data row
        row = reader[1]
        assert row[0] == "3000.png"
        assert float(row[1]) == pytest.approx(15.3)
        assert float(row[2]) == pytest.approx(0.12)
        assert float(row[3]) == pytest.approx(0.5)
        assert float(row[4]) == pytest.approx(0.0)


class TestStop:
    """Tests for the stop() method."""

    def test_stop_sets_running_false(self, output_dir):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10, num_workers=1)
        logger_inst.start()
        logger_inst.stop()
        assert logger_inst._running is False

    def test_stop_flushes_queue(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=100, num_workers=2)
        logger_inst.start()

        for ts in range(5):
            logger_inst.enqueue_frame(ts * 100, sample_image, sample_vehicle_state)

        logger_inst.stop()

        # Queue should be empty after stop
        assert logger_inst._queue.empty()

        # All 5 images should be written
        png_files = list(logger_inst._images_dir.glob("*.png"))
        assert len(png_files) == 5

    def test_stop_flushes_csv_rows(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=100, num_workers=2)
        logger_inst.start()

        for ts in range(5):
            logger_inst.enqueue_frame(ts * 100, sample_image, sample_vehicle_state)

        logger_inst.stop()

        with open(logger_inst._csv_path, "r") as f:
            reader = list(csv.reader(f))

        # header + 5 data rows
        assert len(reader) == 6


class TestMultiFrameCSV:
    """Tests for CSV correctness with multiple frames."""

    def test_csv_contains_correct_data_for_multiple_frames(self, output_dir, sample_image):
        states = [
            VehicleState(speed=10.0, steering=-0.5, throttle=0.3, brake=0.1),
            VehicleState(speed=20.0, steering=0.0, throttle=0.8, brake=0.0),
            VehicleState(speed=5.5, steering=1.0, throttle=0.0, brake=1.0),
        ]
        timestamps = [1000, 2000, 3000]

        logger_inst = AsyncDataLogger(output_dir, queue_size=100, num_workers=1)
        logger_inst.start()

        for ts, vs in zip(timestamps, states):
            logger_inst.enqueue_frame(ts, sample_image, vs)

        logger_inst.stop()

        with open(logger_inst._csv_path, "r") as f:
            reader = list(csv.reader(f))

        assert reader[0] == ["image_filename", "speed", "steering", "throttle", "brake"]
        assert len(reader) == 4  # header + 3 rows

        # Collect rows into a dict keyed by filename for order-independent check
        data_rows = {row[0]: row for row in reader[1:]}
        for ts, vs in zip(timestamps, states):
            fname = f"{ts}.png"
            assert fname in data_rows
            row = data_rows[fname]
            assert float(row[1]) == pytest.approx(vs.speed)
            assert float(row[2]) == pytest.approx(vs.steering)
            assert float(row[3]) == pytest.approx(vs.throttle)
            assert float(row[4]) == pytest.approx(vs.brake)
