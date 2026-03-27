"""
Unit tests for AsyncDataLogger.enqueue_frame() (Task 3.2).

Tests cover:
- Successful enqueue creates FrameData and places it on the queue
- Queue overflow increments frame_drops counter
- Warning logged when queue reaches 90% capacity
"""

import logging
import queue
import tempfile

import numpy as np
import pytest

from data_pipeline.async_logger import AsyncDataLogger
from data_pipeline.models import FrameData, VehicleState


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    return str(tmp_path / "test_out")


@pytest.fixture
def sample_image():
    """800x600 RGB image."""
    return np.zeros((600, 800, 3), dtype=np.uint8)


@pytest.fixture
def sample_vehicle_state():
    return VehicleState(speed=10.0, steering=0.5, throttle=0.7, brake=0.0)


class TestEnqueueFrame:
    """Tests for the enqueue_frame producer method."""

    def test_enqueue_places_frame_on_queue(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10)
        logger_inst.enqueue_frame(1000, sample_image, sample_vehicle_state)

        assert logger_inst._queue.qsize() == 1
        frame = logger_inst._queue.get_nowait()
        assert isinstance(frame, FrameData)
        assert frame.timestamp_ms == 1000
        assert frame.frame_id == 1000
        assert np.array_equal(frame.image, sample_image)
        assert frame.vehicle_state == sample_vehicle_state

    def test_enqueue_multiple_frames(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10)
        for ts in range(5):
            logger_inst.enqueue_frame(ts * 100, sample_image, sample_vehicle_state)

        assert logger_inst._queue.qsize() == 5
        assert logger_inst.frame_drops == 0

    def test_queue_overflow_increments_frame_drops(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=2)
        # Fill the queue
        logger_inst.enqueue_frame(100, sample_image, sample_vehicle_state)
        logger_inst.enqueue_frame(200, sample_image, sample_vehicle_state)
        # This should overflow
        logger_inst.enqueue_frame(300, sample_image, sample_vehicle_state)

        assert logger_inst.frame_drops == 1
        assert logger_inst._queue.qsize() == 2  # Queue still at max

    def test_multiple_overflows_tracked(self, output_dir, sample_image, sample_vehicle_state):
        logger_inst = AsyncDataLogger(output_dir, queue_size=1)
        logger_inst.enqueue_frame(100, sample_image, sample_vehicle_state)
        logger_inst.enqueue_frame(200, sample_image, sample_vehicle_state)
        logger_inst.enqueue_frame(300, sample_image, sample_vehicle_state)

        assert logger_inst.frame_drops == 2

    def test_capacity_warning_at_90_percent(self, output_dir, sample_image, sample_vehicle_state, caplog):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10)
        # Fill to 9 items (90% of 10)
        for i in range(9):
            logger_inst.enqueue_frame(i * 100, sample_image, sample_vehicle_state)

        # The 10th enqueue should trigger the warning (queue is at 9/10 = 90%)
        with caplog.at_level(logging.WARNING, logger="data_pipeline.async_logger"):
            logger_inst.enqueue_frame(900, sample_image, sample_vehicle_state)

        assert any("capacity" in record.message.lower() for record in caplog.records)

    def test_no_warning_below_90_percent(self, output_dir, sample_image, sample_vehicle_state, caplog):
        logger_inst = AsyncDataLogger(output_dir, queue_size=10)
        # Fill to 8 items (80% of 10) — below threshold
        with caplog.at_level(logging.WARNING, logger="data_pipeline.async_logger"):
            for i in range(8):
                logger_inst.enqueue_frame(i * 100, sample_image, sample_vehicle_state)

        assert not any("capacity" in record.message.lower() for record in caplog.records)

    def test_overflow_logs_warning(self, output_dir, sample_image, sample_vehicle_state, caplog):
        logger_inst = AsyncDataLogger(output_dir, queue_size=1)
        logger_inst.enqueue_frame(100, sample_image, sample_vehicle_state)

        with caplog.at_level(logging.WARNING, logger="data_pipeline.async_logger"):
            logger_inst.enqueue_frame(200, sample_image, sample_vehicle_state)

        assert any("overflow" in record.message.lower() for record in caplog.records)
