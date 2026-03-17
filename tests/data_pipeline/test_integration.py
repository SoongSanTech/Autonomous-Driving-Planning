"""
Integration test for a 10-second data collection session (Task 8.1).

This test simulates the full pipeline flow with mocked CARLA interactions
but a real AsyncDataLogger writing actual PNG and CSV files to disk.
It verifies:
- PNG files are created in {output}/images/
- driving_log.csv is created in {output}/labels/
- File counts match expected frame count (~100 frames at 10Hz for 10s)
- CSV has correct headers and row count

Requirements: 2.1, 3.1, 3.2, 3.3, 4.1, 4.2
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from data_pipeline.async_logger import AsyncDataLogger
from data_pipeline.models import VehicleState


TICK_RATE_HZ = 10.0
DURATION_SEC = 10.0
EXPECTED_FRAMES = int(TICK_RATE_HZ * DURATION_SEC)  # 100
FIXED_DELTA_MS = int(1000 / TICK_RATE_HZ)  # 100ms per tick


@pytest.fixture
def sample_image():
    """800x600 RGB image with random content."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)


@pytest.mark.integration
class TestTenSecondCollectionSession:
    """
    Integration test simulating a 10-second collection session.

    Uses a real AsyncDataLogger to write actual files to a temporary
    directory while simulating the tick loop that would normally be
    driven by the CARLA synchronous mode controller.
    """

    def test_full_pipeline_produces_expected_output(self, tmp_path, sample_image):
        """
        Simulate 10 seconds of collection at 10Hz and verify all outputs.

        Validates: Requirements 2.1, 3.1, 3.2, 3.3, 4.1, 4.2
        """
        output_dir = str(tmp_path)
        logger_inst = AsyncDataLogger(
            output_dir=output_dir,
            queue_size=200,
            num_workers=2,
            png_compression=3,
        )
        logger_inst.start()

        # Simulate 10 seconds at 10Hz — enqueue 100 frames
        for i in range(EXPECTED_FRAMES):
            timestamp_ms = (i + 1) * FIXED_DELTA_MS  # 100, 200, ..., 10000
            vehicle_state = VehicleState(
                speed=10.0 + (i % 10) * 0.5,
                steering=round(-0.5 + (i % 20) * 0.05, 2),
                throttle=0.5,
                brake=0.0,
            )
            logger_inst.enqueue_frame(timestamp_ms, sample_image, vehicle_state)

        # Graceful shutdown — flushes the queue
        logger_inst.stop()

        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        csv_path = labels_dir / "driving_log.csv"

        # --- Verify image files ---
        png_files = sorted(images_dir.glob("*.png"))
        assert len(png_files) == EXPECTED_FRAMES, (
            f"Expected {EXPECTED_FRAMES} PNG files, got {len(png_files)}"
        )

        # Verify each file is a valid, decodable PNG with correct dimensions
        for png_file in png_files:
            decoded = cv2.imread(str(png_file))
            assert decoded is not None, f"Failed to decode {png_file.name}"
            assert decoded.shape == (600, 800, 3), (
                f"{png_file.name} has wrong shape: {decoded.shape}"
            )

        # Verify filenames follow {timestamp}.png pattern
        expected_names = {f"{(i + 1) * FIXED_DELTA_MS}.png" for i in range(EXPECTED_FRAMES)}
        actual_names = {f.name for f in png_files}
        assert actual_names == expected_names

        # --- Verify CSV file ---
        assert csv_path.exists(), "driving_log.csv not found"

        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))

        # Check headers
        assert rows[0] == ["image_filename", "speed", "steering", "throttle", "brake"]

        # Check row count (header + 100 data rows)
        data_rows = rows[1:]
        assert len(data_rows) == EXPECTED_FRAMES, (
            f"Expected {EXPECTED_FRAMES} CSV data rows, got {len(data_rows)}"
        )

        # Verify every CSV row references an existing PNG file
        csv_filenames = {row[0] for row in data_rows}
        assert csv_filenames == expected_names

        # Verify zero frame drops
        assert logger_inst.frame_drops == 0

from unittest.mock import MagicMock, patch
from data_pipeline.episode_manager import EpisodeManager, WeatherPreset, TimeOfDay


# --- Constants for episode transition test ---
# Use short episode duration (5s) so 2 episodes fit in 12s of simulated time.
# This keeps the test fast while still validating episode transitions.
EPISODE_DURATION_SEC = 5.0  # 5 seconds per episode (shortened for test speed)
TRANSITION_DURATION_SEC = 12.0  # 12 seconds total (2+ episodes)
TRANSITION_TICK_HZ = 10.0
TRANSITION_FRAMES = int(TRANSITION_TICK_HZ * TRANSITION_DURATION_SEC)  # 120
TRANSITION_DELTA_MS = int(1000 / TRANSITION_TICK_HZ)  # 100ms


def _make_mock_carla_world():
    """Create a mock CARLA world that supports weather get/set operations."""
    world = MagicMock()
    # Mutable weather object that get_weather returns by reference
    current_weather = MagicMock()
    current_weather.sun_altitude_angle = 0.0
    world.get_weather.return_value = current_weather
    world.set_weather.return_value = None
    return world


@pytest.mark.integration
class TestEpisodeTransitionSession:
    """
    Integration test simulating a 12-second collection session spanning
    2+ episodes (with shortened 5-second episode duration).

    Uses a real EpisodeManager and AsyncDataLogger with a mocked CARLA
    world to verify seamless weather/time transitions and zero frame
    drops during episode resets.

    Validates: Requirements 5.5, 5.6, 6.4
    """

    @patch("data_pipeline.episode_manager.carla")
    def test_episode_transitions_produce_diverse_conditions(
        self, mock_carla_module, tmp_path
    ):
        """
        Simulate 12 seconds of collection at 10Hz across 2+ episodes.

        Verifies:
        - At least 2 episodes occurred (weather_history >= 2 entries)
        - At least 2 different weather or time conditions were applied
        - All 120 frames captured with zero drops
        - All PNG files and CSV rows are present
        """
        # Small image to keep I/O fast for 3600 frames
        small_image = np.zeros((600, 800, 3), dtype=np.uint8)

        # --- Setup mock CARLA module ---
        # carla.WeatherParameters.<preset> must be resolvable via getattr
        mock_weather_params = MagicMock()
        mock_carla_module.WeatherParameters = mock_weather_params

        mock_world = _make_mock_carla_world()

        # Seed random for deterministic episode variation
        import random as _random
        _random.seed(42)

        # --- Create real EpisodeManager ---
        episode_mgr = EpisodeManager(
            world=mock_world,
            episode_duration_sec=EPISODE_DURATION_SEC,
        )

        # --- Create real AsyncDataLogger ---
        output_dir = str(tmp_path)
        logger_inst = AsyncDataLogger(
            output_dir=output_dir,
            queue_size=2000,
            num_workers=2,
            png_compression=1,  # Fastest compression for test speed
        )
        logger_inst.start()

        # --- Start first episode ---
        episode_mgr.start_new_episode()
        episode_start_sec = 0.0

        # --- Simulate 6 minutes at 10Hz ---
        # Enqueue in batches with backpressure to avoid queue overflow.
        # The real pipeline runs at 10Hz (one frame per 100ms), giving
        # the writer threads time to drain. We simulate this by waiting
        # for the queue to have room before enqueuing each batch.
        import time

        for i in range(TRANSITION_FRAMES):
            elapsed_total_sec = (i + 1) / TRANSITION_TICK_HZ
            elapsed_in_episode = elapsed_total_sec - episode_start_sec

            # Check for episode reset
            if episode_mgr.should_reset_episode(elapsed_in_episode):
                episode_mgr.start_new_episode()
                episode_start_sec = elapsed_total_sec

            # Wait for queue space before enqueuing (backpressure)
            while logger_inst._queue.qsize() >= logger_inst.queue_size - 10:
                time.sleep(0.01)

            # Enqueue frame
            timestamp_ms = (i + 1) * TRANSITION_DELTA_MS
            vehicle_state = VehicleState(
                speed=10.0 + (i % 10) * 0.5,
                steering=round(-0.5 + (i % 20) * 0.05, 2),
                throttle=0.5,
                brake=0.0,
            )
            logger_inst.enqueue_frame(timestamp_ms, small_image, vehicle_state)

        # --- Graceful shutdown ---
        logger_inst.stop()

        # --- Verify at least 2 episodes occurred ---
        assert len(episode_mgr._weather_history) >= 2, (
            f"Expected >= 2 weather entries, got {len(episode_mgr._weather_history)}"
        )
        assert len(episode_mgr._time_history) >= 2, (
            f"Expected >= 2 time entries, got {len(episode_mgr._time_history)}"
        )

        # --- Verify at least 2 different conditions were applied ---
        unique_weather = set(episode_mgr._weather_history)
        unique_time = set(episode_mgr._time_history)
        # With random selection from 3 options over 2+ episodes, we expect
        # diversity in at least one dimension (weather or time of day).
        has_weather_diversity = len(unique_weather) >= 2
        has_time_diversity = len(unique_time) >= 2
        assert has_weather_diversity or has_time_diversity, (
            f"Expected diversity: unique weather={unique_weather}, "
            f"unique time={unique_time}. "
            f"At least 2 different weather OR time conditions should appear."
        )

        # --- Verify all frames captured (no drops during episode reset) ---
        assert logger_inst.frame_drops == 0, (
            f"Expected zero frame drops, got {logger_inst.frame_drops}"
        )

        # --- Verify PNG files ---
        images_dir = tmp_path / "images"
        png_files = sorted(images_dir.glob("*.png"))
        assert len(png_files) == TRANSITION_FRAMES, (
            f"Expected {TRANSITION_FRAMES} PNG files, got {len(png_files)}"
        )

        expected_names = {
            f"{(i + 1) * TRANSITION_DELTA_MS}.png"
            for i in range(TRANSITION_FRAMES)
        }
        actual_names = {f.name for f in png_files}
        assert actual_names == expected_names

        # --- Verify CSV rows ---
        labels_dir = tmp_path / "labels"
        csv_path = labels_dir / "driving_log.csv"
        assert csv_path.exists(), "driving_log.csv not found"

        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "image_filename", "speed", "steering", "throttle", "brake"
        ]

        data_rows = rows[1:]
        assert len(data_rows) == TRANSITION_FRAMES, (
            f"Expected {TRANSITION_FRAMES} CSV data rows, got {len(data_rows)}"
        )

        csv_filenames = {row[0] for row in data_rows}
        assert csv_filenames == expected_names
