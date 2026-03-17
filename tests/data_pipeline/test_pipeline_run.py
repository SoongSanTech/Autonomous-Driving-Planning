"""
Unit tests for DataPipeline.run() method.

Tests cover:
- Collection loop runs for specified duration
- CARLA crash detection stops collection and preserves data
- Episode reset occurs at correct intervals
- Headless mode logs progress every 100 frames
- Frame data is enqueued correctly with vehicle state
"""

import logging

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_pipeline_with_mocks(headless=True):
    """Create a DataPipeline with all subsystems mocked out."""
    with patch("data_pipeline.pipeline.carla"), \
         patch("data_pipeline.pipeline.AsyncDataLogger"):
        from data_pipeline.pipeline import DataPipeline
        pipeline = DataPipeline(headless=headless)

    pipeline.sync_controller = MagicMock()
    pipeline.data_logger = MagicMock()
    pipeline.data_logger.frame_drops = 0
    pipeline.episode_manager = MagicMock()
    pipeline.episode_manager.should_reset_episode.return_value = False

    mock_vehicle = MagicMock()
    mock_velocity = MagicMock()
    mock_velocity.x, mock_velocity.y, mock_velocity.z = 3.0, 4.0, 0.0
    mock_vehicle.get_velocity.return_value = mock_velocity
    mock_control = MagicMock()
    mock_control.steer = 0.1
    mock_control.throttle = 0.5
    mock_control.brake = 0.0
    mock_vehicle.get_control.return_value = mock_control

    mock_actors = MagicMock()
    mock_actors.filter.return_value = [mock_vehicle]
    mock_world = MagicMock()
    mock_world.get_actors.return_value = mock_actors
    pipeline.world = mock_world
    pipeline.vehicle = mock_vehicle

    pipeline.latest_image = np.zeros((600, 800, 3), dtype=np.uint8)
    pipeline.sync_controller.tick.return_value = 1
    pipeline.sync_controller.get_timestamp_ms.return_value = 100

    return pipeline, mock_vehicle



def _stop_before_tick_n(pipeline, n):
    """Tick side_effect: sets running=False BEFORE tick n executes.

    This means ticks 1..n-1 complete normally (capturing frames),
    and on tick n, running is set to False first. Since the while
    loop checks running at the top, tick n never happens.

    Actually, the running flag is checked at `while self.running`,
    and tick happens AFTER the elapsed/episode checks. So we set
    running=False on tick n, but the current iteration still
    completes. The while check catches it on the NEXT iteration.

    To get exactly N frames, we raise RuntimeError on tick N+1
    or set running=False on tick N (which gives N frames since
    the Nth tick still completes, then while exits).

    Wait - let's trace: if we set running=False inside tick call N,
    the tick returns, frame is captured (frame N), then while
    checks self.running -> False -> exits. So N ticks = N frames.
    """
    count = [0]

    def tick():
        count[0] += 1
        if count[0] >= n + 1:
            # This tick shouldn't produce a frame
            pipeline.running = False
            raise RuntimeError("stop")
        if count[0] == n:
            # Last tick that produces a frame; set running=False
            # so the while loop exits after this iteration
            pipeline.running = False
        return count[0]

    return tick


@pytest.mark.unit
class TestRunDuration:
    """Tests for run() duration control."""

    def test_run_stops_after_n_frames(self):
        """Verify run() captures exactly N frames when stopped."""
        pipeline, _ = _make_pipeline_with_mocks()
        num_frames = 5

        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(
            pipeline, num_frames
        )

        pipeline.run(duration_sec=3600.0)

        assert pipeline.frames_captured == num_frames
        assert pipeline.running is False

    def test_run_initializes_subsystems(self):
        """Verify run() enables sync mode, starts logger, and starts episode."""
        pipeline, _ = _make_pipeline_with_mocks()
        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(pipeline, 0)

        pipeline.run(duration_sec=3600.0)

        pipeline.sync_controller.enable_synchronous_mode.assert_called_once()
        pipeline.data_logger.start.assert_called_once()
        pipeline.episode_manager.start_new_episode.assert_called()

    def test_run_respects_duration_via_time(self):
        """Verify run() exits when elapsed time exceeds duration."""
        pipeline, _ = _make_pipeline_with_mocks()

        with patch("data_pipeline.pipeline.time") as mock_time:
            base = 1000.0
            # start_time=base, first elapsed check exceeds duration
            mock_time.time.side_effect = [base, base + 10.0, base + 10.0]
            pipeline.run(duration_sec=1.0)

        assert pipeline.frames_captured == 0



@pytest.mark.unit
class TestCrashDetection:
    """Tests for CARLA server crash detection."""

    def test_crash_detected_on_tick_runtime_error(self, caplog):
        """Verify RuntimeError from tick() triggers crash handling."""
        pipeline, _ = _make_pipeline_with_mocks()
        pipeline.sync_controller.tick.side_effect = RuntimeError("timeout")

        with caplog.at_level(logging.ERROR, logger="data_pipeline.pipeline"):
            pipeline.run(duration_sec=3600.0)

        assert pipeline.running is False
        assert "CARLA server crash detected" in caplog.text

    def test_crash_preserves_frame_count(self, caplog):
        """Verify frames captured before crash are reported."""
        pipeline, _ = _make_pipeline_with_mocks()
        tick_count = [0]

        def tick_with_crash():
            tick_count[0] += 1
            if tick_count[0] > 3:
                raise RuntimeError("server crashed")
            return tick_count[0]

        pipeline.sync_controller.tick.side_effect = tick_with_crash

        with caplog.at_level(logging.INFO, logger="data_pipeline.pipeline"):
            pipeline.run(duration_sec=3600.0)

        assert pipeline.frames_captured == 3
        assert "3 frames saved before crash" in caplog.text


@pytest.mark.unit
class TestEpisodeReset:
    """Tests for episode reset logic."""

    def test_episode_resets_when_duration_elapsed(self):
        """Verify episode resets when should_reset_episode returns True."""
        pipeline, _ = _make_pipeline_with_mocks()
        check_count = [0]

        def should_reset(elapsed):
            check_count[0] += 1
            return check_count[0] == 2

        pipeline.episode_manager.should_reset_episode.side_effect = should_reset
        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(pipeline, 3)

        pipeline.run(duration_sec=3600.0)

        # start_new_episode: once at init + once on reset
        assert pipeline.episode_manager.start_new_episode.call_count == 2



@pytest.mark.unit
class TestHeadlessLogging:
    """Tests for headless mode progress logging."""

    def test_headless_logs_every_100_frames(self, caplog):
        """Verify progress is logged every 100 frames in headless mode."""
        pipeline, _ = _make_pipeline_with_mocks(headless=True)
        num_frames = 200

        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(
            pipeline, num_frames
        )

        with caplog.at_level(logging.INFO, logger="data_pipeline.pipeline"):
            pipeline.run(duration_sec=3600.0)

        assert pipeline.frames_captured == num_frames
        progress_logs = [r for r in caplog.records if "Progress:" in r.message]
        assert len(progress_logs) == 2  # at frame 100 and 200

    def test_non_headless_no_progress_logs(self, caplog):
        """Verify no progress logging when headless is False."""
        pipeline, _ = _make_pipeline_with_mocks(headless=False)
        num_frames = 150

        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(
            pipeline, num_frames
        )

        with caplog.at_level(logging.INFO, logger="data_pipeline.pipeline"):
            pipeline.run(duration_sec=3600.0)

        assert pipeline.frames_captured == num_frames
        progress_logs = [r for r in caplog.records if "Progress:" in r.message]
        assert len(progress_logs) == 0


@pytest.mark.unit
class TestFrameEnqueue:
    """Tests for frame data enqueue correctness."""

    def test_frame_enqueued_with_correct_vehicle_state(self):
        """Verify enqueue_frame is called with correct vehicle state data."""
        pipeline, _ = _make_pipeline_with_mocks()
        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(pipeline, 1)

        pipeline.run(duration_sec=3600.0)

        pipeline.data_logger.enqueue_frame.assert_called_once()
        call_args = pipeline.data_logger.enqueue_frame.call_args
        timestamp_ms = call_args[0][0]
        image = call_args[0][1]
        vehicle_state = call_args[0][2]

        assert timestamp_ms == 100
        assert image.shape == (600, 800, 3)
        # speed = sqrt(3^2 + 4^2 + 0^2) = 5.0
        assert abs(vehicle_state.speed - 5.0) < 0.01
        assert vehicle_state.steering == 0.1
        assert vehicle_state.throttle == 0.5
        assert vehicle_state.brake == 0.0

    def test_no_enqueue_when_image_is_none(self):
        """Verify no frame is enqueued when latest_image is None."""
        pipeline, _ = _make_pipeline_with_mocks()
        pipeline.latest_image = None
        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(pipeline, 1)

        pipeline.run(duration_sec=3600.0)

        pipeline.data_logger.enqueue_frame.assert_not_called()
        assert pipeline.frames_captured == 0

    def test_frame_drops_tracked_from_logger(self):
        """Verify frame_drops is read from data_logger at shutdown."""
        pipeline, _ = _make_pipeline_with_mocks()
        pipeline.data_logger.frame_drops = 5
        pipeline.sync_controller.tick.side_effect = _stop_before_tick_n(pipeline, 0)

        pipeline.run(duration_sec=3600.0)

        assert pipeline.frame_drops == 5

    def test_running_flag_set_and_cleared(self):
        """Verify running flag is True during collection and False after."""
        pipeline, _ = _make_pipeline_with_mocks()
        running_during = []

        def capture_running_tick():
            running_during.append(pipeline.running)
            pipeline.running = False
            return 1

        pipeline.sync_controller.tick.side_effect = capture_running_tick

        pipeline.run(duration_sec=3600.0)

        assert all(running_during), "running should be True during collection"
        assert pipeline.running is False, "running should be False after collection"
