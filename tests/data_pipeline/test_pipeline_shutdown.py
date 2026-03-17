"""
Unit tests for DataPipeline shutdown, signal handling, and file existence check.

Tests cover:
- shutdown() stops data logger (flushes queue)
- shutdown() destroys camera sensor
- shutdown() reports statistics
- Signal handler sets running to False
- File existence check prevents overwrite of existing data
"""

import logging
import signal
import os

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


def _make_pipeline(**kwargs):
    """Create a DataPipeline with mocked CARLA and AsyncDataLogger."""
    with patch("data_pipeline.pipeline.carla"), \
         patch("data_pipeline.pipeline.AsyncDataLogger") as mock_logger_cls, \
         patch("data_pipeline.pipeline.signal"):
        mock_logger_instance = MagicMock()
        mock_logger_instance.frame_drops = 0
        mock_logger_cls.return_value = mock_logger_instance

        from data_pipeline.pipeline import DataPipeline
        pipeline = DataPipeline(**kwargs)

    return pipeline


@pytest.mark.unit
class TestShutdown:
    """Tests for DataPipeline.shutdown()."""

    def test_shutdown_stops_data_logger(self):
        """Verify shutdown calls data_logger.stop() to flush the queue."""
        pipeline = _make_pipeline()
        pipeline.data_logger = MagicMock()
        pipeline.camera = None

        pipeline.shutdown()

        pipeline.data_logger.stop.assert_called_once()

    def test_shutdown_destroys_camera_sensor(self):
        """Verify shutdown stops and destroys the camera sensor."""
        pipeline = _make_pipeline()
        pipeline.data_logger = MagicMock()
        mock_camera = MagicMock()
        pipeline.camera = mock_camera

        pipeline.shutdown()

        mock_camera.stop.assert_called_once()
        mock_camera.destroy.assert_called_once()

    def test_shutdown_skips_camera_when_none(self):
        """Verify shutdown handles missing camera gracefully."""
        pipeline = _make_pipeline()
        pipeline.data_logger = MagicMock()
        pipeline.camera = None

        # Should not raise
        pipeline.shutdown()

    def test_shutdown_sets_running_false(self):
        """Verify shutdown sets running flag to False."""
        pipeline = _make_pipeline()
        pipeline.data_logger = MagicMock()
        pipeline.camera = None
        pipeline.running = True

        pipeline.shutdown()

        assert pipeline.running is False

    def test_shutdown_reports_statistics(self, caplog):
        """Verify shutdown logs frames saved and drops."""
        pipeline = _make_pipeline()
        pipeline.data_logger = MagicMock()
        pipeline.camera = None
        pipeline.frames_captured = 42
        pipeline.frame_drops = 3

        with caplog.at_level(logging.INFO, logger="data_pipeline.pipeline"):
            pipeline.shutdown()

        assert "42 frames saved" in caplog.text
        assert "3 drops" in caplog.text

    def test_shutdown_logs_initiation(self, caplog):
        """Verify shutdown logs the initiation message."""
        pipeline = _make_pipeline()
        pipeline.data_logger = MagicMock()
        pipeline.camera = None

        with caplog.at_level(logging.INFO, logger="data_pipeline.pipeline"):
            pipeline.shutdown()

        assert "Initiating graceful shutdown" in caplog.text

    def test_shutdown_skips_logger_when_none(self):
        """Verify shutdown handles missing data_logger gracefully."""
        pipeline = _make_pipeline()
        pipeline.data_logger = None
        pipeline.camera = None

        # Should not raise
        pipeline.shutdown()


@pytest.mark.unit
class TestSignalHandler:
    """Tests for signal handler registration and behavior."""

    def test_signal_handler_sets_running_false(self):
        """Verify _signal_handler sets running to False."""
        pipeline = _make_pipeline()
        pipeline.running = True

        pipeline._signal_handler(signal.SIGINT, None)

        assert pipeline.running is False

    def test_signal_handler_logs_signal_number(self, caplog):
        """Verify _signal_handler logs the received signal number."""
        pipeline = _make_pipeline()
        pipeline.running = True

        with caplog.at_level(logging.INFO, logger="data_pipeline.pipeline"):
            pipeline._signal_handler(signal.SIGTERM, None)

        assert "Signal" in caplog.text
        assert "shutting down" in caplog.text

    def test_signal_handlers_registered_on_init(self):
        """Verify SIGINT and SIGTERM handlers are registered during __init__."""
        with patch("data_pipeline.pipeline.carla"), \
             patch("data_pipeline.pipeline.AsyncDataLogger"), \
             patch("data_pipeline.pipeline.signal") as mock_signal:
            from data_pipeline.pipeline import DataPipeline
            pipeline = DataPipeline()

        # signal.signal should be called for SIGINT and SIGTERM
        calls = mock_signal.signal.call_args_list
        registered_signals = [c[0][0] for c in calls]
        assert mock_signal.SIGINT in registered_signals
        assert mock_signal.SIGTERM in registered_signals


@pytest.mark.unit
class TestOutputDirResolution:
    """Tests for daytime-based output directory resolution."""

    def test_output_dir_uses_daytime_subfolder(self, tmp_path):
        """Verify output_dir is resolved to a daytime-based subfolder."""
        base_dir = str(tmp_path / "data")
        with patch("data_pipeline.pipeline.carla"), \
             patch("data_pipeline.pipeline.AsyncDataLogger"), \
             patch("data_pipeline.pipeline.signal"):
            from data_pipeline.pipeline import DataPipeline
            pipeline = DataPipeline(output_dir=base_dir)

        # output_dir should be base_dir/{YYYY-MM-DD_HHMMSS}
        assert pipeline.output_dir.startswith(base_dir + os.sep) or \
               pipeline.output_dir.startswith(base_dir + "/")
        subfolder = pipeline.output_dir[len(base_dir) + 1:]
        # Verify daytime format: YYYY-MM-DD_HHMMSS
        assert len(subfolder) == 17  # e.g. 2026-03-13_143022
        assert subfolder[4] == "-"
        assert subfolder[7] == "-"
        assert subfolder[10] == "_"

    def test_multiple_runs_get_different_subfolders(self, tmp_path):
        """Verify consecutive runs create different daytime subfolders."""
        import time as _time
        base_dir = str(tmp_path / "data")

        with patch("data_pipeline.pipeline.carla"), \
             patch("data_pipeline.pipeline.AsyncDataLogger"), \
             patch("data_pipeline.pipeline.signal"):
            from data_pipeline.pipeline import DataPipeline
            p1 = DataPipeline(output_dir=base_dir)
            _time.sleep(1.1)  # Ensure different second
            p2 = DataPipeline(output_dir=base_dir)

        assert p1.output_dir != p2.output_dir

    def test_creates_data_logger_with_resolved_dir(self, tmp_path):
        """Verify AsyncDataLogger is created once with the resolved daytime dir."""
        base_dir = str(tmp_path / "data")
        with patch("data_pipeline.pipeline.carla"), \
             patch("data_pipeline.pipeline.AsyncDataLogger") as mock_logger_cls, \
             patch("data_pipeline.pipeline.signal"):
            from data_pipeline.pipeline import DataPipeline
            pipeline = DataPipeline(output_dir=base_dir)

        # Called once in _resolve_output_dir with daytime subfolder
        assert mock_logger_cls.call_count == 1
        call_output = mock_logger_cls.call_args_list[0][1]["output_dir"]
        assert call_output == pipeline.output_dir
        assert base_dir in call_output
