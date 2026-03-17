"""
Unit tests for SynchronousModeController.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock
from data_pipeline.sync_controller import SynchronousModeController


def _make_mock_world(elapsed_seconds: float = 0.0):
    """Create a mock CARLA world with configurable snapshot time."""
    world = MagicMock()

    # WorldSettings mock
    settings = MagicMock()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.0
    world.get_settings.return_value = settings

    # tick returns a frame id
    world.tick.return_value = 1

    # snapshot / timestamp
    timestamp = MagicMock()
    timestamp.elapsed_seconds = elapsed_seconds
    snapshot = MagicMock()
    snapshot.timestamp = timestamp
    world.get_snapshot.return_value = snapshot

    return world, settings


@pytest.mark.unit
class TestSynchronousModeController:
    """Tests for SynchronousModeController."""

    def test_default_tick_rate(self):
        """Verify default tick rate is 10 Hz."""
        world, _ = _make_mock_world()
        ctrl = SynchronousModeController(world)

        assert ctrl._tick_rate_hz == 10.0
        assert ctrl._fixed_delta_seconds == pytest.approx(0.1)

    def test_custom_tick_rate(self):
        """Verify custom tick rate is accepted."""
        world, _ = _make_mock_world()
        ctrl = SynchronousModeController(world, tick_rate_hz=20.0)

        assert ctrl._tick_rate_hz == 20.0
        assert ctrl._fixed_delta_seconds == pytest.approx(0.05)

    def test_enable_synchronous_mode_applies_settings(self):
        """Verify enable_synchronous_mode sets correct world settings."""
        world, settings = _make_mock_world()
        ctrl = SynchronousModeController(world)

        ctrl.enable_synchronous_mode()

        # Settings should have been mutated and applied
        assert settings.synchronous_mode is True
        assert settings.fixed_delta_seconds == pytest.approx(0.1)
        world.apply_settings.assert_called_once_with(settings)

    def test_tick_returns_frame_id(self):
        """Verify tick() advances simulation and returns frame_id."""
        world, _ = _make_mock_world()
        world.tick.return_value = 42
        ctrl = SynchronousModeController(world)

        frame_id = ctrl.tick()

        assert frame_id == 42
        world.tick.assert_called_once()

    def test_get_timestamp_ms_zero(self):
        """Verify get_timestamp_ms returns 0 at simulation start."""
        world, _ = _make_mock_world(elapsed_seconds=0.0)
        ctrl = SynchronousModeController(world)

        assert ctrl.get_timestamp_ms() == 0

    def test_get_timestamp_ms_precision(self):
        """Verify get_timestamp_ms returns integer milliseconds."""
        world, _ = _make_mock_world(elapsed_seconds=1.5)
        ctrl = SynchronousModeController(world)

        ts = ctrl.get_timestamp_ms()
        assert isinstance(ts, int)
        assert ts == 1500

    def test_get_timestamp_ms_fractional(self):
        """Verify sub-millisecond values are truncated (not rounded)."""
        world, _ = _make_mock_world(elapsed_seconds=0.1009)
        ctrl = SynchronousModeController(world)

        ts = ctrl.get_timestamp_ms()
        assert isinstance(ts, int)
        assert ts == 100  # int(100.9) == 100

    def test_get_timestamp_ms_large_value(self):
        """Verify timestamp works for long sessions (1 hour)."""
        world, _ = _make_mock_world(elapsed_seconds=3600.0)
        ctrl = SynchronousModeController(world)

        ts = ctrl.get_timestamp_ms()
        assert ts == 3_600_000
