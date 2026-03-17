"""
Unit tests for DataPipeline connect() and setup_sensors() methods.

Tests cover:
- Successful connection on first attempt
- Connection with retry (fail then succeed)
- Connection failure after max retries raises ConnectionError
- WSL2 host IP handling
- setup_sensors configures correct camera resolution (800x600)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call


def _make_mock_carla_client(world=None):
    """Create a mock carla.Client that returns a mock world."""
    client = MagicMock()
    if world is None:
        world = MagicMock()
    client.get_world.return_value = world
    return client, world


def _make_mock_world_with_vehicle():
    """Create a mock CARLA world with a vehicle and blueprint library."""
    world = MagicMock()

    # Blueprint library setup
    camera_bp = MagicMock()
    blueprint_library = MagicMock()
    blueprint_library.find.return_value = camera_bp
    world.get_blueprint_library.return_value = blueprint_library

    # Vehicle actor
    vehicle = MagicMock()
    actors = MagicMock()
    actors.filter.return_value = [vehicle]
    actors.__len__ = lambda self: 1
    world.get_actors.return_value = actors

    # spawn_actor returns a mock camera
    mock_camera = MagicMock()
    world.spawn_actor.return_value = mock_camera

    return world, camera_bp, vehicle, mock_camera



@pytest.mark.unit
class TestConnect:
    """Tests for DataPipeline.connect()."""

    @patch("data_pipeline.pipeline.carla")
    @patch("data_pipeline.pipeline.time")
    def test_successful_connection_first_attempt(self, mock_time, mock_carla):
        """Verify connect succeeds on first try and initializes subsystems."""
        mock_client, mock_world = _make_mock_carla_client()
        mock_carla.Client.return_value = mock_client

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(carla_host="localhost", carla_port=2000)
        pipeline.connect()

        mock_carla.Client.assert_called_once_with("localhost", 2000)
        mock_client.set_timeout.assert_called_once_with(10.0)
        assert pipeline.client is mock_client
        assert pipeline.world is mock_world
        assert pipeline.sync_controller is not None
        assert pipeline.episode_manager is not None
        # No sleep should be called on first success
        mock_time.sleep.assert_not_called()

    @patch("data_pipeline.pipeline.carla")
    @patch("data_pipeline.pipeline.time")
    def test_connection_retry_then_succeed(self, mock_time, mock_carla):
        """Verify connect retries on failure and succeeds on later attempt."""
        mock_client, mock_world = _make_mock_carla_client()

        # Fail twice, then succeed
        mock_carla.Client.side_effect = [
            RuntimeError("Connection refused"),
            RuntimeError("Connection refused"),
            mock_client,
        ]

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.connect()

        assert mock_carla.Client.call_count == 3
        assert pipeline.client is mock_client
        assert pipeline.world is mock_world
        # Exponential backoff: 2^0=1s, 2^1=2s
        assert mock_time.sleep.call_args_list == [call(1), call(2)]

    @patch("data_pipeline.pipeline.carla")
    @patch("data_pipeline.pipeline.time")
    def test_connection_failure_after_max_retries(self, mock_time, mock_carla):
        """Verify ConnectionError raised after 5 failed attempts."""
        mock_carla.Client.side_effect = RuntimeError("Connection refused")

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()

        with pytest.raises(ConnectionError, match="Failed to connect to CARLA"):
            pipeline.connect()

        assert mock_carla.Client.call_count == 5
        # 4 sleeps (attempts 0-3 sleep, attempt 4 raises)
        assert mock_time.sleep.call_count == 4

    @patch("data_pipeline.pipeline.carla")
    @patch("data_pipeline.pipeline.time")
    def test_connection_error_message_includes_host_port(self, mock_time, mock_carla):
        """Verify error message contains host and port for debugging."""
        mock_carla.Client.side_effect = RuntimeError("Connection refused")

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(carla_host="192.168.1.100", carla_port=2000)

        with pytest.raises(
            ConnectionError, match="192.168.1.100:2000"
        ):
            pipeline.connect()

    @patch("data_pipeline.pipeline.carla")
    @patch("data_pipeline.pipeline.time")
    def test_wsl2_explicit_host_ip(self, mock_time, mock_carla):
        """Verify WSL2 host IP is passed through to carla.Client."""
        mock_client, _ = _make_mock_carla_client()
        mock_carla.Client.return_value = mock_client

        from data_pipeline.pipeline import DataPipeline

        wsl2_host = "172.22.160.1"
        pipeline = DataPipeline(carla_host=wsl2_host, carla_port=2000)
        pipeline.connect()

        mock_carla.Client.assert_called_once_with(wsl2_host, 2000)

    @patch("data_pipeline.pipeline.carla")
    @patch("data_pipeline.pipeline.time")
    def test_exponential_backoff_timing(self, mock_time, mock_carla):
        """Verify exponential backoff waits: 1s, 2s, 4s, 8s."""
        mock_client, _ = _make_mock_carla_client()

        # Fail 4 times, succeed on 5th
        mock_carla.Client.side_effect = [
            RuntimeError("fail"),
            RuntimeError("fail"),
            RuntimeError("fail"),
            RuntimeError("fail"),
            mock_client,
        ]

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.connect()

        expected_waits = [call(1), call(2), call(4), call(8)]
        assert mock_time.sleep.call_args_list == expected_waits


@pytest.mark.unit
class TestSetupSensors:
    """Tests for DataPipeline.setup_sensors()."""

    @patch("data_pipeline.pipeline.carla")
    def test_camera_resolution_800x600(self, mock_carla):
        """Verify camera is configured with 800x600 resolution."""
        world, camera_bp, vehicle, mock_camera = _make_mock_world_with_vehicle()

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.world = world

        pipeline.setup_sensors()

        camera_bp.set_attribute.assert_any_call("image_size_x", "800")
        camera_bp.set_attribute.assert_any_call("image_size_y", "600")

    @patch("data_pipeline.pipeline.carla")
    def test_camera_attached_to_vehicle(self, mock_carla):
        """Verify camera is spawned and attached to the ego vehicle."""
        world, camera_bp, vehicle, mock_camera = _make_mock_world_with_vehicle()

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.world = world

        pipeline.setup_sensors()

        world.spawn_actor.assert_called_once()
        spawn_args = world.spawn_actor.call_args
        assert spawn_args[0][0] is camera_bp  # blueprint
        assert spawn_args[1].get("attach_to") or spawn_args[0][2] is vehicle

    @patch("data_pipeline.pipeline.carla")
    def test_camera_listen_callback_registered(self, mock_carla):
        """Verify camera.listen() is called with a callback."""
        world, camera_bp, vehicle, mock_camera = _make_mock_world_with_vehicle()

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.world = world

        pipeline.setup_sensors()

        mock_camera.listen.assert_called_once()
        assert pipeline.camera is mock_camera

    @patch("data_pipeline.pipeline.carla")
    def test_no_vehicle_no_spawn_points_raises_error(self, mock_carla):
        """Verify RuntimeError when no vehicle exists and no spawn points available."""
        world = MagicMock()
        blueprint_library = MagicMock()
        world.get_blueprint_library.return_value = blueprint_library

        # No existing vehicles
        actors = MagicMock()
        actors.filter.return_value = []
        actors.__len__ = lambda self: 0
        actors.__bool__ = lambda self: False
        world.get_actors.return_value = actors

        # No spawn points
        mock_map = MagicMock()
        mock_map.get_spawn_points.return_value = []
        world.get_map.return_value = mock_map

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.world = world

        with pytest.raises(RuntimeError, match="No spawn points"):
            pipeline.setup_sensors()

    @patch("data_pipeline.pipeline.carla")
    def test_camera_callback_converts_image(self, mock_carla):
        """Verify camera callback converts BGRA to BGR numpy array."""
        world, camera_bp, vehicle, mock_camera = _make_mock_world_with_vehicle()

        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline()
        pipeline.world = world

        pipeline.setup_sensors()

        # Get the callback that was registered
        callback = mock_camera.listen.call_args[0][0]

        # Create a fake CARLA image (800x600 BGRA = 1,920,000 bytes)
        mock_image = MagicMock()
        mock_image.raw_data = bytes(np.zeros(600 * 800 * 4, dtype=np.uint8))

        callback(mock_image)

        assert pipeline.latest_image is not None
        assert pipeline.latest_image.shape == (600, 800, 3)
