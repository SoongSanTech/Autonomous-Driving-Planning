"""
DataPipeline main orchestrator.

This module coordinates all subsystems (SynchronousModeController,
AsyncDataLogger, EpisodeManager) and manages the data collection lifecycle.
"""

import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import carla
import numpy as np

from data_pipeline.async_logger import AsyncDataLogger
from data_pipeline.episode_manager import EpisodeManager
from data_pipeline.models import VehicleState
from data_pipeline.sync_controller import SynchronousModeController

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main orchestrator for synchronized multi-modal data collection.

    Coordinates the Synchronous Mode Controller, Asynchronous Data Logger,
    and Episode Manager to capture RGB camera images paired with vehicle
    state telemetry from the CARLA simulator.
    """

    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        output_dir: str = "src/data",
        headless: bool = True,
    ):
        """
        Initialize data pipeline.

        Args:
            carla_host: CARLA server hostname (use explicit IP for WSL2).
            carla_port: CARLA server port (default 2000).
            output_dir: Output directory for images and labels.
            headless: Run without graphical display.
        """
        # Connection parameters
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.output_dir = output_dir
        self.headless = headless

        # Tracking variables
        self.frames_captured: int = 0
        self.frame_drops: int = 0
        self.running: bool = False

        # CARLA references (created in connect())
        self.client = None
        self.world = None

        # Subsystems requiring CARLA connection (created in connect())
        self.sync_controller: SynchronousModeController | None = None
        self.episode_manager: EpisodeManager | None = None

        # AsyncDataLogger — resolve daytime subfolder first, then create once
        self.data_logger = None

        # Sensor references (created in setup_sensors())
        self.camera = None
        self.vehicle = None
        self.latest_image = None

        # Resolve output directory with daytime-based subfolder and create logger
        self._resolve_output_dir()

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        logger.info(
            "DataPipeline initialized: host=%s, port=%d, output_dir=%s, headless=%s",
            self.carla_host,
            self.carla_port,
            self.output_dir,
            self.headless,
        )

    def _resolve_output_dir(self) -> None:
        """Resolve output directory to a daytime-based subfolder.

        Creates a subfolder under the base output_dir using the current
        datetime (e.g. src/data/2026-03-13_143022/) so that each
        collection session is stored separately without polluting the
        parent directory with timestamped siblings.
        """
        daytime_folder = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = str(Path(self.output_dir) / daytime_folder)
        self.data_logger = AsyncDataLogger(output_dir=self.output_dir)
        logger.info("Output directory resolved to: %s", self.output_dir)

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for SIGINT and SIGTERM for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle OS signals by initiating graceful shutdown.

        Args:
            signum: Signal number received.
            frame: Current stack frame (unused).
        """
        logger.info("Signal %d received, shutting down...", signum)
        self.running = False

    def connect(self) -> None:
        """Establish TCP connection to CARLA server with retry logic.

        Uses exponential backoff with a maximum of 5 attempts.
        Accepts explicit host IP to handle WSL2 localhost mapping issues.

        Raises:
            ConnectionError: If all connection attempts fail.
        """
        for attempt in range(5):
            try:
                client = carla.Client(self.carla_host, self.carla_port)
                client.set_timeout(10.0)
                self.client = client
                self.world = client.get_world()
                # Initialize subsystems that need CARLA
                self.sync_controller = SynchronousModeController(self.world)
                self.episode_manager = EpisodeManager(self.world)
                logger.info(
                    "Connected to CARLA at %s:%d",
                    self.carla_host,
                    self.carla_port,
                )
                return
            except RuntimeError as e:
                if attempt == 4:  # max_retries - 1
                    raise ConnectionError(
                        f"Failed to connect to CARLA at {self.carla_host}:{self.carla_port}"
                    ) from e
                wait_time = 2 ** attempt
                logger.warning(
                    "Connection attempt %d failed, retrying in %ds...",
                    attempt + 1,
                    wait_time,
                )
                time.sleep(wait_time)

    def setup_sensors(self) -> None:
        """Spawn ego vehicle with autopilot and attach camera sensor.

        If no vehicle exists in the world, spawns a Tesla Model 3 at a
        random spawn point with autopilot enabled. Then attaches an RGB
        camera at 800x600 resolution.
        """
        blueprint_library = self.world.get_blueprint_library()

        # --- Ego vehicle ---
        vehicles = self.world.get_actors().filter("vehicle.*")
        if vehicles:
            vehicle = vehicles[0]
            logger.info("Using existing vehicle: %s", vehicle.type_id)
        else:
            vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in the CARLA map")
            spawn_point = spawn_points[0]
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)
            logger.info(
                "Spawned ego vehicle: %s at %s (autopilot ON)",
                vehicle.type_id,
                spawn_point.location,
            )
        self.vehicle = vehicle

        # --- RGB Camera ---
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )

        self.latest_image = None

        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((600, 800, 4))  # BGRA
            self.latest_image = array[:, :, :3]  # Drop alpha, keep BGR

        self.camera.listen(camera_callback)
        logger.info("Camera sensor attached to %s", self.vehicle.type_id)

    def run(self, duration_sec: float = 3600.0) -> None:
        """
        Execute data collection for specified duration.

        Main loop ticks the synchronous controller, captures frame data
        (image + vehicle state), and enqueues it to the async logger.
        Detects CARLA server crashes via tick timeout and preserves
        partial data on failure.

        Args:
            duration_sec: Collection duration in seconds (default 1 hour).
        """
        self.running = True
        self.sync_controller.enable_synchronous_mode()
        self.data_logger.start()
        self.episode_manager.start_new_episode()

        start_time = time.time()
        episode_start_time = start_time

        try:
            while self.running:
                elapsed = time.time() - start_time
                if elapsed >= duration_sec:
                    break

                # Episode reset check
                episode_elapsed = time.time() - episode_start_time
                if self.episode_manager.should_reset_episode(episode_elapsed):
                    self.episode_manager.start_new_episode()
                    episode_start_time = time.time()

                # Tick simulation — RuntimeError signals CARLA crash
                try:
                    frame_id = self.sync_controller.tick()
                except RuntimeError:
                    logger.error("CARLA server crash detected!")
                    logger.info(
                        "Flushing queue and saving partial data... "
                        "%d frames saved before crash.",
                        self.frames_captured,
                    )
                    break

                # Get simulation timestamp
                timestamp_ms = self.sync_controller.get_timestamp_ms()

                # Capture frame if image is available
                if self.latest_image is not None:
                    try:
                        velocity = self.vehicle.get_velocity()
                        control = self.vehicle.get_control()
                    except RuntimeError:
                        logger.error(
                            "Ego vehicle destroyed by CARLA! "
                            "Saving %d frames collected so far.",
                            self.frames_captured,
                        )
                        break

                    speed = (
                        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                    ) ** 0.5
                    vehicle_state = VehicleState(
                        speed=speed,
                        steering=control.steer,
                        throttle=control.throttle,
                        brake=control.brake,
                    )

                    self.data_logger.enqueue_frame(
                        timestamp_ms, self.latest_image.copy(), vehicle_state
                    )
                    self.frames_captured += 1

                    # Headless mode progress logging
                    if self.headless and self.frames_captured % 100 == 0:
                        logger.info(
                            "Progress: %d frames captured, %.1fs elapsed",
                            self.frames_captured,
                            elapsed,
                        )
        finally:
            self.running = False
            self.frame_drops = self.data_logger.frame_drops
            elapsed = time.time() - start_time
            logger.info(
                "Collection stopped: %d frames, %d drops, %.1fs",
                self.frames_captured,
                self.frame_drops,
                elapsed,
            )


    def shutdown(self) -> None:
        """Clean up resources and close connections.

        Performs graceful termination:
        1. Stops the data logger (flushes remaining queue items to disk)
        2. Destroys the camera sensor
        3. Reports collection statistics (frames saved, drops, duration)
        """
        logger.info("Initiating graceful shutdown...")
        self.running = False

        # Stop data logger (flushes queue)
        if self.data_logger:
            self.data_logger.stop()

        # Destroy camera sensor
        if self.camera:
            try:
                self.camera.stop()
                self.camera.destroy()
            except RuntimeError:
                logger.debug("Camera already destroyed")

        # Destroy ego vehicle
        if self.vehicle:
            try:
                self.vehicle.destroy()
                logger.info("Ego vehicle destroyed")
            except RuntimeError:
                logger.debug("Vehicle already destroyed")

        # Report statistics
        logger.info(
            "Collection complete: %d frames saved, %d drops",
            self.frames_captured,
            self.frame_drops,
        )
