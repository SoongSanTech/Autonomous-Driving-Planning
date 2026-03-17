"""
Synchronous Mode Controller for CARLA simulation.

This module enforces deterministic simulation stepping and coordinates
data capture timing at a configurable tick rate (default 10Hz).
"""

import logging
import carla

logger = logging.getLogger(__name__)


class SynchronousModeController:
    """
    Enforce deterministic simulation stepping and coordinate data capture timing.

    Uses CARLA's synchronous mode to advance the simulation by a fixed time step
    on each tick, ensuring consistent 10Hz operation and perfect temporal alignment
    between camera images and vehicle telemetry.
    """

    def __init__(self, world: carla.World, tick_rate_hz: float = 10.0):
        """
        Initialize synchronous mode controller.

        Args:
            world: CARLA world instance
            tick_rate_hz: Simulation frequency in Hz (default 10.0)
        """
        self._world = world
        self._tick_rate_hz = tick_rate_hz
        self._fixed_delta_seconds = 1.0 / tick_rate_hz

    def enable_synchronous_mode(self) -> None:
        """
        Configure CARLA world for synchronous mode.

        Sets fixed_delta_seconds = 1.0 / tick_rate_hz and enables
        synchronous_mode so the server waits for a client tick before
        advancing the simulation.
        """
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._fixed_delta_seconds
        self._world.apply_settings(settings)
        logger.info(
            "Synchronous mode enabled: fixed_delta=%.4fs (%.1f Hz)",
            self._fixed_delta_seconds,
            self._tick_rate_hz,
        )

    def tick(self) -> int:
        """
        Advance simulation by one time step.

        Returns:
            frame_id: Unique frame identifier from CARLA.
        """
        frame_id = self._world.tick()
        return frame_id

    def get_timestamp_ms(self) -> int:
        """
        Get current simulation timestamp in milliseconds.

        The timestamp is derived from CARLA's simulation elapsed time
        (not wall clock) for reproducibility.

        Returns:
            Millisecond-precision integer timestamp.
        """
        snapshot = self._world.get_snapshot()
        elapsed_seconds = snapshot.timestamp.elapsed_seconds
        return int(elapsed_seconds * 1000)
