"""
Core data models for the data pipeline.

This module defines the data structures used to represent vehicle state
and frame data captured from the CARLA simulator.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    """
    Vehicle telemetry data captured at a single point in time.
    
    Attributes:
        speed: Vehicle speed in meters per second
        steering: Steering angle normalized to [-1.0, 1.0] range
        throttle: Throttle input normalized to [0.0, 1.0] range
        brake: Brake input normalized to [0.0, 1.0] range
    """
    speed: float      # m/s
    steering: float   # [-1.0, 1.0]
    throttle: float   # [0.0, 1.0]
    brake: float      # [0.0, 1.0]


@dataclass
class FrameData:
    """
    Complete data for a single collection cycle.
    
    Attributes:
        timestamp_ms: Millisecond-precision timestamp from simulation
        frame_id: Unique frame identifier from CARLA
        image: RGB camera image as numpy array with shape (600, 800, 3) and dtype uint8
        vehicle_state: Vehicle telemetry data synchronized with the image
    """
    timestamp_ms: int
    frame_id: int
    image: np.ndarray  # Shape: (600, 800, 3), dtype: uint8
    vehicle_state: VehicleState
