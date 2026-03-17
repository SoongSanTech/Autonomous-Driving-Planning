"""
Data Pipeline - Synchronized multi-modal data collection for autonomous driving ML.

This package provides a data collection system that captures RGB camera images
paired with vehicle state telemetry from the CARLA simulator at 10Hz.
"""

from data_pipeline.models import VehicleState, FrameData
from data_pipeline.sync_controller import SynchronousModeController
from data_pipeline.async_logger import AsyncDataLogger
from data_pipeline.episode_manager import EpisodeManager, WeatherPreset, TimeOfDay
from data_pipeline.pipeline import DataPipeline

__all__ = [
    "VehicleState",
    "FrameData",
    "SynchronousModeController",
    "AsyncDataLogger",
    "EpisodeManager",
    "WeatherPreset",
    "TimeOfDay",
    "DataPipeline",
]
