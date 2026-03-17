"""
Episode Manager for CARLA scenario variation.

This module orchestrates environment resets and scenario diversity by
randomizing weather conditions and time of day at configurable intervals.
"""

import logging
import random
from enum import Enum
from typing import List, Optional

import carla

logger = logging.getLogger(__name__)


class WeatherPreset(Enum):
    """Available weather presets mapped to CARLA WeatherParameters attributes."""

    CLEAR = "ClearNoon"
    RAIN = "WetCloudyNoon"
    FOG = "SoftRainSunset"


class TimeOfDay(Enum):
    """Time of day presets defined by sun altitude angle in degrees."""

    DAYTIME = 0.0      # Sun angle 0 degrees
    NIGHT = 180.0      # Sun angle 180 degrees
    BACKLIGHT = 90.0   # Sun angle 90 degrees


class EpisodeManager:
    """
    Orchestrate scenario variation and environment resets for data diversity.

    Resets the environment at configurable intervals (default 5 minutes),
    randomizing weather conditions and time of day to generate varied
    training data.
    """

    def __init__(self, world: carla.World, episode_duration_sec: float = 300.0):
        """
        Initialize episode manager.

        Args:
            world: CARLA world instance
            episode_duration_sec: Duration per episode in seconds (default 5 minutes)
        """
        self._world = world
        self._episode_duration_sec = episode_duration_sec
        self._current_weather: Optional[WeatherPreset] = None
        self._current_time_of_day: Optional[TimeOfDay] = None
        self._weather_history: List[WeatherPreset] = []
        self._time_history: List[TimeOfDay] = []

    def start_new_episode(self) -> None:
        """Reset environment and apply random scenario configuration."""
        weather = random.choice(list(WeatherPreset))
        time_of_day = random.choice(list(TimeOfDay))
        self.apply_weather(weather)
        self.apply_time_of_day(time_of_day)
        logger.info(f"New episode: weather={weather.value}, time={time_of_day.name}")


    def should_reset_episode(self, elapsed_time_sec: float) -> bool:
        """Check if episode duration has elapsed."""
        return elapsed_time_sec >= self._episode_duration_sec


    def apply_weather(self, preset: WeatherPreset) -> None:
        """Apply weather configuration to CARLA world."""
        try:
            weather = getattr(carla.WeatherParameters, preset.value)
            self._world.set_weather(weather)
            self._current_weather = preset
            self._weather_history.append(preset)
            logger.info(f"Applied weather: {preset.value}")
        except Exception as e:
            logger.error(f"Failed to apply weather {preset.value}: {e}")


    def apply_time_of_day(self, time: TimeOfDay) -> None:
        """Apply sun angle configuration to CARLA world."""
        try:
            weather = self._world.get_weather()
            weather.sun_altitude_angle = time.value
            self._world.set_weather(weather)
            self._current_time_of_day = time
            self._time_history.append(time)
            logger.info(f"Applied time of day: {time.name} (sun angle: {time.value})")
        except Exception as e:
            logger.error(f"Failed to apply time of day {time.name}: {e}")

