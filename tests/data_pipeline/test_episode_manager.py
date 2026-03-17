"""
Unit tests for EpisodeManager scenario configuration methods.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from data_pipeline.episode_manager import (
    EpisodeManager,
    WeatherPreset,
    TimeOfDay,
)


def _make_mock_world():
    """Create a mock CARLA world for testing."""
    world = MagicMock()
    # get_weather returns a mock weather object with a mutable sun_altitude_angle
    weather_obj = MagicMock()
    weather_obj.sun_altitude_angle = 0.0
    world.get_weather.return_value = weather_obj
    return world


@pytest.mark.unit
class TestApplyWeather:
    """Tests for apply_weather method."""

    def test_sets_correct_weather_preset(self):
        """Verify apply_weather calls set_weather with the correct CARLA preset."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        with patch("carla.WeatherParameters") as mock_wp:
            mock_wp.ClearNoon = MagicMock(name="ClearNoon")
            em.apply_weather(WeatherPreset.CLEAR)

            world.set_weather.assert_called_once_with(mock_wp.ClearNoon)

    def test_stores_current_weather(self):
        """Verify apply_weather updates _current_weather."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        em.apply_weather(WeatherPreset.RAIN)

        assert em._current_weather == WeatherPreset.RAIN

    def test_appends_to_weather_history(self):
        """Verify apply_weather appends to _weather_history."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        em.apply_weather(WeatherPreset.CLEAR)
        em.apply_weather(WeatherPreset.FOG)

        assert em._weather_history == [WeatherPreset.CLEAR, WeatherPreset.FOG]

    def test_error_handling_continues_silently(self):
        """Verify apply_weather logs error and continues when CARLA fails."""
        world = _make_mock_world()
        world.set_weather.side_effect = RuntimeError("CARLA error")
        em = EpisodeManager(world)

        # Should not raise
        em.apply_weather(WeatherPreset.CLEAR)

        # State should NOT be updated on failure
        assert em._current_weather is None
        assert em._weather_history == []


@pytest.mark.unit
class TestApplyTimeOfDay:
    """Tests for apply_time_of_day method."""

    def test_sets_correct_sun_angle(self):
        """Verify apply_time_of_day sets sun_altitude_angle on weather."""
        world = _make_mock_world()
        weather_obj = world.get_weather.return_value
        em = EpisodeManager(world)

        em.apply_time_of_day(TimeOfDay.NIGHT)

        assert weather_obj.sun_altitude_angle == 180.0
        world.set_weather.assert_called_once_with(weather_obj)

    def test_stores_current_time_of_day(self):
        """Verify apply_time_of_day updates _current_time_of_day."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        em.apply_time_of_day(TimeOfDay.BACKLIGHT)

        assert em._current_time_of_day == TimeOfDay.BACKLIGHT

    def test_appends_to_time_history(self):
        """Verify apply_time_of_day appends to _time_history."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        em.apply_time_of_day(TimeOfDay.DAYTIME)
        em.apply_time_of_day(TimeOfDay.NIGHT)

        assert em._time_history == [TimeOfDay.DAYTIME, TimeOfDay.NIGHT]

    def test_error_handling_continues_silently(self):
        """Verify apply_time_of_day logs error and continues when CARLA fails."""
        world = _make_mock_world()
        world.get_weather.side_effect = RuntimeError("CARLA error")
        em = EpisodeManager(world)

        # Should not raise
        em.apply_time_of_day(TimeOfDay.NIGHT)

        # State should NOT be updated on failure
        assert em._current_time_of_day is None
        assert em._time_history == []


@pytest.mark.unit
class TestStartNewEpisode:
    """Tests for start_new_episode method."""

    def test_applies_weather_and_time(self):
        """Verify start_new_episode calls apply_weather and apply_time_of_day."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        em.start_new_episode()

        # Both should have been set
        assert em._current_weather is not None
        assert em._current_time_of_day is not None
        assert len(em._weather_history) == 1
        assert len(em._time_history) == 1

    def test_selects_from_valid_presets(self):
        """Verify start_new_episode picks valid enum members."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        em.start_new_episode()

        assert em._current_weather in list(WeatherPreset)
        assert em._current_time_of_day in list(TimeOfDay)

    @patch("data_pipeline.episode_manager.random.choice")
    def test_uses_random_selection(self, mock_choice):
        """Verify start_new_episode uses random.choice for selection."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        mock_choice.side_effect = [WeatherPreset.FOG, TimeOfDay.BACKLIGHT]
        em.start_new_episode()

        assert em._current_weather == WeatherPreset.FOG
        assert em._current_time_of_day == TimeOfDay.BACKLIGHT


@pytest.mark.unit
class TestShouldResetEpisode:
    """Tests for should_reset_episode method."""

    def test_returns_false_before_duration(self):
        """Verify returns False when elapsed time is less than duration."""
        world = _make_mock_world()
        em = EpisodeManager(world, episode_duration_sec=300.0)

        assert em.should_reset_episode(100.0) is False
        assert em.should_reset_episode(299.9) is False

    def test_returns_true_at_duration(self):
        """Verify returns True when elapsed time equals duration."""
        world = _make_mock_world()
        em = EpisodeManager(world, episode_duration_sec=300.0)

        assert em.should_reset_episode(300.0) is True

    def test_returns_true_after_duration(self):
        """Verify returns True when elapsed time exceeds duration."""
        world = _make_mock_world()
        em = EpisodeManager(world, episode_duration_sec=300.0)

        assert em.should_reset_episode(500.0) is True

    def test_custom_duration(self):
        """Verify works with custom episode duration."""
        world = _make_mock_world()
        em = EpisodeManager(world, episode_duration_sec=60.0)

        assert em.should_reset_episode(59.9) is False
        assert em.should_reset_episode(60.0) is True

    def test_zero_elapsed(self):
        """Verify returns False at start of episode."""
        world = _make_mock_world()
        em = EpisodeManager(world)

        assert em.should_reset_episode(0.0) is False
