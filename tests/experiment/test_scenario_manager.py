"""Unit tests for ScenarioManager."""

from unittest.mock import MagicMock, patch

import pytest

from experiment.scenario_manager import (
    EvalScenario,
    ScenarioManager,
    STANDARD_SCENARIOS,
)


class TestStandardScenarios:
    def test_seven_scenarios_defined(self):
        assert len(STANDARD_SCENARIOS) == 7

    def test_all_scenario_ids_unique(self):
        ids = [s.scenario_id for s in STANDARD_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_all_road_types_covered(self):
        road_types = {s.road_type for s in STANDARD_SCENARIOS}
        assert road_types == {"straight", "intersection", "curve"}

    def test_scenario_fields_populated(self):
        for s in STANDARD_SCENARIOS:
            assert s.scenario_id
            assert s.road_type
            assert s.weather
            assert s.max_steps > 0
            assert s.description


class TestGetScenario:
    def test_get_existing(self):
        mgr = ScenarioManager()
        s = mgr.get_scenario("straight_clear_day")
        assert s.scenario_id == "straight_clear_day"
        assert s.road_type == "straight"

    def test_get_nonexistent_raises(self):
        mgr = ScenarioManager()
        with pytest.raises(KeyError, match="Unknown scenario"):
            mgr.get_scenario("nonexistent_scenario")


class TestListScenarios:
    def test_list_all(self):
        mgr = ScenarioManager()
        all_scenarios = mgr.list_scenarios()
        assert len(all_scenarios) == 7

    def test_filter_by_road_type(self):
        mgr = ScenarioManager()
        intersections = mgr.list_scenarios(road_type="intersection")
        assert len(intersections) == 3
        assert all(s.road_type == "intersection" for s in intersections)

    def test_filter_straight(self):
        mgr = ScenarioManager()
        straights = mgr.list_scenarios(road_type="straight")
        assert len(straights) == 2

    def test_filter_curve(self):
        mgr = ScenarioManager()
        curves = mgr.list_scenarios(road_type="curve")
        assert len(curves) == 2

    def test_filter_nonexistent_type(self):
        mgr = ScenarioManager()
        result = mgr.list_scenarios(road_type="highway")
        assert len(result) == 0


class TestApplyScenario:
    """CARLA 모킹으로 apply_scenario 로직 검증."""

    def test_apply_sets_weather_and_seed(self):
        import sys
        mock_carla = MagicMock()
        mock_weather = MagicMock()
        mock_carla.WeatherParameters.ClearNoon = mock_weather
        setattr(mock_carla.WeatherParameters, "ClearNoon", mock_weather)

        mgr = ScenarioManager()
        scenario = mgr.get_scenario("straight_clear_day")

        mock_env = MagicMock()
        mock_spawn_points = [MagicMock() for _ in range(10)]
        mock_env._world.get_map.return_value.get_spawn_points.return_value = mock_spawn_points

        with patch.dict(sys.modules, {"carla": mock_carla}):
            with patch("experiment.scenario_manager.carla", mock_carla):
                mgr.apply_scenario(mock_env, scenario)
                mock_env._world.set_weather.assert_called_once()

    def test_apply_spawn_point_fallback(self):
        import sys
        mock_carla = MagicMock()
        mock_weather = MagicMock()
        mock_carla.WeatherParameters.SoftRainSunset = mock_weather
        mock_carla.WeatherParameters.ClearNoon = mock_weather

        mgr = ScenarioManager()
        # intersection_fog_backlight has spawn_point_index=300
        scenario = mgr.get_scenario("intersection_fog_backlight")

        mock_env = MagicMock()
        # Only 5 spawn points available → should fallback
        mock_spawn_points = [MagicMock() for _ in range(5)]
        mock_env._world.get_map.return_value.get_spawn_points.return_value = mock_spawn_points

        with patch.dict(sys.modules, {"carla": mock_carla}):
            with patch("experiment.scenario_manager.carla", mock_carla):
                mgr.apply_scenario(mock_env, scenario)
                # Should have set _spawn_point to last available (index 4)
                assert mock_env._spawn_point == mock_spawn_points[4]


class TestRunEvaluation:
    """CARLA/모델 모킹으로 run_evaluation 로직 검증."""

    @patch("experiment.scenario_manager.ScenarioManager.apply_scenario")
    @patch("model.evaluator.ModelEvaluator")
    def test_run_evaluation_returns_metrics(self, MockEvaluator, mock_apply):
        mgr = ScenarioManager()
        scenario = mgr.get_scenario("straight_clear_day")

        mock_model = MagicMock()
        mock_env = MagicMock()

        mock_eval_metrics = {
            "collision_count": 1,
            "avg_lane_distance": 0.5,
            "avg_survival_time": 30.0,
            "avg_episode_reward": 10.0,
            "num_episodes": 5,
        }

        mock_evaluator = MockEvaluator.return_value
        mock_evaluator.evaluate_online.return_value = mock_eval_metrics

        result = mgr.run_evaluation(mock_model, scenario, num_runs=5, env=mock_env)

        assert result["scenario_id"] == "straight_clear_day"
        assert result["collision_count"] == 1
        assert result["num_runs"] == 5
        mock_apply.assert_called_once_with(mock_env, scenario)


class TestRunFullEvaluation:
    @patch("experiment.scenario_manager.ScenarioManager.run_evaluation")
    def test_full_evaluation_all_scenarios(self, mock_run_eval):
        mgr = ScenarioManager()
        mock_model = MagicMock()

        mock_run_eval.return_value = {
            "scenario_id": "test",
            "collision_count": 0,
            "avg_survival_time": 60.0,
        }

        result = mgr.run_full_evaluation(mock_model, num_runs=3)
        assert result["total_scenarios"] == 7
        assert result["completed"] == 7
        assert mock_run_eval.call_count == 7

    @patch("experiment.scenario_manager.ScenarioManager.run_evaluation")
    def test_full_evaluation_subset(self, mock_run_eval):
        mgr = ScenarioManager()
        mock_model = MagicMock()

        mock_run_eval.return_value = {"scenario_id": "test"}

        result = mgr.run_full_evaluation(
            mock_model,
            scenario_ids=["straight_clear_day", "curve_clear_day"],
            num_runs=2,
        )
        assert result["total_scenarios"] == 2
        assert mock_run_eval.call_count == 2

    @patch("experiment.scenario_manager.ScenarioManager.run_evaluation")
    def test_full_evaluation_handles_failure(self, mock_run_eval):
        mgr = ScenarioManager()
        mock_model = MagicMock()

        mock_run_eval.side_effect = ConnectionError("CARLA not running")

        result = mgr.run_full_evaluation(
            mock_model,
            scenario_ids=["straight_clear_day"],
            num_runs=1,
        )
        assert result["completed"] == 0
        assert "error" in result["results"]["straight_clear_day"]
