"""
ScenarioManager: 고정 시드 기반 표준 평가 시나리오 관리.

7개 표준 시나리오를 정의하고, CARLA 환경에 적용하여
재현 가능한 성능 비교를 보장한다.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import carla
except ImportError:
    carla = None

logger = logging.getLogger(__name__)


@dataclass
class EvalScenario:
    """평가 시나리오 정의."""
    scenario_id: str
    road_type: str
    weather: str
    time_of_day: float
    spawn_point_index: int
    seed: int
    max_steps: int
    description: str


# 7개 표준 평가 시나리오
STANDARD_SCENARIOS = [
    EvalScenario(
        scenario_id="straight_clear_day",
        road_type="straight",
        weather="ClearNoon",
        time_of_day=0,
        spawn_point_index=0,
        seed=42,
        max_steps=1000,
        description="직선 도로, 맑은 낮",
    ),
    EvalScenario(
        scenario_id="straight_rain_day",
        road_type="straight",
        weather="WetCloudyNoon",
        time_of_day=0,
        spawn_point_index=0,
        seed=42,
        max_steps=1000,
        description="직선 도로, 비 오는 낮",
    ),
    EvalScenario(
        scenario_id="intersection_clear_day",
        road_type="intersection",
        weather="ClearNoon",
        time_of_day=0,
        spawn_point_index=100,
        seed=100,
        max_steps=1000,
        description="교차로, 맑은 낮",
    ),
    EvalScenario(
        scenario_id="intersection_rain_night",
        road_type="intersection",
        weather="WetCloudyNoon",
        time_of_day=180,
        spawn_point_index=100,
        seed=100,
        max_steps=1000,
        description="교차로, 비 오는 밤",
    ),
    EvalScenario(
        scenario_id="curve_clear_day",
        road_type="curve",
        weather="ClearNoon",
        time_of_day=0,
        spawn_point_index=200,
        seed=200,
        max_steps=1000,
        description="커브, 맑은 낮",
    ),
    EvalScenario(
        scenario_id="curve_fog_night",
        road_type="curve",
        weather="SoftRainSunset",
        time_of_day=180,
        spawn_point_index=200,
        seed=200,
        max_steps=1000,
        description="커브, 안개 낀 밤",
    ),
    EvalScenario(
        scenario_id="intersection_fog_backlight",
        road_type="intersection",
        weather="SoftRainSunset",
        time_of_day=90,
        spawn_point_index=300,
        seed=300,
        max_steps=1000,
        description="교차로, 안개+역광",
    ),
]


class ScenarioManager:
    """통제된 CARLA 평가 시나리오 관리."""

    def __init__(self, carla_host: str = "localhost", carla_port: int = 2000):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self._scenarios = {s.scenario_id: s for s in STANDARD_SCENARIOS}

    def get_scenario(self, scenario_id: str) -> EvalScenario:
        """ID로 시나리오 조회. 없으면 KeyError."""
        if scenario_id not in self._scenarios:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        return self._scenarios[scenario_id]

    def list_scenarios(self, road_type: Optional[str] = None) -> list[EvalScenario]:
        """시나리오 목록 (도로 유형별 필터링)."""
        scenarios = list(self._scenarios.values())
        if road_type is not None:
            scenarios = [s for s in scenarios if s.road_type == road_type]
        return scenarios

    def apply_scenario(self, env, scenario: EvalScenario) -> None:
        """시나리오 설정을 CARLA 환경에 적용.

        Args:
            env: CARLAGymEnv 인스턴스.
            scenario: 적용할 시나리오.
        """
        try:
            if carla is None:
                logger.warning("CARLA not available, skipping weather/spawn setup")
                return
        except Exception:
            logger.warning("CARLA not available, skipping weather/spawn setup")
            return

        # 날씨 설정
        try:
            weather = getattr(carla.WeatherParameters, scenario.weather)
            if hasattr(weather, 'sun_altitude_angle'):
                weather.sun_altitude_angle = scenario.time_of_day
            env._world.set_weather(weather)
        except (AttributeError, Exception) as e:
            logger.warning("날씨 설정 실패, 기본값 사용: %s", e)
            env._world.set_weather(carla.WeatherParameters.ClearNoon)

        # 시드 설정
        np.random.seed(scenario.seed)

        # 스폰 포인트 설정
        spawn_points = env._world.get_map().get_spawn_points()
        if scenario.spawn_point_index < len(spawn_points):
            env._spawn_point = spawn_points[scenario.spawn_point_index]
        else:
            # fallback: 가장 가까운 유효 스폰 포인트
            fallback_idx = min(scenario.spawn_point_index, len(spawn_points) - 1)
            env._spawn_point = spawn_points[fallback_idx]
            logger.warning(
                "스폰 포인트 %d 범위 초과, fallback to %d",
                scenario.spawn_point_index, fallback_idx,
            )

        logger.info("시나리오 적용: %s (%s)", scenario.scenario_id, scenario.description)

    def run_evaluation(self, model, scenario: EvalScenario,
                       num_runs: int = 10, env=None) -> dict:
        """특정 시나리오에서 모델 평가 실행.

        Args:
            model: 평가할 모델.
            scenario: 평가 시나리오.
            num_runs: 반복 횟수.
            env: CARLAGymEnv (None이면 새로 생성).

        Returns:
            시나리오별 평가 결과 dict.
        """
        from model.evaluator import ModelEvaluator

        if env is None:
            from model.carla_gym_env import CARLAGymEnv
            env = CARLAGymEnv(
                host=self.carla_host,
                port=self.carla_port,
                max_steps=scenario.max_steps,
            )

        self.apply_scenario(env, scenario)
        evaluator = ModelEvaluator(device="cpu")
        metrics = evaluator.evaluate_online(model, env, num_episodes=num_runs)

        result = {
            "scenario_id": scenario.scenario_id,
            "description": scenario.description,
            "num_runs": num_runs,
            **metrics,
        }

        logger.info("평가 완료: %s — %s", scenario.scenario_id, metrics)
        return result

    def run_full_evaluation(self, model, scenario_ids: Optional[list[str]] = None,
                            num_runs: int = 10) -> dict:
        """전체 표준 시나리오 평가 실행.

        Args:
            model: 평가할 모델.
            scenario_ids: 평가할 시나리오 ID 목록 (None이면 전체).
            num_runs: 시나리오당 반복 횟수.

        Returns:
            전체 평가 결과 dict.
        """
        if scenario_ids is None:
            scenarios = list(self._scenarios.values())
        else:
            scenarios = [self.get_scenario(sid) for sid in scenario_ids]

        results = {}
        for scenario in scenarios:
            try:
                result = self.run_evaluation(model, scenario, num_runs=num_runs)
                results[scenario.scenario_id] = result
            except Exception as e:
                logger.error("시나리오 %s 평가 실패: %s", scenario.scenario_id, e)
                results[scenario.scenario_id] = {
                    "scenario_id": scenario.scenario_id,
                    "error": str(e),
                }

        return {
            "total_scenarios": len(scenarios),
            "completed": sum(1 for r in results.values() if "error" not in r),
            "results": results,
        }
