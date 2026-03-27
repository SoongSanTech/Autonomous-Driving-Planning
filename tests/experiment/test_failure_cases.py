"""Unit + Property tests for FailureCase.

Property 11: 실패 사례 기록 완전성 — 모든 필드 보존.
"""

import pytest
from hypothesis import given, settings, strategies as st

from experiment.analysis import FailureCase


class TestFailureCaseUnit:
    def test_create_failure_case(self):
        fc = FailureCase(
            timestamp=1234.5,
            failure_type="collision",
            image_path="/tmp/img.png",
            steering_pred=0.3,
            throttle_pred=0.7,
            lane_distance=1.2,
            velocity=5.0,
            scenario_id="straight_clear_day",
        )
        assert fc.failure_type == "collision"
        assert fc.timestamp == 1234.5
        assert fc.image_path == "/tmp/img.png"
        assert fc.steering_pred == 0.3
        assert fc.throttle_pred == 0.7
        assert fc.lane_distance == 1.2
        assert fc.velocity == 5.0
        assert fc.scenario_id == "straight_clear_day"

    def test_failure_case_none_image(self):
        fc = FailureCase(
            timestamp=0.0,
            failure_type="lane_departure",
            image_path=None,
            steering_pred=-0.5,
            throttle_pred=0.2,
            lane_distance=3.5,
            velocity=2.0,
            scenario_id="curve_fog_night",
        )
        assert fc.image_path is None

    def test_failure_case_to_dict(self):
        fc = FailureCase(
            timestamp=100.0,
            failure_type="stopped",
            image_path=None,
            steering_pred=0.0,
            throttle_pred=0.0,
            lane_distance=0.1,
            velocity=0.0,
            scenario_id="intersection_clear_day",
        )
        d = {
            "timestamp": fc.timestamp,
            "failure_type": fc.failure_type,
            "image_path": fc.image_path,
            "steering_pred": fc.steering_pred,
            "throttle_pred": fc.throttle_pred,
            "lane_distance": fc.lane_distance,
            "velocity": fc.velocity,
            "scenario_id": fc.scenario_id,
        }
        # Roundtrip: dict → FailureCase
        fc2 = FailureCase(**d)
        assert fc2.timestamp == fc.timestamp
        assert fc2.failure_type == fc.failure_type
        assert fc2.scenario_id == fc.scenario_id


# Feature: experiment-validation, Property 11: 실패 사례 기록 완전성
class TestProperty11FailureCaseCompleteness:
    @settings(max_examples=100, deadline=None)
    @given(
        timestamp=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
        failure_type=st.sampled_from(["collision", "lane_departure", "stopped"]),
        steering_pred=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        throttle_pred=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        lane_distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        velocity=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
        scenario_id=st.sampled_from([
            "straight_clear_day", "intersection_rain_night", "curve_fog_night",
        ]),
    )
    def test_all_fields_preserved(self, timestamp, failure_type, steering_pred,
                                   throttle_pred, lane_distance, velocity, scenario_id):
        """FailureCase 저장/조회 시 모든 필드 보존."""
        fc = FailureCase(
            timestamp=timestamp,
            failure_type=failure_type,
            image_path=None,
            steering_pred=steering_pred,
            throttle_pred=throttle_pred,
            lane_distance=lane_distance,
            velocity=velocity,
            scenario_id=scenario_id,
        )

        # Dict roundtrip
        d = {
            "timestamp": fc.timestamp,
            "failure_type": fc.failure_type,
            "image_path": fc.image_path,
            "steering_pred": fc.steering_pred,
            "throttle_pred": fc.throttle_pred,
            "lane_distance": fc.lane_distance,
            "velocity": fc.velocity,
            "scenario_id": fc.scenario_id,
        }

        fc2 = FailureCase(**d)
        assert fc2.timestamp == timestamp
        assert fc2.failure_type == failure_type
        assert fc2.steering_pred == steering_pred
        assert fc2.throttle_pred == throttle_pred
        assert fc2.lane_distance == lane_distance
        assert fc2.velocity == velocity
        assert fc2.scenario_id == scenario_id
