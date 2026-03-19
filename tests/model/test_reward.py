"""
Tests for RewardFunction.

Property-based tests:
- Property 8: Lane centering incentive
- Property 9: Collision penalty
- Property 10: Steering smoothness
- Property 11: Scalar output
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.reward import RewardFunction


@pytest.fixture
def reward_fn():
    return RewardFunction()


class TestRewardBasic:
    """Basic reward computation tests."""

    def test_zero_state(self, reward_fn):
        info = {"lane_distance": 0.0, "collision": False, "velocity": 0.0, "heading_error": 0.0}
        r = reward_fn.compute(info, [0.0, 0.5])
        assert isinstance(r, float)

    def test_collision_negative(self, reward_fn):
        info = {"lane_distance": 0.0, "collision": True, "velocity": 0.0, "heading_error": 0.0}
        r = reward_fn.compute(info, [0.0, 0.5])
        assert r < 0

    def test_no_collision_no_penalty(self, reward_fn):
        info = {"lane_distance": 0.0, "collision": False, "velocity": 5.0, "heading_error": 0.0}
        r = reward_fn.compute(info, [0.0, 0.5])
        assert r >= 0  # progress reward should be positive

    def test_custom_weights(self):
        fn = RewardFunction(w_lane=0.0, w_collision=0.0, w_steering=0.0, w_progress=1.0)
        info = {"lane_distance": 10.0, "collision": True, "velocity": 5.0, "heading_error": 0.0}
        r = fn.compute(info, [0.0, 0.5])
        assert r == pytest.approx(5.0)  # only progress component


# --- Property-based tests ---

class TestRewardProperties:
    """Property-based tests for reward function."""

    @given(
        d1=st.floats(min_value=0.0, max_value=5.0),
        d2=st.floats(min_value=0.0, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_lane_centering_incentive(self, d1, d2):
        """Property 8: Closer to center → higher reward (all else equal)."""
        fn = RewardFunction()
        base_info = {"collision": False, "velocity": 5.0, "heading_error": 0.0}
        action = [0.0, 0.5]

        r1 = fn.compute({**base_info, "lane_distance": d1}, action)
        r2 = fn.compute({**base_info, "lane_distance": d2}, action)

        if d1 < d2:
            assert r1 >= r2
        elif d1 > d2:
            assert r1 <= r2

    @given(collision=st.booleans())
    @settings(max_examples=10)
    def test_collision_penalty(self, collision):
        """Property 9: Collision → negative reward component."""
        fn = RewardFunction(w_lane=0.0, w_steering=0.0, w_progress=0.0)
        info = {"lane_distance": 0.0, "collision": collision, "velocity": 0.0, "heading_error": 0.0}
        r = fn.compute(info, [0.0, 0.5])

        if collision:
            assert r < 0
        else:
            assert r == 0.0

    @given(
        steering=st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_steering_smoothness(self, steering):
        """Property 10: |steering| > 0.3 → negative penalty."""
        fn = RewardFunction(w_lane=0.0, w_collision=0.0, w_progress=0.0)
        info = {"lane_distance": 0.0, "collision": False, "velocity": 0.0, "heading_error": 0.0}
        r = fn.compute(info, [steering, 0.5])

        if abs(steering) > 0.3:
            assert r < 0
        else:
            assert r == 0.0

    @given(
        lane_dist=st.floats(min_value=0.0, max_value=10.0),
        velocity=st.floats(min_value=0.0, max_value=30.0),
        heading=st.floats(min_value=-math.pi, max_value=math.pi),
        steering=st.floats(min_value=-1.0, max_value=1.0),
        collision=st.booleans(),
    )
    @settings(max_examples=50)
    def test_scalar_output(self, lane_dist, velocity, heading, steering, collision):
        """Property 11: Always returns a single float."""
        fn = RewardFunction()
        info = {
            "lane_distance": lane_dist,
            "collision": collision,
            "velocity": velocity,
            "heading_error": heading,
        }
        r = fn.compute(info, [steering, 0.5])
        assert isinstance(r, float)
        assert not math.isnan(r)
        assert not math.isinf(r)
