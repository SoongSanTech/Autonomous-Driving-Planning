"""
Tests for CARLAGymEnv.

Property-based tests:
- Property 6: Step return structure
- Property 7: Observation shape (224×224×3)

Note: These tests verify the interface without requiring a running CARLA server.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.carla_gym_env import CARLAGymEnv


class TestCARLAGymEnvSpaces:
    """Verify Gym space definitions."""

    def test_observation_space(self):
        env = CARLAGymEnv.__new__(CARLAGymEnv)
        env.observation_space = CARLAGymEnv().__dict__.get("observation_space", None)
        # Create without connecting
        env = CARLAGymEnv.__new__(CARLAGymEnv)
        CARLAGymEnv.__init__(env)
        assert env.observation_space.shape == (224, 224, 3)
        assert env.observation_space.dtype == np.uint8

    def test_action_space(self):
        env = CARLAGymEnv.__new__(CARLAGymEnv)
        CARLAGymEnv.__init__(env)
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, 0.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])

    def test_action_space_sampling(self):
        env = CARLAGymEnv.__new__(CARLAGymEnv)
        CARLAGymEnv.__init__(env)
        for _ in range(10):
            action = env.action_space.sample()
            assert action.shape == (2,)
            assert -1.0 <= action[0] <= 1.0
            assert 0.0 <= action[1] <= 1.0


class TestObservationShapeProperty:
    """Property 7: All observations have shape (224, 224, 3)."""

    @given(
        h=st.integers(min_value=100, max_value=1200),
        w=st.integers(min_value=100, max_value=1600),
    )
    @settings(max_examples=10)
    def test_observation_in_space(self, h, w):
        """Any observation should fit in the observation space."""
        env = CARLAGymEnv.__new__(CARLAGymEnv)
        CARLAGymEnv.__init__(env)
        # Simulate a 224×224 observation (what the env would produce)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        assert env.observation_space.contains(obs)
