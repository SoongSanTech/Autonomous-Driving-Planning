"""
Tests for RLTrainer (PPO).

Uses a mock Gym environment to test training loop,
checkpoint saving, and trajectory collection.
"""

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from model.rl_policy import RLPolicyNetwork
from model.rl_trainer import RLTrainer, RolloutBuffer


class MockGymEnv(gym.Env):
    """Simple mock environment for testing."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8)
        self.action_space = spaces.Box(
            np.float32([-1.0, 0.0]), np.float32([1.0, 1.0]), dtype=np.float32
        )
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        reward = float(np.random.randn())
        terminated = self._step >= 5  # Short episodes for testing
        return obs, reward, terminated, False, {"lane_distance": 0.0, "collision": False}


class TestRolloutBuffer:
    """RolloutBuffer tests."""

    def test_add_and_clear(self):
        buf = RolloutBuffer()
        buf.add(torch.randn(3, 224, 224), torch.randn(2), torch.tensor(0.1), 1.0, 0.5, 0.0)
        assert len(buf.states) == 1
        buf.clear()
        assert len(buf.states) == 0

    def test_compute_returns(self):
        buf = RolloutBuffer()
        for _ in range(5):
            buf.add(torch.randn(3, 224, 224), torch.randn(2), torch.tensor(0.1), 1.0, 0.5, 0.0)
        buf.dones[-1] = 1.0
        buf.compute_returns_and_advantages()
        assert len(buf.returns) == 5
        assert len(buf.advantages) == 5

    def test_get_batches(self):
        buf = RolloutBuffer()
        for _ in range(10):
            buf.add(torch.randn(3, 224, 224), torch.randn(2), torch.tensor(0.1), 1.0, 0.5, 0.0)
        buf.dones[-1] = 1.0
        buf.compute_returns_and_advantages()

        batches = list(buf.get_batches(batch_size=4))
        assert len(batches) >= 2  # 10 samples / 4 batch = 3 batches


class TestRLTrainer:
    """RL trainer tests with mock environment."""

    def test_train_runs(self, tmp_path):
        policy = RLPolicyNetwork(pretrained=False)
        env = MockGymEnv()
        trainer = RLTrainer(
            policy, env, checkpoint_dir=str(tmp_path), lr=1e-4
        )
        result = trainer.train(
            num_episodes=2, frozen_episodes=1,
            checkpoint_interval=2, ppo_epochs=1, batch_size=4,
        )
        assert "episode_rewards" in result
        assert len(result["episode_rewards"]) == 2

    def test_checkpoint_saved(self, tmp_path):
        policy = RLPolicyNetwork(pretrained=False)
        env = MockGymEnv()
        trainer = RLTrainer(
            policy, env, checkpoint_dir=str(tmp_path), lr=1e-4
        )
        result = trainer.train(
            num_episodes=2, frozen_episodes=1,
            checkpoint_interval=2, ppo_epochs=1, batch_size=4,
        )
        assert result["best_checkpoint"] is not None

    def test_phase_transition(self, tmp_path):
        policy = RLPolicyNetwork(pretrained=False)
        env = MockGymEnv()
        trainer = RLTrainer(
            policy, env, checkpoint_dir=str(tmp_path), lr=1e-4
        )
        # After training past frozen_episodes, backbone should be unfrozen
        trainer.train(
            num_episodes=3, frozen_episodes=1,
            checkpoint_interval=3, ppo_epochs=1, batch_size=4,
        )
        for p in policy.backbone.parameters():
            assert p.requires_grad
