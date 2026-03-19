"""
RLTrainer: PPO training pipeline with BC warm-start.

Two-phase training:
  Phase 1: Backbone frozen (100 episodes) — LR 3e-5
  Phase 2: Full fine-tune — LR 1e-5

PPO hyperparameters: GAE λ=0.95, clip ratio=0.2, γ=0.99
"""

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from model.checkpoint import CheckpointManager
from model.rl_policy import RLPolicyNetwork

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Stores trajectory data for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions_raw = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action_raw, log_prob, reward, value, done):
        self.states.append(state)
        self.actions_raw.append(action_raw)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """Compute GAE advantages and discounted returns."""
        advantages = []
        returns = []
        gae = 0.0

        values = self.values + [0.0]  # bootstrap with 0 for terminal

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int, device: str = "cpu"):
        """Yield mini-batches for PPO update."""
        n = len(self.states)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            states = torch.stack([self.states[i] for i in batch_idx]).to(device)
            actions = torch.stack([self.actions_raw[i] for i in batch_idx]).to(device)
            old_log_probs = torch.stack([self.log_probs[i] for i in batch_idx]).to(device)
            returns = torch.tensor(
                [self.returns[i] for i in batch_idx], dtype=torch.float32
            ).unsqueeze(1).to(device)
            advantages = torch.tensor(
                [self.advantages[i] for i in batch_idx], dtype=torch.float32
            ).unsqueeze(1).to(device)

            # Normalize advantages
            if len(advantages) > 1 and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            yield states, actions, old_log_probs, returns, advantages

    def clear(self):
        self.states.clear()
        self.actions_raw.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class RLTrainer:
    """
    PPO trainer with BC warm-start policy.

    Args:
        policy: RLPolicyNetwork (warm-started from BC).
        env: Gymnasium-compatible environment.
        lr: Initial learning rate.
        gae_lambda: GAE lambda.
        clip_ratio: PPO clip ratio.
        gamma: Discount factor.
        device: Training device.
        checkpoint_dir: Directory for checkpoints.
    """

    def __init__(
        self,
        policy: RLPolicyNetwork,
        env,
        lr: float = 3e-5,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.policy = policy.to(device)
        self.env = env
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.device = device

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, policy.parameters()), lr=lr
        )
        self.ckpt_manager = CheckpointManager(checkpoint_dir)
        self.buffer = RolloutBuffer()

    def train(
        self,
        num_episodes: int = 5000,
        frozen_episodes: int = 100,
        finetune_lr: float = 1e-5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        checkpoint_interval: int = 100,
        max_grad_norm: float = 0.5,
    ) -> dict:
        """
        Run PPO training loop.

        Args:
            num_episodes: Total training episodes.
            frozen_episodes: Episodes with backbone frozen.
            finetune_lr: LR after unfreezing backbone.
            ppo_epochs: PPO update epochs per rollout.
            batch_size: Mini-batch size for PPO updates.
            checkpoint_interval: Save checkpoint every N episodes.
            max_grad_norm: Gradient clipping norm.

        Returns:
            Training metrics dict.
        """
        # Phase 1: freeze backbone
        self.policy.freeze_backbone()
        logger.info("Phase 1: backbone frozen for %d episodes", frozen_episodes)

        episode_rewards = []
        episode_lengths = []
        best_avg_reward = float("-inf")
        best_path = None

        for episode in range(1, num_episodes + 1):
            # Phase transition
            if episode == frozen_episodes + 1:
                self.policy.unfreeze_backbone()
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.policy.parameters()),
                    lr=finetune_lr,
                )
                logger.info("Phase 2: backbone unfrozen, LR=%e", finetune_lr)

            # Collect trajectory
            ep_reward, ep_length = self._collect_episode()
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            # PPO update
            self.buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
            policy_loss, value_loss = self._ppo_update(
                ppo_epochs, batch_size, max_grad_norm
            )
            self.buffer.clear()

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                logger.info(
                    "Episode %d/%d — avg_reward: %.2f, avg_length: %.0f, "
                    "policy_loss: %.4f, value_loss: %.4f",
                    episode, num_episodes, avg_reward, avg_length,
                    policy_loss, value_loss,
                )

            # Checkpoint
            if episode % checkpoint_interval == 0:
                avg_reward = np.mean(episode_rewards[-checkpoint_interval:])
                metrics = {
                    "avg_reward": avg_reward,
                    "avg_length": np.mean(episode_lengths[-checkpoint_interval:]),
                    "episode": episode,
                }
                path = self.ckpt_manager.save(
                    self.policy, self.optimizer, episode, metrics, model_type="rl",
                )
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_path = path

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "best_avg_reward": best_avg_reward,
            "best_checkpoint": best_path,
        }

    def _collect_episode(self) -> tuple:
        """Collect one episode of experience."""
        from model.dataset import default_transform
        from PIL import Image as PILImage

        transform = default_transform()

        obs, info = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Preprocess observation
            pil_img = PILImage.fromarray(obs)
            state_tensor = transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_np, log_prob, value, _ = self.policy.get_action(state_tensor)

            obs_next, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Store raw action (pre-activation) for PPO update
            features = self.policy.backbone(state_tensor)
            features = features.view(features.size(0), -1)
            raw_action = self.policy.actor_head(features).detach().squeeze(0)

            self.buffer.add(
                state=state_tensor.squeeze(0).cpu(),
                action_raw=raw_action.cpu(),
                log_prob=log_prob.detach().cpu(),
                reward=reward,
                value=value.item(),
                done=float(done),
            )

            obs = obs_next
            total_reward += reward
            steps += 1

        return total_reward, steps

    def _ppo_update(
        self, ppo_epochs: int, batch_size: int, max_grad_norm: float
    ) -> tuple:
        """Run PPO update on collected buffer."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            for states, actions, old_log_probs, returns, advantages in \
                    self.buffer.get_batches(batch_size, self.device):

                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean() * 0.01

                loss = policy_loss + 0.5 * value_loss + entropy_loss

                if math.isnan(loss.item()):
                    logger.warning("NaN loss in PPO update, skipping batch")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1

        n = max(num_updates, 1)
        return total_policy_loss / n, total_value_loss / n
