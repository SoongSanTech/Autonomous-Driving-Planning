"""
ModelEvaluator: Quantitative evaluation of driving models.

Computes:
- MAE steering / throttle (offline, on test data)
- Collision event count (online, in CARLA)
- Average lane center distance
- Average survival time
- Inference latency
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates driving model performance.

    Supports both offline evaluation (on test DataLoader)
    and online evaluation (in CARLA Gym environment).

    Args:
        device: Evaluation device.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @torch.no_grad()
    def evaluate_offline(self, model, test_loader: DataLoader) -> dict:
        """
        Evaluate model on test data (offline).

        Args:
            model: BC or RL model with forward(image) → (steering, throttle).
            test_loader: DataLoader with (image, controls) pairs.

        Returns:
            Dict with mae_steering, mae_throttle, inference_time_ms.
        """
        model.eval()
        model.to(self.device)

        total_mae_steer = 0.0
        total_mae_throttle = 0.0
        total_samples = 0
        latencies = []

        for images, controls in test_loader:
            images = images.to(self.device)
            steering_gt = controls[:, 0]
            throttle_gt = controls[:, 1]

            start = time.perf_counter()
            steering_pred, throttle_pred = model(images)
            latency = (time.perf_counter() - start) * 1000

            steering_pred = steering_pred.squeeze(-1).cpu()
            throttle_pred = throttle_pred.squeeze(-1).cpu()

            total_mae_steer += (steering_pred - steering_gt).abs().sum().item()
            total_mae_throttle += (throttle_pred - throttle_gt).abs().sum().item()
            total_samples += len(images)
            latencies.append(latency / len(images))

        n = max(total_samples, 1)
        metrics = {
            "mae_steering": total_mae_steer / n,
            "mae_throttle": total_mae_throttle / n,
            "inference_time_ms": np.mean(latencies) if latencies else 0.0,
            "total_samples": total_samples,
        }

        logger.info(
            "Offline eval: MAE steer=%.4f, MAE throttle=%.4f, "
            "latency=%.1fms, samples=%d",
            metrics["mae_steering"], metrics["mae_throttle"],
            metrics["inference_time_ms"], total_samples,
        )
        return metrics

    def evaluate_online(
        self, model, env, num_episodes: int = 10, transform=None
    ) -> dict:
        """
        Evaluate model in CARLA environment (online).

        Args:
            model: Driving model.
            env: Gymnasium-compatible environment.
            num_episodes: Number of evaluation episodes.
            transform: Image preprocessing transform.

        Returns:
            Dict with collision_count, avg_lane_distance,
            avg_survival_time, success_rate.
        """
        from PIL import Image as PILImage
        from model.dataset import default_transform

        if transform is None:
            transform = default_transform()

        model.eval()
        model.to(self.device)

        collision_count = 0
        lane_distances = []
        survival_times = []
        episode_rewards = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            ep_lane_dists = []

            while not done:
                pil_img = PILImage.fromarray(obs)
                tensor = transform(pil_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    steering, throttle = model(tensor)

                action = np.array([steering.item(), throttle.item()])
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_steps += 1
                ep_lane_dists.append(info.get("lane_distance", 0.0))

                if info.get("collision", False):
                    collision_count += 1

            survival_times.append(ep_steps / 10.0)  # 10Hz → seconds
            lane_distances.extend(ep_lane_dists)
            episode_rewards.append(ep_reward)

            logger.info(
                "Episode %d/%d: reward=%.2f, steps=%d, collision=%s",
                ep + 1, num_episodes, ep_reward, ep_steps,
                info.get("collision", False),
            )

        metrics = {
            "collision_count": collision_count,
            "avg_lane_distance": np.mean(lane_distances) if lane_distances else 0.0,
            "avg_survival_time": np.mean(survival_times) if survival_times else 0.0,
            "avg_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "num_episodes": num_episodes,
        }

        logger.info(
            "Online eval: collisions=%d, avg_lane_dist=%.3f, "
            "avg_survival=%.1fs, avg_reward=%.2f",
            metrics["collision_count"], metrics["avg_lane_distance"],
            metrics["avg_survival_time"], metrics["avg_episode_reward"],
        )
        return metrics
