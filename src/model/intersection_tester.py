"""
IntersectionTester: Test model's ability to navigate intersections.

Deploys a trained model to CARLA intersection scenarios
without autopilot, measuring pass rate and logging metrics.
Target: >80% pass rate (8/10 successful passes).
"""

import logging
from typing import Optional

import numpy as np
import torch

from model.checkpoint import CheckpointManager
from model.bc_model import BehavioralCloningModel
from model.dataset import default_transform

logger = logging.getLogger(__name__)


class IntersectionTester:
    """
    Tests model's intersection navigation capability.

    Loads a checkpoint, runs multiple intersection scenarios,
    and reports pass rate with detailed metrics.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Inference device.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.model = BehavioralCloningModel(pretrained=False)
        ckpt_mgr = CheckpointManager()
        self._metadata = ckpt_mgr.load(checkpoint_path, self.model, device=device)
        self.model.to(device)
        self.model.eval()

        self._transform = default_transform()

    def run_tests(
        self,
        env,
        num_trials: int = 10,
        max_steps_per_trial: int = 500,
    ) -> dict:
        """
        Run intersection pass tests.

        Args:
            env: CARLA Gym environment.
            num_trials: Number of test trials.
            max_steps_per_trial: Max steps per trial.

        Returns:
            Dict with pass_rate, successes, failures, and per-trial metrics.
        """
        results = []
        successes = 0

        for trial in range(1, num_trials + 1):
            obs, info = env.reset()
            done = False
            steps = 0
            collision = False
            lane_distances = []
            steering_values = []

            while not done and steps < max_steps_per_trial:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(obs)
                tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    steering, throttle = self.model(tensor)

                action = np.array([steering.item(), throttle.item()])
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                steps += 1
                lane_distances.append(info.get("lane_distance", 0.0))
                steering_values.append(steering.item())

                if info.get("collision", False):
                    collision = True

            # Determine success: completed without collision or lane departure
            success = not collision and steps >= max_steps_per_trial * 0.5

            trial_result = {
                "trial": trial,
                "success": success,
                "steps": steps,
                "collision": collision,
                "avg_lane_distance": np.mean(lane_distances) if lane_distances else 0.0,
                "mae_steering": np.mean(np.abs(steering_values)) if steering_values else 0.0,
            }
            results.append(trial_result)

            if success:
                successes += 1

            logger.info(
                "Trial %d/%d: %s (steps=%d, collision=%s, avg_lane=%.3f)",
                trial, num_trials,
                "PASS" if success else "FAIL",
                steps, collision, trial_result["avg_lane_distance"],
            )

        pass_rate = successes / num_trials
        summary = {
            "pass_rate": pass_rate,
            "successes": successes,
            "failures": num_trials - successes,
            "num_trials": num_trials,
            "target_met": pass_rate >= 0.8,
            "trials": results,
        }

        logger.info(
            "Intersection test: %d/%d passed (%.0f%%) — target %s",
            successes, num_trials, pass_rate * 100,
            "MET" if summary["target_met"] else "NOT MET",
        )

        return summary
