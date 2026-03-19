"""
RewardFunction: 4-component weighted reward for RL driving.

R = w_lane × R_lane + w_collision × R_collision
  + w_steering × R_steering + w_progress × R_progress

Where:
  R_lane = -|d_lane|
  R_collision = -100 if collision, 0 otherwise
  R_steering = -|steer| if |steer| > threshold, 0 otherwise
  R_progress = v_forward × cos(heading_error)
"""

import math


class RewardFunction:
    """
    Configurable reward function for autonomous driving RL.

    Args:
        w_lane: Weight for lane centering component.
        w_collision: Weight for collision penalty.
        w_steering: Weight for steering smoothness.
        w_progress: Weight for forward progress.
        steering_threshold: Steering magnitude above which penalty applies.
        collision_penalty: Penalty value for collision events.
    """

    def __init__(
        self,
        w_lane: float = 1.0,
        w_collision: float = 1.0,
        w_steering: float = 0.5,
        w_progress: float = 0.1,
        steering_threshold: float = 0.3,
        collision_penalty: float = 100.0,
    ):
        self.w_lane = w_lane
        self.w_collision = w_collision
        self.w_steering = w_steering
        self.w_progress = w_progress
        self.steering_threshold = steering_threshold
        self.collision_penalty = collision_penalty

    def compute(self, state_info: dict, action) -> float:
        """
        Compute scalar reward.

        Args:
            state_info: Dict with keys:
                - lane_distance (float): distance from lane center in meters
                - collision (bool): whether collision occurred
                - velocity (float): forward velocity in m/s
                - heading_error (float): heading error in radians
            action: [steering, throttle] array or tuple.

        Returns:
            Scalar reward value.
        """
        lane_dist = state_info.get("lane_distance", 0.0)
        collision = state_info.get("collision", False)
        velocity = state_info.get("velocity", 0.0)
        heading_error = state_info.get("heading_error", 0.0)
        steering = float(action[0]) if action is not None else 0.0

        r_lane = -abs(lane_dist)
        r_collision = -self.collision_penalty if collision else 0.0
        r_steering = -abs(steering) if abs(steering) > self.steering_threshold else 0.0
        r_progress = velocity * math.cos(heading_error)

        reward = (
            self.w_lane * r_lane
            + self.w_collision * r_collision
            + self.w_steering * r_steering
            + self.w_progress * r_progress
        )

        return float(reward)
