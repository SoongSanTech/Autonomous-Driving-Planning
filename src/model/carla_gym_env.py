"""
CARLAGymEnv: Gymnasium-compatible wrapper for CARLA simulator.

observation_space: Box(0, 255, (224, 224, 3), uint8)
action_space: Box([-1.0, 0.0], [1.0, 1.0], float32)

Episode termination: collision, lane departure > 3m, timeout 1000 steps.
Random weather per episode for robustness.
"""

import logging
import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Weather presets for randomization
WEATHER_PRESETS = [
    "ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset",
]


class CARLAGymEnv(gym.Env):
    """
    CARLA simulator wrapped as a Gymnasium environment.

    Connects to a running CARLA server, spawns a vehicle with
    front RGB camera and collision sensor, and provides step/reset
    interface for RL training.

    Args:
        host: CARLA server hostname.
        port: CARLA server port.
        town: CARLA map name.
        max_steps: Maximum steps per episode (timeout).
        target_fps: Simulation tick rate.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town03",
        max_steps: int = 1000,
        target_fps: float = 10.0,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.town = town
        self.max_steps = max_steps
        self.target_fps = target_fps

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=np.float32([-1.0, 0.0]),
            high=np.float32([1.0, 1.0]),
            dtype=np.float32,
        )

        # CARLA objects (initialized on first reset)
        self._client = None
        self._world = None
        self._vehicle = None
        self._camera = None
        self._collision_sensor = None
        self._latest_image: Optional[np.ndarray] = None
        self._collision_occurred = False
        self._episode_step = 0
        self._connected = False

    def _connect(self):
        """Connect to CARLA server with retry logic."""
        try:
            import carla
        except ImportError:
            raise RuntimeError(
                "CARLA Python API not available. "
                "Install with: pip install carla==0.9.15"
            )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._client = carla.Client(self.host, self.port)
                self._client.set_timeout(10.0)
                self._world = self._client.get_world()
                self._connected = True
                logger.info("Connected to CARLA at %s:%d", self.host, self.port)
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "CARLA connection attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Failed to connect to CARLA at {self.host}:{self.port} "
            f"after {max_retries} attempts"
        )

    def _set_random_weather(self):
        """Set random weather for episode diversity."""
        import carla
        preset_name = np.random.choice(WEATHER_PRESETS)
        weather = getattr(carla.WeatherParameters, preset_name)
        self._world.set_weather(weather)
        logger.debug("Weather set to %s", preset_name)

    def _spawn_vehicle(self):
        """Spawn ego vehicle at random spawn point."""
        import carla

        bp_lib = self._world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)

        for sp in spawn_points:
            self._vehicle = self._world.try_spawn_actor(vehicle_bp, sp)
            if self._vehicle is not None:
                return

        raise RuntimeError("Failed to spawn vehicle at any spawn point")

    def _attach_sensors(self):
        """Attach RGB camera and collision sensor."""
        import carla

        bp_lib = self._world.get_blueprint_library()

        # RGB Camera (800×600, FOV 90°)
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self._camera = self._world.spawn_actor(
            camera_bp, camera_transform, attach_to=self._vehicle
        )

        self._latest_image = None
        self._camera.listen(self._on_camera_image)

        # Collision sensor
        collision_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._vehicle
        )
        self._collision_occurred = False
        self._collision_sensor.listen(self._on_collision)

    def _on_camera_image(self, data):
        """Camera callback: convert raw data to numpy array."""
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((data.height, data.width, 4))[:, :, :3]  # Drop alpha
        # Resize to 224×224
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(array)
        pil_img = pil_img.resize((224, 224), PILImage.BILINEAR)
        self._latest_image = np.array(pil_img)

    def _on_collision(self, event):
        """Collision callback."""
        self._collision_occurred = True

    def _get_lane_distance(self) -> float:
        """Get perpendicular distance from lane center."""
        if self._vehicle is None or self._world is None:
            return 0.0

        vehicle_loc = self._vehicle.get_location()
        waypoint = self._world.get_map().get_waypoint(vehicle_loc)
        if waypoint is None:
            return 0.0

        wp_loc = waypoint.transform.location
        dx = vehicle_loc.x - wp_loc.x
        dy = vehicle_loc.y - wp_loc.y
        return (dx ** 2 + dy ** 2) ** 0.5

    def _get_velocity(self) -> float:
        """Get vehicle forward velocity in m/s."""
        if self._vehicle is None:
            return 0.0
        vel = self._vehicle.get_velocity()
        return (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5

    def _get_heading_error(self) -> float:
        """Get heading error relative to road direction."""
        import math

        if self._vehicle is None or self._world is None:
            return 0.0

        vehicle_transform = self._vehicle.get_transform()
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        waypoint = self._world.get_map().get_waypoint(vehicle_transform.location)
        if waypoint is None:
            return 0.0

        road_yaw = math.radians(waypoint.transform.rotation.yaw)
        error = vehicle_yaw - road_yaw

        # Normalize to [-pi, pi]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        return error

    def _cleanup(self):
        """Destroy all CARLA actors."""
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
            self._camera = None

        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None

        if self._vehicle is not None:
            self._vehicle.destroy()
            self._vehicle = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.

        Returns:
            (observation, info) — observation is (224, 224, 3) uint8.
        """
        super().reset(seed=seed)

        self._cleanup()

        if not self._connected:
            self._connect()

        self._set_random_weather()
        self._spawn_vehicle()
        self._attach_sensors()
        self._episode_step = 0
        self._collision_occurred = False

        # Wait for first camera frame
        for _ in range(50):
            self._world.tick()
            if self._latest_image is not None:
                break

        obs = self._latest_image if self._latest_image is not None else np.zeros(
            (224, 224, 3), dtype=np.uint8
        )

        info = {
            "lane_distance": 0.0,
            "collision": False,
            "velocity": 0.0,
            "heading_error": 0.0,
            "episode_step": 0,
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.

        Args:
            action: [steering, throttle] array.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        import carla

        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))

        control = carla.VehicleControl(steer=steering, throttle=throttle, brake=0.0)
        self._vehicle.apply_control(control)
        self._world.tick()

        self._episode_step += 1

        # Get state info
        lane_distance = self._get_lane_distance()
        velocity = self._get_velocity()
        heading_error = self._get_heading_error()

        info = {
            "lane_distance": lane_distance,
            "collision": self._collision_occurred,
            "velocity": velocity,
            "heading_error": heading_error,
            "episode_step": self._episode_step,
        }

        # Compute reward (import here to avoid circular)
        from model.reward import RewardFunction
        reward_fn = RewardFunction()
        reward = reward_fn.compute(info, action)

        # Termination conditions
        terminated = False
        truncated = False

        if self._collision_occurred:
            terminated = True
        elif lane_distance > 3.0:
            terminated = True
        elif self._episode_step >= self.max_steps:
            truncated = True

        obs = self._latest_image if self._latest_image is not None else np.zeros(
            (224, 224, 3), dtype=np.uint8
        )

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up CARLA resources."""
        self._cleanup()
        self._connected = False
        logger.info("CARLA environment closed")
