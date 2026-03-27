"""
BC Inference: Real-time control loop for CARLA.

Loads a trained BC model from checkpoint and runs inference
on Front RGB camera images, outputting steering/throttle controls.
Measures inference latency per frame.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.bc_model import BehavioralCloningModel
from model.checkpoint import CheckpointManager
from model.dataset import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class BCInferenceEngine:
    """
    Real-time inference engine for Behavioral Cloning model.

    Loads a checkpoint, preprocesses camera images, and predicts
    steering/throttle controls with latency measurement.

    Args:
        checkpoint_path: Path to BC model checkpoint.
        device: Inference device ('cuda' or 'cpu').
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device

        self.model = BehavioralCloningModel(pretrained=False)
        ckpt_mgr = CheckpointManager()
        self._metadata = ckpt_mgr.load(checkpoint_path, self.model, device=device)

        self.model.to(device)
        self.model.eval()

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._latency_history: list[float] = []

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict steering and throttle from a camera image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            (steering, throttle, latency_ms)
        """
        start = time.perf_counter()

        pil_image = Image.fromarray(image)
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

        steering, throttle = self.model(tensor)

        latency_ms = (time.perf_counter() - start) * 1000
        self._latency_history.append(latency_ms)

        return steering.item(), throttle.item(), latency_ms

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> Tuple[float, float, float]:
        """
        Predict from a pre-processed tensor (batch=1).

        Args:
            tensor: (1, 3, 224, 224) normalized tensor.

        Returns:
            (steering, throttle, latency_ms)
        """
        start = time.perf_counter()
        tensor = tensor.to(self.device)
        steering, throttle = self.model(tensor)
        latency_ms = (time.perf_counter() - start) * 1000
        self._latency_history.append(latency_ms)
        return steering.item(), throttle.item(), latency_ms

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    @property
    def metadata(self) -> dict:
        """Checkpoint metadata."""
        return self._metadata


class CARLAControlLoop:
    """
    CARLA real-time control loop using BC inference.

    Connects to CARLA, receives camera images, runs inference,
    and applies vehicle controls.

    Args:
        engine: BCInferenceEngine instance.
        host: CARLA server host.
        port: CARLA server port.
    """

    def __init__(
        self,
        engine: BCInferenceEngine,
        host: str = "localhost",
        port: int = 2000,
    ):
        self.engine = engine
        self.host = host
        self.port = port
        self._running = False

    def run(self, max_steps: Optional[int] = None):
        """
        Run the control loop.

        Requires CARLA server to be running. Connects, spawns vehicle,
        attaches camera, and runs inference loop.

        Args:
            max_steps: Maximum number of control steps (None = infinite).
        """
        try:
            import carla
        except ImportError:
            raise RuntimeError("CARLA Python API not available")

        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)
        world = client.get_world()

        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")

        # Attach camera
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        latest_image = [None]

        def on_image(data):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))[:, :, :3]
            latest_image[0] = array

        camera.listen(on_image)
        self._running = True

        try:
            step = 0
            while self._running:
                if max_steps and step >= max_steps:
                    break

                world.tick()

                if latest_image[0] is None:
                    continue

                steering, throttle, latency = self.engine.predict(latest_image[0])

                control = carla.VehicleControl(
                    steer=float(steering),
                    throttle=float(throttle),
                    brake=0.0,
                )
                vehicle.apply_control(control)

                if step % 100 == 0:
                    logger.info(
                        "Step %d: steer=%.3f, throttle=%.3f, latency=%.1fms",
                        step, steering, throttle, latency,
                    )

                step += 1

        finally:
            self._running = False
            camera.stop()
            camera.destroy()
            vehicle.destroy()
            logger.info(
                "Control loop ended. Avg latency: %.1fms",
                self.engine.avg_latency_ms,
            )

    def stop(self):
        """Stop the control loop."""
        self._running = False
