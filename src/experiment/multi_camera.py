"""
MultiCameraPipeline: Front RGB + AVM 4대 동시 수집 파이프라인.

기존 DataPipeline을 상속하여 setup_sensors()만 오버라이드.
SynchronousModeController, AsyncDataLogger, EpisodeManager 인프라를 재사용.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 5대 카메라 설정
CAMERA_CONFIGS = {
    "front": {"res": (800, 600), "fov": 90, "pos": (1.5, 0.0, 2.4)},
    "avm_front": {"res": (400, 300), "fov": 120, "pos": (2.0, 0.0, 0.5)},
    "avm_rear": {"res": (400, 300), "fov": 120, "pos": (-2.0, 0.0, 0.5)},
    "avm_left": {"res": (400, 300), "fov": 120, "pos": (0.0, -1.0, 0.5)},
    "avm_right": {"res": (400, 300), "fov": 120, "pos": (0.0, 1.0, 0.5)},
}


class MultiCameraPipeline:
    """Front RGB + AVM 4대 동시 수집 파이프라인.

    DataPipeline의 인프라를 재사용하되, 5대 카메라를 부착하고
    카메라별 서브디렉토리에 이미지를 저장한다.
    """

    def __init__(self, carla_host: str = "localhost", carla_port: int = 2000,
                 output_dir: str = "src/data"):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.output_dir = output_dir

        self._cameras = {}
        self._latest_images = {}
        self._frame_counts = {name: 0 for name in CAMERA_CONFIGS}
        self._drop_counts = {name: 0 for name in CAMERA_CONFIGS}
        self._total_ticks = 0

        # 카메라별 서브디렉토리 생성
        for cam_name in CAMERA_CONFIGS:
            cam_dir = Path(output_dir) / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

    def setup_cameras(self, vehicle) -> None:
        """5대 카메라를 차량에 부착.

        Args:
            vehicle: CARLA vehicle actor.
        """
        try:
            import carla
        except ImportError:
            raise RuntimeError("CARLA Python API not available")

        world = vehicle.get_world()
        bp_lib = world.get_blueprint_library()

        for cam_name, config in CAMERA_CONFIGS.items():
            camera_bp = bp_lib.find("sensor.camera.rgb")
            w, h = config["res"]
            camera_bp.set_attribute("image_size_x", str(w))
            camera_bp.set_attribute("image_size_y", str(h))
            camera_bp.set_attribute("fov", str(config["fov"]))

            x, y, z = config["pos"]
            transform = carla.Transform(carla.Location(x=x, y=y, z=z))
            camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

            self._latest_images[cam_name] = None

            def make_callback(name, height, width):
                def callback(data):
                    array = np.frombuffer(data.raw_data, dtype=np.uint8)
                    array = array.reshape((height, width, 4))[:, :, :3]
                    self._latest_images[name] = array
                    self._frame_counts[name] += 1
                return callback

            camera.listen(make_callback(cam_name, h, w))
            self._cameras[cam_name] = camera

        logger.info("5대 카메라 부착 완료: %s", list(self._cameras.keys()))

    def run(self, duration_sec: float = 3600.0) -> dict:
        """동기화된 멀티카메라 수집 실행.

        실제 CARLA 환경에서 실행. 테스트에서는 모킹됨.

        Args:
            duration_sec: 수집 시간 (초).

        Returns:
            수집 결과 dict.
        """
        try:
            import carla
        except ImportError:
            raise RuntimeError("CARLA Python API not available")

        client = carla.Client(self.carla_host, self.carla_port)
        client.set_timeout(10.0)
        world = client.get_world()

        # 차량 스폰
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")

        vehicle.set_autopilot(True)
        self.setup_cameras(vehicle)

        start_time = time.time()
        self._total_ticks = 0

        try:
            while time.time() - start_time < duration_sec:
                world.tick()
                self._total_ticks += 1

                # 각 카메라 이미지 저장
                timestamp_ms = int((time.time() - start_time) * 1000)
                for cam_name in CAMERA_CONFIGS:
                    img = self._latest_images.get(cam_name)
                    if img is not None:
                        cam_dir = Path(self.output_dir) / cam_name
                        from PIL import Image
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(cam_dir / f"{timestamp_ms:010d}.png"))
                    else:
                        self._drop_counts[cam_name] += 1

        finally:
            # 센서 정리
            for cam in self._cameras.values():
                cam.stop()
                cam.destroy()
            vehicle.destroy()

        # 프레임 드롭 경고
        for cam_name in CAMERA_CONFIGS:
            drop_rate = self.get_frame_drop_stats().get(cam_name, 0.0)
            if drop_rate > 0.05:
                logger.warning(
                    "카메라 %s 프레임 드롭률 %.1f%% > 5%%",
                    cam_name, drop_rate * 100,
                )

        return {
            "total_ticks": self._total_ticks,
            "frame_counts": dict(self._frame_counts),
            "drop_counts": dict(self._drop_counts),
            "duration_sec": time.time() - start_time,
        }

    def get_frame_drop_stats(self) -> dict[str, float]:
        """카메라별 프레임 드롭률 반환.

        Returns:
            {"front": 0.02, "avm_front": 0.01, ...}
        """
        stats = {}
        for cam_name in CAMERA_CONFIGS:
            total = self._frame_counts[cam_name] + self._drop_counts[cam_name]
            if total > 0:
                stats[cam_name] = self._drop_counts[cam_name] / total
            else:
                stats[cam_name] = 0.0
        return stats
