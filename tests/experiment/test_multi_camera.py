"""Unit tests for MultiCameraPipeline."""

from pathlib import Path

import pytest

from experiment.multi_camera import MultiCameraPipeline, CAMERA_CONFIGS


class TestCameraConfigs:
    def test_five_cameras_defined(self):
        assert len(CAMERA_CONFIGS) == 5

    def test_camera_names(self):
        expected = {"front", "avm_front", "avm_rear", "avm_left", "avm_right"}
        assert set(CAMERA_CONFIGS.keys()) == expected

    def test_front_camera_resolution(self):
        assert CAMERA_CONFIGS["front"]["res"] == (800, 600)
        assert CAMERA_CONFIGS["front"]["fov"] == 90

    def test_avm_camera_resolution(self):
        for name in ["avm_front", "avm_rear", "avm_left", "avm_right"]:
            assert CAMERA_CONFIGS[name]["res"] == (400, 300)
            assert CAMERA_CONFIGS[name]["fov"] == 120

    def test_camera_positions(self):
        # front: x=1.5, z=2.4
        assert CAMERA_CONFIGS["front"]["pos"] == (1.5, 0.0, 2.4)
        # avm_front: x=2.0, z=0.5
        assert CAMERA_CONFIGS["avm_front"]["pos"] == (2.0, 0.0, 0.5)
        # avm_rear: x=-2.0, z=0.5
        assert CAMERA_CONFIGS["avm_rear"]["pos"] == (-2.0, 0.0, 0.5)
        # avm_left: y=-1.0, z=0.5
        assert CAMERA_CONFIGS["avm_left"]["pos"] == (0.0, -1.0, 0.5)
        # avm_right: y=1.0, z=0.5
        assert CAMERA_CONFIGS["avm_right"]["pos"] == (0.0, 1.0, 0.5)


class TestDirectoryCreation:
    def test_creates_subdirectories(self, tmp_path):
        output_dir = str(tmp_path / "multi_cam")
        pipeline = MultiCameraPipeline(output_dir=output_dir)

        for cam_name in CAMERA_CONFIGS:
            cam_dir = Path(output_dir) / cam_name
            assert cam_dir.exists()
            assert cam_dir.is_dir()


class TestFrameDropStats:
    def test_initial_stats_zero(self, tmp_path):
        pipeline = MultiCameraPipeline(output_dir=str(tmp_path / "mc"))
        stats = pipeline.get_frame_drop_stats()
        for cam_name in CAMERA_CONFIGS:
            assert stats[cam_name] == 0.0

    def test_drop_rate_calculation(self, tmp_path):
        pipeline = MultiCameraPipeline(output_dir=str(tmp_path / "mc"))
        # Simulate: front got 90 frames, dropped 10
        pipeline._frame_counts["front"] = 90
        pipeline._drop_counts["front"] = 10
        stats = pipeline.get_frame_drop_stats()
        assert abs(stats["front"] - 0.1) < 1e-6  # 10%

    def test_no_frames_no_division_error(self, tmp_path):
        pipeline = MultiCameraPipeline(output_dir=str(tmp_path / "mc"))
        pipeline._frame_counts["front"] = 0
        pipeline._drop_counts["front"] = 0
        stats = pipeline.get_frame_drop_stats()
        assert stats["front"] == 0.0
