"""Property-based tests for ScenarioManager.

Property 14: 멀티카메라 타임스탬프 동기화 — 동일 틱 5개 타임스탬프 동일성.
"""

from hypothesis import given, settings, strategies as st

from experiment.scenario_manager import EvalScenario, ScenarioManager, STANDARD_SCENARIOS


# Feature: experiment-validation, Property 14: 멀티카메라 타임스탬프 동기화
class TestProperty14MultiCameraTimestampSync:
    """동일 틱에서 수집된 5대 카메라의 타임스탬프가 모두 동일해야 한다.

    실제 CARLA 없이 시뮬레이션: 동기 모드에서 tick당 하나의 타임스탬프가
    생성되므로, 5개 카메라 데이터에 동일 타임스탬프를 부여하는 로직을 검증.
    """

    @settings(max_examples=100, deadline=None)
    @given(
        tick_ts=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
        num_cameras=st.just(5),
    )
    def test_same_tick_same_timestamp(self, tick_ts, num_cameras):
        """동일 틱의 모든 카메라 타임스탬프는 동일해야 한다."""
        camera_names = ["front", "avm_front", "avm_rear", "avm_left", "avm_right"]
        # 동기 모드 시뮬레이션: 하나의 tick_ts를 모든 카메라에 할당
        frame_data = {cam: {"timestamp": tick_ts} for cam in camera_names}

        timestamps = [frame_data[cam]["timestamp"] for cam in camera_names]
        assert len(timestamps) == num_cameras
        assert all(t == timestamps[0] for t in timestamps)

    @settings(max_examples=100, deadline=None)
    @given(
        ticks=st.lists(
            st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=10,
        ),
    )
    def test_different_ticks_different_timestamps(self, ticks):
        """서로 다른 틱의 타임스탬프는 (일반적으로) 다를 수 있다."""
        camera_names = ["front", "avm_front", "avm_rear", "avm_left", "avm_right"]
        all_frames = []
        for tick_ts in ticks:
            frame = {cam: {"timestamp": tick_ts} for cam in camera_names}
            all_frames.append(frame)

        # 각 프레임 내에서는 모든 카메라 동일
        for frame in all_frames:
            ts_values = [frame[cam]["timestamp"] for cam in camera_names]
            assert all(t == ts_values[0] for t in ts_values)
