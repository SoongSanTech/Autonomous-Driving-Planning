"""Property-based tests for DataValidator.

Properties 1, 2, 3, 4 from design doc.
"""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings, strategies as st, assume
from PIL import Image

from experiment.data_validator import DataValidator, DataValidationReport


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_session(base_dir, steering_vals, throttle_vals, timestamps,
                  corrupt_indices=None):
    """Build a session directory with given values."""
    session = Path(base_dir) / "session"
    if session.exists():
        shutil.rmtree(session)
    images_dir = session / "front"
    labels_dir = session / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    corrupt_indices = corrupt_indices or set()
    n = len(steering_vals)

    csv_path = labels_dir / "driving_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "speed", "steering", "throttle", "brake"])
        for i in range(n):
            fname = f"{timestamps[i]}.png"
            img_path = images_dir / fname
            if i in corrupt_indices:
                img_path.write_bytes(b"not a png at all")
            else:
                img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
                img.save(str(img_path))
            writer.writerow([fname, 10.0, steering_vals[i], throttle_vals[i], 0.0])

    return str(session)


# ── Strategies ───────────────────────────────────────────────────────────

float_arrays = st.lists(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    min_size=5, max_size=50,
)


# Feature: experiment-validation, Property 1: 분포 분석 정확성
class TestProperty1DistributionAccuracy:
    @settings(max_examples=100, deadline=None)
    @given(
        steering=float_arrays,
        throttle=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=50,
        ),
    )
    def test_histogram_sum_equals_length_and_mean_matches(self, steering, throttle):
        n = min(len(steering), len(throttle))
        steering = steering[:n]
        throttle = throttle[:n]
        timestamps = [1000 + i * 100 for i in range(n)]

        td = tempfile.mkdtemp()
        try:
            session = _make_session(td, steering, throttle, timestamps)
            v = DataValidator()
            result = v.analyze_distribution(session)

            s_counts = result["steering_histogram"]["counts"]
            assert sum(s_counts) == n

            t_counts = result["throttle_histogram"]["counts"]
            assert sum(t_counts) == n

            assert abs(result["steering_mean"] - float(np.mean(steering))) < 1e-5
            assert abs(result["throttle_mean"] - float(np.mean(throttle))) < 1e-5
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 2: 이미지 무결성 검증
class TestProperty2ImageIntegrity:
    @settings(max_examples=50, deadline=None)
    @given(
        num_valid=st.integers(min_value=1, max_value=10),
        num_corrupt=st.integers(min_value=0, max_value=5),
    )
    def test_valid_and_corrupt_classification(self, num_valid, num_corrupt):
        total = num_valid + num_corrupt
        corrupt_indices = set(range(num_valid, total))
        steering = [0.0] * total
        throttle = [0.5] * total
        timestamps = [1000 + i * 100 for i in range(total)]

        td = tempfile.mkdtemp()
        try:
            session = _make_session(td, steering, throttle, timestamps,
                                    corrupt_indices=corrupt_indices)
            v = DataValidator()
            result = v.validate_images(session)

            assert result["total"] == total
            assert result["valid"] == num_valid
            assert result["corrupted"] == num_corrupt
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 3: 레이블 범위 및 타이밍 검증
class TestProperty3LabelRangeAndTiming:
    @settings(max_examples=100, deadline=None)
    @given(
        steering=st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=30,
        ),
        throttle=st.lists(
            st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=30,
        ),
        intervals=st.lists(
            st.integers(min_value=50, max_value=200),
            min_size=4, max_size=29,
        ),
    )
    def test_out_of_range_counts_accurate(self, steering, throttle, intervals):
        n = min(len(steering), len(throttle), len(intervals) + 1)
        steering = steering[:n]
        throttle = throttle[:n]
        intervals = intervals[:n - 1]

        timestamps = [1000]
        for iv in intervals:
            timestamps.append(timestamps[-1] + iv)

        td = tempfile.mkdtemp()
        try:
            session = _make_session(td, steering, throttle, timestamps)
            v = DataValidator()
            result = v.validate_labels(session)

            expected_steer_oor = sum(1 for s in steering if s < -1.0 or s > 1.0)
            assert result["out_of_range_steering"] == expected_steer_oor

            expected_throttle_oor = sum(1 for t in throttle if t < 0.0 or t > 1.0)
            assert result["out_of_range_throttle"] == expected_throttle_oor

            expected_timing = sum(1 for iv in intervals if iv < 80 or iv > 120)
            assert result["timing_anomalies"] == expected_timing
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 4: 검증 보고서 완전성
class TestProperty4ReportCompleteness:
    @settings(max_examples=50, deadline=None)
    @given(
        num_frames=st.integers(min_value=3, max_value=20),
        num_corrupt=st.integers(min_value=0, max_value=3),
    )
    def test_report_fields_and_invariant(self, num_frames, num_corrupt):
        assume(num_corrupt <= num_frames)
        corrupt_indices = set(range(num_corrupt))
        steering = [0.0] * num_frames
        throttle = [0.5] * num_frames
        timestamps = [1000 + i * 100 for i in range(num_frames)]

        td = tempfile.mkdtemp()
        try:
            session = _make_session(td, steering, throttle, timestamps,
                                    corrupt_indices=corrupt_indices)
            v = DataValidator()
            report = v.validate_session(session)

            assert hasattr(report, "total_frames")
            assert hasattr(report, "valid_frames")
            assert hasattr(report, "corrupted_frames")
            assert hasattr(report, "out_of_range_steering")
            assert hasattr(report, "out_of_range_throttle")
            assert hasattr(report, "timing_anomalies")

            assert report.valid_frames + report.corrupted_frames <= report.total_frames
        finally:
            shutil.rmtree(td, ignore_errors=True)
