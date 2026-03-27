"""
DataValidator: 수집된 데이터의 품질을 정량적으로 검증.

이미지 무결성, 레이블 범위, 타이밍 간격, 분포 분석을 수행하고
DataValidationReport를 생성한다.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DataValidationReport:
    """데이터 품질 검증 보고서."""
    session_dir: str
    total_frames: int = 0
    valid_frames: int = 0
    corrupted_frames: int = 0
    out_of_range_steering: int = 0
    out_of_range_throttle: int = 0
    timing_anomalies: int = 0
    frame_drop_rate: float = 0.0
    steering_mean: float = 0.0
    steering_std: float = 0.0
    throttle_mean: float = 0.0
    throttle_std: float = 0.0
    needs_recollection: bool = False
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """데이터 품질 검증."""

    def __init__(self, experiment_logger=None):
        self._logger = experiment_logger

    def validate_session(self, session_dir: str) -> DataValidationReport:
        """세션 디렉토리의 데이터 품질 전체 검증."""
        session_path = Path(session_dir)
        if not session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        report = DataValidationReport(session_dir=session_dir)

        # Image validation
        img_result = self.validate_images(session_dir)
        report.total_frames = img_result["total"]
        report.valid_frames = img_result["valid"]
        report.corrupted_frames = img_result["corrupted"]

        # Label validation
        label_result = self.validate_labels(session_dir)
        report.out_of_range_steering = label_result["out_of_range_steering"]
        report.out_of_range_throttle = label_result["out_of_range_throttle"]
        report.timing_anomalies = label_result["timing_anomalies"]

        # Distribution analysis
        dist_result = self.analyze_distribution(session_dir)
        report.steering_mean = dist_result.get("steering_mean", 0.0)
        report.steering_std = dist_result.get("steering_std", 0.0)
        report.throttle_mean = dist_result.get("throttle_mean", 0.0)
        report.throttle_std = dist_result.get("throttle_std", 0.0)

        # Frame drop rate
        if report.total_frames > 0:
            report.frame_drop_rate = report.corrupted_frames / report.total_frames
        else:
            report.frame_drop_rate = 0.0

        # Recollection check: corrupted > 5%
        if report.total_frames > 0 and (report.corrupted_frames / report.total_frames) > 0.05:
            report.needs_recollection = True
            msg = (
                f"손상 비율 {report.corrupted_frames}/{report.total_frames} "
                f"({report.corrupted_frames / report.total_frames * 100:.1f}%) > 5%. 재수집 권고."
            )
            report.warnings.append(msg)
            logger.warning(msg)

        return report

    def validate_images(self, session_dir: str) -> dict:
        """이미지 무결성 검사."""
        session_path = Path(session_dir)
        # Support both 'front/' and 'images/' subdirectories
        images_dir = session_path / "front"
        if not images_dir.exists():
            images_dir = session_path / "images"
        if not images_dir.exists():
            return {"total": 0, "valid": 0, "corrupted": 0}

        png_files = sorted(images_dir.glob("*.png"))
        total = len(png_files)
        valid = 0
        corrupted = 0

        for img_path in png_files:
            if img_path.stat().st_size == 0:
                corrupted += 1
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid += 1
            except Exception:
                corrupted += 1

        return {"total": total, "valid": valid, "corrupted": corrupted}

    def validate_labels(self, session_dir: str) -> dict:
        """레이블 범위 및 타이밍 검증."""
        session_path = Path(session_dir)
        csv_path = session_path / "labels" / "driving_log.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Label file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        out_of_range_steering = 0
        out_of_range_throttle = 0
        timing_anomalies = 0

        if "steering" in df.columns:
            out_of_range_steering = int(((df["steering"] < -1.0) | (df["steering"] > 1.0)).sum())
        if "throttle" in df.columns:
            out_of_range_throttle = int(((df["throttle"] < 0.0) | (df["throttle"] > 1.0)).sum())

        # Timing validation: extract timestamps from image_filename (e.g. "12345.png" → 12345 ms)
        if "image_filename" in df.columns and len(df) > 1:
            try:
                timestamps = df["image_filename"].apply(
                    lambda x: int(str(x).replace(".png", ""))
                ).values
                intervals = np.diff(timestamps)
                timing_anomalies = int(((intervals < 80) | (intervals > 120)).sum())
            except (ValueError, TypeError):
                pass

        return {
            "out_of_range_steering": out_of_range_steering,
            "out_of_range_throttle": out_of_range_throttle,
            "timing_anomalies": timing_anomalies,
        }

    def analyze_distribution(self, session_dir: str) -> dict:
        """steering/throttle 분포 분석."""
        session_path = Path(session_dir)
        csv_path = session_path / "labels" / "driving_log.csv"
        if not csv_path.exists():
            return {
                "steering_mean": 0.0, "steering_std": 0.0,
                "throttle_mean": 0.0, "throttle_std": 0.0,
                "steering_histogram": {}, "throttle_histogram": {},
            }

        df = pd.read_csv(csv_path)
        result = {}

        for col in ("steering", "throttle"):
            if col not in df.columns or len(df[col]) == 0:
                result[f"{col}_mean"] = 0.0
                result[f"{col}_std"] = 0.0
                result[f"{col}_histogram"] = {"counts": [], "bin_edges": []}
                continue

            values = df[col].dropna().values.astype(float)
            result[f"{col}_mean"] = float(np.mean(values)) if len(values) > 0 else 0.0
            result[f"{col}_std"] = float(np.std(values)) if len(values) > 0 else 0.0

            if len(values) > 0:
                counts, bin_edges = np.histogram(values, bins=20)
                result[f"{col}_histogram"] = {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            else:
                result[f"{col}_histogram"] = {"counts": [], "bin_edges": []}

        return result
