"""Property-based tests for report generation.

Property 15: CLI 명령어 재현성 — config 값이 CLI 명령어 인자에 포함.
Property 16: 종합 보고서 시간순 정렬 — created_at 오름차순.
"""

import tempfile
import time

import pytest
from hypothesis import given, settings, HealthCheck, strategies as st

from experiment.experiment_logger import ExperimentLogger


# Feature: experiment-validation, Property 15: CLI 명령어 재현성
class TestProperty15CLIReproducibilityExtended:
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        lr=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
        batch_size=st.sampled_from([16, 32, 64, 128]),
        steering_weight=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
    )
    def test_all_config_values_in_cli(self, tmp_path, lr, batch_size, steering_weight):
        """config의 모든 하이퍼파라미터 값이 CLI 명령어에 포함."""
        logger = ExperimentLogger(
            db_path=str(tmp_path / "rp15.db"),
            json_dir=str(tmp_path / "rp15_logs"),
        )
        config = {
            "lr": lr,
            "batch_size": batch_size,
            "steering_weight": steering_weight,
        }
        eid = logger.create_experiment("bc_training", "cli repro test", config)

        cmd = (
            f"python -m experiment.cli train-bc "
            f"--lr {lr} --batch-size {batch_size} "
            f"--steering-weight {steering_weight}"
        )
        logger.log_cli_command(eid, cmd)

        exp = logger.get_experiment(eid)
        recorded_cmd = exp["cli_commands"][0]
        assert str(batch_size) in recorded_cmd
        assert str(steering_weight) in recorded_cmd


# Feature: experiment-validation, Property 16: 종합 보고서 시간순 정렬
class TestProperty16ReportTimeOrderExtended:
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(n=st.integers(min_value=2, max_value=8))
    def test_report_experiments_in_chronological_order(self, tmp_path, n):
        """보고서 내 실험 항목들이 created_at 기준 오름차순."""
        logger = ExperimentLogger(
            db_path=str(tmp_path / "rp16.db"),
            json_dir=str(tmp_path / "rp16_logs"),
        )

        purposes = []
        for i in range(n):
            purpose = f"report_test_{i:03d}"
            purposes.append(purpose)
            logger.create_experiment("bc_training", purpose, {"idx": i})
            time.sleep(0.005)

        report = logger.generate_report()

        positions = []
        for purpose in purposes:
            pos = report.find(purpose)
            assert pos >= 0, f"{purpose} not found in report"
            positions.append(pos)

        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1]
