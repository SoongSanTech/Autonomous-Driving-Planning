"""Unit tests for ExperimentLogger."""

import json
import os
import tempfile

import numpy as np
import pytest

from experiment.experiment_logger import ExperimentLogger


@pytest.fixture
def logger_instance(tmp_path):
    """Create ExperimentLogger with temp directories."""
    db_path = str(tmp_path / "test.db")
    json_dir = str(tmp_path / "logs")
    return ExperimentLogger(db_path=db_path, json_dir=json_dir)


class TestCreateExperiment:
    def test_create_returns_uuid(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {"lr": 1e-4})
        assert isinstance(eid, str)
        assert len(eid) == 36  # UUID format

    def test_create_stores_config(self, logger_instance):
        config = {"lr": 1e-4, "batch_size": 32, "augment": True}
        eid = logger_instance.create_experiment("bc_training", "test purpose", config)
        exp = logger_instance.get_experiment(eid)
        assert exp["config"]["lr"] == 1e-4
        assert exp["config"]["batch_size"] == 32
        assert exp["config"]["augment"] is True

    def test_create_with_parent_id(self, logger_instance):
        parent = logger_instance.create_experiment("bc_grid_search", "grid", {})
        child = logger_instance.create_experiment("bc_training", "child", {"lr": 1e-4}, parent_id=parent)
        exp = logger_instance.get_experiment(child)
        assert exp["parent_id"] == parent


class TestGetExperiment:
    def test_get_nonexistent_raises(self, logger_instance):
        with pytest.raises(KeyError):
            logger_instance.get_experiment("nonexistent-id")

    def test_get_returns_all_fields(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {"lr": 0.001})
        exp = logger_instance.get_experiment(eid)
        assert exp["experiment_id"] == eid
        assert exp["experiment_type"] == "bc_training"
        assert exp["purpose"] == "test"
        assert "created_at" in exp
        assert exp["status"] == "running"
        assert exp["config"] == {"lr": 0.001}


class TestLogMetrics:
    def test_log_and_retrieve_metrics(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {})
        logger_instance.log_metrics(eid, {"mae_steering": 0.087, "mae_throttle": 0.065})
        exp = logger_instance.get_experiment(eid)
        assert abs(exp["metrics"]["mae_steering"] - 0.087) < 1e-6
        assert abs(exp["metrics"]["mae_throttle"] - 0.065) < 1e-6

    def test_log_numpy_metrics(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {})
        logger_instance.log_metrics(eid, {
            "val_loss": np.float64(0.0234),
            "epoch": np.int64(35),
        })
        exp = logger_instance.get_experiment(eid)
        assert abs(exp["metrics"]["val_loss"] - 0.0234) < 1e-6

    def test_empty_db_list(self, logger_instance):
        result = logger_instance.list_experiments()
        assert result == []


class TestLogAnalysis:
    def test_log_and_retrieve_analysis(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {})
        logger_instance.log_analysis(eid, "과적합 미감지", ["데이터 증강 유지"])
        exp = logger_instance.get_experiment(eid)
        assert exp["analysis"]["text"] == "과적합 미감지"
        assert exp["analysis"]["recommendations"] == ["데이터 증강 유지"]


class TestLogCliCommand:
    def test_log_and_retrieve_cli(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {"lr": 1e-4})
        cmd = "python -m model.train_bc --lr 1e-4 --batch_size 32"
        logger_instance.log_cli_command(eid, cmd)
        exp = logger_instance.get_experiment(eid)
        assert cmd in exp["cli_commands"]


class TestListExperiments:
    def test_list_all(self, logger_instance):
        logger_instance.create_experiment("bc_training", "a", {})
        logger_instance.create_experiment("rl_training", "b", {})
        result = logger_instance.list_experiments()
        assert len(result) == 2

    def test_list_by_type(self, logger_instance):
        logger_instance.create_experiment("bc_training", "a", {})
        logger_instance.create_experiment("rl_training", "b", {})
        logger_instance.create_experiment("bc_training", "c", {})
        result = logger_instance.list_experiments("bc_training")
        assert len(result) == 2
        assert all(r["experiment_type"] == "bc_training" for r in result)


class TestCompareExperiments:
    def test_compare_two_experiments(self, logger_instance):
        e1 = logger_instance.create_experiment("bc_training", "baseline", {})
        e2 = logger_instance.create_experiment("bc_training", "improved", {})
        logger_instance.log_metrics(e1, {"mae_steering": 0.12, "val_loss": 0.05})
        logger_instance.log_metrics(e2, {"mae_steering": 0.08, "val_loss": 0.03})

        result = logger_instance.compare_experiments([e1, e2])
        assert result["comparisons"]["mae_steering"]["improved"] is True
        assert result["comparisons"]["val_loss"]["improved"] is True
        assert abs(result["comparisons"]["mae_steering"]["delta"] - (-0.04)) < 1e-6

    def test_compare_requires_two(self, logger_instance):
        e1 = logger_instance.create_experiment("bc_training", "a", {})
        with pytest.raises(ValueError):
            logger_instance.compare_experiments([e1])


class TestGenerateReport:
    def test_report_contains_experiments(self, logger_instance):
        e1 = logger_instance.create_experiment("bc_training", "first", {"lr": 1e-4})
        logger_instance.log_metrics(e1, {"mae_steering": 0.1})
        report = logger_instance.generate_report()
        assert "bc_training" in report
        assert "first" in report
        assert "mae_steering" in report

    def test_report_time_ordered(self, logger_instance):
        import time
        e1 = logger_instance.create_experiment("bc_training", "first", {})
        time.sleep(0.01)
        e2 = logger_instance.create_experiment("bc_training", "second", {})
        report = logger_instance.generate_report()
        pos1 = report.find("first")
        pos2 = report.find("second")
        assert pos1 < pos2

    def test_empty_report(self, logger_instance):
        report = logger_instance.generate_report()
        assert "총 실험 수: 0" in report


class TestUpdateStatus:
    def test_update_status(self, logger_instance):
        eid = logger_instance.create_experiment("bc_training", "test", {})
        logger_instance.update_status(eid, "completed")
        exp = logger_instance.get_experiment(eid)
        assert exp["status"] == "completed"
