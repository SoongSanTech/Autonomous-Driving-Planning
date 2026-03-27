"""Property-based tests for ExperimentLogger.

Properties 5, 6, 8, 15, 16 from design doc.
"""

import shutil
import tempfile
import time

import pytest
from hypothesis import given, settings, strategies as st

from experiment.experiment_logger import ExperimentLogger


# Strategies
experiment_types = st.sampled_from([
    "bc_training", "rl_training", "bc_inference",
    "data_collection", "data_validation",
    "bc_grid_search", "rl_reward_search", "condition_optimization",
])

config_values = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
)

config_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N", "Pd"))),
    values=config_values,
    min_size=1,
    max_size=5,
)

metric_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N", "Pd"))),
    values=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=5,
)


# Feature: experiment-validation, Property 5: 실험 기록 완전성
class TestProperty5RecordCompleteness:
    @settings(max_examples=100, deadline=None)
    @given(exp_type=experiment_types, config=config_strategy, metrics=metric_strategy)
    def test_all_keys_preserved(self, exp_type, config, metrics):
        td = tempfile.mkdtemp()
        try:
            logger = ExperimentLogger(
                db_path=f"{td}/p5.db",
                json_dir=f"{td}/p5_logs",
            )
            eid = logger.create_experiment(exp_type, "property test", config)
            logger.log_metrics(eid, metrics)

            exp = logger.get_experiment(eid)
            assert exp["experiment_id"] == eid
            assert "created_at" in exp
            for key in config:
                assert key in exp["config"]
            for key in metrics:
                assert key in exp["metrics"]
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 6: 실험 기록 라운드트립
class TestProperty6RoundTrip:
    @settings(max_examples=100, deadline=None)
    @given(config=config_strategy, metrics=metric_strategy)
    def test_json_sqlite_roundtrip(self, config, metrics):
        td = tempfile.mkdtemp()
        try:
            logger = ExperimentLogger(
                db_path=f"{td}/p6.db",
                json_dir=f"{td}/p6_logs",
            )
            eid = logger.create_experiment("bc_training", "roundtrip", config)
            logger.log_metrics(eid, metrics)

            # SQLite roundtrip
            exp_db = logger.get_experiment(eid)
            for key, val in config.items():
                db_val = exp_db["config"][key]
                if isinstance(val, float):
                    assert abs(db_val - val) < 1e-6 or str(db_val) == str(val)
                else:
                    assert db_val == val

            for key, val in metrics.items():
                assert abs(exp_db["metrics"][key] - float(val)) < 1e-4

            # JSON roundtrip
            json_data = logger._load_json(eid)
            assert json_data is not None
            for key in metrics:
                assert key in json_data["metrics"]
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 8: 실험 비교
class TestProperty8Comparison:
    @settings(max_examples=100, deadline=None)
    @given(
        m1=st.dictionaries(
            keys=st.sampled_from(["mae_steering", "val_loss", "reward"]),
            values=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            min_size=1, max_size=3,
        ),
        m2=st.dictionaries(
            keys=st.sampled_from(["mae_steering", "val_loss", "reward"]),
            values=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            min_size=1, max_size=3,
        ),
    )
    def test_delta_and_improved(self, m1, m2):
        common = set(m1.keys()) & set(m2.keys())
        if not common:
            return

        td = tempfile.mkdtemp()
        try:
            logger = ExperimentLogger(
                db_path=f"{td}/p8.db",
                json_dir=f"{td}/p8_logs",
            )
            e1 = logger.create_experiment("bc_training", "a", {})
            e2 = logger.create_experiment("bc_training", "b", {})
            logger.log_metrics(e1, m1)
            logger.log_metrics(e2, m2)

            result = logger.compare_experiments([e1, e2])
            for key in common:
                comp = result["comparisons"][key]
                expected_delta = m2[key] - m1[key]
                assert abs(comp["delta"] - expected_delta) < 1e-4
                assert comp["improved"] is not None
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 15: CLI 명령어 재현성
class TestProperty15CLIReproducibility:
    @settings(max_examples=100, deadline=None)
    @given(
        config=st.fixed_dictionaries({
            "lr": st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
            "batch_size": st.sampled_from([16, 32, 64]),
        })
    )
    def test_config_in_cli_command(self, config):
        td = tempfile.mkdtemp()
        try:
            logger = ExperimentLogger(
                db_path=f"{td}/p15.db",
                json_dir=f"{td}/p15_logs",
            )
            eid = logger.create_experiment("bc_training", "cli test", config)
            cmd = f"python -m model.train_bc --lr {config['lr']} --batch_size {config['batch_size']}"
            logger.log_cli_command(eid, cmd)

            exp = logger.get_experiment(eid)
            assert len(exp["cli_commands"]) > 0
            recorded_cmd = exp["cli_commands"][0]
            assert str(config["batch_size"]) in recorded_cmd
        finally:
            shutil.rmtree(td, ignore_errors=True)


# Feature: experiment-validation, Property 16: 종합 보고서 시간순 정렬
class TestProperty16ReportTimeOrder:
    @settings(max_examples=30, deadline=None)
    @given(n=st.integers(min_value=2, max_value=5))
    def test_report_sorted_by_created_at(self, n):
        td = tempfile.mkdtemp()
        try:
            logger = ExperimentLogger(
                db_path=f"{td}/p16.db",
                json_dir=f"{td}/p16_logs",
            )
            eids = []
            for i in range(n):
                eid = logger.create_experiment("bc_training", f"exp_{i}", {"idx": i})
                eids.append(eid)
                time.sleep(0.005)

            report = logger.generate_report()
            positions = []
            for i in range(n):
                pos = report.find(f"exp_{i}")
                assert pos >= 0
                positions.append(pos)

            for i in range(len(positions) - 1):
                assert positions[i] < positions[i + 1]
        finally:
            shutil.rmtree(td, ignore_errors=True)
