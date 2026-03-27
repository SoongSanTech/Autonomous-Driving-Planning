"""Unit tests for GridSearchOrchestrator.

CARLA/GPU 모킹: BCTrainer, RLTrainer 등을 mock.patch로 모킹하여
파라미터 조합 순회 + 로깅 연결 로직만 검증.
"""

from unittest.mock import MagicMock, patch

import pytest

from experiment.grid_search import GridSearchOrchestrator


@pytest.fixture
def mock_logger(tmp_path):
    """ExperimentLogger mock."""
    from experiment.experiment_logger import ExperimentLogger
    logger = ExperimentLogger(
        db_path=str(tmp_path / "grid.db"),
        json_dir=str(tmp_path / "grid_logs"),
    )
    return logger


class TestGenerateCombinations:
    def test_single_param(self):
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations({"lr": [1e-4, 3e-4]})
        assert len(combos) == 2
        assert combos[0] == {"lr": 1e-4}
        assert combos[1] == {"lr": 3e-4}

    def test_multiple_params(self):
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations({
            "lr": [1e-4, 3e-4],
            "batch_size": [16, 32],
        })
        assert len(combos) == 4
        keys = {frozenset(c.keys()) for c in combos}
        assert keys == {frozenset(["lr", "batch_size"])}

    def test_full_grid_27(self):
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations({
            "lr": [5e-5, 1e-4, 3e-4],
            "batch_size": [16, 32, 64],
            "steering_weight": [1.5, 2.0, 3.0],
        })
        assert len(combos) == 27

    def test_empty_grid_raises(self):
        orch = GridSearchOrchestrator(MagicMock())
        with pytest.raises(ValueError):
            orch._generate_combinations({})

    def test_empty_values_raises(self):
        orch = GridSearchOrchestrator(MagicMock())
        with pytest.raises(ValueError):
            orch._generate_combinations({"lr": []})

    def test_all_combinations_unique(self):
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations({
            "a": [1, 2, 3],
            "b": [10, 20],
        })
        combo_tuples = [tuple(sorted(c.items())) for c in combos]
        assert len(combo_tuples) == len(set(combo_tuples))

    def test_all_keys_present(self):
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations({
            "x": [1], "y": [2], "z": [3],
        })
        for c in combos:
            assert set(c.keys()) == {"x", "y", "z"}


class TestRunBcGridSearch:
    @patch.object(GridSearchOrchestrator, "_run_single_bc")
    def test_bc_grid_search_runs_all_combos(self, mock_run, mock_logger):
        mock_run.return_value = {"best_val_loss": 0.05}

        orch = GridSearchOrchestrator(mock_logger, device="cpu")
        ids = orch.run_bc_grid_search(
            data_path="/tmp/data",
            param_grid={"lr": [1e-4, 3e-4], "batch_size": [32]},
        )

        assert len(ids) == 2
        assert mock_run.call_count == 2

    @patch.object(GridSearchOrchestrator, "_run_single_bc")
    def test_bc_grid_search_handles_failure(self, mock_run, mock_logger):
        mock_run.side_effect = RuntimeError("training failed")

        orch = GridSearchOrchestrator(mock_logger, device="cpu")
        ids = orch.run_bc_grid_search(
            data_path="/tmp/data",
            param_grid={"lr": [1e-4]},
        )

        assert len(ids) == 0

    @patch.object(GridSearchOrchestrator, "_run_single_bc")
    def test_bc_grid_search_oom_retry(self, mock_run, mock_logger):
        # First call OOM, retry succeeds
        mock_run.side_effect = [
            RuntimeError("CUDA out of memory"),
            {"best_val_loss": 0.1},
        ]

        orch = GridSearchOrchestrator(mock_logger, device="cpu")
        ids = orch.run_bc_grid_search(
            data_path="/tmp/data",
            param_grid={"lr": [1e-4], "batch_size": [64]},
        )

        assert len(ids) == 1
        # Second call should have halved batch_size
        retry_config = mock_run.call_args_list[1][0][1]
        assert retry_config["batch_size"] == 32

    @patch.object(GridSearchOrchestrator, "_run_single_bc")
    def test_bc_grid_search_stops_at_50pct_failure(self, mock_run, mock_logger):
        """50% 이상 실패 시 중단."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("fail")
            return {"best_val_loss": 0.05, "best_checkpoint": ""}

        mock_run.side_effect = side_effect

        orch = GridSearchOrchestrator(mock_logger, device="cpu")
        ids = orch.run_bc_grid_search(
            data_path="/tmp/data",
            param_grid={"lr": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5]},
        )

        # Should stop before completing all 6
        assert mock_run.call_count < 6


class TestRunRlRewardGridSearch:
    @patch.object(GridSearchOrchestrator, "_run_single_rl")
    def test_rl_grid_search_runs_all(self, mock_run, mock_logger):
        mock_run.return_value = {"best_avg_reward": 50.0}

        orch = GridSearchOrchestrator(mock_logger, device="cpu")
        ids = orch.run_rl_reward_grid_search(
            bc_checkpoint="/tmp/bc.pth",
            reward_grid={"w_progress": [0.1, 0.3], "w_collision": [1.0]},
        )

        assert len(ids) == 2
        assert mock_run.call_count == 2

    @patch.object(GridSearchOrchestrator, "_run_single_rl")
    def test_rl_grid_search_handles_failure(self, mock_run, mock_logger):
        mock_run.side_effect = ConnectionError("CARLA not running")

        orch = GridSearchOrchestrator(mock_logger, device="cpu")
        ids = orch.run_rl_reward_grid_search(
            bc_checkpoint="/tmp/bc.pth",
            reward_grid={"w_progress": [0.1]},
        )

        assert len(ids) == 0
