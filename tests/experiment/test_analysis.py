"""Unit tests for ExperimentAnalyzer."""

import pytest

from experiment.analysis import ExperimentAnalyzer, FailureCase


@pytest.fixture
def analyzer():
    return ExperimentAnalyzer()


class TestAnalyzeOverfitting:
    def test_no_overfitting(self, analyzer):
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
            "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4, 0.3],
        }
        result = analyzer.analyze_overfitting(history)
        assert result["overfitting_detected"] is False
        assert result["overfitting_start_epoch"] is None

    def test_overfitting_detected(self, analyzer):
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15],
            "val_loss": [1.0, 0.8, 0.7, 0.75, 0.80, 0.85, 0.90],
        }
        result = analyzer.analyze_overfitting(history)
        assert result["overfitting_detected"] is True
        assert result["overfitting_start_epoch"] is not None
        assert result["severity"] > 0
        assert len(result["recommendations"]) > 0

    def test_short_history(self, analyzer):
        history = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}
        result = analyzer.analyze_overfitting(history)
        assert result["overfitting_detected"] is False


class TestAnalyzeBcGap:
    def test_all_targets_met(self, analyzer):
        metrics = {"mae_steering": 0.05, "mae_throttle": 0.04}
        targets = {"mae_steering": 0.10, "mae_throttle": 0.08}
        result = analyzer.analyze_bc_gap(metrics, targets)
        assert result["all_targets_met"] is True
        assert len(result["recommendations"]) == 0

    def test_gap_exists(self, analyzer):
        metrics = {"mae_steering": 0.15, "mae_throttle": 0.04}
        targets = {"mae_steering": 0.10, "mae_throttle": 0.08}
        result = analyzer.analyze_bc_gap(metrics, targets)
        assert result["all_targets_met"] is False
        assert result["gaps"]["mae_steering"]["met"] is False
        assert result["gaps"]["mae_throttle"]["met"] is True
        assert len(result["recommendations"]) > 0

    def test_intersection_pass_rate_gap(self, analyzer):
        metrics = {"intersection_pass_rate": 0.40}
        targets = {"intersection_pass_rate": 0.80}
        result = analyzer.analyze_bc_gap(metrics, targets)
        assert result["all_targets_met"] is False
        assert len(result["recommendations"]) > 0

    def test_boundary_mae_steering_exactly_target(self, analyzer):
        """경계값: mae_steering == 0.10 → 미달성 (< 0.10 필요)."""
        metrics = {"mae_steering": 0.10}
        targets = {"mae_steering": 0.10}
        result = analyzer.analyze_bc_gap(metrics, targets)
        assert result["gaps"]["mae_steering"]["met"] is False

    def test_boundary_pass_rate_exactly_50pct(self, analyzer):
        """경계값: 교차로 통과율 정확히 50%."""
        metrics = {"intersection_pass_rate": 0.50}
        targets = {"intersection_pass_rate": 0.50}
        result = analyzer.analyze_bc_gap(metrics, targets)
        assert result["gaps"]["intersection_pass_rate"]["met"] is True


class TestAnalyzeFailureCases:
    def test_empty_failures(self, analyzer):
        result = analyzer.analyze_failure_cases([])
        assert result["total"] == 0

    def test_failure_type_counts(self, analyzer):
        failures = [
            {"failure_type": "collision"},
            {"failure_type": "collision"},
            {"failure_type": "lane_departure"},
            {"failure_type": "stopped"},
        ]
        result = analyzer.analyze_failure_cases(failures)
        assert result["total"] == 4
        assert result["by_type"]["collision"] == 2
        assert result["by_type"]["lane_departure"] == 1
        assert result["most_frequent"] == "collision"
        assert len(result["recommendations"]) == 3


class TestAnalyzeConvergence:
    def test_converging_series(self, analyzer):
        # 단조 증가 시계열
        rewards = list(range(600))  # 0, 1, 2, ..., 599
        result = analyzer.analyze_convergence(rewards, window=500)
        assert result["converging"] is True

    def test_non_converging_series(self, analyzer):
        # 단조 감소 시계열
        rewards = list(range(600, 0, -1))
        result = analyzer.analyze_convergence(rewards, window=500)
        assert result["converging"] is False

    def test_insufficient_data(self, analyzer):
        rewards = [1.0] * 100
        result = analyzer.analyze_convergence(rewards, window=500)
        assert result["converging"] is False
        assert "데이터 부족" in result["reason"]


class TestCheckPhaseCriteria:
    def test_phase_2a_all_met(self, analyzer):
        metrics = {
            "mae_steering": 0.05,
            "mae_throttle": 0.04,
            "intersection_pass_rate": 0.90,
        }
        result = analyzer.check_phase_criteria(metrics, phase="2A")
        assert result["passed"] is True
        assert len(result["unmet_criteria"]) == 0

    def test_phase_2a_partial_met(self, analyzer):
        metrics = {
            "mae_steering": 0.15,  # 미달
            "mae_throttle": 0.04,
            "intersection_pass_rate": 0.90,
        }
        result = analyzer.check_phase_criteria(metrics, phase="2A")
        assert result["passed"] is False
        assert "mae_steering" in result["unmet_criteria"]

    def test_phase_2b_met(self, analyzer):
        metrics = {"survival_time": 65.0}
        result = analyzer.check_phase_criteria(metrics, phase="2B")
        assert result["passed"] is True

    def test_phase_2b_not_met(self, analyzer):
        metrics = {"survival_time": 30.0}
        result = analyzer.check_phase_criteria(metrics, phase="2B")
        assert result["passed"] is False

    def test_boundary_avg_speed_exactly_1(self, analyzer):
        """경계값: 평균 속도 정확히 1.0 m/s."""
        metrics = {
            "mae_steering": 0.05,
            "mae_throttle": 0.04,
            "intersection_pass_rate": 0.90,
        }
        result = analyzer.check_phase_criteria(metrics, phase="2A")
        assert result["passed"] is True

    def test_missing_metric(self, analyzer):
        metrics = {"mae_steering": 0.05}  # mae_throttle, intersection_pass_rate 누락
        result = analyzer.check_phase_criteria(metrics, phase="2A")
        assert result["passed"] is False
        assert "mae_throttle" in result["unmet_criteria"]


class TestAnalyzeRlImprovement:
    def test_significant_improvement(self, analyzer):
        bc = {"avg_survival_time": 30.0, "collision_count": 5}
        rl = {"avg_survival_time": 65.0, "collision_count": 2}
        result = analyzer.analyze_rl_improvement(bc, rl)
        assert result["marginal_improvement"] is False
        assert result["improvements"]["avg_survival_time"]["delta"] == 35.0

    def test_marginal_improvement(self, analyzer):
        bc = {"avg_survival_time": 30.0}
        rl = {"avg_survival_time": 35.0}
        result = analyzer.analyze_rl_improvement(bc, rl)
        assert result["marginal_improvement"] is True
        assert len(result["recommendations"]) > 0


class TestGenerateCorrectionPlan:
    def test_plan_with_issues(self, analyzer):
        all_results = {
            "bc_gap": {
                "all_targets_met": False,
                "recommendations": ["steering 가중치 상향"],
            },
            "overfitting": {
                "overfitting_detected": True,
                "recommendations": ["dropout 증가"],
            },
        }
        plan = analyzer.generate_correction_plan(all_results)
        assert len(plan["actions"]) == 2
        assert "bc_improvement" in plan["priority"]
        assert "overfitting_mitigation" in plan["priority"]

    def test_plan_no_issues(self, analyzer):
        all_results = {
            "bc_gap": {"all_targets_met": True, "recommendations": []},
            "overfitting": {"overfitting_detected": False},
        }
        plan = analyzer.generate_correction_plan(all_results)
        assert len(plan["actions"]) == 0
