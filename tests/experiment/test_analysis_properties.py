"""Property-based tests for ExperimentAnalyzer.

Properties 7, 10, 12, 13, 17, 18 from design doc.
"""

import pytest
from hypothesis import given, settings, assume, strategies as st

from experiment.analysis import ExperimentAnalyzer


@pytest.fixture
def analyzer():
    return ExperimentAnalyzer()


# Feature: experiment-validation, Property 7: 과적합 분석
class TestProperty7Overfitting:
    @settings(max_examples=100, deadline=None)
    @given(
        base_val=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
        drop=st.floats(min_value=0.01, max_value=0.3, allow_nan=False),
        rise=st.floats(min_value=0.01, max_value=0.3, allow_nan=False),
    )
    def test_overfitting_pattern_detected(self, base_val, drop, rise):
        """val_loss가 3 에포크 연속 상승하면 overfitting_detected=True."""
        # 먼저 감소 후 3 에포크 연속 상승
        val_loss = [
            base_val,
            base_val - drop,
            base_val - drop * 2,
            base_val - drop * 2 + rise,
            base_val - drop * 2 + rise * 2,
            base_val - drop * 2 + rise * 3,
            base_val - drop * 2 + rise * 4,
        ]
        train_loss = [base_val - i * drop for i in range(7)]

        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze_overfitting({"train_loss": train_loss, "val_loss": val_loss})
        assert result["overfitting_detected"] is True
        assert result["overfitting_start_epoch"] is not None
        assert len(result["recommendations"]) > 0


# Feature: experiment-validation, Property 10: 그리드 서치 결과 분석 및 목표 판정
class TestProperty10GridSearchAnalysis:
    @settings(max_examples=100, deadline=None)
    @given(
        mae_steer=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        mae_throttle=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    )
    def test_target_classification(self, mae_steer, mae_throttle):
        """달성/미달성 분류 정확성."""
        analyzer = ExperimentAnalyzer()
        metrics = {"mae_steering": mae_steer, "mae_throttle": mae_throttle}
        targets = {"mae_steering": 0.10, "mae_throttle": 0.08}

        result = analyzer.analyze_bc_gap(metrics, targets)

        steer_met = mae_steer < 0.10
        throttle_met = mae_throttle < 0.08

        assert result["gaps"]["mae_steering"]["met"] == steer_met
        assert result["gaps"]["mae_throttle"]["met"] == throttle_met
        assert result["all_targets_met"] == (steer_met and throttle_met)

        if not (steer_met and throttle_met):
            assert len(result["recommendations"]) > 0


# Feature: experiment-validation, Property 12: 실패 유형 집계 및 권고
class TestProperty12FailureAggregation:
    @settings(max_examples=100, deadline=None)
    @given(
        failures=st.lists(
            st.fixed_dictionaries({
                "failure_type": st.sampled_from(["collision", "lane_departure", "stopped"]),
            }),
            min_size=1, max_size=50,
        ),
    )
    def test_frequency_sum_equals_total(self, failures):
        """빈도 합 = 입력 목록 길이, 각 유형별 recommendation 존재."""
        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze_failure_cases(failures)

        assert result["total"] == len(failures)
        assert sum(result["by_type"].values()) == len(failures)
        # 각 실패 유형에 대해 최소 1개 recommendation
        assert len(result["recommendations"]) >= len(result["by_type"])


# Feature: experiment-validation, Property 13: 수렴 추세 감지
class TestProperty13Convergence:
    @settings(max_examples=100, deadline=None)
    @given(
        n=st.integers(min_value=502, max_value=600),
        step=st.floats(min_value=0.001, max_value=1.0, allow_nan=False),
    )
    def test_monotone_increasing_converging(self, n, step):
        """단조 증가 시계열 → converging=True."""
        rewards = [i * step for i in range(n)]
        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze_convergence(rewards, window=500)
        assert result["converging"] is True

    @settings(max_examples=100, deadline=None)
    @given(
        n=st.integers(min_value=502, max_value=600),
        step=st.floats(min_value=0.001, max_value=1.0, allow_nan=False),
    )
    def test_monotone_decreasing_not_converging(self, n, step):
        """단조 감소 시계열 → converging=False."""
        rewards = [1000.0 - i * step for i in range(n)]
        analyzer = ExperimentAnalyzer()
        result = analyzer.analyze_convergence(rewards, window=500)
        assert result["converging"] is False


# Feature: experiment-validation, Property 17: Phase 성공 기준 종합 판정
class TestProperty17PhaseCriteria:
    @settings(max_examples=100, deadline=None)
    @given(
        mae_steer=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        mae_throttle=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        pass_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_all_met_iff_passed(self, mae_steer, mae_throttle, pass_rate):
        """모든 메트릭 충족 시만 passed=True."""
        analyzer = ExperimentAnalyzer()
        metrics = {
            "mae_steering": mae_steer,
            "mae_throttle": mae_throttle,
            "intersection_pass_rate": pass_rate,
        }
        result = analyzer.check_phase_criteria(metrics, phase="2A")

        expected = (mae_steer < 0.10 and mae_throttle < 0.08 and pass_rate >= 0.80)
        assert result["passed"] == expected

        if not expected:
            assert len(result["unmet_criteria"]) > 0


# Feature: experiment-validation, Property 18: 격차 분석 및 보정 제안
class TestProperty18GapAnalysis:
    @settings(max_examples=100, deadline=None)
    @given(
        actual=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        target=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
    )
    def test_gap_implies_recommendations(self, actual, target):
        """격차 존재 시 recommendations 비어있지 않음."""
        analyzer = ExperimentAnalyzer()
        metrics = {"mae_steering": actual}
        targets = {"mae_steering": target}
        result = analyzer.analyze_bc_gap(metrics, targets)

        has_gap = actual >= target  # mae: 낮을수록 좋음
        if has_gap:
            assert len(result["recommendations"]) > 0
        else:
            # 모든 목표 달성 시 recommendations 비어있을 수 있음
            assert result["all_targets_met"] is True
