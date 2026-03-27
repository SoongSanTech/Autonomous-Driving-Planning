"""
ExperimentAnalyzer: 실험 결과 분석 및 보정 제안.

과적합 진단, 격차 분석, 실패 사례 분석, 수렴 감지,
Phase 성공 기준 판정을 수행한다.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    """실패 사례 기록."""
    timestamp: float
    failure_type: str
    image_path: Optional[str]
    steering_pred: float
    throttle_pred: float
    lane_distance: float
    velocity: float
    scenario_id: str


class ExperimentAnalyzer:
    """실험 결과 분석 및 보정 제안."""

    # Phase 2-A 성공 기준
    PHASE_2A_TARGETS = {
        "mae_steering": 0.10,      # < 0.10
        "mae_throttle": 0.08,      # < 0.08
        "intersection_pass_rate": 0.80,  # > 80%
    }

    # Phase 2-B 성공 기준
    PHASE_2B_TARGETS = {
        "survival_time": 60.0,     # > 60초
    }

    def __init__(self, experiment_logger=None):
        self._logger = experiment_logger

    def analyze_overfitting(self, history: dict) -> dict:
        """과적합 진단.

        val_loss가 3 에포크 연속 상승하면 과적합으로 판정.

        Args:
            history: {"train_loss": [...], "val_loss": [...]}

        Returns:
            과적합 분석 결과 dict.
        """
        val_loss = history.get("val_loss", [])
        train_loss = history.get("train_loss", [])

        result = {
            "overfitting_detected": False,
            "overfitting_start_epoch": None,
            "severity": 0.0,
            "recommendations": [],
        }

        if len(val_loss) < 4:
            return result

        # 3 에포크 연속 상승 감지
        for i in range(len(val_loss) - 3):
            if (val_loss[i + 1] > val_loss[i] and
                val_loss[i + 2] > val_loss[i + 1] and
                val_loss[i + 3] > val_loss[i + 2]):
                result["overfitting_detected"] = True
                result["overfitting_start_epoch"] = i
                result["severity"] = val_loss[i + 3] - val_loss[i]
                result["recommendations"] = [
                    "데이터 증강 강화 (flip_prob, brightness 조정)",
                    "dropout 비율 증가",
                    "데이터 추가 수집 권고",
                    f"에포크 {i} 이후 과적합 시작 — early stopping 조정 권고",
                ]
                break

        return result

    def analyze_bc_gap(self, metrics: dict, targets: dict) -> dict:
        """BC 성능과 목표 간 격차 분석 + 보정 제안.

        Args:
            metrics: 달성된 메트릭 {"mae_steering": ..., "mae_throttle": ...}
            targets: 목표 메트릭 (같은 키 구조)

        Returns:
            격차 분석 결과 dict.
        """
        gaps = {}
        all_met = True
        recommendations = []

        for key, target_val in targets.items():
            actual = metrics.get(key, None)
            if actual is None:
                continue

            # mae 계열: 낮을수록 좋음 (actual < target이면 달성)
            # rate 계열: 높을수록 좋음 (actual > target이면 달성)
            if "mae" in key or "loss" in key:
                met = actual < target_val
                gap = actual - target_val if not met else 0.0
            else:
                met = actual >= target_val
                gap = target_val - actual if not met else 0.0

            gaps[key] = {
                "actual": actual,
                "target": target_val,
                "gap": gap,
                "met": met,
            }

            if not met:
                all_met = False

        if not all_met:
            for key, info in gaps.items():
                if not info["met"]:
                    if "steering" in key:
                        recommendations.append("steering loss 가중치 상향 권고")
                        recommendations.append("교차로 구간 데이터 비중 증가")
                    elif "throttle" in key:
                        recommendations.append("throttle 데이터 분포 확인 및 증강")
                    elif "pass_rate" in key or "intersection" in key:
                        recommendations.append("교차로 데이터 추가 수집")
                        recommendations.append("추가 학습 에포크 권고")
                    elif "survival" in key:
                        recommendations.append("RL reward 가중치 재조정")
                    else:
                        recommendations.append(f"{key} 개선을 위한 하이퍼파라미터 재탐색")

        return {
            "all_targets_met": all_met,
            "gaps": gaps,
            "recommendations": recommendations,
        }

    def analyze_rl_improvement(self, bc_metrics: dict, rl_metrics: dict) -> dict:
        """BC 대비 RL 성능 향상 분석.

        Args:
            bc_metrics: BC 모델 평가 결과.
            rl_metrics: RL 모델 평가 결과.

        Returns:
            향상 분석 결과 dict.
        """
        improvements = {}
        for key in set(bc_metrics.keys()) & set(rl_metrics.keys()):
            bc_val = bc_metrics[key]
            rl_val = rl_metrics[key]
            delta = rl_val - bc_val
            improvements[key] = {
                "bc": bc_val,
                "rl": rl_val,
                "delta": delta,
            }

        # 생존 시간 개선 < 10초이면 미미
        survival_delta = improvements.get("avg_survival_time", {}).get("delta", 0)
        marginal = abs(survival_delta) < 10.0

        recommendations = []
        if marginal:
            recommendations.append("RL 성능 향상 미미 — reward 가중치 재조정 권고")
            recommendations.append("BC warm-start 체크포인트 품질 확인")

        return {
            "improvements": improvements,
            "marginal_improvement": marginal,
            "recommendations": recommendations,
        }

    def analyze_failure_cases(self, failure_logs: list[dict]) -> dict:
        """실패 사례 유형별 분석.

        Args:
            failure_logs: FailureCase dict 목록.

        Returns:
            실패 유형별 빈도 및 권고.
        """
        if not failure_logs:
            return {"total": 0, "by_type": {}, "recommendations": []}

        type_counts = {}
        for case in failure_logs:
            ft = case.get("failure_type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1

        recommendations = []
        type_recommendations = {
            "collision": "충돌 방지를 위한 w_collision 가중치 상향",
            "lane_departure": "차선 유지를 위한 steering 학습 데이터 보강",
            "stopped": "정지 문제 해결을 위한 w_progress 가중치 상향",
        }

        for ft in type_counts:
            rec = type_recommendations.get(ft, f"{ft} 유형 실패에 대한 추가 분석 필요")
            recommendations.append(rec)

        return {
            "total": len(failure_logs),
            "by_type": type_counts,
            "most_frequent": max(type_counts, key=type_counts.get),
            "recommendations": recommendations,
        }

    def generate_correction_plan(self, all_results: dict) -> dict:
        """종합 보정 계획 생성.

        Args:
            all_results: 모든 실험 결과를 포함하는 dict.

        Returns:
            보정 계획 dict.
        """
        plan = {"actions": [], "priority": []}

        # BC 격차 분석
        if "bc_gap" in all_results:
            gap_info = all_results["bc_gap"]
            if not gap_info.get("all_targets_met", True):
                plan["actions"].extend(gap_info.get("recommendations", []))
                plan["priority"].append("bc_improvement")

        # 과적합 분석
        if "overfitting" in all_results:
            of_info = all_results["overfitting"]
            if of_info.get("overfitting_detected", False):
                plan["actions"].extend(of_info.get("recommendations", []))
                plan["priority"].append("overfitting_mitigation")

        # RL 향상 분석
        if "rl_improvement" in all_results:
            rl_info = all_results["rl_improvement"]
            if rl_info.get("marginal_improvement", False):
                plan["actions"].extend(rl_info.get("recommendations", []))
                plan["priority"].append("rl_tuning")

        return plan

    def analyze_convergence(self, rewards: list[float],
                            window: int = 500) -> dict:
        """수렴 추세 감지 (이동 평균 기반).

        Args:
            rewards: 에피소드별 reward 시계열.
            window: 이동 평균 윈도우 크기.

        Returns:
            수렴 분석 결과 dict.
        """
        if len(rewards) < window + 1:
            return {
                "converging": False,
                "reason": f"데이터 부족 (최소 {window + 1}개 필요, 현재 {len(rewards)}개)",
                "moving_averages": [],
            }

        # 이동 평균 계산
        ma = []
        for i in range(window, len(rewards)):
            avg = float(np.mean(rewards[i - window:i]))
            ma.append(avg)

        # 단조 증가 여부 판정
        if len(ma) < 2:
            return {"converging": False, "reason": "이동 평균 데이터 부족", "moving_averages": ma}

        increasing = all(ma[i + 1] >= ma[i] - 1e-9 for i in range(len(ma) - 1))
        non_increasing = all(ma[i + 1] <= ma[i] + 1e-9 for i in range(len(ma) - 1))

        if increasing:
            converging = True
            reason = "이동 평균 단조 증가 — 수렴 중"
        elif non_increasing:
            converging = False
            reason = "이동 평균 단조 비증가 — 수렴 미감지"
        else:
            # 혼합 패턴: 마지막 구간 추세로 판단
            recent = ma[-min(5, len(ma)):]
            converging = recent[-1] > recent[0]
            reason = "혼합 패턴 — 최근 추세 기반 판정"

        return {
            "converging": converging,
            "reason": reason,
            "moving_averages": ma,
        }

    def check_phase_criteria(self, metrics: dict, phase: str = "2A") -> dict:
        """Phase 성공 기준 종합 판정.

        Args:
            metrics: 달성된 메트릭.
            phase: "2A" 또는 "2B".

        Returns:
            판정 결과 dict.
        """
        if phase == "2A":
            targets = self.PHASE_2A_TARGETS
        elif phase == "2B":
            targets = self.PHASE_2B_TARGETS
        else:
            targets = {**self.PHASE_2A_TARGETS, **self.PHASE_2B_TARGETS}

        unmet = []
        details = {}

        for key, target_val in targets.items():
            actual = metrics.get(key, None)
            if actual is None:
                unmet.append(key)
                details[key] = {"actual": None, "target": target_val, "met": False}
                continue

            if "mae" in key or "loss" in key:
                met = actual < target_val
            else:
                met = actual >= target_val

            details[key] = {"actual": actual, "target": target_val, "met": met}
            if not met:
                unmet.append(key)

        passed = len(unmet) == 0

        return {
            "phase": phase,
            "passed": passed,
            "unmet_criteria": unmet,
            "details": details,
        }
