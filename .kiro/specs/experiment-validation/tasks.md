# Implementation Plan: Experiment Validation (실험 검증)

## Overview

기존 `src/model/`과 `src/data_pipeline/` 코드를 수정하지 않고, `src/experiment/` 모듈에서 이들을 조합하여 체계적 실험 검증 인프라를 구축한다. 사용자가 지정한 Dependency Order(Tier 1→4)에 따라 구현하며, 테스트는 `tests/experiment/`에 작성한다.

## Tasks

- [x] 1. Tier 1: 기반 인프라 — ExperimentLogger
  - [x] 1.1 `src/experiment/__init__.py` 생성 및 `src/experiment/experiment_logger.py` 구현
    - SQLite DB 스키마 생성 (experiments, experiment_configs, experiment_metrics, experiment_analysis, experiment_cli_commands, experiment_artifacts 테이블)
    - **SQLite 동시성 안전**: DB 연결 시 `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=5000` 적용 (야간 그리드 서치 중 CLI 조회 동시 접근 대비)
    - `ExperimentLogger` 클래스: `__init__`, `create_experiment`, `log_metrics`, `log_analysis`, `log_cli_command`, `get_experiment`, `list_experiments`, `compare_experiments`, `generate_report` 메서드
    - JSON 파일 이중 저장 로직 (SQLite 실패 시 JSON fallback)
    - numpy 타입 자동 변환 직렬화 처리
    - _Requirements: 3.4, 5.2, 6.5, 8.3, 9.2, 12.1, 12.2, 12.3, 12.4_

  - [x] 1.2 `tests/experiment/__init__.py` 생성 및 `tests/experiment/test_experiment_logger.py` 단위 테스트 작성
    - DB 생성, 실험 CRUD, 빈 DB 조회, 메트릭 기록/조회, 분석 기록, CLI 명령어 기록
    - 실험 비교(compare_experiments) 테스트
    - 보고서 생성(generate_report) 테스트
    - _Requirements: 3.4, 12.1, 12.2, 12.3_

  - [x]* 1.3 `tests/experiment/test_experiment_logger_properties.py` Property 테스트 작성
    - **Property 5: 실험 기록 완전성** — 임의의 실험 데이터 저장/조회 시 모든 키 보존
    - **Validates: Requirements 3.4, 5.2, 6.5, 8.3, 9.2, 12.1**
    - **Property 6: 실험 기록 라운드트립** — JSON/SQLite 저장 후 로드 시 동일성
    - **Validates: Requirements 12.1, 12.2**
    - **Property 8: 실험 비교** — 두 실험 메트릭 delta 및 improved 판정 정확성
    - **Validates: Requirements 4.3, 10.4, 13.3**
    - **Property 15: CLI 명령어 재현성** — config 값이 CLI 명령어 인자에 포함
    - **Validates: Requirements 12.2**
    - **Property 16: 종합 보고서 시간순 정렬** — 보고서 내 실험 항목 created_at 오름차순
    - **Validates: Requirements 12.3**

- [x] 2. Tier 1: 기반 인프라 — DataValidator
  - [x] 2.1 `src/experiment/data_validator.py` 구현
    - `DataValidationReport` 데이터클래스 정의
    - `DataValidator` 클래스: `__init__`, `validate_session`, `validate_images`, `validate_labels`, `analyze_distribution` 메서드
    - 이미지 무결성 검사 (파일 크기 > 0, PNG 디코딩 가능)
    - 레이블 범위 검증 (steering [-1,1], throttle [0,1])
    - 타이밍 검증 (연속 프레임 간격 80ms~120ms)
    - 손상 비율 > 5% 시 `needs_recollection=True` 및 경고
    - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 `tests/experiment/test_data_validator.py` 단위 테스트 작성
    - 빈 세션 디렉토리, CSV 없는 경우, 모든 이미지 손상된 경우
    - 정상 데이터 검증, 범위 밖 레코드 감지, 타이밍 이상 감지
    - 손상 비율 정확히 5% 경계값 테스트
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 2.3 `tests/experiment/test_data_validator_properties.py` Property 테스트 작성
    - **Property 1: 분포 분석 정확성** — histogram bin counts 합 = 입력 길이, mean = numpy.mean
    - **Validates: Requirements 1.4**
    - **Property 2: 이미지 무결성 검증** — 유효/손상 이미지 정확 분류
    - **Validates: Requirements 2.1**
    - **Property 3: 레이블 범위 및 타이밍 검증** — 범위 밖 카운트 정확 계산
    - **Validates: Requirements 2.2, 2.3**
    - **Property 4: 검증 보고서 완전성** — 필수 필드 존재 및 valid + corrupted ≤ total
    - **Validates: Requirements 2.4**

- [x] 3. Tier 1 Checkpoint
  - 41/41 tests passed (480s). Tier 1 완료.

- [x] 4. Tier 2: 평가 통제 — ScenarioManager
  - [x] 4.1 `src/experiment/scenario_manager.py` 구현
    - `EvalScenario` 데이터클래스 정의
    - 7개 표준 평가 시나리오 정의 (straight_clear_day, straight_rain_day, intersection_clear_day, intersection_rain_night, curve_clear_day, curve_fog_night, intersection_fog_backlight)
    - `ScenarioManager` 클래스: `__init__`, `get_scenario`, `list_scenarios`, `apply_scenario`, `run_evaluation`, `run_full_evaluation` 메서드
    - 고정 시드 기반 환경 설정 (날씨, 스폰 포인트, 시드)
    - 존재하지 않는 scenario_id 시 `KeyError`, 스폰 포인트 범위 초과 시 fallback
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 13.1_

  - [x] 4.2 `tests/experiment/test_scenario_manager.py` 단위 테스트 작성
    - 시나리오 조회, 목록 필터링, 존재하지 않는 ID 에러
    - 표준 시나리오 7개 정의 확인
    - **CARLA 모킹**: `apply_scenario`, `run_evaluation` 테스트 시 CARLA 서버 의존성을 `unittest.mock.patch`로 모킹하여 시나리오 설정 로직만 검증
    - _Requirements: 6.1, 6.3, 6.4_

  - [x]* 4.3 `tests/experiment/test_scenario_manager_properties.py` Property 테스트 작성
    - **Property 14: 멀티카메라 타임스탬프 동기화** — 동일 틱 5개 타임스탬프 동일성
    - **Validates: Requirements 11.2**

- [x] 5. Tier 2: 평가 통제 — ExperimentAnalyzer
  - [x] 5.1 `src/experiment/analysis.py` 구현
    - `FailureCase` 데이터클래스 정의
    - `ExperimentAnalyzer` 클래스: `__init__`, `analyze_overfitting`, `analyze_bc_gap`, `analyze_rl_improvement`, `analyze_failure_cases`, `generate_correction_plan`, `analyze_convergence`, `check_phase_criteria` 메서드
    - 과적합 진단: val_loss 3 에포크 연속 상승 감지
    - 격차 분석: 달성 메트릭 vs 목표 메트릭 비교 및 보정 제안
    - 실패 유형별 집계 및 권고
    - 수렴 추세 감지 (이동 평균 기반)
    - Phase 2-A/2-B 성공 기준 종합 판정
    - _Requirements: 4.1, 4.2, 4.3, 7.1, 7.2, 7.3, 8.5, 10.1, 10.2, 10.3, 10.4, 12.4_

  - [x] 5.2 `tests/experiment/test_analysis.py` 단위 테스트 작성
    - 과적합 없는 시계열, 모든 목표 달성 케이스
    - 실패 사례 분석, 수렴 감지, Phase 판정
    - 경계값: 평균 속도 정확히 1.0 m/s, 교차로 통과율 정확히 50%
    - _Requirements: 4.1, 4.2, 7.1, 7.2, 8.5, 10.1, 12.4_

  - [x]* 5.3 `tests/experiment/test_analysis_properties.py` Property 테스트 작성
    - **Property 7: 과적합 분석** — 과적합 패턴 시계열에서 감지 정확성
    - **Validates: Requirements 4.1, 4.2**
    - **Property 10: 그리드 서치 결과 분석 및 목표 판정** — 달성/미달성 분류 정확성
    - **Validates: Requirements 5.3, 9.3**
    - **Property 12: 실패 유형 집계 및 권고** — 빈도 합 = 목록 길이, 각 유형별 recommendation 존재
    - **Validates: Requirements 7.2, 7.3**
    - **Property 13: 수렴 추세 감지** — 단조 증가/비증가 시계열 판정 정확성
    - **Validates: Requirements 8.5**
    - **Property 17: Phase 성공 기준 종합 판정** — 모든 메트릭 충족 시만 passed=True
    - **Validates: Requirements 12.4**
    - **Property 18: 격차 분석 및 보정 제안** — 격차 존재 시 recommendations 비어있지 않음
    - **Validates: Requirements 10.1, 10.2, 10.3**

  - [x]* 5.4 `tests/experiment/test_failure_cases.py` 단위 + Property 테스트 작성
    - FailureCase 저장/조회 필드 보존 단위 테스트
    - **Property 11: 실패 사례 기록 완전성** — 모든 필드 보존
    - **Validates: Requirements 7.1**

- [x] 6. Tier 2 Checkpoint
  - 132/132 tests passed (567s). Tier 2 완료.

- [x] 7. Tier 3: 학습 자동화 — GridSearchOrchestrator
  - [x] 7.1 `src/experiment/grid_search.py` 구현
    - `GridSearchOrchestrator` 클래스: `__init__`, `run_bc_grid_search`, `run_rl_reward_grid_search`, `_generate_combinations` 메서드
    - 파라미터 그리드에서 모든 조합 생성 (itertools.product 활용)
    - BC 그리드 서치: BCTrainer + ModelEvaluator 조합 자동 실행
    - RL Reward 그리드 서치: RLTrainer + RewardFunction 가중치 조합 자동 실행
    - 개별 조합 실패 시 해당 조합만 `failed` 표시, 나머지 계속 실행
    - GPU OOM 시 batch_size 절반 재시도
    - 50% 이상 실패 시 그리드 서치 중단 및 부분 결과 반환
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.1, 9.2, 9.3, 9.4_

  - [x] 7.2 `tests/experiment/test_grid_search.py` 단위 테스트 작성
    - 빈 그리드 ValueError, 단일 파라미터 그리드
    - 조합 생성 정확성, 실패 처리 로직
    - **CARLA/GPU 모킹 필수**: `BCTrainer.train()`, `RLTrainer.train()`, `ModelEvaluator` 등을 `unittest.mock.patch`로 모킹하여 더미 loss/메트릭을 즉시 반환하도록 처리. 테스트 목표는 "파라미터 조합 순회 + 로깅 연결"이 올바른지 검증하는 것이며, 실제 모델 학습이 아님
    - _Requirements: 5.1, 9.1_

  - [x]* 7.3 `tests/experiment/test_grid_search_properties.py` Property 테스트 작성
    - **Property 9: 그리드 조합 생성** — 조합 수 = 각 값 리스트 길이의 곱, 모든 조합 고유, 모든 키 포함
    - **Validates: Requirements 5.1, 9.1**

- [x] 8. Tier 3 Checkpoint
  - 132/132 tests passed (567s). Tier 3 완료.

- [x] 9. Tier 4: 통합 및 확장 — CLI 엔트리포인트
  - [x] 9.1 `src/experiment/cli.py` 구현
    - argparse 기반 CLI: `validate`, `train-bc`, `train-rl`, `grid-search-bc`, `grid-search-rl`, `evaluate`, `report` 서브커맨드
    - 각 서브커맨드에서 ExperimentLogger, DataValidator, GridSearchOrchestrator, ScenarioManager, ExperimentAnalyzer 연결
    - 실험 재현용 CLI 명령어 자동 기록 (`log_cli_command`)
    - _Requirements: 12.1, 12.2_

  - [x] 9.2 `tests/experiment/test_cli.py` 단위 테스트 작성
    - 각 서브커맨드 파서 테스트, 인자 파싱 정확성
    - _Requirements: 12.2_

- [x] 10. Tier 4: 통합 및 확장 — MultiCameraPipeline
  - [x] 10.1 `src/experiment/multi_camera.py` 구현
    - **기존 `DataPipeline` 상속**: `src/data_pipeline/pipeline.py`의 `DataPipeline`을 상속하여 `setup_sensors()`만 오버라이드. 기존 `SynchronousModeController`, `AsyncDataLogger`, `EpisodeManager` 인프라를 그대로 재사용하여 코드 중복 최소화 (DRY 원칙)
    - `MultiCameraPipeline` 클래스: `setup_sensors()` 오버라이드 (5대 카메라 부착), `run()` 오버라이드 (카메라별 이미지 분배), `get_frame_drop_stats` 메서드
    - CAMERA_CONFIGS 정의 (front + avm_front/rear/left/right)
    - 5대 카메라 동기화 수집 (동일 타임스탬프)
    - 카메라별 서브디렉토리 저장 (front/, avm_front/, avm_rear/, avm_left/, avm_right/)
    - 프레임 드롭률 > 5% 시 경고 로그
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 10.2 `tests/experiment/test_multi_camera.py` 단위 테스트 작성
    - 카메라 설정 검증, 디렉토리 구조 생성 확인
    - 프레임 드롭 통계 계산 테스트
    - _Requirements: 11.1, 11.3, 11.5_

- [x] 11. Tier 4: 보고서 Property 테스트
  - [x]* 11.1 `tests/experiment/test_report_properties.py` Property 테스트 작성
    - **Property 15: CLI 명령어 재현성** — config 값이 CLI 명령어 인자에 포함 (ExperimentLogger 연동)
    - **Validates: Requirements 12.2**
    - **Property 16: 종합 보고서 시간순 정렬** — 보고서 내 실험 항목 created_at 오름차순
    - **Validates: Requirements 12.3**

- [x] 12. Final Checkpoint
  - 132/132 tests passed (567s). 전체 구현 완료.

## Notes

- `*` 표시된 태스크는 optional이며 빠른 MVP를 위해 건너뛸 수 있음
- 기존 코드(`src/model/`, `src/data_pipeline/`)는 수정하지 않고 `src/experiment/`에서 조합
- 테스트는 `tests/experiment/` 디렉토리에 작성
- Property-Based 테스트는 Hypothesis 사용 (`@settings(max_examples=100, deadline=None)`)
- 각 Tier 완료 후 checkpoint 태스크로 테스트 통과 확인
- 모든 태스크는 이전 태스크의 결과물 위에 점진적으로 구축됨

### 엔지니어링 원칙

1. **SQLite WAL 모드**: ExperimentLogger의 DB 연결 시 `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=5000` 필수 적용. 야간 그리드 서치 중 CLI 조회 등 동시 접근 시 `database is locked` 에러 방지
2. **CARLA/GPU 모킹**: Tier 2~3 단위 테스트에서 BCTrainer, RLTrainer, CARLAGymEnv 등은 `unittest.mock.patch`로 모킹. 테스트 목표는 오케스트레이션 로직 검증이며, 실제 학습/시뮬레이션 실행이 아님
3. **DRY — DataPipeline 상속**: MultiCameraPipeline은 기존 `DataPipeline`을 상속하여 `setup_sensors()`/`run()`만 오버라이드. SynchronousModeController, AsyncDataLogger, EpisodeManager 인프라를 재사용하여 코드 중복 최소화
