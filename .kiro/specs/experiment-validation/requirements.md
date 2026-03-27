# Requirements Document: Experiment Validation (실험 검증)

## Introduction

Productization 단계 진입 이전에 수행하는 체계적 실험 검증 spec이다. 실제 CARLA 시뮬레이터에서 데이터를 수집하고, BC/RL 모델을 학습하며, 실험 결과를 기반으로 데이터 수집 환경, 하이퍼파라미터, 방법론을 보정·개선하는 연구 과정을 엄격하게 정의한다.

이 spec은 이미 구현 완료된 data-pipeline(Phase 1, 89 tests)과 experiment-ml-modeling(Phase 2-A/B 코드, 97 tests)의 코드를 실제 CARLA 환경에서 실행하여 검증하고, 로드맵의 Phase 2-A/2-B 성공 기준을 달성하는 것을 목표로 한다.

## Glossary

- **Data_Collection_Pipeline**: `src/data_pipeline/` 모듈. CARLA autopilot 데이터를 Front RGB 이미지 + driving_log.csv로 수집하는 시스템
- **BC_Model**: `src/model/bc_model.py`의 BehavioralCloningModel. ResNet18 backbone + FC Head로 steering/throttle을 예측하는 행동복제 모델
- **BC_Trainer**: `src/model/bc_trainer.py`의 BCTrainer. 2-phase 학습(frozen→fine-tune), MSE loss, early stopping을 수행하는 학습 파이프라인
- **RL_Trainer**: `src/model/rl_trainer.py`의 RLTrainer. PPO 알고리즘으로 BC warm-start 후 강화학습을 수행하는 학습 파이프라인
- **CARLA_Gym_Env**: `src/model/carla_gym_env.py`의 CARLAGymEnv. CARLA를 Gymnasium 인터페이스로 래핑한 RL 환경
- **Model_Evaluator**: `src/model/evaluator.py`의 ModelEvaluator. MAE, 충돌, 차선 거리, 생존 시간 등을 측정하는 평가 시스템
- **BC_Inference_Engine**: `src/model/inference.py`의 BCInferenceEngine. 학습된 BC 모델로 CARLA 내 실시간 자율주행을 수행하는 추론 엔진
- **Experiment_Log**: 각 실험의 설정, 결과, 분석을 기록하는 구조화된 문서 (Markdown 또는 JSON)
- **Session_Directory**: `src/data/{YYYY-MM-DD_HHMMSS}/` 형식의 데이터 수집 세션 디렉토리
- **Checkpoint**: `checkpoints/` 디렉토리에 저장되는 `.pth` 형식의 모델 가중치 파일
- **MAE**: Mean Absolute Error. 예측값과 실제값 간 절대 오차의 평균
- **Steering_MAE_Target**: 로드맵 Phase 2-A 성공 기준인 MAE Steering < 0.10
- **Throttle_MAE_Target**: 로드맵 Phase 2-A 성공 기준인 MAE Throttle < 0.08
- **Intersection_Pass_Rate_Target**: 로드맵 Phase 2-A 성공 기준인 교차로 통과율 > 80% (10회 중 8회)
- **Survival_Time_Target**: 로드맵 Phase 2-B 성공 기준인 충돌 없이 주행 시간 > 60초
- **Multi_Camera_Pipeline**: Front RGB + AVM 4대(전/후/좌/우) 총 5대 카메라 동시 수집 파이프라인


## Requirements

### Requirement 1: 데이터 수집 실험 — 초기 데이터셋 확보

**User Story:** 연구자로서, CARLA autopilot으로 최소 1시간 분량의 Front RGB 주행 데이터를 수집하여, BC 모델 학습에 충분한 초기 데이터셋을 확보하고 싶다.

#### Acceptance Criteria

1. WHEN Data_Collection_Pipeline이 ClearNoon/DAYTIME 조건에서 1시간 동안 실행되면, THE Data_Collection_Pipeline SHALL 최소 36,000 프레임의 Front RGB 이미지(800×600 PNG)와 대응하는 driving_log.csv 레코드를 Session_Directory에 저장한다
2. WHEN 데이터 수집이 완료되면, THE Data_Collection_Pipeline SHALL 수집된 프레임 수, 드롭된 프레임 수, 총 수집 시간을 로그로 출력한다
3. THE Data_Collection_Pipeline SHALL 프레임 드롭률을 전체 프레임의 1% 미만으로 유지한다
4. WHEN 수집된 데이터셋이 존재하면, THE Model_Evaluator SHALL driving_log.csv의 steering 값 분포(히스토그램)와 throttle 값 분포를 분석하여 데이터 편향 여부를 Experiment_Log에 기록한다
5. IF 수집 중 CARLA 서버 크래시가 발생하면, THEN THE Data_Collection_Pipeline SHALL 크래시 시점까지 수집된 데이터를 보존하고 부분 저장 프레임 수를 로그로 출력한다

---

### Requirement 2: 데이터 품질 검증

**User Story:** 연구자로서, 수집된 데이터의 품질을 정량적으로 검증하여, 학습에 부적합한 데이터를 사전에 식별하고 제거하고 싶다.

#### Acceptance Criteria

1. WHEN 수집된 Session_Directory가 주어지면, THE Data_Collection_Pipeline SHALL 각 이미지 파일의 무결성(파일 크기 > 0, PNG 디코딩 가능 여부)을 검사하고 손상된 파일 수를 보고한다
2. WHEN driving_log.csv가 주어지면, THE Data_Collection_Pipeline SHALL steering 값이 [-1.0, 1.0] 범위 내에 있고 throttle 값이 [0.0, 1.0] 범위 내에 있는지 검증하며, 범위 밖 레코드 수를 보고한다
3. WHEN driving_log.csv가 주어지면, THE Data_Collection_Pipeline SHALL 연속 프레임 간 타임스탬프 간격이 80ms~120ms(10Hz ± 20%) 범위 내인지 검증하며, 범위 밖 간격 수를 보고한다
4. WHEN 데이터 품질 검증이 완료되면, THE Data_Collection_Pipeline SHALL 총 프레임 수, 유효 프레임 수, 손상 프레임 수, 범위 밖 레코드 수, 타이밍 이상 수를 포함하는 품질 보고서를 Experiment_Log에 기록한다
5. IF 손상된 이미지 비율이 전체의 5%를 초과하면, THEN THE Data_Collection_Pipeline SHALL 경고 메시지를 출력하고 재수집을 권고한다

---

### Requirement 3: BC 모델 학습 실험 — 기본 학습

**User Story:** 연구자로서, 수집된 데이터로 BC 모델을 학습하고 학습 곡선을 분석하여, 모델이 정상적으로 수렴하는지 확인하고 싶다.

#### Acceptance Criteria

1. WHEN 유효한 Session_Directory와 기본 하이퍼파라미터(batch_size=32, lr=1e-4, epochs=50, frozen_epochs=10, patience=10)가 주어지면, THE BC_Trainer SHALL 학습을 완료하고 best val_loss 기준 Checkpoint를 저장한다
2. WHILE 학습이 진행되는 동안, THE BC_Trainer SHALL 매 에포크마다 train_loss, val_loss, mae_steering, mae_throttle, learning_rate를 로그로 출력한다
3. WHEN 학습이 완료되면, THE BC_Trainer SHALL train_loss와 val_loss의 에포크별 변화를 포함하는 학습 이력(history)을 반환한다
4. WHEN 학습 이력이 주어지면, THE Experiment_Log SHALL 다음 항목을 기록한다: 최종 train_loss, 최종 val_loss, best val_loss, best epoch, 총 학습 에포크 수, Phase 1→2 전환 시점의 loss 변화
5. IF val_loss가 10 에포크 연속 개선되지 않으면, THEN THE BC_Trainer SHALL early stopping을 실행하고 해당 에포크 번호를 로그로 출력한다

---

### Requirement 4: BC 모델 학습 실험 — 과적합 진단

**User Story:** 연구자로서, 학습된 BC 모델의 과적합 여부를 진단하여, 데이터 증강이나 정규화 전략을 조정할 근거를 확보하고 싶다.

#### Acceptance Criteria

1. WHEN 학습 이력(train_loss, val_loss 시계열)이 주어지면, THE Experiment_Log SHALL train_loss와 val_loss 간 격차(gap)가 에포크 진행에 따라 증가하는지 분석하고 과적합 시작 에포크를 식별한다
2. WHEN 과적합이 감지되면(val_loss가 3 에포크 연속 상승하면서 train_loss는 하락), THE Experiment_Log SHALL 과적합 심각도(val_loss 상승폭)와 권장 조치(데이터 증강 강화, dropout 조정, 데이터 추가 수집)를 기록한다
3. WHEN 데이터 증강 설정(flip_prob, brightness_range, noise_std)이 변경된 후 재학습되면, THE Experiment_Log SHALL 이전 학습 결과와 비교하여 val_loss 개선 여부를 기록한다

---

### Requirement 5: BC 모델 학습 실험 — 하이퍼파라미터 탐색

**User Story:** 연구자로서, 주요 하이퍼파라미터를 체계적으로 탐색하여, 로드맵 성공 기준(MAE Steering < 0.10, MAE Throttle < 0.08)을 달성하는 최적 설정을 찾고 싶다.

#### Acceptance Criteria

1. THE BC_Trainer SHALL 다음 하이퍼파라미터 조합을 순차적으로 실험한다: learning_rate ∈ {5e-5, 1e-4, 3e-4}, batch_size ∈ {16, 32, 64}, steering_weight ∈ {1.5, 2.0, 3.0}
2. WHEN 각 하이퍼파라미터 조합의 학습이 완료되면, THE Experiment_Log SHALL 해당 조합의 best val_loss, mae_steering, mae_throttle, 학습 에포크 수를 기록한다
3. WHEN 모든 하이퍼파라미터 탐색이 완료되면, THE Experiment_Log SHALL 각 하이퍼파라미터가 성능에 미치는 영향을 비교 분석하고, Steering_MAE_Target과 Throttle_MAE_Target 달성 여부를 판정한다
4. IF 기본 하이퍼파라미터 조합으로 Steering_MAE_Target 또는 Throttle_MAE_Target을 달성하지 못하면, THEN THE Experiment_Log SHALL 추가 탐색 범위(lr, weight, augmentation 변경)를 제안한다

---

### Requirement 6: BC 추론 검증 — CARLA 시뮬레이터 내 자율주행 테스트

**User Story:** 연구자로서, 학습된 BC 모델을 CARLA 시뮬레이터에서 autopilot 없이 실행하여, 실제 자율주행 성능을 검증하고 싶다.

#### Acceptance Criteria

1. WHEN best Checkpoint가 BC_Inference_Engine에 로드되면, THE BC_Inference_Engine SHALL CARLA 시뮬레이터 내에서 autopilot 없이 차량을 제어한다
2. WHEN BC_Inference_Engine이 실행되면, THE BC_Inference_Engine SHALL 매 프레임의 추론 지연(latency)을 측정하고, 평균 추론 지연이 100ms 미만인지 검증한다
3. WHEN 직선 도로 주행 테스트가 수행되면, THE Model_Evaluator SHALL 충돌 없이 주행한 시간을 측정하고, 최소 30초 이상 자율주행 가능 여부를 판정한다
4. WHEN 교차로 통과 테스트가 10회 수행되면, THE Model_Evaluator SHALL 성공 횟수를 기록하고, Intersection_Pass_Rate_Target(80%) 달성 여부를 판정한다
5. WHEN 추론 테스트가 완료되면, THE Experiment_Log SHALL 평균 추론 지연, 직선 주행 생존 시간, 교차로 통과율, 차선 이탈 횟수를 기록한다
6. IF 교차로 통과율이 50% 미만이면, THEN THE Experiment_Log SHALL 실패 원인 분석(교차로 데이터 부족, steering 예측 편향, throttle 과다/과소)과 개선 방안을 기록한다

---

### Requirement 7: BC 추론 검증 — 실패 사례 분석

**User Story:** 연구자로서, BC 모델이 실패하는 구체적 상황을 분석하여, 데이터 수집 전략이나 모델 구조 개선의 근거를 확보하고 싶다.

#### Acceptance Criteria

1. WHEN BC_Inference_Engine이 충돌 또는 차선 이탈로 종료되면, THE Experiment_Log SHALL 종료 시점의 이미지, steering 예측값, throttle 예측값, 차선 거리, 속도를 기록한다
2. WHEN 10회 이상의 추론 테스트가 완료되면, THE Experiment_Log SHALL 실패 유형별(충돌, 차선 이탈, 정지) 빈도를 집계하고 가장 빈번한 실패 유형을 식별한다
3. WHEN 실패 사례 분석이 완료되면, THE Experiment_Log SHALL 각 실패 유형에 대한 개선 방안(데이터 추가 수집 조건, 증강 전략 변경, loss 가중치 조정)을 제안한다

---

### Requirement 8: RL 학습 실험 — BC→RL Warm-Start 및 수렴 분석

**User Story:** 연구자로서, BC 모델을 warm-start로 사용하여 PPO RL 학습을 수행하고, 수렴 여부와 성능 향상을 분석하고 싶다.

#### Acceptance Criteria

1. WHEN best BC Checkpoint가 주어지면, THE RL_Trainer SHALL BC backbone과 Actor Head 가중치를 warm-start로 로드하고 PPO 학습을 시작한다
2. WHILE RL 학습이 진행되는 동안, THE RL_Trainer SHALL 매 10 에피소드마다 평균 reward, 평균 에피소드 길이, policy_loss, value_loss를 로그로 출력한다
3. WHEN RL 학습이 완료되면(최소 1,000 에피소드), THE Experiment_Log SHALL 에피소드별 reward 곡선, 수렴 에피소드 번호(reward가 안정화되는 시점), 최종 평균 reward를 기록한다
4. WHEN RL 학습 결과가 주어지면, THE Model_Evaluator SHALL 충돌 없이 주행 시간을 측정하고, Survival_Time_Target(60초) 달성 여부를 판정한다
5. IF RL 학습이 500 에피소드 이후에도 평균 reward가 상승 추세를 보이지 않으면, THEN THE Experiment_Log SHALL reward 가중치 조정(w_progress 상향, w_collision 조정) 또는 학습률 변경을 제안한다

---

### Requirement 9: RL 학습 실험 — Reward 함수 튜닝

**User Story:** 연구자로서, Reward 함수의 가중치를 체계적으로 조정하여, RL 에이전트가 "가만히 서 있기" 지역 최적해에 빠지지 않고 적극적으로 주행하도록 하고 싶다.

#### Acceptance Criteria

1. THE RL_Trainer SHALL 다음 reward 가중치 조합을 실험한다: w_progress ∈ {0.1, 0.3, 0.5}, w_collision ∈ {0.5, 1.0, 2.0}, w_steering ∈ {0.3, 0.5, 1.0}
2. WHEN 각 reward 가중치 조합의 학습이 완료되면, THE Experiment_Log SHALL 해당 조합의 평균 reward, 평균 에피소드 길이, 충돌률, 평균 속도를 기록한다
3. WHEN 모든 reward 튜닝 실험이 완료되면, THE Experiment_Log SHALL 각 가중치가 에이전트 행동에 미치는 영향을 분석하고, "가만히 서 있기" 문제 발생 여부와 해결 조합을 식별한다
4. IF 에이전트의 평균 속도가 1.0 m/s 미만인 조합이 발견되면, THEN THE Experiment_Log SHALL 해당 조합을 "정지 문제 발생"으로 표시하고 w_progress 상향을 권고한다

---

### Requirement 10: 방법론 보정 — 실험 결과 기반 개선 사이클

**User Story:** 연구자로서, 각 실험 단계의 결과를 기반으로 데이터 수집 전략, 학습 설정, 모델 구조를 체계적으로 보정하여, 로드맵 성공 기준을 달성하고 싶다.

#### Acceptance Criteria

1. WHEN BC 학습 실험(Requirement 3~5)이 완료되면, THE Experiment_Log SHALL 달성된 MAE 값과 목표 MAE 값의 격차를 분석하고, 격차가 존재할 경우 다음 중 하나 이상의 보정 조치를 제안한다: (a) 데이터 추가 수집, (b) 하이퍼파라미터 재탐색, (c) 데이터 증강 전략 변경
2. WHEN BC 추론 검증(Requirement 6~7)이 완료되면, THE Experiment_Log SHALL 교차로 통과율과 목표의 격차를 분석하고, 격차가 존재할 경우 다음 중 하나 이상의 보정 조치를 제안한다: (a) 교차로 구간 데이터 비중 증가, (b) steering loss 가중치 상향, (c) 추가 학습 에포크
3. WHEN RL 학습 실험(Requirement 8~9)이 완료되면, THE Experiment_Log SHALL BC 대비 RL의 성능 향상 폭을 정량적으로 기록하고, 향상이 미미할 경우(충돌 없이 주행 시간 개선 < 10초) 원인 분석과 추가 실험 방향을 제안한다
4. WHEN 보정 조치가 실행된 후 재실험이 완료되면, THE Experiment_Log SHALL 보정 전후의 성능 비교표를 작성하고, 보정의 효과를 정량적으로 평가한다

---

### Requirement 11: 멀티카메라 파이프라인 확장

**User Story:** 연구자로서, 기존 단일 Front RGB 카메라 파이프라인을 5대 카메라(Front RGB + AVM 4대) 동시 수집으로 확장하여, Phase 3 연구 트랙에 필요한 데이터 인프라를 구축하고 싶다.

#### Acceptance Criteria

1. THE Multi_Camera_Pipeline SHALL Front RGB(800×600, FOV 90°, x=1.5, z=2.4) + AVM Front(400×300, FOV 120°, x=2.0, z=0.5) + AVM Rear(400×300, FOV 120°, x=-2.0, z=0.5) + AVM Left(400×300, FOV 120°, y=-1.0, z=0.5) + AVM Right(400×300, FOV 120°, y=1.0, z=0.5) 총 5대 카메라를 동시에 CARLA 차량에 부착한다
2. WHILE 멀티카메라 수집이 진행되는 동안, THE Multi_Camera_Pipeline SHALL 5대 카메라의 이미지를 동일 타임스탬프로 동기화하여 저장한다
3. WHEN 멀티카메라 수집이 완료되면, THE Multi_Camera_Pipeline SHALL Session_Directory 내에 front/, avm_front/, avm_rear/, avm_left/, avm_right/ 서브디렉토리를 생성하고 각 카메라의 이미지를 해당 디렉토리에 저장한다
4. WHEN 멀티카메라 수집이 1시간 동안 실행되면, THE Multi_Camera_Pipeline SHALL 5대 카메라 각각에서 최소 36,000 프레임을 수집한다
5. IF 특정 카메라의 프레임 드롭률이 5%를 초과하면, THEN THE Multi_Camera_Pipeline SHALL 해당 카메라 ID와 드롭률을 경고 로그로 출력한다

---

### Requirement 12: 실험 결과 문서화 및 재현성

**User Story:** 연구자로서, 모든 실험의 설정과 결과를 체계적으로 문서화하여, 실험을 재현할 수 있고 productization 단계 진입 판단의 근거로 활용하고 싶다.

#### Acceptance Criteria

1. THE Experiment_Log SHALL 각 실험에 대해 다음 항목을 기록한다: 실험 ID, 실험 일시, 실험 목적, 사용된 데이터셋(Session_Directory 경로), 하이퍼파라미터 전체 목록, 사용된 Checkpoint 경로, 결과 메트릭, 분석 및 결론
2. WHEN 실험이 완료되면, THE Experiment_Log SHALL 해당 실험을 재현하기 위한 CLI 명령어(정확한 인자 포함)를 기록한다
3. THE Experiment_Log SHALL 모든 실험 결과를 시간순으로 정리한 종합 보고서를 포함하며, 각 실험 간의 인과 관계(이전 실험 결과가 다음 실험 설정에 미친 영향)를 명시한다
4. WHEN 모든 실험이 완료되면, THE Experiment_Log SHALL Phase 2-A/2-B 성공 기준 달성 여부를 종합 판정하고, productization 단계 진입 가능 여부를 명시한다

---

### Requirement 13: 데이터 수집 조건 최적화 실험

**User Story:** 연구자로서, 다양한 수집 조건(날씨, 시간대, 맵)에서 데이터를 수집하여, BC 모델의 일반화 성능에 가장 효과적인 수집 전략을 식별하고 싶다.

#### Acceptance Criteria

1. THE Data_Collection_Pipeline SHALL 최소 3가지 날씨 조건(ClearNoon, WetCloudyNoon, SoftRainSunset)에서 각각 20분 이상의 데이터를 수집한다
2. WHEN 각 조건별 데이터로 BC 모델을 학습한 후, THE Model_Evaluator SHALL 조건별 mae_steering과 mae_throttle을 측정하고 비교한다
3. WHEN 단일 조건 데이터(1시간)와 혼합 조건 데이터(3가지 조건 각 20분)로 학습한 모델을 비교하면, THE Experiment_Log SHALL 두 모델의 성능 차이를 기록하고 혼합 데이터의 효과를 분석한다
4. WHEN 수집 조건 최적화 실험이 완료되면, THE Experiment_Log SHALL 본격 학습(6시간)에 사용할 최적 수집 전략(단일 조건 vs 혼합 조건, 조건별 비율)을 결정하고 근거를 기록한다
