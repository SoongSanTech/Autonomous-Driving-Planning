# Implementation Plan: Experiment & ML Modeling

## Overview

BC(Behavioral Cloning) + RL(Reinforcement Learning) 자율주행 제어 모델 시스템 구현 계획. ResNet18 기반 경량 아키텍처로, Front RGB 이미지(224×224)를 입력받아 steering/throttle을 출력한다. BC 학습 후 동일 backbone을 warm-start하여 PPO RL로 미세조정한다.

구현 순서: 데이터 인프라 → BC 모델 → BC 학습 → BC 추론 → Gym 환경 → Reward → RL 정책 → RL 학습 → 평가 → 통합 테스트

## Tasks

- [ ] 1. 프로젝트 구조 및 인프라 설정
  - `src/model/` 디렉토리 생성 + `__init__.py`
  - `tests/model/` 디렉토리 생성 + `__init__.py`
  - `requirements.txt`에 의존성 추가: torch, torchvision, numpy, pandas, pillow, opencv-python, gymnasium, hypothesis, pytest
  - _Requirements: 전체_

- [ ] 2. 데이터 로딩 및 전처리 구현
  - [ ] 2.1 DrivingDataset 클래스 구현
    - `src/data/{session}/front/*.png` 이미지 로드
    - `src/data/{session}/labels/driving_log.csv` 파싱
    - 800×600 → 224×224 리사이즈 + ImageNet mean/std 정규화
    - 누락/손상 파일 스킵 + 경고 로그
    - _Requirements: 2.1, 2.2, 2.3, 9.1, 9.2, 9.3_

  - [ ]* 2.2 데이터 페어링 property test
    - **Property 3: Data pairing consistency**
    - **Validates: Requirements 2.1, 2.2**

  - [ ] 2.3 DataLoaderFactory 구현
    - 80/20 train/val 분할
    - 데이터 증강: horizontal flip (steering 부호 반전), brightness ±20%, Gaussian noise
    - batch size 32, 셔플
    - _Requirements: 2.8, 2.9, 9.4, 9.5_

  - [ ]* 2.4 전처리 property tests
    - **Property 13: Image normalization range**
    - **Property 14: Dataset split completeness**
    - **Validates: Requirements 9.2, 9.4**

- [ ] 3. 체크포인트 관리 시스템 구현
  - [ ] 3.1 CheckpointManager 클래스 구현
    - 저장: {model_type}_{timestamp}_epoch{N}.pth
    - 내용: model_type, epoch, model_state_dict, optimizer_state_dict, metrics, config (input_shape=(3,224,224))
    - 로드 + 무결성 검증
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 3.2 체크포인트 property tests
    - **Property 4: Checkpoint round-trip preservation**
    - **Property 15: Checkpoint filename format**
    - **Property 16: Checkpoint content completeness**
    - **Validates: Requirements 10.1, 10.2, 10.3**

- [ ] 4. Checkpoint — 데이터 파이프라인 검증
  - 모든 테스트 통과 확인

- [ ] 5. Behavioral Cloning 모델 구현
  - [ ] 5.1 BehavioralCloningModel 클래스 구현
    - ResNet18 backbone (ImageNet pretrained) → 512-d feature
    - FC Head: 512→256 (ReLU+Dropout0.5) → 128 (ReLU+Dropout0.3) → 2
    - steering: tanh [-1,1], throttle: sigmoid [0,1]
    - freeze_backbone() / unfreeze_backbone() 메서드
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 5.2 BC 모델 property tests
    - **Property 1: Model output shape consistency** (input 224×224 → 2 outputs)
    - **Property 2: Model output range constraints** (steering [-1,1], throttle [0,1])
    - **Validates: Requirements 1.2, 1.4, 1.5**

- [ ] 6. BC 학습 파이프라인 구현
  - [ ] 6.1 BCTrainer 클래스 구현
    - MSE loss: steering×2.0 + throttle×1.0
    - Adam optimizer, LR 1e-4, ReduceLROnPlateau
    - Phase 1: backbone frozen (10 epochs) → Phase 2: full fine-tune
    - Early stopping (patience=10, val_loss 기준)
    - Gradient clipping, NaN loss 감지
    - 주기적 체크포인트 저장
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.10_

  - [ ]* 6.2 BC 학습 unit tests
    - 합성 데이터로 학습 루프 테스트
    - 체크포인트 저장 테스트
    - NaN loss, OOM 에러 핸들링 테스트

- [ ] 7. BC 추론 파이프라인 구현
  - [ ] 7.1 BC 추론 모듈 구현
    - 체크포인트에서 모델 로드
    - CARLA 연동 실시간 제어 루프 (이미지 수신 → 224×224 리사이즈 → 예측 → 제어)
    - 추론 지연 측정 (< 100ms 검증)
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 7.2 추론 지연 property test
    - **Property 5: Inference latency constraint**
    - **Validates: Requirements 3.2**

- [ ] 8. Checkpoint — BC 파이프라인 검증
  - 모든 테스트 통과 확인

- [ ] 9. CARLA Gym 환경 구현
  - [ ] 9.1 CARLAGymEnv 클래스 구현
    - observation_space: Box(0, 255, (224, 224, 3))
    - action_space: Box([-1.0, 0.0], [1.0, 1.0])
    - CARLA 연결 + 재시도 로직 (3회, exponential backoff)
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 9.2 reset() 구현
    - 랜덤 spawn point + 랜덤 날씨
    - RGB 카메라 센서 부착 (800×600 → 224×224 리사이즈)
    - 충돌 센서 부착
    - _Requirements: 4.4, 4.7_

  - [ ] 9.3 step() 구현
    - action 적용 (steering, throttle)
    - 224×224 observation 반환
    - RewardFunction으로 보상 계산
    - done 조건: 충돌 / 차선 이탈 3m / 타임아웃 1000 steps
    - info dict 반환
    - _Requirements: 4.5, 4.6_

  - [ ]* 9.4 Gym 환경 property tests
    - **Property 6: Step return structure**
    - **Property 7: Observation shape (224×224×3)**
    - **Validates: Requirements 4.2, 4.5**

- [ ] 10. Reward Function 구현
  - [ ] 10.1 RewardFunction 클래스 구현
    - R_lane = -|d_lane|
    - R_collision = -100 if collision
    - R_steering = -|steer| if |steer| > 0.3
    - R_progress = v_forward × cos(θ_heading)
    - 가중치: w_lane=1.0, w_collision=1.0, w_steering=0.5, w_progress=0.1
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 10.2 Reward property tests
    - **Property 8: Lane centering incentive**
    - **Property 9: Collision penalty**
    - **Property 10: Steering smoothness**
    - **Property 11: Scalar output**
    - **Validates: Requirements 5.2, 5.3, 5.4**

- [ ] 11. Checkpoint — 환경 및 보상 검증
  - 모든 테스트 통과 확인

- [ ] 12. RL 정책 네트워크 구현
  - [ ] 12.1 RLPolicyNetwork 클래스 구현
    - from_bc_checkpoint() 클래스 메서드: BC backbone + Actor Head 복사, Critic Head 신규
    - Actor Head: BC FC Head 재사용 (512→256→128→2)
    - Critic Head: 512→256→1 (state value)
    - 공유 ResNet18 backbone
    - get_action() (deterministic/stochastic)
    - freeze_backbone() / unfreeze_backbone()
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 12.2 RL 정책 property tests
    - **Property 12: BC→RL warm-start weight preservation**
    - Output shape/range 테스트
    - **Validates: Requirements 6.2**

- [ ] 13. RL 학습 파이프라인 구현
  - [ ] 13.1 RLTrainer 클래스 구현 (PPO)
    - PPO: LR 3e-5, GAE λ=0.95, clip ratio=0.2
    - Phase 1: backbone frozen (100 에피소드)
    - Phase 2: full fine-tune (LR 1e-5)
    - 3,000~5,000 에피소드 학습
    - 체크포인트 저장 (rl_*.pth)
    - NaN loss, gradient explosion 핸들링
    - _Requirements: 6.1, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 13.2 RL 학습 integration tests
    - Mock 환경으로 학습 루프 테스트
    - 체크포인트 저장/로드 테스트
    - Trajectory 수집 테스트

- [ ] 14. Checkpoint — 모든 모델 및 학습 검증
  - 모든 테스트 통과 확인

- [ ] 15. 평가 메트릭 시스템 구현
  - [ ] 15.1 ModelEvaluator 클래스 구현
    - MAE steering / throttle 계산
    - 충돌 이벤트 카운트
    - 평균 차선 중앙 거리
    - 평균 생존 시간
    - 추론 지연 측정
    - 메트릭 로깅 (파일 + 콘솔)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 15.2 평가 메트릭 property tests
    - MAE 계산 정확도 테스트
    - 충돌 카운팅 정확도 테스트

- [ ] 16. 교차로 통과 테스트 구현
  - [ ] 16.1 IntersectionTester 클래스 구현
    - 체크포인트에서 모델 로드
    - CARLA 교차로 시나리오 배포 (autopilot 없이)
    - 자율 통과 감지 + 로깅
    - 목표: 80% 통과율 (10회 중 8회)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 16.2 교차로 테스트 integration test
    - 사전 학습 모델로 테스트
    - 성공 로깅 검증

- [ ] 17. 학습/평가 CLI 스크립트 생성
  - [ ] 17.1 train_bc.py 스크립트
    - CLI 인자: data_path, batch_size, epochs, lr
    - DataLoader 초기화 → BC 모델 학습 → 체크포인트 저장
    - _Requirements: 2.4, 2.10_

  - [ ] 17.2 train_rl.py 스크립트
    - CLI 인자: bc_checkpoint, carla_host/port, episodes, lr
    - BC warm-start → PPO 학습 → 체크포인트 저장
    - _Requirements: 6.1, 6.8_

  - [ ] 17.3 evaluate.py 스크립트
    - CLI 인자: checkpoint_path, test_data_path
    - 모델 로드 → 평가 → 메트릭 출력
    - _Requirements: 8.1~8.6_

  - [ ] 17.4 inference.py 스크립트
    - CLI 인자: checkpoint_path, carla_host/port
    - 모델 로드 → CARLA 실시간 제어 루프
    - _Requirements: 3.1, 3.2, 3.3_

- [ ] 18. Final Checkpoint — 통합 테스트
  - 모든 테스트 통과 확인

## Notes

- `*` 표시 태스크는 선택적 property-based / unit test (MVP 속도를 위해 스킵 가능)
- 모든 모델은 ResNet18 + 224×224 입력 — Jetson 30Hz 추론 전제
- BC/RL 학습에는 Front RGB만 사용. AVM 데이터는 Phase 3+ BEV 연구용
- 데이터 소스: `src/data/{session}/front/*.png` + `src/data/{session}/labels/driving_log.csv`
- 체크포인트: `.pth` 형식, {model_type}_{timestamp}_epoch{N}.pth
- CARLA 서버: Windows Host, WSL2 Client에서 Python 클라이언트 연결
- RL은 BC 가중치 warm-start 필수 (random init 아님)
- CycleGAN/Domain Adaptation은 제거됨 — 로드맵에서 불필요 판단
