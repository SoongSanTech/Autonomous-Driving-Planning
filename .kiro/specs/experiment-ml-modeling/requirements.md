# Requirements Document

## Introduction

Experiment & ML Modeling 기능은 Data Pipeline에서 수집한 Front RGB 카메라 데이터를 활용하여 PyTorch 기반 자율주행 제어 모델을 학습한다. 시스템은 두 가지 접근법을 구현한다: (1) End-to-End Behavioral Cloning(BC)으로 이미지→제어 직접 매핑, (2) Deep Reinforcement Learning(RL)으로 BC 가중치를 warm-start하여 PPO 기반 정책 최적화. 학습은 WSL2 Linux 환경에서 PyTorch + NVIDIA RTX 3090 Ti / 4090 워크스테이션으로 수행하며, 모든 모델은 Jetson 30Hz 추론을 전제로 ResNet18 경량 아키텍처를 채택한다.

AVM 4대 카메라 데이터는 Phase 1(Data Pipeline)에서 동시 수집되지만, BC/RL 학습에는 Front RGB만 사용한다. AVM 데이터는 Phase 3 이후 BEV Free-Space 연구에 활용한다.

## Glossary

- **ML_System**: 학습, 평가, 배포를 담당하는 전체 머신러닝 시스템
- **Behavioral_Cloning_Model**: ResNet18 backbone + FC Head로 Front RGB 이미지를 steering/throttle로 매핑하는 신경망
- **RL_Agent**: BC 가중치로 초기화된 PPO 기반 강화학습 에이전트
- **Training_Dataset**: Data Pipeline 출력 중 Front RGB 이미지(`src/data/{session}/front/`)와 라벨(`src/data/{session}/labels/driving_log.csv`)
- **Model_Checkpoint**: `.pth` 형식의 PyTorch 모델 파일 (가중치 + 옵티마이저 상태 + 메타데이터)
- **Steering_Value**: [-1.0, 1.0] 범위의 좌우 조향 제어값
- **Throttle_Value**: [0.0, 1.0] 범위의 가속 제어값
- **Gym_Environment**: CARLA를 래핑한 OpenAI Gymnasium 호환 인터페이스
- **Reward_Function**: 차선 유지, 충돌 회피, 조향 부드러움, 전진 보상을 가중합한 스칼라 보상 함수
- **Intersection_Scenario**: 교통 신호와 차선이 있는 CARLA 교차로 테스트 시나리오
- **Autonomous_Pass**: 충돌/차선 이탈 없이 교차로를 통과하는 성공 이벤트
- **Front_RGB_Image**: 전방 카메라에서 촬영한 800×600 RGB 이미지 (224×224로 리사이즈 후 모델 입력)
- **ResNet18_Backbone**: ImageNet pretrained ResNet18의 convolutional layers (512-d feature vector 출력)
- **Actor_Head**: BC의 FC Head를 재사용한 RL 정책 출력 (steering + throttle)
- **Critic_Head**: RL에서 신규 추가되는 상태 가치 추정 헤드 (512 → 256 → 1)
- **PPO**: Proximal Policy Optimization, 안정적인 정책 경사 알고리즘
- **Output_Session_Dir**: `src/data/{YYYY-MM-DD_HHMMSS}/` 형식의 데이터 수집 세션 디렉토리

## Requirements

### Requirement 1: Behavioral Cloning 모델 아키텍처

**User Story:** ML 엔지니어로서, Front RGB 이미지를 처리하여 제어값을 출력하는 신경망 아키텍처가 필요하다.

#### Acceptance Criteria

1. THE Behavioral_Cloning_Model SHALL accept Front_RGB_Image resized to shape (224, 224, 3)
2. THE Behavioral_Cloning_Model SHALL use ResNet18_Backbone (ImageNet pretrained) for feature extraction, outputting a 512-dimensional feature vector
3. THE Behavioral_Cloning_Model SHALL use FC Head: 512 → 256 (ReLU + Dropout 0.5) → 128 (ReLU + Dropout 0.3) → 2
4. THE Behavioral_Cloning_Model SHALL output Steering_Value via tanh activation in range [-1.0, 1.0]
5. THE Behavioral_Cloning_Model SHALL output Throttle_Value via sigmoid activation in range [0.0, 1.0]
6. THE Behavioral_Cloning_Model SHALL support two-phase training: Phase 1 backbone frozen (10 epochs) → Phase 2 full fine-tune

### Requirement 2: Behavioral Cloning 학습

**User Story:** ML 엔지니어로서, 수집된 Front RGB 데이터로 BC 모델을 학습하여 주행 행동을 모방하고 싶다.

#### Acceptance Criteria

1. THE ML_System SHALL load Front_RGB_Images from `src/data/{session}/front/` directory
2. THE ML_System SHALL load Vehicle_State records from `src/data/{session}/labels/driving_log.csv`
3. THE ML_System SHALL resize images from 800×600 to 224×224 and normalize with ImageNet mean/std
4. THE ML_System SHALL use Adam optimizer with initial learning rate 1e-4 and ReduceLROnPlateau scheduler
5. THE ML_System SHALL use MSE loss with steering weight 2.0 and throttle weight 1.0
6. THE ML_System SHALL use batch size 32
7. THE ML_System SHALL train for up to 50 epochs with early stopping (patience=10, val_loss 기준)
8. THE ML_System SHALL apply data augmentation: horizontal flip (steering sign inversion), brightness ±20%, Gaussian noise
9. THE ML_System SHALL split dataset 80% train / 20% validation (에피소드 단위 분할 권장)
10. THE ML_System SHALL save Model_Checkpoint in `.pth` format with model weights, optimizer state, epoch, metrics, and config

### Requirement 3: Behavioral Cloning 추론

**User Story:** ML 엔지니어로서, 학습된 모델을 로드하여 CARLA에서 실시간 제어를 수행하고 싶다.

#### Acceptance Criteria

1. THE ML_System SHALL load a Behavioral_Cloning_Model from a Model_Checkpoint file
2. WHEN provided with a Front_RGB_Image, THE Behavioral_Cloning_Model SHALL predict Steering_Value and Throttle_Value within 100ms
3. THE ML_System SHALL integrate with CARLA Python Client for real-time control loop (이미지 수신 → 예측 → 제어 적용)
4. THE ML_System SHALL achieve MAE Steering < 0.10 and MAE Throttle < 0.08 on validation data

### Requirement 4: CARLA Gym 환경 인터페이스

**User Story:** RL 연구자로서, CARLA를 Gymnasium 환경으로 래핑하여 표준 RL 알고리즘으로 학습하고 싶다.

#### Acceptance Criteria

1. THE Gym_Environment SHALL wrap CARLA simulator as a Gymnasium-compatible environment
2. THE Gym_Environment observation_space SHALL be Box(0, 255, shape=(224, 224, 3), dtype=uint8)
3. THE Gym_Environment action_space SHALL be Box([-1.0, 0.0], [1.0, 1.0], dtype=float32) for [steering, throttle]
4. THE Gym_Environment SHALL implement reset() that spawns vehicle at random spawn point and returns initial observation
5. THE Gym_Environment SHALL implement step(action) returning (observation, reward, done, info)
6. THE Gym_Environment SHALL terminate episodes on: collision, lane departure > 3m, or timeout (1,000 steps = 100초 @ 10Hz)
7. THE Gym_Environment SHALL randomize weather per episode for robustness

### Requirement 5: Reward Function 설계

**User Story:** RL 연구자로서, 안전한 주행 행동을 유도하는 보상 함수가 필요하다.

#### Acceptance Criteria

1. THE Reward_Function SHALL compute: R = w_lane × R_lane + w_collision × R_collision + w_steering × R_steering + w_progress × R_progress
2. R_lane SHALL be -|d_lane| where d_lane is perpendicular distance from lane center (meters)
3. R_collision SHALL be -100 if collision occurred, 0 otherwise
4. R_steering SHALL be -|steering| if |steering| > 0.3, 0 otherwise
5. R_progress SHALL be v_forward × cos(θ_heading)
6. THE default weights SHALL be: w_lane=1.0, w_collision=1.0, w_steering=0.5, w_progress=0.1
7. THE Reward_Function weights SHALL be configurable for tuning experiments

### Requirement 6: RL Agent 학습 (PPO + BC Warm-Start)

**User Story:** RL 연구자로서, BC 가중치를 초기값으로 PPO 에이전트를 학습하여 충돌 회피와 차선 유지를 강화하고 싶다.

#### Acceptance Criteria

1. THE RL_Agent SHALL use PPO (Proximal Policy Optimization) algorithm
2. THE RL_Agent SHALL initialize from BC Model_Checkpoint (warm-start): ResNet18_Backbone + Actor_Head reused from BC FC Head
3. THE RL_Agent SHALL add a new Critic_Head: 512 → 256 (ReLU) → 1 for state value estimation
4. THE RL_Agent SHALL share ResNet18_Backbone between Actor and Critic (parameter efficient)
5. THE RL_Agent SHALL start with backbone frozen for first 100 episodes, then full fine-tune with LR 1e-5
6. THE RL_Agent SHALL use PPO hyperparameters: LR 3e-5, GAE λ=0.95, clip ratio=0.2
7. THE RL_Agent SHALL train for 3,000~5,000 episodes until convergence
8. THE ML_System SHALL save RL Model_Checkpoint in `.pth` format

### Requirement 7: 교차로 통과 테스트

**User Story:** 검증 엔지니어로서, 학습된 모델을 교차로 시나리오에서 테스트하여 자율주행 능력을 검증하고 싶다.

#### Acceptance Criteria

1. THE ML_System SHALL deploy a trained model to CARLA without Rule_Based_Autopilot
2. THE ML_System SHALL achieve at least one Autonomous_Pass through an Intersection_Scenario
3. THE ML_System SHALL target > 80% intersection pass rate (10회 중 8회)
4. THE ML_System SHALL log successful completions with metrics (steering MAE, collision count, lane distance)

### Requirement 8: 모델 평가 메트릭

**User Story:** ML 엔지니어로서, 모델 성능을 정량적으로 측정하여 학습 접근법을 비교하고 싶다.

#### Acceptance Criteria

1. THE ML_System SHALL compute MAE between predicted and actual Steering_Value on test data
2. THE ML_System SHALL compute MAE between predicted and actual Throttle_Value on test data
3. THE ML_System SHALL count Collision_Events during evaluation episodes
4. THE ML_System SHALL measure average lane center distance during evaluation
5. THE ML_System SHALL measure average survival time (충돌 없이 주행한 시간)
6. THE ML_System SHALL log all metrics to file and console

### Requirement 9: 학습 데이터 전처리

**User Story:** ML 엔지니어로서, 학습 데이터를 적절히 전처리하여 모델이 효과적으로 학습하도록 하고 싶다.

#### Acceptance Criteria

1. THE ML_System SHALL resize Front_RGB_Image from 800×600 to 224×224
2. THE ML_System SHALL normalize pixel values using ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
3. WHEN a Front_RGB_Image file is corrupted or missing, THE ML_System SHALL skip that sample and log a warning
4. THE ML_System SHALL split Training_Dataset into train/validation subsets (80/20)
5. THE ML_System SHALL shuffle training samples before each epoch

### Requirement 10: 체크포인트 관리

**User Story:** ML 엔지니어로서, 모델 체크포인트를 저장/로드하여 학습 재개와 배포를 하고 싶다.

#### Acceptance Criteria

1. THE Model_Checkpoint SHALL contain: model_type ('bc' or 'rl'), epoch, model_state_dict, optimizer_state_dict, metrics, config (architecture, input_shape=(3,224,224)), timestamp
2. THE Model_Checkpoint filename SHALL follow format: {model_type}_{timestamp}_epoch{N}.pth
3. WHEN loading a Model_Checkpoint, THE ML_System SHALL restore both model weights and optimizer state
4. THE ML_System SHALL verify checkpoint file integrity before loading
5. WHEN a checkpoint is corrupted, THE ML_System SHALL return a descriptive error message
