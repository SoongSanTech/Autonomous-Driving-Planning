# 숭산텍(Soongsan Tech) 자율주행 기술 로드맵

> 작성일: 2026-03-13
> 버전: v1.3 (멀티카메라 구성 확정 — 전방 RGB + AVM 4대)
> 팀 규모: 2명 | 컴퓨팅: RTX 4090 × 1, RTX 3090 Ti × 1
> 시뮬레이터: CARLA 0.9.15 (Windows Host + WSL2 Client)

---

## 1. 프로젝트 비전 및 방향성

### 1.1 최종 목표

**"엣지 디바이스 제약 하의 Sim-to-Real 경량 자율주행 시스템 구현 및 체계적 실증 분석"**

CARLA 시뮬레이터에서 수집한 데이터로 End-to-End 주행 모델을 학습하고,
TensorRT 양자화 + ROS2 미들웨어를 통해 RC카(Jetson) 실차 배포까지 달성한다.
이후 이 완전한 Sim-to-Real 루프를 활용하여, 자율주행 학계가 요구하는
'데이터 효율성', 'BC-RL 전이 역학', '양자화 안전성'에 대한 희소한 실증 데이터를 산출한다.

### 1.2 방향성 원칙

| 원칙 | 설명 | 근거 |
|------|------|------|
| Sim-First | 모든 학습과 검증은 CARLA에서 먼저 수행 | Wayve/Waabi 트렌드 + 2명 팀 현실 |
| BC → RL 2단계 | 행동복제로 초기 정책 확보 후 강화학습 미세조정 | NVIDIA PilotNet, UC Berkeley RAIL 검증 |
| 시스템 우선 | 새로운 아키텍처 발명 배제, 검증된 모델(ResNet18) 채택 | 2인 팀 한계 극복, 모델링 병목 제거 |
| 엣지 최적화 | 모든 모델은 Jetson 30Hz 추론을 전제로 설계 | RC카 실차 배포 목표 |
| 실증 연구 집중 | 모델 개발 대신, 완전한 파이프라인 기반 체계적 실험 | 탑티어 학회 Novelty 기준 충족 |

### 1.3 기술 스택

| 계층 | 기술 | 비고 |
|------|------|------|
| 시뮬레이터 | CARLA 0.9.15 | Windows Host, TCP 2000 |
| 프레임워크 | PyTorch 2.x | 학습 및 추론 |
| 추론 최적화 | TensorRT (FP16/INT8) | ONNX 경유 변환 |
| 미들웨어 | ROS2 Humble | 노드 분리 배포 |
| 엣지 디바이스 | NVIDIA Jetson Xavier/Orin | RC카 탑재 |
| 데이터 수집 | src/data_pipeline (구현 완료) | 89 tests passing |

### 1.4 센서 구성 (5대 카메라)

| 카메라 | 용도 | 위치 | 해상도 | FOV |
|--------|------|------|--------|-----|
| Front RGB | BC/RL 주행 제어 입력 | 전방 중앙 (x=1.5, z=2.4) | 800×600 | 90° |
| AVM Front | BEV 스티칭 | 전방 하단 (x=2.0, z=0.5) | 400×300 | 120° |
| AVM Rear | BEV 스티칭 | 후방 하단 (x=-2.0, z=0.5) | 400×300 | 120° |
| AVM Left | BEV 스티칭 | 좌측 하단 (y=-1.0, z=0.5) | 400×300 | 120° |
| AVM Right | BEV 스티칭 | 우측 하단 (y=1.0, z=0.5) | 400×300 | 120° |

- Front RGB: 주행 제어 모델(BC/RL)의 직접 입력. 224×224로 리사이즈 후 ResNet18에 투입
- AVM 4대: 호모그래피 스티칭 → 단일 BEV 이미지 생성 → Free-Space 탐지 입력
- 두 스트림은 독립적으로 수집되며, 동일 타임스탬프로 동기화

---

## 2. 전체 로드맵 개요

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    상용화 트랙 (v1.0)                                │
│                                                                     │
│  Phase 1        Phase 2-A       Phase 2-B       Phase 2-C          │
│  데이터 수집 ──→ BC 모델 학습 ──→ RL 미세조정 ──→ 엣지 배포         │
│  (완료)         (4주)           (4주)           (6주)               │
│                                                                     │
│  ────────────────── 총 14주 (약 3.5개월) ──────────────────         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    연구 트랙 (v2.0) — 상용화 트랙 완료 후           │
│                                                                     │
│  Phase 3-A          Phase 3-B             Phase 3-C                │
│  Sim-to-Real        BC-to-RL Transfer     Quantization Safety      │
│  Data Efficiency    Efficiency            Analysis                 │
│  (4주)              (4주)                 (4주)                    │
│                                                                     │
│  ────────────────── 총 12주 (약 3개월) ──────────────────────       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    장기 비전 (v3.0) — Future Works                  │
│                                                                     │
│  AVM Free-Space BEV │ RGB-Guided Depth Completion │ XAI (Grad-CAM) │
│  (별도 연구 트랙)                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: 데이터 수집 파이프라인 (✅ 완료)

### 3.1 현재 상태

| 항목 | 상태 | 상세 |
|------|------|------|
| 동기 수집 엔진 | ✅ 완료 | SynchronousModeController, 10Hz 틱 |
| 비동기 I/O | ✅ 완료 | AsyncDataLogger, 2 worker threads, 1000 큐 |
| 에피소드 관리 | ✅ 완료 | 5분 주기 날씨/시간대 자동 전환 |
| 차량 자동 스폰 | ✅ 완료 | Tesla Model 3 + autopilot |
| 크래시 복원 | ✅ 완료 | CARLA 서버 크래시 시 데이터 보존 |
| 테스트 커버리지 | ✅ 89 tests | 단위 + 통합 테스트 |
| 멀티카메라 (5대) | 🔲 미구현 | Front RGB + AVM 4대 확장 필요 |
| BEV 스티칭 | 🔲 미구현 | AVM 4대 → 호모그래피 → BEV 이미지 |

### 3.2 수집 데이터 형식

```
src/data/{YYYY-MM-DD_HHMMSS}/
├── front/                     ← 전방 RGB 800×600 PNG, 10Hz
│   ├── {timestamp_ms}.png
│   └── ...
├── avm_front/                 ← AVM 전방 400×300 PNG, 10Hz
│   ├── {timestamp_ms}.png
│   └── ...
├── avm_rear/                  ← AVM 후방 400×300 PNG, 10Hz
│   ├── {timestamp_ms}.png
│   └── ...
├── avm_left/                  ← AVM 좌측 400×300 PNG, 10Hz
│   ├── {timestamp_ms}.png
│   └── ...
├── avm_right/                 ← AVM 우측 400×300 PNG, 10Hz
│   ├── {timestamp_ms}.png
│   └── ...
├── bev/                       ← 스티칭된 BEV 이미지 (후처리 생성)
│   ├── {timestamp_ms}.png
│   └── ...
└── labels/
    └── driving_log.csv        ← image_filename, speed, steering, throttle, brake
```

> `bev/` 디렉토리는 AVM 4장을 호모그래피 스티칭한 결과물이며,
> 수집 시점에 실시간 생성하거나 후처리로 일괄 생성할 수 있다.

### 3.3 수집 권장량

| 목적 | 수집 시간 | 프레임 수 | 용량 (5카메라) | 에피소드 수 |
|------|-----------|-----------|---------------|------------|
| BC 초기 실험 | 1시간 | 36,000 × 5cam | ~85 GB | 12 |
| BC 본격 학습 | 6시간 | 216,000 × 5cam | ~510 GB | 72 |
| Phase 3-A 실험용 | 조건별 1시간씩 | 조건당 36,000 × 5cam | 조건당 ~85 GB | 조건당 12 |

> 용량 산정: Front RGB(800×600) ~1.4MB/frame + AVM 4대(400×300) ~0.36MB/frame × 4 ≈ 2.84MB/frame 총합.
> BC 학습에는 Front RGB만 사용하므로, AVM 데이터는 Phase 3 이후 BEV 실험에 활용.

---

## 4. Phase 2-A: Behavioral Cloning 모델 학습 (4주)

### 4.1 목표

수집된 데이터 중 Front RGB 이미지로 ResNet18 기반 BC 모델을 학습하여,
CARLA 시뮬레이터 내에서 autopilot 없이 직선 도로 주행 및 교차로 통과를 달성한다.
AVM 데이터는 이 단계에서는 사용하지 않으며, Phase 3 이후 BEV 실험에 활용한다.

### 4.2 모델 아키텍처

```
Input: Front RGB Image (800×600×3) → Resize to 224×224×3
    ↓
ResNet18 (ImageNet pretrained, conv layers frozen initially)
    → Feature vector (512-d)
    ↓
FC Head
    → FC1: 512 → 256 (ReLU + Dropout 0.5)
    → FC2: 256 → 128 (ReLU + Dropout 0.3)
    → FC3: 128 → 2
    ↓
Output Heads
    → steering: tanh → [-1, 1]
    → throttle: sigmoid → [0, 1]
```

### 4.3 학습 파이프라인 상세

#### 데이터 전처리

| 단계 | 처리 | 비고 |
|------|------|------|
| 로드 | driving_log.csv 파싱 + 이미지 경로 매핑 | 누락/손상 파일 자동 스킵 |
| 정규화 | 픽셀값 [0, 255] → [0.0, 1.0] | ImageNet mean/std 적용 |
| 리사이즈 | 800×600 → 224×224 (ResNet 입력) | torchvision.transforms |
| 증강 | 좌우 반전 (steering 부호 반전), 밝기 ±20%, 가우시안 노이즈 | 과적합 방지 |
| 분할 | 80% train / 20% validation | 에피소드 단위 분할 권장 |

#### 학습 설정

| 하이퍼파라미터 | 값 | 근거 |
|---------------|-----|------|
| Optimizer | Adam | 수렴 안정성 |
| Learning Rate | 1e-4 (초기), ReduceLROnPlateau | ResNet pretrained 미세조정 |
| Batch Size | 32 | 4090 VRAM 24GB 기준 |
| Loss | MSE (steering) + MSE (throttle) | 회귀 태스크 |
| Loss 가중치 | steering × 2.0, throttle × 1.0 | 조향이 더 중요 |
| Epochs | 50 (early stopping patience=10) | val_loss 기준 |
| 학습 전략 | Phase 1: backbone frozen (10 epoch) → Phase 2: full fine-tune | 전이학습 표준 |

#### 체크포인트 저장

```python
{
    'model_type': 'bc',
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'metrics': {'train_loss', 'val_loss', 'mae_steering', 'mae_throttle'},
    'config': {'architecture': 'resnet18', 'input_shape': (3, 224, 224)},
    'timestamp': str  # ISO format
}
```

### 4.4 주간 마일스톤

| 주차 | 산출물 | 성공 기준 |
|------|--------|----------|
| Week 1 | DrivingDataset + DataLoader 구현 | 데이터 로드 + 전처리 파이프라인 동작 |
| Week 2 | BehavioralCloningModel + 학습 루프 | val_loss 수렴, MAE steering < 0.15 |
| Week 3 | CARLA 추론 루프 (모델 → 차량 제어) | 직선 도로 30초 이상 자율주행 |
| Week 4 | 교차로 통과 테스트 | autopilot 없이 1회 이상 교차로 통과 |

---

## 5. Phase 2-B: 강화학습 미세조정 (4주)

### 5.1 목표

BC 모델의 가중치를 초기값으로 사용하여 PPO 기반 RL 에이전트로 전환하고,
CARLA Gym 환경에서 충돌 회피 + 차선 유지 능력을 강화한다.

### 5.2 CARLA Gym 환경 설계

```python
class CARLAGymEnv(gym.Env):
    observation_space = Box(0, 255, shape=(224, 224, 3), dtype=uint8)
    action_space = Box([-1.0, 0.0], [1.0, 1.0], dtype=float32)
    # action[0] = steering, action[1] = throttle
```

#### 에피소드 구성

| 항목 | 값 | 비고 |
|------|-----|------|
| 최대 스텝 | 1,000 (100초 @ 10Hz) | 타임아웃 종료 |
| 종료 조건 | 충돌 / 차선 이탈 3m / 타임아웃 | done=True |
| 스폰 | 랜덤 spawn point | 일반화 |
| 날씨 | 에피소드마다 랜덤 | 로버스트니스 |

### 5.3 Reward Function

```
R(s, a, s') = w_lane × R_lane
            + w_collision × R_collision
            + w_steering × R_steering
            + w_progress × R_progress

R_lane      = -|d_lane|                          (차선 중앙 유지)
R_collision = -100 if collision else 0            (충돌 페널티)
R_steering  = -|steer| if |steer| > 0.3 else 0   (과도 조향 억제)
R_progress  = v_forward × cos(θ_heading)          (전진 보상)

가중치: w_lane=1.0, w_collision=1.0, w_steering=0.5, w_progress=0.1
```

> 가중치는 초기값이며, 실험을 통해 튜닝 필요.
> 에이전트가 "가만히 서 있기" 지역 최적해에 빠지면 w_progress를 0.3~0.5로 상향.

### 5.4 RL 학습 설정

| 항목 | 값 | 근거 |
|------|-----|------|
| 알고리즘 | PPO (Proximal Policy Optimization) | 안정성 최우선, 튜닝 용이 |
| 정책 네트워크 | BC 모델 가중치로 초기화 (warm-start) | 수렴 가속 |
| Actor-Critic | 공유 ResNet18 backbone + 분리된 head | 파라미터 효율 |
| 학습률 | 3e-5 (BC보다 10배 낮게) | 사전학습 가중치 보존 |
| GAE λ | 0.95 | 표준값 |
| Clip ratio | 0.2 | PPO 표준 |
| 에피소드 수 | 3,000~5,000 | 수렴까지 |
| 병렬 환경 | 1 (CARLA 단일 인스턴스) | 리소스 제약 |

### 5.5 BC → RL 전환 절차

```
1. BC 체크포인트 로드 (best val_loss 기준)
2. FC Head를 Actor Head로 재사용
3. Value Head (Critic) 신규 추가: 512 → 256 → 1
4. Backbone은 frozen 상태로 시작 (100 에피소드)
5. 이후 전체 fine-tune (learning rate 1e-5)
```

### 5.6 주간 마일스톤

| 주차 | 산출물 | 성공 기준 |
|------|--------|----------|
| Week 5 | CARLAGymEnv + RewardFunction | env.step() 정상 동작, reward 출력 확인 |
| Week 6 | PPO 학습 루프 (BC warm-start) | 500 에피소드 후 평균 reward 상승 추세 |
| Week 7 | Reward 튜닝 + 학습 안정화 | 충돌 없이 60초 이상 주행 |
| Week 8 | 교차로 + 곡선 도로 통과 | 3회 연속 교차로 통과 |

---

## 6. Phase 2-C: 엣지 배포 및 실차 검증 (6주)

### 6.1 목표

학습된 모델을 TensorRT로 양자화하고, ROS2 노드로 분리하여
RC카(Jetson) 위에서 30Hz 실시간 자율주행을 달성한다.

### 6.2 모델 양자화 파이프라인

```
PyTorch (.pth)
    ↓ torch.onnx.export (opset 13, input: 1×3×224×224)
ONNX (.onnx)
    ↓ trtexec --fp16 (1차 시도)
TensorRT Engine (.trt, FP16)
    ↓ 정확도 검증 (PyTorch 대비 오차 < 5%)
    ↓ 통과 시 FP16 확정, 미통과 시 FP32 fallback
    ↓
    ↓ (선택) INT8 양자화 시도
    ↓ 캘리브레이션 데이터 500장 생성
    ↓ 정확도 검증 (오차 < 5%)
최종 Engine (.trt)
```

#### 양자화 성능 목표

| 메트릭 | FP32 (PyTorch) | FP16 (TensorRT) | INT8 (TensorRT) |
|--------|:-:|:-:|:-:|
| 추론 지연 | ~15ms (4090) | ~5ms (Jetson) | ~3ms (Jetson) |
| 정확도 오차 | baseline | < 2% | < 5% |
| 모델 크기 | ~45MB | ~23MB | ~12MB |

### 6.3 ROS2 노드 아키텍처

```
┌──────────────┐    /sensor/front       ┌──────────────┐    /inference/cmd    ┌──────────────┐
│  Front Cam   │ ──────────────────→   │ Inference    │ ──────────────────→ │ Control Node │
│  Node        │    Image (30Hz)       │ Node (TRT)  │   Steering+Throttle │ (액추에이터)  │
└──────────────┘                        └──────────────┘                     └──────────────┘
                                              ↑
┌──────────────┐    /sensor/avm/*       ┌──────────────┐
│  AVM 4-Cam   │ ──────────────────→   │ BEV Stitch  │ → /sensor/bev (Free-Space용)
│  Node        │    4× Image (10Hz)    │ Node        │
└──────────────┘                        └──────────────┘
```

#### 노드별 책임

| 노드 | 입력 | 출력 | 주기 |
|------|------|------|------|
| Front Cam Node | 전방 카메라 raw image | /sensor/front (800×600 RGB) | 30Hz |
| AVM 4-Cam Node | AVM 4대 raw image | /sensor/avm/{front,rear,left,right} | 10Hz |
| BEV Stitch Node | /sensor/avm/* | /sensor/bev (스티칭된 BEV) | 10Hz |
| Inference Node | /sensor/front | /inference/cmd (steering, throttle) | 30Hz |
| Control Node | /inference/cmd | 차량 제어 명령 | 30Hz |

> Phase 2에서는 Front Cam → Inference → Control 경로만 활성화.
> BEV Stitch Node는 Future Works(v3.0) AVM Free-Space 연구 시 활성화.

### 6.4 RC카 하드웨어 구성

| 항목 | 사양 | 비고 |
|------|------|------|
| 컴퓨팅 | NVIDIA Jetson Xavier NX / Orin Nano | TensorRT 지원 |
| 전방 카메라 | USB 또는 CSI 카메라 (800×600) | 주행 제어용 1대 |
| AVM 카메라 | 광각 카메라 × 4 (400×300) | 전/후/좌/우, BEV 스티칭용 |
| 차체 | 1/10 스케일 RC카 | 조향 서보 + ESC |
| 통신 | ROS2 → PWM 변환 | servo_node 추가 |

### 6.5 주간 마일스톤

| 주차 | 산출물 | 성공 기준 |
|------|--------|----------|
| Week 9 | ONNX 변환 + TensorRT FP16 빌드 | Jetson에서 추론 < 10ms |
| Week 10 | ROS2 Sensor + Inference Node | 토픽 통신 정상, 30Hz 유지 |
| Week 11 | ROS2 Control Node + CARLA 연동 | 시뮬레이터에서 ROS2 경유 자율주행 |
| Week 12 | RC카 하드웨어 조립 + 카메라 연동 | 실시간 이미지 스트리밍 확인 |
| Week 13-14 | RC카 실차 주행 테스트 | 직선 + 완만한 커브 자율주행 |

---

## 7. Phase 3-A: Sim-to-Real 데이터 효율성 실증 (4주)

### 7.1 연구 질문

**"Sim-to-Real 전이에서 데이터의 '절대량(Quantity)'과 '분포의 다양성(Diversity)' 중
무엇이 지배적인 변수인가?"**

### 7.2 가설

> H1: 6시간의 단일 조건(ClearNoon/DAYTIME) 데이터로 학습한 BC 모델보다,
> 1시간의 전 조건 혼합(9가지 날씨×시간대 조합) 데이터로 학습한 BC 모델이
> 미학습 조건(unseen condition)에서의 CARLA 교차로 통과 성공률이 더 높을 것이다.

> H2: 데이터 양 증가의 한계 수익 체감점(diminishing returns threshold)이 존재하며,
> 이 임계점 이후에는 다양성 확보가 양 증가보다 성능 향상에 더 기여할 것이다.

### 7.3 실험 설계

#### 독립 변수

| 조건 ID | 데이터 양 | 날씨/시간대 다양성 | 설명 |
|---------|-----------|-------------------|------|
| Q-10m | 10분 (6,000 프레임) | 단일 (ClearNoon/DAYTIME) | 최소 양 |
| Q-30m | 30분 (18,000) | 단일 | 중간 양 |
| Q-1h | 1시간 (36,000) | 단일 | 기본 양 |
| Q-3h | 3시간 (108,000) | 단일 | 대량 |
| Q-6h | 6시간 (216,000) | 단일 | 최대 양 |
| D-1h | 1시간 (36,000) | 9가지 전 조합 (각 ~4,000) | 다양성 극대화 |
| D-3h | 3시간 (108,000) | 9가지 전 조합 (각 ~12,000) | 양 + 다양성 |

#### 날씨×시간대 조합 (9가지)

| | DAYTIME (0°) | BACKLIGHT (90°) | NIGHT (180°) |
|---|:---:|:---:|:---:|
| CLEAR (ClearNoon) | ✅ | ✅ | ✅ |
| RAIN (WetCloudyNoon) | ✅ | ✅ | ✅ |
| FOG (SoftRainSunset) | ✅ | ✅ | ✅ |

#### 종속 변수 (평가 메트릭)

| 메트릭 | 설명 | 평가 환경 |
|--------|------|----------|
| 교차로 통과율 | 10회 시도 중 성공 횟수 | CARLA Town03 (학습 맵) |
| 교차로 통과율 (unseen) | 10회 시도 중 성공 횟수 | CARLA Town05 (미학습 맵) |
| 평균 생존 시간 | 충돌 없이 주행한 평균 시간 | 랜덤 스폰, 3분 제한 |
| MAE Steering | 예측 vs autopilot GT | 테스트 데이터셋 |
| RC카 직선 주행 거리 | 실차 전이 성공 지표 | 실내 직선 코스 (가능한 경우) |

#### 통제 변수

모든 조건에서 동일: ResNet18 아키텍처, 학습 하이퍼파라미터, 평가 시나리오

### 7.4 기대 산출물

- 데이터 양 vs 성능 곡선 (Scaling Law 그래프)
- 다양성 vs 양의 교차 비교 테이블
- 한계 수익 체감점 식별
- 논문 타겟: ICRA/IROS 또는 IEEE IV Workshop

### 7.5 주간 마일스톤

| 주차 | 작업 | 산출물 |
|------|------|--------|
| Week 15 | 7가지 조건별 데이터 수집 (자동) | 수집 완료 데이터셋 |
| Week 16 | 7개 BC 모델 학습 (동일 설정) | 7개 체크포인트 |
| Week 17 | CARLA 평가 (Town03 + Town05) | 메트릭 테이블 |
| Week 18 | 분석 + 그래프 + 논문 초고 | Scaling Law 그래프, 가설 검증 결과 |

---

## 8. Phase 3-B: BC-to-RL 전이 효율성 실증 (4주)

### 8.1 연구 질문

**"불완전한 행동복제(Sub-optimal BC)가 강화학습 수렴에 미치는 영향에
임계점(Tipping Point)이 존재하는가?"**

### 8.2 가설

> H3: 완벽한 autopilot 데이터만으로 학습한 BC(Clean BC)보다,
> 조향에 의도적 노이즈(±0.2 Gaussian)를 주입하고 autopilot이 복구하는
> 과정이 포함된 BC(Noisy BC)가 PPO 에이전트의 수렴 속도를 높일 수 있다.

> H4: 그러나 노이즈 수준이 일정 임계값(예: ±0.5)을 초과하면,
> BC pretrain이 오히려 random initialization보다 RL 수렴을 방해하는
> "negative transfer" 영역이 존재할 것이다.

### 8.3 실험 설계

#### 노이즈 주입 방법 (CARLA 수집 시)

```python
# autopilot 제어값에 Gaussian noise 추가 후 기록
noisy_steering = autopilot_steering + np.random.normal(0, noise_level)
noisy_steering = np.clip(noisy_steering, -1.0, 1.0)
# autopilot은 다음 틱에서 자연스럽게 복구 → 복구 궤적이 데이터에 포함됨
```

#### 독립 변수

| 조건 ID | BC Pretrain | 노이즈 수준 | 설명 |
|---------|------------|------------|------|
| No-BC | 없음 (random init) | N/A | RL baseline |
| Clean-BC | autopilot 1시간 | σ=0.0 | 완벽한 BC |
| Noisy-0.1 | autopilot 1시간 | σ=0.1 | 약한 노이즈 |
| Noisy-0.2 | autopilot 1시간 | σ=0.2 | 중간 노이즈 |
| Noisy-0.3 | autopilot 1시간 | σ=0.3 | 강한 노이즈 |
| Noisy-0.5 | autopilot 1시간 | σ=0.5 | 극단적 노이즈 |

#### 종속 변수

| 메트릭 | 설명 |
|--------|------|
| 수렴 에피소드 수 | 평균 reward가 임계값 도달까지 필요한 에피소드 |
| 최종 평균 reward | 5,000 에피소드 후 최종 성능 |
| 충돌률 | 평가 에피소드 100회 중 충돌 비율 |
| 교차로 통과율 | 평가 에피소드 중 교차로 성공 비율 |
| 수렴 곡선 형태 | 지역 최적해 빠짐 여부 시각적 분석 |

### 8.4 기대 산출물

- 노이즈 수준 vs RL 수렴 속도 그래프
- Negative transfer 임계점 식별
- "최적의 BC pretrain 품질" 가이드라인
- 논문 타겟: ICRA/IROS 또는 CoRL

### 8.5 주간 마일스톤

| 주차 | 작업 | 산출물 |
|------|------|--------|
| Week 19 | 6가지 노이즈 조건 데이터 수집 + BC 학습 | 6개 BC 체크포인트 |
| Week 20 | 6개 조건 PPO 학습 (각 5,000 에피소드) | 수렴 곡선 6개 |
| Week 21 | CARLA 평가 (교차로, 곡선, 야간) | 메트릭 테이블 |
| Week 22 | 분석 + 임계점 식별 + 논문 초고 | Tipping Point 그래프 |

---

## 9. Phase 3-C: 양자화 안전성 분석 (4주)

### 9.1 연구 질문

**"FP32 → FP16 → INT8 양자화 시, 정확도 손실이 '어떤 주행 상황에서'
안전 임계점을 넘어서는가?"**

### 9.2 가설

> H5: INT8 양자화 모델은 주간/맑은 조건에서 FP32 대비 5% 이내의 성능 저하를
> 보이지만, 야간 + 폭우 조건에서는 충돌률이 FP32 대비 2배 이상 증가할 것이다.

> H6: 양자화로 인한 성능 저하는 steering 예측에서 throttle 예측보다
> 더 크게 나타날 것이다. (조향은 미세한 값 차이가 궤적에 큰 영향)

### 9.3 실험 설계

#### 독립 변수

| 모델 | 정밀도 | 추론 환경 |
|------|--------|----------|
| FP32 | PyTorch 원본 | RTX 4090 |
| FP16 | TensorRT FP16 | Jetson |
| INT8 | TensorRT INT8 | Jetson |

#### 평가 시나리오 (CARLA)

| 시나리오 ID | 도로 | 날씨 | 시간대 | 난이도 |
|------------|------|------|--------|--------|
| S1 | 직선 | Clear | Daytime | 쉬움 |
| S2 | 커브 | Clear | Daytime | 보통 |
| S3 | 교차로 | Clear | Daytime | 보통 |
| S4 | 직선 | Rain | Daytime | 보통 |
| S5 | 교차로 | Rain | Night | 어려움 |
| S6 | 커브 | Fog | Night | 어려움 |
| S7 | 교차로 | Rain | Backlight | 매우 어려움 |

#### 종속 변수

| 메트릭 | 설명 |
|--------|------|
| MAE Steering | 시나리오별 조향 오차 |
| MAE Throttle | 시나리오별 가속 오차 |
| 충돌률 | 시나리오별 10회 중 충돌 횟수 |
| 차선 이탈률 | 시나리오별 차선 이탈 비율 |
| 추론 지연 | 시나리오별 평균 추론 시간 |

#### 실차 테스트 (Phase 2-C 완료 시)

RC카 하드웨어가 동작하는 경우, 추가로 물리적 환경 변수 테스트:

| 환경 | 설명 | 목적 |
|------|------|------|
| 밝은 실내 | 형광등 조명 | 기본 조건 |
| 어두운 복도 | 저조도 환경 | 야간 시뮬레이션 |
| 역광 | 창문 앞 주행 | 카메라 포화 테스트 |

> 실차 테스트는 "가능한 경우 추가"이며, 시뮬레이터 결과가 기본 논문 데이터.

### 9.4 기대 산출물

- 양자화 수준 × 시나리오 난이도 히트맵 (성능 degradation profile)
- "INT8에서만 실패하는 코너 케이스" 카탈로그
- 양자화 수준 선택 가이드라인 (시나리오별 권장 정밀도)
- 논문 타겟: IEEE IV, SAE International, 또는 ICRA Safety Workshop

### 9.5 주간 마일스톤

| 주차 | 작업 | 산출물 |
|------|------|--------|
| Week 23 | FP32/FP16/INT8 엔진 3개 준비 | TensorRT 엔진 파일 |
| Week 24 | 7개 시나리오 × 3개 모델 = 21회 평가 | 메트릭 테이블 |
| Week 25 | 실차 테스트 (가능한 경우) + 코너 케이스 분석 | 실패 사례 카탈로그 |
| Week 26 | 분석 + 히트맵 + 논문 초고 | Degradation Profile |

---

## 10. 평가 체계 종합

### 10.1 상용화 트랙 평가 메트릭

| 메트릭 | 목표 | Phase |
|--------|------|-------|
| MAE Steering | < 0.10 | 2-A |
| MAE Throttle | < 0.08 | 2-A |
| 교차로 통과율 | > 80% (10회 중 8회) | 2-A, 2-B |
| 충돌 없이 주행 시간 | > 60초 | 2-B |
| 추론 지연 (Jetson) | < 10ms | 2-C |
| ROS2 제어 주기 | ≥ 30Hz | 2-C |
| RC카 직선 주행 | > 10m | 2-C |

### 10.2 연구 트랙 평가 메트릭

| 메트릭 | 목표 | Phase |
|--------|------|-------|
| Scaling Law 그래프 | 체감점 식별 | 3-A |
| Diversity vs Quantity 비교 | 통계적 유의성 | 3-A |
| Negative Transfer 임계점 | 노이즈 σ 값 식별 | 3-B |
| 양자화 Degradation 히트맵 | 시나리오별 프로파일 | 3-C |

---

## 11. 리스크 관리

### 11.1 기술 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|:---:|:---:|----------|
| BC 모델이 교차로에서 실패 | 높음 | 중간 | 교차로 데이터 비중 증가, 데이터 증강 |
| RL 학습 미수렴 | 중간 | 높음 | BC warm-start, reward 가중치 그리드 서치 |
| TensorRT 변환 시 정확도 손실 > 5% | 낮음 | 중간 | FP16 fallback, 캘리브레이션 데이터 증가 |
| RC카 하드웨어 호환성 | 중간 | 높음 | 시뮬레이터 검증 완료 후 하드웨어 진입 |
| Phase 3 실험에서 가설 기각 | 중간 | 낮음 | 기각 자체도 논문 기여 (null result) |

### 11.2 일정 리스크

| 리스크 | 완화 전략 |
|--------|----------|
| Phase 2가 예상보다 지연 | Phase 3는 독립 연구 트랙이므로 영향 없음 |
| 2명 중 1명 이탈 | 상용화 트랙만 집중, 연구 트랙 보류 |
| 번아웃 | 2주 스프린트 + 1일 회고, Phase 3는 여유 시 진행 |

---

## 12. 코드 구조

```
src/
├── data_pipeline/          ← Phase 1 (✅ 완료)
│   ├── pipeline.py
│   ├── async_logger.py
│   ├── episode_manager.py
│   ├── sync_controller.py
│   ├── models.py
│   └── cli.py
│
├── model/                  ← Phase 2 (예정)
│   ├── bc_model.py         ← ResNet18 + FC Head
│   ├── bc_trainer.py       ← 학습 루프
│   ├── rl_policy.py        ← Actor-Critic (PPO)
│   ├── rl_trainer.py       ← PPO 학습 루프
│   ├── carla_gym_env.py    ← Gym 래퍼
│   ├── reward.py           ← Reward Function
│   ├── dataset.py          ← DrivingDataset + DataLoader
│   ├── checkpoint.py       ← 체크포인트 관리
│   └── inference.py        ← CARLA 추론 루프
│
├── deploy/                 ← Phase 2-C (예정)
│   ├── quantizer.py        ← PyTorch → ONNX → TensorRT
│   ├── sensor_node.py      ← ROS2 Sensor Node
│   ├── inference_node.py   ← ROS2 Inference Node
│   └── control_node.py     ← ROS2 Control Node
│
├── experiments/            ← Phase 3 (예정)
│   ├── data_efficiency/    ← Phase 3-A 실험 스크립트
│   ├── bc_rl_transfer/     ← Phase 3-B 실험 스크립트
│   └── quant_safety/       ← Phase 3-C 실험 스크립트
│
└── data/                   ← 수집 데이터 (gitignored)
    └── {YYYY-MM-DD_HHMMSS}/
        ├── front/          ← 전방 RGB 800×600
        ├── avm_front/      ← AVM 전방 400×300
        ├── avm_rear/       ← AVM 후방 400×300
        ├── avm_left/       ← AVM 좌측 400×300
        ├── avm_right/      ← AVM 우측 400×300
        ├── bev/            ← 스티칭된 BEV (후처리)
        └── labels/         ← driving_log.csv
```


---

## 13. 즉시 실행 항목 (Immediate Next Actions)

Phase 2-A 진입을 위한 최우선 작업 목록. 순서대로 실행한다.

| 순서 | 작업 | 파일 | 설명 |
|:---:|------|------|------|
| 0 | 멀티카메라 파이프라인 확장 | `src/data_pipeline/pipeline.py` | Front RGB + AVM 4대 센서 부착, 5채널 동기 수집 |
| 1 | CARLA 데이터 수집 (1시간) | `src/data_pipeline/cli.py` | ClearNoon/DAYTIME 조건, 10Hz, 36,000 프레임 × 5cam |
| 2 | DrivingDataset 구현 | `src/model/dataset.py` | front/ 이미지 + driving_log.csv 파싱, 전처리, train/val 분할 |
| 3 | BehavioralCloningModel 구현 | `src/model/bc_model.py` | ResNet18 backbone + FC Head (steering, throttle) |
| 4 | BC 학습 루프 구현 | `src/model/bc_trainer.py` | Adam, MSE loss, early stopping, 체크포인트 저장 |
| 5 | CARLA 추론 루프 구현 | `src/model/inference.py` | 학습된 모델 로드 → 전방 카메라 입력 → 차량 제어 실시간 루프 |

> 작업 0이 새로 추가됨: 기존 단일 카메라 파이프라인을 5대 멀티카메라로 확장.
> 작업 2~5에서 BC 학습은 Front RGB만 사용. AVM 데이터는 수집만 하고 Phase 3에서 활용.

---

## 14. Future Works (v3.0 — 장기 비전)

상용화 트랙(v1.0)과 연구 트랙(v2.0) 완료 후, 추가 연구 확장이 가능한 방향.
각 항목은 독립적인 연구 주제이며, 팀 역량과 시간에 따라 선택적으로 진행한다.

### 14.1 AVM 기반 Free-Space BEV 모델링

- 데이터 수집 인프라는 상용화 트랙(Phase 1)에서 이미 5대 카메라로 구축 완료
- AVM 4대 이미지의 호모그래피 스티칭 → 단일 BEV 이미지 생성
- 물체 인식이 아닌 **주행 가능 영역(Free-Space) 탐지**에 집중
- 호모그래피의 수직 물체 왜곡을 역이용: 왜곡 픽셀 = 장애물로 분류
- 엣지 디바이스(Jetson)에서 실시간 처리 가능한 경량 세그멘테이션 모델 적용
- BEV Free-Space 출력을 주행 제어 모델의 보조 입력으로 융합하는 실험

### 14.2 RGB-Guided Depth Completion

- 전방 카메라(RGB)와 2D LiDAR의 센서 퓨전
- 카메라의 상대 깊이(Relative Depth) 추정 + 2D LiDAR의 절대 거리(Absolute Depth)로 스케일 교정
- Pretrained Depth Estimation 모델(예: Depth Anything V2) + Affine Alignment 활용
- **목표**: 저비용 센서 셋업(15만 원)으로 Metric Depth Estimation 달성
- ⚠️ "64채널 3D LiDAR 수준 복원"은 정보 이론적으로 불가능 — Metric Depth Map 생성이 현실적 목표

### 14.3 해석 가능한(Interpretable) Multi-Task E2E

- 주행 제어(steering/throttle) + 객체 탐지를 단일 네트워크에서 동시 수행
- **Grad-CAM**을 Control Head에 연결하여 제동/조향 시 활성화 영역 시각화
- Grad-CAM 히트맵만으로는 인과관계 증명 불충분 → **Counterfactual Test** 추가
  - 활성화 영역을 마스킹한 입력으로 재추론 → 제어값 변화 측정
  - 변화가 유의미하면 인과관계 성립
- "설명 가능한(Explainable)" 대신 **"해석 가능한(Interpretable)"**으로 표현 (학술적 정확성)

---

> **문서 이력**
> - v1.0 (2026-03-13): 초안 작성
> - v1.1 (2026-03-13): 실증 연구 및 시스템 안정성 피벗 반영
> - v1.2 (2026-03-14): 연구 트랙 가설 예리화, Future Works 추가
> - v1.3 (2026-03-14): 멀티카메라 구성 확정 (Front RGB + AVM 4대 = 5대), 데이터 형식/ROS2 아키텍처/코드 구조 전면 반영
