# 숭산텍(Soongsan Tech) — Sim-to-Real 경량 자율주행 시스템

> **"엣지 디바이스 제약 하의 Sim-to-Real 경량 자율주행 시스템 구현 및 체계적 실증 분석"**

CARLA 시뮬레이터에서 수집한 데이터로 End-to-End 주행 모델을 학습하고,
TensorRT 양자화 + ROS2 미들웨어를 통해 RC카(Jetson) 실차 배포까지 달성합니다.
이후 이 완전한 Sim-to-Real 루프를 활용하여, 자율주행 학계가 요구하는
'데이터 효율성', 'BC-RL 전이 역학', '양자화 안전성'에 대한 희소한 실증 데이터를 산출합니다.

| 항목 | 상세 |
|------|------|
| 팀 규모 | 2명 |
| 컴퓨팅 | RTX 4090 × 1, RTX 3090 Ti × 1 |
| 시뮬레이터 | CARLA 0.9.15 (Windows Host + WSL2 Client) |
| 프레임워크 | PyTorch 2.x, TensorRT, ROS2 Humble |
| 엣지 디바이스 | NVIDIA Jetson Xavier NX / Orin Nano |
| 센서 구성 | 전방 RGB 1대 + AVM 4대 (전/후/좌/우) = 총 5대 카메라 |

---

## Spec 문서 개요

프로젝트는 3개의 Spec으로 구성되며, 각 Spec은 요구사항(requirements.md), 설계(design.md), 구현 태스크(tasks.md)를 포함합니다.

### 1. Data Pipeline (`data-pipeline`)

CARLA 시뮬레이터에서 멀티카메라(5대) 이미지와 차량 상태 데이터를 동기 수집하는 파이프라인.

- **아키텍처**: Producer-Consumer 패턴 (비동기 I/O 큐)
- **수집 주파수**: 10Hz (100ms 주기)
- **카메라 구성**: Front RGB (800×600, FOV 90°) + AVM 4대 (400×300, FOV 120°, 하향)
- **핵심 컴포넌트**: SynchronousModeController, AsyncDataLogger, EpisodeManager
- **에피소드 관리**: 5분 주기 날씨(맑음/비/안개) × 시간대(주간/야간/역광) 자동 전환
- **장애 복원**: CARLA 서버 크래시 시 수집 데이터 보존
- **상태**: Phase 1 구현 완료 (단일 카메라, 89 tests), 멀티카메라 확장 예정 (Task 11-12)

### 2. Experiment & ML Modeling (`experiment-ml-modeling`)

Front RGB 이미지로 자율주행 제어 모델을 학습하는 ML 파이프라인.

- **BC (Behavioral Cloning)**: ResNet18 backbone + FC Head → steering/throttle 출력
  - 입력: 224×224 (800×600에서 리사이즈), ImageNet pretrained
  - 학습: MSE loss (steering×2.0), Adam LR 1e-4, early stopping
  - 2단계 학습: backbone frozen (10 epoch) → full fine-tune
- **RL (Reinforcement Learning)**: BC 가중치를 warm-start하여 PPO 미세조정
  - 공유 ResNet18 backbone + Actor Head (BC FC Head 재사용) + Critic Head (신규)
  - CARLA Gym 환경: 224×224 observation, 충돌/차선이탈/타임아웃 종료
  - Reward: 차선 유지 + 충돌 페널티 + 조향 부드러움 + 전진 보상
  - PPO: LR 3e-5, GAE λ=0.95, clip 0.2, backbone frozen 100 에피소드 후 full fine-tune
- **데이터 소스**: `src/data/{session}/front/` (이미지) + `labels/driving_log.csv`
- **AVM 데이터**: BC/RL 학습에는 미사용, Phase 3+ BEV 연구에 활용

### 3. Productization (`productization`)

실험 코드를 프로덕션 ROS2 시스템으로 변환하여 RC카(Jetson) 실차 배포.

- **ROS2 노드 분리**: Front Cam(30Hz) → Inference(TRT) → Control → PWM Servo
- **멀티카메라 지원**: AVM 4-Cam Node(10Hz) + BEV Stitch Node (Phase 3+ 활성화)
- **양자화**: FP16 우선 (PyTorch → ONNX → TensorRT), INT8 선택적
  - 추론 지연 목표: < 10ms (Jetson), 정확도 오차 < 5%
- **RC카 제어**: PWM 서보/ESC (CAN 대신), steering [-1,1] → 1000~2000μs
- **듀얼 모드**: simulation(CARLA) / hardware(실차) 코드 수정 없이 전환
- **토픽**: `/sensor/front`, `/sensor/avm/*`, `/sensor/bev`, `/inference/cmd`, `/control/vehicle_command`

---

## 마일스톤

```
상용화 트랙 (v1.0) — 총 14주
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (완료)  →  Phase 2-A (4주)  →  Phase 2-B (4주)  →  Phase 2-C (6주)
데이터 수집        BC 모델 학습        RL 미세조정         엣지 배포

연구 트랙 (v2.0) — 총 12주 (상용화 트랙 완료 후)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3-A (4주)      Phase 3-B (4주)         Phase 3-C (4주)
Data Efficiency      BC→RL Transfer          Quantization Safety
```

| Phase | 기간 | 목표 | 성공 기준 | Spec |
|-------|------|------|----------|------|
| **Phase 1** | ✅ 완료 | 멀티카메라 데이터 수집 파이프라인 | 89 tests, 10Hz 동기 수집, 1시간 연속 | data-pipeline |
| **Phase 2-A** | 4주 | BC 모델 학습 (ResNet18) | MAE steering < 0.10, 교차로 1회 통과 | experiment-ml-modeling |
| **Phase 2-B** | 4주 | RL 미세조정 (PPO) | 충돌 없이 60초 주행, 교차로 80% 통과 | experiment-ml-modeling |
| **Phase 2-C** | 6주 | 엣지 배포 (Jetson + ROS2) | 추론 < 10ms, 30Hz 제어, RC카 직선 주행 | productization |
| **Phase 3-A** | 4주 | Sim-to-Real 데이터 효율성 실증 | Scaling Law 그래프, 체감점 식별 | (연구) |
| **Phase 3-B** | 4주 | BC→RL 전이 효율성 실증 | Negative Transfer 임계점 식별 | (연구) |
| **Phase 3-C** | 4주 | 양자화 안전성 분석 | 시나리오별 Degradation 히트맵 | (연구) |

---

## 센서 구성 (5대 카메라)

| 카메라 | 용도 | 위치 | 해상도 | FOV |
|--------|------|------|--------|-----|
| Front RGB | BC/RL 주행 제어 입력 | 전방 중앙 (x=1.5, z=2.4) | 800×600 | 90° |
| AVM Front | BEV 스티칭 | 전방 하단 (x=2.0, z=0.5) | 400×300 | 120° |
| AVM Rear | BEV 스티칭 | 후방 하단 (x=-2.0, z=0.5) | 400×300 | 120° |
| AVM Left | BEV 스티칭 | 좌측 하단 (y=-1.0, z=0.5) | 400×300 | 120° |
| AVM Right | BEV 스티칭 | 우측 하단 (y=1.0, z=0.5) | 400×300 | 120° |

- Front RGB → 224×224 리사이즈 후 ResNet18 입력 (BC/RL 학습)
- AVM 4대 → 호모그래피 스티칭 → BEV 이미지 → Free-Space 탐지 (Phase 3+)
- 두 스트림은 독립 수집, 동일 타임스탬프로 동기화

---

## 프로젝트 구조

```
soongsantech_carla/
├── src/
│   ├── data_pipeline/      ← Phase 1: 데이터 수집 (✅ 완료, 89 tests)
│   │   ├── pipeline.py          메인 오케스트레이터
│   │   ├── async_logger.py      비동기 I/O (Producer-Consumer 큐)
│   │   ├── episode_manager.py   에피소드 관리 (날씨/시간대 전환)
│   │   ├── sync_controller.py   CARLA 동기 모드 제어
│   │   ├── models.py            VehicleState, FrameData 데이터 모델
│   │   └── cli.py               CLI 진입점
│   │
│   ├── model/              ← Phase 2-A/B: BC/RL 모델 학습 (예정)
│   │   ├── bc_model.py          ResNet18 + FC Head (steering, throttle)
│   │   ├── bc_trainer.py        BC 학습 루프 (MSE, Adam, early stopping)
│   │   ├── rl_policy.py         Actor-Critic (BC warm-start + Critic Head)
│   │   ├── rl_trainer.py        PPO 학습 루프
│   │   ├── carla_gym_env.py     CARLA Gymnasium 래퍼
│   │   ├── reward.py            4-component Reward Function
│   │   ├── dataset.py           DrivingDataset + DataLoaderFactory
│   │   ├── checkpoint.py        체크포인트 관리
│   │   ├── evaluator.py         평가 메트릭 (MAE, 충돌, 생존시간)
│   │   └── inference.py         CARLA 실시간 추론 루프
│   │
│   ├── deploy/             ← Phase 2-C: 엣지 배포 (예정)
│   │   ├── sensor_node/         Front Cam + AVM 4-Cam + BEV Stitch
│   │   ├── inference_node/      TensorRT 추론 + 전처리
│   │   ├── control_node/        제어 명령 + 주파수 모니터링
│   │   ├── servo_node/          PWM 서보/ESC 제어
│   │   ├── quantizer/           PyTorch → ONNX → TensorRT
│   │   └── common/              성능 모니터 + 로거
│   │
│   └── data/               ← 수집 데이터 (gitignored)
│       └── {YYYY-MM-DD_HHMMSS}/
│           ├── front/           전방 RGB 800×600 PNG
│           ├── avm_front/       AVM 전방 400×300
│           ├── avm_rear/        AVM 후방 400×300
│           ├── avm_left/        AVM 좌측 400×300
│           ├── avm_right/       AVM 우측 400×300
│           ├── bev/             스티칭된 BEV (후처리)
│           └── labels/          driving_log.csv
│
├── tests/                  ← 테스트 코드
│   └── data_pipeline/          89 tests (단위 + 통합)
├── docs/                   ← 기술 문서
│   ├── 00_AutoDriving_Method_Analysis.md
│   ├── 00_Master_Spec.md
│   ├── 01_Technical_Roadmap_Final.md   ← 기술 로드맵 v1.3 (최종)
│   ├── 01_Data_Pipeline.md
│   ├── 02_Experiment.md
│   └── 03_Productization.md
├── .kiro/specs/            ← Spec 문서 (요구사항 + 설계 + 태스크)
│   ├── data-pipeline/
│   ├── experiment-ml-modeling/
│   └── productization/
├── basic/                  ← CARLA 기초 예제
├── requirements.txt
└── pytest.ini
```

---

## 핵심 아키텍처 결정

| 결정 | 선택 | 근거 |
|------|------|------|
| 모델 아키텍처 | ResNet18 + 224×224 입력 | Jetson 30Hz 추론 전제, 검증된 경량 모델 |
| 학습 전략 | BC → RL 2단계 (warm-start) | BC로 초기 정책 확보 후 PPO 미세조정 |
| BC → RL 전환 | 공유 backbone + Actor Head 재사용 + Critic Head 추가 | 파라미터 효율, 수렴 가속 |
| 양자화 순서 | FP16 우선, INT8 선택적 | FP16으로 충분한 성능, INT8은 정확도 검증 후 |
| 차량 제어 | PWM 서보/ESC (CAN 대신) | RC카 직접 구동, 하드웨어 단순화 |
| 미들웨어 | ROS2 Humble | 노드 분리 배포, 표준 로보틱스 프레임워크 |
| 연구 방향 | 체계적 실증 분석 | 모델 발명 배제, 완전한 파이프라인 기반 실험 |

---

## 기술 스택

| 계층 | 기술 |
|------|------|
| 시뮬레이터 | CARLA 0.9.15 |
| 프레임워크 | PyTorch 2.x |
| 추론 최적화 | TensorRT (FP16/INT8) via ONNX |
| 미들웨어 | ROS2 Humble |
| 엣지 디바이스 | NVIDIA Jetson Xavier NX / Orin Nano |
| 데이터 수집 | `src/data_pipeline` (구현 완료) |
| RL 환경 | Gymnasium (CARLA 래퍼) |
| 테스트 | pytest + Hypothesis (property-based testing) |

---

## 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/SoongSanTech/soongsantech_carla.git
cd soongsantech_carla
```

### 2. 가상 환경 생성 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. CARLA 서버 실행
Windows 호스트에서 CARLA 0.9.15 서버를 실행합니다.
WSL2에서는 Windows 호스트 IP(예: `172.28.224.1`)로 접속합니다.

### 4. 데이터 수집 실행
```bash
python -m data_pipeline.cli --host 172.28.224.1 --duration 3600
```

### 5. 테스트 실행
```bash
venv/bin/pytest tests/ -v
```

---

## 방향성 원칙

| 원칙 | 설명 |
|------|------|
| Sim-First | 모든 학습과 검증은 CARLA에서 먼저 수행 |
| BC → RL 2단계 | 행동복제로 초기 정책 확보 후 강화학습 미세조정 |
| 시스템 우선 | 새로운 아키텍처 발명 배제, 검증된 모델(ResNet18) 채택 |
| 엣지 최적화 | 모든 모델은 Jetson 30Hz 추론을 전제로 설계 |
| 실증 연구 집중 | 모델 개발 대신, 완전한 파이프라인 기반 체계적 실험 |

---

## Future Works (v3.0)

| 주제 | 설명 |
|------|------|
| AVM Free-Space BEV | AVM 4대 호모그래피 스티칭 → 주행 가능 영역 탐지 |
| RGB-Guided Depth Completion | 전방 RGB + 2D LiDAR 센서 퓨전 → Metric Depth Map |
| Interpretable Multi-Task E2E | Grad-CAM + Counterfactual Test → 해석 가능한 제어 |

---

## 협업 가이드

### 브랜치 전략
- `main`: 안정적인 코드만 포함 (PR을 통해서만 병합)
- `feature/*`: 기능 개발
- `fix/*`: 버그 수정

### 커밋 메시지 규칙
- `feat:` 새로운 기능
- `fix:` 버그 수정
- `docs:` 문서 수정
- `refactor:` 코드 리팩토링
- `test:` 테스트 코드

---

## 라이선스

Soongsan Tech 내부 프로젝트

---

> 상세 기술 로드맵: [`docs/01_Technical_Roadmap_Final.md`](docs/01_Technical_Roadmap_Final.md) (v1.3)
