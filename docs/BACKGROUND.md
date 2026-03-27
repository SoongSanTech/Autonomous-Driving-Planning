# Background: 숭산텍 Sim-to-Real 경량 자율주행 기술 백서

본 문서는 숭산텍(Soongsan Tech)의 '엣지 디바이스 제약 하의 Sim-to-Real 경량 자율주행 시스템'을 구현하기 위해 요구되는 단계별 핵심 기술과 이론적 배경을 정의합니다. 

## 1. Data Pipeline & Simulation 환경 (Phase 1)

다중 센서 데이터의 동기화 및 실시간 병목 현상을 제어하기 위한 기반 지식입니다.

* **비동기 I/O 및 분산 처리 (Producer-Consumer Pattern):**
    * **개념:** 데이터 생성(CARLA 카메라 센서)과 소비(디스크 I/O 로깅)를 독립적인 스레드/프로세스로 분리하는 아키텍처입니다.
    * **적용:** 10Hz 주기로 들어오는 5대의 카메라(Front 1, AVM 4)와 차량 상태 데이터를 큐(Queue)에 담아 비동기 처리함으로써 시뮬레이터의 프레임 드랍을 방지합니다.
* **센서 기하학 및 뷰 변환 (Homography & BEV):**
    * **개념:** 서로 다른 시점의 이미지를 기하학적으로 변환하여 하나의 평면으로 투영하는 기법입니다.
    * **적용:** 4대의 AVM(어라운드 뷰 모니터) 카메라 영상을 결합하여 차량 위에서 내려다보는 BEV(Bird's Eye View)를 생성, 향후 Free-Space 탐지에 활용합니다.
* **공간 좌표계 변환 (Coordinate Transformations):**
    * **개념:** World, Ego(차량), Sensor(카메라) 좌표계 간의 3D Translation 및 Rotation(Roll, Pitch, Yaw) 연산입니다.

## 2. Visual Perception & 모방 학습 (Phase 2-A)

전문가(인간 또는 자동 차량)의 주행 데이터를 기반으로 초기 주행 정책(Policy)을 수립합니다.

* **행동 복제 (Behavioral Cloning, BC):**
    * **개념:** 시각적 관측치(State)를 입력받아 전문가의 조향 및 가속(Action)을 지도 학습(Supervised Learning) 방식으로 모방하는 기법입니다.
    * **한계 인지:** 누적 오차로 인해 학습 데이터 분포를 벗어나는 상황에서 실패하는 **공변량 변화(Covariate Shift)** 및 과거 상태에 과적합되는 **관성(Inertia)** 문제가 발생할 수 있습니다.
* **CNN 기반 경량 특징 추출 (ResNet18):**
    * **적용:** 30Hz 엣지 추론을 목표로 잔차 연결(Residual Connection)이 적용된 ResNet18을 Backbone으로 사용합니다. 
    * **전이 학습 (Transfer Learning):** ImageNet Pre-trained 가중치를 사용하여 Backbone을 동결(Frozen)한 후 FC Head를 먼저 학습시키고, 이후 전체를 미세 조정(Fine-tuning)하여 수렴을 가속합니다.

## 3. 정책 최적화 및 심층 강화학습 (Phase 2-B)

BC로 확보된 기초 정책을 동적이고 불확실한 환경에서 스스로 최적화하도록 강화합니다.

* **Actor-Critic 아키텍처:**
    * **개념:** 상태를 입력받아 최적의 행동 확률을 반환하는 Actor(BC의 FC Head 재사용)와, 현재 상태의 가치를 평가하는 Critic Head가 협력하는 구조입니다.
* **근접 정책 최적화 (Proximal Policy Optimization, PPO):**
    * **개념:** 정책 업데이트 시 이전 정책과 현재 정책의 비율(Ratio)을 계산하고, 이를 일정 범위(Clip) 내로 제한하여 학습의 급격한 붕괴를 막는 On-policy 강화학습 알고리즘입니다.
* **일반화된 이점 추정 (Generalized Advantage Estimation, GAE):**
    * **개념:** 강화학습에서 에이전트가 받는 보상 신호의 분산(Variance)과 편향(Bias) 사이의 트레이드오프를 λ(lambda) 파라미터로 조절하여 학습 안정성을 높이는 기법입니다.

## 4. 로보틱스 미들웨어 및 엣지 배포 (Phase 2-C)

가상 환경에서 학습된 모델을 Jetson Orin / Xavier NX 실차 환경에 통합합니다.

* **ROS2 (Robot Operating System 2) 아키텍처:**
    * **개념:** DDS(Data Distribution Service) 기반의 분산 통신 프레임워크입니다.
    * **적용:** 추론 노드(Inference Node), 센서 노드, 제어 노드를 완전히 분리(Decoupling)하여 토픽(Topic) 기반으로 비동기 통신을 수행합니다. 시뮬레이션 코드와 실차 코드를 수정 없이 전환할 수 있는 유연성을 제공합니다.
* **하드웨어 동역학 제어 (PWM & ESC):**
    * **개념:** 모델의 출력값(조향 [-1, 1])을 물리적인 조향각으로 변환하기 위해 1000~2000μs의 PWM(Pulse Width Modulation) 신호로 매핑하여 서보모터와 ESC를 직접 제어합니다.

## 5. 최적화 및 Sim-to-Real 실증 분석 (Phase 3)

모델의 실제 구동 속도를 확보하고 가상과 현실의 도메인 격차를 극복합니다.

* **신경망 양자화 (Neural Network Quantization) 및 TensorRT:**
    * **개념:** FP32 파라미터를 FP16 또는 INT8로 압축하여 추론 속도와 메모리 효율을 극대화합니다.
    * **적용:** PyTorch 모델을 ONNX로 변환 후, NVIDIA TensorRT를 통해 Layer Fusion 및 최적화된 엔진을 빌드합니다. INT8 적용 시 발생할 수 있는 정확도 절벽(Performance Cliff)을 막기 위해 **학습 후 양자화(PTQ)**의 캘리브레이션 또는 **양자화 인지 학습(QAT)**을 전략적으로 선택해야 합니다.
* **도메인 무작위화 (Domain Randomization):**
    * **개념:** 시뮬레이터의 질감, 조명, 날씨(Visual Randomization) 및 마찰력, 질량(Dynamics Randomization)을 무작위로 변경하며 학습시켜, 모델이 특정 시뮬레이션 환경에 과적합되지 않고 현실 세계로 유연하게 전이(Transfer)될 수 있도록 하는 기법입니다.