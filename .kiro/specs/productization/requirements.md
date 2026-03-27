# Requirements Document

## Introduction

Productization 기능은 실험 단계의 Python 스크립트를 프로덕션 수준의 배포 가능한 자율주행 소프트웨어 스택으로 변환한다. ROS2 아키텍처로 노드를 분리하고, TensorRT 양자화(FP16 우선, INT8 선택)로 엣지 디바이스 최적화를 수행하며, RC카(Jetson) 실차 배포를 준비한다. 멀티카메라 구성(Front RGB 1대 + AVM 4대 = 5대)을 지원하며, PWM 서보 제어로 RC카를 구동한다.

## Glossary

- **Productization_System**: 실험 코드를 프로덕션 배포 가능한 소프트웨어로 변환하는 전체 파이프라인
- **ROS2_Framework**: 노드 간 publish-subscribe 통신을 제공하는 Robot Operating System 2 미들웨어
- **Front_Cam_Node**: 전방 RGB 카메라 데이터를 `/sensor/front` 토픽으로 발행하는 ROS2 노드
- **AVM_Cam_Node**: AVM 4대 카메라 데이터를 `/sensor/avm/{front,rear,left,right}` 토픽으로 발행하는 ROS2 노드
- **BEV_Stitch_Node**: AVM 4대 이미지를 호모그래피 스티칭하여 `/sensor/bev` 토픽으로 발행하는 ROS2 노드 (Phase 3+ 활성화)
- **Inference_Node**: `/sensor/front` 이미지를 수신하여 TensorRT 추론 후 `/inference/cmd` 토픽으로 제어 명령을 발행하는 노드
- **Control_Node**: `/inference/cmd`를 수신하여 차량 제어 명령을 발행하는 노드
- **Model_Quantizer**: FP32 PyTorch 모델을 FP16 또는 INT8 TensorRT 엔진으로 변환하는 컴포넌트
- **Edge_Device**: NVIDIA Jetson Xavier NX / Orin Nano 임베디드 컴퓨팅 플랫폼
- **PWM_Servo_Controller**: ROS2 제어 명령을 PWM 신호로 변환하여 RC카 서보/ESC를 구동하는 컴포넌트
- **Control_Frequency**: 제어 명령 생성 주기 (Hz)
- **TensorRT_Engine**: TensorRT로 최적화된 추론 엔진 파일 (.trt)

## Requirements

### Requirement 1: ROS2 노드 분리 (멀티카메라)

**User Story:** 로보틱스 엔지니어로서, 모놀리식 스크립트를 멀티카메라 지원 ROS2 노드로 분리하여 모듈화된 자율주행 시스템을 배포하고 싶다.

#### Acceptance Criteria

1. THE Productization_System SHALL decompose into: Front_Cam_Node, AVM_Cam_Node, BEV_Stitch_Node, Inference_Node, Control_Node
2. THE Front_Cam_Node SHALL publish 800×600 RGB images to `/sensor/front` at 30Hz
3. THE AVM_Cam_Node SHALL publish 400×300 images to `/sensor/avm/front`, `/sensor/avm/rear`, `/sensor/avm/left`, `/sensor/avm/right` at 10Hz
4. THE BEV_Stitch_Node SHALL subscribe to `/sensor/avm/*` and publish stitched BEV to `/sensor/bev` at 10Hz (Phase 3+ 활성화)
5. THE Inference_Node SHALL subscribe to `/sensor/front` and publish to `/inference/cmd` (steering + throttle)
6. THE Control_Node SHALL subscribe to `/inference/cmd` and publish to `/control/vehicle_command`
7. WHEN any node publishes a message, THE ROS2_Framework SHALL deliver within 10ms

### Requirement 2: 노드 간 통신

**User Story:** 시스템 아키텍트로서, 노드 간 ROS2 토픽 통신으로 느슨한 결합과 확장성을 유지하고 싶다.

#### Acceptance Criteria

1. ALL nodes SHALL use ROS2 Publisher/Subscriber pattern
2. WHEN a node fails, THE ROS2_Framework SHALL continue delivering messages to remaining healthy nodes
3. THE Productization_System SHALL support adding new subscriber nodes without modifying existing publishers
4. THE Phase 2 active path SHALL be: Front_Cam_Node → Inference_Node → Control_Node
5. THE BEV_Stitch_Node SHALL be disabled by default and activated for Phase 3+ experiments

### Requirement 3: 모델 양자화 (FP16 우선, INT8 선택)

**User Story:** 배포 엔지니어로서, FP32 PyTorch 모델을 FP16/INT8로 양자화하여 엣지 디바이스에서 실시간 추론을 달성하고 싶다.

#### Acceptance Criteria

1. WHEN provided a PyTorch_Checkpoint, THE Model_Quantizer SHALL first convert to TensorRT FP16 engine
2. THE Model_Quantizer SHALL validate FP16 engine accuracy (PyTorch 대비 오차 < 5%)
3. THE Model_Quantizer SHALL use ONNX as intermediate format (opset 13, input shape: 1×3×224×224)
4. WHEN FP16 passes validation, THE Model_Quantizer SHALL optionally attempt INT8 with 500+ calibration samples
5. WHERE INT8 accuracy degradation exceeds 5%, THE system SHALL use FP16 engine
6. THE TensorRT_Engine SHALL be compatible with NVIDIA Jetson devices

### Requirement 4: 엣지 디바이스 추론

**User Story:** 차량 시스템 엔지니어로서, 양자화된 모델을 Jetson에서 실행하여 실시간 추론을 수행하고 싶다.

#### Acceptance Criteria

1. THE Inference_Node SHALL load TensorRT_Engine on Edge_Device at startup
2. WHEN receiving a `/sensor/front` image, THE Inference_Node SHALL complete inference within 10ms on Edge_Device
3. THE Inference_Node SHALL utilize Jetson GPU for TensorRT inference
4. IF TensorRT_Engine loading fails, THEN terminate with exit code 1
5. THE Inference_Node SHALL monitor GPU memory and warn if usage exceeds 80%

### Requirement 5: 제어 주파수 달성

**User Story:** 제어 시스템 엔지니어로서, 30Hz 제어 주파수를 달성하여 차량이 부드럽게 반응하도록 하고 싶다.

#### Acceptance Criteria

1. THE Productization_System SHALL maintain Control_Frequency ≥ 30Hz
2. THE end-to-end latency (sensor publish → control publish) SHALL be < 33ms
3. THE ROS2_Framework communication delay SHALL not exceed 5ms between any two nodes
4. THE system SHALL log warning if Control_Frequency drops below 30Hz for > 100ms
5. THE Inference_Node SHALL maintain accuracy within 2% of FP32 model at 30Hz

### Requirement 6: PWM 서보 제어 (RC카)

**User Story:** 하드웨어 통합 엔지니어로서, ROS2 제어 명령을 PWM 신호로 변환하여 RC카 서보/ESC를 구동하고 싶다.

#### Acceptance Criteria

1. THE PWM_Servo_Controller SHALL subscribe to `/control/vehicle_command`
2. THE PWM_Servo_Controller SHALL convert steering value [-1.0, 1.0] to servo PWM pulse width (1000~2000μs)
3. THE PWM_Servo_Controller SHALL convert throttle value [0.0, 1.0] to ESC PWM pulse width (1000~2000μs)
4. THE PWM_Servo_Controller SHALL output PWM at the same rate as received commands
5. IF PWM output fails, THE PWM_Servo_Controller SHALL log error and continue

### Requirement 7: Sim-to-Real 검증 준비

**User Story:** 검증 엔지니어로서, 시뮬레이션과 실차 모드를 전환하여 sim-to-real 전이를 검증하고 싶다.

#### Acceptance Criteria

1. THE Productization_System SHALL support "simulation" and "hardware" modes without code modification
2. THE configuration parameter SHALL select between modes
3. WHEN in "hardware" mode, THE PWM_Servo_Controller SHALL be activated
4. WHEN in "simulation" mode, THE CARLA actuator SHALL be used
5. THE system SHALL log operating mode at startup

### Requirement 8: 모델 형식 변환

**User Story:** ML 엔지니어로서, PyTorch 체크포인트를 TensorRT 형식으로 변환하여 엣지 디바이스에 배포하고 싶다.

#### Acceptance Criteria

1. THE Model_Quantizer SHALL load PyTorch model from .pth checkpoint
2. THE Model_Quantizer SHALL export to ONNX with input shape (1, 3, 224, 224)
3. THE Model_Quantizer SHALL build TensorRT engine with specified precision (FP16 or INT8)
4. THE Model_Quantizer SHALL validate output within 5% error vs PyTorch baseline
5. IF conversion fails, THE Model_Quantizer SHALL log error and terminate with exit code 2

### Requirement 9: ROS2 패키지 구조

**User Story:** 소프트웨어 개발자로서, 각 노드를 표준 ROS2 패키지로 구성하여 colcon으로 빌드/배포하고 싶다.

#### Acceptance Criteria

1. THE system SHALL create ROS2 packages: sensor_node (Front + AVM), inference_node, control_node, servo_node
2. EACH package SHALL have package.xml and CMakeLists.txt
3. WHEN executing "colcon build", ALL packages SHALL build without errors
4. THE packages SHALL install to standard ROS2 directories

### Requirement 10: 성능 모니터링 및 로깅

**User Story:** 시스템 운영자로서, 운영 중 성능 메트릭을 로깅하여 시스템 상태를 진단하고 싶다.

#### Acceptance Criteria

1. THE Front_Cam_Node SHALL log timestamp and frame number for each published message
2. THE Inference_Node SHALL log inference latency for each frame
3. THE Control_Node SHALL log Control_Frequency every 1 second
4. WHEN Control_Frequency < 30Hz, THE system SHALL log warning with measured value
5. ALL logs SHALL write to console and rotating log file (max 100MB)
