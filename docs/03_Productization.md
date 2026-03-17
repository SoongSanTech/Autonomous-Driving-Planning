# Spec: Productization & Sim-to-Real Deployment

## 1. Objective
시뮬레이터에 종속된 Python 스크립트를 독립적이고 배포 가능한 형태의 상용 자율주행 소프트웨어 스택으로 전환한다.

## 2. Key Requirements
* **ROS2 아키텍처 전환:**
    * 단일 스크립트를 분해하여 `sensor_node`, `inference_node`, `control_node` 등 ROS2 패키지로 재구성한다.
    * 각 노드는 Publisher/Subscriber 패턴을 통해 토픽을 교환해야 한다.
* **Edge 모델 경량화 (Quantization):**
    * 4090 데스크탑에서 훈련된 FP32 PyTorch 모델을 실제 차량용 엣지 디바이스(NVIDIA Jetson 등)에서 실시간 추론이 가능하도록 TensorRT를 이용해 INT8/FP16으로 양자화한다.
* **실물 하드웨어 연동 준비 (Hardware-in-the-Loop):**
    * Sim-to-Real 검증을 위해, CARLA의 조향 명령을 실제 조향 모터의 CAN 통신 신호로 변환하는 브릿지 코드를 명세화한다.

## 3. Acceptance Criteria
* ROS2 프레임워크 상에서 시뮬레이터와 AI 추론 모델이 통신 병목 없이 30Hz 이상의 제어 주기를 달성해야 함.