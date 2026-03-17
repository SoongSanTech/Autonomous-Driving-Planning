# 숭산텍(Sungsan Tech) Autonomous Driving Master Spec

## 1. Project Overview
본 프로젝트는 CARLA 시뮬레이터를 활용하여 현실 세계(Real-world)의 데이터 희소성 문제를 해결하고, Sim-to-Real 전이가 가능한 상용 수준의 자율주행 파이프라인을 구축하는 것을 목표로 한다. 

## 2. Core Architecture (3 Pillars)
본 시스템은 상호 의존적인 3개의 핵심 모듈로 구성되며, 단일 루프(Data -> Experiment -> Productization -> Data)로 동작한다.

* **Pillar 1: Data Pipeline (데이터)**
    * 가상 환경에서 멀티모달(Image, Depth, LiDAR) 센서 데이터와 텔레메트리(Telemetry) 데이터를 완벽히 동기화하여 수집 및 정제한다.
* **Pillar 2: Experiment & Modeling (실험 및 모델링)**
    * 수집된 데이터를 바탕으로 End-to-End 행동 복제(Behavioral Cloning) 및 강화학습(RL) 기반의 제어 모델을 학습하고 검증한다.
* **Pillar 3: Productization (제품화)**
    * 학습된 모델을 ROS2 미들웨어 환경으로 모듈화 및 경량화(TensorRT)하여 실제 차량(Edge Device) 탑재를 준비한다.

## 3. System Requirements
* **OS:** Windows Host (CARLA Server 0.9.15) + WSL2 Linux (Python Client)
* **Compute:** NVIDIA RTX 3090 Ti / 4090 Workstations
* **Network:** Host-WSL TCP Bridge (Port 2000)