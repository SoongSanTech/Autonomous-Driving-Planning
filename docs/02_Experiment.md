# Spec: Experiment & ML Modeling

## 1. Objective
Data Pipeline에서 확보된 정형/비정형 데이터를 활용하여 차량을 제어할 수 있는 지능(Intelligence)을 PyTorch 기반으로 구현한다.

## 2. Key Requirements
* **End-to-End 행동 복제 (Behavioral Cloning):**
    * 입력: (800x600x3) RGB 이미지 1장
    * 출력: 예측된 Steering (-1.0 ~ 1.0) 및 Throttle (0.0 ~ 1.0) 값
    * 네트워크 아키텍처: ResNet 기반의 경량화된 CNN 추출기 + FCN 층 적용.
* **심층 강화학습 환경 구축 (Deep RL):**
    * CARLA 환경을 에이전트 학습용으로 추상화하기 위해 OpenAI Gym(또는 Gymnasium) 인터페이스로 래핑(Wrapping)한다.
    * Reward Function 설계: 차선 중앙 유지 시 (+), 충돌 시 (-), 과도한 조향 시 (-) 보상 부여.
* **Sim-to-Real 도메인 적응 (Domain Adaptation):**
    * 시뮬레이션 그래픽의 질감을 실제 블랙박스 데이터 질감으로 변환하는 CycleGAN 등의 스타일 트랜스퍼 실험 환경 구성.

## 3. Acceptance Criteria
* 학습된 모델(pth)을 CARLA 클라이언트에 로드했을 때, 룰베이스 오토파일럿 개입 없이 단일 교차로를 성공적으로 통과해야 함.