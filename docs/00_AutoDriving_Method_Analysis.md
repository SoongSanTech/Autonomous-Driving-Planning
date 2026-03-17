# 📊 자율주행 아키텍처 비교 분석 명세서 (Vision vs LiDAR Fusion)

## 1. 테슬라 FSD 아키텍처 (Pure Vision End-to-End)

### 구조적 형태
* **Vision-In, Control-Out:** 8대의 카메라에서 수집된 2D 이미지를 공간-시간 변환기(Spatial-Temporal Transformer)를 통해 3D/4D BEV(Bird's Eye View)로 변환한 뒤, 직접 조향(Steering)과 가속(Throttle)을 출력하는 단일 거대 신경망.
* **의존성 배제:** 고화질 지도(HD Map)나 LiDAR/Radar 없이 오직 광학적 시각 정보와 신경망의 추론에만 의존.

### 장점 (Pros)
* **무한한 확장성(Scalability):** 센서 세트가 저렴하여 수백만 대의 양산차에서 데이터를 수집할 수 있으며, 데이터가 쌓일수록 모델 성능이 선형적으로 증가.
* **높은 의미론적 밀도:** 신호등 색상, 표지판 텍스트, 보행자의 시선 등 주행에 필요한 '문맥(Context)'을 파악하는 데 압도적으로 유리.
* **센서 충돌 제로:** 단일 모달리티(Vision)만 사용하므로 이기종 센서 간의 정보 충돌(예: 카메라는 뚫려있다고 판단, 레이더는 막혀있다고 판단)로 인한 유령 브레이크(Phantom Braking)가 발생하지 않음.

### 단점 (Cons)
* **심도(Depth) 추론의 불안정성:** 거리를 '측정'하는 것이 아니라 픽셀의 변화로 '추론'하므로, 착시 환경이나 한 번도 본 적 없는 기하학적 형태 앞에서는 치명적인 오판 가능성 존재.
* **조명/날씨 취약성:** 역광, 폭우, 심야 등 렌즈가 확보할 수 있는 광량이 제한된 상황에서 성능이 급격히 저하됨.
* **초기 학습 데이터의 천문학적 요구량:** 3D 공간 지각 능력을 2D 이미지로만 '깨닫게' 하려면 엑사바이트(EB) 단위의 방대한 데이터와 H100 클러스터 규모의 컴퓨팅 파워가 필요함.

---

## 2. LiDAR 융합 아키텍처 (Multi-modal Fusion)

### 구조적 이점 (Structural Advantages)
* **Ground Truth 기반의 절대적 3D 지각:** 빛의 반사 시간(ToF)을 통해 오차 없는 정밀한 거리와 형태 정보(Point Cloud)를 획득하여, 카메라의 심도 추론 실패를 완벽히 보완.
* **조명 불변성 (Illumination Invariance):** 자체적으로 레이저를 조사하는 능동형 센서이므로 야간이나 강렬한 역광에서도 주위 환경을 주간과 똑같이 100% 인식.
* **다중 레이어 안전성:** 비전 신경망이 판단을 내리기 전/후에 LiDAR 데이터를 기반으로 한 절대적인 '안전 경계(Safety Envelope/Collision Avoidance)'를 룰베이스로 덧씌울 수 있음.

### 비용 효율성 (Cost Efficiency)
* **하드웨어 비용 (Low):** 실제 차량 탑재 시 LiDAR 센서 자체의 가격은 카메라 대비 매우 높음 (상용화의 최대 걸림돌).
* **데이터 비용 (High):** 신경망이 3D 공간을 이해하기 위해 필요한 **'데이터 요구량'을 획기적으로 낮춰줌**. 적은 양의 데이터로도 충돌 회피 학습이 가능하므로, 숭산텍과 같은 연구 목적(Research-scale)에서는 오히려 시간과 데이터 수집 비용을 아껴주는 엄청난 이점을 제공함.

### 연산 효율성 (Computational Efficiency)
* **입력 처리 병목:** 포인트 클라우드(3D Voxel) 연산은 무거움. 카메라 특징 추출맵과 정합(Calibration)하는 과정에서 병목 발생 가능.
* **학습 수렴 속도 향상:** 거리를 추측하기 위해 신경망이 낭비하는 연산을 줄이고, LiDAR가 제공하는 확정된 거리 값을 입력받음으로써 모델의 학습 수렴(Convergence) 속도가 비전 전용 모델보다 훨씬 빠름.

---

# 🚀 [Sungsan Tech] 자율주행 R&D 및 아키텍처 기획서 (Master Document)

## 1. 프로젝트 목표 (Goal)
**"Software-Defined Sensor 및 Hybrid E2E 아키텍처 기반의 초저비용 Sim-to-Real 자율주행 구현"**

숭산텍은 물리적 센서의 비용 한계와 대규모 컴퓨팅 연산의 병목을 소프트웨어(AI)로 극복합니다. CARLA 시뮬레이터를 활용하여 완벽한 데이터 파이프라인을 구축하고, 어라운드 뷰(AVM)와 저가형 LiDAR를 융합한 **설명 가능한 Multi-Task 강화학습 모델**을 개발하여 실제 RC카(Edge Device) 상용화 배포 및 원천 기술(IP) 확보를 동시에 달성합니다.

---

## 2. 실현 과정 (Implementation Process)
시스템은 '데이터 획득 ➡️ 하이브리드 모델링 ➡️ 엣지 배포'의 파이프라인으로 실현됩니다.

* **A. 센서 데이터 전처리 (Sensor Fusion & Sync):**
    * **AVM 병합:** 차량 전/후/좌/우 4대의 초광각 카메라 영상을 OpenCV 호모그래피를 통해 단일 1채널 조감도(BEV, Bird's Eye View) 이미지로 병합 연산.
    * **동기화 로깅:** CARLA 동기 모드(Synchronous Mode)를 적용하여 AVM 이미지, 2D LiDAR 점군, 텔레메트리(제어값)를 정확히 10Hz(0.1초) 주기로 매핑 및 저장.
* **B. Multi-Task 하이브리드 모델링 (Shared Backbone, Dual Heads):**
    * **공유 백본 (ResNet18):** 1장의 AVM 이미지와 2D LiDAR 데이터를 입력받아 주변 환경의 3D/4D 시공간 특징 맵(Feature Map) 추출.
    * **Head 1 (Perception):** 추출된 특징을 바탕으로 객체(차량, 보행자)의 2D/3D Bounding Box를 출력하여 모니터 UI에 시각화 (XAI 디버깅 용도).
    * **Head 2 (Control):** 특징 맵과 인식된 객체 궤적을 바탕으로 Steering(-1~1) 및 Throttle(0~1) 출력.
* **C. 2단계 강화학습 (BC to RL):**
    * **Step 1 (Warm-start):** 시뮬레이터 오토파일럿 주행 데이터(CSV)로 행동 복제(Behavioral Cloning) 지도학습 선행.
    * **Step 2 (Fine-tuning):** PPO/SAC 기반 강화학습 에이전트로 전환하여, CARLA Gym 환경에서 충돌 및 차선 이탈에 대한 페널티를 주며 복원력(Recovery) 극대화.
* **D. 제품화 (Productization):**
    * 학습된 PyTorch 모델을 TensorRT (FP16/INT8)로 양자화(Quantization) 및 ROS2 미들웨어 기반 노드 분리.

---

## 3. 산업 및 학계의 접근 (Industry & Academia Approach)
숭산텍의 아키텍처는 글로벌 SOTA(State-of-the-Art) 트렌드를 정확히 조준하고 있습니다.

* **산업계 (Tesla, Wayve, Waabi):**
    * **Tesla FSD (Occupancy Network):** 모듈형 코드를 폐기하고 순수 비전(Pure Vision) 기반의 E2E 네트워크 상에서 주행 제어와 UI 시각화(객체 인식)를 동시에 처리하는 파운데이션 모델 상용화.
    * **Wayve (AV2.0) / Waabi:** 방대한 실제 도로 데이터 대신, '월드 모델(World Models)'과 시뮬레이터 내의 심층 강화학습(Deep RL)을 통해 자율주행 에이전트의 강건성을 확보하는 방식 주도.
* **학계 (UniAD, TCP, SR Networks):**
    * **UniAD (CVPR 2023 Best Paper):** 인지, 예측, 제어를 하나의 E2E Transformer 신경망으로 통합한 통합 자율주행 모델 증명.
    * **Point Cloud Super-Resolution:** 값싼 2D/희소 점군 데이터를 고해상도 3D 점군으로 업스케일링하여 하드웨어 비용을 낮추는 소프트웨어 센서 연구 활발.

---

## 4. 숭산텍의 압도적 차별성 (Sungsan Tech's Differentiation)
글로벌 빅테크가 자본(H100 클러스터)으로 밀어붙이는 한계를, 숭산텍은 '설계의 우수성'으로 우회합니다.

1. **Software-Defined Sensor (비용 파괴):** 15만 원대 2D LiDAR 데이터를 카메라 이미지와 융합하여 125만 원 이상의 64채널 3D LiDAR 해상도로 복원하는 Super-Resolution 모델을 구축. 고가 센서 없이도 완벽한 3D 공간 지각 달성.
2. **AVM 전처리 융합 (연산 파괴):** 8대의 카메라 텐서를 각각 AI 모델에 밀어 넣는 테슬라 방식 대신, 전처리 단계에서 4대의 영상을 1장의 BEV로 스티칭함. 엣지 디바이스(RC카 Jetson)에서 병목 현상 없이 30Hz 실시간 추론 가능.
3. **설명 가능한 E2E (블랙박스 극복):** 단순히 핸들만 돌리는 모델이 아니라, Perception Head가 동시 출력하는 '객체 인식 UI 시각화'를 통해 AI가 왜 제동을 걸었는지(또는 실패했는지) 원인을 명확히 추적할 수 있는 하이브리드 구조.

---

## 5. 마일스톤 내역 (Milestone Details)
제한된 인력(2명)과 리소스(4090 1대, 3090Ti 1대)를 극대화하기 위한 투-트랙(Two-Track) 마일스톤입니다.

### 📌 Phase 1: 360도 동기화 데이터 파이프라인 (M1)
* **목표:** CARLA 동기 모드(Sync Mode) 기반의 결함 없는 데이터 수집기 완성.
* **산출물:** - 4채널 AVM 카메라 스티칭 스크립트.
  - 2D 가상 라이다 + 64채널 정답지 라이다 + 텔레메트리 + 2D Bounding Box(정답지)가 10Hz로 동시 로깅되는 `advanced_data_collector.py`.

### 📌 Phase 2: Track 1 RC카 실증 및 배포 (M2)
* **목표:** AVM 기반 경량 E2E 모델(BC) 학습 및 Sim-to-Real 1회전 관통.
* **산출물:**
  - ResNet18 기반 AVM E2E 주행 모델 (CARLA 내 교차로 통과 성공 기준).
  - TensorRT 양자화 모델 및 ROS2 연동을 통한 RC카 주행 테스트.

### 📌 Phase 3: Track 2 멀티 태스크 & 강화학습 고도화 (M3)
* **목표:** 논문/특허 산출을 위한 LiDAR SR 및 PPO 하이브리드 모델 완성.
* **산출물:**
  - CARLA Gym Environment (상태, 행동, 보상 함수) 래퍼 클래스 구축.
  - 저가형 LiDAR Super-Resolution 모델 및 PPO 강화학습 에이전트를 결합한 최종 Foundation Model.