# Implementation Plan: Productization

## Overview

실험 코드를 프로덕션 ROS2 시스템으로 변환. 멀티카메라(5대) 지원, TensorRT FP16 양자화, PWM 서보 RC카 제어. Phase 2에서는 Front Cam → Inference → Control 경로만 활성화하고, BEV Stitch는 Phase 3+에서 활성화.

구현 순서: ROS2 인프라 → 센서 노드 (Front + AVM) → 양자화 파이프라인 → 추론 노드 → 제어 노드 → PWM 서보 → 통합 테스트

## Tasks

- [ ] 1. ROS2 워크스페이스 및 메시지 정의
  - `src/deploy/` 디렉토리 구조 생성
  - CameraImage.msg, SteeringCommand.msg, VehicleCommand.msg 정의
  - 메시지 패키지 빌드
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 2. Front Cam Node 구현
  - [ ] 2.1 FrontCamNode 클래스 구현
    - `/sensor/front` 토픽으로 800×600 RGB 발행 (30Hz)
    - simulation mode: CARLA 카메라 연결 + 재시도 (3회, 2초)
    - hardware mode: USB/CSI 카메라 연결
    - 타임스탬프 + 프레임 번호 로깅
    - _Requirements: 1.2, 7.1, 7.4, 10.1_

  - [ ] 2.2 AVM_Cam_Node 구현
    - `/sensor/avm/{front,rear,left,right}` 토픽으로 400×300 발행 (10Hz)
    - 4대 카메라 동시 관리
    - _Requirements: 1.3_

  - [ ] 2.3 BEV_Stitch_Node 스텁 구현
    - `/sensor/avm/*` 구독 → 호모그래피 스티칭 → `/sensor/bev` 발행
    - 기본 비활성 (config로 활성화)
    - _Requirements: 1.4, 2.5_

  - [ ]* 2.4 센서 노드 unit tests
    - 토픽 이름 검증, 모드 전환, 로깅 테스트
    - _Requirements: 1.2, 1.3, 7.1, 10.1_

- [ ] 3. 모델 양자화 파이프라인 구현
  - [ ] 3.1 ModelQuantizer 클래스 구현
    - PyTorch .pth 로드 + 검증
    - _Requirements: 3.1, 8.1_

  - [ ] 3.2 ONNX export 구현
    - input shape (1, 3, 224, 224), opset 13
    - _Requirements: 3.3, 8.2_

  - [ ] 3.3 TensorRT FP16 엔진 빌드
    - FP16 우선 빌드
    - _Requirements: 3.1, 8.3_

  - [ ] 3.4 캘리브레이션 데이터 생성 (INT8 선택)
    - 500+ 대표 이미지 생성
    - _Requirements: 3.4_

  - [ ] 3.5 엔진 검증 + 정밀도 fallback
    - PyTorch 대비 오차 < 5% 검증
    - INT8 오차 > 5% → FP16 유지
    - _Requirements: 3.2, 3.5, 8.4_

  - [ ] 3.6 전체 변환 파이프라인 통합
    - load → ONNX → TRT → validate → save
    - _Requirements: 3.1~3.6, 8.1~8.5_

  - [ ]* 3.7 양자화 property tests
    - **Property 2: Pipeline completeness**
    - **Property 3: Precision fallback**
    - **Property 4: Engine validation**

- [ ] 4. Checkpoint — 양자화 파이프라인 검증
  - 샘플 체크포인트로 변환 테스트
  - 모든 테스트 통과 확인

- [ ] 5. Inference Node 구현
  - [ ] 5.1 TensorRT 런타임 래퍼
    - 엔진 로드, 실행 컨텍스트, GPU 메모리 할당
    - 로드 실패 시 exit code 1
    - _Requirements: 4.1, 4.4_

  - [ ] 5.2 이미지 전처리
    - 800×600 → 224×224 리사이즈 + 정규화
    - (1, 3, 224, 224) 텐서 변환
    - _Requirements: 4.2_

  - [ ] 5.3 InferenceNode 클래스 구현
    - `/sensor/front` 구독 → TRT 추론 → `/inference/cmd` 발행
    - 추론 지연 측정 + 로깅 (> 10ms 경고)
    - GPU 메모리 모니터링 (> 80% 경고)
    - _Requirements: 1.5, 4.1, 4.2, 4.3, 4.5, 10.2_

  - [ ]* 5.4 추론 노드 property tests
    - **Property 6: Real-time inference latency (<10ms)**
    - **Property 10: Multi-camera topic consistency**

- [ ] 6. Control Node 구현
  - [ ] 6.1 ControlNode 클래스 구현
    - `/inference/cmd` 구독 → `/control/vehicle_command` 발행
    - 30Hz 주파수 모니터링 + 1초마다 로깅
    - < 30Hz 100ms 이상 시 경고
    - _Requirements: 1.6, 5.1, 5.4, 10.3, 10.4_

  - [ ]* 6.2 제어 노드 property tests
    - **Property 7: Control frequency ≥ 30Hz**
    - **Property 8: E2E latency < 33ms**

- [ ] 7. PWM Servo Controller 구현
  - [ ] 7.1 PWMServoController 클래스 구현
    - `/control/vehicle_command` 구독
    - steering [-1, 1] → PWM 1000~2000μs
    - throttle [0, 1] → ESC PWM 1000~2000μs
    - 출력 실패 시 에러 로그 + 계속
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 7.2 PWM property tests
    - **Property 9: PWM conversion accuracy**

- [ ] 8. Checkpoint — 노드 통합 검증
  - 모든 노드 개별 테스트 통과
  - ROS2 토픽 통신 검증

- [ ] 9. 듀얼 모드 지원 구현
  - [ ] 9.1 Mode Manager 구현
    - config.yaml에서 mode 파라미터 읽기
    - simulation: CARLA 카메라 + CARLA actuator
    - hardware: 실제 카메라 + PWM servo
    - 시작 시 모드 로깅
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 9.2 모드 전환 integration tests
    - simulation/hardware 모드 전환 테스트

- [ ] 10. 성능 모니터링 및 로깅 구현
  - [ ] 10.1 PerformanceMonitor 구현
    - 지연, 주파수, GPU 메모리 추적
    - 임계값 초과 시 경고
    - _Requirements: 5.1, 5.2, 5.3, 10.4_

  - [ ] 10.2 ProductionLogger 구현
    - 콘솔 + 로테이팅 파일 (100MB)
    - 구조화 로그 포맷
    - _Requirements: 10.5_

- [ ] 11. Launch 파일 및 설정
  - simulation.launch.py: CARLA 모드 전체 노드 실행
  - hardware.launch.py: 실차 모드 전체 노드 실행
  - config/simulation.yaml, config/hardware.yaml
  - _Requirements: 7.2, 9.1~9.4_

- [ ] 12. Final Checkpoint — 통합 테스트
  - E2E 파이프라인 테스트 (sensor → inference → control)
  - 모든 테스트 통과 확인

## Notes

- `*` 표시 태스크는 선택적 property-based / unit test
- Phase 2 active path: Front Cam → Inference → Control (BEV Stitch는 Phase 3+)
- 모델 입력: (1, 3, 224, 224) — ResNet18 기준
- 양자화: FP16 우선, INT8은 선택적 (캘리브레이션 500장)
- RC카 제어: CAN 대신 PWM 서보/ESC
- 추론 지연 목표: < 10ms (Jetson)
- 제어 주파수: ≥ 30Hz
- 토픽: /sensor/front, /sensor/avm/*, /sensor/bev, /inference/cmd, /control/vehicle_command
