# Spec: Data Pipeline & Generation

## 1. Objective
단순 시뮬레이터 구동을 넘어, 머신러닝 모델이 학습 가능한 '완벽하게 동기화된(Synchronized)' 고품질 데이터셋을 대량으로 확보한다.

## 2. Key Requirements
* **멀티모달 동기화 (Multi-modal Synchronization):**
    * 전방 RGB 카메라 이미지(800x600)와 차량의 상태 데이터(Speed, Steering, Throttle, Brake)를 정확히 동일한 타임스탬프(ms) 기준으로 짝지어(Pairing) 저장해야 한다.
    * 수집 주기: 최소 10Hz (초당 10프레임).
* **Edge-case 시나리오 자동 생성:**
    * 단순 직진 주행뿐만 아니라 CARLA Python API를 활용하여 날씨(비, 안개)와 시간대(야간, 역광)를 무작위로 변경하는 스크립트를 구현한다.
* **데이터 포맷 표준화:**
    * 이미지: `_out/images/{timestamp}.png`
    * 라벨링: `_out/labels/driving_log.csv` (각 로우에 이미지 파일명과 센서값 매핑)

## 3. Acceptance Criteria
* WSL 환경에서 헤드리스(Headless)로 스크립트 실행 시, 프레임 드랍 없이 1시간 이상의 주행 데이터가 디스크에 정상 로깅되어야 함.