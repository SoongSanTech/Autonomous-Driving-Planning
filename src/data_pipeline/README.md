# CARLA 자율주행 데이터 수집 파이프라인

CARLA 0.9.15 시뮬레이터에서 RGB 카메라 이미지와 차량 상태 텔레메트리를 동기 수집하여, End-to-End 자율주행 모델 학습용 데이터셋을 생성합니다.

---

## 1. 데이터 수집 구성

### 센서 구성

| 항목 | 값 | 비고 |
|------|-----|------|
| 카메라 | RGB (`sensor.camera.rgb`) | 전방 장착 |
| 해상도 | 800 × 600 px | BGRA → BGR 변환 |
| 장착 위치 | x=1.5m, z=2.4m (차량 기준) | 전방 루프탑 시점 |
| 이미지 포맷 | PNG (압축 레벨 3) | 0=빠름/큼, 9=느림/작음 |

### 수집 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| 틱 레이트 | 10 Hz | 0.1초 간격 동기 틱 (`fixed_delta_seconds`) |
| 수집 시간 | 3,600초 (1시간) | `--duration`으로 변경 가능 |
| 예상 프레임 수 | 36,000 프레임/시간 | 10Hz × 3,600초 |
| 에피소드 주기 | 300초 (5분) | 에피소드마다 날씨/시간대 랜덤 변경 |
| 에피소드 수 | 12회/시간 | 3,600초 ÷ 300초 |

### 에피소드 시나리오 다양성

수집 중 5분마다 자동으로 날씨와 시간대가 랜덤 변경되어 다양한 조건의 데이터를 확보합니다.

| 날씨 프리셋 | CARLA 매핑 |
|-------------|-----------|
| CLEAR (맑음) | `ClearNoon` |
| RAIN (비) | `WetCloudyNoon` |
| FOG (안개) | `SoftRainSunset` |

| 시간대 프리셋 | 태양 고도각 |
|--------------|-----------|
| DAYTIME (낮) | 0° |
| BACKLIGHT (역광) | 90° |
| NIGHT (밤) | 180° |

### 차량 상태 텔레메트리

각 프레임마다 이미지와 함께 아래 4개 값이 동기 기록됩니다.

| 필드 | 타입 | 범위 | 단위 |
|------|------|------|------|
| `speed` | float | 0 이상 | m/s |
| `steering` | float | [-1.0, 1.0] | 정규화 조향각 |
| `throttle` | float | [0.0, 1.0] | 가속 입력 |
| `brake` | float | [0.0, 1.0] | 제동 입력 |

### 비동기 I/O 설정

| 항목 | 값 |
|------|-----|
| 큐 크기 | 1,000 프레임 (10Hz 기준 ~100초 버퍼) |
| 워커 스레드 | 2개 (`ThreadPoolExecutor`) |
| 큐 경고 임계값 | 90% (900 프레임) |
| 큐 오버플로 시 | 프레임 드롭 + 경고 로그 |

---

## 2. 데이터 수집 프로시저

### 사전 준비

| 항목 | 상세 |
|------|------|
| OS | Windows 10/11 (CARLA 서버) + WSL2 Linux (Python 클라이언트) |
| CARLA | 0.9.15 (Windows에서 `CarlaUE4.exe` 실행) |
| Python | 3.8+ (WSL2 내부, venv 사용) |
| GPU | NVIDIA RTX 3090 Ti / 4090 권장 |

### Step 1. 환경 설정

```bash
# WSL2 터미널에서 실행
cd /mnt/d/00_mrlee/workdir/soongsantech_carla
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2. CARLA 서버 실행 (Windows)

Windows에서 CARLA 시뮬레이터를 실행합니다.

```powershell
# Windows PowerShell
cd C:\CARLA_0.9.15
.\CarlaUE4.exe
```

서버가 완전히 로드될 때까지 대기합니다 (보통 30초~1분).

### Step 3. WSL2 호스트 IP 확인

WSL2는 `localhost`로 Windows 호스트에 접근할 수 없으므로 IP를 확인합니다.

```powershell
# Windows PowerShell
ipconfig
# "vEthernet (WSL)" 어댑터의 IPv4 주소 확인 (예: 172.28.224.1)
```

### Step 4. 데이터 수집 실행

`src/data_pipeline/cli.py`가 실제 데이터 수집 진입점입니다.
(`tests/` 폴더의 스크립트는 CARLA 없이 mock으로 동작하는 검증용이므로 실제 수집에 사용하지 않습니다.)

```bash
# WSL2 터미널 (venv 활성화 상태)
source venv/bin/activate

# 기본 실행 — 1시간 수집 (36,000 프레임)
PYTHONPATH=src python -m data_pipeline.cli --host 172.28.224.1

# 10분 테스트 수집 (6,000 프레임)
PYTHONPATH=src python -m data_pipeline.cli \
    --host 172.28.224.1 \
    --duration 600

# 전체 옵션 지정
PYTHONPATH=src python -m data_pipeline.cli \
    --host 172.28.224.1 \
    --port 2000 \
    --output-dir src/data \
    --duration 3600 \
    --headless
```

### CLI 옵션

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--host` | `localhost` | CARLA 서버 IP (WSL2에서는 Windows IP 필수) |
| `--port` | `2000` | CARLA 서버 TCP 포트 |
| `--output-dir` | `src/data` | 데이터 출력 베이스 디렉토리 |
| `--duration` | `3600` | 수집 시간 (초) |
| `--headless` | `True` | 디스플레이 없이 실행 |
| `--no-headless` | — | CARLA 뷰어 디스플레이 활성화 |

### Step 5. 수집 중 모니터링

수집이 시작되면 100프레임(10초)마다 진행 상황이 로그에 출력됩니다.

```
2026-03-13 14:30:22 [INFO] data_pipeline.pipeline: Connected to CARLA at 172.28.224.1:2000
2026-03-13 14:30:22 [INFO] data_pipeline.pipeline: Output directory resolved to: src/data/2026-03-13_143022
2026-03-13 14:30:22 [INFO] data_pipeline.episode_manager: New episode: weather=ClearNoon, time=DAYTIME
2026-03-13 14:30:32 [INFO] data_pipeline.pipeline: Progress: 100 frames captured, 10.0s elapsed
2026-03-13 14:30:42 [INFO] data_pipeline.pipeline: Progress: 200 frames captured, 20.0s elapsed
...
2026-03-13 14:35:22 [INFO] data_pipeline.episode_manager: New episode: weather=WetCloudyNoon, time=NIGHT
```

### Step 6. 수집 종료

- 지정한 `--duration` 시간이 경과하면 자동 종료
- 수동 종료: `Ctrl+C` (SIGINT) → 큐에 남은 프레임을 디스크에 플러시 후 안전 종료
- CARLA 서버 크래시 시 → 자동 감지, 수집된 데이터 보존 후 종료

```
2026-03-13 15:30:22 [INFO] data_pipeline.pipeline: Collection stopped: 36000 frames, 0 drops, 3600.0s
2026-03-13 15:30:23 [INFO] data_pipeline.pipeline: Collection complete: 36000 frames saved, 0 drops
```

---

## 3. 데이터 수집 결과

### 출력 경로

수집 데이터는 `src/data/` 하위에 수집 시작 시각 기반 폴더로 저장됩니다.

```
src/data/
├── 2026-03-13_143022/              ← 1차 수집 (3/13 14:30)
│   ├── images/
│   │   ├── 100.png                 ← 800×600 RGB PNG
│   │   ├── 200.png
│   │   ├── 300.png
│   │   └── ... (36,000개)
│   └── labels/
│       └── driving_log.csv         ← 프레임별 차량 상태
├── 2026-03-13_160510/              ← 2차 수집 (3/13 16:05)
│   ├── images/
│   └── labels/
└── 2026-03-14_091500/              ← 3차 수집 (3/14 09:15)
    ├── images/
    └── labels/
```

### driving_log.csv 스키마

```csv
image_filename,speed,steering,throttle,brake
100.png,12.5,0.03,0.5,0.0
200.png,13.1,-0.12,0.6,0.0
300.png,11.8,0.25,0.4,0.1
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `image_filename` | string | `{timestamp_ms}.png` 형식 |
| `speed` | float | 차량 속도 (m/s) |
| `steering` | float | 조향각 [-1.0, 1.0] |
| `throttle` | float | 가속 입력 [0.0, 1.0] |
| `brake` | float | 제동 입력 [0.0, 1.0] |

### 수집량 예시

| 수집 시간 | 프레임 수 | 이미지 용량 (추정) | CSV 행 수 |
|-----------|-----------|-------------------|-----------|
| 10분 | 6,000 | ~8.5 GB | 6,001 (헤더 포함) |
| 1시간 | 36,000 | ~51 GB | 36,001 |
| 6시간 | 216,000 | ~306 GB | 216,001 |

> 용량 추정: 800×600 PNG (압축 레벨 3) ≈ 1.4 MB/프레임 기준

---

## 4. 테스트 (개발 검증용)

`tests/` 폴더의 테스트는 CARLA 서버 없이 mock으로 파이프라인 동작을 검증합니다.
실제 데이터 수집과는 무관합니다.

```bash
source venv/bin/activate

# 전체 테스트
pytest tests/data_pipeline/ -v

# 단위 테스트만
pytest tests/data_pipeline/ -v -m unit

# 통합 테스트만
pytest tests/data_pipeline/ -v -m integration
```

---

## 5. 장애 대응

### CARLA 연결 실패

파이프라인은 지수 백오프로 최대 5회 재시도합니다 (1초 → 2초 → 4초 → 8초 → 16초).

1. CARLA 서버 실행 확인: `CarlaUE4.exe` 프로세스 확인
2. 포트 확인: `Test-NetConnection -ComputerName 172.28.224.1 -Port 2000` (PowerShell)
3. Windows 방화벽: TCP 2000 인바운드 허용

### 프레임 드롭 (큐 오버플로)

`Queue overflow` 경고 발생 시 디스크 쓰기가 10Hz를 따라가지 못하는 상태입니다.

- NVMe SSD 사용 권장
- 다른 디스크 I/O 부하 최소화
- PNG 압축 레벨 조정 (코드 내 `png_compression` 파라미터)

### CARLA 서버 크래시

서버 크래시 시 파이프라인이 자동으로:
1. 큐에 남은 프레임을 디스크에 플러시
2. CSV 파일 정상 종료
3. 크래시 전까지 저장된 프레임 수 로그 출력

데이터는 보존되며, 재실행 시 새로운 daytime 폴더가 생성되므로 기존 데이터를 덮어쓰지 않습니다.
