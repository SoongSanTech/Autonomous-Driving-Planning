# 숭산텍 실험 플레이북 (Experiment Playbook)

> 작성일: 2026-03-20
> 목적: CARLA 데이터 수집 → BC 학습 → 추론 테스트까지의 전체 실험 절차를 단계별로 기술
> 대상: 직접 터미널에서 실행하며 실험을 진행하는 연구자
> 전제: `venv` 활성화 완료, CARLA 0.9.15 Windows Host 실행 가능
> CARLA Host IP: `172.28.224.1`

---

## 0. 사전 준비 (Pre-flight Checklist)

### 0.1 환경 확인

```bash
# WSL2 터미널에서 실행
source ./venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

# Python 버전 확인 (3.10+)
python --version

# 핵심 패키지 확인
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import carla; print('CARLA API OK')"

# GPU 상태 확인
nvidia-smi
```

### 0.2 CARLA 서버 실행 (Windows Host)

```powershell
# Windows PowerShell에서 실행
# CARLA 설치 경로로 이동 후:
.\CarlaUE4.exe -quality-level=Low -RenderOffScreen
```

> `-RenderOffScreen` 옵션은 GPU 메모리를 절약합니다. 모니터로 확인하고 싶으면 제거하세요.

### 0.3 CARLA 연결 테스트

```bash
python -c "
import carla
client = carla.Client('172.28.224.1', 2000)
client.set_timeout(10.0)
world = client.get_world()
print(f'Connected: {world.get_map().name}')
print(f'Spawn points: {len(world.get_map().get_spawn_points())}')
"
```

### 0.4 PYTHONPATH 설정

모든 `python -m` 명령어는 `src/` 디렉토리를 모듈 경로에 포함해야 합니다.
셸 세션 시작 시 한 번만 설정하세요:

```bash
export PYTHONPATH=src:$PYTHONPATH
```

> 이하 모든 명령어는 위 설정이 되어 있다고 가정합니다.
> 설정을 잊었다면 각 명령어 앞에 `PYTHONPATH=src`를 붙여도 됩니다.

### 0.5 디렉토리 구조 확인

```bash
ls src/data_pipeline/pipeline.py
ls src/model/train_bc.py
ls src/experiment/cli.py

mkdir -p checkpoints
mkdir -p src/data
mkdir -p experiments/logs
```

### 0.6 테스트 스위트 실행 (선택)

```bash
# 전체 테스트 (약 15분 소요, torch 초기 로드 느림)
python -m pytest tests/ -v --tb=short

# 빠른 확인 (data_pipeline만, ~1분)
python -m pytest tests/data_pipeline/ -v --tb=short
```

---

## 1. Stage 1: 10분 마이크로 루프 (Micro-Loop) — 파이프라인 개통

> 목표: 10분(6,000 프레임) 수집 → 검증 → 5 에포크 학습 → 추론 테스트
> 예상 소요: 약 40분 (수집 10분 + 검증 1분 + 학습 15분 + 추론 5분)
> 성공 기준: 차가 10초라도 차선을 따라가면 파이프라인 개통 완료

### Step 1.1: 10분 데이터 수집

```bash
python -m data_pipeline.cli \
  --host 172.28.224.1 \
  --port 2000 \
  --output-dir src/data \
  --duration 600
```

**확인 사항:**
- 터미널에 `Frames captured: ...` 로그가 10Hz로 출력되는지
- `Ctrl+C`로 중단해도 데이터가 보존되는지 (크래시 복원 기능)

**수집 완료 후 확인:**

```bash
# 생성된 세션 디렉토리 확인
ls src/data/
# 예: 2026-03-20_143000/

# 프레임 수 확인 (약 6,000개 예상)
ls src/data/{SESSION}/front/ | wc -l

# CSV 확인
head -5 src/data/{SESSION}/labels/driving_log.csv
```

> `{SESSION}`은 실제 생성된 디렉토리명(예: `2026-03-20_143000`)으로 대체하세요.

### Step 1.2: 데이터 품질 검증 + 조향 분포 확인 ⚠️

```bash
python -c "
from experiment.data_validator import DataValidator
from experiment.experiment_logger import ExperimentLogger

logger = ExperimentLogger(db_path='experiments/experiment_log.db')
validator = DataValidator(experiment_logger=logger)
report = validator.validate_session('src/data/{SESSION}')

print('=== 데이터 검증 보고서 ===')
print(f'총 프레임: {report.total_frames}')
print(f'유효 프레임: {report.valid_frames}')
print(f'손상 프레임: {report.corrupted_frames}')
print(f'범위 밖 steering: {report.out_of_range_steering}')
print(f'범위 밖 throttle: {report.out_of_range_throttle}')
print(f'타이밍 이상: {report.timing_anomalies}')
print(f'재수집 필요: {report.needs_recollection}')
print()
print('=== 조향 분포 (핵심 확인 항목) ===')
print(f'Steering 평균: {report.steering_mean:.4f}')
print(f'Steering 표준편차: {report.steering_std:.4f}')
print(f'Throttle 평균: {report.throttle_mean:.4f}')
print(f'Throttle 표준편차: {report.throttle_std:.4f}')

# 조향 분포 히스토그램 출력
dist = validator.analyze_distribution('src/data/{SESSION}')
hist = dist.get('steering_histogram', {})
counts = hist.get('counts', [])
edges = hist.get('bin_edges', [])
if counts:
    print()
    print('=== Steering 히스토그램 ===')
    max_count = max(counts) if counts else 1
    for i, c in enumerate(counts):
        bar = '#' * int(40 * c / max_count) if max_count > 0 else ''
        lo, hi = edges[i], edges[i+1]
        pct = c / sum(counts) * 100
        print(f'  [{lo:+.2f}, {hi:+.2f}) {bar} {c} ({pct:.1f}%)')
    
    # 직진 편향 경고
    center_bins = [c for i, c in enumerate(counts) if abs((edges[i]+edges[i+1])/2) < 0.05]
    center_ratio = sum(center_bins) / sum(counts) * 100 if sum(counts) > 0 else 0
    print(f'\n직진(|steering| < 0.05) 비율: {center_ratio:.1f}%')
    if center_ratio > 70:
        print('WARNING: 직진 편향 심각 (>70%). Under-sampling 또는 Weighted Sampling 필요')
    elif center_ratio > 50:
        print('CAUTION: 직진 비율 높음 (>50%). 데이터 증강으로 보완 가능')
    else:
        print('OK: 조향 분포 양호')
"
```

**판단 기준:**

| 항목 | 정상 | 주의 | 위험 (재수집) |
|------|:----:|:----:|:------------:|
| 손상 프레임 비율 | < 1% | 1~5% | > 5% |
| 범위 밖 레이블 | 0 | < 10 | > 10 |
| 타이밍 이상 | < 5% | 5~10% | > 10% |
| 직진 비율 | < 50% | 50~70% | > 70% |

> **직진 비율 > 70%이면 Step 1.3으로 진행하지 마세요.** 부록 A "조향 편향 대응"을 먼저 참고하세요.

### Step 1.3: BC 5 에포크 학습 (Micro Training)

```bash
python -m model.train_bc \
  --data_path src/data/{SESSION} \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --frozen_epochs 3 \
  --patience 10 \
  --checkpoint_dir checkpoints \
  --num_workers 2
```

**확인 사항:**
- `Phase 1: backbone frozen` 로그 출력
- 에포크마다 `train_loss`, `val_loss`, `mae_steer`, `mae_throttle` 출력
- `checkpoints/` 디렉토리에 `.pth` 파일 생성

**예상 출력 예시:**
```
Epoch 1/5 — train_loss: 0.8234, val_loss: 0.7891, mae_steer: 0.3210, mae_throttle: 0.1543
Epoch 2/5 — train_loss: 0.5123, val_loss: 0.4987, mae_steer: 0.2456, mae_throttle: 0.1234
...
```

> 5 에포크는 수렴을 기대하는 것이 아닙니다. 파이프라인이 동작하는지 확인하는 것이 목적입니다.

```bash
# 생성된 체크포인트 확인
ls -la checkpoints/bc_*.pth
```

### Step 1.4: CARLA 추론 테스트 (10초 생존 확인)

```bash
python -m model.run_inference \
  --checkpoint checkpoints/bc_*.pth \
  --carla_host 172.28.224.1 \
  --carla_port 2000 \
  --max_steps 300
```

> `--max_steps 300` = 30초 (10Hz 기준). 차가 충돌하면 자동 종료됩니다.

**관찰 포인트:**
- CARLA 화면에서 차량이 움직이는지 확인
- 터미널 로그에서 `steer`, `throttle` 값이 변하는지 확인
- `latency` 값이 100ms 이하인지 확인

**결과 판단:**

| 결과 | 의미 | 다음 행동 |
|------|------|----------|
| 10초 이상 차선 추종 | 파이프라인 개통 성공 | Stage 2로 진행 |
| 즉시 충돌/벽으로 돌진 | 데이터 또는 모델 문제 | Step 1.2 재확인 |
| 차가 움직이지 않음 | throttle 출력 문제 | 체크포인트 메트릭 확인 |
| 연결 오류 | CARLA 서버 문제 | Step 0.3 재실행 |

> **10초라도 차선을 따라가면 성공입니다.** 완벽한 주행은 Stage 2에서 달성합니다.

### Step 1.5: Micro-Loop 결과 기록

```bash
python -c "
from experiment.experiment_logger import ExperimentLogger

logger = ExperimentLogger(db_path='experiments/experiment_log.db')
eid = logger.create_experiment(
    'micro_loop', 'Stage 1 마이크로 루프 파이프라인 개통 테스트',
    {
        'session': '{SESSION}',
        'epochs': 5,
        'duration_sec': 600,
        'frames': 6000,
    }
)
# 아래 값은 실제 관찰 결과로 대체하세요
logger.log_metrics(eid, {
    'train_loss': 0.0,       # 마지막 에포크 train_loss
    'val_loss': 0.0,         # 마지막 에포크 val_loss
    'mae_steering': 0.0,     # 마지막 에포크 mae_steer
    'mae_throttle': 0.0,     # 마지막 에포크 mae_throttle
    'survival_seconds': 0.0, # CARLA 추론 시 생존 시간(초)
    'pipeline_ok': 1,        # 파이프라인 개통 여부 (1=성공, 0=실패)
})
logger.update_status(eid, 'completed')
print(f'실험 기록 완료: {eid}')
"
```

---

## 2. Stage 2: 1시간 본격 수집 + BC 본학습

> 전제: Stage 1 Micro-Loop 성공 (파이프라인 개통 확인)
> 목표: 1시간(36,000 프레임) 수집 → 조향 분포 게이트 → BC 50 에포크 학습
> 예상 소요: 약 3~4시간 (수집 1시간 + 학습 1~2시간 + 평가 30분)

### Step 2.1: 1시간 데이터 수집

```bash
python -m data_pipeline.cli \
  --host 172.28.224.1 \
  --port 2000 \
  --output-dir src/data \
  --duration 3600
```

**수집 중 모니터링:**
- 터미널 로그에서 에피소드 전환(날씨/시간대 변경) 확인 (5분마다)
- 프레임 드롭 경고가 없는지 확인
- 수집 완료 시 통계 출력 확인 (frames saved, drops, duration)

```bash
# 수집 완료 후 확인
ls src/data/{SESSION_1H}/front/ | wc -l    # 약 36,000개 예상
du -sh src/data/{SESSION_1H}/              # 약 50GB 예상 (Front RGB only)
```

### Step 2.2: 조향 분포 게이트 ⚠️

Step 1.2와 동일한 검증 스크립트를 `{SESSION_1H}`로 실행합니다.

```bash
python -c "
from experiment.data_validator import DataValidator
from experiment.experiment_logger import ExperimentLogger

logger = ExperimentLogger(db_path='experiments/experiment_log.db')
validator = DataValidator(experiment_logger=logger)
report = validator.validate_session('src/data/{SESSION_1H}')

print(f'총 프레임: {report.total_frames}')
print(f'유효: {report.valid_frames}, 손상: {report.corrupted_frames}')
print(f'Steering 평균: {report.steering_mean:.4f}, 표준편차: {report.steering_std:.4f}')

dist = validator.analyze_distribution('src/data/{SESSION_1H}')
hist = dist.get('steering_histogram', {})
counts = hist.get('counts', [])
edges = hist.get('bin_edges', [])
if counts:
    center_bins = [c for i, c in enumerate(counts) if abs((edges[i]+edges[i+1])/2) < 0.05]
    center_ratio = sum(center_bins) / sum(counts) * 100 if sum(counts) > 0 else 0
    print(f'직진 비율: {center_ratio:.1f}%')
    if center_ratio > 70:
        print('WARNING: 직진 편향 심각. 부록 A 참고')
    else:
        print('OK: 학습 진행 가능')
"
```

> **직진 비율 > 70%이면 Step 2.3으로 진행하지 마세요.** 부록 A를 먼저 수행하세요.

### Step 2.3: BC 본학습 (50 에포크)

```bash
python -m model.train_bc \
  --data_path src/data/{SESSION_1H} \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --frozen_epochs 10 \
  --patience 10 \
  --checkpoint_dir checkpoints \
  --num_workers 4
```

**학습 중 모니터링:**
- Phase 1 (에포크 1~10): backbone frozen, FC head만 학습
- Phase 2 (에포크 11~): backbone unfreeze, LR 1e-5로 감소
- Early stopping: val_loss가 10 에포크 연속 개선 없으면 자동 종료
- 최종 출력에서 `Best val_loss`와 `Best checkpoint` 경로 기록

**학습 완료 후 확인:**

```bash
# 체크포인트 목록
ls -la checkpoints/bc_*.pth

# 최적 체크포인트 메트릭 확인
python -c "
import torch
ckpt = torch.load('checkpoints/bc_*.pth', map_location='cpu')
print('Epoch:', ckpt.get('epoch'))
print('Metrics:', ckpt.get('metrics'))
"
```

### Step 2.4: BC 오프라인 평가

```bash
python -m model.evaluate \
  --checkpoint checkpoints/bc_*.pth \
  --test_data src/data/{SESSION_1H} \
  --batch_size 32
```

**목표 메트릭:**

| 메트릭 | 목표 | 비고 |
|--------|:----:|------|
| MAE Steering | < 0.15 | 0.10 이하면 우수 |
| MAE Throttle | < 0.10 | 0.08 이하면 우수 |

> 목표 미달 시: 에포크 수 증가, LR 조정, 데이터 증강 강화 후 Step 2.3 재실행

### Step 2.5: BC CARLA 온라인 평가 (BC 품질 게이트) 🚧

```bash
# 방법 1: 실시간 추론 (시각적 확인, 5분)
python -m model.run_inference \
  --checkpoint checkpoints/bc_*.pth \
  --carla_host 172.28.224.1 \
  --max_steps 3000
```

```bash
# 방법 2: 정량 평가 (10 에피소드)
python -m model.evaluate \
  --checkpoint checkpoints/bc_*.pth \
  --online \
  --carla_host 172.28.224.1 \
  --num_episodes 10
```

**BC 품질 게이트 — RL 진입 최소 기준:**

| 메트릭 | 최소 기준 | 확인 방법 |
|--------|:---------:|----------|
| 직선 도로 생존 시간 | ≥ 30초 | 실시간 추론 관찰 |
| 교차로 통과 | ≥ 1회 성공 | 10 에피소드 온라인 평가 |
| MAE Steering | < 0.15 | 오프라인 평가 |

**게이트 미통과 시 대응:**

| 증상 | 원인 추정 | 대응 |
|------|----------|------|
| 즉시 벽 충돌 | 데이터 편향 또는 학습 부족 | 조향 분포 재확인 + 에포크 증가 |
| 직선은 OK, 커브에서 실패 | 커브 데이터 부족 | 커브 구간 데이터 추가 수집 |
| 교차로에서 무조건 직진 | 직진 편향 | 부록 A 수행 후 재학습 |
| 차가 멈춤 | throttle 학습 실패 | throttle_weight 증가 (1.0→2.0) |
| 지그재그 주행 | steering 노이즈 | steering_weight 증가 (2.0→3.0) |

> **이 게이트를 통과할 때까지 Step 2.3~2.5를 반복합니다. RL로 넘어가지 마세요.**

### Step 2.6: BC 결과 기록

```bash
python -c "
from experiment.experiment_logger import ExperimentLogger

logger = ExperimentLogger(db_path='experiments/experiment_log.db')
eid = logger.create_experiment(
    'bc_training', 'Stage 2 BC 본학습',
    {
        'session': '{SESSION_1H}',
        'epochs': 50,
        'lr': 1e-4,
        'batch_size': 32,
        'frozen_epochs': 10,
        'steering_weight': 2.0,
    }
)
# 아래 값은 실제 결과로 대체하세요
logger.log_metrics(eid, {
    'best_val_loss': 0.0,
    'mae_steering': 0.0,
    'mae_throttle': 0.0,
    'survival_seconds': 0.0,
    'intersection_pass': 0,    # 교차로 통과 횟수 (10회 중)
    'gate_passed': 0,          # BC 품질 게이트 통과 여부 (1/0)
})
logger.update_status(eid, 'completed')
print(f'실험 기록 완료: {eid}')
"
```

---

## 3. Stage 3: RL 미세조정 (PPO)

> 전제: Stage 2 BC 품질 게이트 통과 (직선 30초 + 교차로 1회)
> 목표: BC 체크포인트를 warm-start하여 PPO로 충돌 회피 + 차선 유지 강화
> 예상 소요: 4~8시간 (3,000~5,000 에피소드)

### Step 3.1: RL 학습 실행

```bash
python -m model.train_rl \
  --bc_checkpoint checkpoints/bc_*.pth \
  --carla_host 172.28.224.1 \
  --carla_port 2000 \
  --episodes 3000 \
  --frozen_episodes 100 \
  --lr 3e-5 \
  --finetune_lr 1e-5 \
  --checkpoint_dir checkpoints \
  --checkpoint_interval 100
```

**학습 중 모니터링:**
- Phase 1 (에피소드 1~100): backbone frozen, Actor/Critic head만 학습
- Phase 2 (에피소드 101~): backbone unfreeze, LR 1e-5
- 100 에피소드마다 체크포인트 저장
- 평균 reward 상승 추세 확인

**주의 사항:**
- CARLA 서버가 학습 중 크래시할 수 있음 → 체크포인트에서 재개 가능
- 에이전트가 "가만히 서 있기" 지역 최적해에 빠지면:
  - reward.py의 `w_progress`를 0.1 → 0.3으로 상향
  - 재학습 필요

### Step 3.2: RL 온라인 평가

```bash
python -m model.evaluate \
  --checkpoint checkpoints/rl_*.pth \
  --online \
  --carla_host 172.28.224.1 \
  --num_episodes 10
```

**목표 메트릭:**

| 메트릭 | 목표 | 비고 |
|--------|:----:|------|
| 충돌 없이 주행 시간 | ≥ 60초 | 평균 생존 시간 |
| 교차로 통과율 | ≥ 80% (8/10) | 10 에피소드 기준 |
| 평균 차선 거리 | < 1.0m | 차선 중앙 유지 |

### Step 3.3: BC vs RL 비교

```bash
python -c "
from experiment.experiment_logger import ExperimentLogger

logger = ExperimentLogger(db_path='experiments/experiment_log.db')
# BC와 RL 실험 ID를 입력하세요
bc_id = '{BC_EXPERIMENT_ID}'
rl_id = '{RL_EXPERIMENT_ID}'
comparison = logger.compare_experiments([bc_id, rl_id])
for key, data in comparison.items():
    print(f'{key}: delta={data[\"delta\"]:.4f}, improved={data[\"improved\"]}')
"
```

---

## 4. Stage 4: 하이퍼파라미터 그리드 서치 (선택)

> 전제: Stage 2 또는 Stage 3 완료 후, 성능 개선이 필요할 때
> 목표: 최적 하이퍼파라미터 조합 탐색

### Step 4.1: BC 그리드 서치

```bash
python -m experiment.cli grid-search-bc \
  --data-path src/data/{SESSION_1H} \
  --db-path experiments/experiment_log.db
```

기본 탐색 범위:
- `lr`: [5e-5, 1e-4, 3e-4]
- `batch_size`: [16, 32, 64]
- `steering_weight`: [1.5, 2.0, 3.0]
- 총 27개 조합 (50% 이상 실패 시 자동 중단)

### Step 4.2: RL Reward 그리드 서치

```bash
python -m experiment.cli grid-search-rl \
  --bc-checkpoint checkpoints/bc_*.pth \
  --carla-host 172.28.224.1 \
  --db-path experiments/experiment_log.db
```

기본 탐색 범위:
- `w_progress`: [0.1, 0.3, 0.5]
- `w_collision`: [0.5, 1.0, 2.0]
- `w_steering`: [0.3, 0.5, 1.0]
- 총 27개 조합

### Step 4.3: 결과 보고서 확인

```bash
python -m experiment.cli report --db-path experiments/experiment_log.db
```

---

## A. 부록: 조향 편향 대응 방안

### A.1 문제 진단

CARLA autopilot은 대부분의 시간을 직진으로 주행합니다. 수집 데이터의 steering 분포가 0 근처에 집중되면:
- BC 모델이 "항상 직진"을 학습
- 교차로/커브에서 조향 불가
- RL warm-start 시에도 조향 능력 부재

### A.2 대응 방법 1: 수집 경로 다양화

CARLA Town03(교차로 많음) 또는 Town04(고속도로 + 커브)에서 수집:

```bash
# Town 변경 후 수집 (pipeline.py에서 Town 설정 필요)
# 현재 코드는 기본 Town 사용. 필요시 코드 수정 요청
python -m data_pipeline.cli --host 172.28.224.1 --duration 600
```

### A.3 대응 방법 2: 학습 시 데이터 증강 활용

현재 `dataset.py`에 구현된 증강:
- 좌우 반전 (steering 부호 반전) — 50% 확률
- 밝기 ±20%
- 가우시안 노이즈

이 증강은 `--no_augment` 플래그 없이 학습하면 자동 적용됩니다.

```bash
# 증강 활성화 확인 (기본값)
python -m model.train_bc --data_path src/data/{SESSION} --epochs 50
# → 로그에 "augment=True" 출력 확인
```

### A.4 대응 방법 3: 직진 프레임 Under-sampling (코드 수정 필요)

직진 비율이 70% 이상이면, 학습 전에 직진 프레임을 일부 제거하는 것이 효과적입니다.

```python
# 아이디어: dataset.py의 DataLoaderFactory에 추가할 로직
# 구현이 필요하면 요청하세요

# 개념:
# 1. driving_log.csv에서 |steering| < 0.05인 행을 식별
# 2. 해당 행의 50%를 랜덤 제거
# 3. 나머지로 학습
```

> 이 로직이 필요하면 "Under-sampling 구현해줘"라고 요청하세요.

---

## B. 부록: 트러블슈팅

### B.1 CARLA 연결 오류

```
RuntimeError: time-out of 10000ms while waiting for the simulator
```

**해결:**
1. Windows에서 CARLA 서버가 실행 중인지 확인
2. WSL2 → Windows 방화벽에서 포트 2000 허용
3. IP 확인: `cat /etc/resolv.conf | grep nameserver`

### B.2 모듈 import 오류

```
ModuleNotFoundError: No module named 'data_pipeline'
```

**해결:**
```bash
export PYTHONPATH=src:$PYTHONPATH
```

### B.3 GPU 메모리 부족 (OOM)

```
RuntimeError: CUDA out of memory
```

**해결:**
```bash
python -m model.train_bc --data_path src/data/{SESSION} --batch_size 16
# 또는
python -m model.train_bc --data_path src/data/{SESSION} --num_workers 0
```

### B.4 torch 초기 로드 느림

WSL2에서 PyTorch 첫 import에 1~2분 소요될 수 있습니다. 정상입니다.

### B.5 데이터 수집 중 CARLA 크래시

파이프라인은 크래시 감지 시 자동으로 큐를 플러시하고 데이터를 보존합니다.
크래시 후 재실행하면 새 세션 디렉토리가 생성됩니다 (기존 데이터 덮어쓰기 없음).

### B.6 학습 중 NaN loss

```
WARNING: NaN loss detected, skipping batch
```

- 간헐적 발생: 정상 (해당 배치만 스킵)
- 연속 발생: LR이 너무 높음 → `--lr 5e-5`로 낮추기

### B.7 RL 에이전트가 정지/회전만 함

- `w_progress` 값을 0.1 → 0.3~0.5로 상향
- BC 체크포인트 품질 재확인 (BC 게이트 미통과 상태에서 RL 진입했을 가능성)

---

## C. 부록: 실험 전체 흐름도

```
                    ┌─────────────────────────┐
                    │  Stage 0: 사전 준비      │
                    │  환경 확인 + CARLA 연결   │
                    │  export PYTHONPATH=src   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Stage 1: Micro-Loop     │
                    │  10분 수집 → 5ep 학습    │
                    │  → 10초 생존 확인        │
                    └────────────┬────────────┘
                                 │
                         파이프라인 개통?
                        NO ↙         ↘ YES
                   ┌────────┐   ┌────────────▼────────────┐
                   │ 디버깅  │   │  Step 2.1: 1시간 수집    │
                   │ → 재시도│   └────────────┬────────────┘
                   └────────┘                │
                                 ┌───────────▼───────────┐
                                 │  Step 2.2: 조향 분포    │
                                 │  게이트 확인            │
                                 └───────────┬───────────┘
                                             │
                                    직진 < 70%?
                                   NO ↙      ↘ YES
                            ┌──────────┐  ┌──────▼──────────┐
                            │ 부록 A    │  │ Step 2.3: BC 학습│
                            │ 편향 대응 │  │ (50 에포크)      │
                            └──────────┘  └──────┬──────────┘
                                                 │
                                     ┌───────────▼───────────┐
                                     │  Step 2.5: BC 게이트   │
                                     │  30초 생존 + 교차로 1회│
                                     └───────────┬───────────┘
                                                 │
                                        게이트 통과?
                                       NO ↙      ↘ YES
                                ┌──────────┐  ┌──────▼──────────┐
                                │ 데이터/   │  │ Stage 3: RL PPO  │
                                │ 하이퍼    │  │ (3,000+ 에피소드)│
                                │ 파라미터  │  └──────┬──────────┘
                                │ 조정 후   │         │
                                │ 재학습    │  ┌──────▼──────────┐
                                └──────────┘  │ 60초 생존 +      │
                                              │ 교차로 80% 통과  │
                                              └─────────────────┘
```

---

## D. 부록: 명령어 빠른 참조 (Quick Reference)

```bash
# === 환경 설정 (셸 세션 시작 시 1회) ===
source ./venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

# === 데이터 수집 ===
python -m data_pipeline.cli --host 172.28.224.1 --duration 600     # 10분
python -m data_pipeline.cli --host 172.28.224.1 --duration 3600    # 1시간

# === 데이터 검증 ===
python -m experiment.cli validate --session-dir src/data/{SESSION}

# === BC 학습 ===
python -m model.train_bc --data_path src/data/{SESSION} --epochs 5   # 마이크로
python -m model.train_bc --data_path src/data/{SESSION} --epochs 50  # 본학습

# === BC 평가 ===
python -m model.evaluate --checkpoint checkpoints/bc_*.pth --test_data src/data/{SESSION}
python -m model.evaluate --checkpoint checkpoints/bc_*.pth --online --carla_host 172.28.224.1

# === BC 추론 (실시간) ===
python -m model.run_inference --checkpoint checkpoints/bc_*.pth --carla_host 172.28.224.1

# === RL 학습 (BC 게이트 통과 후에만) ===
python -m model.train_rl --bc_checkpoint checkpoints/bc_*.pth --carla_host 172.28.224.1

# === RL 평가 ===
python -m model.evaluate --checkpoint checkpoints/rl_*.pth --online --carla_host 172.28.224.1

# === 그리드 서치 ===
python -m experiment.cli grid-search-bc --data-path src/data/{SESSION}
python -m experiment.cli grid-search-rl --bc-checkpoint checkpoints/bc_*.pth --carla-host 172.28.224.1

# === 실험 보고서 ===
python -m experiment.cli report
```

---

> 이 문서는 실험 진행에 따라 업데이트됩니다.
> 문제 발생 시 부록 B(트러블슈팅)를 먼저 참고하세요.
