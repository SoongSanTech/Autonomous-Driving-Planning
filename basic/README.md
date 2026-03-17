# CARLA 자율주행 시뮬레이션 프로젝트

CARLA 시뮬레이터를 활용한 자율주행 알고리즘 개발 및 테스트 프로젝트입니다.

## 프로젝트 구조

```
soongsantech_carla/
├── first_ride.py          # 기본 차량 생성 및 카메라 부착 예제
├── requirements.txt       # Python 패키지 의존성
├── README.md             # 프로젝트 문서
└── .gitignore            # Git 제외 파일 목록
```

## 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/SoongSanTech/Autonomous-Driving-Planning.git
cd Autonomous-Driving-Planning
```

### 2. 가상 환경 생성 및 활성화

**Linux/WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 실행 방법

### 1. CARLA 서버 실행
먼저 CARLA 시뮬레이터를 실행합니다. 서버가 `127.0.0.1:2000` 또는 지정된 IP에서 실행 중이어야 합니다.

### 2. 스크립트 실행
```bash
python first_ride.py
# 또는 WSL/Linux
python3 first_ride.py
```

### 3. 결과 확인
- CARLA 서버 창에서 차량이 생성되고 자동으로 주행합니다
- 서버 화면의 시점이 차량 카메라 위치로 자동 전환됩니다
- 15초간 실행 후 자동으로 종료됩니다

## 주의사항

- CARLA Python 패키지 버전(0.9.15)은 서버 버전과 반드시 일치해야 합니다
- 스크립트 실행 전 CARLA 서버가 실행 중이어야 합니다
- WSL 환경에서는 Windows에서 실행 중인 CARLA 서버의 IP를 확인하여 코드에서 수정해야 할 수 있습니다

## 협업 가이드

이 저장소는 Organization 저장소로, 여러 팀원이 함께 작업합니다. 아래 워크플로우를 따라주세요.

### 브랜치 전략

**main 브랜치**
- 프로덕션 레벨의 안정적인 코드만 포함
- 직접 push 금지 (보호된 브랜치)
- Pull Request를 통해서만 병합 가능

**개발 워크플로우**
1. 작업 시작 전 최신 코드 동기화
2. 기능별로 새 브랜치 생성
3. 작업 완료 후 Pull Request 생성
4. 코드 리뷰 후 main에 병합

### 작업 프로세스

#### 1. 최신 코드 가져오기
```bash
git checkout main
git pull origin main
```

#### 2. 새 브랜치 생성
```bash
# 브랜치 명명 규칙: feature/기능명, fix/버그명, docs/문서명
git checkout -b feature/lane-detection
# 또는
git checkout -b fix/camera-position
```

#### 3. 작업 및 커밋
```bash
# 변경사항 확인
git status

# 파일 추가
git add .

# 커밋 (명확한 메시지 작성)
git commit -m "feat: Add lane detection algorithm"
```

**커밋 메시지 규칙:**
- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정
- `refactor:` 코드 리팩토링
- `test:` 테스트 코드 추가
- `chore:` 빌드, 설정 파일 수정

#### 4. 원격 저장소에 푸시
```bash
git push origin feature/lane-detection
```

#### 5. Pull Request 생성
1. GitHub 웹사이트에서 저장소 접속
2. "Pull requests" 탭 클릭
3. "New pull request" 버튼 클릭
4. base: `main` ← compare: `feature/lane-detection` 선택
5. 제목과 설명 작성
   - 무엇을 변경했는지
   - 왜 변경했는지
   - 테스트 방법
6. Reviewers 지정 (팀원 선택)
7. "Create pull request" 클릭

#### 6. 코드 리뷰 및 병합
- 최소 1명 이상의 승인 필요
- 리뷰어의 피드백 반영
- 승인 후 "Merge pull request" 클릭
- 병합 후 로컬 브랜치 정리:
```bash
git checkout main
git pull origin main
git branch -d feature/lane-detection
```

### Fork vs Branch

**Branch 방식 (권장)**
- Organization 멤버는 브랜치 방식 사용
- 저장소에 직접 접근 권한이 있는 경우
- 위에서 설명한 워크플로우 사용

**Fork 방식**
- 외부 기여자나 임시 협력자
- 저장소 접근 권한이 없는 경우
- 개인 계정으로 Fork 후 작업

### 충돌 해결

브랜치 작업 중 main이 업데이트된 경우:

```bash
# main 브랜치 최신화
git checkout main
git pull origin main

# 작업 브랜치로 돌아가서 병합
git checkout feature/lane-detection
git merge main

# 충돌 발생 시 파일 수정 후
git add .
git commit -m "Merge main into feature/lane-detection"
git push origin feature/lane-detection
```

### 코드 리뷰 체크리스트

- [ ] 코드가 정상적으로 실행되는가?
- [ ] CARLA 서버 버전과 호환되는가?
- [ ] 주석과 문서가 충분한가?
- [ ] 불필요한 파일(venv, __pycache__ 등)이 포함되지 않았는가?
- [ ] 커밋 메시지가 명확한가?

### 문의 및 이슈

- 버그 발견 시: GitHub Issues에 등록
- 기능 제안: GitHub Discussions 활용
- 긴급 문의: 팀 채널 활용

## 라이선스

이 프로젝트는 Organization 내부 프로젝트입니다.
