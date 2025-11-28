# 🎬 실시간 영상 기반 GraphRAG 시스템 가이드

## 📋 개요

이 시스템은 **영상을 재생하면서 실시간으로 지식 그래프가 생성**되는 인터랙티브 GraphRAG 플랫폼입니다.

### 주요 기능

1. **영상 업로드**: 영상 파일 + 키워드 JSON 자동 업로드
2. **실시간 그래프 생성**: 영상 재생 시 현재 시점의 키워드가 자동으로 그래프 노드로 추가
3. **인터랙티브 시각화**: 노드 클릭 시 관련 문서 즉시 표시
4. **GraphRAG 답변**: 지식 그래프 기반 질문-답변

---

## 🚀 빠른 시작

### 1. 환경 설정

#### Backend (Python)

```bash
# 의존성 설치
pip install fastapi uvicorn python-multipart python-arango
pip install langchain sentence-transformers torch

# ArangoDB 실행 (Docker)
docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD="" arangodb/arangodb:latest
```

#### Frontend (TypeScript/React)

```bash
cd frontend

# 의존성 설치
npm install

# 환경변수 설정
echo "VITE_API_BASE=http://localhost:8080" > .env
```

### 2. 실행

#### Backend 실행

```bash
python api_server.py
```

서버가 `http://localhost:8080`에서 실행됩니다.

#### Frontend 실행

```bash
cd frontend
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

---

## 📖 사용 시나리오

### 시나리오 1: 영상 업로드 및 자동 등록

1. **영상 준비**:
   - 영상 파일: `example_video.mp4`

2. **업로드**:
   - "0. 영상 업로드" 섹션에서:
     - 영상 ID: `example_video` 영상 파일 선택
     - "업로드" 버튼 클릭

3. **자동 처리**:
   - 영상이 `uploaded_videos/` 폴더에 저장됨
   - 키워드 JSON이 `video_keywords/` 폴더에 저장됨
   - `video_keywords/registry.json`에 자동 등록됨

### 시나리오 2: 실시간 그래프 생성

1. **영상 재생**:
   - 헤더의 영상 플레이어에서 영상 재생
   - 영상 ID에 `example_video` 입력

2. **"현재 구간 키워드 가져오기" 클릭**:
   - 현재 재생 시점(예: 60초)의 키워드가 자동으로:
     - ArangoDB에 엔티티로 추가됨
     - 그래프에 노드로 표시됨
     - 관련 관계가 엣지로 연결됨

3. **자동 업데이트**:
   - 영상이 진행되면서 새로운 키워드 등장 시:
     - 실시간으로 노드가 추가됨
     - 기존 노드와의 관계가 자동으로 연결됨

### 시나리오 3: 노드 클릭 및 문서 탐색

1. **노드 클릭**:
   - 그래프에서 관심 있는 엔티티 노드 클릭 (예: "이순신")

2. **자동 표시**:
   - 그래프 아래에 "선택된 엔티티" 패널이 나타남
   - **관련 문서**: 해당 엔티티가 언급된 모든 문서 표시
   - **엔티티 정보**: 타입, 카테고리, 출처 등 메타데이터

3. **탐색**:
   - 문서를 읽으면서 다른 노드를 클릭하여 지식 확장

### 시나리오 4: GraphRAG 질문 답변

1. **질문 입력**:
   - "5. 질문 및 GraphRAG 답변" 섹션에서:
     - 질문: "한산도 대첩에 대해 설명해주세요"
     - (선택) 영상 ID: `example_video`
     - (선택) 집중 키워드: "이순신, 조선 수군"

2. **답변 생성**:
   - GraphRAG가 다음을 종합하여 답변 생성:
     - 벡터 검색으로 찾은 관련 문서
     - 지식 그래프의 엔티티 관계
     - 그래프 확장 검색 결과

3. **결과 확인**:
   - LLM이 생성한 답변
   - 참고된 엔티티 목록
   - 그래프 키워드

---

## 🎨 그래프 시각화 기능

### 노드 타입별 색상

- **Person (인물)**: 파란색 (`#3b82f6`)
- **Event (사건)**: 보라색 (`#8b5cf6`)
- **Location (장소)**: 초록색
- **Date (날짜)**: 노란색

### 인터랙션

- **드래그**: 노드를 자유롭게 이동
- **클릭**: 노드 클릭 시 관련 문서 표시
- **줌**: 마우스 휠로 확대/축소
- **팬**: 빈 공간 드래그하여 이동

### 컨트롤

- **Zoom +/-**: 확대/축소 버튼
- **Fit View**: 모든 노드가 보이도록 자동 조정
- **MiniMap**: 전체 그래프 미니맵

---

## 🗂️ 키워드 JSON 형식

```json
{
  "slices": [
    {
      "start": 0,
      "end": 60,
      "keywords": [
        {"term": "임진왜란", "score": 0.9},
        {"term": "1592년", "score": 0.7}
      ],
      "entities": [
        {"name": "임진왜란", "score": 0.92},
        {"name": "선조", "score": 0.6}
      ]
    },
    {
      "start": 60,
      "end": 120,
      "keywords": [
        {"term": "한산도 대첩", "score": 0.95},
        {"term": "이순신", "score": 0.8}
      ],
      "entities": [
        {"name": "이순신", "score": 0.9},
        {"name": "조선 수군", "score": 0.7}
      ]
    }
  ]
}
```

### 필드 설명

- `start`: 시작 시간(초)
- `end`: 종료 시간(초)
- `keywords`: 해당 구간의 키워드 목록
  - `term`: 키워드 텍스트
  - `score`: 중요도 점수 (0~1)
- `entities`: 그래프 엔티티 목록
  - `name`: 엔티티 이름
  - `score`: 신뢰도 점수

---

## 🔧 API 엔드포인트

### 영상 업로드

```bash
POST /videos/upload
Content-Type: multipart/form-data

Parameters:
  - video_id: 영상 ID
  - file: 영상 파일
  - keyword_file: 키워드 JSON (선택)
```

### 실시간 키워드 스트리밍

```bash
GET /videos/{video_id}/stream-keywords?current_time=60&window=5

Response: Server-Sent Events (SSE)
```

### 키워드 조회

```bash
GET /videos/{video_id}/keywords?start=60&end=120&top_k=5
```

### 엔티티 조회

```bash
GET /entity/{entity_name}?depth=1
```

### GraphRAG 질문

```bash
POST /query
{
  "query": "한산도 대첩에 대해 설명해주세요",
  "video_id": "example_video",
  "focus_keywords": ["이순신", "조선 수군"]
}
```

---

## 🐛 문제 해결

### CORS 에러

- Backend가 올바른 CORS 설정을 포함하는지 확인
- Frontend `.env` 파일의 `VITE_API_BASE` 확인

### 그래프가 표시되지 않음

- `npm install reactflow` 실행 확인
- 브라우저 콘솔에서 에러 메시지 확인
- ArangoDB가 실행 중인지 확인 (`localhost:8529`)

### 영상 업로드 실패

- 파일 크기 제한 확인 (FastAPI 기본: 10MB)
- Backend 로그 확인
- `uploaded_videos/` 폴더 권한 확인

### SSE 연결 끊김

- 방화벽/프록시 설정 확인
- Backend 로그에서 타임아웃 메시지 확인
- `interval` 파라미터 조정 (기본: 1초)

---

## 📦 프로젝트 구조

```
history/
├── api_server.py                # FastAPI 서버
├── graph_context_service.py     # GraphRAG 래퍼
├── video_keyword_store.py       # 키워드 저장소
├── video_registry.py            # 영상 레지스트리
├── graph_db.py                  # ArangoDB 인터페이스
├── rag_graph.py                 # GraphRAG 코어
├── video_keywords/              # 키워드 JSON 저장소
│   ├── registry.json
│   └── example_video.json
├── uploaded_videos/             # 업로드된 영상 파일
└── frontend/
    ├── src/
    │   ├── App.tsx              # 메인 앱
    │   ├── GraphView.tsx        # 그래프 시각화
    │   ├── api.ts               # API 클라이언트
    │   └── styles.css
    ├── package.json
    └── vite.config.ts
```

---

## 🎓 고급 사용법

### 커스텀 키워드 추출

영상에서 자동으로 키워드를 추출하려면:

```python
# TODO: Whisper + NER 파이프라인 구현
from whisper import load_model
from transformers import pipeline

# 1. 음성 → 텍스트 (Whisper)
model = load_model("base")
result = model.transcribe("video.mp4")

# 2. NER 추출
ner = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")
entities = ner(result["text"])

# 3. 키워드 JSON 생성
# ... (타임스탬프별 그룹화)
```

### 다중 언어 지원

```python
# graph_db.py의 KOREAN_STOPWORDS에 언어별 불용어 추가
ENGLISH_STOPWORDS = {...}
JAPANESE_STOPWORDS = {...}
```

### 성능 최적화

- **배치 크기 조정**: `graph_db.py`의 `batch_size` 파라미터
- **임베딩 캐싱**: 엔티티 임베딩을 Redis에 캐싱
- **WebSocket**: SSE 대신 WebSocket 사용

---

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 PR 환영합니다!

