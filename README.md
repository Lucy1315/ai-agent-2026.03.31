# YouTube Thumbnail Generator Agent

LangGraph 기반의 YouTube 썸네일 생성 에이전트입니다. YouTube URL을 입력하면 AI로 썸네일을 생성하고, 일반 검색어를 입력하면 웹 검색 결과를 요약합니다.

## 주요 기능

### 노드 구성 (9개)

| 노드 | 역할 |
|------|------|
| `memory_load` | 이전 대화 기록 로드 |
| `input_classifier` | YouTube URL / 검색어 분류 |
| `youtube_analyzer` | 비디오 메타데이터 수집 및 썸네일 전략 결정 |
| `thumbnail_style_gen` | 특정 스타일의 썸네일 생성 (병렬 실행) |
| `thumbnail_selector` | 병렬 생성 결과에서 최적 선택 |
| `thumbnail_downloader` | YouTube 기존 썸네일 다운로드 |
| `web_search` | DuckDuckGo 웹 검색 + Claude 요약 |
| `memory_save` | 대화 기록 저장 |
| `result` | 최종 결과 포맷 |

### Conditional Edges (2개)

- **`route_by_input_type`** - 입력이 YouTube URL이면 `youtube_analyzer`, 검색어면 `web_search`로 라우팅
- **`route_by_thumbnail_mode`** - `ai_generate` 모드면 Send API로 병렬 생성, `download` 모드면 기존 썸네일 다운로드

### Tool 연동 (4개)

| Tool | 라이브러리 | 역할 |
|------|-----------|------|
| `get_youtube_info` | yt-dlp | 비디오 메타데이터 및 썸네일 URL 추출 |
| `web_search` | duckduckgo-search | DuckDuckGo 웹 검색 |
| `generate_image` | Replicate / Stability AI / PIL | AI 이미지 생성 (폴백 체인) |
| `download_image` | requests | URL에서 이미지 다운로드 |

### 병렬 실행 (Send API)

`youtube_analyzer` 이후 `ai_generate` 모드에서 **Send API**를 사용하여 2개 스타일의 썸네일을 동시에 생성합니다:

- **bold_dramatic** - 강렬한 색상, 시네마틱 조명, 높은 대비
- **clean_minimal** - 깔끔한 디자인, 파스텔 톤, 모던 타이포그래피

생성된 결과는 `thumbnail_selector` 노드에서 병합됩니다.

### 메모리 기능

`memory.json` 파일에 대화 기록을 자동 저장/로드합니다. 이전 대화 컨텍스트가 Claude 프롬프트에 주입되어 연속적인 대화가 가능합니다.

## 그래프 구조

```
START
  ↓
[memory_load]
  ↓
[input_classifier]
  ├─ youtube → [youtube_analyzer]
  │              ├─ ai_generate → Send API 병렬:
  │              │    ├─ [thumbnail_style_gen] (bold_dramatic)
  │              │    └─ [thumbnail_style_gen] (clean_minimal)
  │              │              ↓
  │              │    [thumbnail_selector]
  │              └─ download → [thumbnail_downloader]
  │                        ↓
  └─ search → [web_search]
                    ↓
              [memory_save]
                    ↓
                [result]
                    ↓
                   END
```

## 설치 및 실행

### 1. 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키를 설정합니다:

```env
# 필수
ANTHROPIC_API_KEY=your_anthropic_api_key

# 선택 - AI 이미지 생성 (없으면 PIL 폴백)
REPLICATE_API_TOKEN=your_replicate_token
STABILITY_API_KEY=your_stability_key

# 썸네일 모드: ai_generate 또는 download
THUMBNAIL_MODE=ai_generate
```

### 3. 실행

```bash
# YouTube URL로 썸네일 생성
python main.py "https://www.youtube.com/watch?v=..."

# 웹 검색
python main.py "LangGraph tutorial"

# 대화형 모드
python main.py
```

## 프로젝트 구조

```
.
├── main.py              # 엔트리포인트
├── agent/
│   ├── state.py         # AgentState 정의 (Annotated 리듀서 포함)
│   ├── graph.py         # LangGraph 그래프 빌드 (Send API 포함)
│   ├── nodes.py         # 노드 구현 (9개)
│   └── tools.py         # Tool 함수 (4개 + 메모리)
├── thumbnails/          # 생성된 썸네일 저장 디렉토리
├── memory.json          # 대화 기록 (자동 생성)
├── requirements.txt
├── .env.example
└── README.md
```
