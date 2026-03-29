"""
LangGraph 그래프 정의

그래프 구조:
  START
    ↓
  [memory_load]                    ← 메모리: 이전 대화 로드
    ↓
  [input_classifier]
    ↓ (조건부 엣지 1: input_type)
    ├─ "youtube" → [youtube_analyzer]
    │                 ↓ (조건부 엣지 2: thumbnail_mode)
    │                 ├─ "ai_generate" → Send API 병렬 실행:
    │                 │    ├─ [thumbnail_style_gen] (bold_dramatic)
    │                 │    └─ [thumbnail_style_gen] (clean_minimal)
    │                 │              ↓
    │                 │    [thumbnail_selector]
    │                 └─ "download" → [thumbnail_downloader]
    │                           ↓
    └─ "search" → [web_search]  ↓
                       ↓    [memory_save]     ← 메모리: 대화 저장
                       └──────┘
                           ↓
                        [result]
                           ↓
                          END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from .state import AgentState
from .nodes import (
    memory_load_node,
    input_classifier_node,
    youtube_analyzer_node,
    thumbnail_style_gen_node,
    thumbnail_selector_node,
    download_thumbnail_node,
    web_search_node,
    memory_save_node,
    result_node,
)


# ── 조건부 엣지 라우터 ────────────────────────────────────────────────────────

def route_by_input_type(state: AgentState) -> str:
    """입력 타입에 따라 youtube 또는 search로 라우팅합니다."""
    return state.get("input_type", "search")


def route_by_thumbnail_mode(state: AgentState):
    """
    썸네일 모드에 따라 라우팅합니다.
    - ai_generate: Send API로 여러 스타일을 병렬 생성
    - download: 기존 썸네일 다운로드
    """
    mode = state.get("thumbnail_mode", "download")
    if mode == "ai_generate":
        info = state.get("video_info", {})
        title = info.get("title", "")
        styles = [
            ("bold_dramatic",
             f"YouTube thumbnail for '{title}', "
             "bold dramatic colors, cinematic lighting, intense mood, "
             "high contrast, dynamic composition, 16:9 aspect ratio"),
            ("clean_minimal",
             f"YouTube thumbnail for '{title}', "
             "clean minimal design, soft pastel colors, modern typography, "
             "elegant whitespace, professional layout, 16:9 aspect ratio"),
        ]
        return [
            Send("thumbnail_style_gen", {
                **state,
                "thumbnail_style": name,
                "image_prompt": prompt,
                "thumbnail_results": [],
            })
            for name, prompt in styles
        ]
    return "thumbnail_downloader"


# ── 그래프 빌더 ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    # 노드 등록
    builder.add_node("memory_load", memory_load_node)
    builder.add_node("input_classifier", input_classifier_node)
    builder.add_node("youtube_analyzer", youtube_analyzer_node)
    builder.add_node("thumbnail_style_gen", thumbnail_style_gen_node)
    builder.add_node("thumbnail_selector", thumbnail_selector_node)
    builder.add_node("thumbnail_downloader", download_thumbnail_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("memory_save", memory_save_node)
    builder.add_node("result", result_node)

    # 시작 → 메모리 로드 → 입력 분류
    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "input_classifier")

    # 조건부 엣지 1: 입력 타입으로 분기
    builder.add_conditional_edges(
        "input_classifier",
        route_by_input_type,
        {"youtube": "youtube_analyzer", "search": "web_search"},
    )

    # 조건부 엣지 2: 썸네일 모드로 분기 (ai_generate → Send API 병렬)
    builder.add_conditional_edges(
        "youtube_analyzer",
        route_by_thumbnail_mode,
        ["thumbnail_style_gen", "thumbnail_downloader"],
    )

    # 병렬 생성 → 선택기
    builder.add_edge("thumbnail_style_gen", "thumbnail_selector")

    # 모든 경로 → 메모리 저장 → 결과 → 종료
    builder.add_edge("thumbnail_selector", "memory_save")
    builder.add_edge("thumbnail_downloader", "memory_save")
    builder.add_edge("web_search", "memory_save")
    builder.add_edge("memory_save", "result")
    builder.add_edge("result", END)

    return builder.compile()
