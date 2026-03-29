"""
노드 구현:
  1. memory_load_node           - 이전 대화 기록 로드 (메모리)
  2. input_classifier_node      - URL인지 검색어인지 분류
  3. youtube_analyzer_node      - 비디오 메타데이터 + 자막 추출 + Claude 요약 + 이미지 프롬프트 생성
  4. thumbnail_style_gen_node   - 특정 스타일의 썸네일 생성 (Send API 병렬 실행)
  5. thumbnail_selector_node    - 병렬 생성 결과에서 최적 선택
  6. download_thumbnail_node    - 기존 YouTube 썸네일 다운로드
  7. web_search_node            - 웹 검색 + Claude 요약
  8. memory_save_node           - 대화 기록 저장 (메모리)
  9. result_node                - 최종 결과 포맷
"""

import re
import os
from anthropic import Anthropic
from .state import AgentState
from .tools import (
    get_youtube_info, web_search,
    generate_image, download_image, load_memory, save_memory,
)


def _get_client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)


def _format_history(history: list) -> str:
    """대화 기록을 Claude 프롬프트에 포함할 문자열로 변환합니다."""
    if not history:
        return ""
    lines = []
    for h in history[-5:]:
        inp = h.get("input", "")
        title = h.get("video_title", "")
        itype = h.get("input_type", "")
        lines.append(f"- [{itype}] {inp}" + (f" → {title}" if title else ""))
    return "이전 대화 기록:\n" + "\n".join(lines) + "\n\n"


# ── 1. memory_load_node ────────────────────────────────────────────────────

def memory_load_node(state: AgentState) -> dict:
    """이전 대화 기록을 메모리에서 로드합니다."""
    history = load_memory()
    return {"conversation_history": history}


# ── 2. input_classifier_node ───────────────────────────────────────────────

def input_classifier_node(state: AgentState) -> dict:
    """사용자 입력이 YouTube URL인지 일반 검색어인지 분류합니다."""
    text = state["user_input"]
    match = _YT_PATTERN.search(text)
    if match:
        return {"input_type": "youtube", "youtube_url": match.group(0)}

    resp = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                "다음 텍스트가 YouTube URL을 포함하면 'youtube', "
                "그 외 검색어면 'search'라고만 답하세요.\n\n" + text
            ),
        }],
    )
    label = resp.content[0].text.strip().lower()
    if "youtube" in label:
        return {"input_type": "youtube", "youtube_url": text.strip()}
    return {"input_type": "search"}


# ── 3. youtube_analyzer_node ───────────────────────────────────────────────

def youtube_analyzer_node(state: AgentState) -> dict:
    """
    비디오 메타데이터와 자막을 추출하고,
    Claude로 내용을 요약한 뒤 썸네일 이미지 프롬프트를 생성합니다.
    """
    url = state.get("youtube_url", "")

    # 1) 메타데이터 수집
    try:
        info = get_youtube_info(url)
    except Exception as e:
        return {"video_info": {}, "thumbnail_mode": "download", "error": str(e)}

    # 2) 자막은 get_youtube_info에서 이미 추출됨
    transcript = info.get("transcript", info.get("description", ""))

    # 3) 썸네일 모드 결정
    force_mode = os.getenv("THUMBNAIL_MODE", "").lower()
    if force_mode not in ("ai_generate", "download"):
        force_mode = "ai_generate"
    thumbnail_mode = force_mode

    # 4) Claude로 영상 요약 + 이미지 프롬프트 생성
    history_ctx = _format_history(state.get("conversation_history", []))
    transcript_text = transcript[:1500] if transcript else "자막 없음"

    resp = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"{history_ctx}"
                f"유튜브 비디오 정보:\n"
                f"제목: {info['title']}\n"
                f"채널: {info['channel']}\n"
                f"설명: {info['description'][:300]}\n"
                f"태그: {', '.join(info.get('tags', []))}\n\n"
                f"자막/트랜스크립트:\n{transcript_text}\n\n"
                "위 영상의 내용을 바탕으로 다음 두 가지를 제공해주세요:\n"
                "1. SUMMARY: 영상 내용 한국어 요약 (2-3문장)\n"
                "2. IMAGE_PROMPT: 영상 내용을 시각적으로 표현하는 영문 이미지 생성 프롬프트 "
                "(DALL-E용, 구체적이고 시각적인 묘사, 텍스트 포함 금지, 16:9 비율)\n\n"
                "정확히 아래 형식으로 답변:\n"
                "SUMMARY: ...\n"
                "IMAGE_PROMPT: ..."
            ),
        }],
    )

    text = resp.content[0].text
    summary = ""
    image_prompt = ""

    for line in text.splitlines():
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("IMAGE_PROMPT:"):
            image_prompt = line.replace("IMAGE_PROMPT:", "").strip()

    if not image_prompt:
        image_prompt = (
            f"A visually striking YouTube thumbnail illustration about '{info['title']}', "
            f"professional digital art, vibrant colors, cinematic composition, 16:9 aspect ratio"
        )

    info["summary"] = summary

    return {
        "video_info": info,
        "thumbnail_mode": thumbnail_mode,
        "image_prompt": image_prompt,
    }


# ── 4. thumbnail_style_gen_node (Send API 병렬 실행) ───────────────────────

def thumbnail_style_gen_node(state: AgentState) -> dict:
    """특정 스타일로 썸네일을 생성합니다. Send API로 병렬 호출됩니다."""
    info = state.get("video_info", {})
    style = state.get("thumbnail_style", "default")
    prompt = state.get("image_prompt", "")
    title = info.get("title", "thumbnail")
    filename = f"ai_{style}_{abs(hash(title)) % 100000}.png"

    try:
        path = generate_image(prompt, filename)
        return {"thumbnail_results": [{"style": style, "path": path, "prompt": prompt}]}
    except Exception as e:
        return {"thumbnail_results": [{"style": style, "path": None, "error": str(e)}]}


# ── 5. thumbnail_selector_node ─────────────────────────────────────────────

def thumbnail_selector_node(state: AgentState) -> dict:
    """병렬 생성된 썸네일 중 최적의 결과를 선택합니다."""
    results = state.get("thumbnail_results", [])
    info = state.get("video_info", {})
    title = info.get("title", "")
    summary = info.get("summary", "")

    successful = [r for r in results if r.get("path")]
    if not successful:
        return {"error": "모든 썸네일 생성 실패",
                "final_response": "썸네일 생성에 실패했습니다."}

    primary = successful[0]
    summary_text = f"\n요약: {summary}" if summary else ""

    return {
        "thumbnail_path": primary["path"],
        "final_response": (
            f"AI 썸네일 {len(successful)}개 스타일 병렬 생성 완료!\n"
            f"비디오: {title}{summary_text}\n"
            + "\n".join(f"  [{r['style']}] {r['path']}" for r in successful)
        ),
    }


# ── 6. download_thumbnail_node ─────────────────────────────────────────────

def download_thumbnail_node(state: AgentState) -> dict:
    """기존 YouTube 썸네일을 다운로드합니다."""
    info = state.get("video_info", {})
    url = info.get("thumbnail", "")
    title = info.get("title", "video")

    if not url:
        return {"error": "썸네일 URL 없음",
                "final_response": "썸네일 URL을 찾을 수 없습니다."}

    filename = f"yt_{abs(hash(title)) % 100000}.jpg"
    try:
        path = download_image(url, filename)
        return {
            "thumbnail_path": path,
            "final_response": (
                f"YouTube 썸네일 다운로드 완료!\n"
                f"비디오: {title}\n"
                f"저장 위치: {path}"
            ),
        }
    except Exception as e:
        return {"error": str(e),
                "final_response": f"썸네일 다운로드 실패: {e}"}


# ── 7. web_search_node ─────────────────────────────────────────────────────

def web_search_node(state: AgentState) -> dict:
    """DuckDuckGo로 검색하고 Claude로 요약합니다."""
    query = state["user_input"]
    history_ctx = _format_history(state.get("conversation_history", []))

    try:
        results = web_search(query, max_results=5)
        context = "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')[:200]}"
            for r in results
        )
        resp = _get_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": (
                    f"{history_ctx}"
                    f"검색어: {query}\n\n"
                    f"검색 결과:\n{context}\n\n"
                    "위 내용을 한국어로 간결하게 요약해주세요."
                ),
            }],
        )
        return {
            "search_results": results,
            "final_response": resp.content[0].text,
        }
    except Exception as e:
        return {"error": str(e), "final_response": f"검색 실패: {e}"}


# ── 8. memory_save_node ────────────────────────────────────────────────────

def memory_save_node(state: AgentState) -> dict:
    """현재 대화를 메모리에 저장합니다."""
    entry = {
        "input": state.get("user_input", ""),
        "input_type": state.get("input_type", ""),
        "video_title": (state.get("video_info") or {}).get("title", ""),
        "summary": (state.get("video_info") or {}).get("summary", ""),
        "thumbnail_path": state.get("thumbnail_path"),
        "had_error": bool(state.get("error")),
    }
    save_memory(entry)
    return {}


# ── 9. result_node ─────────────────────────────────────────────────────────

def result_node(state: AgentState) -> dict:
    """최종 결과를 정리하고, 요약본이 있으면 파일로 저장합니다."""
    from datetime import datetime

    history = state.get("conversation_history", [])
    history_note = ""
    if history:
        history_note = f"\n(메모리: 이전 {len(history)}건의 대화 기록 참조됨)"

    if state.get("error") and not state.get("thumbnail_path"):
        return {"final_response": f"오류: {state['error']}{history_note}"}

    # 요약본 파일 저장
    info = state.get("video_info") or {}
    summary = info.get("summary", "")
    summary_path = ""
    if summary and info.get("title"):
        os.makedirs("summaries", exist_ok=True)
        safe_title = re.sub(r'[^\w\s-]', '', info["title"])[:50].strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"summaries/{timestamp}_{safe_title}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"제목: {info['title']}\n")
            f.write(f"채널: {info.get('channel', '')}\n")
            f.write(f"URL: {state.get('youtube_url', '')}\n")
            f.write(f"날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'─' * 50}\n\n")
            f.write(f"요약:\n{summary}\n")
            transcript = info.get("transcript", "")
            if transcript:
                f.write(f"\n{'─' * 50}\n\n")
                f.write(f"자막 원문:\n{transcript}\n")

    response = state.get("final_response", "완료")
    if summary_path:
        response += f"\n요약 저장: {summary_path}"
    return {"final_response": response + history_note}
