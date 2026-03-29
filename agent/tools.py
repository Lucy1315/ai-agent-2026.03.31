"""
Tools:
  1. get_youtube_info       - yt-dlp으로 비디오 메타데이터 및 썸네일 URL 추출
  2. get_youtube_transcript - yt-dlp으로 비디오 자막/트랜스크립트 추출
  3. web_search             - DuckDuckGo 웹 검색
  4. generate_image         - AI 이미지 생성 (Replicate → Stability → OpenAI DALL-E → PIL 폴백)
  5. download_image         - URL에서 이미지 다운로드
  6. load_memory / save_memory - 대화 기록 저장/로드
"""

import os
import json
import base64
import textwrap
from datetime import datetime
import requests
import yt_dlp
from duckduckgo_search import DDGS

MEMORY_FILE = "memory.json"


def get_youtube_info(url: str) -> dict:
    """YouTube 비디오 메타데이터, 썸네일 URL, 자막을 한 번에 추출합니다."""
    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # 자막 추출
    transcript = ""
    for source in [info.get("subtitles", {}), info.get("automatic_captions", {})]:
        if transcript:
            break
        for lang in ["ko", "en"]:
            if lang not in source:
                continue
            for fmt in source[lang]:
                if fmt.get("ext") == "json3":
                    try:
                        resp = requests.get(fmt["url"], timeout=30)
                        data = resp.json()
                        texts = []
                        for event in data.get("events", []):
                            for seg in event.get("segs", []):
                                text = seg.get("utf8", "").strip()
                                if text and text != "\n":
                                    texts.append(text)
                        transcript = " ".join(texts)[:3000]
                        if transcript:
                            break
                    except Exception:
                        continue
            if transcript:
                break

    if not transcript:
        transcript = (info.get("description") or "")[:1500]

    return {
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:500],
        "channel": info.get("channel", ""),
        "tags": (info.get("tags") or [])[:10],
        "thumbnail": info.get("thumbnail", ""),
        "view_count": info.get("view_count", 0),
        "duration": info.get("duration", 0),
        "transcript": transcript,
    }


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo로 웹 검색 결과를 반환합니다."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def generate_image(prompt: str, filename: str) -> str:
    """
    AI 이미지를 생성하고 저장 경로를 반환합니다.
    우선순위: Replicate → Stability AI → OpenAI DALL-E → PIL 폴백
    """
    os.makedirs("thumbnails", exist_ok=True)
    filepath = f"thumbnails/{filename}"

    if os.getenv("REPLICATE_API_TOKEN"):
        try:
            return _replicate(prompt, filepath)
        except Exception:
            pass

    if os.getenv("STABILITY_API_KEY"):
        try:
            return _stability(prompt, filepath)
        except Exception:
            pass

    if os.getenv("OPENAI_API_KEY"):
        try:
            return _openai_dalle(prompt, filepath)
        except Exception:
            pass

    return _pil_fallback(prompt, filepath)


def download_image(url: str, filename: str) -> str:
    """URL에서 이미지를 다운로드하고 저장 경로를 반환합니다."""
    os.makedirs("thumbnails", exist_ok=True)
    filepath = f"thumbnails/{filename}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(resp.content)
    return filepath


# ── 메모리 (대화 기록) ────────────────────────────────────────────────────────

def load_memory(max_entries: int = 10) -> list:
    """이전 대화 기록을 로드합니다."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[-max_entries:]
    except Exception:
        return []


def save_memory(entry: dict) -> None:
    """대화 기록을 저장합니다."""
    history = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
    entry["timestamp"] = datetime.now().isoformat()
    history.append(entry)
    history = history[-50:]  # 최근 50개만 유지
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── 이미지 생성 백엔드 ────────────────────────────────────────────────────────

def _replicate(prompt: str, filepath: str) -> str:
    import replicate
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt, "num_inference_steps": 4, "aspect_ratio": "16:9"},
    )
    with open(filepath, "wb") as f:
        f.write(output[0].read())
    return filepath


def _stability(prompt: str, filepath: str) -> str:
    api_key = os.getenv("STABILITY_API_KEY")
    resp = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": 7,
            "width": 1280,
            "height": 720,
            "samples": 1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    img_bytes = base64.b64decode(resp.json()["artifacts"][0]["base64"])
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return filepath


def _openai_dalle(prompt: str, filepath: str) -> str:
    """OpenAI DALL-E 3로 이미지를 생성합니다."""
    from openai import OpenAI
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt[:4000],
        size="1792x1024",  # 16:9에 가장 가까운 옵션
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(img_resp.content)
    return filepath


def _pil_fallback(prompt: str, filepath: str) -> str:
    """API 키가 없을 때 PIL로 텍스트 기반 썸네일을 생성합니다."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1280, 720), color=(15, 15, 30))
    draw = ImageDraw.Draw(img)

    # 그라디언트 배경
    for y in range(720):
        r = int(15 + (y / 720) * 40)
        g = int(15 + (y / 720) * 20)
        b = int(30 + (y / 720) * 60)
        draw.line([(0, y), (1280, y)], fill=(r, g, b))

    # 장식 사각형
    draw.rectangle([40, 40, 1240, 680], outline=(80, 120, 220), width=3)
    draw.rectangle([60, 60, 1220, 660], outline=(40, 80, 180), width=1)

    font_large = _load_font(64)
    font_small = _load_font(32)

    lines = textwrap.wrap(prompt[:120], width=28)
    total_h = len(lines) * 80
    y = (720 - total_h) // 2

    for line in lines[:5]:
        draw.text((640, y), line, fill=(240, 240, 255), font=font_large, anchor="mm")
        y += 80

    draw.text((640, 650), "AI Generated Thumbnail", fill=(120, 140, 200),
              font=font_small, anchor="mm")

    if not filepath.endswith(".png"):
        filepath = filepath.rsplit(".", 1)[0] + ".png"
    img.save(filepath)
    return filepath


def _load_font(size: int):
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()
