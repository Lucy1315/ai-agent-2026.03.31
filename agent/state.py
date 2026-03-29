import operator
from typing import TypedDict, Optional, Annotated


class AgentState(TypedDict):
    user_input: str
    input_type: str              # "youtube" | "search"
    youtube_url: Optional[str]
    video_info: Optional[dict]
    thumbnail_mode: str          # "ai_generate" | "download"
    thumbnail_style: str         # 병렬 생성 시 스타일 이름
    thumbnail_path: Optional[str]
    thumbnail_results: Annotated[list, operator.add]  # Send API 병렬 결과 리듀서
    image_prompt: Optional[str]
    search_results: Optional[list]
    conversation_history: list   # 메모리: 이전 대화 기록
    final_response: str
    error: Optional[str]
