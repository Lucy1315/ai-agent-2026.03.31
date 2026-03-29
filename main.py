"""
YouTube Thumbnail Generator Agent
사용법:
  python main.py "https://www.youtube.com/watch?v=..."   # 썸네일 생성
  python main.py "LangGraph tutorial"                     # 웹 검색
  python main.py                                          # 대화형 모드

실행 전 .env 파일에 ANTHROPIC_API_KEY를 설정하세요.
"""

import sys
from dotenv import load_dotenv
from agent.graph import build_graph
from agent.state import AgentState

load_dotenv()


def run(user_input: str) -> AgentState:
    graph = build_graph()

    initial: AgentState = {
        "user_input": user_input,
        "input_type": "",
        "youtube_url": None,
        "video_info": None,
        "thumbnail_mode": "download",
        "thumbnail_style": "",
        "thumbnail_path": None,
        "thumbnail_results": [],
        "image_prompt": None,
        "search_results": None,
        "conversation_history": [],
        "final_response": "",
        "error": None,
    }

    print(f"\n입력: {user_input}\n" + "─" * 50)

    # stream(mode="updates")로 노드별 진행 상황을 출력하면서 최종 상태를 축적합니다
    final_state = dict(initial)
    for chunk in graph.stream(initial, stream_mode="updates"):
        for node_name, updates in chunk.items():
            print(f"  ✓ {node_name}")
            if updates:
                final_state.update(updates)

    print("\n" + "=" * 50)
    print(final_state.get("final_response", "완료"))
    if final_state.get("thumbnail_path"):
        print(f"\n썸네일 저장: {final_state['thumbnail_path']}")
    print("=" * 50)

    return final_state


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        print("YouTube Thumbnail Generator Agent")
        print("YouTube URL 또는 검색어를 입력하세요. (종료: q)\n")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit"):
                break
            run(user_input)
