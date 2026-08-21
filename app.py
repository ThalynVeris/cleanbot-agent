from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str) -> Any:
    with httpx.Client(timeout=8) as client:
        response = client.get(f"{API_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


def load_session_messages(session_id: str) -> list[dict[str, Any]]:
    stored_messages = api_get(f"/api/v1/sessions/{session_id}/messages")

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "sources": message.get("sources", []),
        }
        for message in stored_messages
        if message["role"] in {"user", "assistant"}
    ]


def start_new_session(selector_key: str) -> None:
    new_session_id = str(uuid.uuid4())

    st.session_state.session_id = new_session_id
    st.session_state.messages = []
    st.session_state[selector_key] = new_session_id


def show_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"参考资料（{len(sources)} 条）"):
        for index, source in enumerate(sources, start=1):
            location = f" · 第 {source['page']} 页" if source.get("page") else ""
            st.markdown(f"**来源 {index}：{source.get('source', 'unknown')}{location}**")
            if source.get("section"):
                st.caption(source["section"])
            st.write(source.get("excerpt", ""))


def stream_answer(payload: dict[str, Any], sources: list[dict[str, Any]], status_box) -> Iterator[str]:
    event_type = ""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{API_BASE_URL}/api/v1/chat/stream", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    event_type = ""
                    continue
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = json.loads(line.split(":", 1)[1].strip())
                if event_type == "token":
                    yield data.get("text", "")
                elif event_type == "source":
                    sources.append(data)
                elif event_type == "status":
                    status_box.caption(data.get("message", "处理中"))
                elif event_type == "error":
                    raise RuntimeError(data.get("message", "服务暂时不可用"))
                elif event_type == "done":
                    first_token_ms = float(data.get("first_token_ms", 0) or 0)
                    latency_ms = float(data.get("latency_ms", 0) or 0)
                    answer_mode = "模型生成" if data.get("model_called") else "固定回答"

                    parts = [
                        "完成",
                        str(data.get("intent", "unknown")),
                        f"首字 {first_token_ms:.0f} ms",
                        f"总计 {latency_ms:.0f} ms",
                        answer_mode,
                    ]

                    token_usage = data.get("token_usage")
                    if isinstance(token_usage, dict):
                        parts.append(f"Token {token_usage.get('total_tokens', 0)}")

                    status_box.caption(" · ".join(parts))


st.set_page_config(page_title="CleanBot Agent", page_icon="🤖", layout="centered")
st.title("🤖 CleanBot 智能客服")
st.caption("LangGraph · Hybrid RAG · 可追溯引用 · 演示数据，不代表真实设备账号")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    users = api_get("/api/v1/demo/users")
except Exception:
    st.error("无法连接 FastAPI 服务。请先运行：`uvicorn cleanbot.api.app:app --reload`。")
    st.stop()

if not users:
    st.error("没有演示用户。请先运行：`python -m cleanbot.cli init-db`。")
    st.stop()

with st.sidebar:
    st.header("演示上下文")

    labels = {f"{user['display_name']} · {user['city']}": user["id"] for user in users}
    selected_label = st.selectbox("用户", list(labels))
    user_id = labels[selected_label]

    user_sessions = api_get(f"/api/v1/demo/users/{user_id}/sessions")
    session_by_id = {session["id"]: session for session in user_sessions}

    selector_key = f"session_selector_{user_id}"
    user_changed = st.session_state.get("active_user_id") != user_id

    if user_changed:
        st.session_state.active_user_id = user_id

        remembered_session_id = st.session_state.get(selector_key)

        if remembered_session_id:
            st.session_state.session_id = remembered_session_id
        elif user_sessions:
            st.session_state.session_id = user_sessions[0]["id"]
        else:
            st.session_state.session_id = str(uuid.uuid4())

        if st.session_state.session_id in session_by_id:
            st.session_state.messages = load_session_messages(st.session_state.session_id)
        else:
            st.session_state.messages = []

        st.session_state[selector_key] = st.session_state.session_id
        st.rerun()

    session_ids = [session["id"] for session in user_sessions]

    if st.session_state.session_id not in session_ids:
        session_ids.insert(
            0,
            st.session_state.session_id,
        )

    session_labels = {
        session["id"]: (f"{session['title']} · {session['message_count']} 条消息")
        for session in user_sessions
    }

    if selector_key not in st.session_state:
        st.session_state[selector_key] = st.session_state.session_id

    selected_session_id = st.selectbox(
        "会话",
        session_ids,
        format_func=lambda session_id: session_labels.get(
            session_id,
            "新会话",
        ),
        key=selector_key,
    )

    if selected_session_id != st.session_state.session_id:
        st.session_state.session_id = selected_session_id

        if selected_session_id in session_by_id:
            st.session_state.messages = load_session_messages(selected_session_id)
        else:
            st.session_state.messages = []

        st.rerun()

    months = api_get(f"/api/v1/demo/users/{user_id}/months")
    selected_month = st.selectbox("报告月份", months) if months else None

    st.caption(f"会话 ID：{st.session_state.session_id[:8]}…")

    st.button(
        "开始新会话",
        on_click=start_new_session,
        args=(selector_key,),
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

prompt = st.chat_input("例如：主刷总被宠物毛发缠住怎么办？")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.empty()
        captured_sources: list[dict[str, Any]] = []
        try:
            answer = st.write_stream(
                stream_answer(
                    {
                        "session_id": st.session_state.session_id,
                        "user_id": user_id,
                        "message": prompt,
                        "month": selected_month,
                    },
                    captured_sources,
                    status_box,
                )
            )
            show_sources(captured_sources)
        except Exception as exc:
            answer = f"请求失败：{exc}"
            st.error(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": captured_sources}
        )
