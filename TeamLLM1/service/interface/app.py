import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()
ip_name = os.getenv("ALLOWED_HOST")
API_URL = f"http://{ip_name}:20000/chat"

st.set_page_config(page_title="Legal RAG Chatbot", page_icon="⚖️", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("⚖️ 법률 관련 고민을 말씀해주세요")

# 사용자 입력
user_msg = st.chat_input("법률 관련 질문을 입력하세요…")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    # FastAPI 호출
    try:
        payload = {"client_id": "user123", "query": user_msg}
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "❌ 응답 없음")
    except Exception as e:
        answer = f"FastAPI 연결 실패: {e}"

    # 답변도 세션에 추가
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 세션 상태 전체를 한 번만 렌더링
for msg in st.session_state.chat_history:
    if msg.get("_rendered"):  # 이미 출력된 메시지면 스킵
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    msg["_rendered"] = True
