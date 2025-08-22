import os

from fastapi import FastAPI, Header
from service.langgraph.state import ChatRequest, ChatResponse, GraphState
from service.langgraph.node import Node
from service.langgraph.utils import save_conversation
from service.langgraph.LangGraph import LangRAGGraph  # 위 파일
import uvicorn
from langsmith import traceable, tracing_context
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="LangGraph RAG API", version="1.0.0")

# 그래프 준비(서버 기동 시 1회)
node = Node(retrieve_k=15, top_k=3)
graph = LangRAGGraph(node=node)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
@traceable(name="Chat API", run_type="chain", tags=["api","rag"])
def chat(req: ChatRequest,
        langsmith_trace: str | None = Header(default=None),  # 분산 트레이싱용(선택)
        baggage: str | None = Header(default=None),):
    """
    LangRAGGraph.invoke 호출만으로 전체 파이프라인이 실행됨.
    """
    init = GraphState(client_id=req.client_id, query=req.query)

    # 프로젝트/태그/메타데이터 + (있다면) 클라이언트에서 넘어온 부모 트레이스 이어붙이기
    with tracing_context(
        project_name="LangRAG",
        tags=[f"client:{req.client_id}"],
        metadata={"retrieve_k": node.retrieve_k, "top_k": node.top_k},
        parent={"langsmith-trace": langsmith_trace, "baggage": baggage} if langsmith_trace else None,
        enabled=True,   # env 없이 강제 켜고 싶을 때
    ):
        out: GraphState = graph.invoke(init)

    # 로그 저장
    save_conversation(req.client_id, req.query, out.answer or "")

    return ChatResponse(client_id=req.client_id, answer=out.answer or "")

if __name__ == "__main__":
    ip_name = os.getenv("ALLOWED_HOST")
    uvicorn.run("service.main:app", host=ip_name, port=20000, reload=True)