from fastapi import FastAPI
from service.langgraph.state import ChatRequest, ChatResponse, GraphState
from service.langgraph.node import Node
from service.langgraph.utils import save_conversation
from service.langgraph.LangGraph import LangRAGGraph  # 위 파일
import uvicorn

app = FastAPI(title="LangGraph RAG API", version="1.0.0")

# 그래프 준비(서버 기동 시 1회)
node = Node(retrieve_k=15, top_k=3)
graph = LangRAGGraph(node=node)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    LangRAGGraph.invoke 호출만으로 전체 파이프라인이 실행됨.
    """
    init = GraphState(client_id=req.client_id, query=req.query)
    out: GraphState = graph.invoke(init)

    # 로그 저장
    save_conversation(req.client_id, req.query, out.answer or "")

    return ChatResponse(client_id=req.client_id, answer=out.answer or "")

if __name__ == "__main__":
    uvicorn.run("service.main:app", host="0.0.0.0", port=20000, reload=True)