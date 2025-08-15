from typing import Any, Dict

from langgraph.graph import StateGraph, END

from service.langgraph.state import GraphState
from service.langgraph.node import Node


class LangRAGGraph:
    """
    LangGraph 파이프라인을 클래스 형태로 래핑.
    - 생성 시 그래프를 빌드/컴파일
    - invoke(state) 한 번으로 전체 파이프라인 수행
    """
    def __init__(self, node: Node) -> None:
        self.node = node
        self.graph = self._build().compile()

    # ----------------------
    # public
    # ----------------------
    def invoke(self, state: GraphState) -> GraphState:
        """GraphState -> GraphState"""
        init_dict = self._to_dict(state)
        out_dict = self.graph.invoke(init_dict)
        return self._to_state(out_dict)

    def run(self, client_id: str, query: str) -> GraphState:
        """편의 함수: client_id, query로 실행"""
        return self.invoke(GraphState(client_id=client_id, query=query))

    # ----------------------
    # internal
    # ----------------------
    def _to_state(self, d: Dict[str, Any]) -> GraphState:
        return GraphState(**d)

    def _to_dict(self, s: GraphState) -> Dict[str, Any]:
        d = s.model_dump()
        return {k: v for k, v in d.items() if v is not None}

    def _build(self) -> StateGraph:
        """add_node / add_edge 로 그래프 구성"""
        g = StateGraph(dict)

        # ---- node wrappers (GraphState <-> dict) ----
        def use_RAG_node(s: Dict[str, Any]) -> Dict[str, Any]:
            st = self._to_state(s)
            need_rag = self.node.use_RAG(st)  # bool
            if not need_rag:
                st2 = self.node.answer_RAG(st)
                return self._to_dict(st2)
            return s

        def has_history_node(s: Dict[str, Any]) -> Dict[str, Any]:
            st = self._to_state(s)
            ok = self.node.has_history(st)  # bool
            if ok:
                st2 = self.node.answer_history(st)
                return self._to_dict(st2)
            return s

        def query_rewrite_node(s: Dict[str, Any]) -> Dict[str, Any]:
            st = self._to_state(s)
            st2 = self.node.query_rewrite(st)  # 현재 noop
            return self._to_dict(st2)

        def retrieve_and_rerank_node(s: Dict[str, Any]) -> Dict[str, Any]:
            st = self._to_state(s)
            st2 = self.node.retrieve_and_rerank(st)
            return self._to_dict(st2)

        def generate_node(s: Dict[str, Any]) -> Dict[str, Any]:
            st = self._to_state(s)
            st2 = self.node.generate(st)
            return self._to_dict(st2)

        # ---- routing functions ----
        def route_after_use_RAG(s: Dict[str, Any]) -> str:
            return "end" if s.get("answer") else "continue"

        def route_after_has_history(s: Dict[str, Any]) -> str:
            return "end" if s.get("answer") else "continue"

        # ---- build graph ----
        g.add_node("use_RAG", use_RAG_node)
        g.add_node("has_history", has_history_node)
        g.add_node("query_rewrite", query_rewrite_node)
        g.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
        g.add_node("generate", generate_node)

        g.set_entry_point("use_RAG")

        g.add_conditional_edges(
            "use_RAG",
            route_after_use_RAG,
            {"continue": "has_history", "end": END},
        )
        g.add_conditional_edges(
            "has_history",
            route_after_has_history,
            {"continue": "query_rewrite", "end": END},
        )
        g.add_edge("query_rewrite", "retrieve_and_rerank")
        g.add_edge("retrieve_and_rerank", "generate")
        g.add_edge("generate", END)

        return g
