from typing import List
from service.langgraph.state import GraphState
from service.langgraph.utils import (
    use_RAG as _use_RAG,
    has_history as _has_history,
    query_rewrite as _query_rewrite,
    retrieve_and_rerank as _retrieve_and_rerank,
    generate as _generate,
    get_conversation as _get_conversation,
)


class Node:
    """
    하나의 클래스에 노드 동작을 메서드로 모았다.
    - 모든 메서드는 GraphState -> GraphState
    - retrieve_k, top_k 등 파라미터는 생성자에서 설정
    """
    def __init__(self, retrieve_k: int = 20, top_k: int = 5):
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    # -------------------------
    # 노드 메서드 (state -> state)
    # -------------------------
    def use_RAG(self, state: GraphState) -> GraphState:
        """
        - LLM이 판단: 법률 지식 필요(False면 비-RAG 경로)
        - 필요 없으면 여기서 바로 not_RAG로 답변 생성하여 state.answer 채움
        - 필요하면 state 그대로 반환(다음 노드로 진행)
        """
        need_rag = _use_RAG(state.query)  # True=RAG 필요, False=불필요
        return need_rag
    
    
    def answer_RAG(self, state:GraphState) -> GraphState:
        answer = _generate({"query": state.query}, key="not_RAG")
        return GraphState(
                client_id=state.client_id,
                query=state.query,
                answer=answer,
            )


    def has_history(self, state: GraphState) -> GraphState:
        """
        - 과거 대화로만 답 가능/중복 여부 판단
        - 가능하면 여기서 바로 히스토리 기반 답변 생성하여 state.answer 채움
        - 아니면 state 그대로 반환(다음 노드로 진행)
        """
        conversation = _get_conversation(state.client_id)
        ok = _has_history(state.client_id, state.query, conversation)
        return ok
    

    def answer_history(self, state: GraphState) -> GraphState:
        conversation = _get_conversation(state.client_id)
        answer = _generate(
                {"query": state.query, "conversation": conversation},
                key="has_history",
            )
        return GraphState(
                client_id=state.client_id,
                query=state.query,
                retrieved_docs=state.retrieved_docs,
                answer=answer,
            )


    def query_rewrite(self, state: GraphState) -> GraphState:
        """
        - 현재 사양상 비활성(None 반환). 향후 리라이트 도입 시 state.query를 업데이트.
        """
        return state


    def retrieve_and_rerank(self, state: GraphState) -> GraphState:
        """
        - retrieve_k개 검색 → CrossEncoder 재순위 → top_k 문서의 '본문'만 state.retrieved_docs(List[str])에 저장
        """
        _, reranked_topk = _retrieve_and_rerank(
            state.query, retrieve_k=self.retrieve_k, top_k=self.top_k
        )
        # reranked_topk: List[Tuple[title, text]] 형태를 본문(text)으로 투영
        docs: List[str] = [t[1] for t in reranked_topk] if reranked_topk else []
        return GraphState(
            client_id=state.client_id,
            query=state.query,
            retrieved_docs=docs,
            answer=state.answer,
        )


    def generate(self, state: GraphState) -> GraphState:
        """
        - RAG 컨텍스트(retrieved_docs)를 사용해 최종 답변 생성
        """
        answer = _generate(
            {"query": state.query, "retrieved_docs": state.retrieved_docs or []},
            key="generate",
        )
        return GraphState(
            client_id=state.client_id,
            query=state.query,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
        )
