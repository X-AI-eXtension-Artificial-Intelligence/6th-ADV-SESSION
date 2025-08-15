# utils.py
import os
import pickle
import pandas as pd
import faiss
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from service.langgraph.prompt import SYSTEM_LAWQA, build_user_lawqa

load_dotenv()

# ==== .env에서 읽어올 설정 ====
RETRIEVE_MODEL_PATH = os.getenv("FAISS_MODEL_PATH")
CROSS_ENCODER_PATH  = os.getenv("RERANK_MODEL_PATH")
FAISS_INDEX_PATH    = os.getenv("FAISS_INDEX_PATH")
LAW_IDS_PKL_PATH    = os.getenv("FAISS_PKL_PATH")
LAW_DF_PATH         = os.getenv("CHUNKING_OUTPUT_CSV")
API_KEY             = os.getenv("API_KEY")
LLM_MODEL           = "gpt-4o-mini"
CONV_DIR = os.getenv("CONV_DIR", "./conversations")
os.makedirs(CONV_DIR, exist_ok=True)


# ==== 모듈 전역 캐시 ====
_bi_encoder: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None
_faiss_index: Optional[faiss.Index] = None
_law_ids: Optional[List[int]] = None
_law_df: Optional[pd.DataFrame] = None


# ---------------- 내부 유틸 ----------------
def _ensure_loaded():
    """모델, 인덱스, 데이터셋을 1회 로드"""
    global _bi_encoder, _cross_encoder, _faiss_index, _law_ids, _law_df
    if _bi_encoder is None:
        _bi_encoder = SentenceTransformer(RETRIEVE_MODEL_PATH)
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)
    if _faiss_index is None:
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    if _law_ids is None:
        with open(LAW_IDS_PKL_PATH, "rb") as f:
            _law_ids = pickle.load(f)
    if _law_df is None:
        _law_df = pd.read_csv(LAW_DF_PATH)


def _conversation_path(client_id: str) -> str:
    """client_id 기반 CSV 경로 (간단한 sanitize 포함)"""
    safe_id = "".join(c for c in client_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(CONV_DIR, f"{safe_id}.csv")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _to_text_list(docs: List[Any], max_docs: int = 10, max_each_chars: int = 1200) -> List[str]:
    """retrieved_docs를 문자열 리스트로 변환"""
    out: List[str] = []
    for d in docs[:max_docs]:
        if isinstance(d, str):
            text = d
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            text = str(d[1])
        elif isinstance(d, dict):
            text = str(d.get("text") or d.get("page_content") or d.get("content") or d.get("법령_텍스트") or d)
        else:
            text = str(d)
        out.append(text[:max_each_chars])
    return out


def _conversation_to_text(conv: Union[str, List[Dict[str, Any]]]) -> str:
    """conversation을 role: content 형태로 합침"""
    if isinstance(conv, str):
        return conv
    if isinstance(conv, list):
        lines = []
        for t in conv:
            role = str(t.get("role", "user")).upper()
            content = str(t.get("content", ""))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    return ""


def _call_openai(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """OpenAI Chat API 호출 후 텍스트 반환"""
    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------- 외부 공개 함수 ----------------
def use_RAG(query: str) -> bool:
    """법률 지식 필요 여부 판단"""
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 라우팅 판단기입니다. "
                "아래 질의가 법률 지식(조문/판례/조항 번호 등) 없이 "
                "답변 가능한지 판단하세요.\n"
                "- 법률 지식이 필요하면 true\n"
                "- 필요 없으면 false\n"
                "출력은 반드시 true 또는 false"
            ),
        },
        {"role": "user", "content": f"질의: {query}\n\n출력: true | false"},
    ]
    return _call_openai(messages, temperature=0.0).lower() == "true"


def has_history(client_id: str, query: str, conversation: Optional[List[Dict[str, Any]]] = None) -> bool:
    """과거 대화만으로 답변 가능 여부 판단"""
    if conversation is None:
        conversation = get_conversation(client_id)
    if not conversation:
        return False

    history_str = _conversation_to_text(conversation)
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 대화 기록 기반 라우팅 판단기입니다. "
                "현재 질의가 과거 대화 기록만으로 "
                "정확히 답변 가능하거나 사실상 동일 질문이 있는지 판단하세요.\n"
                "- 가능하면 true\n"
                "- 아니면 false\n"
                "틀린 정보는 그대로 사용하지 마세요."
            ),
        },
        {
            "role": "user",
            "content": f"현재 질의:\n{query}\n\n과거 대화:\n{history_str}\n\n출력: true | false",
        },
    ]
    return _call_openai(messages, temperature=0.0).lower() == "true"


def query_rewrite():
    return None


def retrieve_and_rerank(query: str, retrieve_k: int = 15, top_k: int = 3) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """retrieve_k개 검색 → CrossEncoder로 rerank → top_k 반환"""
    _ensure_loaded()
    q_emb = _bi_encoder.encode([query], convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    q_emb = _l2_normalize(q_emb)

    D, I = _faiss_index.search(q_emb, retrieve_k)
    faiss_indices = I[0].tolist()

    candidates = []
    for idx in faiss_indices:
        row = _law_df.iloc[_law_ids[idx]]
        candidates.append((str(row["법령_조문"]), str(row["법령_텍스트"])))

    ce_inputs = [[query, passage] for (_title, passage) in candidates]
    scores = _cross_encoder.predict(ce_inputs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return candidates[:top_k], [item[0] for item in ranked[:top_k]]


def generate(payload: Dict[str, Any], key: str) -> str:
    """질문 유형(key)에 따라 답변 생성"""
    query = payload.get("query", "")
    if not query:
        raise ValueError("payload에 query가 없습니다.")

    if key == "not_RAG":
        system = "법률 인용 없이 일반 상식과 논리로 정확히 답하세요."
        user = f"질의:\n{query}"
        return _call_openai([{"role": "system", "content": system},
                             {"role": "user", "content": user}], temperature=0.2)

    elif key == "has_history":
        conv = payload.get("conversation_text") or payload.get("conversation") or ""
        conv_text = _conversation_to_text(conv)
        system = "과거 대화를 참고하되, 사실과 논리를 우선하여 답변하세요."
        user = f"현재 질의:\n{query}\n\n과거 대화:\n{conv_text}"
        return _call_openai([{"role": "system", "content": system},
                             {"role": "user", "content": user}], temperature=0.2)

    elif key == "generate":
        docs = payload.get("retrieved_docs") or []
        doc_texts = _to_text_list(docs)
        context = "\n\n".join(f"- {t}" for t in doc_texts) if doc_texts else "(컨텍스트 없음)"

        system = SYSTEM_LAWQA
        user = build_user_lawqa(query=query, context=context)

        return _call_openai(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.1
        )

    else:
        raise ValueError(f"알 수 없는 key: {key}")


def save_conversation(client_id: str, query: str, answer: str) -> None:
    """pandas를 사용하여 role/content 저장"""
    path = _conversation_path(client_id)
    new_rows = pd.DataFrame([
        {"role": "user", "content": query},
        {"role": "system", "content": answer}
    ])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, new_rows], ignore_index=True)
    else:
        df = new_rows
    df.to_csv(path, index=False, encoding="utf-8")


def get_conversation(client_id: str) -> Optional[List[Dict[str, Any]]]:
    """pandas로 읽어 리스트[dict] 반환"""
    path = _conversation_path(client_id)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.to_dict(orient="records")