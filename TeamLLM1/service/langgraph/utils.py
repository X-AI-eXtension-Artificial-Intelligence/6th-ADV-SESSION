# utils.py
import os
import pickle
import pandas as pd
import faiss
import torch
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import time, hashlib, tempfile, shutil
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
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
    # if _faiss_index is None:
    #     _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    # if _law_ids is None:
    #     with open(LAW_IDS_PKL_PATH, "rb") as f:
    #         _law_ids = pickle.load(f)
    # if _law_df is None:
    #     _law_df = pd.read_csv(LAW_DF_PATH)


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

# 검색
def _search(query: str, k: int, handle: Dict[str, Any]):
    m = handle["method"]; docs = handle["docs"]
    if m == "bm25":
        tok = handle["tokenizer"]; bm25 = handle["bm25"]
        q_tokens = tok.tokenize(query)
        scores = bm25.get_scores(q_tokens)
        idx = np.argsort(scores)[::-1][:k]
        return [(docs[i], float(scores[i])) for i in idx]
    if m == "dense":
        model = handle["dense_model"]; emb = handle["docs_embeddings"]
        qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        sims = emb @ qv
        idx = np.argsort(sims)[::-1][:k]
        return [(docs[i], float(sims[i])) for i in idx]
    if m == "faiss":
        model = handle["dense_model"]; index = handle["faiss_index"]
        qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, k)
        return [(docs[int(i)], float(D[0, j])) for j, i in enumerate(I[0])]
    raise ValueError("unknown method in handle")
class PrebuiltIndexNotFound(Exception): ...
class ArtifactMissing(Exception): ...

def _safe_makedirs(p: str): os.makedirs(p, exist_ok=True)

def _atomic_write_bytes(dst: str, data: bytes):
    d = os.path.dirname(dst); _safe_makedirs(d)
    fd, tmp = tempfile.mkstemp(dir=d)
    try:
        with os.fdopen(fd, "wb", buffering=0) as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, dst)
    except Exception:
        try: os.remove(tmp)
        finally: raise

def _save_json(path: str, obj: Any):
    _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

def _load_json(path: str) -> Any:
    if not os.path.exists(path): raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def _save_numpy_atomic(path: str, arr: np.ndarray):
    d = os.path.dirname(path); _safe_makedirs(d)
    fd, tmp = tempfile.mkstemp(dir=d)
    try:
        with os.fdopen(fd, "wb", buffering=0) as f:
            np.save(f, arr); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        finally: raise

def _load_numpy(path: str) -> np.ndarray:
    if not os.path.exists(path): raise FileNotFoundError(path)
    return np.load(path)

def _save_faiss(path: str, index: faiss.Index):
    _safe_makedirs(os.path.dirname(path)); faiss.write_index(index, path)

def _load_faiss(path: str) -> faiss.Index:
    if not os.path.exists(path): raise FileNotFoundError(path)
    return faiss.read_index(path)

# 코퍼스 키: 문서+instruction만 반영
def _hash_corpus(docs: List[str], instruction: bool) -> str:
    h = hashlib.sha256()
    h.update(b"inst1" if instruction else b"inst0")
    for d in docs:
        h.update(b"\x1e"); h.update(d.encode("utf-8"))
    return h.hexdigest()[:16]

def _paths_corpus(store_dir: str, corpus_key: str) -> Dict[str, str]:
    root = os.path.join(store_dir, corpus_key)
    return {
        "root": root,
        "corpus": os.path.join(root, "corpus.json"),
        "corpus_config": os.path.join(root, "corpus_config.json"),
        "bm25_tokens": os.path.join(root, "bm25_tokens.json"),
        "variants_dir": os.path.join(root, "variants"),
    }

def _paths_variant(store_dir: str, corpus_key: str, variant: str) -> Dict[str, str]:
    vd = os.path.join(store_dir, corpus_key, "variants", variant)
    return {
        "dir": vd,
        "config": os.path.join(vd, "config.json"),
        "dense_npy": os.path.join(vd, "dense_emb.npy"),
        "faiss_index": os.path.join(vd, "faiss.index"),
    }

# 임베딩 저장
def build_and_save_index(
    *,
    docs: List[str],
    instruction: bool,
    variant: str,                 # ex) "bge-m3" | "sbert" | "other"
    embed_model_name: str,        # ex) "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2", .
    method: str,                  # "dense" | "faiss" | "bm25"
    store_dir: str = "data/vector_db",
    corpus_key: Optional[str] = None,   # 미지정 시 자동 생성
    overwrite_variant: bool = False,    # 같은 variant 덮어쓸지
    build_bm25_once: bool = False,      # BM25 토큰도 함께 만들고 싶으면 True
) -> str:
    # instruction 프리픽스 적용
    _docs = [f"Document: {d}" for d in docs] if instruction else list(docs)
    corpus_key = corpus_key or _hash_corpus(_docs, instruction)

    PC = _paths_corpus(store_dir, corpus_key)
    PV = _paths_variant(store_dir, corpus_key, variant)

    # 코퍼스 메타/문서 저장(없으면)
    _safe_makedirs(PC["root"])
    if not os.path.exists(PC["corpus"]):
        _save_json(PC["corpus"], _docs)
        _save_json(PC["corpus_config"], {
            "instruction": instruction,
            "created_at": int(time.time()),
            "version": 1
        })

    # BM25 토큰 생성(옵션, 중복 방지)
    if build_bm25_once and not os.path.exists(PC["bm25_tokens"]):
        tok = AutoTokenizer.from_pretrained(embed_model_name)
        tokenized = [tok.tokenize(d) for d in _docs]
        _save_json(PC["bm25_tokens"], tokenized)

    # 변종 생성
    if (os.path.exists(PV["config"]) or os.path.exists(PV["dense_npy"]) or os.path.exists(PV["faiss_index"])) and not overwrite_variant:
        return corpus_key  # 이미 있음

    _safe_makedirs(PV["dir"])

    if method == "bm25":
        # 변종에 별도 파일은 없음(코퍼스 공용 bm25_tokens 사용)
        _save_json(PV["config"], {
            "variant": variant, "embed_model_name": embed_model_name,
            "method": "bm25", "created_at": int(time.time())
        })
        return corpus_key

    if method not in {"dense", "faiss"}:
        raise ValueError("method must be one of {'bm25','dense','faiss'}")

    model = SentenceTransformer(embed_model_name)
    emb = model.encode(_docs, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _save_numpy_atomic(PV["dense_npy"], emb)

    if method == "faiss":
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        _save_faiss(PV["faiss_index"], index)

    _save_json(PV["config"], {
        "variant": variant,
        "embed_model_name": embed_model_name,
        "method": method,
        "dim": int(emb.shape[1]) if method in {"dense","faiss"} else None,
        "created_at": int(time.time())
    })
    return corpus_key

def _load_index(
    *,
    store_dir: str,
    corpus_key: str,
    variant: str,
    use_gpu_for_faiss: bool = True
) -> Dict[str, Any]:
    PC = _paths_corpus(store_dir, corpus_key)
    PV = _paths_variant(store_dir, corpus_key, variant)
    if not os.path.exists(PC["corpus"]):
        raise PrebuiltIndexNotFound(f"Corpus not found: {PC['root']}")

    docs = _load_json(PC["corpus"])
    if not os.path.exists(PV["config"]):
        raise PrebuiltIndexNotFound(f"Variant not found: {PV['dir']}")

    vcfg = _load_json(PV["config"])
    method = vcfg["method"]

    if method == "bm25":
        if not os.path.exists(PC["bm25_tokens"]):
            raise ArtifactMissing("bm25_tokens.json missing for corpus")
        tokens = _load_json(PC["bm25_tokens"])
        tokenizer = AutoTokenizer.from_pretrained(vcfg["embed_model_name"])
        bm25 = BM25Okapi(tokens)
        return {"method": "bm25", "docs": docs, "bm25": bm25, "tokenizer": tokenizer}

    if method == "dense":
        if not os.path.exists(PV["dense_npy"]):
            raise ArtifactMissing("dense_emb.npy missing for variant")
        emb = _load_numpy(PV["dense_npy"]).astype("float32")
        model = SentenceTransformer(vcfg["embed_model_name"])
        return {"method": "dense", "docs": docs, "docs_embeddings": emb, "dense_model": model}

    if method == "faiss":
        if not (os.path.exists(PV["dense_npy"]) and os.path.exists(PV["faiss_index"])):
            raise ArtifactMissing("faiss.index or dense_emb.npy missing for variant")
        emb = _load_numpy(PV["dense_npy"]).astype("float32")
        index = _load_faiss(PV["faiss_index"])  # CPU
        if use_gpu_for_faiss and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        model = SentenceTransformer(vcfg["embed_model_name"])
        return {"method": "faiss", "docs": docs, "docs_embeddings": emb, "dense_model": model, "faiss_index": index}

    raise ValueError(f"unknown method in variant config: {method}")
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


def retrieve_trained_with_weak_query(query: str, retrieve_k: int = 15) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """재작성 ---> 새롭게 만들어진 weak query로 학습된 임베딩 모델 기반 retriever 후 리랭크"""
    corpus_key = "corpus-generated-train"
    variant = "bge-m3@faiss"
    use_gpu_for_faiss = False

    index_setting = _load_index(store_dir="data/vector_db", corpus_key=corpus_key, variant=variant,
                               use_gpu_for_faiss=use_gpu_for_faiss)
    if 'bge' in variant:
        query = f"Query: {query}"
    candidates = _search(query, k=retrieve_k, handle=index_setting)  # [15개의 문서]
    candidates = [('Document', c[0].replace('Document: ', '')) for c in candidates]
    ce_inputs = [[query, passage] for (_title, passage) in candidates]

    return candidates, ce_inputs

def rerank(candidates: List, ce_inputs: List, top_k: int = 3) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """retrieve_k개 검색 → CrossEncoder로 rerank → top_k 반환"""
    _ensure_loaded()
    # q_emb = _bi_encoder.encode([query], convert_to_numpy=True)
    # if q_emb.ndim == 1:
    #     q_emb = q_emb.reshape(1, -1)
    # q_emb = _l2_normalize(q_emb)
    #
    # D, I = _faiss_index.search(q_emb, retrieve_k)
    # faiss_indices = I[0].tolist()
    #
    # # candidates = []
    # # for idx in faiss_indices:
    # #     row = _law_df.iloc[_law_ids[idx]]
    # #     candidates.append((str(row["법령_조문"]), str(row["법령_텍스트"])))
    #
    # ce_inputs = [[query, passage] for (_title, passage) in candidates]
    print('ce_inputs 이거 잘받는 거 맞음? ', ce_inputs )
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