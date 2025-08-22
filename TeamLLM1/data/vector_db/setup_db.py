# -*- coding: utf-8 -*-
"""
preprocessing.py
청킹된 CSV(law_db)에서 지정한 텍스트 컬럼을 임베딩하여
FAISS 인덱스(.index)와 ID 매핑(.pkl)을 지정된 output_path에 저장합니다.
(IndexFlatIP + 코사인 유사도용 L2 정규화)

사용 예:
python preprocessing.py --input_csv law_db.csv --text_col 법령_텍스트 \
  --output_path ./faiss_output --model_path ./downloaded_model

※ .env(선택):
FAISS_MODEL_PATH=./downloaded_model
FAISS_OUTPUT_PATH=./faiss_output
INPUT_CSV=law_db.csv
TEXT_COL=법령_텍스트
ENCODE_BATCH_SIZE=32
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import faiss

try:
    from dotenv import load_dotenv  # 선택 사항
    load_dotenv()
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """코사인 유사도용 L2 정규화"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def main():
    parser = argparse.ArgumentParser(description="CSV 특정 텍스트 컬럼으로 FAISS 인덱스 구축")
    parser.add_argument("--input_csv",  default=os.getenv("CHUNKING_OUTPUT_CSV", "law_db.csv"), help="청킹된 CSV 경로")
    parser.add_argument("--text_col",   default=os.getenv("TEXT_COL", "법령_텍스트"), help="임베딩할 텍스트 컬럼명")
    parser.add_argument("--output_path", default=os.getenv("FAISS_OUTPUT_PATH", "./faiss_output"), help="인덱스와 ID 파일이 저장될 디렉토리 경로")
    parser.add_argument("--model_path", default=os.getenv("FAISS_MODEL_PATH"), help="다운로드된 SentenceTransformer 모델 경로")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("ENCODE_BATCH_SIZE", "32")), help="encode 배치 크기")
    args = parser.parse_args()

    # 입력 CSV 체크
    if not os.path.exists(args.input_csv):
        print(f"[오류] 입력 CSV가 존재하지 않습니다: {args.input_csv}")
        sys.exit(1)

    # 모델 경로 체크
    if not args.model_path or not os.path.exists(args.model_path):
        print("[오류] --model_path 또는 .env의 FAISS_MODEL_PATH가 존재하지 않습니다.")
        sys.exit(1)

    # output_path 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    index_path = os.path.join(args.output_path, "faiss_law.index")
    id_path = os.path.join(args.output_path, "faiss_law_ids.pkl")

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        print(f"[오류] 입력 CSV에 지정한 컬럼이 없습니다: {args.text_col}")
        sys.exit(1)

    texts = df[args.text_col].astype(str).fillna("").tolist()

    print(f"모델 로딩 중... ({args.model_path})")
    from sentence_transformers import SentenceTransformer
    bi_encoder = SentenceTransformer(args.model_path)

    # 기존 인덱스가 있으면 로드
    if os.path.exists(index_path) and os.path.exists(id_path):
        print(" FAISS 인덱스 불러오는 중...")
        index = faiss.read_index(index_path)
        with open(id_path, "rb") as f:
            id_list = pickle.load(f)
        print(f" 인덱스 차원: {index.d} / 문서 수: {index.ntotal}")
        if index.ntotal != len(id_list):
            print("[경고] 인덱스 수와 id 리스트 길이가 다릅니다. 재생성을 권장합니다.")
        return

    print(" FAISS 인덱스 생성 중...")
    embeddings = bi_encoder.encode(
        texts,
        show_progress_bar=True,
        batch_size=args.batch_size,
        convert_to_numpy=True
    ).astype("float32")

    embeddings = _normalize(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    id_list = df.index.to_list()

    faiss.write_index(index, index_path)
    with open(id_path, "wb") as f:
        pickle.dump(id_list, f)

    print(f" 인덱스 차원: {index.d} / 문서 수: {index.ntotal}")
    print(f"[완료] 인덱스 저장: {index_path}")
    print(f"[완료] ID 매핑 저장: {id_path}")


if __name__ == "__main__":
    main()
