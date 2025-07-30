import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv

def build_faiss_index(input_path: str, output_path: str, model_name: str):
    """
    CSV 파일로부터 텍스트를 읽어들여 FAISS 인덱스를 구축하고 저장합니다.

    Args:
        input_path (str): 입력 CSV 파일 경로.
        output_path (str): 생성된 FAISS 인덱스를 저장할 파일 경로.
        model_name (str): 사용할 Sentence Transformer 모델 이름.
    """
    # --- 1. 데이터 로드 및 전처리 ---
    print(f"🔄 [1/5] '{input_path}' 파일에서 데이터를 로드합니다...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"에러: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다.")
        return

    print("[2/5] 임베딩할 텍스트 데이터를 전처리합니다...")
    df['text_for_embedding'] = df['법령명'].astype(str) + " " + \
                               df['조문 번호'].astype(str) + " " + \
                               df['조문 내용'].astype(str)
    texts = df['text_for_embedding'].tolist()
    print("데이터 전처리 완료.")
    print("-" * 50)

    # --- 2. 임베딩 모델 로드 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[3/5] 임베딩 모델 '{model_name}'을(를) 로드합니다... ({device} 사용)")
    model = SentenceTransformer(model_name, device=device)
    print("임베딩 모델 로드 완료.")
    print("-" * 50)

    # --- 3. 텍스트 임베딩 생성 ---
    print(f"[4/5] 총 {len(texts)}개의 텍스트에 대한 임베딩을 생성합니다...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().numpy()
    print(f"텍스트 임베딩 생성 완료. 벡터 차원: {embeddings_np.shape}")
    print("-" * 50)

    # --- 4. FAISS 인덱스 구축 및 저장 ---
    print("[5/5] FAISS 인덱스를 구축하고 저장합니다...")
    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    faiss.write_index(index, output_path)
    print(f"FAISS 인덱스 구축 완료.")
    print(f"   - 총 {index.ntotal}개의 벡터가 인덱스에 추가되었습니다.")
    print(f"   - '{output_path}' 파일로 인덱스 저장 완료.")


if __name__ == '__main__':
    # .env 파일에서 환경변수 로드
    load_dotenv()

    # 환경변수에서 설정값 가져오기
    input_csv_path = os.getenv("FAISS_INPUT_DATA")
    faiss_index_path = os.getenv("FAISS_OUTPUT_DATA")
    output_path = os.path.join(faiss_index_path , "db.index")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    print(output_path)
    # 환경변수가 제대로 설정되었는지 확인
    if not all([input_csv_path, output_path, embedding_model]):
        raise ValueError("환경변수(INPUT_CSV_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL)를 .env 파일에 설정해주세요.")
    
    # 메인 함수 실행
    build_faiss_index(
        input_path=input_csv_path,
        output_path=output_path,
        model_name=embedding_model
    )