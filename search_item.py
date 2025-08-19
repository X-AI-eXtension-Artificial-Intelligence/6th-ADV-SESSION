import os, torch
from transformers import AutoProcessor, VisionTextDualEncoderModel
import pinecone

# 1. 설정
BASE_DIR = "/home/work/XAI_ADV"
MODEL_DIR = os.path.join(BASE_DIR, "koclip_lora_adapter", "koclip_finetuning_model")

PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "multimodal-qa"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 모델 로드
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = VisionTextDualEncoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# 3. Pinecone 초기화
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# 4. 텍스트 → 임베딩 변환
def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # normalize
    return emb.squeeze().cpu().tolist()

# 5. 유사도 검색 함수
def search_similar_items(query, top_k = 3):
    query_emb = get_text_embedding(query)

    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)

    items = []
    for match in results["matches"]:
        meta = match["metadata"]
        items.append({
            "image_path": meta.get("image_path", ""),
            "img_prod_nm": meta.get("img_prod_nm", ""),
            "score": match["score"]
        })

    return items

# 6. 테스트 실행
if __name__ == "__main__":
    query = "파란색 봉지 과자"
    results = search_similar_items(query, top_k=3)

    for r in results:
        print(f"상품명: {r['img_prod_nm']} | 이미지: {r['image_path']} | 유사도: {r['score']:.4f}")
