import os, torch, json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pinecone

from transformers import (
    VisionTextDualEncoderModel,
    AutoProcessor
)

# 1. 설정
BASE_DIR = "/home/work/XAI_ADV"
CSV_PATH = os.path.join(BASE_DIR, "Training", "inference.csv")
MODEL_DIR = os.path.join(BASE_DIR, "koclip_lora_adapter", "koclip_finetuning_model")

# Pinecone 설정
PINECONE_API_KEY = " "   # 파인콘 API Key
PINECONE_ENV = "us-east-1"                 
INDEX_NAME = "multimodal_qa"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 모델 로드
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = VisionTextDualEncoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# 3. Pinecone 초기화
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=model.config.projection_dim, metric="cosine")

index = pinecone.Index(INDEX_NAME)

# 4. 임베딩 함수
def get_image_embedding(image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"이미지 로드 실패: {image_path} ({e})")
        return None
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # normalize
    return emb.squeeze().cpu().tolist()

# 5. CSV 로드
df = pd.read_csv(CSV_PATH)

# 6. Pinecone 업서트
vectors = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    emb = get_image_embedding(row["image_path"])
    if emb is None:
        continue
    # metadata JSON 파싱
    try:
        meta = json.loads(row["metadata"])
    except Exception:
        meta = {"metadata": row["metadata"]}

    # image_path도 함께 저장
    meta["image_path"] = row["image_path"]

    vectors.append((
        str(idx),
        emb,
        meta
    ))

# 배치 업서트
for i in range(0, len(vectors), 100):
    batch = vectors[i:i+100]
    index.upsert(vectors=batch)

print(f"{len(vectors)} image embeddings upserted into Pinecone index: {INDEX_NAME}")
