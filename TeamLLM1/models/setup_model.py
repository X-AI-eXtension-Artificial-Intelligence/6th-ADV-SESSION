import os
import gdown
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# ---- ENV ----
soft_prompt_id = "1Cbhm5GfJzWdpCh18jqcLa0no9n7LmQ6i"
hf_token = os.getenv("HUGGINGFACE_TOKEN")
base_dir = os.getenv("BASE_MODEL_DIR")
retrieve_id = os.getenv("RETRIEVE_MODEL_NAME")
rerank_id = os.getenv("RERANK_MODEL_NAME")
generate_id = os.getenv("GENERATE_MODEL")
generate_bool = os.getenv("GENERATE_MODEL_BOOL", "false").lower() == "true"

if not all([base_dir, retrieve_id, rerank_id, generate_id]):
    raise ValueError(
        "환경변수(BASE_MODEL_DIR, RETRIEVE_MODEL_NAME, RERANK_MODEL_NAME, GENERATE_MODEL)를 확인하세요."
    )

# HF 로그인
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"[경고] Hugging Face 로그인 실패: {e}")
else:
    print("[주의] HUGGINGFACE_TOKEN이 없어 Gated 모델 다운로드가 실패할 수 있습니다.")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def download_embedding_model(model_id: str, save_path: str):
    try:
        print(f"[임베딩] 다운로드 시작: {model_id}")
        model = SentenceTransformer(model_id)
        model.save(save_path)
        print(f"[임베딩] 저장 완료: {save_path}")
    except Exception as e:
        print(f"[임베딩] 오류({model_id}): {e}")


def download_rerank_model(model_id: str, save_path: str):
    try:
        print(f"[리랭크] 다운로드 시작: {model_id}")
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
            )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
            )
        tok.save_pretrained(save_path)
        mdl.save_pretrained(save_path)
        print(f"[리랭크] 저장 완료: {save_path}")
    except Exception as e:
        print(f"[리랭크] 오류({model_id}): {e}")


def download_soft_prompt_model(save_path: str, file_id: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model from: {url}")

    gdown.download(url, save_path, quiet=False, fuzzy=True)
    print(f"✅ Model downloaded and saved to: {save_path}")


def download_generate_model(model_id: str, save_path: str):
    try:
        print(f"[생성] 다운로드 시작: {model_id}")
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        )
        tok.save_pretrained(save_path)
        mdl.save_pretrained(save_path)
        print(f"[생성] 저장 완료: {save_path}")
    except Exception as e:
        print(f"[생성] 오류({model_id}): {e}")


def download_model_files():
    retrieve_path = os.path.join(base_dir, "retrieve")
    rerank_path = os.path.join(base_dir, "rerank")
    generate_path = os.path.join(base_dir, "generate")
    soft_prompt_path = os.path.join(
        base_dir, "train_retriever_result", "bge-m3-contrastive-01", "model.safetensors"
    )

    ensure_dir(base_dir)

    # soft prompt
    if not os.path.exists(soft_prompt_path):
        download_soft_prompt_model(
            save_path=soft_prompt_path,
            file_id=soft_prompt_id
        )
    else:
        print("✅ soft_prompt 모델 이미 존재")

    # Retrieve (임베딩)
    if not os.path.exists(retrieve_path) or not os.listdir(retrieve_path):
        ensure_dir(retrieve_path)
        download_embedding_model(retrieve_id, retrieve_path)
    else:
        print(f"[임베딩] 이미 존재: {retrieve_path}")

    # Rerank (크로스인코더)
    if not os.path.exists(rerank_path) or not os.listdir(rerank_path):
        ensure_dir(rerank_path)
        download_rerank_model(rerank_id, rerank_path)
    else:
        print(f"[리랭크] 이미 존재: {rerank_path}")

    # Generate (HF CausalLM only; BOOL 플래그로 제어)
    if generate_bool:
        if not os.path.exists(generate_path) or not os.listdir(generate_path):
            ensure_dir(generate_path)
            download_generate_model(generate_id, generate_path)
        else:
            print(f"[생성] 이미 존재: {generate_path}")
    else:
        print(f"[생성] GENERATE_MODEL_BOOL=False → 다운로드 스킵: {generate_id}")


if __name__ == "__main__":
    download_model_files()
