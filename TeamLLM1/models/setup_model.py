import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer 
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 토큰으로 로그인
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Hugging Face 토큰이 없어 Gated Model 다운로드가 실패할 수 있습니다.")


def download_embedding_model(model_id, save_path):
    """SentenceTransformer 모델을 다운로드합니다."""
    try:
        print(f"Downloading embedding model: {model_id}")
        model = SentenceTransformer(model_id, token=hf_token)
        model.save(save_path)
        print(f"Embedding model saved to: {save_path}")
    except Exception as e:
        print(f"Error downloading {model_id}: {e}")


def download_llm_model(model_id, save_path):
    """Causal LM 모델을 다운로드합니다."""
    try:
        print(f"Downloading LLM: {model_id}")
        # trust_remote_code와 token 인자를 함께 사용
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"LLM saved to: {save_path}")
    except Exception as e:
        print(f"Error downloading {model_id}: {e}")


def download_model_files():
    base_path = os.getenv("BASE_MODEL_DIR")
    embedding_model_id = os.getenv("EMBEDDING_MODEL")
    generate_model_id = os.getenv("GENERATE_MODEL")
    
    if not all([base_path, embedding_model_id, generate_model_id]):
        raise ValueError("환경변수(BASE_MODEL_DIR, EMBEDDING_MODEL, GENERATE_MODEL)가 비어 있습니다.")

    embedding_path = os.path.join(base_path, "embedding_model")
    generate_path = os.path.join(base_path, "generate_model")

    # 각 모델에 맞는 다운로드 함수 호출
    if not os.path.exists(embedding_path):
        print(embedding_path)
        download_embedding_model(embedding_model_id, embedding_path)
    else:
        print(f"Embedding model already exists at: {embedding_path}")

    if not os.path.exists(generate_path):
        print(generate_path)
        #download_llm_model(generate_model_id, generate_path)
    else:
        print(f"LLM already exists at: {generate_path}")

if __name__ == "__main__":
    download_model_files()