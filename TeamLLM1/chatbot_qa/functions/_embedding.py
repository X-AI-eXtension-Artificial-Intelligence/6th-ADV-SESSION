import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings


class QwenEmbeddingModel:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings  # Tensor

    def load_embedding_function(self):
        # 기본 함수형 임베딩 반환 (List[str] → Tensor)
        return self._embed

    def get_langchain_embedding(self):
        # LangChain-compatible Embedding class 리턴
        return self.LangchainEmbeddingWrapper(self._embed)

    class LangchainEmbeddingWrapper(Embeddings):
        def __init__(self, embed_fn):
            self.embed_fn = embed_fn

        def embed_documents(self, texts):
            return self.embed_fn(texts).cpu().tolist()

        def embed_query(self, text):
            return self.embed_fn([text])[0].cpu().tolist()


if __name__ == "__main__":  # python -m chatbot_qa.functions._embedding
    model = QwenEmbeddingModel.getInstance()
    embed = model.load_embedding_function()
    vecs = embed(["형법 제250조는 살인죄를 규정합니다."])
    print(vecs.shape)

    lc_embed = model.get_langchain_embedding()
    print("query vector (LC):", lc_embed.embed_query("살인죄 요건이 궁금합니다."))
