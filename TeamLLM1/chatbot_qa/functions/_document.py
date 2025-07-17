import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

class DocumentProcessFunc:
    PROMPT_TEMPLATE = "assets/prompt_config.json"
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

    def _load_prompt_config(self):
        with open(self.PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        documents = []
        for item in qa_data:
            question = item['question']
            answer = item['answer']
            documents.append(f"Question: {question}\nAnswer: {answer}")
        print("document: ", documents)
        return documents

    def build_llm_prompt(self, question, rag_context, prompt_key="llm_prompt_ver1"):
        template = self._load_prompt_config()[prompt_key]["template"]
        return template.format(question=question, rag_context=rag_context)

    # 학습시 instruction-aware 방식 사용한다면 아래 함수 사용
    def build_retriever_instruction(self, query, prompt_key="retriever_instruction"):
        template = self._load_prompt_config()[prompt_key]["template"]
        return template.format(question=query)


if __name__ == "__main__": # python -m chatbot_qa.functions._document
    BotInstance = DocumentProcessFunc()
    docu = BotInstance.load_document("assets/data_sample.json")
    template = BotInstance.build_llm_prompt(question="집에 가고싶어요",rag_context="")
    print("template: \n", template)
