import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from chatbot_qa.functions._document import DocumentProcessFunc
from chatbot_qa.functions._embedding import QwenEmbeddingModel
from chatbot_qa.functions._langchain import LangchainFunc
from chatbot_qa.functions._llm import LLMFunc
class ChatbotQA:
    FILE_PATH = 'assets/data_sample.json'
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__document_func = DocumentProcessFunc.getInstance()
            cls.__instance.__embedding_func = QwenEmbeddingModel.getInstance()
            cls.__instance.__langchain_func = LangchainFunc.getInstance()
            cls.__instance.__llm_func = LLMFunc()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def get_faiss_index(self):
        document = self.__document_func.load_document(self.FILE_PATH)
        embedding = self.__embedding_func.get_langchain_embedding()
        self.__langchain_func.create_faiss_index(document, embedding)

    def get_answer(self, user_question):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding = self.__embedding_func.get_langchain_embedding()

        faissIndex = self.__langchain_func.load_faiss_index(faissIndexPath='assets/faiss_index_file',
                                                            embeddings=embedding)
        # llm = self.__llm_func.load_bllossom_chain(device)
        llm = self.__llm_func.load_openai_chain()
        prompt = self.__langchain_func.load_prompt_template()
        chain = self.__langchain_func.create_chain(llm, prompt, faissIndex)
        response = self.__langchain_func.invoke_chain(chain, user_question)
        print("result: ", response)
        return {'result': response}
