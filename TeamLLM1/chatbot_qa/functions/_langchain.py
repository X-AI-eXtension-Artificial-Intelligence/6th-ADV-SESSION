import json
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

class LangchainFunc:
    store = {}
    FAISS_INDEX_PATH = "assets/faiss_index_file"
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

    def load_prompt_template(self, prompt_key="llm_prompt_ver1"):
        prompt_config = self._load_prompt_config()[prompt_key]["template"]
        system_prefix = (
            "당신은 법률 상담을 수행하는 법률 전문가입니다. "
            "사용자의 질문에 대해 법률적인 답변을 생성해주세요.\n"
        )

        system_prompt = system_prefix + prompt_config
        print(system_prompt)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        return prompt_template

    def create_faiss_index(self, texts, embedding):
        faissIndex = FAISS.from_texts(texts=texts, embedding=embedding)
        faissIndex.save_local(self.FAISS_INDEX_PATH)
        print("index가 성공적으로 저장됨")

    def load_faiss_index(self, faissIndexPath, embeddings):
        faissIndex = FAISS.load_local(folder_path=faissIndexPath, embeddings=embeddings, allow_dangerous_deserialization=True)
        return faissIndex

    def _get_session_history(self, session_ids='always_same'):
        # print(f"[대화 세션ID]: {session_ids}")
        if session_ids not in self.store:
            self.store[session_ids] = ChatMessageHistory()
        return self.store[session_ids]

    def create_chain(self, llm, prompt, faiss_index):
        chain = (
                {
                    "rag_context": itemgetter("question") | faiss_index.as_retriever(),  # retriever
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
                | prompt  # 프롬프트 추가
                | llm
                | StrOutputParser()  # 언어 모델의 출력을 문자열로 변환
        )

        qa_chain = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
            history_messages_key="chat_history",  # 기록 메시지의 키
        )

        return qa_chain

    def invoke_chain(self, chain, user_question):
        print("invoke_chain => 답변 생성 중")
        result = chain.invoke({"question": user_question},
                              config={"configurable": {"session_id": "always_same"}})
        return result

if __name__ == "__main__": # python -m chatbot_qa.functions._langchain
    print('')