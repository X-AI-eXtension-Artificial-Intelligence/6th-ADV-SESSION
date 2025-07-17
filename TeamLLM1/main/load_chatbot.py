from chatbot_qa.chatbot_service import ChatbotQA

ChatbotInstance = ChatbotQA()
def create_index():
    ChatbotInstance.get_faiss_index()

def chatbot(question):
    ChatbotInstance.get_answer(question)

if __name__ == "__main__":
    print('')
    # create_index()
    chatbot(question="집에 가고 싶어.")
