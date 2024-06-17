from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage,AIMessage
import csv,sqlite3,ast,re,json
import os
from source.history.process_history import HistoryProcessor
# from langchain.pydantic_v1 import BaseModel, Field

# class HumanInput(BaseModel):
#     question_human: str = Field(description="Là câu hỏi của người dùng, không chỉnh sửa cắt bỏ hay thêm bớt gì")

class RetrieverCreator():
    def __init__(self) -> None:
        pass
    def create_retriever_chain(self,llm,db_retriever):
        # ví dụ llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        ### Construct retriever ###
        retriever = db_retriever.as_retriever(search_kwargs={"k": 3})
        ### Contextualize question ###
        contextualize_q_system_prompt = """Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng có thể tham khảo ngữ cảnh trong lịch sử trò chuyện, hãy tạo một câu hỏi độc lập
        có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, chỉ cần sửa lại câu hỏi nếu cần và nếu không thì trả lại như cũ."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        ### Answer question ###
        qa_system_prompt = """Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi.Luôn nhớ các thông tin bạn có thể cung cấp được liên quan đến thiết bị điện,điện tử, đồ gia dụng...hoặc sản phẩm tương tự.
        Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. Không được bịa ra sản phẩm không có trong cơ sở dữ liệu.
        Nếu không có sản phẩm người dùng hỏi, hoặc vấn đề người dùng hỏi thì nói về phạm vi hoạt động của bạn là gì và không thể thực hiện yêu cầu với phạm vi câu hỏi của người dùng,
        Có thể "GỢI Ý" cho người dùng cách để giải đáp được câu hỏi của họ nhưng không đựa bịa sản phẩm mà bạn không có trong cơ sở dữ liệu.
        Nếu bạn không biết chính xác câu trả lời liên quan đến câu hỏi người dùng, chỉ cần nói rằng bạn không biết.
        Yêu cầu người dùng cung cấp thêm thông tin cụ thể hơn. Không để lộ bạn sử dụng ngữ cảnh hay nói đến cơ sở dữ liệu, hay bất kỳ thông tin do ai cung cấp.
        Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn, đủ ý.

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain