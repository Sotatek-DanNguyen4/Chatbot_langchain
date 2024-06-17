from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import HumanMessage,AIMessage
from langchain.chains.llm import LLMChain
import csv,sqlite3,ast,re,json
import os,sys,time
from pydantic import BaseModel
import streamlit as st 
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
# Get the absolute path of the project directory
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project directory to sys.path
sys.path.append(project_path)
from source.sql_qa.sql_creator import SQLCreatorServer
from source.sql_qa.agent_creator import AgentCreatorSQL
from source.rag_qa.rag_creator import RetrieverCreator
from source.history.process_history import HistoryProcessor
# from langchain.pydantic_v1 import BaseModel, Field
# class HumanInput(BaseModel):
#     question_human: str = Field(description="Là câu hỏi của người dùng, không chỉnh sửa cắt bỏ hay thêm bớt gì")
# # os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# create model embedding in openAi
model_embed = OpenAIEmbeddings(model="text-embedding-3-large")
#Tạo database retriever
db_retriever=FAISS.load_local("D:/Chatbot_langchains_openAI/vectorstore/store_204_items",model_embed,allow_dangerous_deserialization=True)
db_sql = SQLDatabase.from_uri("sqlite:///D:/Chatbot_langchains_openAI/streamlit/database_204_items.db")
# status= False #"Lần đầu" = True 
# if status:
#     SQLCreatorServer().create_table_SQLite(csv_file_path_local="D:/Chatbot_langchains_openAI/data/data_csv/204_final_edited.csv",
#                                        name_table="data_items")
sql_agent=AgentCreatorSQL().create_SQL_agent(llm,model_embed,db_sql)
agent_retriever=RetrieverCreator().create_retriever_chain(llm,db_retriever)
def answer(id_user,id_conversation,question):
    chat_history=HistoryProcessor().load_history(id_user,id_conversation)
    format_chat_his=[]
    for tmp_his in chat_history[-3:]:
        format_chat_his.append(HumanMessage(content=tmp_his["HumanMessage"]))
        format_chat_his.append(AIMessage(content=tmp_his["AIMessage"]))
    examples = [
        {"input": "Nhà của tôi đang không có dụng cụ nấu ăn", "tool_selected": "retriever"},
        {"input": "Có tất cả bao nhiêu sản phẩm máy giặt", "tool_selected": "sql"},
        {"input": "Tôi cần một số gợi ý về sản phẩm hỗ trợ chiếu sáng trong nhà", "tool_selected": "retriever"},
        {"input": "So sánh giá các sản phẩm đèn năng lượng mặt trời giúp tôi được không?", "tool_selected": "sql"},
    ]
    # Mẫu prompt
    example_template = "Input: {input}\ntool_selected: {tool_selected}"
    # Tạo PromptTemplate cho các ví dụ
    example_prompt = PromptTemplate(
        input_variables=["input", "tool_selected"],
        template=example_template
    )
    # Tạo FewShotPromptTemplate
    system_prefix="""Bạn là một trợ lý hỗ trợ nhận biết tool được sử dụng để trả lời câu hỏi. Chỉ trả ra "retriever" hoặc "sql", không có tool nào khác.
    Tôi có 2 tool hỗ trợ trả lời câu hỏi là "retriever" và "sql".
    Tool retriever sử dụng khi câu hỏi có ý định là 1 số loại sau: chào hỏi giao tiếp xã giao bình thường, muốn tư vấn nhưng không nói cụ thể về sản phẩm là gì.KHÔNG dùng tool này để trả lời các câu hỏi về số lượng.KHÔNG dùng tool này để trả lời các câu hỏi so sánh trên 2 sản phẩm khác nhau.Không dùng tool này khi mục đích là tính toán tiền sản phẩm.
    Tool sql sử dụng khi câu hỏi có ý định là 1 số loại sau: có đề cập đến hỏi số lượng, có đề cập đến việc so sánh gì đó, có đề cập vấn đề tính toán tiền sản phẩm
    Dưới đây là một số ví dụ về phân loại tool sử dụng để trả lời câu hỏi. Dựa trên các ví dụ này, phân loại tool cho văn bản câu hỏi mới.
    """
    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=system_prefix,
        suffix="Input: {question}\ntool_selected:",
        input_variables=["question"]
    )
    # Tạo LLMChain với prompt template
    tool_classification_chain = LLMChain(llm=llm, prompt=prompt_template)
    # print("chat_history: ",chat_history)
    tool_selected=tool_classification_chain.run({"question": question})
    print("tool_selected: ",tool_selected)
    if tool_selected.find("sql") != -1 :
        response=sql_agent.invoke({"input":question,
                                   #  'top_k': 3,
                                 "history":format_chat_his})
        # print(response)
        print("Câu hỏi: ",response['input'])
        print("Lịch sử hội thoại: ",response['history'])
        print("Trả lời: ",response['output'])
        result=response['output']
    else:
        response=agent_retriever.invoke({"input":question,
                                        "chat_history":format_chat_his})
        # print(response)
        print("Câu hỏi: ",response['input'])
        print("Lịch sử hội thoại: ",response['chat_history'])
        print("Context đầu vào LLM: ", response['context'])
        print("Trả lời: ",response['answer'])
        result=response["answer"]
    HistoryProcessor().update_history(id_user,id_conversation,HumanMessage=question,AIMessage=result)
    return result

# streamlit ##################################################
original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">✨Langchain 🦜🔗 </h1>'
st.markdown(original_title, unsafe_allow_html=True)
# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://hoanghamobile.com/tin-tuc/wp-content/webp-express/webp-images/uploads/2023/07/hinh-dep-2.jpg.webp");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .reportview-container .main .block-container div[data-baseweb="toast"] {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('🦜🔗 CHATBOT')
with st.form('my_form'):
    # User inputs
    user_id = st.text_input('Nhập ID của bạn:')
    id_conversation = st.text_input('Nhập id hội thoại:')
    text = st.text_area('Anh Huy hỏi em điiii:')
    # print("id_conversation: ", id_conversation)
    # print("user_id: ", user_id)
    # print("text: ", text)
    submitted = st.form_submit_button('Ấn rồi đợi em xíuuuuuu...')
    start_time=time.time()
    result=answer(user_id,id_conversation,text)
    print("total_time: ",time.time()-start_time)
    print("*****************************************************")
    print("*****************************************************")
    print("*****************************************************")
    if submitted:
        st.info(result)
