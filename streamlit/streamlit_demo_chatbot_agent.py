from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import HumanMessage,AIMessage
import os,sys,time
from pydantic import BaseModel
import streamlit as st 
# Get the absolute path of the project directory
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project directory to sys.path
sys.path.append(project_path)
from source.sql_qa.sql_creator import SQLCreatorServer
from source.sql_qa.agent_creator import AgentCreatorSQL
from source.rag_qa.rag_creator import RetrieverCreator
from source.history.process_history import HistoryProcessor
from langchain.pydantic_v1 import BaseModel, Field
class HumanInput(BaseModel):
    question_human: str = Field(description="Là câu hỏi của người dùng, không chỉnh sửa cắt bỏ hay thêm bớt gì")
# # os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# create model embedding in openAi
model_embed = OpenAIEmbeddings(model="text-embedding-3-large")
#Tạo database retriever
db_retriever=FAISS.load_local("D:/Chatbot_langchains_openAI/vectorstore/store_113_items",model_embed,allow_dangerous_deserialization=True)
db_sql = SQLDatabase.from_uri("sqlite:///D:/Chatbot_langchains_openAI/streamlit/database_204_items.db")
# status= False #"Lần đầu" = True 
# if status:
#     SQLCreatorServer().create_table_SQLite(csv_file_path_local="D:/Chatbot_langchains_openAI/data/data_csv/113_final_oke_edited.csv",
#                                        name_table="data_items")
sql_agent=AgentCreatorSQL().create_SQL_agent(llm,model_embed,db_sql)
agent_retriever=RetrieverCreator().create_retriever_chain(llm,db_retriever)
def answer(id_user,id_conversation,question):
    chat_history=HistoryProcessor().load_history(id_user,id_conversation)
    format_chat_his=[]
    for i in range(0,len(chat_history)):
        format_chat_his.append(HumanMessage(content=chat_history[i]["HumanMessage"]))
        format_chat_his.append(AIMessage(content=chat_history[i]["AIMessage"]))
    # print("chat_history: ",chat_history)
    @tool(args_schema=HumanInput,return_direct=True)
    def retriever_tool(question_human : str) -> str:
        """
        Sử dụng tool này khi người dùng muốn tư vấn nhưng không nói rõ sản phẩm là gì, chỉ nói chung chung về mục đích.
        KHÔNG dùng tool này để trả lời các câu hỏi về số lượng.KHÔNG dùng tool này để trả lời các câu hỏi so sánh trên 2 sản phẩm khác nhau.
        Suy nghĩ mọi thứ bằng tiếng việt để đưa ra hành động.
        question đầu vào tool là câu hỏi gốc của người dùng, không thêm bớt.
        Trả lời lại bằng tiếng Việt.
        """
        # print("QUestion: ",question_human)
        result=agent_retriever.invoke(
            {"input": question_human,
            "chat_history": format_chat_his},
        )
        print(result)
        answer=result["answer"]
        # print(chat_history)
        return answer
    @tool(args_schema=HumanInput,return_direct=True)
    def sql_query_tool(question_human : str) -> str:
        """
        Sử dụng tool này khi người dùng hỏi các câu hỏi có mục đích như : hỏi liên quan đến số lượng,hỏi giá cả, so sánh nhiều sản phẩm với nhau,
        tính toán tổng hợp hoặc thống kê, hỏi sản phẩm nào đắt hoặc rẻ nhất (có thể đề cập đến hoặc không nói cụ thể nó thuộc loại nào),khi có nhắc đến hơn 2 sản phẩm trong câu hỏi.
        Suy nghĩ mọi thứ bằng tiếng việt để đưa ra hành động.
        question đầu vào tool là câu hỏi gốc của người dùng.Trả lời lại bằng tiếng Việt.
        """
        result=sql_agent.invoke(
            {"input": question_human,
            "history": format_chat_his},
        )
        print(result)
        answer=result["output"]
        return answer
    tools_retriever_SQL=[retriever_tool,sql_query_tool]
    agent_main = initialize_agent(
        tools_retriever_SQL,
        # OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    result=agent_main.run(question)
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
    if submitted:
        st.info(result)
