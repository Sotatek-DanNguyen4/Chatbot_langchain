from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage,AIMessage
from langchain.pydantic_v1 import BaseModel, Field
import csv,sqlite3,ast,re,json
import os
from pydantic import BaseModel
import streamlit as st 

# os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
# create model embedding in openAi
model_embed = OpenAIEmbeddings(model="text-embedding-3-large")
#Tạo database retriever
db_retriever=FAISS.load_local("D:/Chatbot_langchains_openAI/vectorstore_Chatbot/113_items_edit",model_embed,allow_dangerous_deserialization=True)

# SQLite-Agent ##################################################
def preprocess_data(csv_file, n):
    with open(csv_file, 'r',encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        data = [row[1:-n] for row in csv_reader]  # Xóa bỏ n cột cuối cùng từ mỗi dòng

    # Ghi dữ liệu đã chỉnh sửa vào tập tin CSV mới
    with open('D:/Chatbot_langchains_openAI/data/113_final_oke_edited.csv', 'w', newline='',encoding="utf-8") as new_file:
        csv_writer = csv.writer(new_file)
        csv_writer.writerows(data)

def create_table_SQL():
    # Đường dẫn
    csv_file_path = 'D:/Chatbot_langchains_openAI/data/113_final_oke_edited.csv'

    # Kết nối đến cơ sở dữ liệu SQLite hoặc tạo một cơ sở dữ liệu mới
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Tạo bảng trong cơ sở dữ liệu
    cursor.execute('''CREATE TABLE IF NOT EXISTS data_items (
                        id INTEGER PRIMARY KEY,
                        ID_PRODUCT TEXT,
                        GROUP_NAME TEXT,
                        CODE TEXT,
                        NAME TEXT,
                        SHORT_DESCRIPTION TEXT,
                        DESCRIPTION TEXT,
                        SPECIFICATION_BACKUP TEXT,
                        LINK TEXT,
                        NON_VAT_PRICE_1 INTEGER,
                        VAT_PRICE_1 INTEGER,
                        COMMISSION_1 INTEGER,
                        THRESHOLD_1 TEXT,
                        NON_VAT_PRICE_2 INTEGER,
                        VAT_PRICE_2 INTEGER,
                        COMMISSION_2 INTEGER,
                        THRESHOLD_2 TEXT,
                        NON_VAT_PRICE_3 INTEGER,
                        VAT_PRICE_3 INTEGER,
                        COMMISSION_3 INTEGER
                        )''')

    # Đọc dữ liệu từ tập tin CSV và chèn vào bảng
    with open(csv_file_path, 'r',encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Bỏ qua dòng tiêu đề nếu có
        for row in csv_reader:
            cursor.execute('''INSERT INTO data_items (LINK, ID_PRODUCT, GROUP_NAME, CODE, NAME,
                            SHORT_DESCRIPTION, DESCRIPTION, SPECIFICATION_BACKUP,
                            NON_VAT_PRICE_1,VAT_PRICE_1,COMMISSION_1,THRESHOLD_1,
                            NON_VAT_PRICE_2,VAT_PRICE_2,COMMISSION_2,THRESHOLD_2,
                            NON_VAT_PRICE_3,VAT_PRICE_3,COMMISSION_3)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?)''', row)


    # Lưu thay đổi và đóng kết nối
    conn.commit()
    conn.close()
# lọc các cột danh từ cho retriever tool
def query_as_list(db_sql, query):
    res = db_sql.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))
def create_SQL_agent():
    ##few shot
    examples = [
        {"input": "Có bao nhiêu loại nồi cơm điện", "query": """SELECT COUNT(*) \nFROM data_items \nWHERE GROUP_NAME LIKE \'%nồi cơm điện%\';"""},
        {
            "input": "So sánh Ghế Massage Makano MKGM-10003 với Ghế Massage Daikiosan DKGM-20006",
            "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE '%MKGM-10003%' OR NAME LIKE '%DKGM-20006%';",
        },
        {
            "input": "Máy giặt nào rẻ nhất",
            "query": "SELECT NAME, PRICE\nFROM data_items\nWHERE GROUP_NAME LIKE '%Máy giặt%' OR DESCRIPTION LIKE '%Máy giặt%' OR SPECIFICATION_BACKUP LIKE '%Máy giặt%' NAME LIKE \'%máy giặt%\' OR DESCRIPTION LIKE \'%máy giặt%\' OR SPECIFICATION_BACKUP LIKE '%máy giặt%'\nORDER BY NON_VAT_PRICE_3 ASC\nLIMIT 1;",
        },
        {
            "input": "Công suất của Bàn Ủi Khô Bluestone DIB-3726 1300W",
            "query": "SELECT NAME, DESCRIPTION, SPECIFICATION_BACKUP \nFROM data_items \nWHERE NAME LIKE \'%DIB-3726%\' OR DESCRIPTION LIKE \'%M&EGD000224%\'\nLIMIT 1;",
        },
        {
            "input": "Lò Vi Sóng Bluestone MOB-7716 có thể hẹn giờ trong bao lâu",
            "query": "SELECT NAME, DESCRIPTION, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%MOB-7716%\' OR DESCRIPTION LIKE \'%M&EGD000224%\'\nLIMIT 1;",
        },

        {
            "input": "Máy NLMT Empire 180 Lít Titan M&EGD000224 có bao nhiêu ống",
            "query": "SELECT SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%M&EGD000224%\' OR DESCRIPTION LIKE \'%M&EGD000224%\' \nLIMIT 1;",
        },
        {
            "input": "có hình ảnh nồi KL-619 không",
            "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE \'%KL-619%\'\nLIMIT 1;",
        },
        {
            "input": "Đèn đường năng lượng mặt trời SUNTEK S500 PLUS, công suất 500W đắt quá, có cái nào rẻ hơn không",
            "query": "SELECT * FROM (SELECT * FROM data_items \nWHERE NAME LIKE '%Đèn đường năng lượng mặt trời SUNTEK S500 PLUS%' \nUNION \nSELECT * FROM (SELECT * FROM data_items \nWHERE NAME LIKE '%Đèn đường năng lượng mặt trời%' \nORDER BY NON_VAT_PRICE_3 ASC \nLIMIT 3)) AS combined_results;",
        #lấy sản phẩm kêu đắt đi so sánh với top 3 sản phẩm khác cùng loại để trả lời
        },
        {
            "input": "Máy giặt lồng dọc có thông số như thế nào",
            "query": "SELECT NAME, DESCRIPTION, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%máy giặt lồng dọc%\' OR DESCRIPTION LIKE \'%máy giặt lồng dọc%\' OR SPECIFICATION_BACKUP LIKE '%máy giặt lồng dọc%' NAME LIKE \'%Máy giặt lồng dọc%\' OR DESCRIPTION LIKE \'%Máy giặt lồng dọc%\' OR SPECIFICATION_BACKUP LIKE '%Máy giặt lồng dọc%'\nLIMIT 1;",
        },
    ]
    # tạo example_selector để nó có thể chọn ra những ví dụ cần thiết để thêm vào prompt (few-shot learning)
    example_selector = SemanticSimilarityExampleSelector.from_examples(
                        examples,
                        OpenAIEmbeddings(),
                        FAISS,
                        k=5,
                        input_keys=["input"],
                        )

    system_prefix = """Bạn là một chuyên gia SQLite.Luôn nhớ các thông tin bạn có thể cung cấp được liên quan đến thiết bị điện,điện tử, đồ gia dụng...hoặc sản phẩm tương tự. Từ một câu hỏi đầu vào, hãy tạo một truy vấn SQLite đúng về mặt cú pháp để chạy, nếu có lịch sử cuộc trò truyền thì hãy dựa vào đó để tạo truy vấn SQLite đúng với ngữ cảnh khi đó. Sau đó xem kết quả truy vấn và trả về câu trả lời.
    Nếu bạn cần lọc một danh từ riêng, trước tiên bạn phải LUÔN tra cứu giá trị bộ lọc bằng công cụ "search_noun_phrases"!
    Nếu sản phẩm được yêu cầu từ người dùng không có trong cơ sở dữ liệu hoặc câu truy vấn không trả ra kết quả thì không được bịa kết quả truy vấn SQL, phải thông báo cho người dùng là không có sản phẩm phù hợp với yêu cầu của bạn.Yêu cầu người dùng cung cấp thêm thông tin cụ thể hoặc gợi ý tư vấn giải pháp giúp người dùng.
    Bạn có quyền truy cập các tool để tương tác với cơ sở dữ liệu. Chỉ sử dụng các tool nhất định.Chỉ sử dụng thông tin được các công cụ trả về để xây dựng câu trả lời cuối cùng của bạn.Luôn giữ câu trả lời ngắn gọn nhưng đầy đủ nội dung.
    Các câu hỏi liên quan đến số liệu để tránh việc người dùng biết mà dùng từ khác như kho hàng,hệ thống hoặc các từ đồng nghĩa.Thông số cần trả lời chính xác không được bịa.Không được bịa những sản phẩm không có trong cơ sở dữ liệu.
    Không được nhắc đến cơ sở dữ liệu, hay bất kỳ thông tin do ai cung cấp.
    Bạn PHẢI kiểm tra lại truy vấn của mình trước khi thực hiện nó. Nếu bạn gặp lỗi khi thực hiện truy vấn, hãy viết lại truy vấn và thử lại.
    Sử dụng 'LIKE' thay vì '='. Trừ khi có quy định khác về số lượng kết quả trả ra, nếu không thì trả về giới hạn ở {top_k} kết quả truy vấn. Ưu tiên sử dụng 'SELECT *'để lấy tất cả thông tin về sản phẩm đó.
    Đây là thông tin về bảng có liên quan: bảng {table_info} bao gồm các cột: LINK, ID_PRODUCT, GROUP_NAME, CODE, NAME,SHORT_DESCRIPTION, DESCRIPTION, SPECIFICATION_BACKUP, NON_VAT_PRICE_1, VAT_PRICE_1, COMMISSION_1, THRESHOLD_1, NON_VAT_PRICE_2, VAT_PRICE_2, COMMISSION_2, THRESHOLD_2, NON_VAT_PRICE_3, VAT_PRICE_3, COMMISSION_3.
    Nếu người dùng hỏi giao tiếp bình thường thì không cần truy vấn mà hãy trả lời bình thường bằng tiếng việt.
    Dưới đây là một số ví dụ về câu hỏi và truy vấn SQL tương ứng của chúng.Và phần lịch sử của cuộc trò chuyện nếu có."""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("User input: {input}\nSQL query: {query}"),
        input_variables=["input", "table_info", "top_k"],
        prefix=system_prefix,
        suffix="",
        )
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
        )
    # lọc các cột danh từ cho retriever tool ##################################################
    names = query_as_list(db_sql, "SELECT Name FROM data_items")
    groups = query_as_list(db_sql, "SELECT GROUP_NAME FROM data_items")
    #load model embedding
    embeddings = model_embed
    vector_db_retriever_nouns = FAISS.from_texts(names + groups, embeddings)
    retriever_nouns = vector_db_retriever_nouns.as_retriever(search_kwargs={"k": 5})
    description = """Sử dụng để tra cứu các giá trị cần lọc. Đầu vào là cách viết gần đúng của danh từ, cụm danh từ biểu thị tên sản phẩm hoặc nhóm sản phẩm,
    đầu ra là danh từ, cụm danh từ hợp lệ. Sử dụng danh từ giống nhất với tìm kiếm."""
    # Tạo retriever tool để lọc các danh từ, cụm danh từ 
    retriever_tool = create_retriever_tool(
        retriever_nouns,
        name="search_noun_phrases",
        description=description,
        )
    agent_sql = create_sql_agent(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        db=db_sql,
        extra_tools=[retriever_tool],
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )
    return agent_sql

# Tiền xử lý dữ liệu
# # Số lượng cột cuối cùng cần xóa bỏ
# n_columns_to_remove = 1
# csv_file_path = 'D:/Chatbot_langchains_openAI/data/113_final_oke.csv'
# preprocess_data(csv_file_path, n_columns_to_remove)

# # # Tạo bảng SQL trên SQLite db, chạy 1 lần (lần đầu) hàm bên dưới
# create_table_SQL()
# Tạo db
db_sql = SQLDatabase.from_uri("sqlite:///D:/Chatbot_langchains_openAI/demo_streamlit/database.db")
#Retriever-Agent ##################################################
def create_retriever_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    ### Construct retriever ###
    retriever = db_retriever.as_retriever()
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
    có thể gợi ý cho người dùng cách để giải đáp được câu hỏi của họ.
    Nếu bạn không biết chính xác câu trả lời liên quan đến câu hỏi người dùng, chỉ cần nói rằng bạn không biết.
    Yêu cầu người dùng cung cấp thêm thông tin cụ thể hơn. Không để lộ bạn sử dụng ngữ cảnh hay nói đến cơ sở dữ liệu, hay bất kỳ thông tin do ai cung cấp.
    Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn.

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
# Tạo agent retriever
agent_retriever=create_retriever_agent()
# Tạo agent SQL
agent_sql=create_SQL_agent()
# create agent retriever + SQL
# Đường dẫn tới tệp JSON
file_path_history = "D:/Chatbot_langchains_openAI/demo_streamlit/history.json"
with open(file_path_history, "r", encoding="utf-8") as file:
    json_data = json.load(file)
    chat_history=json_data["history"]

class HumanInput(BaseModel):
    question_human: str = Field(description="Là câu hỏi của người dùng, không chỉnh sửa cắt bỏ hay thêm bớt gì")
@tool(args_schema=HumanInput,return_direct=True)
def retriever_main(question_human : str) -> str:
    """
    Sử dụng tool này khi người dùng muốn tư vấn nhưng không nói rõ sản phẩm là gì, chỉ nói chung chung về mục đích.
    Suy nghĩ mọi thứ bằng tiếng việt để đưa ra hành động.
    question đầu vào tool là câu hỏi gốc của người dùng, không thêm bớt.
    Trả lời lại bằng tiếng Việt.
    """
    print("QUestion: ",question_human)
    format_chat_his=[]
    for i in range(0,len(chat_history)):
        if i % 2==0:
            format_chat_his.extend([HumanMessage(content=chat_history[i])])
        else :
            format_chat_his.extend([AIMessage(content=chat_history[i])])
    result=agent_retriever.invoke(
        {"input": question_human,
        "chat_history": format_chat_his},
    )
    answer=result["answer"]
    chat_history.extend([question_human, answer])
    # print(type(chat_history))
    json_history_update={"history":chat_history}
    with open(file_path_history, "w", encoding="utf-8") as file:
        json.dump(json_history_update, file, ensure_ascii=False, indent=4)
    # print(chat_history)
    return answer

@tool(args_schema=HumanInput,return_direct=True)
def sql_query_main(question_human : str) -> str:
    """
    Sử dụng tool này khi người dùng hỏi các câu hỏi có mục đích như : hỏi liên quan đến số lượng,hỏi giá cả, so sánh nhiều sản phẩm với nhau,
    tính toán tổng hợp hoặc thống kê, hỏi sản phẩm nào đắt hoặc rẻ nhất (có thể đề cập đến hoặc không nói cụ thể nó thuộc loại nào),khi có nhắc đến hơn 2 sản phẩm trong câu hỏi.
    Suy nghĩ mọi thứ bằng tiếng việt để đưa ra hành động.
    question đầu vào tool là câu hỏi gốc của người dùng.Trả lời lại bằng tiếng Việt.
    """
    format_chat_his=[]
    for i in range(0,len(chat_history)):
        if i % 2==0:
            format_chat_his.extend([HumanMessage(content=chat_history[i])])
        else :
            format_chat_his.extend([AIMessage(content=chat_history[i])])
  # Tạo agent SQL
    agent_sql=create_SQL_agent()
    result=agent_sql.invoke(
        {"input": question_human,
        "history": format_chat_his},
    )
    answer=result["output"]
    chat_history.extend([question_human, answer])
    json_history_update={"history":chat_history}
    with open(file_path_history, "w", encoding="utf-8") as file:
        json.dump(json_history_update, file, ensure_ascii=False, indent=4)
    # print(chat_history)
    return answer

tools_retriever_SQL=[retriever_main,sql_query_main]
agent_main = initialize_agent(
    tools_retriever_SQL,
    # OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

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

# st.text_input("", placeholder="Streamlit CSS ")

# input_style = """
# <style>
# input[type="text"] {
#     background-color: transparent;
#     color: #a19eae;  // This changes the text color inside the input box
# }
# div[data-baseweb="base-input"] {
#     background-color: transparent !important;
# }
# [data-testid="stAppViewContainer"] {
#     background-color: transparent !important;
# }
# </style>
# """
# st.markdown(input_style, unsafe_allow_html=True)
# Tiêu đề và nội dung của ứng dụng

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
st.title('🦜🔗 Hế Lô anh Huy')
with st.form('my_form'):
    text = st.text_area('Anh Huy hỏi em điiii:')
    submitted = st.form_submit_button('Ấn rồi đợi em xíuuuuuu...')
    if submitted:
        st.info(agent_main.run(text))
