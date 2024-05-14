from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import FAISS
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
import csv
import os
import sqlite3
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

os.environ["OPENAI_API_KEY"] = ""

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

def create_agent_SQL(db):
    ##few shot
    examples = [
        {"input": "Có bao nhiêu loại nồi cơm điện", "query": """SELECT COUNT(*) \nFROM data_items \nWHERE GROUP_NAME LIKE \'%nồi cơm điện%\';"""},
        {
            "input": "So sánh Ghế Massage Makano MKGM-10003 với Ghế Massage Daikiosan DKGM-20006",
            "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE '%MKGM-10003%' OR NAME LIKE '%DKGM-20006%';",
        },
        {
            "input": "Máy giặt nào rẻ nhất",
            "query": "SELECT NAME, PRICE\nFROM data_items\nWHERE GROUP_NAME LIKE '%Máy giặt%'\nORDER BY NON_VAT_PRICE_3 ASC\nLIMIT 1;",
        },
        {
            "input": "Công suất của Bàn Ủi Khô Bluestone DIB-3726 1300W",
            "query": "SELECT NAME, DESCRIPTION, SPECIFICATION_BACKUP \nFROM data_items \nWHERE NAME LIKE \'%DIB-3726%\'\nLIMIT 1;",
        },
        {
            "input": "Lò Vi Sóng Bluestone MOB-7716 có thể hẹn giờ trong bao lâu",
            "query": "SELECT NAME, DESCRIPTION, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%MOB-7716%\'\nLIMIT 1;",
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

    ]
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )
    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotPromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
        SystemMessagePromptTemplate,
    )

    system_prefix = """Bạn là một chuyên gia SQLite. Từ một câu hỏi đầu vào, hãy tạo một truy vấn SQLite đúng về mặt cú pháp để chạy, sau đó xem kết quả truy vấn và trả về câu trả lời.
        Bạn có quyền truy cập các tool để tương tác với cơ sở dữ liệu. Chỉ sử dụng các tool nhất định.Chỉ sử dụng thông tin được các công cụ trả về để xây dựng câu trả lời cuối cùng của bạn.
        Bạn PHẢI kiểm tra lại truy vấn của mình trước khi thực hiện nó. Nếu bạn gặp lỗi khi thực hiện truy vấn, hãy viết lại truy vấn và thử lại.
        Sử dụng 'LIKE' thay vì '='. Trừ khi có quy định khác về số lượng kết quả trả ra, nếu không thì trả về giới hạn ở {top_k} kết quả truy vấn. Ưu tiên sử dụng 'SELECT *'để lấy tất cả thông tin về sản phẩm đó.
        Dưới đây là thông tin bảng có liên quan: bảng {table_info} bao gồm các cột: LINK, ID_PRODUCT, GROUP_NAME, CODE, NAME,SHORT_DESCRIPTION, DESCRIPTION, SPECIFICATION_BACKUP, NON_VAT_PRICE_1, VAT_PRICE_1, COMMISSION_1, THRESHOLD_1, NON_VAT_PRICE_2, VAT_PRICE_2, COMMISSION_2, THRESHOLD_2, NON_VAT_PRICE_3, VAT_PRICE_3, COMMISSION_3.
        Dưới đây là một số ví dụ về câu hỏi và truy vấn SQL tương ứng của chúng."""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "table_info", "top_k"],
        prefix=system_prefix,
        suffix="",
    )
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_sql_agent(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        db=db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )
    return agent

# Tiền xử lý dữ liệu
# Số lượng cột cuối cùng cần xóa bỏ
n_columns_to_remove = 1
csv_file_path = 'D:/Chatbot_langchains_openAI/data/113_final_oke.csv'
preprocess_data(csv_file_path, n_columns_to_remove)

# # Tạo bảng SQL trên SQLite db, chạy 1 lần (lần đầu) hàm bên dưới
# create_table_SQL()
# Tạo db
db = SQLDatabase.from_uri("sqlite:///database.db")
# Tạo agent SQL
agent=create_agent_SQL(db)

app = FastAPI()
class TextRequest(BaseModel):
    text: str
@app.post("/chat")
def chat(request: TextRequest):
    response = agent.run({"input": request.text})
    return response
if __name__ == "__main__":
    uvicorn.run(app, port=1234, host='0.0.0.0')