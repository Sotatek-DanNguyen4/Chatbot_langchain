from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
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
from pydantic import BaseModel
from source.sql_qa.sql_creator import SQLCreatorServer
from source.history.process_history import HistoryProcessor
# from langchain.pydantic_v1 import BaseModel, Field

# class HumanInput(BaseModel):
#     question_human: str = Field(description="Là câu hỏi của người dùng, không chỉnh sửa cắt bỏ hay thêm bớt gì")

class AgentCreatorSQL:
    def __init__(self) -> None:
        pass
    def create_SQL_agent(self,llm, model_embed, db_sql):
        ##few shot
        examples = [
            {"input": "Có bao nhiêu loại nồi cơm điện", "query": """SELECT COUNT(*) \nFROM data_items \nWHERE GROUP_NAME LIKE \'%nồi cơm điện%\';"""},
            {
                "input": "So sánh Ghế Massage Makano MKGM-10003 với Ghế Massage Daikiosan DKGM-20006",
                "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE '%MKGM-10003%' OR NAME LIKE '%DKGM-20006%';",
            },
            {
                "input": "Máy giặt nào rẻ nhất",
                "query": "SELECT NAME, PRICE\nFROM data_items\nWHERE GROUP_NAME LIKE '%Máy giặt%' OR SPECIFICATION_BACKUP LIKE '%Máy giặt%' OR NAME LIKE \'%máy giặt%\' \nORDER BY RAW_PRICE ASC\nLIMIT 1;",
            },
            {
                "input": "Công suất của Bàn Ủi Khô Bluestone DIB-3726 1300W",
                "query": "SELECT NAME, SPECIFICATION_BACKUP \nFROM data_items \nWHERE NAME LIKE \'%DIB-3726%\' OR SPECIFICATION_BACKUP LIKE \'%DIB-3726%\'\nLIMIT 1;",
            },
            {
                "input": "Lò Vi Sóng Bluestone MOB-7716 có thể hẹn giờ trong bao lâu",
                "query": "SELECT NAME, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%MOB-7716%\' OR SPECIFICATION_BACKUP LIKE \'%MOB-7716%\'\nLIMIT 1;",
            },
            {
                "input": "Máy NLMT Empire 180 Lít Titan M&EGD000224 có bao nhiêu ống",
                "query": "SELECT SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%M&EGD000224%\' OR NAME LIKE \'%Empire%\' OR SPECIFICATION_BACKUP LIKE \'%M&EGD000224%\' \nLIMIT 3;",
            },
            {
                "input": "có hình ảnh nồi KL-619 không",
                "query": "SELECT * \nFROM data_items \nWHERE NAME LIKE \'%KL-619%\'\nLIMIT 1;",
            },
            {
                "input": "Đèn đường năng lượng mặt trời SUNTEK S500 PLUS, công suất 500W đắt quá, có cái nào rẻ hơn không",
                "query": "SELECT * FROM (SELECT * FROM data_items \nWHERE NAME LIKE '%Đèn đường năng lượng mặt trời SUNTEK S500 PLUS%' \nUNION \nSELECT * FROM (SELECT * FROM data_items \nWHERE NAME LIKE '%Đèn đường năng lượng mặt trời%' \nORDER BY RAW_PRICE ASC \nLIMIT 3)) AS combined_results;",
            #lấy sản phẩm kêu đắt đi so sánh với top 3 sản phẩm khác cùng loại để trả lời
            },
            {
                "input": "Máy giặt lồng dọc có thông số như thế nào",
                "query": "SELECT NAME, SPECIFICATION_BACKUP\nFROM data_items \nWHERE NAME LIKE \'%máy giặt lồng dọc%\' OR SPECIFICATION_BACKUP LIKE '%máy giặt lồng dọc%' NAME LIKE \'%Máy giặt lồng dọc%\' OR SPECIFICATION_BACKUP LIKE '%Máy giặt lồng dọc%'\nLIMIT 3;",
            },
            {
                "input": "Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE đã bán bao nhiêu sản phẩm",
                "query": "SELECT QUANTITY_SOLD\nFROM data_items \nWHERE NAME LIKE \'%Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE%\' OR SPECIFICATION_BACKUP LIKE \'%Bình nước nóng gián tiếp 30 lít SL2 30 B 2.5 FE%\' \nLIMIT 1;"
            },
            {
                "input": "Giá gốc của sản phẩm Bếp từ đơn AIO Smart kèm nồi",
                "query": "SELECT RAW_PRICE\nFROM data_items \nWHERE NAME LIKE \'%Bếp từ đơn AIO Smart kèm nồi%\' OR SPECIFICATION_BACKUP LIKE \'%Bếp từ đơn AIO Smart kèm nồi%\' \nLIMIT 1;"
            }
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
        Đây là thông tin về bảng có liên quan: bảng {table_info} bao gồm các cột: LINK, ID_PRODUCT, GROUP_NAME, CODE, NAME,SPECIFICATION_BACKUP, NON_VAT_PRICE_1, VAT_PRICE_1, COMMISSION_1, THRESHOLD_1, NON_VAT_PRICE_2, VAT_PRICE_2, COMMISSION_2, THRESHOLD_2, NON_VAT_PRICE_3, VAT_PRICE_3, COMMISSION_3, RAW_PRICE, QUANTITY_SOLD.
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
        names = SQLCreatorServer().query_noun(db_sql, "SELECT Name FROM data_items")
        groups = SQLCreatorServer().query_noun(db_sql, "SELECT GROUP_NAME FROM data_items")
        vector_db_retriever_nouns = FAISS.from_texts(names + groups, model_embed)
        retriever_nouns = vector_db_retriever_nouns.as_retriever(search_kwargs={"k": 3})
        description = """Sử dụng để tra cứu các giá trị cần lọc. Đầu vào là cách viết gần đúng của danh từ, cụm danh từ biểu thị tên sản phẩm hoặc nhóm sản phẩm,
        đầu ra là danh từ, cụm danh từ hợp lệ. Sử dụng danh từ giống nhất với tìm kiếm."""
        # Tạo retriever tool để lọc các danh từ, cụm danh từ 
        retriever_tool = create_retriever_tool(
            retriever_nouns,
            name="search_noun_phrases",
            description=description,
            )
        agent_sql = create_sql_agent(
            llm, # ví dụ llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            db=db_sql,
            extra_tools=[retriever_tool],
            prompt=full_prompt,
            verbose=True,
            agent_type="openai-tools",
        )
        return agent_sql