from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import csv
# # load data csv
# loader = CSVLoader(file_path='D:/Chatbot_langchains_openAI/data/204_final_edited.csv')
# data = loader.load()
# # Split data
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
# texts = text_splitter.split_documents(data)

# Xử lý file CSV 113 sản phẩm( data mới) thành dạng text #####################################
def csv2txt(csv_link):
    data_text = ''
    with open(csv_link, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Lấy thông tin từ mỗi hàng của file CSV
            name = row['PRODUCT_NAME']  # Thay 'Tên Sản Phẩm' bằng tên cột chứa tên sản phẩm trong file CSV của bạn
            id = row['PRODUCT_INFO_ID']  # Thay 'ID' bằng tên cột chứa ID sản phẩm trong file CSV của bạn
            code = row['PRODUCT_CODE']  # Thay 'Code' bằng tên cột chứa mã code sản phẩm trong file CSV của bạn
            group = row['GROUP_PRODUCT_NAME']  # Thay 'Nhóm' bằng tên cột chứa nhóm sản phẩm trong file CSV của bạn
            spec_prd=row['SPECIFICATION_BACKUP']
            link = row['LINK_SP']
            nv1 = row['NON_VAT_PRICE_1']
            v1 = row['VAT_PRICE_1']
            comm1 = row['COMMISSION_1']
            thresh_hold_1=row['THRESHOLD_1'].lower()
            nv2 = row['NON_VAT_PRICE_2']
            v2 = row['VAT_PRICE_2']
            comm2 = row['COMMISSION_2']
            thresh_hold_2=row['THRESHOLD_2'].lower()
            nv3 = row['NON_VAT_PRICE_3']
            v3 = row['VAT_PRICE_3']
            comm3 = row['COMMISSION_3']
            raw_price=row['RAW_PRICE']
            quantity_sold=['QUANTITY_SOLD']
            # In ra văn bản theo định dạng mong muốn
            s = f"Sản phẩm \"{name}\" có ID là {id} và mã sản phẩm(mã Code) là {code}.Sản phẩm \"{name}\" thuộc nhóm \" {group} \".Thông số kỹ thuật của sản phẩm \"{name}\": {spec_prd}. Liên kết(Link) của sản phẩm \"{name}\" là {link}. Về giá của sản phẩm \"{name}\": Nếu tổng {thresh_hold_1} thì giá sản phẩm \"{name}\" không bao gồm VAT là {nv1}, giá sản phẩm \"{name}\" bao gồm VAT là {v1} và tiền hoa hồng sản phẩm \"{name}\" là {comm1}, nếu tổng {thresh_hold_2} thì giá sản phẩm \"{name}\" không bao gồm VAT là {nv2}, giá sản phẩm \"{name}\" bao gồm VAT là {v2} và tiền hoa hồng sản phẩm \"{name}\" là {comm2}, nếu tổng giá trị đơn hàng dưới mức {thresh_hold_2} thì giá sản phẩm \"{name}\" không bao gồm VAT là {nv3}, giá sản phẩm \"{name}\" bao gồm VAT là {v3} và tiền hoa hồng sản phẩm \"{name}\" là {comm3}. Giá gốc của sản phẩm \"{name}\" là {raw_price}. Số lượng sản phẩm \"{name}\" đã bán là {quantity_sold}."
            s = s.replace('\n', ' ')
            s = s.replace('  ', ',')
            s = s.replace('..', ',')
            data_text = data_text + s + '.'
            # print(s)
    return data_text

data_text = csv2txt('D:/Chatbot_langchains_openAI/data/204_final_edited.csv')

# apikeys openAI
import os
os.environ["OPENAI_API_KEY"] = ""
# create model embedding in openAi
model_embed = OpenAIEmbeddings(model="text-embedding-3-large",request_timeout=120, max_retries=10)

documents = data_text
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=1024,
    chunk_overlap=64,
    length_function=len
)
# print(len(data_text))
chunks = text_splitter.split_text(documents)
embeddings = model_embed
db = FAISS.from_texts(chunks, embeddings)

# Embeding
# embedding_model = GPT4AllEmbeddings(model_file="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.from_documents(chunks, embedding_model)
db.save_local("D:/Chatbot_langchains_openAI/vectorstore_Chatbot/204_items_db")