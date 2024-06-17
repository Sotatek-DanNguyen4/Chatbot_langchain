import csv,sqlite3,re,ast
class SQLCreatorServer:
    def __init__(self) -> None:
        pass
    def create_table_SQLite(self, csv_file_path_local, name_table):
        # # Đường dẫn mẫu
        # csv_file_path = 'D:/Chatbot_langchains_openAI/data/113_final_oke_edited.csv'
        # Kết nối đến cơ sở dữ liệu SQLite hoặc tạo một cơ sở dữ liệu mới
        conn = sqlite3.connect('database_204_items.db')
        cursor = conn.cursor()
        # Tạo bảng trong cơ sở dữ liệu,ví dụ name_table ='data_items'
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS {name_table} (
                            id INTEGER PRIMARY KEY,
                            ID_PRODUCT TEXT,
                            GROUP_NAME TEXT,
                            CODE TEXT,
                            NAME TEXT,
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
                            COMMISSION_3 INTEGER,
                            RAW_PRICE INTEGER,
                            QUANTITY_SOLD INTEGER
                            )''')

        # Đọc dữ liệu từ tập tin CSV và chèn vào bảng
        with open(csv_file_path_local, 'r',encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Bỏ qua dòng tiêu đề nếu có
            for row in csv_reader:
                cursor.execute('''INSERT INTO data_items (LINK, ID_PRODUCT, GROUP_NAME, CODE,
                                NON_VAT_PRICE_1,VAT_PRICE_1,COMMISSION_1,THRESHOLD_1,
                                NON_VAT_PRICE_2,VAT_PRICE_2,COMMISSION_2,THRESHOLD_2,
                                NON_VAT_PRICE_3,VAT_PRICE_3,COMMISSION_3,
                                NAME,SPECIFICATION_BACKUP,
                                RAW_PRICE,QUANTITY_SOLD)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?)''', row)
        # Lưu thay đổi và đóng kết nối
        conn.commit()
        conn.close()

    # lọc các cột danh từ cho retriever tool trong SQL agent
    def query_noun(self, db_sql, query):
        res = db_sql.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return list(set(res))
    def load_db (self, path_db):
        pass