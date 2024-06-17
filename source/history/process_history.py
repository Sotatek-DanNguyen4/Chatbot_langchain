import os, json
class HistoryProcessor:
    def __init__(self) -> None:
        pass
    def create_new_history_file_data(self,id_user):
        file_path = f"D:/Chatbot_langchains_openAI/data/data_history/{id_user}.json"
        # Tạo file mới với cấu trúc cơ bản
        data = {
            "conversations": []
        }
        # Lưu file mới
        with open(file_path, 'w',encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        # print(f"Created new file for user {id_user}.")
    def load_history(self,id_user,id_conversation):
        # Đường dẫn tới file JSON của người dùng
        file_path = f"D:/Chatbot_langchains_openAI/data/data_history/{id_user}.json"
        
        # Kiểm tra xem file có tồn tại không, không thì trả về list empty
        if not os.path.exists(file_path):
            return []
        
        # Mở và đọc nội dung file JSON
        with open(file_path, 'r',encoding="utf-8") as file:
            data = json.load(file)
    
        # Tìm cuộc trò chuyện theo id_conversation
        conversations = data.get('conversations', [])
        for conversation in conversations:
            if conversation['id_conversation'] == id_conversation:
                return conversation['messages']
        
        # Nếu không tìm thấy cuộc trò chuyện, trả về None hoặc raise exception
        return []
    def update_history(self,id_user, id_conversation, HumanMessage, AIMessage):
        # Đường dẫn tới file JSON của người dùng
        file_path = f"D:/Chatbot_langchains_openAI/data/data_history/{id_user}.json"
        
        # Nếu file không tồn tại, tạo file mới cho người dùng
        if not os.path.exists(file_path):
            self.create_new_history_file_data(id_user)
        
        # Mở và đọc nội dung file JSON
        with open(file_path, 'r',encoding="utf-8") as file:
            data = json.load(file)
        
        # Tìm hoặc tạo cuộc trò chuyện theo id_conversation
        conversations = data.get('conversations', [])
        for conversation in conversations:
            if conversation['id_conversation'] == id_conversation:
                # Thêm tin nhắn mới vào cuộc trò chuyện
                conversation['messages'].append({"HumanMessage": HumanMessage, "AIMessage": AIMessage})
                break
        else:
            # Nếu không tìm thấy, tạo cuộc trò chuyện mới
            new_conversation = {
                "id_conversation": id_conversation,
                "messages": [{"HumanMessage": HumanMessage, "AIMessage": AIMessage}]
            }
            conversations.append(new_conversation)
        
        # Lưu dữ liệu cập nhật trở lại file JSON
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)