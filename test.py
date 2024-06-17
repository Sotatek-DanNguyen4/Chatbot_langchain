# from source.history.process_history import HistoryProcessor
# # Ví dụ sử dụng
# if __name__ == "__main__":
#     id_user = "1120042"
#     id_conversation = "1"
#     x=HistoryProcessor()
#     # Tải lịch sử cuộc trò chuyện
#     history = HistoryProcessor().load_history(id_user, id_conversation)
#     print(f"History for conversation {id_conversation}:", history)
    
#     # Cập nhật lịch sử cuộc trò chuyện
#     HistoryProcessor().update_history(id_user=id_user, id_conversation=id_conversation,HumanMessage= "user", AIMessage="How are you?")
#     print(f"Updated history for conversation {id_conversation}:")
    
#     # Tải lại lịch sử cuộc trò chuyện để kiểm tra cập nhật
#     updated_history = HistoryProcessor().load_history(id_user=id_user, id_conversation=id_conversation)
#     print(updated_history)

