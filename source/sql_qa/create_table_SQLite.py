import os ,sys
# # Get the absolute path of the project directory
# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(project_path)
# # Add the project directory to sys.path
# sys.path.append(project_path)
from sql_creator import SQLCreatorServer
status= True #"Lần đầu" = True 
if status:
    SQLCreatorServer().create_table_SQLite(csv_file_path_local="D:/Chatbot_langchains_openAI/data/data_csv/204_final_edited.csv",
                                       name_table="data_items")