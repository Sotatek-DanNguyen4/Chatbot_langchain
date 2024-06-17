import csv
class DataProcessor_CSV:
    def __init__(self):
        pass
    def remove_columns_last(self, original_file_path, destination_file_path, num_col_delete):
        with open(original_file_path, 'r',encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = [row[1:-num_col_delete] for row in csv_reader]  # Xóa bỏ các cột cuối cùng được chỉ định từ mỗi dòng
        # Ghi dữ liệu đã chỉnh sửa vào tập tin CSV mới
        with open(destination_file_path, 'w', newline='',encoding="utf-8") as new_file:
            csv_writer = csv.writer(new_file)
            csv_writer.writerows(data)