import os

# 設定來源和目標資料夾的路徑
root_folder = "./data/input"

keyfolderName = "Source"

# 遞迴處理資料夾中的檔案
for input_folder_path, dirs, files in os.walk(root_folder, topdown=False):
    # 如果資料夾中包含 keyfolderName 將內容移動到上一層資料夾

    if keyfolderName in input_folder_path:
        # 將內容移動到上一層資料夾
        for full_file_name in files:
            file_name, file_extension = os.path.splitext(full_file_name)
            input_file_path = os.path.join(input_folder_path, full_file_name)
            output_file_path = os.path.join(input_folder_path, "..", full_file_name)
            os.rename(input_file_path, output_file_path)
            print(f"moved: {files}")
    
    print(f"completed: {input_folder_path}")
