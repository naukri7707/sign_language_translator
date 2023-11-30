import os
import json

# 設定來源和目標資料夾的路徑
root_folder = "./data/input"

remove_renamer_file = True

# 遞迴處理資料夾中的影片檔案
for input_folder_path, dirs, files in os.walk(root_folder, topdown=False):
    # 如果目錄中包含 "renamer.txt" 則進入
    
    if "renamer.json" in files:
        renamer_file_path = os.path.join(input_folder_path, "renamer.json")
        # 讀取 JSON 檔案內容
        with open(renamer_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # 將該資料夾中的所有檔案重新命名
            # 規則為 : 將名稱片段中包含的 key 改為 value
            for full_file_name in files:
                file_name, file_extension = os.path.splitext(full_file_name)

                new_file_name = file_name
                for key, value in data.items():
                    new_file_name = new_file_name.replace(key, value)
                new_file_name = f"{new_file_name}{file_extension}"

                old_file_path = os.path.join(input_folder_path, full_file_name)
                new_file_path = os.path.join(input_folder_path, new_file_name)

                os.rename(old_file_path, new_file_path)
                print(f"renamed: {full_file_name} to {new_file_name}")
        
        if remove_renamer_file:
            os.remove(renamer_file_path)

        print(f"completed: {input_folder_path}")