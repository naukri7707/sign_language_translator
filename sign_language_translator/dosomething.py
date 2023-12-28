import glob
import os
from pathlib import Path
import helpers.singword_db as swdb

db = swdb.read_database('words_db.json')

class_ids_for_name = {}

for data in db:
    class_ids_for_name[data.zh_name] = data.class_id

# 列出目標資料夾下的所有子資料夾
root_folder = '~data/dataset/src_lmdata'

# 取得所有 .mp4 檔案
mp4_files = glob.glob(f'{root_folder}/**/*.json')  # 替換成你的目標資料夾路徑

# 如果有找到 .mp4 檔案
for file_path in mp4_files:
    file_name = os.path.basename(file_path)  # 取得檔案名稱

    # 以底線分隔檔名
    name_parts = file_name.split('_')  # 使用底線分割檔名
    
    # 判斷分割後的首項
    label = name_parts[0]

    if not label in class_ids_for_name:
        if label == '松鼠':
            new_path = file_path.replace('松鼠', '松鼠(A)')
            Path(new_path).parent.mkdir(parents=True, exist_ok=True)
            os.rename(file_path, new_path)
            continue
        pass

    # 在這裡進行首項的判斷和後續的處理
    print("分割後的首項：", label)