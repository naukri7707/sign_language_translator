import json
import file_walker as fw
from typing import List

# 讀取資料庫
def read_database(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        serialized_data  = json.load(file)
        return serialized_data

# 寫入資料庫
def write_database(data, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

file_path = '~data/1~10/3_lmdata/1/1_方思雯_front_360p_lmdata.json'
data = read_database(file_path)


def rename(input_file_path, outputs_file_path: str):
    data = read_database(input_file_path)
    for fi in data["frame_infos"]:
        pose_info = fi["pose_info"]
        landmarks = pose_info.get("landmark")   # 取得 "landmark" 欄位的值
        if landmarks is not None:
            pose_info["landmarks"] = landmarks  # 將資料移到 "landmarks"
            del pose_info["landmark"]           # 刪除原始的 "landmark" 欄位

        hand_infos = fi["hand_infos"]
        for hand in hand_infos:
            landmarks = hand.get("landmark")    # 取得 "landmark" 欄位的值
            if landmarks is not None:
                hand["landmarks"] = landmarks   # 將資料移到 "landmarks"
                del hand["landmark"]            # 刪除原始的 "landmark" 欄位

    write_database(data, outputs_file_path)

fw.walk_old(
    '~data/1~10/3_lmdata',
    '~data/1~10/3_1_lmdata',
    "named",
    rename
)