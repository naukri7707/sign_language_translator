import os
import ffmpeg

# 設定來源和目標資料夾的路徑
input_root_folder = "./data/input"
output_root_folder = "./data/output"

# 保存原始 print 函式
original_print = print
def disable_print(*args, **kwargs):
    pass

# 遞迴處理資料夾中的影片檔案
for input_folder_path, dirs, files in os.walk(input_root_folder, topdown=False):
    
    # 產生映射路徑
    sub_folder_dir = os.path.relpath(input_folder_path, input_root_folder)
    output_folder_path = os.path.join(output_root_folder, sub_folder_dir)
    os.makedirs(output_folder_path, exist_ok=True)

    for full_file_name in files:
        file_name, file_extension = os.path.splitext(full_file_name)

        input_file_path = os.path.join(input_folder_path, full_file_name)

        output_compressed_file_path = os.path.join(output_folder_path, f"{file_name}_480p{file_extension}")
        output_compressed_hflipped_file_path = os.path.join(output_folder_path, f"{file_name}_hflipped_480p{file_extension}")

        # 暫時關閉 print 以避免 ffmpeg 輸出訊息
        print = disable_print

        # 壓縮至 480p 並移除聲音
        (
            ffmpeg
            .input(input_file_path)
            .output(output_compressed_file_path, vf="scale=854:480", an=None)
            .run()
        )

        # 水平翻轉壓縮後的影片
        (
            ffmpeg
            .input(output_compressed_file_path)
            .output(output_compressed_hflipped_file_path, vf="hflip", acodec="copy")
            .run()
        )

        # 還原 print
        print = original_print

        print(f"completed: {input_file_path}")
    
    print(f"completed: {input_folder_path}")
    print("-----------------------------------")
