"""
將影片轉換成序列圖
"""
import os
import cv2

def convert(input_file_path, output_file_path):
    """
    將影片轉換成序列圖
    """
    cap = cv2.VideoCapture(input_file_path)
    
    frame = 0

    dir_name = os.path.dirname(output_file_path)
    name_without_extension = os.path.splitext(os.path.basename(output_file_path))[0]

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        

        # output file path + 序列號
        frame_name = os.path.join(dir_name, f"{name_without_extension}_f{frame:03d}.jpg")


        # 匯出成 jpg
        cv2.imwrite(frame_name, img)
        
        frame += 1

    cap.release()

convert(
    "~data/test/1.mp4",
    "~data/test-ok/1.mp4",
)