import json
import os
import cv2
import numpy as np
from data_structures import SerializedFrameData, HandInfo, PoseInfo, FrameInfo
# import mediapipe as mp

# 使用MP4編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def convert(video_path):
    video_path = '~data/test/1.mp4'
    json_path = '~data/test/1.json'

    frame_datas = None
    with open(json_path, 'r') as file:
        frame_datas = json.load(file, object_hook=lambda d: SerializedFrameData(**d))

    if frame_datas is None:
        return

    # 檔案路徑
    file_name, file_ext_name = os.path.splitext(video_path)
    convert_video_path = f"{file_name}-converted{file_ext_name}"
    cap = cv2.VideoCapture(video_path)

    # 設置影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(convert_video_path, fourcc, fps, (width, height))

    while True:
        frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while cap.isOpened():
            # 讀取影格
            success, image = cap.read()
            if not success:
                break

            data = frame_datas[frame]
            frame += 1

            # 將影格轉換為RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 建立一個灰色影像用以輸出
            output_img = np.zeros(image.shape, np.uint8)
            output_img[:] = (64, 64, 64)

            mask = np.zeros_like(image_rgb)
            mm = data.get_hand_mask(mask)
            img_with_mask = cv2.add(image_rgb, mm)
            img_with_mask = cv2.cvtColor(img_with_mask, cv2.COLOR_RGB2BGR)

            # 寫入影格輸出
            out.write(img_with_mask)

            # 顯示預覽
            cv2.imshow('Hand Gesture Recognition', img_with_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 釋放資源
    cap.release()
    out.release()
    # 關閉預覽視窗
    cv2.destroyAllWindows()

convert('../~data/test/1.mp4')