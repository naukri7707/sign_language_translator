import os
import cv2
import numpy as np
import mediapipe as mp

# 使用MP4編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 初始化MediaPipe手部關節檢測器
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
draw_style = mp_drawing.DrawingSpec(
    color = (255, 192, 0),
    thickness = 2,
    circle_radius = 1
    )
hands_reorganize = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def convert(video_path):
    # 檔案路徑
    file_name, file_ext_name = os.path.splitext(video_path)
    convert_video_path = f"{file_name}-converted{file_ext_name}"
    cap = cv2.VideoCapture(video_path)

    # 設置影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(convert_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # 讀取影格
        success, image = cap.read()
        if not success:
            break

        # 將影格轉換為RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 建立一個灰色影像用以輸出
        output_img = np.zeros(image.shape, np.uint8)
        output_img[:] = (64, 64, 64)

        # 使用手部關節檢測模型進行辨識
        results = hands_reorganize.process(image_rgb)

        # 繪製手部關節
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output_img,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    draw_style
                    )
        else:
            print("No hand")
        # 寫入影格輸出
        out.write(output_img)

        # 顯示預覽
        cv2.imshow('Hand Gesture Recognition', output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    # 關閉預覽視窗
    cv2.destroyAllWindows()

if __name__ == '__main__':
    convert('../~data/test/1.mp4')