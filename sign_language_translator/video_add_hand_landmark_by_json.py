import json
import os
import cv2
import numpy as np
from types import SimpleNamespace
# import mediapipe as mp

# 使用MP4編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 初始化MediaPipe手部關節檢測器
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# draw_style = mp_drawing.DrawingSpec(
#     color = (255, 192, 0),
#     thickness = 2,
#     circle_radius = 1
#     )
# hands_reorganize = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

class FrameRecord(SimpleNamespace):
    def get_hand_mask(self, mask):
        height, width, _ = mask.shape
        if self.hands:
            if len(self.hands) > 1:
                pass
            for hand in self.hands:
                # 絕對座標
                abs_landmarks = []
                for i in range(21):
                    abs_landmarks.append(
                            (
                            int(hand.landmark[i].x * width),
                            int(hand.landmark[i].y * height) 
                            )
                        )

                def draw_line(start, end):
                    cv2.line(mask, abs_landmarks[start], abs_landmarks[end], (255, 192, 0), 2)

                # Thumb
                draw_line(0,1)
                draw_line(1,2)
                draw_line(2,3)
                draw_line(3,4)
                # Index
                draw_line(0,5)
                draw_line(5,6)
                draw_line(6,7)
                draw_line(7,8)
                # Middle
                draw_line(9,10)
                draw_line(10,11)
                draw_line(11,12)
                # Ring
                draw_line(13,14)
                draw_line(14,15)
                draw_line(15,16)
                # Pinky
                draw_line(0,17)
                draw_line(17,18)
                draw_line(18,19)
                draw_line(19,20)
                # Palm
                draw_line(5,9)
                draw_line(9,13)
                draw_line(13,17)

        return mask


def convert(video_path):
    video_path = '~data/test/1.mp4'
    json_path = '~data/test/1.json'

    frame_datas = None
    with open(json_path, 'r') as file:
        frame_datas = json.load(file, object_hook=lambda d: FrameRecord(**d))

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

    frame = 0
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

if __name__ == '__main__':
    convert('../~data/test/1.mp4')