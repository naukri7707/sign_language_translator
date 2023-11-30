import os
import time
import cv2
import mediapipe as mp

# 初始化MediaPipe手部關節檢測器
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def convert(video_path):
    start_time = time.time()
    # 檔案路徑
    file_name, file_ext_name = os.path.splitext(video_path)
    convert_video_path = f"{file_name}-converted{file_ext_name}"
    cap = cv2.VideoCapture(video_path)

    # 設置影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編碼器
    out = cv2.VideoWriter(convert_video_path, fourcc, fps, (width, height))

    # 初始化手部關節檢測模型
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
        while cap.isOpened():
            # 讀取影格
            success, image = cap.read()
            if not success:
                break

            # 將影格轉換為RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 使用手部關節檢測模型進行辨識
            results = hands.process(image_rgb)

            # 繪製手部關節
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                print("No hand")
                a = 10
            # 寫入影格輸出
            out.write(image)

            # 顯示預覽
            # cv2.imshow('Hand Gesture Recognition', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    cap.release()
    out.release()
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"[Done] Time cost:{cost_time}")
    # 關閉預覽視窗
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    convert('./data/input/1_front_480.mp4')