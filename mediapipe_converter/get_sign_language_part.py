from math import isnan, nan
import re
from typing import List
import numpy as np
import file_enumerate as fe
import cv2
import mediapipe as mp

# 連續片段無法辨識容許值
max_undetected_frame = 10
total_offset_transhold = 10

# 初始化MediaPipe手部關節檢測器
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# 使用MP4編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class FrameHands:
    def __init__(self, frame: int, hands):
        self.frame = frame
        self.hands = hands
        self.hands_avg = [
                self.__get_hand_landmark_avg(idx)
                for idx in range(len(self.hands))
            ]
        
        # sort by avg_x
        if len(self.hands) > 1:
            sorted_data = sorted(zip(self.hands, self.hands_avg), key=lambda x: x[1][0])
            self.hands, self.hands_avg = zip(*sorted_data)
        pass

    def min_avg_y(self):

        if len(self.hands_avg) == 0:
            return nan
        return min([avg[1] for avg in self.hands_avg])
    
    def get_max_offset_hand(self, other):
        length = min(len(self.hands), len(other.hands))
        max_offset = 0
        for idx in range(length):
            for (self_hand_landmark, other_hand_landmark) in zip(self.hands[idx].landmark, other.hands[idx].landmark):
                offset_x = abs(self_hand_landmark.x - other_hand_landmark.x)
                offset_y = abs(self_hand_landmark.y - other_hand_landmark.y)
                offset_z = abs(self_hand_landmark.z - other_hand_landmark.z)
                
                offset = offset_x + offset_y + offset_z
                max_offset = max(max_offset, offset)

        return max_offset

    def __get_hand_landmark_avg(self, hand_index):
        hand = self.hands[hand_index]
        return np.mean([
            [pt.x, pt.y, pt.z] for pt in hand.landmark
        ],axis=0)
    

def get_part(input_file_path, output_file_path):


    # 檔案路徑
    cap = cv2.VideoCapture(input_file_path)

    # 設置影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
   
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    frame = 0

    frameHands: List[FrameHands] = []

    while cap.isOpened():
        # 讀取影格
        success, img = cap.read()
        if not success:
            break

        frame += 1

        # 將影格轉換為RGB格式
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 使用手部關節檢測模型進行辨識
        results = hands.process(image_rgb)
        hand_detected = results.multi_hand_landmarks is not None

        detected_multi_hand_landmarks = results.multi_hand_landmarks if hand_detected else []
        instance = FrameHands(frame, detected_multi_hand_landmarks)
        frameHands.append(instance)

    # 輸出最佳片段
    frame = 0 # 影格計數器歸零
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    gray_scale = True

    while cap.isOpened():
        # 讀取影格
        success, img = cap.read()
        if not success:
            break

        min_avg_y = frameHands[frame].min_avg_y()
        print(min_avg_y)

        frame += 1

        # 將 offset 印在影格左上
        cv2.putText(img, str(min_avg_y), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 如果 y > 0.95 將影格灰階化
        if min_avg_y > 0.95:
            gray_scale = True
        # 用 elif 因為有 nan 的可能
        elif min_avg_y < 0.95: 
            gray_scale = False

        if gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 寫入影格輸出
        out.write(img)
    cap.release()
    out.release()

input_dir = "../~data/test"
output_dir = "../~pdata"

fe.walk(input_dir, output_dir, "part", get_part)