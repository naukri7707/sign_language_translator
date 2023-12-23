from typing import List
import cv2
import mediapipe as mp
from preprocess.models import FrameInfo, FrameInfoContainer

# 手部骨架檢測器
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
draw_style = mp_drawing.DrawingSpec(
    color = (255, 192, 0),
    thickness = 2,
    circle_radius = 1
    )
hands_reorganizer = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# 姿勢骨架檢測器
mp_pose = mp.solutions.pose # 姿勢追蹤方法
pose_reorganizer = mp_pose.Pose(
    static_image_mode=False,      # 靜態圖模式，False: 置信度高時繼續跟蹤，True: 實時跟蹤檢測新的結果
    # upper_body_only=True,       # 是否只檢測上半身 
    smooth_landmarks=True,        # 平滑，一般為True 
    min_detection_confidence=0.5, # 檢測置信度 
    min_tracking_confidence=0.5   # 跟蹤置信度 
    )

def generate_lmdata(input_file_path, mapping_paths: List[str]):

    output_file_path = mapping_paths[0]

    cap = cv2.VideoCapture(input_file_path)
    frame = 0
    container = FrameInfoContainer()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame += 1
        pose_data = pose_reorganizer.process(image_rgb)
        hands_data = hands_reorganizer.process(image_rgb)

        frame_info = FrameInfo.from_mp_data(
            previous = container.frame_infos[-1] if container.frame_infos and len(container.frame_infos) > 0 else None,
            frame = frame,
            mp_pose_data = pose_data,
            mp_hands_data = hands_data
        )
        container.append(frame_info)

    FrameInfoContainer.dump(container, output_file_path)
    cap.release()