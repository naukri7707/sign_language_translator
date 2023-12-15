import cv2
import mediapipe as mp
import file_walker as fw
from data_structures import FrameInfo, FrameInfoContainer

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

def videos_to_lmdatas(input_file_path, outputs_file_path: str):
    
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
            previous = container.last(),
            frame = frame,
            mp_pose_data = pose_data,
            mp_hands_data = hands_data
        )
        container.append(frame_info)

    FrameInfoContainer.dump(container, outputs_file_path)
    cap.release()


fw.walk(
    '~data/2_360p',
    '~data/3_lmdata',
    'lmdata', 
    videos_to_lmdatas,
    output_extension='.json',
    skip=6439
    )