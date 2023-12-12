import json
import cv2
from matplotlib.pylab import f
import mediapipe as mp
import time

class FrameRecord:
    def __init__(self, frame, pose_center, hands):
        self.frame = frame
        self.body_center = pose_center
        self.hands = hands

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

def get_serial_landmark(landmark):
    return {
        "x": round(landmark.x, 3),
        "y": round(landmark.y, 3), 
        "z": round(landmark.z, 3), 
        "v": round(landmark.visibility, 3),
        } 

def videos_to_records(input_file_path, outputs_file_path: str):
    frame = 0
    cap = cv2.VideoCapture(input_file_path)

    serialize_data = []

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame += 1
        pose_data = pose_reorganizer.process(image_rgb)
        hands_data = hands_reorganizer.process(image_rgb)

        serialize_data.append({
            "frame": frame,
            "pose":
            {
                "landmark": 
                [
                    get_serial_landmark(lm)
                    for lm in pose_data.pose_landmarks.landmark
                ],
            } if pose_data.pose_landmarks else None,
            "hands":
            [
                {
                    "landmark":
                    [
                        get_serial_landmark(lm)
                        for lm in hand.landmark
                    ]
                }
                for hand in hands_data.multi_hand_landmarks # hands
            ] if hands_data.multi_hand_landmarks else None,
        })
        
    with open(outputs_file_path, 'w') as file:
        json.dump(serialize_data, file)
    cap.release()


# 讀取 test.png
videos_to_records('~data/test/1.mp4','~data/test/1.json')