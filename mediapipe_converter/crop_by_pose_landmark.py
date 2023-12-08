import os
import cv2
import mediapipe as mp 
import numpy as np
import file_enumerate as fe

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編碼器

# 匯入姿勢追蹤模型 
mpPose = mp.solutions.pose # 姿勢追蹤方法

pose = mpPose.Pose(
    static_image_mode=False, # 靜態圖模式，False: 置信度高時繼續跟蹤，True: 實時跟蹤檢測新的結果
    # upper_body_only=True, # 是否只檢測上半身 
    smooth_landmarks=True, # 平滑，一般為True 
    min_detection_confidence=0.5, # 檢測置信度 
    min_tracking_confidence=0.5 # 跟蹤置信度 
    )

# 導入繪圖方法
mpDraw = mp.solutions.drawing_utils 


def crop(input_file_path, output_file_path):
    # 輸入影片
    cap = cv2.VideoCapture(input_file_path) 

    # 設置影片參數
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    video_avg_sum_x, video_avg_sum_y = 0, 0

    while cap.isOpened(): 
        success, img = cap.read() 
        
        if not success:
            break
        
        # 取得影像寬高，不能使用 videoWidth, videoHeight，因為有可能會不同
        imageHeight, imageWidth, _ = img.shape

        frame_count += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        # 取得姿勢標記
        results = pose.process(imgRGB) 
        
        if results.pose_landmarks: 
            landmarks = results.pose_landmarks.landmark

            # left_shoulder, right_shoulder, left_hip, right_hip
            targets = [11, 12, 23, 24]
            target_landmarks = [landmarks[i] for i in targets]

            avg_x, avg_y = np.mean([(lm.x * imageWidth, lm.y * imageHeight,) for lm in target_landmarks], axis=0)
            avg_x, avg_y = int(avg_x), int(avg_y)

            video_avg_sum_x += avg_x
            video_avg_sum_y += avg_y


    video_avg_avg_x = video_avg_sum_x // frame_count
    # 實際上用不到
    # video_avg_avg_y = video_avg_sum_y // frame_count

    # 匯出裁切影片
    cropped_size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (cropped_size, cropped_size))

    # x=身體中心點, y=圖片中心點
    crop_center_x, crop_center_y = video_avg_avg_x, imageHeight // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        imageHeight, imageWidth, _ = img.shape
        imageHalfHeight = imageHeight // 2

        img = img[max(crop_center_y - imageHalfHeight, 0): crop_center_y + imageHalfHeight, max(crop_center_x - imageHalfHeight, 0): crop_center_x + imageHalfHeight, : ]
        if img.shape[1] < cropped_size:
            padded = np.zeros((img.shape[0], max(0, cropped_size - img.shape[1]), img.shape[-1]), dtype=np.uint8)
            if crop_center_x - imageHalfHeight < 0:
                img = np.concatenate((padded, img,), axis=1)
            else:
                img = np.concatenate((img, padded,), axis=1)

        # 寫入影格
        out.write(img)
    cap.release()
    out.release()


input_dir = "../~data/360p"
output_dir = "../~data/cropped"

fe.walk(input_dir, output_dir, "cropped", crop)