import file_enumerate as fe
import cv2
import mediapipe as mp

# 連續片段無法辨識容許值
max_undetected_frame = 10

# 初始化MediaPipe手部關節檢測器
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# 使用MP4編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def get_part(input_file_path, output_file_path):


    # 檔案路徑
    cap = cv2.VideoCapture(input_file_path)

    # 設置影片參數
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
   
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    frame = 0

    best_start_frame = -1
    best_frame_count = 0

    part_start_frame = -1
    part_frame_count = 0
    part_undetected_counter = 0
        
    while cap.isOpened():
        # 讀取影格
        success, img = cap.read()
        if not success:
            break

        # 將影格轉換為RGB格式
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 使用手部關節檢測模型進行辨識
        results = hands.process(image_rgb)

        hand_detected = results.multi_hand_landmarks is not None

        frame += 1
        # 只有在片段開始時才會更新片段影格數
        if part_start_frame != -1:
            part_frame_count += 1
            part_undetected_counter += 1 if hand_detected else 0 

        if hand_detected:
            # 如果片段尚未開始記錄，則從此影格開始記錄
            if part_start_frame == -1:
                part_start_frame = frame
                part_frame_count = 0
        else:
            # 如果連續 max_undetected_frame 個影格都無法辨識，則判斷片段結束
            if part_undetected_counter > max_undetected_frame:
                # 如果片段影格數大於最佳片段，則更新最佳片段
                if part_frame_count > best_frame_count:
                    # 更新最佳片段
                    best_start_frame = part_start_frame
                    best_frame_count = part_frame_count
                    # 重置片段影格計數器
                    part_start_frame = -1

    if(best_start_frame == -1):
        best_start_frame = part_start_frame
        best_frame_count = part_frame_count
    # 輸出最佳片段
    frame = 0 # 影格計數器歸零
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        # 讀取影格
        success, img = cap.read()
        if not success:
            break

        frame += 1

        # 如果影格不在最佳片段範圍內，則跳過
        if frame < best_start_frame:
            continue
        # 如果影格超過最佳片段範圍，則結束
        elif frame > best_start_frame + best_frame_count:
            break

        # 將影格轉換為RGB格式
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 使用手部關節檢測模型進行辨識
        results = hands.process(image_rgb)

        hand_detected = results.multi_hand_landmarks is not None

        # 如果影格辨識成功，則繪製手部關節
        if hand_detected:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 寫入影格輸出
        out.write(img)
    cap.release()
    out.release()

input_dir = "../~data"
output_dir = "../~pdata"

fe.walk(input_dir, output_dir, "part", get_part)