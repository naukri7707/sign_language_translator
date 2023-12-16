
import cv2
import numpy as np
import mask_drawer as md

from models import FrameInfoContainer

# 由於影片幀率為 58 fps，因此每 3 幀取一次 ~= 0.05 秒 / 次
step = 3

# 尾巴長度，這會追蹤前 tail_frame_count 幀的資訊來產生
tail_frame_count = 7

cap = cv2.VideoCapture("~data/1~10/2_360p/1/1_方思雯_front_360p.mp4")
container = FrameInfoContainer.load("~data/1~10/3_1_lmdata/1/1_方思雯_front_360p_lmdata_named.json", step)

np_frame_info = np.array(container.frame_infos)

def is_hand_start(frame_info):
    if frame_info.hand_infos:
        for hand in frame_info.hand_infos:
            if hand.get_avg_y() < 0.95:
                return True
    return False


frame_infos = container.frame_infos
# 取出第一個手勢出現的幀數，額外加一幀使其可以產生光流
start_idx = next((index for index, fi in enumerate(frame_infos) if is_hand_start(fi)), None) + 1
# 取出最後一個手勢出現的幀數
end_idx = (len(frame_infos) - next((index for index, fi in enumerate(reversed(frame_infos)) if is_hand_start(fi)), None))

while True:
    frame = 0
    stepped_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame += 1

        if frame % step != 0:
            continue
        
        stepped_frame += 1

        if stepped_frame < start_idx:
            continue
        if stepped_frame > end_idx:
            break

        frame_info = container.frame_infos[stepped_frame - 1]
        
        hand_mask = md.create_hand_mask(frame_info, image_rgb)
        optical_flow_mask = md.create_optical_flow_mask(frame_info, image_rgb, tail_frame_count)
        
        image_black = np.zeros_like(image_rgb)

        image_with_mask = image_black
        image_with_mask = cv2.add(image_with_mask, hand_mask)
        image_with_mask = cv2.add(image_with_mask, optical_flow_mask)
        output_image = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)

        # 顯示預覽
        cv2.imshow('Hand Gesture Recognition', output_image)
        if cv2.waitKey(-1) & 0xFF == ord('q'):
            break

cap.release()