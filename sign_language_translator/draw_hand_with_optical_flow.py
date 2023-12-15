
import cv2
import numpy as np
from mask_drawer import create_hand_mask, create_optical_flow_mask

from models import FrameInfoContainer

# 由於影片幀率為 58 fps，因此每 6 幀取一次 ~= 0.1 秒 / 次
step = 6

# 尾巴長度，這會追蹤前 tail_frame_count 幀的資訊來產生
tail_frame_count = 7

cap = cv2.VideoCapture("~data/2_360p/睡袋/睡袋_方思雯_front_360p.mp4")
container = FrameInfoContainer.load("~data/3_lmdata/睡袋/睡袋_方思雯_front_360p_lmdata.json", step)

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
        frame_info = container.frame_infos[stepped_frame - 1]
        
        hand_mask = create_hand_mask(frame_info, image_rgb)
        optical_flow_mask = create_optical_flow_mask(frame_info, image_rgb, tail_frame_count)
        
        zero = np.zeros_like(image_rgb)

        image_with_mask = image_rgb
        image_with_mask = cv2.add(image_with_mask, hand_mask)
        image_with_mask = cv2.add(image_with_mask, optical_flow_mask)

        output_image = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)

        vis = sum([hand.get_avg_visibility() for hand in frame_info.hand_infos]) / len(frame_info.hand_infos) \
              if frame_info.hand_infos else 0
        
        # 顯示預覽
        cv2.imshow('Hand Gesture Recognition', output_image)
        if cv2.waitKey(-1) & 0xFF == ord('q'):
            break

cap.release()