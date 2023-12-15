
import cv2
from draw import create_hand_mask, create_optical_flow_mask

from models import FrameInfoContainer

cap = cv2.VideoCapture("~data/test/1.mp4")
step = 3
container = FrameInfoContainer.load("~data/test/1.json", step)

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

        frame_info = container.get_info(stepped_frame - 1)

        hand_mask = create_hand_mask(frame_info, image_rgb)
        optical_flow_mask = create_optical_flow_mask(frame_info, image_rgb, 7)

        image_with_mask = cv2.add(image_rgb, hand_mask)
        image_with_mask = cv2.add(image_with_mask, optical_flow_mask)

        output_image = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)
        # 顯示預覽
        cv2.imshow('Hand Gesture Recognition', output_image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

cap.release()