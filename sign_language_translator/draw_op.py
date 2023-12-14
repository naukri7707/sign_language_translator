
import cv2
from data_structures import FrameInfoContainer

cap = cv2.VideoCapture("~data/test/1.mp4")
container = FrameInfoContainer.load("~data/test/1.json")

while True:
    frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame += 1

        frame_info = container.get_info(frame - 1)

        hand_mask = frame_info.create_hand_mask(image_rgb)

        image_with_mask = cv2.add(image_rgb, hand_mask)

        output_image = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)
        # 顯示預覽
        cv2.imshow('Hand Gesture Recognition', output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()