
import cv2
import numpy as np

from preprocess.models import FrameInfo, HandInfo

def create_pose_mask(frame_info: FrameInfo, src_img: cv2.typing.MatLike, draw_on_src: bool = False) -> cv2.typing.MatLike:
    mask = src_img.copy() if draw_on_src else np.zeros_like(src_img) 
    height, width, _ = mask.shape

    if frame_info.pose_info:
        pose = frame_info.pose_info
        def draw_line(start, end):
            # 絕對座標
            start_x, start_y = pose.get_abs_landmark(start, width, height)
            end_x, end_y = pose.get_abs_landmark(end, width, height)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), (255, 25, 255), 2)
        # body
        draw_line(11,12)
        draw_line(12,24)
        draw_line(24,23)
        draw_line(23,11)
        # left arm and hand
        draw_line(11,13)
        draw_line(13,15)
        draw_line(15,21)
        draw_line(15,17)
        draw_line(15,19)
        draw_line(17,19)
        # right arm and hand
        draw_line(12,14)
        draw_line(14,16)
        draw_line(16,22)
        draw_line(16,18)
        draw_line(16,20)
        draw_line(18,20)
        # left leg
        draw_line(23,25)
        draw_line(25,27)
        draw_line(27,29)
        draw_line(27,31)
        draw_line(29,31)
        # right leg
        draw_line(24,26)
        draw_line(26,28)
        draw_line(28,30)
        draw_line(28,32)
        draw_line(30,32)
    return mask

def create_hand_mask(frame_info: FrameInfo, src_img: cv2.typing.MatLike, draw_on_src: bool = False) -> cv2.typing.MatLike:
    mask = src_img.copy() if draw_on_src else np.zeros_like(src_img) 
    height, width, _ = mask.shape

    if frame_info.hand_infos:
        for hand in frame_info.hand_infos:
            
            def draw_line(start, end):
                # 絕對座標
                start_x, start_y = hand.get_abs_landmark(start, width, height)
                end_x, end_y = hand.get_abs_landmark(end, width, height)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), (0, 0, 225), 4)
            # Thumb
            draw_line(0,1)
            draw_line(1,2)
            draw_line(2,3)
            draw_line(3,4)
            # Index
            draw_line(0,5)
            draw_line(5,6)
            draw_line(6,7)
            draw_line(7,8)
            # Middle
            draw_line(9,10)
            draw_line(10,11)
            draw_line(11,12)
            # Ring
            draw_line(13,14)
            draw_line(14,15)
            draw_line(15,16)
            # Pinky
            draw_line(0,17)
            draw_line(17,18)
            draw_line(18,19)
            draw_line(19,20)
            # Palm
            draw_line(5,9)
            draw_line(9,13)
            draw_line(13,17)
    return mask

def create_timeline_flow_mask(
    frame_info: FrameInfo,
    src_img: cv2.typing.MatLike,
    timeline_length = 7,           # 尾巴長度，這會追蹤前 tail_length 幀的資訊來產生
    draw_on_src: bool = False,
    ) -> cv2.typing.MatLike:

    frame_infos = []
    current = frame_info
    for i in range(timeline_length):
        frame_infos.append(frame_info)
        if not current.previous:
            break
        current = current.previous
    
    frame_infos.reverse()

    mask = src_img.copy() if draw_on_src else np.zeros_like(src_img) 
    height, width, _ = mask.shape

    def draw_claw_marks(start_hand_info: HandInfo, end_hand_info: HandInfo, ratio: bool) -> None:
        for i in [4, 8, 12, 16, 20]:
            start = start_hand_info.get_abs_landmark(i, width, height)
            end = end_hand_info.get_abs_landmark(i, width, height)

            # 由於 z 軸不是一個非常精確的數值且非 Normalized 的數值，因此不將 z 軸納入計算
            # 見: https://github.com/google/mediapipe/issues/742

            # 由於手部辨識模型並沒有 visibility 的資訊，因此不將 visibility 納入計算
            # 見 https://github.com/google/mediapipe/issues/4240#issuecomment-1502225717

            # 使用 ratio 為線條粗細及顏色深淺的依據
            # ratio越小的線段，顏色越淺、越細
            color = (0, int(ratio * 65), int(ratio * 255)) 
            
            # 線段粗細最小值為 1
            thickness = max(int(ratio * 10), 1)
            cv2.line(mask, start, end, color, thickness)
    
    end = min(len(frame_infos), timeline_length)
    for i in range(1, end):
        if frame_infos[i].hand_infos:
            for hand in frame_infos[i].hand_infos:
                if hand.tracking_previous_hand:
                    draw_claw_marks(hand, hand.tracking_previous_hand, i / timeline_length)
                    
                
    return mask