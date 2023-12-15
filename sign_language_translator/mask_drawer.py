
import cv2
import numpy as np
import algorithms as alg

from models import FrameInfo, HandInfo

def create_hand_mask(frame_info: FrameInfo, src_img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask = np.zeros_like(src_img)
        height, width, _ = mask.shape

        if frame_info.hand_infos:
            for hand in frame_info.hand_infos:
                
                def draw_line(start, end):
                    # 絕對座標
                    start_x, start_y = hand.get_abs_landmark(start, width, height)
                    end_x, end_y = hand.get_abs_landmark(end, width, height)
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), (255, 165, 0), 4)
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

def create_optical_flow_mask(
    frame_info: FrameInfo,
    src_img: cv2.typing.MatLike,
    tail_length = 7                 # 尾巴長度，這會追蹤前 tail_length 幀的資訊來產生
    ) -> cv2.typing.MatLike:
    def cofm(
        current_frame_info: FrameInfo,
        src_img: cv2.typing.MatLike,
        current_previous_frame
        ) -> cv2.typing.MatLike:
        if current_previous_frame >= tail_length or not current_frame_info.previous:
            return np.zeros_like(src_img)

        mask = cofm(
            current_frame_info.previous,
            src_img,
            current_previous_frame + 1
        )

        height, width, _ = mask.shape

        def draw_claw_marks(start_hand_info: HandInfo, end_hand_info: HandInfo, is_right_hand: bool) -> None:
            for i in [4, 8, 12, 16, 20]:
                start = start_hand_info.get_abs_landmark(i, width, height)
                end = end_hand_info.get_abs_landmark(i, width, height)

                # 由於 z 軸不是一個非常精確的數值且非 Normalized 的數值，因此不將 z 軸納入計算
                # 見: https://github.com/google/mediapipe/issues/742

                # 由於手部辨識模型並沒有 visibility 的資訊，因此不將 visibility 納入計算
                # 見 https://github.com/google/mediapipe/issues/4240#issuecomment-1502225717

                # 使用時間差異作為線條粗細及顏色深淺的依據
                # 越舊的線段，顏色越淺、越細
                # time_ratio 最小值為 (n / 1)
                time_ratio = (tail_length - current_previous_frame) / tail_length
                color = (int(time_ratio * 255), int(time_ratio * 65), 0) if is_right_hand \
                        else (0, int(time_ratio * 65), int(time_ratio * 255)) 
                
                # 線段粗細最小值為 1
                thickness = max(int(time_ratio * 10), 1)
                cv2.line(mask, start, end, color, thickness)

        if current_frame_info.previous:
            previous_frame_info = current_frame_info.previous
            if current_frame_info.left_hand_info and previous_frame_info.left_hand_info:
                draw_claw_marks(current_frame_info.left_hand_info, previous_frame_info.left_hand_info, False)
            if current_frame_info.right_hand_info and previous_frame_info.right_hand_info:
                draw_claw_marks(current_frame_info.right_hand_info, previous_frame_info.right_hand_info, True)
        
        return mask

    return cofm(frame_info, src_img, 0)