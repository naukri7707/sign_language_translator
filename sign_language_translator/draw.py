
import cv2
import numpy as np
import algorithms as  alg

from models import FrameInfo

def create_hand_mask(frame_info: FrameInfo, src_img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask = np.zeros_like(src_img)
        height, width, _ = mask.shape

        if frame_info.hand_infos:
            if len(frame_info.hand_infos) > 1:
                pass
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
    end_previous_frame = 7
    ) -> cv2.typing.MatLike:
    
    def cofm(
        current_frame_info: FrameInfo,
        src_img: cv2.typing.MatLike,
        current_previous_frame
        ) -> cv2.typing.MatLike:

        if current_previous_frame >= end_previous_frame or not current_frame_info.previous:
            return np.zeros_like(src_img)

        mask = cofm(
            current_frame_info.previous,
            src_img,
            current_previous_frame + 1
        )

        height, width, _ = mask.shape

        if current_frame_info.hand_infos:
            map = alg.create_distance_map(
                current_frame_info.hand_infos,
                current_frame_info.previous.hand_infos,
                lambda ch, ph: ch.offset_sum_from(ph),
            )
            distance, matches = alg.calc_min_distance_matching(map)

            for idx, match in enumerate(matches):
                start_hand_info = current_frame_info.hand_infos[idx]
                end_hand_info = current_frame_info.previous.hand_infos[match]
                for i in [4, 8, 12, 16, 20]:
                    start_x, start_y = start_hand_info.get_abs_landmark(i, width, height)
                    end_x, end_y = end_hand_info.get_abs_landmark(i, width, height)

                    time_ratio = (end_previous_frame - current_previous_frame) / end_previous_frame
                    thickness = int(time_ratio * 10) + 1

                    cv2.line(
                        mask,
                        (start_x, start_y),
                        (end_x, end_y),
                        (0, int(time_ratio * 65), int(time_ratio * 255)),
                        thickness
                        )
        return mask

    return cofm(frame_info, src_img, 0)