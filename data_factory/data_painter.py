from typing import List

import cv2
import numpy as np

from data_factory.model.frame import LandmarksFrame

DEFAULT_RADIUS = 6
DEFAULT_THICKNESS = 5

# 定義手部骨架的連接順序
HAND_CONNECTIONS = [
    # 拇指
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # 食指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # 中指
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # 無名指
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # 小指
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# 定義身體姿態骨架的連接順序
POSE_CONNECTIONS = [
    # 頭部和肩膀
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # 上半身
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    # 手掌
    # (15, 17),
    # (16, 18),
    # (15, 19),
    # (16, 20),
    # (15, 21),
    # (16, 22),
    # (17, 19),
    # (18, 20),
    # 下半身和大腿
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    # 大腿和小腿
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    # 小腿和腳
    (27, 31),
    (28, 32),
]

POSE_JOINTS = [
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]


def draw_hand_skeleton(
    image: np.ndarray,
    landmarks_frame: LandmarksFrame,
    color: int = 255,
    thickness: int = DEFAULT_THICKNESS,
) -> np.ndarray:

    # 獲取影像的寬度和高度
    image_height, image_width = image.shape

    for hand in landmarks_frame.hands:
        # 繪製手部骨架
        for connection in HAND_CONNECTIONS:
            start_index, end_index = connection
            start_x = int(hand.landmarks[start_index].x * image_width)
            start_y = int(hand.landmarks[start_index].y * image_height)
            end_x = int(hand.landmarks[end_index].x * image_width)
            end_y = int(hand.landmarks[end_index].y * image_height)
            cv2.line(
                image,
                (start_x, start_y),
                (end_x, end_y),
                color,  # type: ignore
                thickness,
            )

    return image


def draw_hand_joint(
    image: np.ndarray,
    landmarks_frame: LandmarksFrame,
    color: int = 255,
    radius: int = DEFAULT_RADIUS,
):
    # 獲取影像的寬度和高度
    image_height, image_width = image.shape

    for hand in landmarks_frame.hands:
        # 繪製手部骨架
        for landmark in hand.landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            cv2.circle(
                image,
                (x, y),
                radius,
                color,  # type: ignore
                -1,  # thickness = -1 表示填滿，以畫出實心圓,
                # cv2.LINE_AA, # 反鋸齒
            )

    return image


def draw_pose_skeleton(
    image: np.ndarray,
    landmarks_frame: LandmarksFrame,
    color: int = 255,
    thickness: int = DEFAULT_THICKNESS,
) -> np.ndarray:

    # 獲取影像的寬度和高度
    image_height, image_width = image.shape

    pose = landmarks_frame.pose

    # 繪製身體姿態骨架
    for connection in POSE_CONNECTIONS:
        start_index, end_index = connection
        start_x = int(pose.landmarks[start_index].x * image_width)
        start_y = int(pose.landmarks[start_index].y * image_height)
        end_x = int(pose.landmarks[end_index].x * image_width)
        end_y = int(pose.landmarks[end_index].y * image_height)
        cv2.line(
            image,
            (start_x, start_y),
            (end_x, end_y),
            color,  # type: ignore
            thickness,
        )

    return image


def draw_pose_joint(
    image: np.ndarray,
    landmarks_frame: LandmarksFrame,
    color: int = 255,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    # 獲取影像的寬度和高度
    image_height, image_width = image.shape

    pose = landmarks_frame.pose

    # 繪製身體姿態關節點
    for joint in POSE_JOINTS:
        landmark = pose.landmarks[joint]
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        cv2.circle(
            image,
            (x, y),
            radius,
            color,  # type: ignore
            -1,  # thickness = -1 表示填滿，以畫出實心圓,
            # lineType=cv2.LINE_AA, # 反鋸齒
        )

    return image


def cut_image_to_square_by_pose(
    image: np.ndarray,
    landmarks_frame: LandmarksFrame,
    *,
    output_size: int = 360,
    re_center_x: bool = True,
    re_center_y: bool = False,
) -> np.ndarray:
    # 獲取影像的高度和寬度
    height, width = image.shape[:2]

    # 將人物姿態的中心點座標轉換為實際像素座標
    center_x = int(landmarks_frame.pose.center.x * width) if re_center_x else width // 2
    center_y = (
        int(landmarks_frame.pose.center.y * height) if re_center_y else height // 2
    )

    # 計算正方形區域的左上角和右下角座標
    left = center_x - output_size // 2
    top = center_y - output_size // 2
    right = left + output_size
    bottom = top + output_size

    # 創建一個黑色背景的輸出影像
    output_image = np.zeros((output_size, output_size), dtype=np.uint8)

    # 計算實際裁切區域在輸出影像中的位置
    output_left = max(0, -left)
    output_top = max(0, -top)
    output_right = min(output_size, width - left)
    output_bottom = min(output_size, height - top)

    # 計算實際裁切區域在原始影像中的位置
    input_left = max(0, left)
    input_top = max(0, top)
    input_right = min(width, right)
    input_bottom = min(height, bottom)

    # 將實際裁切區域複製到輸出影像中
    output_image[output_top:output_bottom, output_left:output_right] = image[
        input_top:input_bottom, input_left:input_right
    ]

    return output_image


def calc_longest_continuous_segment_range(
    landmarks_frames: List[LandmarksFrame],
    *,
    threshold=0.9,
    tolerance_frames=5,
):
    start_frame_index = 0
    end_frame_index = 0
    current_start_index = 0
    current_end_index = 0
    bad_frame_count = 0

    for i, frame in enumerate(landmarks_frames):
        if frame.hands is None or len(frame.hands) == 0:
            bad_frame_count += 1
        else:
            any_hand_within_threshold = False
            for hand in frame.hands:
                if hand.center.y < threshold:
                    any_hand_within_threshold = True
                    break

            if any_hand_within_threshold:
                bad_frame_count = 0
            else:
                bad_frame_count += 1

        if bad_frame_count > tolerance_frames:
            if (
                current_end_index - current_start_index
                > end_frame_index - start_frame_index
            ):
                start_frame_index = current_start_index
                end_frame_index = current_end_index
            current_start_index = i + 1
            current_end_index = i + 1
            bad_frame_count = 0
        else:
            current_end_index = i

    if current_end_index - current_start_index > end_frame_index - start_frame_index:
        start_frame_index = current_start_index
        end_frame_index = current_end_index

    return start_frame_index, end_frame_index
