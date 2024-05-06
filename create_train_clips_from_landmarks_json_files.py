from pathlib import Path
from typing import List

import cv2
import numpy as np

import data_factory.data_painter as data_painter
from data_factory import converter, utils
from data_factory.model.frame import GrayscaleImageFrame, LandmarksFrame


def create_traning_clip(
    landmarks_frames: List[LandmarksFrame],
    image_shape=(360, 360),  # 單通道的話不需要定義 shape
) -> List[GrayscaleImageFrame]:
    start_frame_index, end_frame_index = (
        data_painter.calc_longest_continuous_segment_range(landmarks_frames)
    )
    output_size = min(image_shape[0], image_shape[1])
    output_shape = (output_size, output_size)

    output_landmarks_frames = landmarks_frames[start_frame_index : end_frame_index + 1]

    history_image: np.ndarray = np.zeros(output_shape, dtype=np.uint8)
    image_frames: List[GrayscaleImageFrame] = []

    for i, current_landmarks_frame in enumerate(output_landmarks_frames):
        image = np.zeros(image_shape, dtype=np.uint8)

        # 骨架
        image = data_painter.draw_pose_skeleton(image, current_landmarks_frame)
        image = data_painter.draw_hand_skeleton(image, current_landmarks_frame)
        image = cv2.GaussianBlur(image, (13, 13), 13)
        image = cv2.GaussianBlur(image, (25, 25), 25)

        # 關節
        image = data_painter.draw_pose_joint(image, current_landmarks_frame)
        image = data_painter.draw_hand_joint(image, current_landmarks_frame)
        image = cv2.GaussianBlur(image, (3, 3), 3)

        # 裁切成正方形
        image = data_painter.cut_image_to_square_by_pose(
            image, current_landmarks_frame, output_size=360
        )

        # 模糊化歷史圖像並將當前圖像加入歷史圖像
        history_image = cv2.GaussianBlur(history_image, (17, 17), 17)
        history_image = cv2.addWeighted(history_image, 0.7, image, 0.3, 1)

        # 將歷史圖像轉換成 ImageFrame 並存入佇列
        rgb_image = cv2.cvtColor(history_image, cv2.COLOR_BGR2RGB)
        image_frarme = GrayscaleImageFrame(
            frame=current_landmarks_frame.frame,
            image=rgb_image,
        )
        image_frames.append(image_frarme)

    return image_frames


src_folder_path = "~data/landmarks"
dst_folder_path = "~data/traning_clips"
dst_extension_name = ".mp4"


def on_traversed(file_path: Path):
    # 建立對應的目標路徑
    target_directory = Path(dst_folder_path)
    target_file_path = target_directory / file_path.relative_to(Path(src_folder_path))
    target_file_path = target_file_path.with_suffix(dst_extension_name)

    # 確保目標目錄存在
    target_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    landmarks_frames = converter.to_landmarks_frame.from_json_file(
        str(file_path),
    )

    traning_clip = create_traning_clip(landmarks_frames)

    converter.to_video.from_grayscale_image_frames(
        str(target_file_path),
        image_frames=traning_clip,
    )


utils.io.traverse_files(
    src_folder_path,
    on_traversed=on_traversed,
)
