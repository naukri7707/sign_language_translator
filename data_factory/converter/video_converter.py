from typing import List, Tuple

import cv2

from data_factory.model.frame import GrayscaleImageFrame, RGBImageFrame


def from_rgb_image_frames(
    video_file_path: str,
    image_frames: List[RGBImageFrame],
    output_size: Tuple[int, int] = (360, 360),
) -> None:
    # 初始化輸出視訊的編解碼器和寫入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    output_video = cv2.VideoWriter(
        video_file_path,
        fourcc,
        30,
        output_size,
    )

    for image_frame in image_frames:
        bgr_image = image_frame.create_bgr()
        output_video.write(bgr_image)

    # 釋放視訊寫入器
    output_video.release()


def from_grayscale_image_frames(
    video_file_path: str,
    image_frames: List[GrayscaleImageFrame],
    output_size: Tuple[int, int] = (360, 360),
) -> None:
    # 初始化輸出視訊的編解碼器和寫入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    output_video = cv2.VideoWriter(
        video_file_path,
        fourcc,
        30,
        output_size,
        True,
    )

    for image_frame in image_frames:
        output_video.write(image_frame.image)

    # 釋放視訊寫入器
    output_video.release()
