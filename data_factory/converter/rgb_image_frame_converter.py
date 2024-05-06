from typing import List

import cv2

from ..model import RGBImageFrame


def from_image(image_path: str, frame: int = 0) -> RGBImageFrame:
    """
    Read an image and create an RGBImageFrame from it

    Args:
        image_path (str): The path of the image file
        frame (int): The frame number of the image

    Returns:
        (RGBImageFrame): The RGBImageFrame created from the image
    """
    # Read the image with OpenCV in RGB color mode
    rgb_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # Create an RGBImageFrame with the RGB image and the frame number
    image_frame = RGBImageFrame(
        image=rgb_image,
        frame=frame,
    )
    return image_frame


def from_video(video_path: str, start_frame: int = 0) -> List[RGBImageFrame]:
    cap = cv2.VideoCapture(video_path)

    current_frame = start_frame
    image_frames = []

    while cap.isOpened():
        success, bgr_image = cap.read()
        if not success:
            break

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        image_frame = RGBImageFrame(
            image=rgb_image,
            frame=current_frame,
        )

        image_frames.append(image_frame)

        current_frame += 1

    return image_frames
