import json
import re
from typing import Any, List

from ..model import Hand, Landmark, LandmarksFrame, Pose, RGBImageFrame
from ..utils import dict_mapper


def from_landmarks_frame(landmark_frame: LandmarksFrame) -> str:
    """Convert LandmarksFrame to JSON string

    Args:
        landmark_frame (LandmarksFrame): Data to be converted

    Returns:
        (str): JSON string after conversion
    """

    data = landmark_frame.to_dict()

    # Set float attributes to 2 decimal places
    new_data = dict_mapper.round_float_attribute(data, decimal_places=2)

    return json.dumps(new_data)


def from_landmarks_frames(
    json_file_path: str, landmark_frames: List[LandmarksFrame]
) -> None:
    """Convert a list of LandmarksFrame to JSON string

    Sort the input list by frame number before conversion

    Args:
        landmark_frames (List[LandmarksFrame]): Data to be converted

    Returns:
        (str): JSON string after conversion
    """

    landmark_frames.sort(key=lambda frame: frame.frame)
    with open(json_file_path, "w") as file:
        json.dump(
            [
                dict_mapper.round_float_attribute(frame.to_dict(), decimal_places=2)
                for frame in landmark_frames
            ],
            file,
        )
