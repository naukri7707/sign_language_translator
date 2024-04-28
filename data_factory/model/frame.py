from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from matplotlib.pylab import f

from .landmark import Hand, Pose


@dataclass(frozen=True)
class FrameBase:
    frame: int


@dataclass(frozen=True)
class ImageFrame(FrameBase):
    rgb_image: np.ndarray

    def create_rgb(self) -> np.ndarray:
        return self.rgb_image.copy()

    def create_bgr(self) -> np.ndarray:
        return cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

    def create_grayscale(self) -> np.ndarray:
        return cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2GRAY)


@dataclass(frozen=True)
class LandmarksFrame(FrameBase):
    hands: List[Hand]
    pose: Pose

    def to_dict(self) -> dict:
        dict = {
            "frame": self.frame,
            "hands": [
                {
                    "landmarks": [
                        {
                            "x": hand.landmarks[index].x,
                            "y": hand.landmarks[index].y,
                            "z": hand.landmarks[index].z,
                        }
                        for index in range(len(hand.landmarks))
                    ]
                }
                for hand in self.hands
            ],
            "pose": {
                "landmarks": [
                    {
                        "x": self.pose.landmarks[index].x,
                        "y": self.pose.landmarks[index].y,
                        "z": self.pose.landmarks[index].z,
                        "v": self.pose.landmarks[index].v,
                    }
                    for index in range(len(self.pose.landmarks))
                ]
            },
        }

        return dict
