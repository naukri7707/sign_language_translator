from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from matplotlib.pylab import f

from .landmark import Hand, Landmark, Pose


@dataclass(frozen=True)
class FrameBase:
    frame: int


class ImageFrame(FrameBase):
    pass


@dataclass(frozen=True)
class RGBImageFrame(ImageFrame):
    image: np.ndarray

    def create_rgb(self) -> np.ndarray:
        return self.image.copy()

    def create_bgr(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def create_grayscale(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)


@dataclass(frozen=True)
class GrayscaleImageFrame(ImageFrame):
    image: np.ndarray


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

    @staticmethod
    def from_dict(dict: dict):
        hands = [
            Hand(
                landmarks=[
                    Landmark(
                        x=landmark["x"],
                        y=landmark["y"],
                        z=landmark["z"],
                    )
                    for landmark in hand["landmarks"]
                ]
            )
            for hand in dict["hands"]
        ]

        pose = Pose(
            landmarks=[
                Landmark(
                    x=landmark["x"],
                    y=landmark["y"],
                    z=landmark["z"],
                    v=landmark["v"],
                )
                for landmark in dict["pose"]["landmarks"]
            ]
        )

        return LandmarksFrame(
            frame=dict["frame"],
            hands=hands,
            pose=pose,
        )
