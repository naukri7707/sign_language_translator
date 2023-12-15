from collections import deque
import json
from types import SimpleNamespace
from typing import Deque, List

import cv2
import numpy as np


class LandmarkInfo:
    def __init__(self, x: float, y: float, z: float, v: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        pass

    @staticmethod
    def from_mp_landmark(mp_landmark):
        return LandmarkInfo(
            x = mp_landmark.x,
            y = mp_landmark.y,
            z = mp_landmark.z,
            v = mp_landmark.visibility,
        )

class PoseInfo:
    landmarks: List[LandmarkInfo]

    def __init__(self, landmarks: List[LandmarkInfo]) -> None:
        self.landmarks = landmarks
        pass


    @staticmethod
    def empty():
        return PoseInfo(landmarks=[])

    @staticmethod
    def from_mp_pose_landmarks(mp_pose_landmarks):
        return PoseInfo(
            landmarks=[
                LandmarkInfo.from_mp_landmark(lm)
                for lm in mp_pose_landmarks.landmark
            ] if mp_pose_landmarks else None,
        )

class HandInfo:
    landmarks: List[LandmarkInfo]

    def __init__(self, landmarks: List[LandmarkInfo]) -> None:
        self.landmarks = landmarks
        pass

    def offset_sum_from(self, other: 'HandInfo') -> float:
        offset_sum = 0

        for lm in range(21):
            offset_x = abs(self.landmarks[lm].x - other.landmarks[lm].x)
            offset_y = abs(self.landmarks[lm].y - other.landmarks[lm].y)
            offset_sum += offset_x + offset_y # 只比較大小，所以使用曼哈頓距離簡化
            pass
        return offset_sum

    def get_abs_landmark(self, index: int, width: int, height: int) -> (int, int):
        return (
            int(self.landmarks[index].x * width),
            int(self.landmarks[index].y * height),
        )

    @staticmethod
    def empty():
        return HandInfo(landmarks=[])

    @staticmethod
    def from_mp_hand_landmarks(mp_hand_landmarks):
        return HandInfo(
                landmarks=[
                    LandmarkInfo.from_mp_landmark(lm)
                    for lm in mp_hand_landmarks
                ] if mp_hand_landmarks else None,
            )

class FrameInfo:
    previous: 'FrameInfo'
    frame: int
    pose_info: PoseInfo
    hand_infos: List[HandInfo]

    def __init__(self,previous: 'FrameInfo', frame: int, pose_info: PoseInfo, hand_infos: List[HandInfo]) -> None:
        self.previous = previous
        self.frame = frame
        self.pose_info = pose_info
        self.hand_infos = hand_infos
        pass

    @staticmethod
    def from_mp_data(previous: 'FrameInfo', frame: int, mp_pose_data, mp_hands_data):
        return FrameInfo(
            previous = previous,
            frame = frame,
            pose_info = PoseInfo.from_mp_pose_landmarks(mp_pose_data.pose_landmarks),
            hand_infos = [
                HandInfo.from_mp_hand_landmarks(hand.landmark)
                for hand in mp_hands_data.multi_hand_landmarks
            ] if mp_hands_data.multi_hand_landmarks else []
        )

class FrameInfoContainer:
    __max_frame_count : int
    __frame_infos : Deque[FrameInfo]

    def __init__(self, max_frame_count: int = -1) -> None:
        self.__max_frame_count = max_frame_count
        self.__frame_infos = deque()
        pass

    def append(self, frame_info: FrameInfo):
        # 如果超過最大數量，就移除最舊的
        if self.__max_frame_count > 0 and len(self.__frame_infos) >= self.__max_frame_count:
                self.__frame_infos.popleft()
        # 加入新的
        self.__frame_infos.append(frame_info)
        pass

    def last(self) -> FrameInfo:
        return self.__frame_infos[-1] if len(self.__frame_infos) > 0 else None

    def get_info(self, index: int) -> FrameInfo:
        return self.__frame_infos[index]

    @staticmethod
    def dump(target: 'FrameInfoContainer', json_file_path: str):

        serialized_data = {
            "max_frame_count" : target.__max_frame_count,
            "frame_infos" : []
        }

        for frame_info in target.__frame_infos:
            serialized_data["frame_infos"].append({
                "frame": frame_info.frame,
                "pose_info":
                {
                    "landmark":
                    [
                        {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "v": lm.v,
                        }
                        for lm in frame_info.pose_info.landmarks
                    ],
                } if frame_info.pose_info else None,
                "hand_infos":
                [
                    {
                        "landmark":
                        [
                            {
                                "x": lm.x,
                                "y": lm.y,
                                "z": lm.z,
                                "v": lm.v,
                            }
                            for lm in hand.landmarks
                        ]
                    }
                    for hand in frame_info.hand_infos
                ] if frame_info.hand_infos else [],
            })

        with open(json_file_path, 'w') as file:
           json.dump(serialized_data, file)
        pass

    @staticmethod
    def load(json_file_path: str, step: int = 1) -> 'FrameInfoContainer':
        with open(json_file_path, 'r') as file:
            serialized_data = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

            if serialized_data is None:
                return None

            container = FrameInfoContainer(
                max_frame_count = serialized_data.max_frame_count
                )

            for idx, frame_info in enumerate(serialized_data.frame_infos):
                if (idx + 1) % step != 0:
                    continue
                container.append(
                    FrameInfo(
                        previous = container.last(),
                        frame = frame_info.frame,
                        pose_info = PoseInfo(
                            landmarks = [
                                LandmarkInfo(
                                    x = lm.x,
                                    y = lm.y,
                                    z = lm.z,
                                    v = lm.v,
                                )
                                for lm in frame_info.pose_info.landmark
                            ] if frame_info.pose_info else None,
                        ),
                        hand_infos = [
                            HandInfo(
                                landmarks = [
                                    LandmarkInfo(
                                        x = lm.x,
                                        y = lm.y,
                                        z = lm.z,
                                        v = lm.v,
                                    )
                                    for lm in hand.landmark
                                ]
                            )
                            for hand in frame_info.hand_infos
                        ] if frame_info.hand_infos else None,
                    )
                )

            return container

