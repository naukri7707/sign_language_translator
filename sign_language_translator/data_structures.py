from scipy.optimize import linear_sum_assignment
from collections import deque
import json
from typing import Deque, List
from types import SimpleNamespace

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

    def offset_distance_from(self, other: 'HandInfo') -> float:
        offset_sum = 0

        for lm in range(21):
            offset_x = abs(self.landmarks[lm].x - other.landmarks[lm].x)
            offset_y = abs(self.landmarks[lm].y - other.landmarks[lm].y)
            offset_sum += offset_x + offset_y # 只比較大小，所以使用曼哈頓距離優化
            pass
        return offset_sum
    
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

    def create_hand_mask(self, src_img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask = np.zeros_like(src_img)
        height, width, _ = mask.shape

        if self.hand_infos:
            if len(self.hand_infos) > 1:
                pass
            for hand in self.hand_infos:
                def draw_line(start, end):
                    # 絕對座標
                    start_x = int(hand.landmarks[start].x * width)
                    start_y = int(hand.landmarks[start].y * height)
                    end_x = int(hand.landmarks[end].x * width)
                    end_y = int(hand.landmarks[end].y * height)
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), (0, 192, 255), 2)
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

    def get_nearest_hand_from_previous_frame(self) -> List[float]:
        previous = self.previous
        for previous_hand in previous.hand_infos:
            for hand in self.hand_infos:
                distance = hand.offset_distance_from(previous_hand)
                pass

        for hand in self.hand_infos:
            if self.previous:
                pass
        return None

    def create_opflow_mask(self, src_img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask = np.zeros_like(src_img)
        height, width, _ = mask.shape

        if self.hand_infos:
            if len(self.hand_infos) > 1:
                pass
            for hand in self.hand_infos:
                def draw_line(start, end):
                    # 絕對座標
                    start_x = int(hand.landmarks[start].x * width)
                    start_y = int(hand.landmarks[start].y * height)
                    end_x = int(hand.landmarks[end].x * width)
                    end_y = int(hand.landmarks[end].y * height)
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), (0, 192, 255), 2)



        
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
    def load(json_file_path: str) -> 'FrameInfoContainer':
        with open(json_file_path, 'r') as file:
            serialized_data = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

            if serialized_data is None:
                return None
            
            container = FrameInfoContainer(
                max_frame_count = serialized_data.max_frame_count
                )
            
            for frame_info in serialized_data.frame_infos:
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