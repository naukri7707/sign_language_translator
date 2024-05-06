import copy
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float
    v: float = 0.0


@dataclass(frozen=True)
class Hand:
    landmarks: List[Landmark] = field(
        default_factory=lambda: [Landmark(0, 0, 0) for _ in range(20)],
    )

    @property
    def wrist(self) -> Landmark:
        return self.landmarks[0]

    @property
    def thumb_cmc(self) -> Landmark:
        return self.landmarks[1]

    @property
    def thumb_mcp(self) -> Landmark:
        return self.landmarks[2]

    @property
    def thumb_ip(self) -> Landmark:
        return self.landmarks[3]

    @property
    def thumb_tip(self) -> Landmark:
        return self.landmarks[4]

    @property
    def index_finger_mcp(self) -> Landmark:
        return self.landmarks[5]

    @property
    def index_finger_pip(self) -> Landmark:
        return self.landmarks[6]

    @property
    def index_finger_dip(self) -> Landmark:
        return self.landmarks[7]

    @property
    def index_finger_tip(self) -> Landmark:
        return self.landmarks[8]

    @property
    def middle_finger_mcp(self) -> Landmark:
        return self.landmarks[9]

    @property
    def middle_finger_pip(self) -> Landmark:
        return self.landmarks[10]

    @property
    def middle_finger_dip(self) -> Landmark:
        return self.landmarks[11]

    @property
    def middle_finger_tip(self) -> Landmark:
        return self.landmarks[12]

    @property
    def ring_finger_mcp(self) -> Landmark:
        return self.landmarks[13]

    @property
    def ring_finger_pip(self) -> Landmark:
        return self.landmarks[14]

    @property
    def ring_finger_dip(self) -> Landmark:
        return self.landmarks[15]

    @property
    def ring_finger_tip(self) -> Landmark:
        return self.landmarks[16]

    @property
    def pinky_mcp(self) -> Landmark:
        return self.landmarks[17]

    @property
    def pinky_pip(self) -> Landmark:
        return self.landmarks[18]

    @property
    def pinky_dip(self) -> Landmark:
        return self.landmarks[19]

    @property
    def pinky_tip(self) -> Landmark:
        return self.landmarks[20]

    @property
    def center(self) -> Landmark:
        calc_landmarks = [
            self.landmarks[0],
            self.landmarks[5],
            self.landmarks[9],
            self.landmarks[13],
            self.landmarks[17],
        ]
        x = sum(landmark.x for landmark in calc_landmarks) / len(calc_landmarks)
        y = sum(landmark.y for landmark in calc_landmarks) / len(calc_landmarks)
        z = sum(landmark.z for landmark in calc_landmarks) / len(calc_landmarks)
        return Landmark(x, y, z)


@dataclass(frozen=True)
class Pose:
    landmarks: List[Landmark] = field(
        default_factory=lambda: [Landmark(0, 0, 0) for _ in range(33)]
    )

    @property
    def nose(self) -> Landmark:
        return self.landmarks[0]

    @property
    def left_eye_inner(self) -> Landmark:
        return self.landmarks[1]

    @property
    def left_eye(self) -> Landmark:
        return self.landmarks[2]

    @property
    def left_eye_outer(self) -> Landmark:
        return self.landmarks[3]

    @property
    def right_eye_inner(self) -> Landmark:
        return self.landmarks[4]

    @property
    def right_eye(self) -> Landmark:
        return self.landmarks[5]

    @property
    def right_eye_outer(self) -> Landmark:
        return self.landmarks[6]

    @property
    def left_ear(self) -> Landmark:
        return self.landmarks[7]

    @property
    def right_ear(self) -> Landmark:
        return self.landmarks[8]

    @property
    def mouth_left(self) -> Landmark:
        return self.landmarks[9]

    @property
    def mouth_right(self) -> Landmark:
        return self.landmarks[10]

    @property
    def left_shoulder(self) -> Landmark:
        return self.landmarks[11]

    @property
    def right_shoulder(self) -> Landmark:
        return self.landmarks[12]

    @property
    def left_elbow(self) -> Landmark:
        return self.landmarks[13]

    @property
    def right_elbow(self) -> Landmark:
        return self.landmarks[14]

    @property
    def left_wrist(self) -> Landmark:
        return self.landmarks[15]

    @property
    def right_wrist(self) -> Landmark:
        return self.landmarks[16]

    @property
    def left_pinky(self) -> Landmark:
        return self.landmarks[17]

    @property
    def right_pinky(self) -> Landmark:
        return self.landmarks[18]

    @property
    def left_index(self) -> Landmark:
        return self.landmarks[19]

    @property
    def right_index(self) -> Landmark:
        return self.landmarks[20]

    @property
    def left_thumb(self) -> Landmark:
        return self.landmarks[21]

    @property
    def right_thumb(self) -> Landmark:
        return self.landmarks[22]

    @property
    def left_hip(self) -> Landmark:
        return self.landmarks[23]

    @property
    def right_hip(self) -> Landmark:
        return self.landmarks[24]

    @property
    def left_knee(self) -> Landmark:
        return self.landmarks[25]

    @property
    def right_knee(self) -> Landmark:
        return self.landmarks[26]

    @property
    def left_ankle(self) -> Landmark:
        return self.landmarks[27]

    @property
    def right_ankle(self) -> Landmark:
        return self.landmarks[28]

    @property
    def left_heel(self) -> Landmark:
        return self.landmarks[29]

    @property
    def right_heel(self) -> Landmark:
        return self.landmarks[30]

    @property
    def left_foot_index(self) -> Landmark:
        return self.landmarks[31]

    @property
    def right_foot_index(self) -> Landmark:
        return self.landmarks[32]

    @property
    def center(self) -> Landmark:
        calc_landmarks = [
            self.landmarks[11],
            self.landmarks[12],
            self.landmarks[23],
            self.landmarks[24],
        ]
        x = sum(landmark.x for landmark in calc_landmarks) / len(calc_landmarks)
        y = sum(landmark.y for landmark in calc_landmarks) / len(calc_landmarks)
        z = sum(landmark.z for landmark in calc_landmarks) / len(calc_landmarks)
        return Landmark(x, y, z)
