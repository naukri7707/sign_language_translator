from typing import Any, List

from ..model import Hand, ImageFrame, Landmark, LandmarksFrame, Pose


def from_image_frame(
    image_frame: ImageFrame,
    mp_hands: Any,
    mp_pose: Any,
) -> LandmarksFrame:
    # 建立圖像副本
    rgb_image = image_frame.create_rgb()

    # 進行手部檢測
    hand_results = mp_hands.process(rgb_image)
    hands = []
    if hand_results.multi_hand_landmarks:
        for mp_hand_landmarks in hand_results.multi_hand_landmarks:
            hand_landmarks = []
            for mp_landmark in mp_hand_landmarks.landmark:
                hand_landmarks.append(
                    Landmark(
                        x=mp_landmark.x,
                        y=mp_landmark.y,
                        z=mp_landmark.z,
                    )
                )
            hands.append(
                Hand(
                    landmarks=hand_landmarks,
                )
            )

    # 進行姿勢檢測
    pose_results = mp_pose.process(rgb_image)
    pose_landmarks = []
    if pose_results.pose_landmarks:
        for mp_landmark in pose_results.pose_landmarks.landmark:
            pose_landmarks.append(
                Landmark(
                    x=mp_landmark.x,
                    y=mp_landmark.y,
                    z=mp_landmark.z,
                    v=mp_landmark.visibility,
                )
            )

    pose = Pose(
        landmarks=pose_landmarks,
    )

    return LandmarksFrame(
        hands=hands,
        pose=pose,
        frame=image_frame.frame,
    )


def from_image_frames(
    image_frames: List[ImageFrame],
    mp_hands: Any,
    mp_pose: Any,
) -> List[LandmarksFrame]:
    """
    Convert a list of ImageFrame to a list of LandmarksFrame

    Sort the input list by frame number before conversion

    Args:
        image_frames (List[ImageFrame]): The input list of ImageFrame
        mp_hands (Any): The MediaPipe hands model
        mp_pose (Any): The MediaPipe pose model

    Returns:
        (List[LandmarksFrame]): The output list of LandmarksFrame
    """
    image_frames.sort(key=lambda image_frame: image_frame.frame)

    return [
        from_image_frame(image_frame, mp_hands, mp_pose) for image_frame in image_frames
    ]
