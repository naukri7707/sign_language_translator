from pathlib import Path

import mediapipe as mp

import data_factory.converter as converter
import data_factory.utils as utils

static_image_mode = False
max_num_hands = 2
min_detection_confidence = 0.5

mp_hands = mp.solutions.hands.Hands(  # type: ignore
    static_image_mode=static_image_mode,
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
)

mp_pose = mp.solutions.pose.Pose(  # type: ignore
    static_image_mode=static_image_mode,
    min_detection_confidence=min_detection_confidence,
)


src_folder_path = "~data/source"
dst_folder_path = "~data/landmarks"
dst_extension_name = ".json"


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

    image_frames = converter.to_rgb_image_frame.from_video(str(file_path))
    landmarks_frames = converter.to_landmarks_frame.from_rgb_image_frames(
        image_frames, mp_hands, mp_pose
    )
    converter.to_json_file.from_landmarks_frames(
        str(target_file_path), landmarks_frames
    )


utils.io.traverse_files(
    src_folder_path,
    on_traversed=on_traversed,
)

mp_hands.close()
mp_pose.close()
