import os
from typing import List
import cv2
import numpy as np
import mask_drawer as md

from models import FrameInfoContainer

# 使用MP4編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 由於影片幀率為 58 fps，因此每 3 幀取一次 ~= 0.05 秒 / 次
step = 3

# 尾巴長度，這會追蹤前 tail_frame_count 幀的資訊來產生
tail_frame_count = 14

# 匯出影片中除了判斷為手語片段外的額外幀數 (前後各 29 幀, 0.5s)
padding = 29

# 匯出裁切影片
crop_size = int(360)

def is_sign_part(frame_info):
    if frame_info.hand_infos:
        for hand in frame_info.hand_infos:
            if hand.get_avg_y() < 0.95:
                return True
    return False

def process_video(video_file_path, output_file_path, frame_infos, start_idx, end_idx):
    parent_folder = os.path.dirname(output_file_path)
    os.makedirs(parent_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_file_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (crop_size, crop_size))
    # 利用 Pose 計算人物 X 軸點以定位裁切中心點
    pose_avg_x = np.average([x.pose_info.get_avg_x() for x in filter(lambda f: f.pose_info and f.pose_info.landmarks, frame_infos)])

    frame = 0
    stepped_frame = 0
    while cap.isOpened():
        success, frame_img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        imageHeight, imageWidth, _ = image_rgb.shape

        frame += 1

        if frame % step != 0:
            continue
        
        stepped_frame += 1

        if stepped_frame < start_idx:
            continue
        if stepped_frame > end_idx:
            break

        # x=身體中心點, y=圖片中心點
        crop_center_x, crop_center_y = int(pose_avg_x * imageWidth), imageHeight // 2
        half_diagonal_length = crop_size // 2

        output_img = frame_img[max(crop_center_y - half_diagonal_length, 0): crop_center_y + half_diagonal_length, max(crop_center_x - half_diagonal_length, 0): crop_center_x + half_diagonal_length, : ]
        if output_img.shape[1] < crop_size:
            padded = np.zeros((output_img.shape[0], max(0, crop_size - output_img.shape[1]), output_img.shape[-1]), dtype=np.uint8)
            if crop_center_x - half_diagonal_length < 0:
                output_img = np.concatenate((padded, output_img,), axis=1)
            else:
                output_img = np.concatenate((output_img, padded,), axis=1)
        
        out.write(output_img)

    cap.release()
    out.release()

def crop_video(input_file_path, mapping_paths: List[str]):
    # 跳過 俯視角
    if input_file_path.endswith('top_360p.mp4'):
        return

    lmdata_file_path = mapping_paths[0]
    output_file_path = mapping_paths[1]

    top_file_path = input_file_path.replace('front', 'top')
    top_lmdata_file_path = lmdata_file_path.replace('front', 'top')
    top_output_path = output_file_path.replace('front', 'top')
    
    container = FrameInfoContainer.load(lmdata_file_path, step)

    front_frame_infos = container.frame_infos

    # 取出第一個手勢出現的幀數
    start_idx = next((index for index, fi in enumerate(front_frame_infos) if is_sign_part(fi)), None) + 1
    # 取出最後一個手勢出現的幀數
    end_idx = (len(front_frame_infos) - next((index for index, fi in enumerate(reversed(front_frame_infos)) if is_sign_part(fi)), None))

    start_idx = max(0, start_idx - padding)
    end_idx = min(len(front_frame_infos), end_idx + padding)

    container = FrameInfoContainer.load(top_lmdata_file_path, step)
    top_frame_infos = container.frame_infos

    process_video(input_file_path, output_file_path, front_frame_infos, start_idx, end_idx)
    process_video(top_file_path, top_output_path, top_frame_infos, start_idx, end_idx)
    pass
# Sample

import file_walker as fw

fw.walk(
    '~data/2_360p/front',
    [
        ('~data/3_lmdata/front', lambda name: f"{name}_lmdata.json"),
        ('~data/4_cropped/front', lambda name: f"{name}_cropped.mp4"),
    ],
    crop_video,
    skip=0,
)