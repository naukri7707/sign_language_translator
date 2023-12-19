
import os
from typing import List
import cv2
import numpy as np
import mask_drawer as md

from models import FrameInfoContainer

# 由於影片幀率為 58 fps，因此每 3 幀取一次 ~= 0.05 秒 / 次
step = 3

# 尾巴長度，這會追蹤前 tail_frame_count 幀的資訊來產生
tail_frame_count = 14

# 匯出裁切影片
crop_size = int(360)

def is_sign_part(frame_info):
    if frame_info.hand_infos:
        for hand in frame_info.hand_infos:
            if hand.get_avg_y() < 0.95:
                return True
    return False

def process_video(video_file_path, output_folder_path, frame_infos, start_idx, end_idx):

    os.makedirs(output_folder_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file_path)

    # 利用 Pose 計算人物 X 軸點以定位裁切中心點
    pose_avg_x = np.average([x.pose_info.get_avg_x() for x in filter(lambda f: f.pose_info and f.pose_info.landmarks, frame_infos)])

    frame = 0
    stepped_frame = 0
    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imageHeight, imageWidth, _ = image_rgb.shape

        frame += 1

        if frame % step != 0:
            continue
        
        stepped_frame += 1

        if stepped_frame < start_idx:
            continue
        if stepped_frame > end_idx:
            break

        frame_info = frame_infos[stepped_frame - 1]
        
        hand_mask = md.create_hand_mask(frame_info, image_rgb)
        optical_flow_mask = md.create_optical_flow_mask(frame_info, image_rgb, tail_frame_count)
        
        image_black = np.zeros_like(image_rgb)

        image_with_mask = image_black
        image_with_mask = cv2.add(image_with_mask, hand_mask)
        image_with_mask = cv2.add(image_with_mask, optical_flow_mask)
        output_image = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)
        
        output_file_true_path = os.path.join(output_folder_path, f"{str(stepped_frame - start_idx).zfill(3)}.jpg")

        # Crop
        # x=身體中心點, y=圖片中心點
        crop_center_x, crop_center_y = int(pose_avg_x * imageWidth), imageHeight // 2
        half_diagonal_length = crop_size // 2
        img = output_image
        img = img[max(crop_center_y - half_diagonal_length, 0): crop_center_y + half_diagonal_length, max(crop_center_x - half_diagonal_length, 0): crop_center_x + half_diagonal_length, : ]
        if img.shape[1] < crop_size:
            padded = np.zeros((img.shape[0], max(0, crop_size - img.shape[1]), img.shape[-1]), dtype=np.uint8)
            if crop_center_x - half_diagonal_length < 0:
                img = np.concatenate((padded, img,), axis=1)
            else:
                img = np.concatenate((img, padded,), axis=1)

        cv2.imwrite(output_file_true_path, img)

    cap.release()

def videos_to_frames(input_file_path, mapping_paths: List[str]):
    
    # 跳過 俯視角
    if input_file_path.endswith('top_360p.mp4'):
        return


    lmdata_file_path = mapping_paths[0]
    output_folder_path = mapping_paths[1]

    top_file_path = input_file_path.replace('front', 'top')
    top_lmdata_file_path = lmdata_file_path.replace('front', 'top')
    top_output_folder_path = output_folder_path.replace('front', 'top')

    
    container = FrameInfoContainer.load(lmdata_file_path, step)

    front_frame_infos = container.frame_infos

    # 取出第一個手勢出現的幀數，額外加一幀使其可以產生光流
    start_idx = next((index for index, fi in enumerate(front_frame_infos) if is_sign_part(fi)), None) + 1
    # 取出最後一個手勢出現的幀數
    end_idx = (len(front_frame_infos) - next((index for index, fi in enumerate(reversed(front_frame_infos)) if is_sign_part(fi)), None))

    container = FrameInfoContainer.load(top_lmdata_file_path, step)
    top_frame_infos = container.frame_infos

    process_video(input_file_path, output_folder_path, front_frame_infos, start_idx, end_idx)
    process_video(top_file_path, top_output_folder_path, top_frame_infos, start_idx, end_idx)