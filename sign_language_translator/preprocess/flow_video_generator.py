
from email.mime import image
import math
from typing import List
import cv2
import numpy as np
from .optical_flow import OpticalFlow
from .mask_drawer import create_hand_mask, create_pose_mask, create_timeline_flow_mask
from .models import FrameInfoContainer

import params

# 輸出編碼器
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# 使用前多少幀產生偽光流
TIMELINE_FRAME_COUNT = 14

# 影片裁切大小
CROP_SIZE = int(360)

def is_sign_part(frame_info):
    if frame_info.hand_infos:
        for hand in frame_info.hand_infos:
            if hand.get_avg_y() < 0.95:
                return True
    return False

def process_video(video_file_path, output_file_path_without_ext, frame_infos, start_idx, end_idx, offset=0):
    optflow = OpticalFlow()

    cap = cv2.VideoCapture(video_file_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_file_true_path = f"{output_file_path_without_ext}_offset{offset + 1}.mp4"
    out = cv2.VideoWriter(output_file_true_path, FOURCC, fps, (CROP_SIZE, CROP_SIZE))

    # 利用 Pose 計算人物 X 軸點以定位裁切中心點
    pose_avg_x = np.average([x.pose_info.get_avg_x() for x in filter(lambda f: f.pose_info and f.pose_info.landmarks, frame_infos)])

    # 如果沒有偵測到人物，則使用 0.5 作為中心點
    if math.isnan(pose_avg_x):
        pose_avg_x = 0.5

    frame = 0
    stepped_frame = 0

    for _ in range(offset):
        success, img = cap.read()

        if not success:
            break

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imageHeight, imageWidth, _ = image_rgb.shape

        frame += 1

        if frame % params.FRAME_STEP != 0:
            continue
        
        stepped_frame += 1

        if stepped_frame < start_idx:
            continue
        if stepped_frame > end_idx:
            break

        frame_info = frame_infos[stepped_frame - 1]
        
        hand_mask = create_hand_mask(frame_info, image_rgb)
        timeline_mask = create_timeline_flow_mask(frame_info, image_rgb, TIMELINE_FRAME_COUNT)
        x_flow, y_flow, rgb = optflow.next(img)

        image_black = np.zeros_like(image_rgb)

        image_with_mask = image_black
        image_with_mask = cv2.add(image_with_mask, hand_mask)
        image_with_mask = cv2.add(image_with_mask, timeline_mask)
        image_with_mask = cv2.add(image_with_mask, rgb)
        masked_image = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)
        
        # Crop
        # x=身體中心點, y=圖片中心點
        crop_center_x, crop_center_y = int(pose_avg_x * imageWidth), imageHeight // 2
        half_diagonal_length = CROP_SIZE // 2
        
        cropped_img = masked_image[max(crop_center_y - half_diagonal_length, 0): crop_center_y + half_diagonal_length, max(crop_center_x - half_diagonal_length, 0): crop_center_x + half_diagonal_length, : ]
        if cropped_img.shape[1] < CROP_SIZE:
            padded = np.zeros((cropped_img.shape[0], max(0, CROP_SIZE - cropped_img.shape[1]), cropped_img.shape[-1]), dtype=np.uint8)
            if crop_center_x - half_diagonal_length < 0:
                cropped_img = np.concatenate((padded, cropped_img,), axis=1)
            else:
                cropped_img = np.concatenate((cropped_img, padded,), axis=1)
        
        out.write(cropped_img)

    cap.release()
    out.release()

def generate_flow_videos(input_file_path, mapping_paths: List[str]):
    
    lmdata_file_path = mapping_paths[0]
    output_folder_path = mapping_paths[1]

    containers = FrameInfoContainer.load(lmdata_file_path, params.FRAME_STEP)
    
    for idx, container in enumerate(containers):

        frame_infos = container.frame_infos

        # 取出第一個手勢出現的幀數，額外加一幀使其可以產生光流
        start_idx = next((index for index, fi in enumerate(frame_infos) if is_sign_part(fi)), None) + 1

        # 取出最後一個手勢出現的幀數
        end_idx = (len(frame_infos) - next((index for index, fi in enumerate(reversed(frame_infos)) if is_sign_part(fi)), None))

        process_video(input_file_path, output_folder_path, frame_infos, start_idx, end_idx, idx)