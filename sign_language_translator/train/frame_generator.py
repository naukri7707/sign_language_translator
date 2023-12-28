import random
from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import train.data_augmentation as da
import params

def data_augmentation(frames, seed):
    augmentated_frames = []

    for frame in frames:
        # 不能用 flip 因為手語中左右手有差異
        agmt_frame = frame
        # agmt_frame = da.random_flip(agmt_frame, seed)
        agmt_frame = da.random_rotation(agmt_frame, params.RANDOM_ROATAION, seed)
        agmt_frame = da.random_zoom(agmt_frame, params.RANDOM_ZOOM , seed)
        augmentated_frames.append(agmt_frame)

    return frames

def format_frames(frame, output_size):
    """
        調整大小並對影片中的圖像進行填充
        
        Args:
        frame: 需要調整大小和填充的影像
        output_size: 輸出影格圖像的像素尺寸

        Return:
        使用指定輸出尺寸的填充格式化影格
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    
    return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 3):
    """
        從每個類別的每個影片檔案中建立影格

        Args:
        video_path: 影片的檔案路徑
        n_frames: 每個影片檔案要建立的影格數量
        output_size: 輸出影格圖像的像素尺寸
        frame_step: 每個影格之間的間隔

        Return:
        shape 為 (n_frames, height, width, channels) 的 NumPy 陣列。
    """

    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    # 計算需要的影格數量
    need_length = 1 + (n_frames - 1) * frame_step

    # 如果影片長度不足，則從開頭開始
    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
            pass
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

class FrameGenerator:
    def __init__(
        self,
        path: str,
        num_frames: int,
        resolution:  Tuple[int, int],
        frame_step: int,
        class_ids_for_name: List[Tuple[str, int]] = None,
        glob_patten = '*/*.mp4',
        training = False):
        """ 
        回傳具有相關標籤的一組影格

        Args:
            path: 影片檔案路徑
            n_frames: 影片檔案路徑
            training: 是否正在建立訓練資料集
        """
        self.path = Path(path)
        self.n_frames = num_frames
        self.resolution = resolution
        self.frame_step = frame_step
        self.glob_patten = glob_patten
        self.training = training
        self.seed = random.randint(0, 2**32)

        self.class_names = sorted([subdir.name for subdir in self.path.iterdir() if subdir.is_dir()])
        self.class_ids_for_name = class_ids_for_name if class_ids_for_name \
                        else dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_data(self):
        video_paths = list(self.path.glob(self.glob_patten))  # 設定路徑 class_label/frames_set
        classes = [p.parent.name for p in video_paths]  # 設定路徑標籤 (基本上都一樣會是 class_label 資料夾)
        return video_paths, classes

    def __call__(self):
        file_paths, classes = self.get_data()

        pairs = list(zip(file_paths, classes))

        # 如果是訓練資料集，則將資料隨機化
        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(
                path,
                self.n_frames,
                self.resolution,
                self.frame_step,
                )
        
            if self.training:
                video_frames = data_augmentation(video_frames, self.seed)

            label = self.class_ids_for_name[name] # 將標籤映射成整數
            yield video_frames, label