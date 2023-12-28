import cv2
import numpy as np

class OpticalFlow():
    def __init__(self, quality = 1, bound = 20) -> None:
        self.DualTVL1 = cv2.optflow.DualTVL1OpticalFlow_create(scaleStep = 0.5, warps = 3, epsilon = 0.02) if quality == 1 \
                        else cv2.DualTVL1OpticalFlow_create(warps = 1) if quality == 2 \
                        else cv2.DualTVL1OpticalFlow_create() if quality == 3 \
                        else None
        if self.DualTVL1 is None:
            raise Exception("Invalid quality value")

        self.bound = bound

        self.prvs_gray = None

        pass

    
    def to_img(self, flow):
        flow_with_3_channel = np.concatenate((flow, self.zero_mask), axis=2)
        img = np.round((flow_with_3_channel + 1.0) * 127.5).astype(np.uint8)
        return img

    def first(self, frame: cv2.typing.MatLike) -> (cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike):
        h, w, _ = frame.shape
        # 初始化一張全黑的 mask 用來與 flow 合併以產生圖片
        self.zero_mask = np.zeros((h, w, 1), dtype = np.float32)
        # save first image in black&white
        self.prvs_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = np.zeros((h, w, 2), dtype = np.float32)
        
        return self.to_img(flow)

    def next(self, frame: cv2.typing.MatLike) -> (cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike):
        # first frame
        if self.prvs_gray is None:
            return self.first(frame)
        
        # get image in black&white
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = self.DualTVL1.calc(self.prvs_gray, next_gray, None)

        # only 2 dims
        flow = flow[:, :, 0:2]

        # truncate to +/-15.0, then rescale to [-1.0, 1.0]
        flow[flow > self.bound] = self.bound 
        flow[flow < -self.bound] = -self.bound
        flow = flow / self.bound

        self.prvs_gray = next_gray

        return self.to_img(flow)