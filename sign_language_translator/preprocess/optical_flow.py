import cv2
import numpy as np

class OpticalFlow():
    def __init__(self) -> None:
        self.previous_gray = None
        self.mask = None
        pass

    def next(self, frame: cv2.typing.MatLike) -> (cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike):
        if self.mask is None:
            self.mask = np.zeros_like(frame)
            self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        previous = self.previous_gray
        mask = self.mask
        
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between the two frames
        flow = cv2.calcOpticalFlowFarneback(previous, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        
        # Sets image hue according to the optical flow  
        # direction 
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        
        # Converts HSV to RGB (BGR) color representation 
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 

        # Normalize x and y components
        x_flow = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
        y_flow = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        x_flow = x_flow.astype('uint8')
        y_flow = y_flow.astype('uint8')

        # Change - Make next frame previous frame
        previous = next_gray.copy()
        
        return x_flow, y_flow, rgb