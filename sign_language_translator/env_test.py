import cv2
import numpy as np
import tensorflow as tf

#### GPU ####
gpus = tf.config.list_physical_devices('GPU')

print("------------------------")
print(f"TensorFlow version: {tf.__version__}")
if gpus:
    print(len(gpus), "GPUs detected!")
else:
    print("No GPUs detected.")
    
print("------------------------")

#### GUI ####

# 創建一個紅色矩形（400x300）
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = (0, 0, 255)  # 設置紅色（BGR格式）

# 顯示圖像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()