import cv2

## -- 通用參數 --

# 總分類數量
CLASS_NUM = 10

# 每隔 n 幀取一張影格
FRAME_STEP = 3

## -- 訓練參數 --

# 訓練迭代次數
EPOCHS = 50

# 以連續 n 幀影格作為訓練資料
N_FRAMES = 20 

# 訓練批次大小
BATCH_SIZE = 8

# 訓練影格大小
FRAME_SIZE = (224, 224)
