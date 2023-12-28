## -- 通用參數 --

# 總分類數量
NUM_CLASSES = 151

# 每隔 n 幀取一張影格
FRAME_STEP = 15

## -- 訓練參數 --

DATASET_PATH = "~data/dataset/src_flow_split"

# MoViNet 模型 ID
MOVINET_MODEL_ID = 'a5'

# 訓練學習率
LEARNING_RATE = 0.001

# 訓練迭代次數
EPOCHS = 100

# 以連續 n 幀影格作為訓練資料
NUM_FRAMES = 16

# 訓練批次大小
BATCH_SIZE = 16

# 訓練影格解析度
RESOLUTION = (224, 224)
