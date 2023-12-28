## -- 通用參數 --

# 總分類數量
# 目前實際上只有 151 個單字，但由於 db 中有 250 個單字，因此這裡設定為 250，否則會因為 label 對不上導致無法訓練 (loss 為 nan)
NUM_CLASSES = 250

# 每隔 n 幀取一張影格
FRAME_STEP = 3

## -- 資料增強 --

RANDOM_ROATAION = (-10, 10)

RANDOM_ZOOM = (90, 100)

## -- 訓練參數 --

WORDS_DB_PATH = "words_db.json"

DATASET_PATH = "~data/dataset/src_hand_split"

# MoViNet 模型 ID
MOVINET_MODEL_ID = 'a2'

# 以連續 n 幀影格作為訓練資料
NUM_FRAMES = 80

# 訓練影格解析度
RESOLUTION = (224, 224)

# 訓練迭代次數
EPOCHS = 100

# 訓練批次大小
BATCH_SIZE = 16

# 訓練學習率
LEARNING_RATE = 0.001

# 訓練時隨機丟棄比例 (降低 overfitting)
DROPOUT_RATE = 0.2