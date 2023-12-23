import numpy as np
import os
import tensorflow as tf
import pathlib

# %% [markdown]
# ## 設定檔案路徑

# %%

data_dir = pathlib.Path('~data/4_frames')

# %% [markdown]
# # 使用 Keras 載入數據

# %%
batch_size = 32
img_height = 360
img_width = 360

train_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# # 確認 class_names (labels)

class_names = train_dataset.class_names

# ## 訓練模型

num_classes = 10 # 有幾種類別 (因為是 1~10 所以是 10)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255), # 標準化數據
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=3
)